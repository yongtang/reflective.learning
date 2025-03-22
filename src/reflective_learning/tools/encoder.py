import json
import os
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class ContextEncoder:
    def __init__(
        self, text_model, image_model, tokenizer, image_processor, device="cpu"
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert tokenizer is not None, "tokenizer is required"
        assert image_processor is not None, "image_processor is required"

        assert text_model.config.hidden_size == image_model.config.hidden_size, (
            f"Text model (hidden={text_model.config.hidden_size}) and image model "
            f"(hidden={image_model.config.hidden_size}) must match."
        )

        self.text_model = text_model.to(device)
        self.image_model = image_model.to(device)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device

    @classmethod
    def from_pretrained(cls, context_dir, device="cpu"):
        with open(os.path.join(context_dir, "context_versions.json")) as f:
            versions = json.load(f)

        text_cfg = versions["pretrained_models"]["gpt2"]
        image_cfg = versions["pretrained_models"]["vit"]

        text_model = AutoModel.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_cfg["model"], revision=text_cfg["revision"]
        )
        image_model = AutoModel.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"]
        )
        image_processor = AutoImageProcessor.from_pretrained(
            image_cfg["model"], revision=image_cfg["revision"], use_fast=True
        )

        return cls(text_model, image_model, tokenizer, image_processor, device=device)

    @lru_cache(maxsize=1024)
    def encode_text_embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            output = self.text_model(input_ids=input_ids)
            return output.last_hidden_state.squeeze(0).detach().cpu()  # shape [T, D]

    @lru_cache(maxsize=1024)
    def encode_image_embed(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            output = self.image_model(pixel_values=pixel_values)
            return output.last_hidden_state.squeeze(0).detach().cpu()  # shape [I, D]

    def encode(self, text: list[str], image: list[str]) -> torch.Tensor:
        segments = []

        # Always create break embedding
        break_dim = self.text_model.config.hidden_size
        break_embed = torch.zeros((1, break_dim), dtype=torch.float32)

        # Add text embeddings
        for t in text:
            segments.append(self.encode_text_embed(t))
        segments.append(break_embed.clone())  # break between text and image

        # Add image embeddings
        for path in image:
            segments.append(self.encode_image_embed(path))
        segments.append(break_embed.clone())  # break between context and tokens

        return torch.cat(segments, dim=0)
