import functools

import PIL.Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class ContextEncoder:
    def __init__(
        self, text_model, image_model, tokenizer, image_processor, device="cpu"
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert tokenizer is not None, "tokenizer is required"
        assert image_processor is not None, "image_processor is required"
        assert text_model.config.hidden_size == image_model.config.hidden_size

        self.text_model = text_model.to(device)
        self.image_model = image_model.to(device)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device

    @classmethod
    def from_pretrained(cls, info, device="cpu"):
        text_cfg = info["text"]
        image_cfg = info["image"]

        text_model = AutoModel.from_pretrained(
            info["text"]["model"], revision=info["text"]["revision"]
        )
        tokenizer = AutoTokenizer.from_pretrained(
            info["text"]["model"], revision=info["text"]["revision"]
        )
        image_model = AutoModel.from_pretrained(
            info["image"]["model"], revision=info["image"]["revision"]
        )
        image_processor = AutoImageProcessor.from_pretrained(
            info["image"]["model"], revision=info["image"]["revision"], use_fast=True
        )

        return cls(text_model, image_model, tokenizer, image_processor, device=device)

    @functools.lru_cache(maxsize=1024)
    def encode_text_embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            output = self.text_model(input_ids=input_ids)
            return output.last_hidden_state.squeeze(0).detach().cpu()

    @functools.lru_cache(maxsize=1024)
    def encode_image_embed(self, image_path: str) -> torch.Tensor:
        image = PIL.Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        with torch.no_grad():
            output = self.image_model(pixel_values=pixel_values)
            return output.last_hidden_state.squeeze(0).detach().cpu()

    def encode(self, text: list[str], image: list[str]) -> torch.Tensor:
        segments = []
        break_embed = torch.zeros(
            (1, self.text_model.config.hidden_size), dtype=torch.float32
        )

        for t in text:
            segments.append(self.encode_text_embed(t))
        segments.append(break_embed.clone())

        for path in image:
            segments.append(self.encode_image_embed(path))
        segments.append(break_embed.clone())

        return torch.cat(segments, dim=0)
