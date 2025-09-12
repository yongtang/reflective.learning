import functools

import PIL.Image
import PIL.ImageOps
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from .model import autocast


class ContextEncoder:
    def __init__(
        self, text_model, image_model, text_tokenizer, image_processor, device="cpu"
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert text_tokenizer is not None, "text_tokenizer is required"
        assert image_processor is not None, "image_processor is required"
        assert text_model.config.hidden_size == image_model.config.hidden_size

        self.device = torch.device(device)
        self.text_model = text_model.to(self.device).eval()
        self.image_model = image_model.to(self.device).eval()
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(cls, info, device="cpu"):
        text_cfg = info["text"]
        image_cfg = info["image"]

        text_model = AutoModel.from_pretrained(
            info["text"]["model"], revision=info["text"]["revision"]
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            info["text"]["model"], revision=info["text"]["revision"]
        )
        image_model = AutoModel.from_pretrained(
            info["image"]["model"], revision=info["image"]["revision"]
        )
        image_processor = AutoImageProcessor.from_pretrained(
            info["image"]["model"], revision=info["image"]["revision"], use_fast=True
        )

        return cls(
            text_model=text_model,
            image_model=image_model,
            text_tokenizer=text_tokenizer,
            image_processor=image_processor,
            device=device,
        )

    @functools.lru_cache(maxsize=1024)
    def encode_text_embed(self, text: str) -> torch.Tensor:
        inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device, non_blocking=True)
        with torch.inference_mode(), autocast():  # your helper
            output = self.text_model(input_ids=input_ids)
            # CPU float32 output, consistent regardless of autocast
            return output.last_hidden_state.squeeze(0).to("cpu", dtype=torch.float32)

    @functools.lru_cache(maxsize=1024)
    def encode_image_embed(self, image_path: str) -> torch.Tensor:
        with PIL.Image.open(image_path) as f:
            image = PIL.ImageOps.exif_transpose(f).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device, non_blocking=True)
        with torch.inference_mode(), autocast():  # your helper
            output = self.image_model(pixel_values=pixel_values)
            # CPU float32 output, consistent regardless of autocast
            return output.last_hidden_state.squeeze(0).to("cpu", dtype=torch.float32)

    def encode_embed(
        self, text: list[torch.Tensor], image: list[torch.Tensor]
    ) -> torch.Tensor:
        segments = []
        break_embed = torch.zeros(
            (1, self.text_model.config.hidden_size), dtype=torch.float32
        )

        for chunk in text:
            segments.append(chunk)
        segments.append(break_embed.clone())

        for chunk in image:
            segments.append(chunk)
        segments.append(break_embed.clone())

        return torch.cat(segments, dim=0)

    def encode(self, text: list[str], image: list[str]) -> torch.Tensor:
        text = list(self.encode_text_embed(chunk) for chunk in text)
        image = list(self.encode_image_embed(chunk) for chunk in image)
        return self.encode_embed(text, image)
