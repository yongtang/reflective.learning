import functools

import PIL.Image
import PIL.ImageOps
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from reflective_learning.model import autocast


class ContextEncoder:
    def __init__(
        self,
        dimension,
        text_model,
        image_model,
        text_tokenizer,
        image_processor,
        device="cpu",
    ):
        assert text_model is not None, "text_model is required"
        assert image_model is not None, "image_model is required"
        assert text_tokenizer is not None, "text_tokenizer is required"
        assert image_processor is not None, "image_processor is required"

        self.dimension = dimension
        self.device = torch.device(device)
        self.text_model = text_model.to(self.device).eval()
        self.image_model = image_model.to(self.device).eval()
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor

    @classmethod
    def from_pretrained(cls, info, device="cpu"):
        text_model = AutoModel.from_pretrained(
            info["context"]["text"]["model"],
            revision=info["context"]["text"]["revision"],
        )
        text_tokenizer = AutoTokenizer.from_pretrained(
            info["context"]["text"]["model"],
            revision=info["context"]["text"]["revision"],
        )
        image_model = AutoModel.from_pretrained(
            info["context"]["image"]["model"],
            revision=info["context"]["image"]["revision"],
        )
        image_processor = AutoProcessor.from_pretrained(
            info["context"]["image"]["model"],
            revision=info["context"]["image"]["revision"],
            use_fast=True,
        )
        dimension = info["layer"]["d_model"]

        return cls(
            dimension=dimension,
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
        with torch.inference_mode(), autocast():
            value = self.text_model(input_ids=input_ids).last_hidden_state.squeeze(0)
            output = torch.nn.functional.pad(
                value, (0, (self.dimension - value.shape[-1]))
            ).view(-1, self.dimension)
            # CPU float32 output, consistent regardless of autocast
            return output.to("cpu", dtype=torch.float32)

    @functools.lru_cache(maxsize=1024)
    def encode_image_embed(self, image) -> torch.Tensor:
        if isinstance(image, str):
            with PIL.Image.open(image) as f:
                image = PIL.ImageOps.exif_transpose(f).convert("RGB")
        with torch.inference_mode(), autocast():
            values = self.image_processor(images=[image], return_tensors="pt")
            values = {
                k: v.to(self.device) for k, v in values.items() if torch.is_tensor(v)
            }
            value = self.image_model.get_image_features(**values).last_hidden_state
            output = torch.nn.functional.pad(
                value, (0, (self.dimension - value.shape[-1]))
            ).view(-1, self.dimension)
            # CPU float32 output, consistent regardless of autocast
            return output.to("cpu", dtype=torch.float32)

    def encode_block_embed(self, block: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), autocast():
            value = torch.as_tensor(block).flatten()
            output = torch.nn.functional.pad(
                value, (0, (-block.numel()) % self.dimension)
            ).view(-1, self.dimension)
            # CPU float32 output, consistent regardless of autocast
            return output.to("cpu", dtype=torch.float32)

    def encode(
        self,
        text: list[str],
        image: list,
        block: torch.Tensor,
    ) -> torch.Tensor:
        text = list(self.encode_text_embed(chunk) for chunk in text)
        image = list(self.encode_image_embed(chunk) for chunk in image)
        block = self.encode_block_embed(block)
        segments = []
        break_embed = torch.zeros((1, self.dimension), dtype=torch.float32)

        for chunk in text:
            segments.append(chunk)
        segments.append(break_embed.clone())

        for chunk in image:
            segments.append(chunk)
        segments.append(break_embed.clone())

        segments.append(block)

        segments.append(break_embed.clone())

        return torch.cat(segments, dim=0)
