import json
import os
from functools import lru_cache

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


class ContextEncoder:
    def __init__(self, text_model, image_model, tokenizer, device="cpu"):
        self.text_model = text_model.to(device) if text_model else None
        self.image_model = image_model.to(device) if image_model else None
        self.tokenizer = tokenizer
        self.device = device

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

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

        return cls(text_model, image_model, tokenizer, device=device)

    @lru_cache(maxsize=1024)
    def encode_text(self, text):
        if self.text_model is None or self.tokenizer is None:
            raise ValueError("text_model and tokenizer must be set for text encoding.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        with torch.no_grad():
            output = self.text_model(input_ids=input_ids)
            return output.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()

    @lru_cache(maxsize=1024)
    def encode_image(self, image_path):
        if self.image_model is None:
            raise ValueError("image_model must be set for image encoding.")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.image_model(pixel_values=image_tensor)
            return output.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu()

    def encode(self, text, image_path):
        text_embed = self.encode_text(text)
        image_embed = self.encode_image(image_path)
        return torch.cat([text_embed, image_embed], dim=0)
