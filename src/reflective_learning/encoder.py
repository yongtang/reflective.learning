import os
import torch
import torch.nn as nn
from functools import lru_cache


class ContextEncoder(nn.Module):
    def __init__(self, text_encoder=None, image_encoder=None, device="cpu"):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.device = torch.device(device)

    def forward(self, text_ids, image_path):
        return self.encode(text_ids, image_path)

    def encode(self, text_ids, image_path):
        device = self.device

        # --- Text encoding ---
        if self.text_encoder and text_ids:
            text_embed = self.text_encoder.embed_text(tuple(text_ids)).to(device)
        elif self.text_encoder:
            text_embed = torch.zeros(self.text_encoder.embedding_dim, device=device)
        else:
            text_embed = None

        # --- Image encoding ---
        if self.image_encoder and image_path and os.path.exists(image_path):
            image_tensor = self.image_encoder.load_image(image_path).to(device)
            image_embed = self.image_encoder(image_tensor).squeeze(0)
        elif self.image_encoder:
            image_embed = torch.zeros(self.image_encoder.embedding_dim, device=device)
        else:
            image_embed = None

        # --- Combine ---
        if text_embed is not None and image_embed is not None:
            combined = torch.cat([text_embed, image_embed], dim=-1)
        elif text_embed is not None:
            combined = text_embed
        elif image_embed is not None:
            combined = image_embed
        else:
            raise ValueError("Both text and image encoders are missing.")

        return combined

    @classmethod
    def load(cls, context_dir, device="cpu"):
        text_encoder = None
        image_encoder = None

        text_path = os.path.join(context_dir, "text_encoder.pt")
        text_vocab_path = os.path.join(context_dir, "text_vocab.json")
        image_path = os.path.join(context_dir, "image_encoder.pt")

        if os.path.exists(text_path):
            text_encoder = load_text_encoder(text_path, text_vocab_path)

        if os.path.exists(image_path):
            image_encoder = load_image_encoder(image_path)

        if not text_encoder and not image_encoder:
            raise ValueError("No context encoders found in directory: " + context_dir)

        return cls(
            text_encoder=text_encoder, image_encoder=image_encoder, device=device
        )


# -------------------------------
# Text encoder loader with cache
# -------------------------------


def load_text_encoder(path, vocab_path):
    model = torch.load(path, map_location="cpu")
    model.eval()
    model.requires_grad_(False)

    @lru_cache(maxsize=256)
    def embed_text(text_ids):  # text_ids: tuple of ints
        with torch.no_grad():
            input_tensor = torch.tensor([list(text_ids)], dtype=torch.long)  # [1, T]
            x = model.transformer.wte(input_tensor)
            x = model.transformer.drop(x)
            for block in model.transformer.h:
                x = block(x)
            x = model.transformer.ln_f(x)
            return x.mean(dim=1).squeeze(0)  # [D]

    model.embed_text = embed_text
    model.embedding_dim = model.transformer.wte.embedding_dim

    return model


# -------------------------------
# Image encoder loader with cache
# -------------------------------


def load_image_encoder(path):
    model = torch.load(path, map_location="cpu")
    model.eval()
    model.requires_grad_(False)

    if not hasattr(model, "embedding_dim"):
        raise ValueError("Image encoder must define `embedding_dim`.")

    @lru_cache(maxsize=64)
    def load_image(image_path):
        from PIL import Image
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        return transform(image).unsqueeze(0)  # [1, 3, H, W]

    model.load_image = load_image
    return model
