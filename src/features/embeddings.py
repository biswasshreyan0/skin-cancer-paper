"""Medical/foundation-model image embedding extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


def extract_embeddings(
    image_paths: Iterable[str | Path],
    model_name: str,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Extract L2-normalized image embeddings with Hugging Face AutoModel/AutoProcessor."""
    if model_name.lower().startswith("medclip"):
        return _extract_medclip_embeddings(image_paths, model_name, batch_size=batch_size, device=device)
    if model_name.startswith("hf-hub:") or "BiomedCLIP" in model_name:
        return _extract_open_clip_embeddings(image_paths, model_name, batch_size=batch_size, device=device)

    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Embedding extraction requires torch and transformers. "
            "Install requirements.txt in the skinai conda environment."
        ) from exc

    image_paths = [str(p) for p in image_paths]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model {model_name!r}. Check the model name, "
            "network access, authentication, and compatible transformers version."
        ) from exc

    model = model.to(device)
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding {model_name}"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}

            if hasattr(model, "get_image_features"):
                outputs = model.get_image_features(**inputs)
            else:
                model_outputs = model(**inputs)
                if hasattr(model_outputs, "image_embeds") and model_outputs.image_embeds is not None:
                    outputs = model_outputs.image_embeds
                elif hasattr(model_outputs, "pooler_output") and model_outputs.pooler_output is not None:
                    outputs = model_outputs.pooler_output
                elif hasattr(model_outputs, "last_hidden_state"):
                    outputs = model_outputs.last_hidden_state[:, 0]
                else:
                    raise RuntimeError(f"Could not locate image embeddings in outputs for {model_name!r}.")

            outputs = torch.nn.functional.normalize(outputs.float(), p=2, dim=1)
            all_embeddings.append(outputs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _extract_medclip_embeddings(
    image_paths: Iterable[str | Path],
    model_name: str,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Extract image embeddings with the optional MedCLIP package."""
    try:
        import torch
        from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModel, MedCLIPVisionModelViT
    except Exception as exc:
        raise RuntimeError(
            "MedCLIP extraction requires the optional medclip package. "
            "Install it before using an embedding model name starting with 'medclip'."
        ) from exc

    image_paths = [str(p) for p in image_paths]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_cls = MedCLIPVisionModelViT if "vit" in model_name.lower() else MedCLIPVisionModel
    checkpoint = None
    if ":" in model_name:
        _, checkpoint = model_name.split(":", 1)
        checkpoint = checkpoint.strip() or None

    try:
        model = MedCLIPModel(vision_cls=vision_cls)
        if checkpoint and checkpoint != "default":
            model.from_pretrained(checkpoint)
        else:
            model.from_pretrained()
        processor = MedCLIPProcessor()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MedCLIP embedding model {model_name!r}. "
            "Use names like 'medclip:default' or 'medclip-vit:default' after installing medclip."
        ) from exc

    model = model.to(device)
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding {model_name}"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
            if hasattr(model, "encode_image"):
                try:
                    outputs = model.encode_image(input_ids=None, pixel_values=inputs["pixel_values"])
                except TypeError:
                    outputs = model.encode_image(inputs["pixel_values"])
            else:
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get("img_embeds")
                    if outputs is None:
                        outputs = outputs.get("image_embeds")
                elif hasattr(outputs, "image_embeds"):
                    outputs = outputs.image_embeds
            if outputs is None:
                raise RuntimeError(f"Could not locate image embeddings in MedCLIP outputs for {model_name!r}.")
            outputs = torch.nn.functional.normalize(outputs.float(), p=2, dim=1)
            all_embeddings.append(outputs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _extract_open_clip_embeddings(
    image_paths: Iterable[str | Path],
    model_name: str,
    batch_size: int = 8,
    device: str | None = None,
) -> np.ndarray:
    """Extract image embeddings for OpenCLIP/HF Hub models such as BiomedCLIP."""
    try:
        import torch
        import open_clip
    except Exception as exc:
        raise RuntimeError(
            "OpenCLIP embedding extraction requires torch and open_clip_torch. "
            "Install requirements.txt in the skinai conda environment."
        ) from exc

    image_paths = [str(p) for p in image_paths]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    open_clip_name = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
    try:
        model, preprocess = open_clip.create_model_from_pretrained(open_clip_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load OpenCLIP embedding model {model_name!r}. Check the model name, "
            "network access, authentication, and open_clip_torch compatibility."
        ) from exc

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding {model_name}"):
            batch_paths = image_paths[start : start + batch_size]
            images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            inputs = torch.stack(images, dim=0).to(device)
            outputs = model.encode_image(inputs)
            outputs = torch.nn.functional.normalize(outputs.float(), p=2, dim=1)
            all_embeddings.append(outputs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)
