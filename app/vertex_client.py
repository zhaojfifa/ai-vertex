"""Client utilities for interacting with Vertex Imagen 3."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

_MODEL_NAME = "imagen-3.0-generate-001"
_DEFAULT_IMAGE_DIMENSIONS = {"width": 1024, "height": 1024}
_DEFAULT_SAFETY_LEVEL = "block_few"

_model: Optional[ImageGenerationModel] = None
_initialized: bool = False


def _ensure_initialized() -> ImageGenerationModel:
    """Initialise Vertex AI and return a cached image generation model."""
    global _initialized, _model

    if _initialized and _model is not None:
        return _model

    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    missing = [name for name, value in (
        ("GCP_PROJECT_ID", project),
        ("GCP_LOCATION", location),
        ("GOOGLE_APPLICATION_CREDENTIALS", credentials),
    ) if not value]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    vertexai.init(project=project, location=location)
    _model = ImageGenerationModel.from_pretrained(_MODEL_NAME)
    _initialized = True
    return _model


def _parse_dimensions(value: Any) -> Dict[str, int]:
    """Convert a size/image_dimensions value into a width/height mapping."""
    if value is None:
        return dict(_DEFAULT_IMAGE_DIMENSIONS)

    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError("size should be formatted as 'WIDTHxHEIGHT'")
        try:
            width, height = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("width and height must be integers") from exc
        return _validate_dimensions(width, height)

    if isinstance(value, dict):
        try:
            width = int(value["width"])
            height = int(value["height"])
        except KeyError as exc:
            raise ValueError("image_dimensions requires 'width' and 'height'") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("width and height must be integers") from exc
        return _validate_dimensions(width, height)

    raise ValueError("Unsupported image dimension format")


def _validate_dimensions(width: int, height: int) -> Dict[str, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    return {"width": width, "height": height}


def generate_image(prompt: str, **params: Any) -> bytes:
    """Generate an image using Vertex Imagen 3 and return it as bytes."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    model = _ensure_initialized()

    normalized: Dict[str, Any] = {}

    # Normalise size/image_dimensions parameters.
    image_dimensions: Optional[Any] = None
    if "image_dimensions" in params:
        image_dimensions = params.pop("image_dimensions")
    elif "size" in params:
        image_dimensions = params.pop("size")
    elif "width" in params or "height" in params:
        if "width" not in params or "height" not in params:
            raise ValueError("both width and height must be provided together")
        width = params.pop("width")
        height = params.pop("height")
        image_dimensions = {"width": width, "height": height}

    normalized["image_dimensions"] = _parse_dimensions(image_dimensions)

    # Propagate other parameters (e.g., aspect_ratio) if provided.
    for key, value in params.items():
        if value is not None:
            normalized[key] = value

    normalized.setdefault("safety_filter_level", _DEFAULT_SAFETY_LEVEL)

    response = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        **normalized,
    )

    if not response.images:
        raise RuntimeError("No image returned from Vertex AI")

    image = response.images[0]
    image_bytes = getattr(image, "image_bytes", None)
    if image_bytes is None:
        # Fallback for SDK versions exposing the private attribute.
        image_bytes = getattr(image, "_image_bytes", None)
    if image_bytes is None:
        raise RuntimeError("Image bytes not available in the response")

    return image_bytes
