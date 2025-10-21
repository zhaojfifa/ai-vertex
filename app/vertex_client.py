"""Client utilities for interacting with Vertex Imagen 3."""
from __future__ import annotations

import math
import os
from typing import Any, Dict

from fastapi import HTTPException
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

_MODEL_NAME = "imagen-3.0-generate-001"
_DEFAULT_ASPECT_RATIO = "1:1"
_DEFAULT_SAFETY_LEVEL = "block_few"
_SUPPORTED_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4"}
_ASPECT_RATIO_VALUES = {
    "1:1": 1.0,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
}
_ALLOWED_PARAMS = {
    "negative_prompt",
    "safety_filter_level",
    "person_generation",
    "language",
    "number_of_images",
}


def _normalize_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise request payload into parameters accepted by the SDK."""

    params = dict(payload or {})
    normalized: Dict[str, Any] = {}

    if params.get("aspect_ratio"):
        normalized["aspect_ratio"] = str(params.pop("aspect_ratio"))
    else:
        width: int | None = None
        height: int | None = None

        if "size" in params and params["size"] is not None:
            width, height = _parse_size_value(params.pop("size"))
        elif "image_dimensions" in params and params["image_dimensions"] is not None:
            width, height = _parse_size_value(params.pop("image_dimensions"))
        elif "width" in params or "height" in params:
            if params.get("width") is None or params.get("height") is None:
                raise HTTPException(
                    status_code=400,
                    detail="Both width and height must be provided together.",
                )
            width = _to_positive_int(params.pop("width"))
            height = _to_positive_int(params.pop("height"))

        if width is not None and height is not None:
            normalized["aspect_ratio"] = _aspect_ratio_from_dimensions(width, height)
        else:
            normalized["aspect_ratio"] = _DEFAULT_ASPECT_RATIO

    for key in ("size", "image_dimensions", "width", "height"):
        params.pop(key, None)

    for key in _ALLOWED_PARAMS:
        value = params.get(key)
        if value is not None:
            normalized[key] = value

    normalized.setdefault("safety_filter_level", _DEFAULT_SAFETY_LEVEL)
    normalized.setdefault("number_of_images", 1)

    return normalized


def _parse_size_value(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise HTTPException(
                status_code=400,
                detail="size/image_dimensions must follow 'WIDTHxHEIGHT' format.",
            )
        try:
            width, height = (int(part) for part in parts)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise HTTPException(
                status_code=400,
                detail="Width and height must be integers.",
            ) from exc
        return _validate_dimensions(width, height)

    if isinstance(value, dict):
        try:
            width = value["width"]
            height = value["height"]
        except KeyError as exc:
            raise HTTPException(
                status_code=400,
                detail="image_dimensions dict requires 'width' and 'height'.",
            ) from exc
        return _validate_dimensions(width, height)

    raise HTTPException(
        status_code=400,
        detail="Unsupported image dimension format.",
    )


def _validate_dimensions(width: Any, height: Any) -> tuple[int, int]:
    width_int = _to_positive_int(width)
    height_int = _to_positive_int(height)
    if width_int <= 0 or height_int <= 0:
        raise HTTPException(
            status_code=400,
            detail="Width and height must be positive integers.",
        )
    return width_int, height_int


def _to_positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive parsing
        raise HTTPException(
            status_code=400,
            detail="Width and height must be integers.",
        ) from exc
    if number <= 0:
        raise HTTPException(
            status_code=400,
            detail="Width and height must be positive integers.",
        )
    return number


def _aspect_ratio_from_dimensions(width: int, height: int) -> str:
    gcd_value = math.gcd(width, height)
    simplified = f"{width // gcd_value}:{height // gcd_value}"
    if simplified in _SUPPORTED_ASPECT_RATIOS:
        return simplified

    ratio_value = width / height
    closest = min(
        _SUPPORTED_ASPECT_RATIOS,
        key=lambda ratio: abs(ratio_value - _ASPECT_RATIO_VALUES[ratio]),
    )
    return closest


def generate_image(prompt: str, **payload: Any) -> bytes:
    """Generate an image using Vertex Imagen 3 and return it as bytes."""

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        project = os.environ["GCP_PROJECT_ID"]
        location = os.environ["GCP_LOCATION"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Missing environment variable: {exc.args[0]}",
        ) from exc

    try:
        vertexai.init(project=project, location=location)
        model = ImageGenerationModel.from_pretrained(_MODEL_NAME)
        params = _normalize_params(payload)
        response = model.generate_images(prompt=prompt, **params)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surface SDK errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not getattr(response, "images", None):
        raise HTTPException(status_code=400, detail="No image returned from Vertex AI")

    image = response.images[0]
    image_bytes = getattr(image, "image_bytes", None) or getattr(image, "_image_bytes", None)
    if image_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="Image bytes not available in the response.",
        )

    return image_bytes
