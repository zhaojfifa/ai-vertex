"""FastAPI application exposing a Vertex Imagen 3 generation service."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.vertex_client import generate_image

app = FastAPI(title="Vertex Imagen Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, bool]:
    """Return a simple health indicator."""
    return {"ok": True}


@app.post("/generate")
async def generate_endpoint(body: Dict[str, Any] = Body(...)) -> Response:
    """Generate an image given a JSON payload and return it as image/jpeg."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    params = {key: value for key, value in body.items() if key != "prompt"}

    try:
        image_bytes = generate_image(prompt, **params)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=image_bytes, media_type="image/jpeg")
