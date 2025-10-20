import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def init_vertex():
    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Missing env GCP_PROJECT_ID")
    vertexai.init(project=project, location=location)

def generate_image(prompt: str, size="1024x1024") -> bytes:
    w, h = map(int, size.split("x"))
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    res = model.generate_images(prompt=prompt, number_of_images=1, size=f"{w}x{h}", safety_filter_level="block_few")
    return res.images[0]._image_bytes
