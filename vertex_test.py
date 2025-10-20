import os
from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION   = os.getenv("GCP_LOCATION", "us-central1")
CRED       = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

assert PROJECT_ID, "GCP_PROJECT_ID is required"
assert CRED and os.path.exists(CRED), "GOOGLE_APPLICATION_CREDENTIALS not found"

vertexai.init(project=PROJECT_ID, location=LOCATION)

def generate_imagen(prompt: str, size="1024x1024", out_path="test-imagen3.jpg"):
    w, h = map(int, size.split("x"))
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    res = model.generate_images(prompt=prompt, number_of_images=1, size=f"{w}x{h}", safety_filter_level="block_few")
    img = res.images[0]._image_bytes
    with open(out_path, "wb") as f:
        f.write(img)
    return out_path

if __name__ == "__main__":
    p = "A futuristic smart kitchen with high-tech appliances, studio lighting, product render."
    print("✅ Saved:", generate_imagen(p))
