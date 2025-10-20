import os
import inspect
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def init_vertex():
    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Missing env GCP_PROJECT_ID")
    vertexai.init(project=project, location=location)

def generate_image(prompt: str, size: str = "1024x1024") -> bytes:
    """Generate one image and return JPEG/PNG bytes."""
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    # 解析 size
    try:
        w, h = [int(x) for x in size.lower().split("x")]
    except Exception:
        w, h = 1024, 1024

    # 读取函数签名，决定用哪个参数名
    sig = inspect.signature(model.generate_images).parameters
    kwargs = dict(prompt=prompt, number_of_images=1, safety_filter_level="block_few")

    if "size" in sig:
        kwargs["size"] = f"{w}x{h}"
    elif "image_dimensions" in sig:
        kwargs["image_dimensions"] = (w, h)
    elif "aspect_ratio" in sig:
        # 退而求其次：用长宽比；仅支持常见比例
        ratio = f"{w}:{h}"
        if ratio not in {"1:1", "16:9", "9:16", "4:3", "3:4"}:
            ratio = "1:1"
        kwargs["aspect_ratio"] = ratio
    # 若都没有，则使用 SDK 默认尺寸

    res = model.generate_images(**kwargs)
    return res.images[0]._image_bytes
