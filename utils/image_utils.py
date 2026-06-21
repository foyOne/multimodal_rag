import base64
import io
import math
from pathlib import Path

from PIL import Image


def scale_image(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    w, h = image.size
    num_pixels = w * h

    if min_pixels <= num_pixels <= max_pixels:
        return image

    round_fn = math.ceil if num_pixels < min_pixels else math.floor
    target_pixels = min_pixels if num_pixels < min_pixels else max_pixels

    scale = 1 / math.sqrt(num_pixels / target_pixels)
    scaled_w = int(round_fn(w * scale))
    scaled_h = int(round_fn(h * scale))

    return image.resize(size=(scaled_w, scaled_h), resample=resample)


def image_path_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as stream:
        return base64.b64encode(stream.read()).decode("utf-8")


def image_to_base64(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(image_base64: str) -> Image.Image:
    img_bytes = base64.b64decode(image_base64)
    buffer = io.BytesIO(img_bytes)
    return Image.open(buffer)
