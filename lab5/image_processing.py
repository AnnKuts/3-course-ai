from io import BytesIO
import base64

import numpy as np
from PIL import Image
import torch


class EmptyDrawingError(ValueError):
    pass


def preprocess_drawing(image: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    gray = image.convert("L")
    a = np.asarray(gray, dtype=np.uint8)
    ys, xs = np.where(a > 28)
    if not len(xs):
        raise EmptyDrawingError("Полотно порожнє. Намалюйте цифру перед розпізнаванням.")

    left, right = int(xs.min()), int(xs.max() + 1)
    top, bottom = int(ys.min()), int(ys.max() + 1)
    cropped = gray.crop((left, top, right, bottom))
    w, h = cropped.size
    factor = min(20 / w, 20 / h)
    w_new, h_new = max(1, round(w * factor)), max(1, round(h * factor))
    cropped = cropped.resize((w_new, h_new), Image.Resampling.LANCZOS)

    normalized = Image.new("L", (28, 28), 0)
    normalized.paste(cropped, ((28 - w_new) // 2, (28 - h_new) // 2))

    pixels = np.asarray(normalized, dtype=np.float32)
    weight = float(pixels.sum())
    if weight > 0:
        yy, xx = np.indices(pixels.shape)
        shift_x = int(round(13.5 - (pixels * xx).sum() / weight))
        shift_y = int(round(13.5 - (pixels * yy).sum() / weight))
        shifted = np.roll(pixels, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0
        normalized = Image.fromarray(np.asarray(shifted, dtype=np.uint8), "L")

    tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32).copy()) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0), normalized


def decode_canvas_image(data_uri: str) -> Image.Image:
    if not isinstance(data_uri, str) or not data_uri.startswith("data:image/png;base64,"):
        raise ValueError("Очікувався формат data:image/png;base64")
    try:
        raw = base64.b64decode(data_uri.split(",", 1)[1], validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError("Не вдалося декодувати base64") from exc

    if len(raw) > 3_000_000:
        raise ValueError("Зображення завелике")

    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except OSError as exc:
        raise ValueError("Не вдалося відкрити зображення") from exc

    if image.width > 2048 or image.height > 2048:
        raise ValueError("Розміри зображення перевищують ліміт")
    return image


def pil_png_data_uri(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
