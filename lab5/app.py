from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from image_processing import EmptyDrawingError, decode_canvas_image, pil_png_data_uri, preprocess_drawing
from network import DigitCNN

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "model" / "mnist_cnn.pt"

app = FastAPI(title="Digit Lab")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PredictRequest(BaseModel):
    image: str


@lru_cache(maxsize=1)
def _load_model(mtime: float):
    model = DigitCNN()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    return model


def get_model():
    if not WEIGHTS.exists():
        raise FileNotFoundError("Модель не навчена: виконайте python train.py")
    return _load_model(WEIGHTS.stat().st_mtime)


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/status")
def status():
    return {"trained": WEIGHTS.exists()}


@app.post("/predict")
def predict(req: PredictRequest):
    if not WEIGHTS.exists():
        raise HTTPException(status_code=503, detail="Модель не навчена: виконайте python train.py")
    try:
        source = decode_canvas_image(req.image)
        x, normalized = preprocess_drawing(source)
    except (ValueError, EmptyDrawingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with torch.inference_mode():
        probs = torch.softmax(get_model()(x), dim=1)[0].tolist()
    label = int(max(range(10), key=lambda i: probs[i]))
    return {
        "digit": label,
        "confidence": round(probs[label] * 100, 2),
        "probabilities": [round(p * 100, 2) for p in probs],
        "processed": pil_png_data_uri(normalized),
    }
