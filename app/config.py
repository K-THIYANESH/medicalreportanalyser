import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"
DATASET_PATH = DATA_DIR / "dataset.csv"

LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "LLM_FALLBACK_MODELS",
        "llama3:latest,deepseek-r1:1.5b,qwen2.5:7b",
    ).split(",")
    if m.strip()
]
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
