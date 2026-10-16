from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import FRONTEND_DIR

app = FastAPI(
    title="Medical Report Analyzer",
    description="Upload blood test reports (PDF/images) for OCR extraction, ML analysis, and LLM summary.",
    version="1.0.0",
)

app.include_router(router, prefix="/api")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Medical Report Analyzer API. Use /api/analyze endpoints."}
