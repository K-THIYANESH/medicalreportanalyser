from fastapi import APIRouter, File, UploadFile

from app.models.schemas import AnalysisResponse
from app.services.analyzer import AnalyzerService
from app.utils.helpers import is_allowed_image, is_allowed_pdf

router = APIRouter()
analyzer = AnalyzerService()


@router.get("/health")
def health():
    return {"status": "ok", "service": "medical-report-analyzer"}


@router.post("/analyze/pdf", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)):
    if not file.filename or not is_allowed_pdf(file.filename):
        return AnalysisResponse(
            success=False,
            error="Invalid file format. Only PDF files are allowed.",
        )

    pdf_bytes = await file.read()
    return analyzer.analyze_pdf(pdf_bytes, file.filename)


@router.post("/analyze/images", response_model=AnalysisResponse)
async def analyze_images(files: list[UploadFile] = File(...)):
    if not files:
        return AnalysisResponse(success=False, error="No image files uploaded.")

    valid_files = [f for f in files if f.filename]
    if not valid_files:
        return AnalysisResponse(success=False, error="No images selected for uploading.")

    for file in valid_files:
        if not is_allowed_image(file.filename):
            return AnalysisResponse(
                success=False,
                error=f"Invalid file format: '{file.filename}'. Only JPG, JPEG, and PNG are allowed.",
            )

    images = []
    for file in valid_files:
        images.append((await file.read(), file.filename))

    return analyzer.analyze_images(images)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(files: list[UploadFile] = File(...)):
    if not files:
        return AnalysisResponse(success=False, error="No files uploaded.")

    valid_files = [f for f in files if f.filename]
    if not valid_files:
        return AnalysisResponse(success=False, error="No files selected for uploading.")

    if len(valid_files) == 1 and is_allowed_pdf(valid_files[0].filename):
        pdf_bytes = await valid_files[0].read()
        return analyzer.analyze_pdf(pdf_bytes, valid_files[0].filename)

    for file in valid_files:
        if not is_allowed_image(file.filename):
            return AnalysisResponse(
                success=False,
                error=(
                    f"Unsupported file '{file.filename}'. "
                    "Upload a single PDF or one/more JPG/PNG images."
                ),
            )

    images = []
    for file in valid_files:
        images.append((await file.read(), file.filename))

    return analyzer.analyze_images(images)
