from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from app.analysis.llm import Summary
from app.analysis.ml import Prediction
from app.models.schemas import AnalysisResponse, CBCResult
from app.ocr.extractor import extract_biomarkers
from app.ocr.pipeline import OCRPipeline
from app.utils.helpers import logger

BIOMARKER_KEYS = ("hemoglobin", "wbc", "rbc", "platelets")


class AnalyzerService:
    def __init__(self):
        self.pipeline = OCRPipeline()
        self.predictor = Prediction()
        self.summarizer = Summary()

    def analyze_pdf(self, pdf_bytes: bytes, filename: str) -> AnalysisResponse:
        lines, method = self.pipeline.extract_from_pdf_bytes(pdf_bytes)
        metadata = {
            "filename": filename,
            "method": method,
            "line_count": len(lines),
            "source_type": "pdf",
        }
        return self._analyze_lines(lines, metadata)

    def analyze_images(self, images: List[Tuple[bytes, str]]) -> AnalysisResponse:
        combined_lines: List[str] = []
        for img_bytes, filename in images:
            logger.info(f"Processing image: '{filename}' ({len(img_bytes)} bytes)")
            page_lines = self.pipeline.extract_from_image_bytes(img_bytes)
            combined_lines.extend(page_lines)

        metadata = {
            "images_processed": len(images),
            "filenames": [name for _, name in images],
            "line_count": len(combined_lines),
            "source_type": "images",
        }
        return self._analyze_lines(combined_lines, metadata)

    def _analyze_lines(self, lines: List[str], metadata: Dict[str, Any]) -> AnalysisResponse:
        try:
            extracted = extract_biomarkers(lines)
            cbc = CBCResult(**extracted)

            ml_predictions = None
            if all(extracted.get(key) is not None for key in BIOMARKER_KEYS):
                sample = [[
                    extracted["hemoglobin"],
                    extracted["wbc"],
                    extracted["rbc"],
                    extracted["platelets"],
                ]]
                ml_predictions = self.predictor.predict(sample)
                metadata["ml_status"] = "completed"
            else:
                missing = [key for key in BIOMARKER_KEYS if extracted.get(key) is None]
                metadata["ml_status"] = "skipped"
                metadata["ml_skip_reason"] = f"Missing biomarkers: {', '.join(missing)}"

            summary_result = self.summarizer.summarize(cbc, ml_predictions)
            if summary_result["success"]:
                metadata["llm_status"] = "completed"
                metadata["llm_model"] = summary_result.get("model_used")
            else:
                metadata["llm_status"] = "failed"
                metadata["llm_error"] = summary_result.get("error")

            return AnalysisResponse(
                success=True,
                data=cbc,
                ml_predictions=ml_predictions,
                summary=summary_result.get("summary"),
                error=None if summary_result["success"] else summary_result.get("error"),
                metadata=metadata,
            )
        except ValidationError as exc:
            logger.error(f"Validation error during analysis: {exc}")
            return AnalysisResponse(success=False, error=f"Data validation failed: {exc}", metadata=metadata)
        except Exception as exc:
            logger.error(f"Analysis pipeline failed: {exc}")
            return AnalysisResponse(success=False, error=str(exc), metadata=metadata)
