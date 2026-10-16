from typing import Dict, List, Optional
import re

import httpx
import ollama

from app.config import LLM_FALLBACK_MODELS, LLM_MODEL
from app.models.schemas import CBCResult
from app.utils.helpers import logger


class Summary:
    @staticmethod
    def _build_report(cbc: CBCResult, ml_predictions: Optional[Dict[str, float]] = None) -> str:
        lines = ["Extracted CBC biomarkers:"]
        labels = {
            "hemoglobin": "Hemoglobin (g/dL)",
            "rbc": "RBC (mill/cumm)",
            "wbc": "WBC (/cumm)",
            "platelets": "Platelets (/cumm)",
        }
        for key, label in labels.items():
            value = getattr(cbc, key)
            lines.append(f"  - {label}: {value if value is not None else 'not detected'}")

        if ml_predictions:
            lines.append("\nML disease risk probabilities:")
            for disease, prob in ml_predictions.items():
                lines.append(f"  - {disease}: {prob}%")

        return "\n".join(lines)

    @staticmethod
    def _models_to_try() -> List[str]:
        ordered: List[str] = []
        for model in [LLM_MODEL, *LLM_FALLBACK_MODELS]:
            if model and model not in ordered:
                ordered.append(model)
        return ordered

    @staticmethod
    def _is_model_missing_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "not found" in message or "404" in message

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = text.strip()
        intro_patterns = [
            r"^here['']?s?(?: a)?(?: patient[- ]friendly)? summary:?\s*",
            r"^patient[- ]friendly summary:?\s*",
            r"^summary:?\s*",
        ]
        for pattern in intro_patterns:
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def summarize(cbc: CBCResult, ml_predictions: Optional[Dict[str, float]] = None) -> dict:
        report = Summary._build_report(cbc, ml_predictions)
        prompt = f"""You are a medical report assistant. Based on the following CBC blood test data,
write a clear, patient-friendly summary in 4-5 sentences.

Explain what the values mean in plain language, mention any notable risk flags from the ML analysis,
and recommend consulting a doctor for proper diagnosis. Do NOT diagnose definitively.

Start directly with the summary content. Do not include a title, heading, or introduction
(e.g. do not write "Here's a patient-friendly summary:" or similar).

{report}
"""
        last_error = None
        for model in Summary._models_to_try():
            try:
                logger.info(f"Generating summary with Ollama model: {model}")
                response = ollama.generate(model=model, prompt=prompt)
                summary = Summary._clean_summary(response["response"])
                return {"success": True, "summary": summary, "model_used": model}
            except httpx.ConnectError as exc:
                return {
                    "success": False,
                    "error": (
                        f"Ollama is not running. Start Ollama, then pull a model "
                        f"(e.g. ollama pull llama3). Details: {exc}"
                    ),
                }
            except Exception as exc:
                last_error = exc
                if Summary._is_model_missing_error(exc):
                    logger.warning(f"Ollama model '{model}' not available: {exc}")
                    continue
                return {"success": False, "error": str(exc)}

        tried = ", ".join(Summary._models_to_try())
        return {
            "success": False,
            "error": (
                f"No Ollama model available. Tried: {tried}. "
                f"Install one with: ollama pull llama3"
            ),
            "details": str(last_error) if last_error else None,
        }
