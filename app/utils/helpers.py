import logging
import os

from app.config import ALLOWED_IMAGE_EXTENSIONS, ALLOWED_PDF_EXTENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger("medical_report_analyzer")


def get_file_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower()


def is_allowed_pdf(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_PDF_EXTENSIONS


def is_allowed_image(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS
