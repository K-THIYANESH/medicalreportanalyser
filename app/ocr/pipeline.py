import os
import cv2
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple, Any, Optional
from app.utils.helpers import logger

# Singleton class to load PaddleOCR only once
class PaddleOCRManager:
    _instance: Optional['PaddleOCRManager'] = None
    _ocr_engine = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PaddleOCRManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def get_ocr(self):
        if self._ocr_engine is None:
            # We import PaddleOCR lazily here so that if we only extract text from
            # a digital PDF, we don't load paddleocr unless we actually run image/scanned OCR.
            logger.info("Initializing PaddleOCR engine...")
            try:
                import os
                os.environ["FLAGS_use_mkldnn"] = "0"
                os.environ["FLAGS_use_onednn"] = "0"
                from paddleocr import PaddleOCR
                # use_angle_cls=True helps detect and correct text orientation
                # lang='en' since medical reports are in English
                # enable_mkldnn=False disables unsupported CPU extensions in some environments
                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
                logger.info("PaddleOCR engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR engine: {e}")
                raise e
        return self._ocr_engine

def group_words_into_lines(words: List[Tuple[float, float, float, float, str, int, int, int]]) -> List[str]:
    """
    Groups individual PDF words into horizontal lines by matching vertical coordinates.
    Each word is: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    """
    if not words:
        return []
        
    # Sort words by top y0 coordinate first, then left x0 coordinate
    sorted_words = sorted(words, key=lambda w: (w[1], w[0]))
    
    lines: List[str] = []
    current_line_words: List[Tuple[float, float, float, float, str, int, int, int]] = []
    current_y0: Optional[float] = None
    current_y1: Optional[float] = None
    
    for w in sorted_words:
        x0, y0, x1, y1, word, _, _, _ = w
        h = y1 - y0
        
        if current_y0 is None:
            current_y0 = y0
            current_y1 = y1
            current_line_words.append(w)
        else:
            # Compute vertical overlap
            overlap_y0 = max(y0, current_y0)
            overlap_y1 = min(y1, current_y1)
            overlap = overlap_y1 - overlap_y0
            min_h = min(h, current_y1 - current_y0)
            
            # If vertical overlap is more than 30% or vertical difference of top is small, group them
            if (overlap > 0.3 * min_h) or (abs(y0 - current_y0) < 8):
                current_line_words.append(w)
                current_y0 = min(current_y0, y0)
                current_y1 = max(current_y1, y1)
            else:
                # Finished line: sort horizontally (by x0) and convert to string
                current_line_words = sorted(current_line_words, key=lambda x: x[0])
                lines.append(" ".join(x[4] for x in current_line_words))
                # Reset for new line
                current_line_words = [w]
                current_y0 = y0
                current_y1 = y1
                
    if current_line_words:
        current_line_words = sorted(current_line_words, key=lambda x: x[0])
        lines.append(" ".join(x[4] for x in current_line_words))
        
    return lines

def group_ocr_blocks_into_lines(ocr_result: List[Any]) -> List[str]:
    """
    Groups OCR bounding boxes and text blocks into horizontal lines.
    Handles both legacy (list-of-lists) and modern (dict-based) PaddleOCR formats.
    """
    if not ocr_result:
        return []
        
    blocks = []
    for item in ocr_result:
        if item is None:
            continue
            
        # Format 1: dict-based (PaddleOCR 3.x / PaddleX)
        if isinstance(item, dict):
            texts = item.get("rec_texts", [])
            boxes = item.get("rec_boxes", [])
            for idx, text in enumerate(texts):
                if idx >= len(boxes):
                    break
                box = boxes[idx]
                # If box is [x_min, y_min, x_max, y_max]
                if len(box) == 4 and not hasattr(box[0], "__len__"):
                    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                else:
                    # box is array of points [[x, y], [x, y], ...]
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                blocks.append((x0, y0, x1, y1, text))
                
        # Format 2: legacy list-based (PaddleOCR 2.x)
        elif isinstance(item, list):
            for block in item:
                if isinstance(block, (list, tuple)) and len(block) == 2:
                    box, val = block
                    if isinstance(val, (list, tuple)) and len(val) >= 1:
                        text = val[0]
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)
                        blocks.append((x0, y0, x1, y1, text))
            
    if not blocks:
        return []
        
    # Sort blocks by vertical top first, then horizontal left
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    
    lines = []
    current_line_blocks = []
    current_y0 = None
    current_y1 = None
    
    for b in blocks:
        x0, y0, x1, y1, text = b
        h = y1 - y0
        
        if current_y0 is None:
            current_y0 = y0
            current_y1 = y1
            current_line_blocks.append(b)
        else:
            overlap_y0 = max(y0, current_y0)
            overlap_y1 = min(y1, current_y1)
            overlap = overlap_y1 - overlap_y0
            min_h = min(h, current_y1 - current_y0)
            
            # Since OCR bounding boxes might be slightly wavy, vertical center proximity is also checked
            center_y = (y0 + y1) / 2
            current_center_y = (current_y0 + current_y1) / 2
            
            if (overlap > 0.3 * min_h) or (abs(center_y - current_center_y) < 15):
                current_line_blocks.append(b)
                current_y0 = min(current_y0, y0)
                current_y1 = max(current_y1, y1)
            else:
                current_line_blocks = sorted(current_line_blocks, key=lambda x: x[0])
                lines.append(" ".join(x[4] for x in current_line_blocks))
                
                current_line_blocks = [b]
                current_y0 = y0
                current_y1 = y1
                
    if current_line_blocks:
        current_line_blocks = sorted(current_line_blocks, key=lambda x: x[0])
        lines.append(" ".join(x[4] for x in current_line_blocks))
        
    return lines

def preprocess_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Converts image to grayscale, and thresholds it to filter out light watermarks
    and background noise. Returns a 3-channel image suitable for PaddleOCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold: values > 200 become 255 (white). This cleans out light-colored
    # background watermarks while keeping the main text crisp.
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed

class OCRPipeline:
    def __init__(self):
        self.ocr_manager = PaddleOCRManager()

    def extract_from_image_bytes(self, image_bytes: bytes) -> List[str]:
        """Runs OCR on raw image bytes and groups text into horizontal lines."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file or decoding failed.")
            
        # Apply image preprocessing to filter background watermarks
        img = preprocess_image_for_ocr(img)
        
        ocr = self.ocr_manager.get_ocr()
        ocr_result = ocr.ocr(img)
        return group_ocr_blocks_into_lines(ocr_result)

    def extract_from_pdf_bytes(self, pdf_bytes: bytes) -> Tuple[List[str], str]:
        """
        Extracts text lines from PDF bytes.
        Attempts native text extraction first. If it's a scanned PDF (little or no text),
        it falls back to rendering PDF pages as images and running PaddleOCR.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_lines: List[str] = []
        method = "pdf_text"
        
        # Check if PDF contains searchable text by sampling the first few pages
        total_words = 0
        for page_num in range(min(5, len(doc))):
            words = doc[page_num].get_text("words")
            total_words += len(words)
            
        # If total words across sampled pages is low, treat as scanned PDF
        if total_words < 10 * min(5, len(doc)):
            logger.info("PDF appears to be scanned. Falling back to rendering pages and running OCR.")
            method = "pdf_ocr"
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to an image. 150 DPI is standard for text extraction
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                
                # Run OCR on page image
                page_lines = self.extract_from_image_bytes(img_data)
                all_lines.extend(page_lines)
        else:
            logger.info("Extracting native text from PDF.")
            for page_num in range(len(doc)):
                page = doc[page_num]
                words = page.get_text("words")
                page_lines = group_words_into_lines(words)
                all_lines.extend(page_lines)
                
        doc.close()
        return all_lines, method
