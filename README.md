# Medical Report Analyzer

End-to-end pipeline for blood test report analysis:

**Upload PDF or images → OCR extraction → JSON biomarkers → ML disease risk → LLM summary**

## Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐    ┌─────────────┐
│ PDF / Images│ -> │ OCR Pipeline │ -> │  JSON CBC   │ -> │ ML Predict │ -> │ LLM Summary │
│   Upload    │    │ PaddleOCR +  │    │  Biomarkers │    │ RandomForest│    │   Ollama    │
│             │    │   PyMuPDF    │    │             │    │            │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘    └─────────────┘
```

## Project Structure

```
medical-report-analyser/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── config.py            # Paths, model names, env settings
│   ├── api/
│   │   └── routes.py        # REST endpoints
│   ├── services/
│   │   └── analyzer.py      # Orchestrates OCR → ML → LLM
│   ├── ocr/
│   │   ├── pipeline.py      # PDF/image OCR (PaddleOCR + PyMuPDF)
│   │   └── extractor.py     # CBC biomarker extraction from text
│   ├── analysis/
│   │   ├── ml.py            # RandomForest disease risk prediction
│   │   └── llm.py           # Ollama LLM summarization
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── utils/
│       └── helpers.py       # Logging, file validation
├── data/
│   └── dataset.csv          # Training data for ML models
├── frontend/
│   └── index.html           # Web UI (multi-file upload)
├── tests/                   # Unit and integration tests
├── scripts/
│   └── run_on_image.py      # CLI tool for single-image OCR test
├── requirements.txt
├── run.py                   # Start the server
└── .env.example
```

## Prerequisites

- **Python 3.10–3.13** (required — **Python 3.14 is not supported** by PaddlePaddle/OCR)
- **Ollama** running locally with the LLM model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Setup (Windows)

**Option A — automated (recommended):**
```powershell
cd medical-report-analyser
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

**Option B — manual:**
```powershell
cd medical-report-analyser
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr
copy .env.example .env        # optional: customize LLM model / port
```

> **Note:** Do not use the system `python`/`pip` if it points to Python 3.14. Always activate `.venv` created with `py -3.10`.

## Run

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Auto-detect: single PDF or multiple images |
| POST | `/api/analyze/pdf` | Analyze a single PDF |
| POST | `/api/analyze/images` | Analyze one or more images |

### Example: Analyze PDF

```bash
curl -X POST -F "file=@report.pdf" http://localhost:8000/api/analyze/pdf
```

### Example: Analyze multiple images

```bash
curl -X POST -F "files=@page1.png" -F "files=@page2.jpg" http://localhost:8000/api/analyze/images
```

### Response

```json
{
  "success": true,
  "data": {
    "hemoglobin": 12.5,
    "rbc": 5.2,
    "wbc": 9000,
    "platelets": 150000
  },
  "ml_predictions": {
    "Anemia": 12.0,
    "Infection": 5.0,
    "Thrombocytopenia": 3.0,
    "Leukocytosis": 2.0
  },
  "summary": "Your hemoglobin and blood cell counts appear within typical ranges...",
  "metadata": {
    "filename": "report.pdf",
    "method": "pdf_text",
    "line_count": 42,
    "source_type": "pdf",
    "ml_status": "completed"
  }
}
```

## Tests

```bash
pytest tests/ -v
```

## CLI: Test OCR on a single image

```bash
python scripts/run_on_image.py path/to/report.jpg
```

## Biomarkers Extracted

| Biomarker | Aliases | Type |
|-----------|---------|------|
| Hemoglobin | hemoglobin, hb, hgb | float (g/dL) |
| RBC | total rbc count, rbc count, rbc | float |
| WBC | white blood cell count, wbc count, wbc | int |
| Platelets | platelet count, platelets, plt | int |

## ML Disease Risk Models

RandomForest classifiers trained on `data/dataset.csv` predict probability for:

- Anemia
- Infection
- Thrombocytopenia
- Leukocytosis

## Notes

- Scanned PDFs are automatically detected and processed with PaddleOCR.
- ML prediction runs only when all four biomarkers are extracted.
- LLM summary requires Ollama to be running locally.
- This tool is for educational/demo purposes — not a substitute for professional medical diagnosis.
