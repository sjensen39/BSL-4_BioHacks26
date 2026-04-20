from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models import AskRequest, AskResponse, DataUploadResponse
from app.services.data import build_dataset_record
from app.services.tutor import build_analysis
from app.state import DATA_STORE

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "viewer_v2.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": settings.app_name}


@app.post("/api/upload-data", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    description: str = Form(...),
) -> DataUploadResponse:
    if not description.strip():
        raise HTTPException(status_code=400, detail="Describe what the dataset is about.")

    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file only.")

    try:
        data = await file.read()
        record = build_dataset_record(filename=filename, description=description, data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not process the dataset: {exc}") from exc

    DATA_STORE[record["data_id"]] = record
    return DataUploadResponse(
        data_id=record["data_id"],
        filename=record["filename"],
        description=record["description"],
        row_count=record["row_count"],
        column_count=record["column_count"],
        columns=record["columns"],
        numeric_columns=record["numeric_columns"],
        categorical_columns=record["categorical_columns"],
        datetime_columns=record["datetime_columns"],
        preview_rows=record["preview_rows"],
        overview_bullets=record["overview_bullets"],
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    record = DATA_STORE.get(request.data_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload the CSV again.")

    result = build_analysis(record, request.question)
    return AskResponse(**result)
