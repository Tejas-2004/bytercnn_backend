import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

try:
    from .model import load_model, predict_file
except ImportError:
    from model import load_model, predict_file

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="ByteRCNN File Fragment Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "https://bytercnnbackend-production.up.railway.app",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "EfficientByteRCNN", "classes": 75}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    block_index: Optional[int] = Query(None),
    block_start: Optional[int] = Query(None),
    block_end: Optional[int] = Query(None),
):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Determine which blocks to analyze
    block_indices = None
    if block_index is not None:
        block_indices = [block_index]
    elif block_start is not None and block_end is not None:
        if block_start < 0 or block_end < block_start:
            raise HTTPException(status_code=400, detail="Invalid block range")
        block_indices = list(range(block_start, block_end + 1))

    result = predict_file(file_bytes, block_indices)

    return {
        "filename": file.filename,
        "file_size": len(file_bytes),
        **result
    }
