# backend/main.py
from fastapi import FastAPI # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LATEST_FILE = "latest_prediction.json"

@app.get("/api/latest")
async def get_latest_prediction():
    if os.path.exists(LATEST_FILE):
        with open(LATEST_FILE, "r") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                return JSONResponse(status_code=500, content={"error": "Corrupted prediction file."})
    return JSONResponse(status_code=404, content={"error": "Prediction not available."})
