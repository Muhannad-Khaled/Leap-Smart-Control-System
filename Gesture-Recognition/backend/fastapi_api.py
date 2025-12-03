# fastapi_api.py (للـ Azure فقط)
# trigger rebuild
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_result = {"result": "No prediction yet", "confidence": 0.0}

@app.get("/")
def root():
    return {"message": "GR-SYSTEM backend is up and running!"}

@app.get("/api/latest")
async def get_latest_prediction():
    return latest_result

@app.post("/api/update")
async def update_prediction(request: Request):
    global latest_result
    latest_result = await request.json()
    return {"status": "updated", "data": latest_result}
