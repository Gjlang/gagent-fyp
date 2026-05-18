from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router


app = FastAPI(
    title="GAgent AI Service",
    description="FastAPI service for UX friction prediction using the trained GAgent ML model.",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "service": "GAgent AI Service",
        "phase": "Phase 5",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/model-info",
            "/predict",
            "/batch-predict",
            "/docs",
        ],
    }