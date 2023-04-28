import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.health import router as health_router
from routes.detect import router as detect_router
from routes.search import router as search_router

PRESET = os.getenv("PRESET")
PRESET = "deploy" if PRESET == None else PRESET
ORIGINS = os.getenv("CORS_URL")
ORIGINS = ["*"] if ORIGINS == None else ORIGINS

app = FastAPI(openapi_url=None if PRESET == "deploy" else "/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(detect_router)
app.include_router(search_router)
