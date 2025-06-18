import os
from typing import Optional
from fastapi import APIRouter
from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    version: Optional[str] = None
    build_time: Optional[str] = None

    class Config:
        env_prefix = "FASTAPI_APP_"

app_config = AppConfig()

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_health():
    node = os.getenv("HOSTNAME", "unknown-node")

    return {
        "status": "OK",
        **app_config.dict(),
        "node": node,
    }
