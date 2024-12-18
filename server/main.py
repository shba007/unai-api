from fastapi import FastAPI

# from .dependencies import get_query_token
from .routers import health, detect, search


app = FastAPI()  # dependencies=[Depends(get_query_token)])

app.include_router(health.router)
app.include_router(detect.router)
app.include_router(search.router)
