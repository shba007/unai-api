from fastapi import FastAPI

# from server.dependencies import get_query_token
from server.routes import health, detect, omr, search

app = FastAPI()  # dependencies=[Depends(get_query_token)])

app.include_router(health.router)
app.include_router(detect.router)
app.include_router(search.router)
app.include_router(omr.router)
