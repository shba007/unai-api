from fastapi import FastAPI

# from server.dependencies import get_query_token

from server.routes import health, image, text

# dependencies=[Depends(get_query_token)])
app = FastAPI()

app.include_router(health.router)
app.include_router(image.detect.router)
app.include_router(image.embedding.router)
app.include_router(text.embedding.router)
