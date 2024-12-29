from typing import List
from fastapi import HTTPException
from pydantic import BaseModel
from transformers import AutoModel

from .router import router

model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)


class RequestBody(BaseModel):
    id: str
    queries: List[str]


@router.post("/embedding")
async def get_embedding(request: RequestBody):
    try:
        truncate_dim = 512

        embeddings = model.encode_text(request.queries, truncate_dim=truncate_dim)

        return {
            "id": request.id,
            "embeddings": [[float(x) for x in embedding] for embedding in embeddings],
        }
        return
    except HTTPException as error:
        print("API similarity POST", error)
        raise error
    except Exception as error:
        print("API similarity POST", error)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
