import asyncio
import json
import math
from typing import List, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import numpy as np
from PIL import Image

from server.core.config import config
from server.utils.convertBoxFormat import convert_box_format
from server.utils.imageLoad import image_load
from server.utils.imageResize import image_resize

CLASS_DIM = [256, 256]


async def preprocess(images: List[np.ndarray]) -> List[List[float]]:
    async def process_image(image: np.ndarray) -> List[float]:
        resized_image = image.tolist()
        return resized_image

    resized_images = await asyncio.gather(*[process_image(image) for image in images])
    # print({"resized_images": resized_images})

    return resized_images


def postprocess(embeddings):
    return embeddings


async def predict(image: np.ndarray, boxes: list):
    image_crops = await crop_image(image, boxes)
    preprocessed_images = await preprocess(image_crops)
    data = json.dumps({"instances": preprocessed_images})

    try:
        detections = httpx.post(
            f"{config.tensorflow_api_url}/v1/models/feature_extractor:predict",
            data=data,
        )

        embeddings = postprocess(detections.json()["predictions"])

        return embeddings
    except Exception as error:
        print("Failed request Tensorflow Serving /feature_extractor:predict", error)
        raise HTTPException(
            status_code=500,
            detail="Failed request Tensorflow Serving /feature_extractor:predict",
        )


async def crop_image(
    image: np.ndarray, boxes: Tuple[float, float, float, float]
) -> np.ndarray:
    img = Image.fromarray(image)

    def process_box(box):
        converted_box = convert_box_format(box, img.size, "CCWH", True, "XYWH", False)
        converted_box = [math.floor(num) for num in converted_box]

        cropped_image, dim = image_resize(
            np.array(
                img.crop(
                    (
                        converted_box[0],
                        converted_box[1],
                        converted_box[0] + converted_box[2],
                        converted_box[1] + converted_box[3],
                    )
                )
            ),
            CLASS_DIM,
        )

        return cropped_image

    tasks = [asyncio.to_thread(process_box, box) for box in boxes]
    cropped_images = await asyncio.gather(*tasks)
    return np.array(cropped_images)


router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


class Object(BaseModel):
    box: List[float]
    confidence: float
    category: str


class RequestBody(BaseModel):
    id: str
    objects: List[Object]


@router.post("/")
async def search(request: RequestBody):
    try:
        image = image_load(request.id)

        searches = await predict(
            image, list(map(lambda object: object.box, request.objects))
        )

        return {"id": request.id, "searches": searches}
    except HTTPException as error:
        print("API search POST", error)
        raise error
    except Exception as error:
        print("API search POST", error)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
