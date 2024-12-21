import json
from typing import List, Tuple
from fastapi import APIRouter, HTTPException
import httpx
from nanoid import generate
import numpy as np
from pydantic import BaseModel

from server.core.config import config
from server.utils.base64ToArray import base64_to_array
from server.utils.convertBoxFormat import convert_box_format
from server.utils.imageResize import image_resize
from server.utils.imageScale import image_scale
from server.utils.imageSave import image_save
from server.utils.labelBox import label_box
from server.utils.nonMaxSuppression import nms

DET_DIM = (640, 640)


def preprocess(image: np.ndarray) -> list:
    resized_image, dim = image_resize(image, DET_DIM)
    return [resized_image.tolist(), dim]


def postprocess(
    predictions: List[Tuple[float, float, float, float, float, float]],
    dim: Tuple[int, int],
    conf_threshold: float = 0.25,
) -> List[Tuple[float, float, float, float, float, int]]:
    predictions_array = np.array(predictions)

    xs, ys, ws, hs = predictions_array[:4]
    labels = predictions_array[4:]
    # Compute confidence and label index
    confs = np.max(labels, axis=0)
    label_indices = np.argmax(labels, axis=0)
    mask = confs > conf_threshold

    # Apply confidence threshold
    filtered_boxes = np.column_stack(
        (xs[mask], ys[mask], ws[mask], hs[mask], confs[mask], label_indices[mask])
    )

    nms_boxes = nms(filtered_boxes, iou_threshold=0.5)
    # print("nms_boxes", nms_boxes)

    scaled_boxes = list(
        map(lambda box: [*image_scale(box[:4], dim), *box[4:]], nms_boxes)
    )
    # print("scaled_boxes", scaled_boxes)

    converted_boxes = list(
        map(
            lambda box: [
                *convert_box_format(box[:4], dim, "CCWH", False, "CCWH", True),
                *box[4:],
            ],
            scaled_boxes,
        )
    )
    # print("converted_boxes", converted_boxes)

    labeled_boxes = list(
        map(
            lambda box: {
                "box": box[:4],
                "confidence": box[4],
                "category": label_box(box[5]),
            },
            converted_boxes,
        )
    )
    # print("labeled_boxes", labeled_boxes)

    return labeled_boxes


def predict(image: np.ndarray):
    preprocessed_image, dim = preprocess(image)

    data = json.dumps({"instances": [preprocessed_image]})

    try:
        detections = httpx.post(
            f"{config.tensorflow_api_url}/v1/models/detector:predict",
            data=data,
        )

        return postprocess(detections.json()["predictions"][0], dim)
    except Exception as error:
        print("Failed request Tensorflow Serving /detector:predict", error)
        raise HTTPException(
            status_code=500,
            detail="Failed request Tensorflow Serving /detector:predict",
        )


router = APIRouter(
    prefix="/detect",
    tags=["detect"],
    responses={404: {"description": "Not found"}},
)


class RequestBody(BaseModel):
    image: str


@router.post("/")
async def detect(request: RequestBody):
    try:
        id = generate()
        image_save(id, request.image)

        image_array = base64_to_array(request.image)
        objects = predict(image_array)

        return {"id": id, "objects": objects}
    except HTTPException as error:
        print("API detect POST", error)
        raise error
    except Exception as error:
        print("API detect POST", error)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
