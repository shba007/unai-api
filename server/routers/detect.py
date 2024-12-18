import io
import uuid
import base64

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from PIL import Image
import tensorflow as tf

from ..utils.helpers_obj import Data, save_file, upload_file
from ..utils.models import Detector

router = APIRouter(
    prefix="/detect",
    tags=["detect"],
    responses={404: {"description": "Not found"}},
)

detector = Detector(output_format="CCWH")


class ImageRequest(BaseModel):
    image: str


def remove_alpha_channel(image_tensor):
    r, g, b, a = tf.split(image_tensor, 4, axis=-1)
    rgb_tensor = tf.concat([r, g, b], axis=-1)

    return rgb_tensor


@router.post("/")
async def detect(background_tasks: BackgroundTasks, request: ImageRequest):
    try:
        # Generate a random UUID
        id = uuid.uuid4()
        file_path = f"assets/images/{id}.jpg"
        # Decode the base64-encoded image data
        request.image = request.image.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(request.image)
        image_bytes = io.BytesIO(image_data)
        image = Image.open(image_bytes)

        save_file(file_path, image_bytes)
        background_tasks.add_task(upload_file, file_path, f"images/{id}.jpg")

        data = Data("temp-1")
        data.images = {} if data.images is None else data.images
        data.images[id] = image

        img = data.get_images()
        img_width, img_height = img[0]["image"].size

        img = tf.keras.utils.img_to_array(img[0]["image"])
        img = tf.expand_dims(img, axis=0)

        if img.shape[-1] > 3:
            img = remove_alpha_channel(img)

        detections = detector.predict(img)[0]

        boxes = []
        for detection in detections:
            x, y, width, height, conf, label = detection

            if int(label) != 0:
                continue

            x *= 100.0
            y *= 100.0
            width *= 100.0
            height *= 100.0

            boxes.append(
                {"x": x, "y": y, "width": width, "height": height, "conf": conf}
            )

        return {"id": id, "boxes": boxes}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
