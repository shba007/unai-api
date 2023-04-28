import io
import uuid
import base64

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from PIL import Image
import tensorflow as tf

from utils.helpers import Data, save_file, upload_file
from utils.models import Detector

router = APIRouter()

detector = Detector(output_format="CCWH")


class ImageRequest(BaseModel):
    image: str


def remove_alpha_channel(image_tensor):
    r, g, b, a = tf.split(image_tensor, 4, axis=-1)
    rgb_tensor = tf.concat([r, g, b], axis=-1)

    return rgb_tensor


@router.post("/detect")
async def detect(background_tasks: BackgroundTasks, request: ImageRequest):
    try:
        # Generate a random UUID
        id = uuid.uuid4()
        file_path = f"assets/images/{id}.jpg"
        # Decode the base64-encoded image data
        request.image = request.image.replace('data:image/jpeg;base64,', '')
        image_data = base64.b64decode(request.image)
        image_bytes = io.BytesIO(image_data)
        image = Image.open(image_bytes)

        save_file(file_path, image_bytes)
        background_tasks.add_task(upload_file, file_path, f"images/{id}.jpg")

        data = Data("temp-1")
        data.images = {} if data.images == None else data.images
        data.images[id] = image

        det_img = data.get_images()
        img_width, img_height = det_img[0]["image"].size

        det_img = tf.keras.utils.img_to_array(det_img[0]["image"])
        det_img = tf.expand_dims(det_img, axis=0)

        if (det_img.shape[-1] > 3):
            det_img = remove_alpha_channel(det_img)

        detections = detector.predict(det_img)[0]

        boxes = []
        for detection in detections:
            x, y, width, height, conf, label = detection

            if (int(label) != 0):
                continue

            x /= img_width
            y /= img_height
            width /= img_width
            height /= img_height

            x *= 100.0
            y *= 100.0
            width *= 100.0
            height *= 100.0
            conf *= 100.0

            boxes.append({
                "x": x,
                "y": y,
                "width":  width,
                "height": height,
                "conf": conf
            })

        return {
            "id": id,
            "boxes": boxes
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")