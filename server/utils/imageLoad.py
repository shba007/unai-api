from PIL import Image
import numpy as np


def image_load(id: str, input_path: str = "./assets/images") -> np.ndarray:
    try:
        image_array = Image.open(f"{input_path}/{id}.jpg")
        image_array = np.array(image_array)

        print(f"Image load successfully to {input_path}/{id}.jpg")
        return image_array
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
