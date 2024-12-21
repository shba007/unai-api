import base64
from io import BytesIO
from PIL import Image
import numpy as np


def base64_to_array(encoded_image: str) -> np.ndarray:
    """
    Convert a Base64-encoded image string to a NumPy array.

    Args:
        base64_string (str): Base64-encoded image string.

    Returns:
        np.ndarray: Image represented as a NumPy array.
    """
    # Remove the data URL scheme if present
    if encoded_image.startswith("data:image"):
        encoded_image = encoded_image.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(encoded_image)

    # Convert bytes to a PIL Image
    image = Image.open(BytesIO(image_bytes))

    return np.array(image, np.uint8)
