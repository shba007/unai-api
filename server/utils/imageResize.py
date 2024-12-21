from typing import Tuple

import cv2
import numpy as np


def image_resize(
    image: np.ndarray,
    dim: Tuple[int, int],
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, Tuple[int, int]]:
    # Get original dimensions
    h, w = image.shape[:2]
    target_width, target_height = dim

    # Calculate aspect ratios
    aspect_ratio = w / h
    target_aspect_ratio = target_width / target_height

    # Compute new width and height maintaining aspect ratio
    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image to fit the target dimensions while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height))

    # Add padding (borders) to make it the correct size
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # Add borders using the background color
    bordered_image = cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=background_color,
    )

    return bordered_image, (h, w)
