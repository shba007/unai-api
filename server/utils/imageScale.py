from typing import Tuple


def image_scale(
    box: Tuple[float, float, float, float], dim: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    x_center, y_center, width, height, *_ = box

    DET_DIM = (640, 640)  # Example DET_DIM value
    factor = max(dim) / max(DET_DIM)
    offset = abs(dim[0] - dim[1])

    x_center *= factor
    y_center *= factor
    width *= factor
    height *= factor

    if dim[0] > dim[1]:
        y_center -= offset / 2
    elif dim[0] < dim[1]:
        x_center -= offset / 2

    scaled_box = [x_center, y_center, width, height]

    return scaled_box
