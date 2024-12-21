from enum import Enum


class BoxFormat(Enum):
    CCWH = "CCWH"
    XYWH = "XYWH"
    XYXY = "XYXY"


def convert_box_format(
    init_box: list[float],
    image_dim,
    init_format: BoxFormat = BoxFormat.XYXY,  # BoxFormat enum
    init_normalized: bool = False,
    final_format: BoxFormat = BoxFormat.CCWH,  # BoxFormat enum
    final_normalized: bool = False,
    is_debug: bool = False,
) -> tuple[float, float, float, float]:
    """
    Converts bounding box formats between CCWH, XYWH, and XYXY.

    Parameters:
    - init_box: List of initial box coordinates [x1, y1, x2, y2]
    - image_dim: Tuple of image dimensions (width, height)
    - init_format: Format of the input box ('CCWH', 'XYWH', 'XYXY')
    - init_normalized: Whether input coordinates are normalized
    - final_format: Desired output format ('CCWH', 'XYWH', 'XYXY')
    - final_normalized: Whether output coordinates should be normalized
    - is_debug: Print debug information if True

    Returns:
    - final_box: List of converted box coordinates [x1, y1, x2, y2]
    """
    final_box = [0, 0, 0, 0]
    img_width, img_height = image_dim

    if init_normalized:
        init_box[0] *= img_width
        init_box[1] *= img_height
        init_box[2] *= img_width
        init_box[3] *= img_height

    if init_format == "XYXY":
        x_center = (init_box[0] + init_box[2]) / 2
        y_center = (init_box[1] + init_box[3]) / 2
        width = init_box[2] - init_box[0]
        height = init_box[3] - init_box[1]
    elif init_format == "XYWH":
        x_center = init_box[0] + init_box[2] / 2
        y_center = init_box[1] + init_box[3] / 2
        width = init_box[2]
        height = init_box[3]
    else:  # 'CCWH'
        x_center, y_center, width, height = init_box

    if is_debug:
        print(
            {
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }
        )

    if final_format == "XYXY":
        final_box[0] = x_center - width / 2
        final_box[1] = y_center - height / 2
        final_box[2] = x_center + width / 2
        final_box[3] = y_center + height / 2
    elif final_format == "XYWH":
        final_box[0] = x_center - width / 2
        final_box[1] = y_center - height / 2
        final_box[2] = width
        final_box[3] = height
    else:  # 'CCWH'
        final_box[0] = x_center
        final_box[1] = y_center
        final_box[2] = width
        final_box[3] = height

    if final_normalized:
        final_box[0] /= img_width
        final_box[1] /= img_height
        final_box[2] /= img_width
        final_box[3] /= img_height

    if is_debug:
        print({"final_box": final_box})

    return final_box
