import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_circle(canvas, x, y, circle_type, value=None):
    draw = ImageDraw.Draw(canvas)

    if circle_type == "alignment":
        draw.ellipse(
            (x - 27.5, y - 27.5, x + 27.5, y + 27.5), outline=(34, 197, 94), width=7
        )
    else:
        color_map = [
            (225, 29, 72),
            (192, 38, 211),
            (147, 51, 234),
            (79, 70, 229),
            (96, 165, 250),
        ]
        color = color_map[value] if value is not None else color_map[0]
        draw.ellipse(
            (x - 12.5, y - 12.5, x + 12.5, y + 12.5),
            fill=color,
            outline=(0, 0, 0),
            width=3,
        )


def get_highlights(image, option_count, inputs, responses):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    is_alignment = True
    is_response = True if responses is not None else False

    inputs = [[choice["chord"] for choice in input["choices"]] for input in inputs]
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for q_index, input in enumerate(inputs):
        for d_index, dot in enumerate(input):
            if dot is None:
                continue

            x, y = dot

            if is_alignment:
                draw_circle(canvas, x, y, "alignment")

            if is_response:
                choice = responses[q_index]["value"] if responses is not None else None
                if choice is not None:
                    if option_count == 2 and choice == 1 - d_index:
                        draw_circle(canvas, x, y, "response", choice * 4)
                    elif option_count == 5 and choice == d_index:
                        draw_circle(canvas, x, y, "response", choice)

    canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    height, width = canvas.shape[:2]
    new_width = int((720 / height) * width)
    canvas = cv2.resize(canvas, (new_width, 720))

    # cv2.imshow("highlighted", canvas)
    # cv2.waitKey(0)

    _, buffer = cv2.imencode(".jpg", canvas)
    image_base64 = base64.b64encode(buffer)
    image_str = image_base64.decode("utf-8")

    return f"data:image/jpeg;base64,{image_str}"
