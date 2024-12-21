import cv2
import numpy as np


def calculate_bw_ratio(image):
    # Threshold the image to get binary image with white pixels
    _, binary = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)

    # Count the white pixels
    num_white_pixels = np.count_nonzero(binary == 255)

    # Calculate the ratio of white pixels to total pixels
    height, width = binary.shape
    num_pixels = width * height
    white_ratio = num_white_pixels / num_pixels

    return white_ratio


def extract_data(image, inputs):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow("align_inputs", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    factor = 4
    threshold = 0.12

    results = []
    for input_data in inputs:
        index = input_data["index"]
        choices = input_data["choices"]

        bw_ratios = []
        for choice in choices:
            chord = choice["chord"]
            if chord is None:
                continue

            x, y = chord
            w, h = 12, 12
            left = int(x - (w / 2) * factor)
            top = int(y - (h / 2) * factor)
            width = int(w * factor)
            height = int(h * factor)

            crop_image = image[top : top + height, left : left + width]

            # cv2.imwrite(f'./temp/inputs/{index}-{choice["name"]}.jpg', crop_image)

            bw_ratio = calculate_bw_ratio(crop_image)
            bw_ratios.append(bw_ratio)

        choice_index = bw_ratios.index(min(bw_ratios)) if bw_ratios else None
        second_choice_index = (
            bw_ratios.index(
                min(bw_ratios[:choice_index] + bw_ratios[choice_index + 1 :])
            )
            if len(bw_ratios) > 1
            else None
        )

        delta_bw_ratio = (
            bw_ratios[choice_index] - bw_ratios[second_choice_index]
            if choice_index is not None and second_choice_index is not None
            else None
        )

        if delta_bw_ratio is not None and abs(delta_bw_ratio) >= threshold:
            choice_index = choice_index if delta_bw_ratio < 0 else second_choice_index
        else:
            choice_index = None

        result = {
            "index": index,
            "value": choices[choice_index]["value"]
            if choice_index is not None
            else choice_index,
            # 'deltaBWRatio': delta_bw_ratio
        }
        results.append(result)

    return results
