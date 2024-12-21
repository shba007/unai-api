import cv2
from fastapi import HTTPException
import numpy as np

from scipy.optimize import linear_sum_assignment


def is_circle_inside(circle_center):
    # from markers 3,5,11,9
    boundary = [
        [70.0, 390.5],
        [2306.0, 390.5],
        [2306.0, 3294.0],
        [70.0, 3294.0],
    ]

    x, y = circle_center
    x_min, y_min = boundary[0]
    x_max, y_max = boundary[2]

    if x_min <= x <= x_max and y_min <= y <= y_max:
        return True
    else:
        return False


def choice_generator(option, index, total):
    factor = 4
    index = index - 1
    unit = 15
    x = 55
    y = 100

    while index < total:
        if index % 40 == 0 and index != 0:
            x += 110
            y = 100
        elif index % 5 == 0 and index != 0:
            y += 15

        y += unit

        choices = None
        if option == 2:
            choices = [
                {"value": 1, "chord": [(x) * factor, (y) * factor]},
                {"value": 0, "chord": [(x + unit) * factor, (y) * factor]},
            ]
        elif option == 5:
            choices = [
                {"value": 0, "chord": [(x) * factor, (y) * factor]},
                {"value": 1, "chord": [(x + 1 * unit) * factor, (y) * factor]},
                {"value": 2, "chord": [(x + 2 * unit) * factor, (y) * factor]},
                {"value": 3, "chord": [(x + 3 * unit) * factor, (y) * factor]},
                {"value": 4, "chord": [(x + 4 * unit) * factor, (y) * factor]},
            ]

        yield {"index": index + 1, "choices": choices}

        index += 1


def align_inputs(image, options_count, choice_start, choice_count):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # cv2.imshow("align_inputs", cv2.resize(image, (0,0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50,
    )

    dest_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)

        for x, y, r in circles:
            if is_circle_inside((x, y)):
                dest_circles.append((x, y))

    choices = list(choice_generator(options_count, choice_start, choice_count))

    src_circles = np.array(
        [choice["chord"] for data in choices for choice in data["choices"]]
    )
    dest_circles = np.array(dest_circles)

    try:
        distances = np.linalg.norm(src_circles[:, np.newaxis] - dest_circles, axis=-1)
    except Exception:
        raise HTTPException(status_code=500, detail="Unable to calculate")

    row_indices, col_indices = linear_sum_assignment(distances)

    matched_pairs = [(i, j) for i, j in zip(row_indices, col_indices)]

    for i in range(len(choices)):
        for j in range(len(choices[i]["choices"])):
            choices[i]["choices"][j]["chord"] = None

    for pair in matched_pairs:
        index = pair[0]
        choices[index // options_count]["choices"][index % options_count]["chord"] = (
            dest_circles[pair[1]].tolist()
        )

    return choices
