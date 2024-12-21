import cv2
import numpy as np

from server.utils.omrDetectMarkers import detect_markers

DEST_MARKERS = [
    {"id": 1, "positions": [69.5, 69.5]},
    {"id": 2, "positions": [2309.5, 69.5]},
    {"id": 3, "positions": [69.5, 389.5]},
    {"id": 4, "positions": [1189.5, 389.5]},
    {"id": 5, "positions": [2309.5, 389.5]},
    {"id": 6, "positions": [69.5, 1839.5]},
    {"id": 7, "positions": [1189.5, 1839.5]},
    {"id": 8, "positions": [2309.5, 1839.5]},
    {"id": 9, "positions": [69.5, 3289.5]},
    {"id": 10, "positions": [1189.5, 3289.5]},
    {"id": 11, "positions": [2309.5, 3289.5]},
]

DIM = [2380, 3368]


def align_crop(image, src_markers):
    width, height = DIM

    corners = []
    for target_key in [1, 2, 11, 9]:
        src_marker = next(
            (
                src_marker
                for src_marker in src_markers
                if src_marker["id"] == target_key
            ),
            None,
        )
        if src_marker is not None:
            corners.append(src_marker["positions"])

    src_points = np.array(corners, dtype=np.float32)
    dest_points = np.array(
        [[70, 70], [width - 74, 70], [width - 74, height - 74], [70, height - 74]],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points)
    cropped_image = cv2.warpPerspective(image, transform_matrix, (width, height))
    # cv2.imwrite("./assets/images/cropped_image.jpg", cropped_image)

    src_markers = [
        element
        for element in detect_markers(cropped_image)
        if element["id"] in [3, 5, 7, 9, 11]
    ]
    dest_markers = DEST_MARKERS

    src_points = np.array(
        [
            src_marker["positions"]
            for src_marker in src_markers
            if (
                dest_marker := next(
                    (d for d in dest_markers if d["id"] == src_marker["id"]), None
                )
            )
            is not None
        ]
    )
    dest_points = np.array(
        [
            dest_marker["positions"]
            for src_marker in src_markers
            if (
                dest_marker := next(
                    (d for d in dest_markers if d["id"] == src_marker["id"]), None
                )
            )
            is not None
        ]
    )

    # print("src_points", src_points, "dest_points", dest_points)

    homography, _ = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(cropped_image, homography, (width, height))
    # cv2.imwrite("./assets/images/aligned_image.jpg", warped_image)

    return warped_image
