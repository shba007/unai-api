from fastapi import HTTPException
import numpy as np

import cv2


def detect_markers(image, findNecessary=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1,  9, -1],
    #                    [-1, -1, -1]])
    # image = cv2.filter2D(image, -1, kernel)

    # cv2.imshow("detect_markers", cv2.resize(image, (0, 0), fx=0.55, fy=0.55))
    # cv2.waitKey(0)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected = cv2.aruco.detectMarkers(
        image=image, dictionary=aruco_dict, parameters=parameters
    )

    if ids is None:
        raise HTTPException(status_code=404, detail="Unable to Detect any marker")

    # print("ids", ids.flatten())
    sufficient = sum(num in ids for num in [1, 2, 9, 11]) >= 4
    # print("total found", list(num in ids for num in [1, 2, 9, 11]))

    if not sufficient and findNecessary:
        raise HTTPException(status_code=404, detail="Unable to Detect Corner markers")

    markers = [
        {
            "id": id[0].tolist(),
            "positions": [
                float(np.mean(corner[0, :, 0])),
                float(np.mean(corner[0, :, 1])),
            ],
        }
        for id, corner in zip(ids, corners)
    ]
    markers.sort(key=lambda x: x["id"])

    return markers
