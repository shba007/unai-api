import json
import cv2
from fastapi import HTTPException


def detect_qr(image):
    x = image.shape[1] - 105 - 380
    y = 55
    image = image[y : y + 380, x : x + 380]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    # Upscale the image
    # image = sr.upsample(image)
    # image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    # kernel = np.array([[-1, -1, -1],
    #                    [-1,  9, -1],
    #                    [-1, -1, -1]])
    # image = cv2.filter2D(image, -1, kernel)

    # cv2.imshow("detect_qr", image)
    # cv2.waitKey(0)

    detector = cv2.QRCodeDetector()
    retval, info, points, _ = detector.detectAndDecodeMulti(image)

    if retval is False:
        raise HTTPException(status_code=404, detail="Unable to detected QR Code")

    # print(info)

    try:
        data = json.loads(info[0])
        return {
            "scale": data["scale"],
            "option": data["option"],
            "choice": {
                "start": data["start"],
                "count": data["count"],
                "total": data["total"],
            },
        }
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid QR Code detected")
