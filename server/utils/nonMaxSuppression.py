import numpy as np


def nms(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes.

    Args:
        boxes: Array of bounding boxes with each row as (x, y, w, h, conf, label).
        iou_threshold: Intersection over Union (IoU) threshold for NMS.

    Returns:
        Array of bounding boxes after NMS.
    """
    if len(boxes) == 0:
        return np.array([])

    # Initialize the list of picked indexes
    pick = []

    # Extract coordinates and confidence scores
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    confs = boxes[:, 4]

    # Compute the area of the bounding boxes and sort by confidence
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(confs)[::-1]

    # Iterate while some indexes remain
    while len(idxs) > 0:
        # Grab the index of the current highest confidence
        i = idxs[0]
        pick.append(i)

        # Compute IoU between the picked box and the rest
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Compute width and height of the overlap
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the ratio of overlap (IoU)
        overlap = (w * h) / area[idxs[1:]]

        # Delete indexes with IoU above the threshold
        idxs = np.delete(
            idxs, np.concatenate(([0], np.where(overlap > iou_threshold)[0] + 1))
        )

    # Return the boxes that were picked
    return boxes[pick]
