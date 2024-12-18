import os
import json
import requests

import tensorflow as tf
from ..utils.helpers_obj import convert_box, resize

DET_DIM = (640, 640)
CLASS_DIM = (256, 256)

TF_SERVING_URL = os.getenv("TF_SERVING_URL")
TF_SERVING_URL = "" if TF_SERVING_URL is None else TF_SERVING_URL


def fetch(path: str, method: str, body=None) -> dict:
    url = f"{TF_SERVING_URL}{path}"
    headers = {"Content-Type": "application/json"}

    if body is not None:
        data = json.dumps(body)

        if method == "GET":
            response = requests.get(url, headers=headers, data=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, data=data)
        elif method == "PUT":
            response = requests.put(url, headers=headers, data=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, data=data)
        else:
            response = requests.get(url, headers=headers, data=data)
    else:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers)
        elif method == "PUT":
            response = requests.put(url, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            response = requests.get(url, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        raise ValueError({"status": response.status_code, "message": response.json()})


class Detector:
    def __init__(self, dim=None, output_format="XYXY") -> None:
        self.dim = dim
        self.output_format = output_format

        return

    def __nms__(
        self, bboxes, score_threshold=0.5, iou_threshold=0.5, max_output_size=100
    ):
        def convert(bbox):
            xmin, ymin, xmax, ymax = convert_box(
                bbox,
                DET_DIM,
                init_format="CCWH",
                init_normalized=False,
                final_format="XYXY",
                final_normalized=True,
            )
            return [ymin, xmin, ymax, xmax]

        bboxes = tf.convert_to_tensor(bboxes).numpy()
        if len(bboxes) == 0:
            return bboxes

        boxes = bboxes[:, :4]
        confs = bboxes[:, -2]

        converted_boxes = list(map(convert, boxes))

        selected_indices = tf.image.non_max_suppression(
            converted_boxes,
            confs,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        bboxes = tf.gather(bboxes, selected_indices)

        return bboxes

    def __scale__(self, box, dim):
        # print("__scale__ bbox, dim", box.numpy().tolist(), dim.numpy().tolist())
        """
        format x_center, y_center, width, height in not normalized for 640, 640 image ->
        format x_center, y_center, width, height in normalized for self.dim image
        """
        x_center, y_center, width, height = box[:4]

        factor = tf.cast(max(dim) / 640, dtype=tf.float32)
        offset = tf.cast(abs(dim[0] - dim[1]), dtype=tf.float32)

        x_center *= factor
        y_center *= factor
        width *= factor
        height *= factor

        # print("before", x_center, y_center, width, height)
        if dim[0] > dim[1]:
            y_center -= offset / 2
        elif dim[0] < dim[1]:
            x_center -= offset / 2

        # print("after", x_center, y_center, width, height)
        scaled_box = tf.stack([x_center, y_center, width, height])
        # print("scaled_box",scaled_box.numpy().tolist())
        dim = tf.cast(dim, tf.float32)

        # print("scaled_box", scaled_box.numpy().tolist())
        converted_box = convert_box(
            scaled_box,
            dim,
            init_format="CCWH",
            final_format=self.output_format,
            final_normalized=True,
            isDebug=True,
        )
        converted_box = tf.convert_to_tensor(converted_box)
        # print("converted_box", converted_box.numpy().tolist())
        return converted_box

    def __postprocess__(self, prediction, dim, conf_threshold=0.25):
        # print("__postprocess__, dim", prediction.shape, dim)
        xs, ys, ws, hs = prediction[:4]
        labels = prediction[4:]

        conf = tf.reduce_max(labels, axis=0)
        label_index = tf.argmax(labels, axis=0)
        # print("label_index", label_index, "conf", conf)

        label_index = tf.cast(label_index, tf.float32)
        boxes = tf.stack([xs, ys, ws, hs, conf, label_index], axis=1)
        mask = tf.greater(conf, conf_threshold)

        filtered_boxes = tf.boolean_mask(boxes, mask)
        # print("Boxes before NMS", filtered_boxes.numpy().tolist())
        # NMS
        nms_boxes = self.__nms__(filtered_boxes)
        # print("Boxes After NMS", nms_boxes.numpy().tolist())
        # Scale
        scaled_boxes = tf.map_fn(lambda x: self.__scale__(x, dim), nms_boxes)
        if len(scaled_boxes) > 0:
            scaled_boxes = tf.concat([scaled_boxes, nms_boxes[:, -2:]], axis=-1)

        # print("Boxes After Scaling", scaled_boxes.numpy().tolist())
        return scaled_boxes

    def predict(self, inputs):
        images = tf.map_fn((lambda x: resize(x.numpy(), DET_DIM)), inputs)
        # images = tf.cast(images, tf.float32)/255.0
        images = tf.expand_dims(images, axis=0)

        if self.dim is None:
            dim = tf.map_fn((lambda x: x.shape[:2][::-1]), inputs, dtype=tf.int32)
        else:
            dim = tf.map_fn(
                (lambda x: tf.convert_to_tensor(self.dim)), inputs, dtype=tf.int32
            )

        predictions = fetch(
            "/detector:predict", "POST", {"instances": images.numpy().tolist()}
        )["predictions"]
        predictions = tf.convert_to_tensor(predictions)

        detections = tf.map_fn(
            lambda x: self.__postprocess__(x[0], x[1]),
            (predictions, dim),
            fn_output_signature=(tf.float32),
        )
        # detections = tf.round(detections)
        detections = detections.numpy().tolist()

        return detections


class OneShotClassifier:
    def __init__(self) -> None:
        return

    def predict(self, inputs) -> list:
        inputs = list(
            map(
                lambda class_img: tf.keras.utils.img_to_array(
                    resize(class_img, CLASS_DIM)
                ),
                inputs,
            )
        )
        inputs = tf.convert_to_tensor(inputs)
        features = fetch(
            "/feature_extractor:predict", "POST", {"instances": inputs.numpy().tolist()}
        )["predictions"]
        return features
