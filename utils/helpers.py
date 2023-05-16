import os
import copy
import json
import asyncio
from io import BytesIO

from firebase_admin import initialize_app, credentials, storage

import numpy as np
from PIL import Image, ImageOps, ExifTags

PRESET = os.getenv("PRESET")
PRESET = "deploy" if PRESET == None else PRESET

FIREBASE_CONFIG = os.getenv("FIREBASE_CONFIG")
FIREBASE_CONFIG = "" if FIREBASE_CONFIG == None else FIREBASE_CONFIG

STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
STORAGE_BUCKET = "" if STORAGE_BUCKET == None else STORAGE_BUCKET

initialize_app(credentials.Certificate(json.loads(FIREBASE_CONFIG)), {'storageBucket': STORAGE_BUCKET})

bucket = storage.bucket()


def file_exists_check(file_path):
    return os.path.isfile(file_path)


def save_file(file_location: str, file_bytes: BytesIO):
    with open(file_location, 'wb') as f:
        content = file_bytes.getvalue()
        f.write(content)


async def delete_file(file_location: str):
    await asyncio.sleep(300)
    os.remove(file_location)
    return


def upload_file(source_file_location: str, dest_file_location: str | None = None):
    if PRESET != "deploy":
        return

    if dest_file_location == None:
        dest_file_location = source_file_location

    blob = bucket.blob(dest_file_location)
    blob.upload_from_filename(source_file_location)

    asyncio.run(delete_file(source_file_location))
    return


def download_file(source_file_location: str, dest_file_location: str | None = None):
    if file_exists_check(dest_file_location):
        return

    if dest_file_location == None:
        dest_file_location = source_file_location

    blob = bucket.blob(source_file_location)
    blob.download_to_filename(dest_file_location)
    return


def pad_name(num: int, size: int, pad_with="0"):
    return format(num, pad_with + str(size))

# TODO: Test
# def pad_name(num: int, size: int, pad_with="0"):
#     format_type = '{'+f":{pad_with}>{size}"+'}'
#     return f"{format_type}".format(num)


"""
formats:
XYXY -> X_min, Y_min, X_max, Y_max
XYWH -> X_min, Y_min, Width, Height
CCWH -> X_center, Y_center, Width, Height

normalized -> X /=  Image Width and Y /=  Image Height
            Width /= Image Width and Y /= Image Height

"""


def convert_box(init_box, image_dim, init_format="XYXY", init_normalized=False, final_format="CCWH", final_normalized=False, isDebug=False):
    final_box = np.zeros_like(init_box)
    img_width, img_height = image_dim

    if init_normalized:
        init_box[0] *= img_width
        init_box[1] *= img_height
        init_box[2] *= img_width
        init_box[3] *= img_height

    # Convert form "XYXY" or "XYWH" to "CCWH"
    if init_format == "XYXY":
        x_center = (init_box[0] + init_box[2])/2
        y_center = (init_box[1] + init_box[3])/2
        width = init_box[2] - init_box[0]
        height = init_box[3] - init_box[1]
    elif init_format == "XYWH":
        x_center = init_box[0] + init_box[2]/2
        y_center = init_box[1] + init_box[3]/2
        width = init_box[2]
        height = init_box[3]
    else:
        x_center, y_center, width, height = init_box

    # if isDebug:
    #     print("\n",x_center, y_center, width, height)
    # Convert form "CCWH" to "XYXY" or "XYWH"
    if final_format == "XYXY":
        final_box[0] = x_center - width/2
        final_box[1] = y_center - height/2
        final_box[2] = x_center + width/2
        final_box[3] = y_center + height/2
    elif final_format == "XYWH":
        final_box[0] = x_center - width/2
        final_box[1] = y_center - height/2
        final_box[2] = width
        final_box[3] = height
    else:
        final_box = [x_center, y_center, width, height]
    
    # if isDebug:
    #     print({ "final_box": np.array(final_box).tolist() })

    if final_normalized:
        final_box[0] /= img_width
        final_box[1] /= img_height
        final_box[2] /= img_width
        final_box[3] /= img_height
    
    # if isDebug:
    #     print({ "finalBoxNormalize": np.array(final_box).tolist() })

    return final_box


def resize(input: str | list, dimension: tuple, annotations: list[list[float]] | None = None, background_color=(255, 255, 255)):
    image = copy.deepcopy(input)

    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, list) or isinstance(image, np.ndarray):
        image = np.array(image).astype(np.uint8)
        image = Image.fromarray(image)
    else:
        try:
            image.size
        except:
            raise ValueError("Enter path or image array")

    width, height = image.size

    if (width != height):
        max_dim = max(width, height)
        result = Image.new(image.mode, (max_dim, max_dim), background_color)
        result.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
        image = result

    image = image.resize(dimension, Image.LANCZOS)

    def pipeline(annotation, dimension, max_dim):
        x_center, y_center, obj_width, obj_height = convert_box(annotation, dimension, init_format="CCWH", init_normalized=True)

        # padding added
        x_center += (max_dim-width)/2
        y_center += (max_dim-height)/2

        annotation = [x_center, y_center, obj_width, obj_height]
        annotation = convert_box(annotation, (max_dim, max_dim), init_format="CCWH", final_normalized=True)

        return annotation

    if annotations:
        max_dim = max(width, height)

        annotations = map(lambda x: pipeline(x, (width, height), max_dim), annotations)
        annotations = np.array(list(annotations))
        return image, annotations
    else:
        return image


# Crop image
# Annotations format XYXY
def crop(images, annotations: list[list[float]], init_format="XYXY", init_normalized=False):
    # bbox format x, y, width, height
    img = copy.deepcopy(images)
    bboxes = copy.deepcopy(annotations)

    img = np.array(img)
    img = Image.fromarray(img)
    crops = []

    for bbox in bboxes:
        bbox = convert_box(bbox, img.size, init_format=init_format, init_normalized=init_normalized, final_format="XYXY")
        crops.append(img.crop(bbox))
    return crops


class Data:
    def __init__(self, id, meta=None, annotations=None, images=None):
        self.id = id
        self.meta = meta
        self.annotations = annotations
        self.images = images

    def __orient__(self, image):
        # Get the EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        try:
            exif = dict(image._getexif().items())
            orientation = exif[orientation]
        except:
            orientation = 1

        # Apply the necessary transformation
        image = ImageOps.exif_transpose(image)

        # Update the image dimensions if necessary
        width, height = image.size
        if orientation in [5, 6, 7, 8]:
            width, height = height, width

        # print("finally", image.size)
        # print("img", np.array(image).shape)

        # Save the transformed image with updated dimensions
        return image

    def __pipeline__(self, info, type, resize_dim, return_annotations):
        if type == "crop-all" or type == "crop-one":
            annotations = copy.deepcopy(self.annotations[info["id"]])
            annotations = annotations[:1] if type == "crop-one" else annotations
            crops = crop(self.images[info["id"]], annotations, init_format="CCXY", init_normalized=True)

            for single_crop, annotation in zip(crops, annotations):
                single_crop, annotations = resize(single_crop, resize_dim, annotations=[annotation]) if resize_dim else (single_crop, annotation)

                if return_annotations == True:
                    yield {"id": info["id"],
                           "photography": info["photography"],
                           "shot": info["shot"],
                           "image": single_crop,
                           "bboxes": annotations}
                else:
                    yield {"id": info["id"],
                           "photography": info["photography"],
                           "shot": info["shot"],
                           "image": single_crop}

        else:
            annotations = copy.deepcopy(self.annotations[info["id"]])
            image, annotations = resize(self.images[info["id"]], resize_dim, annotations=annotations) if resize_dim else (self.images[info["id"]], annotations)

            if return_annotations == True:
                yield {"id": info["id"],
                       "photography": info["photography"],
                       "shot": info["shot"],
                       "image": image,
                       "bboxes": annotations}
            else:
                yield {"id": info["id"],
                       "photography": info["photography"],
                       "shot": info["shot"],
                       "image": image}

    def __img_pipeline__(self, input, type, resize_dim, return_annotations):
        id, image = input
        try:
            if type == "crop-all" or type == "crop-one":
                annotations = copy.deepcopy(self.annotations[id])
                annotations = annotations[:1] if type == "crop-one" else annotations
                crops = crop(self.images[id], annotations, init_format="CCXY", init_normalized=True)

                for single_crop, annotation in zip(crops, annotations):
                    single_crop, annotations = resize(single_crop, resize_dim, annotations=[annotation]) if resize_dim else (single_crop, annotation)

                    if return_annotations == True:
                        yield {"id": id,
                               "image": single_crop,
                               "bboxes": annotations}
                    else:
                        yield {"id": id,
                               "image": single_crop}
            else:
                if self.annotations == None:
                    annotations = None
                    image = resize(self.images[id], resize_dim) if resize_dim else self.images[id]
                else:
                    annotations = copy.deepcopy(self.annotations[id])
                    image, annotations = resize(self.images[id], resize_dim, annotations=annotations) if resize_dim else (self.images[id], annotations)

                if return_annotations == True:
                    yield {"id": id,
                           "image": image,
                           "bboxes": annotations}
                else:
                    yield {"id": id,
                           "image": image}
        except:
            print(f"Error in {id}")

    def get_image(self, select, type="full", resize_dim=None, return_annotations=False):
        id = self.meta["profile-image"] if select == "profile" else select
        info = list(filter(lambda info: info["id"] == id, self.meta["images"]))[0]
        result = self.__pipeline__(info, type, resize_dim, return_annotations)

        return [item for item in result]

    def get_images(self, select=None, type="full", resize_dim=None, return_annotations=False):
        if self.meta == None:
            lst = map(lambda x: self.__img_pipeline__(x, type, resize_dim, return_annotations), self.images.items())
        else:
            if select == None:
                # print("Id", self.id)
                filtered_images = self.meta["images"]
            elif select == "face":
                filtered_images = [image for image in self.meta["images"] if image["shot"] == "model" or image["shot"] == "ear" or image["shot"] == "mannequin"]

            lst = map(lambda x: self.__pipeline__(x, type, resize_dim, return_annotations), filtered_images)

        return [item for sublist in lst for item in sublist]

        # else:
        #     map(lambda x: x  self.images)
