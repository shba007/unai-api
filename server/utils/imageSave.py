import base64


def image_save(
    id: str, encoded_image: str, output_path: str = "./assets/images"
) -> None:
    try:
        if encoded_image.startswith("data:image"):
            encoded_image = encoded_image.split(",")[1]

        image_data = base64.b64decode(encoded_image)

        with open(f"{output_path}/{id}.jpg", "wb") as image_file:
            image_file.write(image_data)

        print(f"Image saved successfully to {output_path}/{id}.jpg")
    except Exception as e:
        raise ValueError(f"Failed to save image: {e}")
