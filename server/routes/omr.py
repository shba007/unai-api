from fastapi import APIRouter, HTTPException
from nanoid import generate
from pydantic import BaseModel

from server.utils.base64ToArray import base64_to_array
from server.utils.imageSave import image_save
from server.utils.omrAlignCrop import align_crop
from server.utils.omrAlignInput import align_inputs
from server.utils.omrDetectMarkers import detect_markers
from server.utils.omrDetectQR import detect_qr
from server.utils.omrExtractData import extract_data
from server.utils.omrHighlights import get_highlights

router = APIRouter(
    prefix="/omr",
    tags=["omr"],
    responses={404: {"description": "Not found"}},
)


class RequestBody(BaseModel):
    image: str


@router.post("/")
async def omr(request: RequestBody):
    try:
        id = generate()
        image_save(id, request.image)

        image_array = base64_to_array(request.image)

        markers = detect_markers(image_array)
        # print("markers", markers)
        cropped_image = align_crop(image_array, markers)
        # print("cropped_image", cropped_image)
        meta_data = detect_qr(cropped_image)
        # print("meta_data", meta_data)

        option_count = meta_data["option"]
        choice_start = meta_data["choice"]["start"]
        choice_count = meta_data["choice"]["count"]

        inputs = align_inputs(cropped_image, option_count, choice_start, choice_count)
        choices = extract_data(cropped_image, inputs)
        highlights = get_highlights(cropped_image, option_count, inputs, choices)

        return {
            "data": {"name": meta_data["scale"], "choices": choices},
            "highlights": highlights,
        }
    except HTTPException as error:
        print("API omr POST", error)
        raise error
    except Exception as error:
        print("API omr POST", error)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
