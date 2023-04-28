import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from PIL import Image
import weaviate as Weaviate
import meilisearch as Meilisearch

from utils.helpers import Data
from utils.models import OneShotClassifier


router = APIRouter()

classifier = OneShotClassifier()

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_URL = "127.0.0.1" if WEAVIATE_URL == None else WEAVIATE_URL

MEILISEARCH_URL = os.getenv('MEILISEARCH_URL')
MEILISEARCH_URL = "127.0.0.1" if MEILISEARCH_URL == None else MEILISEARCH_URL
MEILISEARCH_API_KEY = os.getenv('MEILISEARCH_SECRET')
MEILISEARCH_API_KEY = "" if MEILISEARCH_API_KEY == None else MEILISEARCH_API_KEY

weaviate = Weaviate.Client(url=WEAVIATE_URL)
meilisearch = Meilisearch.Client(MEILISEARCH_URL, MEILISEARCH_API_KEY)


class Box(BaseModel):
    x: float
    y: float
    width:  float
    height: float
    conf: float


class ImageRequest(BaseModel):
    id: str
    boxes: List[Box]


def format(data):
    return {
        "id": data["id"],
        "image": data["image"],
        "banner": data["banner"],
        "name": data["name"],
        "category": data["category"],
        "price": {
            "original": data["priceOriginal"],
            "discounted": data["priceDiscounted"]
        },
        "rank": data["rank"],
        "ratings": data["ratings"]
    }


@router.post("/search")
async def search(request: ImageRequest):
    try:
        id = request.id
        file_path = f"assets/images/{id}.jpg"
        data = Data("temp-1")

        img = Image.open(file_path)
        img = data.__orient__(img)
        data.images = {}
        data.images[id] = img

        anno = [[box.x/100, box.y/100, box.width/100, box.height/100] for box in request.boxes]
        data.annotations = {}
        data.annotations[id] = anno

        class_img = data.get_images(type="crop-all")
        class_img = [x["image"] for x in class_img]

        embeddings = classifier.predict(class_img)
        results = []

        for embedding in embeddings:
            embedding = {
                "distance": 1.0,
                "vector": embedding
            }
            result = weaviate.query.get('Earrings', ['lakeId', 'sku']).with_near_vector(embedding).do()
            results.append(result)

        # print("results", results)
        filtered_metadatas = []
        for item in results:  # type: ignore
            item = item['data']['Get']['Earrings']
            filtered_metadatas.append([x for x in item if x['sku'] != ''])

        # print("filtered_metadatas", filtered_metadatas)
        products_results = []
        # Remove duplicate
        for item in filtered_metadatas:
            # print("\nitem", item, "\n")
            seen = set()
            unique_ids = [x["sku"] for x in item if not (x['sku'] in seen or seen.add(x['sku']))]
            # print("\nunique_ids", unique_ids, "\n")
            queries = [{'indexUid': 'products', 'q': f'"{unique_id}"'} for unique_id in unique_ids]
            # print("\nqueries", queries, "\n")
            products = meilisearch.multi_search(queries)
            # print("\nhits", products, "\n")
            products = [format(product["hits"][0]) for product in products["results"] if len(product["hits"])]
            # print("\nproducts", products, "\n")
            products_results.append({"products": products})

        return products_results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not Found")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Some Unknown Error Found")
