import { readFileSync } from "fs";
import { join } from "path";

import { MeiliSearch } from "meilisearch";
import { QdrantClient } from '@qdrant/js-client-rest';
import sharp from "sharp";
import { Box, Detection, Product } from "../utils/models";

const path = { join }

const CLASS_DIM: [number, number] = [256, 256]

const meilisearch = new MeiliSearch({
  host: process.env.MEILISEARCH_URL as string,
  apiKey: process.env.MEILISEARCH_SECRET as string,
});

const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL as string,
  apiKey: process.env.QDRANT_SECRET as string
});

async function openImage(id: string): Promise<Buffer> {
  const filePath = path.join(process.cwd(), `assets/images/${id}.jpg`)

  try {
    const imageBuffer = readFileSync(filePath);
    return sharp(imageBuffer).toBuffer();
  } catch (error) {
    throw createError({ statusCode: 404, statusMessage: 'File not found' })
  }
}

async function cropImage(image: Buffer, boxes: Box[]): Promise<Buffer[]> {
  const imageSharp = sharp(image);
  const { width: imgWidth, height: imgHeight } = await imageSharp.metadata()
  // Convert boxes of CCWH to XYWH int
  return Promise.all(boxes.map(async ({ x, y, width, height }) => {
    const box = convertBox([x / 100, y / 100, width / 100, height / 100], [imgWidth, imgHeight], "CCWH", true, "XYWH", false).map((num) => Math.floor(num));
    const temp = await imageSharp.resize({ width: imgWidth, height: imgHeight })
      .extract({ left: box[0], top: box[1], width: box[2], height: box[3] })
      .toBuffer();

    // saveUint8ArrayAsImage(temp, [imgWidth, imgHeight], `assets/test-${Math.floor(x)}.jpg`)
    return temp
  }))
}

async function preprocess(images: Buffer[]): Promise<number[][]> {
  const reshapedImagesArray = await Promise.all(images.map(async (image) => {
    // Resize into 640x640
    const [resizedImageArray, dim] = await resize(image, CLASS_DIM)
    // console.log({ resizedImageArray })
    return reshape(resizedImageArray, CLASS_DIM)
  }))
  // console.log({ reshapedImageArray })

  return reshapedImagesArray
}

async function predict(images: Buffer[]): Promise<string[][]> {
  const config = useRuntimeConfig()

  const preprocessedImages = await preprocess(images)
  try {
    const embeddings = await $fetch("/feature_extractor:predict", { baseURL: config.apiUrl, method: "POST", body: { "instances": preprocessedImages } })

    return postprocess(embeddings['predictions'])
  } catch (error) {
    throw Error("Failed request Tensorflow Serving /feature_extractor:predict", error)
  }
}

async function postprocess(embeddings: number[][]): Promise<string[][]> {
  const filteredProductsSKUs: string[][] = []

  await Promise.all(embeddings.map(async (embedding) => {
    try {
      const result = await qdrant.search("Earrings", {
        vector: embedding,
        limit: 20,
      });

      const products: { lakeId: string, sku: string }[] = result.map(({ payload }) => ({ lakeId: payload.lakeId as string, sku: payload.sku as string }))
      const filteredProductSKUs = new Set<string>()
      products.filter((product) => product.sku !== '')
        .forEach((product) => filteredProductSKUs.add(product.sku))

      filteredProductsSKUs.push([...filteredProductSKUs])
    } catch (error) {
      throw Error("Failed request Qdrant", error)
    }
  }))
  // console.log(filteredProductsSKUs);

  return filteredProductsSKUs
}

export interface SearchProduct extends Omit<Product, 'ratings'> {
  totalRating: number
  averageRating: number,
}

export default defineEventHandler<any>(async (event) => {
  try {
    const { id, boxes } = await readBody<Detection>(event);
    // Open image with that id
    const image = await openImage(id)
    // Make image Crops using the boxes
    const images = await cropImage(image, boxes)
    // console.log({ images });
    const labels = await predict(images)
    // console.log({ labels })

    return Promise.all(labels.map(async (labels) => {
      try {
        const products = await meilisearch.multiSearch({ queries: labels.map((id) => ({ 'indexUid': 'products', 'q': `'${id}'` })) })

        return {
          products: products.results.map((data) => data.hits[0])
            .filter(a => {
              return a != undefined
            })
            .map(({ id, image, banner, name, categories, priceOriginal, priceDiscounted, rank, totalRating, averageRating, stock }): Product =>
            ({
              id,
              image,
              banner,
              name,
              categories,
              price: { original: priceOriginal, discounted: priceDiscounted },
              rank,
              totalRating,
              averageRating,
              stock
            }))
        }
      } catch (error) {
        throw Error("Failed request MeiliSearch", error)
      }
    }))
  } catch (error: any) {
    if (error?.statusCode === 404)
      throw error

    console.error("API search POST", error);

    throw createError({ statusCode: 500, statusMessage: 'Some Unknown Error Found' })
  }
})
