import { unlinkSync } from "fs";
import { join } from "path";

import tf from '@tensorflow/tfjs';
import firebase from 'firebase-admin';
import { initializeApp } from 'firebase-admin/app';
import sharp from "sharp";

import { Detection } from '../utils/models';
import { randomUUID } from "crypto";

const { storage, credential } = firebase;

const FIREBASE_CONFIG = process.env["FIREBASE_CONFIG"] == null ? "" : process.env["FIREBASE_CONFIG"]
const STORAGE_BUCKET = process.env["STORAGE_BUCKET"] == null ? "" : process.env["STORAGE_BUCKET"]
const PRESET = process.env["PRESET"] == null ? "" : process.env["PRESET"]

// Initialize Firebase Admin SDK
const firebaseApp = initializeApp({
	credential: credential.cert(FIREBASE_CONFIG),
	storageBucket: STORAGE_BUCKET
})
const bucket = storage().bucket()

const path = { join }
const fs = { unlinkSync }

const DET_DIM: [number, number] = [640, 640]
const DELETE_TIMEOUT = 5 * 60 * 1000

async function saveImage(id: string, image: Buffer): Promise<boolean> {
	const storage = useStorage()
	const filePath = path.join(process.cwd(), `assets/images/${id}.jpg`)
	try {
		storage.setItem(id, "pending")
		await sharp(image).toFile(filePath)
		storage.setItem(id, true)
		console.log(`Image saved to: ${id}`);
		return true
	} catch (error) {
		storage.setItem(id, false)
		console.error(`Error save image: ${id}`, error);
		return false
	}
}

async function uploadImage(id: string, image: Buffer): Promise<boolean> {
	if (PRESET === "dev")
		return

	try {
		// Upload the image buffer to Firebase Storage
		const file = bucket.file(`images/${id}.jpg`);
		await file.save(image, {
			metadata: {
				contentType: 'image/jpeg', // Change the content type if necessary
			},
		});

		console.log(`Image uploaded to: ${id}`);
		return new Promise((resolve) => setTimeout(() => resolve(deleteImage(id)), DELETE_TIMEOUT))
	} catch (error) {
		console.error(`Error upload image: ${id}`, error);
		return false
	}
}

function deleteImage(id: string): boolean {
	const storage = useStorage()
	const filePath = path.join(process.cwd(), `assets/images/${id}.jpg`)

	try {
		fs.unlinkSync(filePath);
		storage.setItem(id, false)
		console.log(`Image deleted: ${id}`);
		return true;
	} catch (error) {
		console.error(`Error deleting image: ${id}`, error);
		return false;
	}
}

async function nms(boxes: [number, number, number, number, number, number][], maxOutputSize = 100, iouThreshold = 0.75, scoreThreshold = 0.5) {
	const boxesTensor = tf.tensor2d(boxes.map((box) => convertBox(box, DET_DIM, "CCWH", false, "XYXY", true)).map((box) => box.slice(0, 4)));
	const confsTensor = tf.tensor1d(boxes.map((box) => box[box.length - 2]));

	const selectedIndices = tf.image.nonMaxSuppression(
		boxesTensor,
		confsTensor,
		maxOutputSize,
		iouThreshold,
		scoreThreshold
	);
	const selectedBoxes = tf.gather(boxes, selectedIndices);
	return (await selectedBoxes.array()) as [number, number, number, number, number, number][];
}

/*
	format x_center, y_center, width, height in not normalized for 640, 640 image ->
	format x_center, y_center, width, height in normalized for self.dim image
 */
function scale(box: number[], dim: [number, number]) {
	let [xCenter, yCenter, width, height, ..._] = box;

	const factor = Math.max(...dim) / Math.max(...DET_DIM);
	const offset = Math.abs(dim[0] - dim[1]);

	xCenter *= factor;
	yCenter *= factor;
	width *= factor;
	height *= factor;

	if (dim[0] > dim[1])
		yCenter -= offset / 2;
	else if (dim[0] < dim[1])
		xCenter -= offset / 2;

	const scaledBox = [xCenter, yCenter, width, height];
	// console.log({ scaledBox });

	const convertedBox = convertBox([...scaledBox], [...dim], "CCWH", false, "CCWH", true);
	// console.log({ convertedBox });

	return convertedBox;
}


async function preprocess(image: Buffer): Promise<[number[], [number, number]]> {
	tf.engine().startScope()
	// Resize into 640x640
	const [resizedImageArray, dim] = await resize(image, DET_DIM)
	// console.log({ resizedImageArray })
	const reshapedImageArray = [[await reshape(resizedImageArray, DET_DIM)]] as unknown as number[]
	// console.log({ reshapedImageArray })
	tf.engine().startScope()

	return [reshapedImageArray, dim]
}

async function predict(image: Buffer): Promise<[number, number, number, number, number, number][]> {
	const config = useRuntimeConfig()

	const [preprocessedImage, dim] = await preprocess(image)

	try {
		const detections = await $fetch("/detector:predict", { baseURL: config.apiUrl, method: "POST", body: { "instances": preprocessedImage } })
		return postprocess(detections['predictions'][0], dim)
	} catch (error) {
		throw Error("Failed request Tensorflow Serving /detector:predict ", error)
	}
}

async function postprocess(predictions: [number, number, number, number, number, number][], dim: [number, number], confThreshold = 0.25): Promise<[number, number, number, number, number, number][]> {
	tf.engine().startScope()
	const [xs, ys, ws, hs, ...labels] = predictions

	const conf = tf.max(labels, 0)
	const labelIndex = tf.cast(tf.argMax(labels, 0), 'float32')
	const boxes = tf.stack([xs, ys, ws, hs, conf, labelIndex], 1)
	const mask = tf.greater(conf, confThreshold)

	// Filter by confidence Threshold
	const filteredBoxes = await (await tf.booleanMaskAsync(boxes, mask)).array() as [number, number, number, number, number, number][]
	// console.log({ filteredBoxes });

	// NMS
	const nmsBoxes = await nms(filteredBoxes)
	// console.log({ nmsBoxes });

	// Scale
	const scaledBoxes = nmsBoxes.map((nmsBox) => [...scale(nmsBox, dim), nmsBox[4], nmsBox[5]] as [number, number, number, number, number, number])
	// console.log({ scaledBoxes });
	tf.engine().startScope()

	return scaledBoxes
}

export default defineEventHandler<Detection>(async (event) => {
	try {

		const { image } = await readBody<{ image: string }>(event);
		// Generate a random UUID
		const id = randomUUID()

		// Convert from base64 encoding to array
		const imageArray = await base64ToArray(image)

		saveImage(id, imageArray)
		uploadImage(id, imageArray)

		// Save and Upload Async
		const detections = await predict(imageArray)
		// console.log({ detections });

		return {
			"id": id,
			"boxes": detections.map(([x, y, width, height, conf, label]) => {
				if (label != 0)
					return null

				x *= 100
				y *= 100
				width *= 100
				height *= 100
				conf *= 100

				return {
					"x": x,
					"y": y,
					"width": width,
					"height": height,
					"conf": conf
				}
			}).filter((box) => box !== null)
		}
	} catch (error: any) {
		console.error("API detect POST", error);

		throw createError({ statusCode: 500, statusMessage: 'Some Unknown Error Found' })
	}
})
