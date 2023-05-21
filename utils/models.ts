export interface Box {
	x: number;
	y: number;
	width: number;
	height: number;
	conf: number;
}

export interface Detection {
	id: string;
	boxes: Box[]
}

export interface Product {
	id: string;
	image: string;
	banner: 'popular' | 'newest' | 'treading' | null;
	name: string;
	category: string;
	price: {
		original: number;
		discounted: number;
	};
	ratings: [number, number, number, number, number];
	stock: boolean;
}
