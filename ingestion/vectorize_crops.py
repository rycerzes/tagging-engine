import os
import json
import glob
import numpy as np
import argparse
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from urllib.parse import urlparse
from pathlib import Path
import hashlib
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()


class CropVectorizer:
    def __init__(
        self,
        processed_data_path: str = "/root/flickd-ai/tagging-engine/data/processed/product_images",
        qdrant_url: str = None,
        qdrant_collection_name: str = None,
        similarity_threshold: float = 0.95,
        max_workers: int = 8,
    ):
        self.processed_data_path = processed_data_path
        self.similarity_threshold = similarity_threshold
        self.max_workers = max_workers

        # Load configuration from environment variables
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.collection_name = qdrant_collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME"
        )

        if not self.qdrant_url:
            raise ValueError(
                "QDRANT_URL must be set in environment variables or passed as parameter"
            )

        if not self.collection_name:
            raise ValueError(
                "QDRANT_COLLECTION_NAME must be set in environment variables or passed as parameter"
            )

        # Parse Qdrant URL to extract host and port
        parsed_url = urlparse(self.qdrant_url)
        qdrant_host = parsed_url.hostname
        qdrant_port = parsed_url.port or 6333

        print(
            f"Connecting to Qdrant at {self.qdrant_url} (collection: {self.collection_name})"
        )

        # Check for CUDA availability and configure device
        self.device = self._setup_device()
        print(f"Using device: {self.device}")

        # Initialize CLIP model and processor
        model_name = "patrickjohncyh/fashion-clip"
        print(f"Loading {model_name} model...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print(f"CLIP model moved to {self.device}")

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._setup_collection()

        # Initialize FAISS index for fast similarity search
        self.faiss_index = faiss.IndexFlatIP(512)  # Inner product for cosine similarity
        self.faiss_lock = threading.Lock()
        self.unique_count = 0
        self.duplicate_count = 0

    def _setup_device(self):
        """Setup and return the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"CUDA available! Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"Available GPUs: {gpu_count}")

            # Clear cache to start fresh
            torch.cuda.empty_cache()
            return device
        else:
            print("CUDA not available. Using CPU (this will be slower)")
            return torch.device("cpu")

    def _setup_collection(self):
        """Initialize Qdrant collection"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")

    def _setup_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        # Use GPU FAISS if available and CUDA is being used
        if self.device.type == "cuda" and faiss.get_num_gpus() > 0:
            print("Using GPU FAISS index")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        else:
            print("Using CPU FAISS index")

    def _load_image_batch(
        self, image_paths: List[str]
    ) -> Tuple[List[Image.Image], List[str]]:
        """Load a batch of images in parallel"""

        def load_single_image(img_path: str) -> Tuple[Image.Image, str]:
            try:
                image = Image.open(img_path).convert("RGB")
                # Basic validation
                if image.size[0] >= 16 and image.size[1] >= 16:
                    return image, img_path
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
            return None, None

        images = []
        valid_paths = []

        # Use ThreadPoolExecutor for parallel image loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(load_single_image, path): path for path in image_paths
            }

            for future in as_completed(future_to_path):
                image, path = future.result()
                if image is not None:
                    images.append(image)
                    valid_paths.append(path)

        return images, valid_paths

    def _generate_image_embeddings(
        self, image_paths: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for multiple images using CLIP with GPU acceleration and parallel loading"""
        try:
            # Increased batch size for better GPU utilization
            batch_size = 64 if self.device.type == "cuda" else 16

            # Load images in parallel
            images, valid_paths = self._load_image_batch(image_paths)

            if not images:
                return None, []

            # Process images in batches
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]

                # Preprocess batch
                inputs = self.processor(
                    images=batch_images, return_tensors="pt", padding=True
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize features for cosine similarity
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )

                all_embeddings.append(image_features.cpu().numpy())

                # Clear GPU cache after each batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # Combine all embeddings
            embeddings = np.concatenate(all_embeddings, axis=0)

            return embeddings, valid_paths

        except Exception as e:
            print(f"Error processing images: {e}")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return None, []

    def _is_duplicate_faiss(self, embedding: np.ndarray) -> bool:
        """Check if embedding is similar to any previously processed embedding using FAISS"""
        with self.faiss_lock:
            if self.faiss_index.ntotal == 0:
                return False

            # Search for similar embeddings
            embedding_normalized = embedding / np.linalg.norm(embedding)
            similarities, _ = self.faiss_index.search(
                embedding_normalized.reshape(1, -1), 1
            )

            max_similarity = similarities[0][0]
            return max_similarity >= self.similarity_threshold

    def _add_to_faiss_index(self, embedding: np.ndarray):
        """Add embedding to FAISS index for future similarity checks"""
        with self.faiss_lock:
            embedding_normalized = embedding / np.linalg.norm(embedding)
            self.faiss_index.add(embedding_normalized.reshape(1, -1))

    def _generate_crop_id(self, file_path: str, product_id: str) -> str:
        """Generate unique ID for crop"""
        crop_name = Path(file_path).stem
        combined = f"{product_id}_{crop_name}"
        return hashlib.md5(combined.encode()).hexdigest()

    def find_product_directories(self) -> List[Dict[str, Any]]:
        """Find all processed product directories"""
        product_dirs = []

        if not os.path.exists(self.processed_data_path):
            print(f"Processed data path does not exist: {self.processed_data_path}")
            return product_dirs

        for item in os.listdir(self.processed_data_path):
            product_path = os.path.join(self.processed_data_path, item)

            if not os.path.isdir(product_path):
                continue

            # Check for product_info.json
            product_info_path = os.path.join(product_path, "product_info.json")
            if not os.path.exists(product_info_path):
                continue

            # Check for bbox_crops directory
            bbox_crops_dir = os.path.join(product_path, "bbox_crops")
            if not os.path.exists(bbox_crops_dir):
                continue

            # Get all bbox crop images
            crop_files = glob.glob(os.path.join(bbox_crops_dir, "*.png"))
            crop_files.extend(glob.glob(os.path.join(bbox_crops_dir, "*.jpg")))

            if not crop_files:
                continue

            product_dirs.append(
                {
                    "path": product_path,
                    "product_info_path": product_info_path,
                    "bbox_crops_dir": bbox_crops_dir,
                    "crop_files": crop_files,
                    "product_name": item,
                }
            )

        return product_dirs

    def process_product_crops(
        self, product_dir: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process all bbox crops for a single product"""
        try:
            # Load product info
            with open(product_dir["product_info_path"], "r") as f:
                product_info = json.load(f)

            product_id = product_info.get("product_id", "unknown")

            # Generate embeddings for all crops
            embeddings, valid_paths = self._generate_image_embeddings(
                product_dir["crop_files"]
            )

            if embeddings is None or len(embeddings) == 0:
                return []

            unique_crops = []

            for embedding, file_path in zip(embeddings, valid_paths):
                # Check for duplicates
                if self._is_duplicate_faiss(embedding):
                    self.duplicate_count += 1
                    continue

                # Add to FAISS index for future deduplication
                self._add_to_faiss_index(embedding)
                self.unique_count += 1

                # Generate crop ID
                crop_id = self._generate_crop_id(file_path, product_id)

                # Extract crop info from filename
                filename = Path(file_path).stem
                parts = filename.split("_")
                class_name = "unknown"
                if len(parts) >= 4:  # format: imagename_crop_i_classname_bbox
                    class_name = parts[3]

                # Prepare metadata
                metadata = {
                    "product_id": product_id,
                    "product_name": product_dir["product_name"],
                    "file_path": file_path,
                    "filename": Path(file_path).name,
                    "class_name": class_name,
                    "title": product_info.get("title", ""),
                    "description": product_info.get("description", ""),
                    "product_type": product_info.get("product_type", ""),
                    "price": product_info.get("price_display_amount", ""),
                    "tags": product_info.get("product_tags", ""),
                    "collections": product_info.get("product_collections", ""),
                }

                unique_crops.append(
                    {"id": crop_id, "embedding": embedding, "metadata": metadata}
                )

            return unique_crops

        except Exception as e:
            print(f"Error processing product {product_dir['product_name']}: {e}")
            return []

    def process_product_crops_batch(
        self, product_dirs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple products together for better GPU utilization"""
        all_crops = []

        # Collect all crop paths from all products
        all_file_paths = []
        path_to_product = {}

        for product_dir in product_dirs:
            for file_path in product_dir["crop_files"]:
                all_file_paths.append(file_path)
                path_to_product[file_path] = product_dir

        if not all_file_paths:
            return []

        # Generate embeddings for all crops at once
        embeddings, valid_paths = self._generate_image_embeddings(all_file_paths)

        if embeddings is None or len(embeddings) == 0:
            return []

        # Process each embedding
        for embedding, file_path in zip(embeddings, valid_paths):
            product_dir = path_to_product[file_path]

            # Check for duplicates using FAISS
            if self._is_duplicate_faiss(embedding):
                self.duplicate_count += 1
                continue

            # Add to FAISS index for future deduplication
            self._add_to_faiss_index(embedding)
            self.unique_count += 1

            # Load product info
            try:
                with open(product_dir["product_info_path"], "r") as f:
                    product_info = json.load(f)
            except:  # noqa
                product_info = {}

            product_id = product_info.get("product_id", "unknown")

            # Generate crop ID
            crop_id = self._generate_crop_id(file_path, product_id)

            # Extract crop info from filename
            filename = Path(file_path).stem
            parts = filename.split("_")
            class_name = "unknown"
            if len(parts) >= 4:  # format: imagename_crop_i_classname_bbox
                class_name = parts[3]

            # Prepare metadata
            metadata = {
                "product_id": product_id,
                "product_name": product_dir["product_name"],
                "file_path": file_path,
                "filename": Path(file_path).name,
                "class_name": class_name,
                "title": product_info.get("title", ""),
                "description": product_info.get("description", ""),
                "product_type": product_info.get("product_type", ""),
                "price": product_info.get("price_display_amount", ""),
                "tags": product_info.get("product_tags", ""),
                "collections": product_info.get("product_collections", ""),
            }

            all_crops.append(
                {"id": crop_id, "embedding": embedding, "metadata": metadata}
            )

        return all_crops

    def vectorize_all_crops(
        self, max_products: int = None, product_batch_size: int = 10
    ):
        """Process all product crops and store unique embeddings in Qdrant with optimized batching"""
        start_time = time.time()
        start_datetime = datetime.now()

        print(
            f"\nStarting crop vectorization at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Similarity threshold for deduplication: {self.similarity_threshold}")
        print(f"Max workers for parallel I/O: {self.max_workers}")
        print(f"Product batch size: {product_batch_size}")

        # Initialize FAISS index
        self._setup_faiss_index()

        # Find all product directories
        product_dirs = self.find_product_directories()

        if not product_dirs:
            print("No valid product directories found!")
            return

        if max_products:
            product_dirs = product_dirs[:max_products]

        total_products = len(product_dirs)
        total_crops_processed = 0
        points_batch = []
        qdrant_batch_size = 200  # Increased batch size for Qdrant

        print(f"Found {total_products} product directories")

        # Monitor GPU memory if using CUDA
        if self.device.type == "cuda":
            print(
                f"Initial GPU memory - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        # Process products in batches
        with tqdm(total=total_products, desc="Processing Products") as pbar:
            for i in range(0, total_products, product_batch_size):
                batch_end = min(i + product_batch_size, total_products)
                product_batch = product_dirs[i:batch_end]

                pbar.set_description(f"Processing batch {i // product_batch_size + 1}")

                # Process batch of products together
                unique_crops = self.process_product_crops_batch(product_batch)

                # Add to Qdrant batch
                for crop_data in unique_crops:
                    point = PointStruct(
                        id=crop_data["id"],
                        vector=crop_data["embedding"].tolist(),
                        payload=crop_data["metadata"],
                    )
                    points_batch.append(point)
                    total_crops_processed += 1

                    # Batch insert when batch is full
                    if len(points_batch) >= qdrant_batch_size:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points_batch
                        )
                        points_batch = []

                # Update progress bar
                pbar.update(len(product_batch))
                pbar.set_postfix(
                    {
                        "Unique": self.unique_count,
                        "Duplicates": self.duplicate_count,
                        "Total": total_crops_processed,
                        "Rate": f"{self.duplicate_count / (self.unique_count + self.duplicate_count) * 100:.1f}%"
                        if (self.unique_count + self.duplicate_count) > 0
                        else "0%",
                    }
                )

                # Show GPU memory usage periodically
                if (
                    self.device.type == "cuda"
                    and (i // product_batch_size + 1) % 5 == 0
                ):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    pbar.write(f"GPU memory allocated: {allocated:.2f} GB")

        # Insert remaining points
        if points_batch:
            self.client.upsert(
                collection_name=self.collection_name, points=points_batch
            )

        # Final statistics
        end_time = time.time()
        end_datetime = datetime.now()
        total_elapsed = end_time - start_time
        elapsed_str = str(timedelta(seconds=int(total_elapsed)))

        print("\n" + "=" * 80)
        print("CROP VECTORIZATION COMPLETE!")
        print("=" * 80)
        print(f"Products processed: {total_products}")
        print(f"Unique crops stored: {self.unique_count}")
        print(f"Duplicates filtered: {self.duplicate_count}")
        print(f"Total crops processed: {total_crops_processed}")
        print(
            f"Deduplication rate: {(self.duplicate_count / (self.unique_count + self.duplicate_count) * 100):.1f}%"
        )
        print(
            f"Processing rate: {total_crops_processed / total_elapsed:.1f} crops/second"
        )
        print(f"Total time elapsed: {elapsed_str}")
        print(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Final GPU memory report
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Final GPU memory allocated: {allocated:.2f} GB")

    def search_crops(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar crops using text query"""
        try:
            # Generate text embedding
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            query_embedding = text_features.cpu().numpy()[0]

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
            )

            # Format results
            results = []
            for result in search_results:
                results.append(
                    {
                        "crop_id": result.id,
                        "score": result.score,
                        "product_id": result.payload.get("product_id"),
                        "product_name": result.payload.get("product_name"),
                        "class_name": result.payload.get("class_name"),
                        "title": result.payload.get("title"),
                        "file_path": result.payload.get("file_path"),
                        "filename": result.payload.get("filename"),
                    }
                )

            return results

        except Exception as e:
            print(f"Error searching crops: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": str(collection_info.status),
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "similarity_threshold": self.similarity_threshold,
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorize fashion product bbox crops using CLIP with deduplication"
    )
    parser.add_argument(
        "--processed-data-path",
        default="/root/flickd-ai/tagging-engine/data/processed/product_images",
        help="Path to processed product images directory",
    )
    parser.add_argument(
        "--max-products", type=int, help="Maximum number of products to process"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for deduplication (0.0-1.0)",
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("QDRANT_COLLECTION_NAME"),
        help="Qdrant collection name (uses QDRANT_COLLECTION_NAME from .env)",
    )
    parser.add_argument(
        "--search", type=str, help="Search query to test the vectorized crops"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of worker threads for parallel I/O",
    )
    parser.add_argument(
        "--product-batch-size",
        type=int,
        default=10,
        help="Number of products to process together in each batch",
    )

    args = parser.parse_args()

    vectorizer = CropVectorizer(
        processed_data_path=args.processed_data_path,
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_collection_name=args.collection_name,
        similarity_threshold=args.similarity_threshold,
        max_workers=args.max_workers,
    )

    if args.search:
        # Search mode
        print(f"\nSearching for: '{args.search}'")
        results = vectorizer.search_crops(args.search, top_k=10)

        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(
                    f"{i}. [{result['score']:.3f}] {result['class_name']} - {result['title'][:50]}..."
                )
                print(
                    f"   Product: {result['product_name']} | File: {result['filename']}"
                )
        else:
            print("No results found")
    else:
        # Vectorization mode
        vectorizer.vectorize_all_crops(
            max_products=args.max_products, product_batch_size=args.product_batch_size
        )

        # Show collection info
        info = vectorizer.get_collection_info()
        if info:
            print("\nCollection Info:")
            print(f"   Name: {info['name']}")
            print(f"   Points: {info['points_count']}")
            print(f"   Vectors: {info['vectors_count']}")
            print(f"   Status: {info['status']}")
