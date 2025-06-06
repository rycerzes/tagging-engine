import numpy as np
import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class QueryService:
    def __init__(self):
        # Qdrant configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.fashion_products_collection = os.getenv("QDRANT_COLLECTION_NAME")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL must be set in environment variables")

        # Parse Qdrant URL to extract host and port
        parsed_url = urlparse(self.qdrant_url)
        qdrant_host = parsed_url.hostname
        qdrant_port = parsed_url.port or 6333

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

    def _query_fashion_products(
        self, embedding: np.ndarray, limit: int = 2
    ) -> List[Dict[str, Any]]:
        """Query similar products from fashion_products collection"""
        try:
            search_results = self.client.query_points(
                collection_name=self.fashion_products_collection,
                query=embedding.tolist(),
                limit=limit,
                score_threshold=0.75,
                with_payload=True,
            )

            results = []
            for result in search_results.points:
                results.append(
                    {
                        "product_id": result.id,
                        "score": result.score,
                        "payload": result.payload,
                    }
                )

            return results
        except Exception as e:
            print(f"Error querying fashion_products collection: {e}")
            return []

    async def query_video_crops(self, video_id: str) -> List[Dict[str, Any]]:
        """Query all crops for a video and their fashion product matches"""
        try:
            # Get all points from the video collection
            scroll_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.scroll(
                    collection_name=video_id,
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                ),
            )

            results = []
            for point in scroll_result[0]:
                # Query fashion products for each embedding
                embedding = np.array(point.vector)
                fashion_matches = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._query_fashion_products(embedding, limit=5)
                )

                result = {
                    "crop_id": point.id,
                    "filename": point.payload.get("filename"),
                    "class_name": point.payload.get("class_name"),
                    "original_class_name": point.payload.get("original_class_name"),
                    "bbox": point.payload.get("bbox", []),
                    "fashion_matches": fashion_matches,
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Error querying video crops: {e}")
            return []
