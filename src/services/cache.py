import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import UPLOAD_DIR

logger = logging.getLogger(__name__)


class FileProcessingCache:
    def __init__(self):
        self.cache_dir = UPLOAD_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for a given hash"""
        return self.cache_dir / f"{file_hash}.json"

    def get_cached_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached processing result"""
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                logger.info(f"Found cached result for hash: {file_hash}")
                return cached_data
        except Exception as e:
            logger.error(f"Error reading cache file {cache_path}: {e}")
            return None

    def store_result(
        self, file_hash: str, result: Dict[str, Any], original_filename: str
    ) -> None:
        """Store processing result in cache"""
        cache_path = self._get_cache_path(file_hash)

        try:
            cache_data = {
                "file_hash": file_hash,
                "original_filename": original_filename,
                "cached_at": datetime.utcnow().isoformat(),
                "result": result,
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

            logger.info(f"Cached result for hash: {file_hash}")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")

    def clear_cache(self) -> int:
        """Clear all cached files and return count of files removed"""
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        return count
