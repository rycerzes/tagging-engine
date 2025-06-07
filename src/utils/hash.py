import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()
