"""File hashing helpers."""
import hashlib
from pathlib import Path


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
