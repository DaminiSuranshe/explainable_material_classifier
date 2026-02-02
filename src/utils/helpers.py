"""
Caching helpers.
"""

import hashlib
from pathlib import Path
import json


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def image_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def get_cached_result(hash_key: str):
    path = CACHE_DIR / f"{hash_key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def set_cached_result(hash_key: str, result: dict):
    path = CACHE_DIR / f"{hash_key}.json"
    path.write_text(json.dumps(result))
