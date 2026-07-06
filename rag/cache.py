import hashlib
import json
import os

import pandas as pd


CACHE_VERSION = 1


def file_sha256(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def build_cache_metadata(dataset_path, chunk_size, embedding_model):
    return {
        "cache_version": CACHE_VERSION,
        "dataset_path": dataset_path,
        "dataset_sha256": file_sha256(dataset_path),
        "chunk_size": chunk_size,
        "embedding_model": embedding_model,
    }


def load_embedding_cache(cache_path, expected_metadata):
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, mode="r", encoding="utf-8") as file:
        cache = json.load(file)

    if cache.get("metadata") != expected_metadata:
        return None

    records = cache.get("records")
    if not isinstance(records, list):
        return None

    return pd.DataFrame(records, columns=["chunk", "embedding"])


def save_embedding_cache(cache_path, metadata, dataframe):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    records = dataframe[["chunk", "embedding"]].to_dict(orient="records")
    payload = {
        "metadata": metadata,
        "records": records,
    }

    with open(cache_path, mode="w", encoding="utf-8") as file:
        json.dump(payload, file)
