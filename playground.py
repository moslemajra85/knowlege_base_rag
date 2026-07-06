 
import os
import hashlib
import json
import pandas as pd

CHUNK_SIZE = 1024
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
DATASET_PATH = "./mini-llama-articles.csv"
CHUNK_SIZE = 1024
CACHE_DIR = "./.cache"
CACHE_PATH = os.path.join(CACHE_DIR, "document_embeddings.json")
CACHE_VERSION = 1

def file_sha256(path):
    hasher = hashlib.sha256()

    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024*1024), b""):
            hasher.update(block)
        
    return hasher.hexdigest()

file_hash = file_sha256("./mini-llama-articles.csv")
print(file_hash)


def build_cache_metadata(dataset_path, chunk_size, embedding_model):
    return {

        "cacher_version": CACHE_VERSION,
        "dataset_path": dataset_path,
        "dataset_sha256": file_sha256(dataset_path),
        "chunk_size": CHUNK_SIZE,
        "embedding_model": embedding_model
    }


def load_embedding_cache(cache_path, expected_metadata):
    if not os.path.exists(cache_path):
        return None
    
    with open(cache_path, 'r', encoding='utf-8') as file:
        cache = json.load(file)

        if cache.get["metadata"] != expected_metadata:
            return None
    
    records = cache.get['records']

    if not isinstance(records, list):
        return None
    
    return pd.DataFrame(records, columns=['chunk', 'embedding'])

    

def save_embedding_cache(cache_path, metadata, dataframe):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)