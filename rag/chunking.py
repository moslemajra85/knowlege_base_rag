def split_into_chunks(text, chunk_size=1024):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i: i + chunk_size])

    return chunks
