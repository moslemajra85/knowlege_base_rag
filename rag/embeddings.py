import pandas as pd
from tqdm import tqdm


def get_embedding(client, text, embedding_model):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=embedding_model)
        return response.data[0].embedding
    except Exception as exc:
        raise RuntimeError(f"Failed to create embedding: {exc}") from exc


def add_embeddings_to_dataframe(dataframe, embedding_fn):
    embeddings = []

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        embeddings.append(embedding_fn(row["chunk"]))

    result = dataframe.copy()
    result.insert(loc=1, column="embedding", value=pd.Series(embeddings))
    return result
