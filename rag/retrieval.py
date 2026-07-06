import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_top_chunks(dataframe, question_embedding, top_k=3):
    similarities = cosine_similarity([question_embedding], dataframe["embedding"].tolist())
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    return dataframe.iloc[top_indices]["chunk"].tolist()
