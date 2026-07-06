import os

from dotenv import load_dotenv
from openai import OpenAI

from rag.cache import build_cache_metadata, load_embedding_cache, save_embedding_cache
from rag.data import load_chunks_from_csv
from rag.embeddings import add_embeddings_to_dataframe, get_embedding
from rag.prompting import SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import retrieve_top_chunks


load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DATASET_PATH = "./mini-llama-articles.csv"
CHUNK_SIZE = 1024
CACHE_PATH = "./.cache/document_embeddings.json"
NUMBER_OF_CHUNKS_TO_RETRIEVE = 3
QUESTION = "How many parameters does the LLaMA2 model have?"


def require_openai_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")


def load_or_create_document_embeddings(client):
    df = load_chunks_from_csv(DATASET_PATH, chunk_size=CHUNK_SIZE)
    cache_metadata = build_cache_metadata(
        dataset_path=DATASET_PATH,
        chunk_size=CHUNK_SIZE,
        embedding_model=OPENAI_EMBEDDING_MODEL,
    )
    cached_df = load_embedding_cache(CACHE_PATH, cache_metadata)

    if cached_df is not None:
        print(f"Loaded {len(cached_df)} document embeddings from {CACHE_PATH}")
        return cached_df

    print("Generating document embeddings")
    df = add_embeddings_to_dataframe(
        df,
        embedding_fn=lambda text: get_embedding(client, text, OPENAI_EMBEDDING_MODEL),
    )
    save_embedding_cache(CACHE_PATH, cache_metadata, df)
    print(f"Saved {len(df)} document embeddings to {CACHE_PATH}")
    return df


def main():
    require_openai_api_key()
    client = OpenAI()

    df = load_or_create_document_embeddings(client)
    question_embedding = get_embedding(client, QUESTION, OPENAI_EMBEDDING_MODEL)
    retrieved_chunks = retrieve_top_chunks(
        df,
        question_embedding=question_embedding,
        top_k=NUMBER_OF_CHUNKS_TO_RETRIEVE,
    )
    prompt = build_user_prompt(question=QUESTION, context_chunks=retrieved_chunks)

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
