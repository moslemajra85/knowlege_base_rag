import csv

import pandas as pd

from rag.chunking import split_into_chunks


def load_chunks_from_csv(dataset_path, chunk_size=1024, content_column_index=1):
    chunks = []

    with open(dataset_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)

        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            chunks.extend(split_into_chunks(row[content_column_index], chunk_size=chunk_size))

    return pd.DataFrame(chunks, columns=["chunk"])
