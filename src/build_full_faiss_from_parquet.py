import os
import json
import sqlite3

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PARQUET_PATH = "data/processed/complaint_embeddings.parquet"

OUTPUT_DIR = "vector_store_full"
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.bin")
SQLITE_PATH = os.path.join(OUTPUT_DIR, "chunk_metadata.sqlite")

BATCH_SIZE = 10000


def find_embedding_column(columns):
    possible_names = ["embedding", "embeddings", "vector", "chunk_embedding"]

    for name in possible_names:
        if name in columns:
            return name

    raise ValueError(f"No embedding column found. Available columns: {columns}")


def convert_embeddings(series):
    embeddings = []

    for value in series:
        if isinstance(value, np.ndarray):
            embeddings.append(value.astype("float32"))
        elif isinstance(value, list):
            embeddings.append(np.array(value, dtype="float32"))
        elif isinstance(value, str):
            parsed = json.loads(value)
            embeddings.append(np.array(parsed, dtype="float32"))
        else:
            raise ValueError(f"Unsupported embedding type: {type(value)}")

    return np.vstack(embeddings).astype("float32")


def clean_value(value):
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    if isinstance(value, (list, dict)):
        return json.dumps(value)

    return str(value)


def clean_metadata_df(df):
    cleaned = df.copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].apply(clean_value)

    return cleaned


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)

    if os.path.exists(SQLITE_PATH):
        os.remove(SQLITE_PATH)

    print("Opening parquet file...")
    parquet_file = pq.ParquetFile(PARQUET_PATH)

    print("Total rows:", parquet_file.metadata.num_rows)
    print("Columns:", parquet_file.schema_arrow.names)

    embedding_col = find_embedding_column(parquet_file.schema_arrow.names)
    print("Using embedding column:", embedding_col)

    conn = sqlite3.connect(SQLITE_PATH)

    faiss_index = None
    total_rows = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        start=1
    ):
        df = batch.to_pandas()

        embeddings = convert_embeddings(df[embedding_col])

        # Use cosine similarity by normalizing embeddings and using inner product
        faiss.normalize_L2(embeddings)

        if faiss_index is None:
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            print("Embedding dimension:", dimension)

        faiss_index.add(embeddings)

        metadata_df = df.drop(columns=[embedding_col])
        metadata_df.insert(
            0,
            "row_id",
            range(total_rows, total_rows + len(metadata_df))
        )

        metadata_df = clean_metadata_df(metadata_df)

        metadata_df.to_sql(
            "metadata",
            conn,
            if_exists="append",
            index=False
        )

        total_rows += len(df)

        print(f"Processed batch {batch_number} | total rows: {total_rows}")

    print("Creating SQLite index...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_row_id ON metadata(row_id)")
    conn.commit()
    conn.close()

    print("Saving FAISS index...")
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    print("Done.")
    print("FAISS vectors:", faiss_index.ntotal)
    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Metadata database saved to:", SQLITE_PATH)


if __name__ == "__main__":
    main()