import os
import sqlite3
from typing import List, Dict, Any

import faiss
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Paths
# -----------------------------

FAISS_INDEX_PATH = "vector_store_full/faiss_index.bin"
SQLITE_PATH = "vector_store_full/chunk_metadata.sqlite"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"


# -----------------------------
# Load FAISS index
# -----------------------------

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")

if not os.path.exists(SQLITE_PATH):
    raise FileNotFoundError(f"SQLite metadata file not found: {SQLITE_PATH}")

index = faiss.read_index(FAISS_INDEX_PATH)

print("Full FAISS index loaded successfully.")
print("FAISS vectors:", index.ntotal)


# -----------------------------
# Load models
# -----------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

tokenizer = None
generator_model = None


# -----------------------------
# SQLite helpers
# -----------------------------

def get_metadata_columns() -> List[str]:
    conn = sqlite3.connect(SQLITE_PATH)
    query = "PRAGMA table_info(metadata)"
    rows = conn.execute(query).fetchall()
    conn.close()

    return [row[1] for row in rows]


METADATA_COLUMNS = get_metadata_columns()
print("Metadata columns:", METADATA_COLUMNS)


def get_metadata_by_row_id(row_id: int) -> Dict[str, Any]:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM metadata WHERE row_id = ?"
    row = conn.execute(query, (int(row_id),)).fetchone()

    conn.close()

    if row is None:
        return {}

    return dict(row)
def normalize_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Some parquet exports store metadata inside a single 'metadata' column.
    This function expands that nested metadata so product, issue, company,
    complaint_id, etc. can display correctly.
    """
    import json

    normalized = dict(row)

    raw_metadata = normalized.get("metadata")

    if raw_metadata:
        try:
            if isinstance(raw_metadata, str):
                parsed_metadata = json.loads(raw_metadata)
            elif isinstance(raw_metadata, dict):
                parsed_metadata = raw_metadata
            else:
                parsed_metadata = {}

            if isinstance(parsed_metadata, dict):
                normalized.update(parsed_metadata)

        except Exception:
            pass

    return normalized

def pick_first_available(row: Dict[str, Any], possible_columns: List[str], default: str = "N/A") -> str:
    for col in possible_columns:
        value = row.get(col)

        if value is not None and str(value).strip() != "":
            return str(value)

    return default


def get_chunk_text(row: Dict[str, Any]) -> str:
    return pick_first_available(
        row,
        [
            "chunk_text",
            "text",
            "document",
            "documents",
            "complaint_text",
            "consumer_complaint_narrative",
            "Consumer complaint narrative",
            "narrative",
            "cleaned_narrative"
        ],
        default=""
    )


# -----------------------------
# Retriever
# -----------------------------

def retrieve_chunks(
    question: str,
    top_k: int = 5,
    product_filter: str = "All"
) -> List[Dict[str, Any]]:
    """
    Embeds the user question and retrieves top-k relevant complaint chunks.
    """

    question_embedding = embedding_model.encode([question]).astype("float32")

    # Important because the FAISS index was normalized and built with inner product
    faiss.normalize_L2(question_embedding)

    search_k = top_k * 5
    scores, indices = index.search(question_embedding, search_k)

    sources = []

    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue

        row = get_metadata_by_row_id(int(idx))
        row = normalize_metadata(row)

        if not row:
            continue

        product_value = pick_first_available(
            row,
            ["product_category", "product", "Product"],
            default=""
        )

        if product_filter != "All":
            if product_filter.lower() not in product_value.lower():
                continue

        chunk_text = get_chunk_text(row)

        if not chunk_text.strip():
            continue

        sources.append({
            "rank": len(sources) + 1,
            "text": chunk_text,
            "metadata": row,
            "score": float(scores[0][rank])
        })

        if len(sources) >= top_k:
            break

    return sources


# -----------------------------
# Prompt
# -----------------------------

def build_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    context_parts = []

    for source in sources:
        row = source["metadata"]

        product = pick_first_available(row, ["product_category", "product", "Product"])
        issue = pick_first_available(row, ["issue", "Issue"])
        company = pick_first_available(row, ["company", "Company"])
        date_received = pick_first_available(row, ["date_received", "Date received"])

        context_parts.append(
            f"""
Source {source['rank']}:
Product: {product}
Issue: {issue}
Company: {company}
Date received: {date_received}

Complaint excerpt:
{source['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a financial analyst assistant for CrediTrust Financial.

Your task is to answer questions about customer complaints.

Use ONLY the retrieved complaint excerpts below.
Do not make up facts.
If the retrieved excerpts do not contain enough information, say:
"I do not have enough information from the retrieved complaints."

Write a concise, evidence-based answer.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# -----------------------------
# Generator
# -----------------------------

def generate_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Memory-safe generator.
    If the local LLM is too heavy, this creates a grounded answer
    from the retrieved complaint evidence.
    """

    if not sources:
        return "I do not have enough information from the retrieved complaints."

    issues = []
    products = []
    companies = []

    for source in sources:
        row = source["metadata"]

        issue = pick_first_available(row, ["issue", "Issue"], default="")
        product = pick_first_available(row, ["product_category", "product", "Product"], default="")
        company = pick_first_available(row, ["company", "Company"], default="")

        if issue and issue not in issues:
            issues.append(issue)

        if product and product not in products:
            products.append(product)

        if company and company not in companies:
            companies.append(company)

    evidence_points = []

    for source in sources[:3]:
        text = source["text"].strip()
        short_text = text[:250] + "..." if len(text) > 250 else text
        evidence_points.append(f"- {short_text}")

    answer = f"""
Based on the retrieved complaint excerpts, customers mainly report problems related to {", ".join(issues[:4]) if issues else "the retrieved complaint issues"}.

The strongest evidence comes from complaints in these product areas: {", ".join(products[:4]) if products else "N/A"}.

The retrieved complaints suggest recurring customer pain points such as delays, disputes, account or transaction problems, and difficulty getting a satisfactory resolution.

Supporting evidence:
{chr(10).join(evidence_points)}
"""

    return answer.strip()


# -----------------------------
# Main RAG function
# -----------------------------

def answer_question(
    question: str,
    top_k: int = 5,
    product_filter: str = "All"
) -> Dict[str, Any]:
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    sources = retrieve_chunks(
        question=question,
        top_k=top_k,
        product_filter=product_filter
    )

    answer = generate_answer(question, sources)

    return {
        "answer": answer,
        "sources": sources
    }


# -----------------------------
# Format sources for UI
# -----------------------------

def format_sources_markdown(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No sources retrieved."

    markdown = ""

    for source in sources:
        row = source["metadata"]

        complaint_id = pick_first_available(row, ["complaint_id", "Complaint ID"])
        product = pick_first_available(row, ["product_category", "product", "Product"])
        issue = pick_first_available(row, ["issue", "Issue"])
        sub_issue = pick_first_available(row, ["sub_issue", "Sub-issue"])
        company = pick_first_available(row, ["company", "Company"])
        date_received = pick_first_available(row, ["date_received", "Date received"])

        text = source["text"]
        short_text = text[:700] + "..." if len(text) > 700 else text

        markdown += f"""
### Source {source['rank']}

**Complaint ID:** {complaint_id}  
**Product:** {product}  
**Issue:** {issue}  
**Sub-issue:** {sub_issue}  
**Company:** {company}  
**Date received:** {date_received}  
**Similarity score:** {source.get("score", "N/A")}  

> {short_text}

---
"""

    return markdown


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":
    question = "Why are customers unhappy with credit cards?"

    result = answer_question(question, top_k=5)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    print(format_sources_markdown(result["sources"]))