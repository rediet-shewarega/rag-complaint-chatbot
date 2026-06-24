# Intelligent Complaint Analysis for Financial Services

## Introduction

CrediTrust Financial receives a large number of customer complaints across products such as credit cards, personal loans, savings accounts, and money transfers. Manually reviewing these complaints is slow and makes it difficult for product, support, and compliance teams to identify important customer pain points quickly.

This project builds a Retrieval-Augmented Generation system that allows internal users to ask plain-English questions about customer complaints and receive evidence-based answers supported by retrieved complaint excerpts.

## Technical Choices

The system uses the provided `complaint_embeddings.parquet` file for Task 3 and Task 4. This file contains pre-built embeddings, text chunks, and metadata for the complaint dataset.

The embeddings were created using `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional sentence embeddings suitable for semantic search. The full dataset was converted into a FAISS vector store for efficient similarity search.

The RAG pipeline has three main stages:

1. Embed the user question using all-MiniLM-L6-v2.
2. Search the FAISS index for the top-k most relevant complaint chunks.
3. Generate an evidence-based answer using the retrieved complaint excerpts.

## Vector Store

The mandatory `complaint_embeddings.parquet` file was used to build the final FAISS vector store.

Final vector store details:

- Vector database: FAISS
- Total vectors/chunks: 1,375,327
- Embedding model: all-MiniLM-L6-v2
- Metadata storage: SQLite
- Source file: complaint_embeddings.parquet

Large files such as `complaint_embeddings.parquet` and `vector_store_full/` are excluded from GitHub because of size limits.

## RAG Pipeline

The RAG logic is implemented in `src/rag_pipeline.py`.

The pipeline includes:

- Loading the full FAISS vector store
- Loading metadata from SQLite
- Embedding the user question
- Retrieving top-k complaint chunks
- Building a grounded prompt
- Returning an answer with source evidence

The prompt instructs the assistant to act as a financial analyst for CrediTrust and to answer only using retrieved complaint excerpts.

## Evaluation

The RAG system was evaluated using representative questions across the supported financial products.

Paste the contents of `reports/evaluation_table.md` here.

## UI Showcase

The Gradio interface is implemented in `app.py`.

The interface includes:

- A question input box
- Product filter
- Top-k source selector
- Ask button
- Clear button
- AI-generated answer section
- Retrieved sources section

Screenshots of the working UI are included in the `reports/images/` folder.

## Challenges

One challenge was working with the large `complaint_embeddings.parquet` file and converting it into a FAISS index locally. Another challenge was managing memory while loading the full vector store and language model on a local machine.

To keep the system stable, the implementation focuses on reliable retrieval and evidence-grounded answer generation using the retrieved complaint chunks. The system is modular, so a stronger LLM can be connected later when more compute resources are available.

## Future Improvements

Future improvements include:

- Deploying the Gradio app online
- Adding stronger LLM support
- Improving product/category filtering
- Adding conversation history
- Adding better source ranking and reranking
- Adding streaming responses

## Conclusion

This project demonstrates how customer complaint data can be transformed into an internal decision-support tool. The RAG chatbot helps users quickly identify complaint themes, inspect supporting evidence, and understand customer pain points across financial services.