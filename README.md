# CFPB Complaint RAG Chatbot - Interim Submission

## Project Overview

This project builds the data preparation and vector indexing pipeline for a Retrieval-Augmented Generation (RAG) chatbot using the CFPB consumer complaint dataset.

The goal is to prepare complaint narratives so they can later be searched semantically and used by a chatbot to answer user questions about financial complaints.

This interim submission includes:

* Task 1: Exploratory Data Analysis and Data Preprocessing
* Task 2: Text Chunking, Embedding, and Vector Store Indexing

## Dataset

The project uses the full CFPB consumer complaint dataset.

The raw dataset should be placed locally at:

```text
data/raw/complaints.csv
```

Because the raw file and cleaned full dataset are large, they are not included in the GitHub repository.

## Task 1: EDA and Data Preprocessing

In Task 1, the full CFPB dataset was loaded and analyzed.

The EDA included:

* Distribution of complaints across products
* Count of complaints with and without consumer complaint narratives
* Narrative length analysis using word count
* Visualization of product distribution
* Visualization of narrative length distribution

The dataset was then filtered to keep only the four required product categories:

* Credit Card
* Personal Loan
* Savings Account
* Money Transfer

Rows with missing or empty consumer complaint narratives were removed.

The complaint narratives were cleaned by:

* Converting text to lowercase
* Removing boilerplate complaint phrases
* Removing URLs
* Removing special characters
* Removing CFPB redaction placeholders such as `XXXX`
* Normalizing extra whitespace

The cleaned dataset was saved as:

```text
data/processed/complaints_task1_cleaned.csv
```

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

In Task 2, a stratified sample of 12,000 complaints was created from the cleaned dataset.

Although the Task 2 instruction mentioned five product categories, Task 1 explicitly required filtering the dataset to four products. Therefore, the stratified sample was created proportionally across the four retained product categories:

* Credit Card
* Personal Loan
* Savings Account
* Money Transfer

Stratified sampling was used to preserve the original product distribution and avoid underrepresenting smaller categories such as Personal Loan.

Long complaint narratives were split into smaller chunks before embedding. This was done because long narratives can lose semantic precision when embedded as a single vector.

The final chunking strategy used was:

```text
chunk_size = 500
chunk_overlap = 100
```

This setting was selected because it provides a good balance between preserving context and keeping each chunk short enough for effective semantic search.

The embedding model used was:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model was selected because it is lightweight, efficient, and suitable for semantic similarity search. Each text chunk was converted into a 384-dimensional vector embedding.

FAISS was used to create the vector store. Each vector was stored with metadata including:

* Complaint ID
* Product category
* Chunk index
* Chunk text

The vector store outputs are saved in:

```text
vector_store/
```

Expected files include:

```text
vector_store/faiss_index.bin
vector_store/chunk_metadata.csv
vector_store/chunk_embeddings.npy
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── complaints.csv
│   └── processed/
│       ├── complaints_task1_cleaned.csv
│       └── task2_stratified_sample.csv
├── notebooks/
│   ├── eda_preprocessing.ipynb
│   └── task2_vector_store.ipynb
├── vector_store/
│   ├── faiss_index.bin
│   ├── chunk_metadata.csv
│   └── chunk_embeddings.npy
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib tqdm ipykernel
python -m pip install sentence-transformers faiss-cpu langchain-text-splitters
```

### 3. Run Task 1 notebook

Open and run:

```text
notebooks/eda_preprocessing.ipynb
```

This notebook performs EDA, filtering, and text preprocessing.

### 4. Run Task 2 notebook

Open and run:

```text
notebooks/task2_vector_store.ipynb
```

This notebook creates the stratified sample, chunks narratives, generates embeddings, builds the FAISS vector store, and tests semantic search.

## Interim Submission Status

Task 1 and Task 2 have been completed.

Completed outputs:

* Cleaned complaint dataset
* Stratified sample
* Text chunks
* Embeddings
* FAISS vector index
* Metadata for tracing chunks back to original complaints
* Semantic search test
