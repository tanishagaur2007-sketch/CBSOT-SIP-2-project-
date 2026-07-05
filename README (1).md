# AI Search Intelligence System

A semantic search engine for machine learning research papers. It combines sentence embeddings, FAISS similarity search, automatic summarization, and keyword extraction, served through a FastAPI backend.

## Overview

Traditional keyword search fails when a query and a paper describe the same idea using different words. This project builds a **semantic** search pipeline over the [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) dataset so that a query like *"deep learning for medical image analysis"* returns conceptually relevant papers, even if they don't share exact keywords, then enriches each result with a generated summary and extracted keywords.

## Features

- **Semantic search** — encodes paper titles + abstracts into dense vector embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Fast similarity search** — indexes embeddings with **FAISS** (`IndexFlatIP` + L2 normalization) for cosine-similarity nearest-neighbor lookup
- **Automatic summarization** — generates concise summaries of matched abstracts using a Hugging Face `transformers` pipeline
- **Keyword extraction (notebook only)** — surfaces key phrases from abstracts using **KeyBERT**; explored in the notebook but not currently exposed by the API (see step 17 below)
- **REST API** — exposes search + summarization through a **FastAPI** endpoint for easy integration
- **Caching** — embeddings and the FAISS index are saved to disk (`paper_embeddings.npy`, `paper_faiss.index`) so they're only computed once

## Tech Stack

| Component | Library |
|---|---|
| Dataset loading | `datasets` (Hugging Face) |
| Data handling | `pandas`, `numpy` |
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| Summarization | `transformers` |
| Keyword extraction | `keybert` |
| API server | `fastapi`, `uvicorn` |

## How It Works

The notebook builds the system in the following exact sequence:

1. **Install dependencies & load the dataset**
   Install `datasets`, then load `CShorten/ML-ArXiv-Papers` (split `train`) from the Hugging Face Hub — 117,592 papers with `title` and `abstract` fields.

2. **Convert to a DataFrame and select relevant columns**
   Load the dataset into a `pandas` DataFrame and keep only the `title` and `abstract` columns.

3. **Subset the data**
   Take the first 15,000 rows (`df.head(15000)`) to keep iteration fast during development.

4. **Check for missing values**
   Run `df.isnull().sum()` to confirm there are no nulls before combining fields.

5. **Build the combined text field**
   Create `paper_text = title + abstract` — this combined string is what gets embedded and searched.

6. **Save the preprocessed DataFrame**
   Persist the DataFrame to `df.pkl` via `pickle`, so it can be reloaded later (in the FastAPI app) without repeating the steps above.

7. **Clean the text**
   Strip newlines (`\n`) from `paper_text` and trim leading/trailing whitespace.

8. **Load the embedding model**
   Install `sentence-transformers`, then load `all-MiniLM-L6-v2`. *(Note: `torchcodec` is uninstalled first in the notebook to resolve a PyTorch/sentence-transformers compatibility conflict, and the Colab runtime must be restarted after that step.)*

9. **Test embeddings on a single paper**
   Encode one sample `paper_text` to confirm the model produces a 384-dimensional vector.

10. **Test embeddings on a small batch and compare similarity**
    Encode 5 sample papers and compute pairwise cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) to sanity-check that similar papers score higher.

11. **Generate embeddings for the full dataset**
    Encode all 15,000 `paper_text` entries in batches (`batch_size=32`) and cache the result to `paper_embeddings.npy`. On subsequent runs, this file is loaded instead of re-encoding.

12. **Build the FAISS index**
    Install `faiss-cpu`. Normalize the embeddings (`faiss.normalize_L2`) so inner product search behaves like cosine similarity, then build a `faiss.IndexFlatIP(384)` index and add all embeddings to it. The index is cached to `paper_faiss.index` and reloaded on later runs.

13. **Run a manual test search**
    Encode a query (`"deep learning for medical image analysis"`), normalize it, and call `index.search(query_embedding, k)` to get the top-k most similar paper indices (`I`) and their similarity scores (`D`). Manually inspect the matched titles/abstracts by index.

14. **Wrap search in a reusable function**
    Define `search_paper(query, k=5)`, which encodes a query, normalizes it, searches the index, and prints the similarity score, title, and abstract snippet for each result.

15. **Add summarization**
    Install `transformers`. Load a `text-generation` pipeline using `gpt2` (used here as a stand-in, since a dedicated `summarization` pipeline wasn't available in the original environment) and generate a short summary for a matched paper's abstract.

16. **Combine search + summarization**
    Extend the search function into `search_and_summarize(query, k=5)`, which runs the FAISS search and generates a summary for each of the top-k matched abstracts.

17. **Explore keyword extraction (notebook only)**
    Install `keybert` and load `KeyBERT` using the same embedding model. Extract representative keywords from a sample abstract (`kw_model.extract_keywords(...)`) as an additional exploratory feature. **This step is not carried over into the FastAPI service in step 19 below** — it exists only in the notebook.

18. **Re-save the DataFrame for the API**
    Save `df.pkl` again immediately before writing the API, ensuring the FastAPI app loads the exact same preprocessed data.

19. **Write the FastAPI app (`main.py`)**
    Using `%%writefile main.py`, generate a FastAPI app that, on startup, loads `df.pkl`, the `all-MiniLM-L6-v2` model, the `gpt2` text-generation pipeline, and the cached FAISS index. It exposes:
    - `GET /shard` — a health-check endpoint returning `{"message": "Hello from FastAPI!"}`
    - `POST /search_and_summarize/` — accepts `{"query": str, "k": int}`, runs the FAISS search, generates a summary for each result, and returns scores, titles, abstract snippets, and summaries. **Keyword extraction is not included in this endpoint.**

20. **Run the API server in-notebook**
    Apply `nest_asyncio`, then launch `uvicorn` in a background thread (`host="0.0.0.0"`, `port=8000`) so the server runs alongside the notebook.

21. **Test the live API**
    Use `requests` to call `GET /shard` (health check) and `POST /search_and_summarize/` with a sample query, confirming the full pipeline works end-to-end.

## Project Structure

```
.
├── AI_Search_Intelligence_System_project.ipynb   # Full development notebook (EDA → embeddings → API)
├── main.py                                       # FastAPI app (generated from the notebook)
├── df.pkl                                        # Cached, preprocessed DataFrame
├── paper_embeddings.npy                          # Cached paper embeddings
└── paper_faiss.index                             # Cached FAISS index
```

## Setup

```bash
pip install datasets pandas sentence-transformers faiss-cpu transformers keybert fastapi uvicorn nest-asyncio
```

## Usage

### Run the notebook
Open `AI_Search_Intelligence_System_project.ipynb` and run all cells in order. This will:
- download and preprocess the dataset
- generate/load cached embeddings
- build/load the FAISS index
- launch the FastAPI server in a background thread

### Query the API

```bash
curl -X POST "http://localhost:8000/search_and_summarize/" \
  -H "Content-Type: application/json" \
  -d '{"query": "deep learning for medical image analysis", "k": 5}'
```

**Response shape:**
```json
{
  "query": "deep learning for medical image analysis",
  "results": [
    {
      "score": 0.83,
      "title": "...",
      "abstract_snippet": "...",
      "summary_text": "..."
    }
  ]
}
```

## Notes / Known Limitations

- The current summarization step uses a `text-generation` pipeline (`gpt2`) as a stand-in, since a dedicated `summarization` pipeline wasn't available in the original environment — swapping in a proper summarization model (e.g. `facebook/bart-large-cnn`) would improve summary quality.
- KeyBERT keyword extraction is implemented and tested in the notebook but is **not** included in the `main.py` API response — it would need to be added to the `/search_and_summarize/` endpoint to be user-facing.
- The notebook processes a 15,000-row subset of the full ~117k-paper dataset for faster iteration; this can be increased for full coverage.
- Embeddings and the FAISS index are cached locally — delete `paper_embeddings.npy` / `paper_faiss.index` to force a rebuild after changing the dataset or model.

## Future Improvements

- Swap in a proper summarization model instead of the `gpt2` text-generation workaround
- Add pagination and filtering (e.g. by year, category) to the search endpoint
- Deploy the FastAPI service (Docker + cloud hosting) instead of running it in-notebook
- Add a lightweight frontend for interactive querying

## License

Add a license of your choice (e.g. MIT) here.
