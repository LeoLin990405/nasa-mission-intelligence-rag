# NASA Mission Intelligence RAG Project

This project implements an end-to-end Retrieval-Augmented Generation (RAG)
chat system for NASA mission documents from Apollo 11, Apollo 13, and
Challenger.

The application processes mission text files, stores embeddings in ChromaDB,
retrieves relevant context for a user question, generates a grounded answer with
an OpenAI-compatible chat model, and evaluates each answer with RAGAS metrics.

## Files

```text
chat.py                 Streamlit chat interface with retrieval and evaluation
embedding_pipeline.py   Text chunking, OpenAI embeddings, and ChromaDB indexing
llm_client.py           OpenAI-compatible chat-completion client
rag_client.py           ChromaDB discovery, retrieval, filtering, context formatting
ragas_evaluator.py      Single-turn and batch RAGAS evaluation utilities
evaluation_dataset.txt  Mission-relevant evaluation questions
requirements.txt        Python dependencies
data_text/              NASA mission source text
```

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set an API key for OpenAI or an OpenAI-compatible provider:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Or create a local `.env` file:

```bash
cp .env.example .env
# edit .env and paste your provider key there
```

For compatible providers, also set a base URL:

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

Do not commit API keys or provider credentials. `.env` is ignored by git, while
`.env.example` documents the required variables without exposing secrets.

## Build The Vector Store

Run the embedding pipeline against the included NASA text files:

```bash
python embedding_pipeline.py \
  --data-path ./data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --update-mode skip
```

Useful variants:

```bash
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --stats-only
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --update-mode update
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --update-mode replace
```

If `OPENAI_API_KEY` is already set in the environment or `.env`, the
`--openai-key` flag is optional.

The pipeline stores per-chunk metadata including source, file path, mission,
data type, document category, chunk index, chunk start/end, and chunk count.

## Run The Chat App

```bash
streamlit run chat.py
```

In the sidebar:

- Select the ChromaDB collection.
- Enter the API key and optional base URL.
- Choose the chat model.
- Configure top-k retrieval and mission filtering.
- Enable RAGAS evaluation when the required dependencies and model endpoint are available.

The assistant is prompted as a NASA mission expert and instructed to cite
retrieved source headers and state uncertainty when the context is insufficient.

## Batch Evaluation

The batch evaluator loads `evaluation_dataset.txt`, runs retrieval and answer
generation for each question, computes RAGAS `response_relevancy` and
`faithfulness`, and writes per-question plus aggregate results.

```bash
python ragas_evaluator.py \
  --dataset evaluation_dataset.txt \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --openai-key "$OPENAI_API_KEY" \
  --model gpt-3.5-turbo \
  --top-k 3 \
  --output batch_evaluation_results.json
```

## Rubric Mapping

Embedding and data pipeline:

- `--chunk-size` and `--chunk-overlap` are runtime CLI flags.
- `chunk_text` keeps every chunk within `chunk_size` characters and applies overlap.
- `get_embedding` calls the configured OpenAI embedding model.
- `add_documents_to_collection` supports `skip`, `update`, and `replace`.
- ChromaDB is persisted through `--chroma-dir` and `--collection-name`.
- `--stats-only` prints collection size and aggregate metadata counts.

Retrieval and LLM integration:

- `rag_client.initialize_rag_system` connects to ChromaDB.
- `retrieve_documents` performs semantic search with configurable top-k and mission filter.
- `format_context` sorts by distance, removes duplicate snippets, and adds source attribution.
- `llm_client.generate_response` maintains recent conversation history and passes context to the model.

Real-time evaluation:

- `ragas_evaluator.evaluate_response_quality` evaluates one question/context/answer triple.
- RAGAS metrics include response relevancy and faithfulness.
- Additional documented metrics include local ROUGE-L and lexical context precision.
- Empty or malformed inputs return clear error dictionaries instead of crashing.
- `ragas_evaluator.py` also provides batch evaluation from `evaluation_dataset.txt`.

## Quick Checks

```bash
python3 -m py_compile chat.py embedding_pipeline.py llm_client.py rag_client.py ragas_evaluator.py
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --stats-only
```

After building the vector store, launch Streamlit and test several questions from
`evaluation_dataset.txt`.
