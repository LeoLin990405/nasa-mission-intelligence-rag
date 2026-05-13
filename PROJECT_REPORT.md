# Project Report: NASA Mission Intelligence RAG

## Objective

The project builds a Retrieval-Augmented Generation chat system that answers
questions about Apollo 11, Apollo 13, and Challenger using retrieved NASA source
text instead of relying only on model memory.

## Architecture

1. `embedding_pipeline.py` scans `data_text/`, chunks mission documents, embeds
   each chunk with an OpenAI-compatible embedding model, and stores the results
   in a persistent ChromaDB collection.
2. `rag_client.py` connects to ChromaDB, embeds the user question through the
   collection embedding function, retrieves top-k relevant chunks, optionally
   filters by mission metadata, and formats the retrieved snippets with source
   attribution.
3. `llm_client.py` sends the user question, retrieved context, and recent chat
   history to an OpenAI-compatible chat model. The system prompt instructs the
   assistant to act as a NASA mission expert, cite retrieved sources, and avoid
   unsupported claims.
4. `ragas_evaluator.py` evaluates answer quality with RAGAS response relevancy
   and faithfulness. It supports both one-off real-time evaluation and batch
   evaluation using `evaluation_dataset.txt`.
5. `chat.py` provides the Streamlit interface, including model selection,
   retrieval top-k, mission filtering, answer generation, and evaluation display.

## Evaluation Dataset

`evaluation_dataset.txt` contains questions spanning overview, emergency,
technical, disaster analysis, crew, timeline, and insufficient-context cases.
Each question includes an expected response description used as a human-readable
reference for testing.

## Rubric Coverage

- Configurable chunk size and overlap are exposed through CLI flags.
- Chunk metadata includes source, file path, mission, document category, and
  chunk position.
- Existing ChromaDB documents can be skipped, updated, or replaced.
- Retrieval supports configurable top-k and mission filtering.
- LLM context includes deduplicated source-attributed snippets sorted by
  retrieval distance.
- Conversation history is preserved across turns in the chat app.
- RAGAS returns structured metric dictionaries and handles malformed inputs with
  clear errors.
- Batch evaluation writes detailed JSON output and aggregate metric means.

## Testing Procedure

1. Compile the Python modules:

   ```bash
   python3 -m py_compile chat.py embedding_pipeline.py llm_client.py rag_client.py ragas_evaluator.py
   ```

2. Build or inspect the vector store:

   ```bash
   python embedding_pipeline.py --data-path ./data_text
   python embedding_pipeline.py --stats-only
   ```

   `OPENAI_API_KEY` can be supplied through the shell environment, a local
   untracked `.env` file, or Streamlit secrets for the UI.

3. Launch the app:

   ```bash
   streamlit run chat.py
   ```

4. Ask several questions from `evaluation_dataset.txt` and verify that answers
   cite retrieved sources and that RAGAS metrics are displayed when enabled.

5. Run batch evaluation:

   ```bash
   python ragas_evaluator.py --dataset evaluation_dataset.txt --openai-key "$OPENAI_API_KEY"
   ```
