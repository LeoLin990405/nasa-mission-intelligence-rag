#!/usr/bin/env python3
"""RAGAS evaluation utilities for the NASA Mission Intelligence project."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    SingleTurnSample = None
    LangchainEmbeddingsWrapper = None
    LangchainLLMWrapper = None
    Faithfulness = None
    ResponseRelevancy = None
    ChatOpenAI = None
    OpenAIEmbeddings = None


def _build_ragas_models():
    """Create RAGAS-compatible LLM and embedding wrappers."""
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY")

    llm_kwargs = {
        "model": os.getenv("RAGAS_LLM_MODEL", "gpt-3.5-turbo"),
        "temperature": 0,
    }
    embedding_kwargs = {
        "model": os.getenv(
            "RAGAS_EMBEDDING_MODEL",
            os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        ),
    }

    if base_url:
        llm_kwargs["base_url"] = base_url
        embedding_kwargs["base_url"] = base_url
    if api_key:
        llm_kwargs["api_key"] = api_key
        embedding_kwargs["api_key"] = api_key

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(**embedding_kwargs))
    return evaluator_llm, evaluator_embeddings


def _normalize_evaluation_inputs(
    question: object,
    answer: object,
    contexts: object,
) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[str]]:
    """Validate and normalize evaluator inputs without raising on bad data."""
    if not isinstance(question, str):
        return None, None, None, "Question must be a string"
    if not question.strip():
        return None, None, None, "Question must not be empty"

    if not isinstance(answer, str):
        return None, None, None, "Answer must be a string"
    if not answer.strip():
        return None, None, None, "Answer must not be empty"

    if isinstance(contexts, str):
        raw_contexts: Sequence[object] = [contexts]
    elif isinstance(contexts, Sequence):
        raw_contexts = contexts
    else:
        return None, None, None, "Contexts must be a list of strings"

    normalized_contexts: List[str] = []
    for index, context in enumerate(raw_contexts):
        if not isinstance(context, str):
            return None, None, None, f"Context at index {index} must be a string"
        cleaned = context.strip()
        if cleaned:
            normalized_contexts.append(cleaned)

    if not normalized_contexts:
        return None, None, None, "No retrieved contexts available for evaluation"

    return question.strip(), answer.strip(), normalized_contexts, None


def _tokenize_for_overlap(text: str) -> List[str]:
    """Tokenize text for lightweight lexical evaluation metrics."""
    import re

    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _rouge_l_score(reference: str, candidate: str) -> float:
    """Compute a compact ROUGE-L F1 score for extra, non-RAGAS coverage."""
    reference_tokens = _tokenize_for_overlap(reference)
    candidate_tokens = _tokenize_for_overlap(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    previous = [0] * (len(candidate_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for column, candidate_token in enumerate(candidate_tokens, start=1):
            if reference_token == candidate_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _context_precision_score(contexts: List[str], answer: str) -> float:
    """Estimate how much of the answer is lexically supported by retrieved context."""
    context_tokens = set(_tokenize_for_overlap(" ".join(contexts)))
    answer_tokens = _tokenize_for_overlap(answer)
    if not context_tokens or not answer_tokens:
        return 0.0
    supported_tokens = sum(1 for token in answer_tokens if token in context_tokens)
    return supported_tokens / len(answer_tokens)


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, object]:
    """Evaluate one (question, answer, retrieved contexts) triple.

    The primary scores come from RAGAS response relevancy and faithfulness. The
    evaluator also reports lightweight ROUGE-L and lexical context precision so
    the project has an additional documented metric path even when RAGAS metric
    availability changes across package versions.
    """
    normalized_question, normalized_answer, normalized_contexts, error = _normalize_evaluation_inputs(
        question,
        answer,
        contexts,
    )
    if error:
        return {"error": error}

    assert normalized_question is not None
    assert normalized_answer is not None
    assert normalized_contexts is not None

    scores: Dict[str, object] = {
        "rouge_l": _rouge_l_score(" ".join(normalized_contexts), normalized_answer),
        "context_precision": _context_precision_score(normalized_contexts, normalized_answer),
    }

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS or langchain-openai is not available", **scores}

    evaluator_llm, evaluator_embeddings = _build_ragas_models()

    sample = SingleTurnSample(
        user_input=normalized_question,
        response=normalized_answer,
        retrieved_contexts=normalized_contexts,
    )

    metrics = {
        "response_relevancy": ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "faithfulness": Faithfulness(llm=evaluator_llm),
    }

    for metric_name, metric in metrics.items():
        try:
            if hasattr(metric, "single_turn_score"):
                score = metric.single_turn_score(sample)
            elif hasattr(metric, "single_turn_ascore"):
                score = metric.single_turn_ascore(sample)
            else:
                raise AttributeError(f"{metric_name} has no single-turn scoring method")

            if asyncio.iscoroutine(score):
                score = asyncio.run(score)
            scores[metric_name] = float(score)
        except Exception as exc:
            scores[f"{metric_name}_error"] = str(exc)

    if not any(isinstance(value, float) for value in scores.values()):
        return {"error": "RAGAS metrics did not produce numeric scores", **scores}

    return scores


def load_evaluation_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load evaluation questions from JSON/JSONL or the Udacity text format."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON evaluation dataset must contain a list of records")
        return data

    if path.suffix.lower() == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    records: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        if line.lower().startswith("question:"):
            current["question"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("expected response:"):
            current["expected_response"] = line.split(":", 1)[1].strip()
    if current:
        records.append(current)

    return records


def batch_evaluate_test_set(
    dataset_path: str,
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
    model: str = "gpt-3.5-turbo",
    top_k: int = 3,
    mission_filter: Optional[str] = None,
) -> Dict[str, object]:
    """Run retrieval, generation, and RAGAS scoring for each dataset question."""
    import llm_client
    import rag_client

    if not openai_key:
        raise ValueError("An OpenAI-compatible API key is required for batch evaluation")

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    records = load_evaluation_dataset(dataset_path)
    if not records:
        raise ValueError("Evaluation dataset contains no questions")

    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Failed to initialize RAG system: {error}")

    results = []
    aggregate_values: Dict[str, List[float]] = {}

    for index, record in enumerate(records, start=1):
        question = record.get("question", "").strip()
        if not question:
            results.append({"index": index, "error": "Missing question"})
            continue

        retrieved = rag_client.retrieve_documents(
            collection,
            question,
            n_results=top_k,
            mission_filter=mission_filter,
        )
        contexts = retrieved.get("documents", [[]])[0] if retrieved else []
        metadatas = retrieved.get("metadatas", [[]])[0] if retrieved else []
        distances = retrieved.get("distances", [[]])[0] if retrieved else []
        context = rag_client.format_context(contexts, metadatas, distances)

        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context,
            conversation_history=[],
            model=model,
        )
        scores = evaluate_response_quality(question, answer, contexts)
        for metric, value in scores.items():
            if isinstance(value, float):
                aggregate_values.setdefault(metric, []).append(value)

        results.append(
            {
                "index": index,
                "question": question,
                "expected_response": record.get("expected_response", ""),
                "answer": answer,
                "sources": [
                    {
                        "source": metadata.get("source"),
                        "mission": metadata.get("mission"),
                        "chunk_index": metadata.get("chunk_index"),
                        "distance": distances[pos] if pos < len(distances) else None,
                    }
                    for pos, metadata in enumerate(metadatas)
                ],
                "scores": scores,
            }
        )

    aggregate = {
        metric: mean(values)
        for metric, values in aggregate_values.items()
        if values
    }
    return {"results": results, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate the NASA RAG system")
    parser.add_argument("--dataset", default="evaluation_dataset.txt")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai")
    parser.add_argument("--collection-name", default="nasa_space_missions_text")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL", "gpt-3.5-turbo"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mission-filter", default=None)
    parser.add_argument("--output", default="batch_evaluation_results.json")
    args = parser.parse_args()

    report = batch_evaluate_test_set(
        dataset_path=args.dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        openai_key=args.openai_key,
        model=args.model,
        top_k=args.top_k,
        mission_filter=args.mission_filter,
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
