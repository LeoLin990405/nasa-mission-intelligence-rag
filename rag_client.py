import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    ignored_parts = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

    chroma_dirs = []
    for path in current_dir.rglob("*"):
        if not path.is_dir():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        name = path.name.lower()
        if "chroma" in name or name in {"db", "vector_db"}:
            chroma_dirs.append(path)

    for chroma_dir in sorted(set(chroma_dirs)):
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for collection_ref in collections:
                collection_name = getattr(collection_ref, "name", str(collection_ref))
                collection = client.get_collection(collection_name)
                key = f"{chroma_dir}:{collection_name}"
                try:
                    document_count = collection.count()
                except Exception:
                    document_count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": (
                        f"{collection_name} ({document_count} docs, "
                        f"{chroma_dir})"
                    ),
                    "document_count": str(document_count),
                }
        except Exception as exc:
            key = f"{chroma_dir}:unavailable"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir} unavailable: {str(exc)[:80]}",
                "document_count": "0",
                "error": str(exc),
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[object, bool, Optional[str]]:
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        embedding_function = None
        openai_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if openai_key:
            embedding_kwargs = {
                "api_key": openai_key,
                "model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            }
            base_url = os.getenv("CHROMA_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            if base_url:
                embedding_kwargs["api_base"] = base_url
            try:
                embedding_function = OpenAIEmbeddingFunction(**embedding_kwargs)
            except TypeError:
                embedding_kwargs.pop("api_base", None)
                embedding_function = OpenAIEmbeddingFunction(**embedding_kwargs)

        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_function,
        )
        return collection, True, None
    except Exception as exc:
        return None, False, str(exc)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    where = None

    if mission_filter and mission_filter.lower() not in {"all", "any", "none"}:
        where = {"mission": mission_filter}

    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

def format_context(
    documents: List[str],
    metadatas: List[Dict],
    distances: Optional[List[float]] = None,
) -> str:
    """Format retrieved documents into a deduplicated, source-attributed context."""
    if not documents:
        return ""

    context_parts = ["Retrieved NASA mission context:"]
    rows = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances if distances is not None else [None] * len(documents),
    ):
        metadata = metadata or {}
        clean_document = " ".join(str(document).split())
        if not clean_document:
            continue
        rows.append((distance, clean_document, metadata))

    rows.sort(key=lambda row: float("inf") if row[0] is None else row[0])

    seen = set()
    source_index = 1
    for distance, clean_document, metadata in rows:
        dedupe_key = (
            metadata.get("source", ""),
            metadata.get("chunk_index", ""),
            clean_document[:200],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        mission = metadata.get("mission", "unknown").replace("_", " ").title()
        source = metadata.get("source", "unknown source")
        category = metadata.get("document_category", "general").replace("_", " ").title()
        chunk_index = metadata.get("chunk_index", "n/a")
        score_label = ""
        if distance is not None:
            score_label = f" | Distance: {float(distance):.4f}"

        context_parts.append(
            f"\n[Source {source_index}] Mission: {mission} | "
            f"Category: {category} | Source: {source} | Chunk: {chunk_index}"
            f"{score_label}"
        )

        if len(clean_document) > 1400:
            clean_document = f"{clean_document[:1400]}..."
        context_parts.append(clean_document)
        source_index += 1

    return "\n".join(context_parts)
