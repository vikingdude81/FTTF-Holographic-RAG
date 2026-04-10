"""
ingest.py — Document ingestion pipeline for FTTF Holographic RAG.

Crawls directories, extracts text, and upserts into ChromaDB using
Ollama embeddings.  Optionally invalidates the holographic cache after
a successful run so it is rebuilt on the next HoloRAG start.
"""

import logging
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from fttf_holo_rag.utils import extract_project_name

log = logging.getLogger(__name__)

# Directories that are never worth crawling
_IGNORE_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    "dist", "build", "target", ".local", "site-packages", ".tox",
    ".mypy_cache", ".eggs", ".conda", "Lib", "lib", "Scripts", "Include",
    ".cache", "egg-info", "$RECYCLE.BIN", "System Volume Information",
})

# File extensions considered for ingestion
_VALID_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".html", ".css", ".md", ".txt", ".json",
    ".yaml", ".cpp", ".h", ".cs", ".rs", ".toml", ".pdf",
    ".go", ".sh", ".bat", ".yml", ".xml", ".ipynb", ".sql",
    ".java", ".rb", ".lua", ".r", ".cfg", ".ini", ".jsx", ".tsx",
})


@dataclass
class IngestResult:
    """Counts returned by :func:`ingest_directories`."""

    added: int = 0
    skipped: int = 0
    failed: int = 0


def _extract_pdf_text(filepath: str, max_chars: int = 2_000) -> str | None:
    """Extract text from a PDF using PyMuPDF.

    Returns the text (truncated to *max_chars*) or ``None`` when the PDF
    contains no selectable text or PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error(
            "PyMuPDF is required for PDF ingestion.  "
            "Install it with:  pip install fttf-holo-rag[pdf]"
        )
        return None

    try:
        doc = fitz.open(filepath)
        text_parts: list[str] = []
        chars = 0
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)
            chars += len(page_text)
            if chars >= max_chars:
                break
        doc.close()
        full_text = "\n".join(text_parts)
        return full_text[:max_chars] if full_text.strip() else None
    except Exception as exc:
        log.warning("PDF extraction failed for %s: %s", filepath, exc)
        return None


def ingest_directories(
    directories: list[str],
    *,
    ollama_url: str = "http://localhost:11434/api/embeddings",
    embed_model: str = "nomic-embed-text",
    chroma_path: str = "./chroma_db_storage",
    collection_name: str = "my_codebase",
    batch_size: int = 50,
    max_chars: int = 2_000,
    include_pdf: bool = False,
    holo_cache_path: str | None = None,
) -> IngestResult:
    """Crawl *directories*, embed new files, and upsert them into ChromaDB.

    Files that are already present in the collection (identified by their
    absolute filepath) are skipped.  After a successful upsert the
    holographic cache pickle is deleted so HoloRAG rebuilds it on the next
    start.

    Parameters
    ----------
    directories:
        List of filesystem paths to crawl recursively.
    ollama_url:
        URL of the Ollama embedding API endpoint.
    embed_model:
        Name of the Ollama model to use for embeddings.
    chroma_path:
        Path to the ChromaDB persistent storage directory.
    collection_name:
        Name of the ChromaDB collection to upsert into.
    batch_size:
        Number of documents per ChromaDB upsert call.
    max_chars:
        Maximum characters to read from each file (truncated if longer).
    include_pdf:
        When ``True``, extract text from ``.pdf`` files using PyMuPDF.
        Requires ``pip install fttf-holo-rag[pdf]``.
    holo_cache_path:
        Optional path to the holographic cache pickle.  When provided and
        a file exists at that path, it is deleted after ingestion so the
        cache is rebuilt on the next HoloRAG start.

    Returns
    -------
    IngestResult
        Dataclass with ``added``, ``skipped``, and ``failed`` counts.
    """
    log.info("Connecting to ChromaDB at %s …", chroma_path)
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=ollama_url,
        model_name=embed_model,
    )
    db_client = chromadb.PersistentClient(path=chroma_path)
    collection = db_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef,
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    log.info("Collection '%s' currently contains %d document(s).", collection_name, len(existing_ids))

    # ------------------------------------------------------------------ #
    #  Crawl                                                               #
    # ------------------------------------------------------------------ #
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    already_present = 0

    valid_exts = _VALID_EXTENSIONS if include_pdf else (_VALID_EXTENSIONS - {".pdf"})

    for folder_path in directories:
        if not os.path.exists(folder_path):
            log.warning("Directory not found, skipping: %s", folder_path)
            continue

        log.info("Crawling %s …", folder_path)

        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]

            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in valid_exts:
                    continue

                filepath = os.path.abspath(os.path.join(root, filename))

                if filepath in existing_ids:
                    already_present += 1
                    continue

                try:
                    if ext == ".pdf":
                        content = _extract_pdf_text(filepath, max_chars=max_chars)
                        if not content:
                            continue
                    else:
                        with open(filepath, "r", encoding="utf-8") as fh:
                            content = fh.read()
                        if not content.strip():
                            continue
                        content = content[:max_chars]

                    documents.append(content)
                    metadatas.append({
                        "filename": filename,
                        "filepath": filepath,
                        "project_name": extract_project_name(filepath),
                    })
                    ids.append(filepath)
                    log.debug("Queued: %s", filename)
                except Exception as exc:
                    log.warning("Could not read %s: %s", filename, exc)

    log.info(
        "Crawl complete — %d new file(s) queued, %d already present.",
        len(documents),
        already_present,
    )

    result = IngestResult(skipped=already_present)

    # ------------------------------------------------------------------ #
    #  Batch upsert                                                        #
    # ------------------------------------------------------------------ #
    if not documents:
        log.info("Nothing new to ingest.")
        return result

    log.info("Upserting %d document(s) in batches of %d …", len(documents), batch_size)

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i: i + batch_size]
        batch_metas = metadatas[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]

        try:
            collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
            result.added += len(batch_docs)
            log.info("Upserted [%d/%d]", result.added, len(documents))
        except Exception as batch_exc:
            log.warning("Batch upsert failed, falling back to one-by-one: %s", batch_exc)
            for doc, meta, doc_id in zip(batch_docs, batch_metas, batch_ids):
                try:
                    collection.upsert(documents=[doc], metadatas=[meta], ids=[doc_id])
                    result.added += 1
                except Exception as single_exc:
                    result.failed += 1
                    log.warning("Skipped %s: %s", meta["filename"], single_exc)
            log.info("Fallback done [%d/%d]", result.added, len(documents))

    log.info(
        "Ingestion complete — added: %d, skipped (duplicate): %d, failed: %d",
        result.added,
        result.skipped,
        result.failed,
    )

    # Invalidate holographic cache so it rebuilds on next HoloRAG start
    if holo_cache_path and os.path.exists(holo_cache_path):
        try:
            os.remove(holo_cache_path)
            log.info("Holographic cache cleared — will rebuild on next start: %s", holo_cache_path)
        except OSError as exc:
            log.warning("Could not remove holographic cache: %s", exc)

    return result
