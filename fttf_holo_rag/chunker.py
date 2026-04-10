"""
chunker.py — Document chunking pipeline for FTTF Holographic RAG.

Splits large text files and PDFs into overlapping chunks suitable for
embedding and ingestion into ChromaDB.
"""

import json
import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

CHUNK_SIZE = 25_000
CHUNK_OVERLAP = 500


@dataclass
class TextChunk:
    """A single text chunk produced by :func:`chunk_text`."""

    text: str
    start: int
    end: int
    index: int


@dataclass
class Manifest:
    """Summary of a :func:`prep_directory` run."""

    input_dir: str
    output_dir: str
    chunks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "chunks": self.chunks,
        }


def extract_pdf_text(pdf_path: str, max_chars: int | None = None) -> str:
    """Extract text from a PDF using PyMuPDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    max_chars:
        If given, truncate the result to this many characters.

    Returns
    -------
    str
        Extracted text (may be empty if the PDF contains no selectable text).

    Raises
    ------
    ImportError
        If PyMuPDF is not installed.  Install with ``pip install fttf-holo-rag[pdf]``.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF support.  "
            "Install it with:  pip install fttf-holo-rag[pdf]"
        ) from exc

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    chars = 0
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- PAGE {i + 1} ---\n{text}")
            chars += len(text)
            if max_chars is not None and chars >= max_chars:
                break
    doc.close()
    full_text = "\n\n".join(pages)
    if max_chars is not None:
        return full_text[:max_chars]
    return full_text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split *text* into overlapping :class:`TextChunk` objects.

    Boundary detection is paragraph-aware: the splitter looks for a
    ``\\n\\n`` break in the last 2 000 characters of each chunk so that
    chunks do not cut through paragraphs.

    Parameters
    ----------
    text:
        Input text to split.
    chunk_size:
        Maximum characters per chunk.
    overlap:
        Characters of overlap between consecutive chunks.

    Returns
    -------
    list[TextChunk]
        Ordered list of chunks.  The list is empty when *text* is empty.
    """
    chunks: list[TextChunk] = []
    pos = 0
    text_len = len(text)
    index = 0

    while pos < text_len:
        end = min(pos + chunk_size, text_len)

        # Paragraph-aware boundary: look for \n\n near the end of the window
        if end < text_len:
            search_start = max(end - 2_000, pos + chunk_size // 2)
            break_pos = text.rfind("\n\n", search_start, end)
            if break_pos > pos:
                end = break_pos + 2  # include the two newlines in the current chunk

        chunk_text_str = text[pos:end]
        chunks.append(TextChunk(text=chunk_text_str, start=pos, end=end, index=index))
        index += 1

        pos = end - overlap if end < text_len else text_len

    return chunks


def prep_directory(
    input_dir: str,
    output_dir: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    include_pdf: bool = False,
) -> Manifest:
    """Process all supported files in *input_dir*, write chunks to *output_dir*.

    Each source file produces one or more numbered chunk files plus a
    ``manifest.json`` in *output_dir* that records every chunk's metadata.

    Parameters
    ----------
    input_dir:
        Directory to scan recursively for source documents.
    output_dir:
        Directory where chunk files and the manifest are written.
    chunk_size:
        Maximum characters per chunk (passed to :func:`chunk_text`).
    overlap:
        Overlap between chunks (passed to :func:`chunk_text`).
    include_pdf:
        When ``True``, attempt to extract text from PDF files using
        PyMuPDF.  Raises :class:`ImportError` if PyMuPDF is not installed.

    Returns
    -------
    Manifest
        Dataclass containing input/output paths and a list of chunk records.
    """
    os.makedirs(output_dir, exist_ok=True)

    text_extensions = {
        ".py", ".js", ".ts", ".html", ".css", ".md", ".txt", ".json",
        ".yaml", ".yml", ".cpp", ".h", ".cs", ".rs", ".toml", ".go",
        ".sh", ".bat", ".xml", ".ipynb", ".sql", ".java", ".rb", ".lua",
        ".r", ".cfg", ".ini", ".jsx", ".tsx",
    }
    if include_pdf:
        text_extensions.add(".pdf")

    manifest = Manifest(input_dir=os.path.abspath(input_dir), output_dir=os.path.abspath(output_dir))
    chunk_counter = 0

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
                         "dist", "build", "target", ".tox", ".mypy_cache"}
        ]
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in text_extensions:
                continue
            filepath = os.path.join(root, filename)
            try:
                if ext == ".pdf":
                    text = extract_pdf_text(filepath)
                else:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                if not text.strip():
                    continue
            except Exception as exc:
                log.warning("prep_directory: could not read %s: %s", filepath, exc)
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                safe_name = (
                    os.path.relpath(filepath, input_dir)
                    .replace(os.sep, "_")
                    .replace("/", "_")
                )
                out_filename = f"{chunk_counter:06d}_{safe_name}.chunk.txt"
                out_path = os.path.join(output_dir, out_filename)

                header = (
                    f"# source: {filepath}\n"
                    f"# chunk: {chunk.index}  start: {chunk.start}  end: {chunk.end}\n"
                    f"# ---\n"
                )
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(header + chunk.text)

                manifest.chunks.append({
                    "chunk_file": out_filename,
                    "source": filepath,
                    "chunk_index": chunk.index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "chars": len(chunk.text),
                })
                chunk_counter += 1
                log.debug("prep_directory: wrote %s", out_filename)

            log.info("prep_directory: %s → %d chunk(s)", filename, len(chunks))

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)

    log.info(
        "prep_directory: done — %d chunk(s) from %s → %s",
        len(manifest.chunks),
        input_dir,
        output_dir,
    )
    return manifest
