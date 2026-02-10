"""
RAG Engine — Document ingestion and semantic search using ChromaDB.
Supports: PDF, TXT, CSV, JSON, Markdown files.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("chatbot-api.rag")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".jsonl"}

# Chunk configuration
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks


def _read_file(filepath: Path) -> str:
    """Read file contents based on extension."""
    ext = filepath.suffix.lower()

    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(filepath))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except ImportError:
            logger.warning(f"PyMuPDF not installed — skipping PDF: {filepath.name}")
            return ""

    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
            return json.dumps(data, ensure_ascii=False, indent=2)

    elif ext == ".jsonl":
        lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return "\n".join(lines)

    elif ext == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    else:  # .txt, .md, etc.
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                for sep in [". ", ".\n", "! ", "? ", "\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def _file_hash(filepath: Path) -> str:
    """Generate a hash for file content to detect changes."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


class RAGEngine:
    """Retrieval-Augmented Generation engine using ChromaDB."""

    def __init__(self, data_dir: str = "/workspace/data", db_dir: str = "/workspace/chromadb"):
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="chatbot_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"ChromaDB initialized at {self.db_dir} — {self.document_count()} chunks")

    def document_count(self) -> int:
        """Return total number of indexed chunks."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def ingest(self, directory: Optional[str] = None) -> int:
        """Ingest all supported documents from the directory."""
        target_dir = Path(directory) if directory else self.data_dir

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        files_processed = 0
        total_chunks = 0

        for filepath in sorted(target_dir.rglob("*")):
            if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if filepath.name.startswith("."):
                continue

            try:
                file_id = _file_hash(filepath)
                source_name = str(filepath.relative_to(target_dir))

                # Check if already ingested (same hash)
                existing = self.collection.get(
                    where={"file_hash": file_id},
                    limit=1,
                )
                if existing and existing["ids"]:
                    logger.debug(f"Skipping (unchanged): {source_name}")
                    continue

                # Remove old chunks from same file
                try:
                    old = self.collection.get(where={"source": source_name})
                    if old and old["ids"]:
                        self.collection.delete(ids=old["ids"])
                except Exception:
                    pass

                # Read and chunk
                text = _read_file(filepath)
                if not text.strip():
                    continue

                chunks = _chunk_text(text)
                if not chunks:
                    continue

                # Add to ChromaDB
                ids = [f"{source_name}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": source_name, "chunk_index": i, "file_hash": file_id}
                    for i in range(len(chunks))
                ]

                self.collection.add(
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas,
                )

                files_processed += 1
                total_chunks += len(chunks)
                logger.info(f"Ingested: {source_name} → {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to ingest {filepath.name}: {e}")

        logger.info(f"Ingestion complete — {files_processed} files, {total_chunks} new chunks, {self.document_count()} total")
        return files_processed

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant document chunks."""
        if self.document_count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.document_count()),
            )

            chunks = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    chunks.append({
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "relevance": round(1 - distance, 4),
                    })

            return chunks

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []

    def clear(self):
        """Clear all indexed documents."""
        self.client.delete_collection("chatbot_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="chatbot_knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Knowledge base cleared")
