from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

from app.content_retrieval import clean_text, extract_text, warm_reranker

StatusCallback = Callable[[str, int], None]


def file_checksum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_loader(path: str) -> str:
    """Extract source text without performing retrieval or LLM work."""
    return extract_text(path)


def preprocessing(text: str) -> str:
    """Normalize extracted text before it is split into vector-store nodes."""
    return clean_text(text)


def chunking(text: str, metadata: dict[str, str]) -> list:
    """Create deterministic overlapping chunks for one document."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.get_nodes_from_documents([Document(text=text, metadata=metadata)])


def embedding_and_vector_store(nodes: list, persist_dir: str) -> VectorStoreIndex:
    """Embed all chunks, create the vector index, and persist it once."""
    index = VectorStoreIndex(nodes, show_progress=False)
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def load_vector_store(persist_dir: str) -> VectorStoreIndex:
    context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded = load_index_from_storage(context)
    if not isinstance(loaded, VectorStoreIndex):
        raise TypeError("Persisted index is not a VectorStoreIndex")
    return loaded


class DocumentIndexingService:
    """Background, hash-deduplicated upload/indexing pipeline."""

    def __init__(self, base_dir: str, max_workers: int = 2) -> None:
        self.base_dir = Path(base_dir)
        self.index_root = self.base_dir / "document_indexes"
        self.database_path = self.base_dir / "document_metadata.db"
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="document-indexer"
        )
        self._prepared: dict[str, tuple[list[VectorStoreIndex], list[str]]] = {}
        self._prepared_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._initialize_database()
        self._restore_prepared_jobs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    checksum TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    index_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS indexing_jobs (
                    id TEXT PRIMARY KEY,
                    checksums TEXT NOT NULL,
                    filenames TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)

    def submit(self, paths: list[str], original_names: list[str] | None = None) -> str:
        documents = []
        seen_checksums: set[str] = set()
        for position, path in enumerate(paths):
            checksum = file_checksum(path)
            if checksum in seen_checksums:
                continue
            seen_checksums.add(checksum)
            original_name = (
                original_names[position]
                if original_names and position < len(original_names)
                else os.path.basename(path)
            )
            documents.append(
                {
                    "checksum": checksum,
                    "path": path,
                    "name": original_name,
                    "size": os.path.getsize(path),
                    "index_path": str(self.index_root / checksum),
                }
            )

        job_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO indexing_jobs
                (id, checksums, filenames, status, progress, error, created_at, updated_at)
                VALUES (?, ?, ?, 'Uploading', 0, NULL, ?, ?)
                """,
                (
                    job_id,
                    json.dumps([item["checksum"] for item in documents]),
                    json.dumps([item["name"] for item in documents]),
                    now,
                    now,
                ),
            )
        for item in documents:
            if self._existing_index(item["checksum"]) is None:
                self._set_document_status(item, "Uploading", 0)
        self._executor.submit(self._run_job, job_id, documents)
        return job_id

    def _set_job_status(
        self, job_id: str, status: str, progress: int, error: str | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE indexing_jobs
                SET status = ?, progress = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, progress, error, time.time(), job_id),
            )

    def _set_document_status(
        self,
        item: dict,
        status: str,
        progress: int,
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                (checksum, original_name, stored_path, index_path, size_bytes,
                 status, progress, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(checksum) DO UPDATE SET
                    original_name = excluded.original_name,
                    stored_path = excluded.stored_path,
                    index_path = excluded.index_path,
                    size_bytes = excluded.size_bytes,
                    status = excluded.status,
                    progress = excluded.progress,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    item["checksum"],
                    item["name"],
                    item["path"],
                    item["index_path"],
                    item["size"],
                    status,
                    progress,
                    error,
                    now,
                    now,
                ),
            )

    def _existing_index(self, checksum: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT index_path, status FROM documents WHERE checksum = ?",
                (checksum,),
            ).fetchone()
        if row and row["status"] == "Indexed" and Path(row["index_path"]).exists():
            return str(row["index_path"])
        return None

    def _index_document(
        self, item: dict, progress_callback: StatusCallback
    ) -> VectorStoreIndex:
        existing_path = self._existing_index(item["checksum"])
        if existing_path:
            progress_callback("Reusing existing index", 90)
            index = load_vector_store(existing_path)
            item["index_path"] = existing_path
            self._set_document_status(item, "Indexed", 100)
            return index

        self._set_document_status(item, "Processing", 10)
        progress_callback("Extracting text", 20)
        text = document_loader(item["path"])
        progress_callback("Preprocessing text", 35)
        text = preprocessing(text)
        if not text:
            raise ValueError(f"No readable text found in {item['name']}")

        metadata = {
            "source": item["name"],
            "document_hash": item["checksum"],
        }
        progress_callback("Chunking document", 50)
        nodes = chunking(text, metadata)
        if not nodes:
            raise ValueError(f"No chunks generated for {item['name']}")

        progress_callback("Generating embeddings", 65)
        index = embedding_and_vector_store(nodes, item["index_path"])
        progress_callback("Storing vector index", 85)
        self._set_document_status(item, "Indexed", 100)
        return index

    def _run_job(self, job_id: str, documents: list[dict]) -> None:
        try:
            self._set_job_status(job_id, "Processing", 5)
            indexes: list[VectorStoreIndex] = []
            checksums: list[str] = []
            total = max(len(documents), 1)

            # Serializing mutations avoids concurrent writes to the same index folder.
            with self._index_lock:
                for position, item in enumerate(documents):
                    base = int(position / total * 90)

                    def update_stage(
                        _label: str,
                        stage_progress: int,
                        current_base: int = base,
                        current_item: dict = item,
                    ) -> None:
                        overall = min(
                            95, current_base + int(stage_progress / total * 0.9)
                        )
                        self._set_document_status(
                            current_item, "Processing", stage_progress
                        )
                        self._set_job_status(job_id, "Processing", overall)

                    try:
                        index = self._index_document(item, update_stage)
                    except Exception as exc:
                        self._set_document_status(item, "Failed", 0, str(exc))
                        raise
                    indexes.append(index)
                    checksums.append(item["checksum"])

                warm_reranker()

            with self._prepared_lock:
                self._prepared[job_id] = (indexes, checksums)
            self._set_job_status(job_id, "Indexed", 100)
        except Exception as exc:  # noqa: BLE001 — background task boundary
            self._set_job_status(job_id, "Failed", 0, str(exc))

    def status(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM indexing_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row:
                checksums = json.loads(row["checksums"])
                placeholders = ",".join("?" for _ in checksums)
                document_rows = connection.execute(
                    f"""SELECT checksum, original_name, size_bytes, status, progress, error
                    FROM documents WHERE checksum IN ({placeholders})""",
                    checksums,
                ).fetchall()
        if not row:
            return None
        return {
            "document_id": row["id"],
            "status": row["status"],
            "progress": row["progress"],
            "error": row["error"],
            "filenames": json.loads(row["filenames"]),
            "documents": [dict(document) for document in document_rows],
        }

    def prepared_indexes(
        self, job_id: str
    ) -> tuple[list[VectorStoreIndex], list[str]] | None:
        with self._prepared_lock:
            return self._prepared.get(job_id)

    def _restore_prepared_jobs(self) -> None:
        with self._connect() as connection:
            jobs = connection.execute(
                "SELECT id, checksums FROM indexing_jobs WHERE status = 'Indexed'"
            ).fetchall()
            documents = {
                row["checksum"]: row["index_path"]
                for row in connection.execute(
                    "SELECT checksum, index_path FROM documents WHERE status = 'Indexed'"
                ).fetchall()
            }
        for job in jobs:
            checksums = list(dict.fromkeys(json.loads(job["checksums"])))
            try:
                indexes = [load_vector_store(documents[value]) for value in checksums]
            except (KeyError, OSError, ValueError, TypeError):
                continue
            self._prepared[job["id"]] = (indexes, checksums)
