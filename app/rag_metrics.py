"""
app/rag_metrics.py
──────────────────
Lightweight per-query metrics capture for the RAG Evaluation Dashboard.

Every time a query runs through the pipeline, the caller builds a
RagMetrics instance, fills in each field as the pipeline progresses,
then calls MetricsStore.save(metrics).

The MetricsStore persists records to a SQLite database (rag_metrics.db
inside the project root) so they survive server restarts and can be
queried / paginated efficiently.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Groq pricing snapshot (August 2026). Update if pricing changes.
# Prices are in USD per 1 000 tokens.
# ---------------------------------------------------------------------------
_PRICE_PER_1K: dict[str, dict[str, float]] = {
    "openai/gpt-oss-120b": {"prompt": 0.00015, "completion": 0.00060},
    "llama-3.3-70b-versatile": {"prompt": 0.00059, "completion": 0.00079},
    "llama-3.1-8b-instant": {"prompt": 0.00005, "completion": 0.00008},
    "mixtral-8x7b-32768": {"prompt": 0.00024, "completion": 0.00024},
}
_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_DEFAULT_PRICE = _PRICE_PER_1K["openai/gpt-oss-120b"]


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = _DEFAULT_MODEL,
) -> float:
    """Return estimated USD cost rounded to 6 decimal places."""
    p = _PRICE_PER_1K.get(model, _DEFAULT_PRICE)
    return round(
        prompt_tokens * p["prompt"] / 1000 + completion_tokens * p["completion"] / 1000,
        6,
    )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class RagMetrics:
    # ── query identity ────────────────────────────────────────────────────────
    query: str = ""  # original query as typed by the user
    session_name: str = ""
    timestamp: float = field(default_factory=time.time)

    # ── query rewriting ───────────────────────────────────────────────────────
    original_query: str = ""  # same as query (kept for explicitness)
    rewritten_query: str = ""  # standalone query after rewriting; "" if not rewritten
    was_rewritten: bool = False  # True when rewrite changed the query

    # ── classification ────────────────────────────────────────────────────────
    query_classification: str = ""  # intent label from QueryClassifier
    query_complexity: float = 0.0  # 0-1 complexity score
    query_word_count: int = 0

    # ── routing / mode ────────────────────────────────────────────────────────
    mode: str = ""  # fast | advanced | conversational | …
    tool_used: str = ""  # cache | docs | llm | web_search | …

    # ── cache ─────────────────────────────────────────────────────────────────
    cache_hit: bool = False

    # ── retrieval ─────────────────────────────────────────────────────────────
    retrieval_latency_ms: float = 0.0
    num_docs_retrieved: int = 0  # after initial vector search
    retrieved_doc_names: list[str] = field(default_factory=list)
    original_similarity_scores: list[float] = field(default_factory=list)  # top-k

    # ── reranker ──────────────────────────────────────────────────────────────
    reranker_used: bool = False
    reranker_scores: list[float] = field(default_factory=list)
    num_final_chunks: int = 0  # after reranking / top-k selection

    # ── LLM call ──────────────────────────────────────────────────────────────
    llm_latency_ms: float = 0.0
    model_name: str = field(default_factory=lambda: _DEFAULT_MODEL)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # ── costs & totals ────────────────────────────────────────────────────────
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    # ── response metadata ─────────────────────────────────────────────────────
    sources: list[str] = field(default_factory=list)
    language: str = ""
    emotion_detected: str = ""

    # ── groundedness / hallucination detection ────────────────────────────────
    # Only populated when tool_used == 'docs'.  -1.0 means "not evaluated".
    groundedness_score: float = -1.0  # 0.0–1.0 fraction of supported claims
    supported_claims: int = 0
    unsupported_claims: int = 0
    groundedness_verdict: str = (
        ""  # fully_supported | partially_supported | unsupported | ""
    )
    groundedness_latency_ms: float = 0.0  # time spent on the verification LLM call

    def finalise(self) -> None:
        """Compute derived fields (call once, just before save)."""
        self.estimated_cost_usd = estimate_cost(
            self.prompt_tokens, self.completion_tokens, self.model_name
        )
        self.total_tokens = self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
class MetricsStore:
    """SQLite-backed store for RagMetrics records."""

    DB_VERSION = 1

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    # ── internal helpers ──────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_metrics (
                    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp                  REAL    NOT NULL,
                    query                      TEXT    NOT NULL,
                    session_name               TEXT    NOT NULL DEFAULT '',
                    original_query             TEXT    NOT NULL DEFAULT '',
                    rewritten_query            TEXT    NOT NULL DEFAULT '',
                    was_rewritten              INTEGER NOT NULL DEFAULT 0,
                    query_classification       TEXT    NOT NULL DEFAULT '',
                    query_complexity           REAL    NOT NULL DEFAULT 0,
                    query_word_count           INTEGER NOT NULL DEFAULT 0,
                    mode                       TEXT    NOT NULL DEFAULT '',
                    tool_used                  TEXT    NOT NULL DEFAULT '',
                    cache_hit                  INTEGER NOT NULL DEFAULT 0,
                    retrieval_latency_ms       REAL    NOT NULL DEFAULT 0,
                    num_docs_retrieved         INTEGER NOT NULL DEFAULT 0,
                    retrieved_doc_names        TEXT    NOT NULL DEFAULT '[]',
                    original_similarity_scores TEXT    NOT NULL DEFAULT '[]',
                    reranker_used              INTEGER NOT NULL DEFAULT 0,
                    reranker_scores            TEXT    NOT NULL DEFAULT '[]',
                    num_final_chunks           INTEGER NOT NULL DEFAULT 0,
                    llm_latency_ms             REAL    NOT NULL DEFAULT 0,
                    model_name                 TEXT    NOT NULL DEFAULT '',
                    prompt_tokens              INTEGER NOT NULL DEFAULT 0,
                    completion_tokens          INTEGER NOT NULL DEFAULT 0,
                    total_tokens               INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd         REAL    NOT NULL DEFAULT 0,
                    total_latency_ms           REAL    NOT NULL DEFAULT 0,
                    sources                    TEXT    NOT NULL DEFAULT '[]',
                    language                   TEXT    NOT NULL DEFAULT '',
                    emotion_detected           TEXT    NOT NULL DEFAULT '',
                    groundedness_score         REAL    NOT NULL DEFAULT -1,
                    supported_claims           INTEGER NOT NULL DEFAULT 0,
                    unsupported_claims         INTEGER NOT NULL DEFAULT 0,
                    groundedness_verdict       TEXT    NOT NULL DEFAULT '',
                    groundedness_latency_ms    REAL    NOT NULL DEFAULT 0
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rm_timestamp ON rag_metrics(timestamp DESC)"
            )
            # ── Live migration: add groundedness columns to existing DBs ──────
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(rag_metrics)").fetchall()
            }
            migrations = [
                ("groundedness_score", "REAL    NOT NULL DEFAULT -1"),
                ("supported_claims", "INTEGER NOT NULL DEFAULT 0"),
                ("unsupported_claims", "INTEGER NOT NULL DEFAULT 0"),
                ("groundedness_verdict", "TEXT    NOT NULL DEFAULT ''"),
                ("groundedness_latency_ms", "REAL    NOT NULL DEFAULT 0"),
                ("original_query", "TEXT    NOT NULL DEFAULT ''"),
                ("rewritten_query", "TEXT    NOT NULL DEFAULT ''"),
                ("was_rewritten", "INTEGER NOT NULL DEFAULT 0"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE rag_metrics ADD COLUMN {col_name} {col_def}"
                    )

    # ── public API ────────────────────────────────────────────────────────────
    def save(self, m: RagMetrics) -> int:
        """Insert one record; return its rowid."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO rag_metrics (
                    timestamp, query, session_name,
                    original_query, rewritten_query, was_rewritten,
                    query_classification, query_complexity, query_word_count,
                    mode, tool_used, cache_hit,
                    retrieval_latency_ms, num_docs_retrieved,
                    retrieved_doc_names, original_similarity_scores,
                    reranker_used, reranker_scores, num_final_chunks,
                    llm_latency_ms, model_name,
                    prompt_tokens, completion_tokens, total_tokens,
                    estimated_cost_usd, total_latency_ms,
                    sources, language, emotion_detected,
                    groundedness_score, supported_claims, unsupported_claims,
                    groundedness_verdict, groundedness_latency_ms
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    m.timestamp,
                    m.query,
                    m.session_name,
                    m.original_query or m.query,
                    m.rewritten_query or m.query,
                    int(m.was_rewritten),
                    m.query_classification,
                    m.query_complexity,
                    m.query_word_count,
                    m.mode,
                    m.tool_used,
                    int(m.cache_hit),
                    m.retrieval_latency_ms,
                    m.num_docs_retrieved,
                    json.dumps(m.retrieved_doc_names),
                    json.dumps([round(s, 4) for s in m.original_similarity_scores]),
                    int(m.reranker_used),
                    json.dumps([round(s, 4) for s in m.reranker_scores]),
                    m.num_final_chunks,
                    m.llm_latency_ms,
                    m.model_name,
                    m.prompt_tokens,
                    m.completion_tokens,
                    m.total_tokens,
                    m.estimated_cost_usd,
                    m.total_latency_ms,
                    json.dumps(m.sources),
                    m.language,
                    m.emotion_detected,
                    m.groundedness_score,
                    m.supported_claims,
                    m.unsupported_claims,
                    m.groundedness_verdict,
                    m.groundedness_latency_ms,
                ),
            )
            return cur.lastrowid or 0

    def fetch(
        self,
        limit: int = 100,
        offset: int = 0,
        session_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return records newest-first as plain dicts."""
        with self._connect() as conn:
            if session_name:
                rows = conn.execute(
                    "SELECT * FROM rag_metrics WHERE session_name = ? "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (session_name, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM rag_metrics "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def aggregate(self) -> dict[str, Any]:
        """Compute summary stats across all stored records."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                                                AS total_queries,
                    SUM(cache_hit)                                          AS cache_hits,
                    ROUND(AVG(retrieval_latency_ms), 1)                     AS avg_retrieval_ms,
                    ROUND(AVG(llm_latency_ms), 1)                           AS avg_llm_ms,
                    ROUND(AVG(total_latency_ms), 1)                         AS avg_total_ms,
                    ROUND(MAX(total_latency_ms), 1)                         AS max_total_ms,
                    SUM(total_tokens)                                       AS total_tokens_used,
                    ROUND(SUM(estimated_cost_usd), 6)                       AS total_cost_usd,
                    ROUND(AVG(num_docs_retrieved), 1)                       AS avg_docs_retrieved,
                    ROUND(AVG(num_final_chunks), 1)                         AS avg_final_chunks,
                    COUNT(CASE WHEN groundedness_score >= 0 THEN 1 END)     AS grounded_queries,
                    ROUND(AVG(CASE WHEN groundedness_score >= 0
                                   THEN groundedness_score END) * 100, 1)   AS avg_groundedness_pct,
                    COUNT(CASE WHEN groundedness_verdict = 'fully_supported'
                                    THEN 1 END)                             AS fully_supported_count,
                    COUNT(CASE WHEN groundedness_verdict = 'partially_supported'
                                    THEN 1 END)                             AS partially_supported_count,
                    COUNT(CASE WHEN groundedness_verdict = 'unsupported'
                                    THEN 1 END)                             AS unsupported_count
                FROM rag_metrics
                """).fetchone()
        d = dict(row) if row else {}
        total = d.get("total_queries") or 0
        hits = d.get("cache_hits") or 0
        d["cache_hit_rate"] = round(hits / total * 100, 1) if total else 0.0
        return d

    def clear(self) -> int:
        """Delete all records; return deleted count."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM rag_metrics")
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM rag_metrics").fetchone()[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for json_col in (
        "retrieved_doc_names",
        "original_similarity_scores",
        "reranker_scores",
        "sources",
    ):
        if json_col in d and isinstance(d[json_col], str):
            try:
                d[json_col] = json.loads(d[json_col])
            except (json.JSONDecodeError, TypeError):
                d[json_col] = []
    d["cache_hit"] = bool(d.get("cache_hit"))
    d["reranker_used"] = bool(d.get("reranker_used"))
    d["was_rewritten"] = bool(d.get("was_rewritten"))
    return d


# ---------------------------------------------------------------------------
# Module-level singleton (initialised once when imported by main.py)
# ---------------------------------------------------------------------------
_store: MetricsStore | None = None


def get_store() -> MetricsStore:
    global _store
    if _store is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "rag_metrics.db",
        )
        _store = MetricsStore(db_path)
    return _store
