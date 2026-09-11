from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import pickle
import re
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar

# RagMetrics imported at runtime so Agent.run() can accept it as a parameter.
# Deferred to avoid circular imports at module load time.
try:
    from app.rag_metrics import RagMetrics
except ImportError:  # graceful fallback if module is absent
    RagMetrics = None  # type: ignore[assignment,misc]

import cv2
import docx
import nltk
import numpy as np
import pdfplumber
import pytesseract
import requests
import whisper
from groq import Groq as GroqClient
from app.llm_fallback import GROQ_MODEL, llm_complete as _llm_complete
from llama_index.core import (
    Document,
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Module logger ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Observability (lazy import avoids circular dependency) ─────────────────────
def _obs():
    try:
        from app.observability import obs  # type: ignore[import]

        return obs
    except Exception:  # noqa: BLE001
        return None


# ── Configuration ──────────────────────────────────────────────────────────────
PERSIST_DIR = os.path.join(os.path.dirname(BASE_DIR), "storage")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
HASH_CACHE_FILE = os.path.join(os.path.dirname(BASE_DIR), "text_hash.txt")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = GroqClient(api_key=GROQ_API_KEY)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retrieval parameters
K_INIT = 20
K_RERANK = 10
K_FINAL = 5
SIMILARITY_THRESHOLD = 0.3
DOC_RELEVANCE_THRESHOLD = 0.25
CACHE_SIZE = 1000
CACHE_SIMILARITY_THRESHOLD = 0.95

_reranker: CrossEncoder | None = None
_reranker_lock = threading.Lock()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ── Cache System ───────────────────────────────────────────────────────────────
@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    response: str
    sources: list[str]
    timestamp: float
    scope: str = ""


class MultiLevelCache:
    """L1 (memory) + L2 (disk) caching with semantic similarity."""

    def __init__(self, cache_dir: str = CACHE_DIR, max_size: int = CACHE_SIZE) -> None:
        self.cache_dir = cache_dir
        self.max_size = max_size
        os.makedirs(cache_dir, exist_ok=True)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache_file = os.path.join(cache_dir, "l2_cache.pkl")
        self._load_l2_cache()

    def _load_l2_cache(self) -> None:
        if os.path.exists(self.l2_cache_file):
            try:
                with open(self.l2_cache_file, "rb") as f:
                    self.l1_cache = pickle.load(f)
            except (OSError, pickle.UnpicklingError) as exc:
                logger.warning("Failed to load L2 cache: %s", exc)

    def _save_l2_cache(self) -> None:
        try:
            with open(self.l2_cache_file, "wb") as f:
                pickle.dump(self.l1_cache, f)
        except OSError as exc:
            logger.warning("Failed to save L2 cache: %s", exc)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

    def get(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        scope: str = "",
    ) -> tuple[str, list[str]] | None:
        key = hashlib.md5(f"{scope}\0{query}".encode()).hexdigest()
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
            e = self.l1_cache[key]
            return e.response, e.sources
        if query_embedding is not None:
            for e in self.l1_cache.values():
                if (
                    getattr(e, "scope", "") == scope
                    and self._cosine(query_embedding, e.embedding)
                    >= CACHE_SIMILARITY_THRESHOLD
                ):
                    return e.response, e.sources
        return None

    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: str,
        sources: list[str],
        scope: str = "",
    ) -> None:
        key = hashlib.md5(f"{scope}\0{query}".encode()).hexdigest()
        if len(self.l1_cache) >= self.max_size:
            self.l1_cache.popitem(last=False)
        self.l1_cache[key] = CacheEntry(
            query=query,
            embedding=query_embedding,
            response=response,
            sources=sources,
            timestamp=time.time(),
            scope=scope,
        )
        if len(self.l1_cache) % 10 == 0:
            self._save_l2_cache()


cache = MultiLevelCache()


# ── Helpers ────────────────────────────────────────────────────────────────────
def initialize_settings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    logger.info("Embed model: %s | LLM: %s (via groq SDK)", EMBEDDING_MODEL, GROQ_MODEL)


def warm_reranker() -> bool:
    """Load the optional cross-encoder once so retrieval queries do not load it."""
    global _reranker
    if _reranker is not None:
        return True
    with _reranker_lock:
        if _reranker is None:
            try:
                _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Cross-encoder reranker ready")
            except Exception as exc:  # noqa: BLE001 — optional model warm-up
                logger.warning("Reranker warm-up failed: %s", exc)
                return False
    return True


def _groq_complete(
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 700,
    trace: object = None,
    span_name: str = "rag-llm-call",
) -> tuple[str, int, int]:
    """Single-turn completion with automatic provider fallback.

    Tries Groq first; falls back to Gemini if Groq is unavailable.
    Accepts an optional Langfuse `trace` object — when provided, each call
    records a Generation span under the parent trace.

    Returns
    -------
    (text, prompt_tokens, completion_tokens)
    """
    text, pt, ct, provider = _llm_complete(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        trace=trace,
        span_name=span_name,
    )
    if provider != "groq":
        logger.warning(
            "_groq_complete: request served by fallback provider '%s'", provider
        )
    return text, pt, ct


# ── File Extractors ────────────────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    except OSError as exc:
        logger.error("PDF extraction failed: %s", exc)
        return ""


def extract_text_from_image(path: str) -> str:
    try:
        return pytesseract.image_to_string(cv2.imread(path))
    except OSError as exc:
        logger.error("Image OCR failed: %s", exc)
        return ""


def extract_text_from_docx(path: str) -> str:
    try:
        return "\n".join([p.text for p in docx.Document(path).paragraphs])
    except OSError as exc:
        logger.error("DOCX extraction failed: %s", exc)
        return ""


def extract_text_from_txt(path: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            logger.error("TXT extraction failed: %s", exc)
            return ""
    return ""


def extract_text_from_csv(path: str) -> str:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return "\n".join([" | ".join(row) for row in csv.reader(f)])
    except OSError as exc:
        logger.error("CSV extraction failed: %s", exc)
        return ""


def extract_text_from_json(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return json.dumps(json.load(f), indent=2)
    except OSError as exc:
        logger.error("JSON extraction failed: %s", exc)
        return ""


def extract_text_from_audio(path: str) -> str:
    try:
        return whisper.load_model("base").transcribe(path)["text"]
    except OSError as exc:
        logger.error("Audio transcription failed: %s", exc)
        return ""


def extract_text_from_zip(path: str) -> str:
    text = ""
    try:
        with zipfile.ZipFile(path, "r") as z, tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            for root, _, files in os.walk(tmp):
                for name in files:
                    text += extract_text(os.path.join(root, name)) + "\n"
    except (OSError, zipfile.BadZipFile) as exc:
        logger.error("ZIP extraction failed: %s", exc)
    return text


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".jpg": extract_text_from_image,
        ".jpeg": extract_text_from_image,
        ".png": extract_text_from_image,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".csv": extract_text_from_csv,
        ".json": extract_text_from_json,
        ".mp3": extract_text_from_audio,
        ".wav": extract_text_from_audio,
        ".mp4": extract_text_from_audio,
        ".zip": extract_text_from_zip,
    }
    fn = extractors.get(ext)
    return fn(path) if fn else ""


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace("•", "-").replace("–", "-").strip()


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def has_file_changed(text: str) -> bool:
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE) as f:
            return compute_hash(text) != f.read().strip()
    return True


def cache_hash(text: str) -> None:
    with open(HASH_CACHE_FILE, "w") as f:
        f.write(compute_hash(text))


# ── Index Creation ─────────────────────────────────────────────────────────────
def create_index(file_paths: list[str]) -> VectorStoreIndex | None:
    docs = []
    for path in file_paths:
        text = clean_text(extract_text(path))
        if text:
            docs.append(
                Document(text=text, metadata={"source": os.path.basename(path)})
            )

    if not docs:
        logger.warning("No valid text extracted from files.")
        return None

    combined = "\n".join([d.text for d in docs])
    if os.path.exists(PERSIST_DIR) and not has_file_changed(combined):
        logger.info("Loading existing index…")
        ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        loaded = load_index_from_storage(ctx)
        if not isinstance(loaded, VectorStoreIndex):
            logger.warning("Loaded index is not a VectorStoreIndex; rebuilding.")
        else:
            return loaded

    logger.info("Building new index…")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    cache_hash(combined)
    return index


# ── Context Compression ────────────────────────────────────────────────────────
def compress_context(chunks: list[str], max_tokens: int = 1500) -> str:
    compressed, token_count = [], 0
    for chunk in chunks:
        n = len(chunk.split())
        if token_count + n > max_tokens:
            compressed.append(" ".join(chunk.split()[: max_tokens - token_count]))
            break
        compressed.append(chunk)
        token_count += n

    seen: set[str] = set()
    sentences: list[str] = []
    for chunk in compressed:
        for s in sent_tokenize(chunk):
            sc = s.strip().lower()
            if sc not in seen and len(sc) > 20:
                seen.add(sc)
                sentences.append(s)
    return " ".join(sentences)


# ── Tools ──────────────────────────────────────────────────────────────────────
class Tools:
    """Stateless tool collection used by the Agent."""

    @staticmethod
    def search_docs(
        index: VectorStoreIndex | list[VectorStoreIndex],
        query: str,
        query_embedding: list[float] | None = None,
    ) -> tuple[str | None, list[str], float, list[float], list[str], bool, list[float]]:
        """Hybrid retrieval: vector similarity + optional cross-encoder reranking.

        Returns
        -------
        (context, sources, best_score,
         original_scores, retrieved_doc_names,
         reranker_used, reranker_scores)
        """
        indexes = index if isinstance(index, list) else [index]
        if query_embedding is None:
            query_embedding = Settings.embed_model.get_query_embedding(query)
        query_bundle = QueryBundle(query_str=query, embedding=query_embedding)
        nodes = []
        for document_index in indexes:
            retriever = document_index.as_retriever(similarity_top_k=K_INIT)
            nodes.extend(retriever.retrieve(query_bundle))
        nodes.sort(key=lambda node: float(node.score or 0.0), reverse=True)

        if not nodes:
            return None, [], 0.0, [], [], False, []

        best_score: float = (
            float(nodes[0].score)
            if hasattr(nodes[0], "score") and nodes[0].score is not None
            else 0.0
        )

        # Capture original similarity scores and doc names before any reranking
        original_scores: list[float] = [
            float(n.score) if n.score is not None else 0.0 for n in nodes[:K_RERANK]
        ]
        all_doc_names: list[str] = [
            n.metadata.get("source", "Unknown") for n in nodes[:K_RERANK]
        ]

        reranker_used = False
        reranker_scores: list[float] = []

        if best_score < SIMILARITY_THRESHOLD and len(nodes) > K_FINAL:
            try:
                if not warm_reranker() or _reranker is None:
                    raise RuntimeError("Cross-encoder reranker is unavailable")
                pairs = [[query, n.text] for n in nodes[:K_RERANK]]
                raw_scores = _reranker.predict(pairs)
                ranked = sorted(
                    zip(nodes[:K_RERANK], raw_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                best_score = float(ranked[0][1]) if ranked else best_score
                reranker_scores = [float(s) for _, s in ranked[:K_FINAL]]
                nodes = [n for n, _ in ranked[:K_FINAL]]
                reranker_used = True
            except (OSError, RuntimeError) as exc:
                logger.warning("Reranking failed: %s. Falling back to top-k.", exc)
                nodes = nodes[:K_FINAL]
        else:
            nodes = nodes[:K_FINAL]

        context = compress_context([n.text for n in nodes])
        sources = list(
            dict.fromkeys(n.metadata.get("source", "Unknown") for n in nodes[:3])
        )
        return (
            context,
            sources,
            best_score,
            original_scores,
            all_doc_names,
            reranker_used,
            reranker_scores,
        )

    @staticmethod
    def web_search(query: str) -> str:
        """Compatibility adapter to the shared DuckDuckGo implementation."""
        from app.web_search import search_web
        try:
            return "\n".join(f"{r['title']} ({r['url']}): {r['snippet']}" for r in search_web(query))
        except Exception:
            logger.warning("DuckDuckGo unavailable")
            return ""

    @staticmethod
    def calculator(expr: str) -> str:
        try:
            safe = re.sub(r"[^0-9+\-*/().% ]", "", expr)
            result = eval(
                safe, {"__builtins__": {}}, {k: getattr(math, k) for k in dir(math)}
            )
            return str(result)
        except (SyntaxError, ArithmeticError, NameError, TypeError, ValueError) as exc:
            return f"Calculation error: {exc}"


class QueryClassifier:
    CONVERSATIONAL: ClassVar[list[str]] = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "good morning",
        "good night",
    ]
    ANALYTICAL: ClassVar[list[str]] = [
        "compare",
        "analyze",
        "why",
        "how can",
        "how does",
        "evaluate",
        "difference",
        "relationship",
    ]
    FACTUAL: ClassVar[list[str]] = [
        "what",
        "who",
        "when",
        "where",
        "define",
        "explain",
        "list",
        "show",
        "tell",
    ]
    TEMPORAL: ClassVar[list[str]] = [
        "latest",
        "current",
        "today",
        "news",
        "price",
        "live",
        "now",
        "2024",
        "2025",
        "2026",
    ]
    MATH_WORDS: ClassVar[list[str]] = [
        "calculate",
        "compute",
        "how many hours",
        "how many minutes",
        "how many days",
        "sum of",
        "total of",
    ]

    @classmethod
    def classify(cls, query: str) -> dict:
        q = query.lower().strip()
        words = q.split()
        wc = len(words)

        is_greeting = wc <= 4 and any(q.startswith(kw) for kw in cls.CONVERSATIONAL)
        has_number = bool(re.search(r"\d", q))
        has_math_op = any(op in q for op in ["+", "-", "*", "/", "^", "%"])
        is_math = (has_number and has_math_op) or (
            has_number and any(w in q for w in cls.MATH_WORDS)
        )
        is_temporal = any(kw in q for kw in cls.TEMPORAL)

        intent = "factual"
        if is_greeting:
            intent = "conversational"
        elif is_math:
            intent = "calculator"
        elif is_temporal:
            intent = "web_search"
        elif any(kw in q for kw in cls.ANALYTICAL):
            intent = "analytical"

        return {
            "intent": intent,
            "complexity": min(1.0, (wc / 20) + (0.3 if intent == "analytical" else 0)),
            "word_count": wc,
            "is_math": is_math,
            "is_temporal": is_temporal,
        }


# ── Groundedness ──────────────────────────────────────────────────────────────
# Minimum fraction of claims that must be supported to keep the LLM answer.
# Below this threshold the answer is replaced with a safe fallback message.
GROUNDEDNESS_MIN_SCORE: float = 0.50  # 50 %

# Label boundaries
_VERDICT_FULLY = "fully_supported"  # score ≥ 0.80
_VERDICT_PARTIAL = "partially_supported"  # 0.50 ≤ score < 0.80
_VERDICT_UNSUPPORTED = "unsupported"  # score < 0.50

GROUNDEDNESS_FALLBACK = (
    "I don't have enough information in the available knowledge base "
    "to answer this reliably. Please try rephrasing your question or "
    "upload additional relevant documents."
)


@dataclass
class GroundednessResult:
    """Output of the groundedness evaluator."""

    score: float  # 0.0 – 1.0
    supported_claims: int
    unsupported_claims: int
    verdict: str  # one of the _VERDICT_* constants
    latency_ms: float = 0.0


def _verdict_from_score(score: float) -> str:
    if score >= 0.80:
        return _VERDICT_FULLY
    if score >= GROUNDEDNESS_MIN_SCORE:
        return _VERDICT_PARTIAL
    return _VERDICT_UNSUPPORTED


# ── Query Rewriter ─────────────────────────────────────────────────────────────
# Minimum turns of prior history needed before rewriting is attempted.
# With 0 prior turns there is nothing to resolve, so we skip the LLM call.
_REWRITE_MIN_HISTORY_TURNS: int = 1

# Pronouns / references that almost certainly need a prior context to resolve.
_REFERENTIAL_TOKENS: frozenset[str] = frozenset(
    {
        "it",
        "its",
        "this",
        "that",
        "they",
        "them",
        "their",
        "those",
        "these",
        "he",
        "she",
        "him",
        "her",
        "such",
        "same",
        "so",
        "do",
        "does",
    }
)

# Short filler phrases that are never standalone queries on their own.
_DEPENDENT_OPENERS: tuple[str, ...] = (
    "what about",
    "how about",
    "and what",
    "but what",
    "why is that",
    "can you explain",
    "tell me more",
    "more about",
    "what else",
    "what if",
    "any other",
)


class QueryRewriter:
    """
    Rewrites an ambiguous follow-up query into a fully self-contained query
    using the recent conversation history.

    Decision heuristic (cheap, no LLM)
    ────────────────────────────────────
    1. No prior history  →  return query unchanged (nothing to resolve)
    2. Query is long (≥ 8 words) AND contains no referential pronouns
       AND does not start with a dependent opener  →  self-contained, skip
    3. Otherwise  →  call the LLM to produce a standalone rewrite

    The LLM is instructed to return ONLY the rewritten query — no explanation,
    no quotation marks, no filler.  If the query is already self-contained the
    LLM is told to return it verbatim.
    """

    @staticmethod
    def needs_rewrite(query: str, history: list[dict]) -> bool:
        """Return True when the LLM rewrite step should be attempted."""
        if len(history) < _REWRITE_MIN_HISTORY_TURNS:
            return False
        q = query.lower().strip()
        words = q.split()
        has_referential = any(w.rstrip("?,!.") in _REFERENTIAL_TOKENS for w in words)
        starts_dependent = any(q.startswith(op) for op in _DEPENDENT_OPENERS)
        is_short = len(words) < 8
        return has_referential or starts_dependent or is_short

    @staticmethod
    def rewrite(query: str, history: list[dict]) -> str:
        """
        Return a standalone version of *query* given *history*.
        Returns the original query unchanged on any failure.
        """
        if not QueryRewriter.needs_rewrite(query, history):
            return query

        # Build a compact, token-efficient history block (last 6 turns max)
        turns: list[str] = []
        for msg in history[-6:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = str(msg.get("content", "")).strip()
            # Truncate very long assistant replies to keep prompt short
            if role == "Assistant" and len(content) > 300:
                content = content[:300] + "…"
            turns.append(f"{role}: {content}")
        history_block = "\n".join(turns)

        prompt = (
            "You are a query rewriting assistant.\n\n"
            "TASK: Given a conversation history and a follow-up question, rewrite "
            "the follow-up question into a fully self-contained, standalone question "
            "that can be understood without any prior context.\n\n"
            "RULES:\n"
            "- Resolve all pronouns, references, and ellipsis using the history.\n"
            "- Preserve the user's original intent exactly — do not add, remove, or "
            "  change the topic.\n"
            "- Output ONLY the rewritten question. No explanation, no quotes, no preamble.\n"
            "- If the question is already fully self-contained, return it word-for-word.\n\n"
            f"Conversation History:\n{history_block}\n\n"
            f"Follow-up Question: {query}\n\n"
            "Standalone Question:"
        )

        try:
            raw, _, _ = _groq_complete(prompt, temperature=0.0, max_tokens=120)
            rewritten = raw.strip().strip('"').strip("'").strip()
            # Sanity guard: if the LLM hallucinated something very long or empty,
            # fall back to the original query.
            if not rewritten or len(rewritten) > 400:
                return query
            return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed: %s — using original", exc)
            return query


# ── Agent ──────────────────────────────────────────────────────────────────────
class Agent:
    """
    Agentic RAG orchestrator.

    Decision flow per query
    ───────────────────────
    0. Query rewriting (if conversation history present)
       → resolve pronouns / references → standalone query
    1. Cache hit?            → return cached answer
    2. Greeting?             → conversational reply
    3. Math expression?      → calculator tool
    4. Temporal / live data? → web search tool
    5. Has index?
       a. Retrieve from docs (hybrid vector + rerank)
       b. Score ≥ DOC_RELEVANCE_THRESHOLD?
          → build context-grounded answer
          → groundedness check (supported / partially / unsupported)
          → if score < GROUNDEDNESS_MIN_SCORE → return fallback message
          → verify; if vague → enrich with web search
       c. Score too low (off-topic for the docs)?
          → answer from LLM general knowledge directly
          → if LLM is unsure → try web search as fallback
    6. No index?             → LLM general knowledge (+ web if needed)
    """

    def __init__(
        self,
        index: VectorStoreIndex | list[VectorStoreIndex] | None,
        cache_scope: str | None = None,
    ) -> None:
        self.index = index
        self.tools = Tools()
        self.cache_scope = cache_scope or (
            "documents" if index is not None else "general"
        )
        # Langfuse trace injected by the caller (main.py) before Agent.run()
        self._obs_trace: object = None

    # ── Private helpers ────────────────────────────────────────────────────────
    def _llm(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 1200,
        span_name: str = "rag-llm-call",
    ) -> tuple[str, int, int]:
        """Call the LLM; pass the Langfuse trace so each call gets a sub-span."""
        return _groq_complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            trace=self._obs_trace,
            span_name=span_name,
        )

    def _is_vague(self, text: str) -> bool:
        vague_phrases = [
            "i don't know",
            "not sure",
            "i cannot",
            "not available",
            "no information",
            "i'm unable",
            "cannot determine",
        ]
        return any(p in text.lower() for p in vague_phrases)

    def _extract_math_expr(self, query: str) -> str:
        prompt = (
            "Extract only the math expression from the query as a Python-evaluable string. "
            "Return ONLY the expression, no explanation.\n"
            f"Query: {query}\nExpression:"
        )
        expr, _, _ = self._llm(prompt, temperature=0.0, max_tokens=60)
        return re.sub(r"[^0-9+\-*/().% ]", "", expr).strip()

    def _answer_from_docs(self, query: str, context: str) -> tuple[str, int, int]:
        prompt = (
            "You are a helpful assistant. Use ONLY the provided document context to answer.\n"
            "Be specific, sufficiently detailed, and easy to understand. Do not omit "
            "relevant details merely to be brief. If the answer is not in the context, say exactly: "
            "'Not available in the documentation.'\n"
            "Formatting requirements:\n"
            "- Start with a direct answer in one or two sentences.\n"
            "- For multi-part answers, use descriptive Markdown headings (## Heading).\n"
            "- Put a blank line before and after every heading and paragraph.\n"
            "- Put every bullet or numbered item on its own line.\n"
            "- Explain important points in short paragraphs, not a dense wall of text.\n"
            "- End substantial answers with a '## Summary' section containing 2-4 concise bullets.\n"
            "- Do not output literal \\n characters; output real line breaks.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return self._llm(prompt, span_name="rag-answer-from-docs")

    def _answer_from_knowledge(self, query: str) -> tuple[str, int, int]:
        prompt = (
            "You are a knowledgeable assistant. The user's question is not covered by any "
            "uploaded documents, so answer using your general knowledge.\n"
            "Be accurate, helpful, sufficiently detailed, and well summarized. Start with "
            "a direct answer. For multi-part answers, use descriptive Markdown headings, "
            "short paragraphs separated by blank lines, and one bullet per line. End a "
            "substantial answer with a '## Summary' section of 2-4 bullets. Use real line "
            "breaks and never return a dense wall of text.\n\n"
            f"Question: {query}\nAnswer:"
        )
        return self._llm(prompt, temperature=0.5, span_name="rag-answer-from-knowledge")

    def _answer_with_web(self, query: str, web_context: str) -> tuple[str, int, int]:
        prompt = (
            "Use the following web search results to answer the question accurately and in "
            "useful detail. Start with a direct answer. For multi-part answers, use Markdown "
            "headings, blank lines between short paragraphs, and one bullet per line. End "
            "substantial answers with a '## Summary' section of 2-4 bullets.\n\n"
            f"Web Results:\n{web_context}\n\nQuestion: {query}\nAnswer:"
        )
        return self._llm(prompt, span_name="rag-answer-with-web")

    def _check_groundedness(self, answer: str, context: str) -> GroundednessResult:
        """
        Second-pass LLM verifier: decompose the answer into atomic claims,
        check each claim against the retrieved context, and return a
        GroundednessResult.

        The LLM is asked to output ONLY a strict JSON object so we can parse
        it deterministically without regex heuristics.
        """
        t0 = time.perf_counter()

        prompt = (
            "You are a strict groundedness evaluator. Your task is to verify whether "
            "the claims in an AI-generated answer are supported by the provided context.\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the CONTEXT carefully.\n"
            "2. Break the ANSWER into individual atomic factual claims "
            "(ignore formatting, headings, and filler phrases).\n"
            "3. For each claim decide: SUPPORTED (explicitly or logically backed by the context) "
            "or UNSUPPORTED (absent, contradicted, or cannot be verified).\n"
            "4. Output ONLY a valid JSON object — no markdown, no explanation — in exactly "
            "this format:\n"
            '{"supported": <int>, "unsupported": <int>, '
            '"claims": [{"text": "<claim>", "verdict": "supported"|"unsupported"}]}\n\n'
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"ANSWER:\n{answer[:2000]}\n\n"
            "JSON output:"
        )

        try:
            raw, _pt, _ct = self._llm(
                prompt,
                temperature=0.0,
                max_tokens=600,
                span_name="rag-groundedness-check",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Groundedness LLM call failed: %s", exc)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return GroundednessResult(
                score=1.0,
                supported_claims=0,
                unsupported_claims=0,
                verdict=_VERDICT_FULLY,
                latency_ms=latency_ms,
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── Parse JSON ────────────────────────────────────────────────────────
        try:
            # Strip any accidental markdown fences
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Extract the first {...} block in case the LLM added preamble
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                raise ValueError("No JSON object found in LLM output")
            parsed = json.loads(m.group())
            supported = int(parsed.get("supported", 0))
            unsupported = int(parsed.get("unsupported", 0))
            total = supported + unsupported
            if total == 0:
                # Edge case: no claims extracted → treat as fully supported
                score = 1.0
            else:
                score = round(supported / total, 4)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Groundedness parse error (%s) — raw: %.120s", exc, raw)
            # Parsing failure: do not penalise; pass through as fully supported
            return GroundednessResult(
                score=1.0,
                supported_claims=0,
                unsupported_claims=0,
                verdict=_VERDICT_FULLY,
                latency_ms=latency_ms,
            )

        verdict = _verdict_from_score(score)
        return GroundednessResult(
            score=score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            verdict=verdict,
            latency_ms=latency_ms,
        )

    # ── Public interface ───────────────────────────────────────────────────────
    def run(
        self,
        query: str,
        metrics: RagMetrics | None = None,
        conv_history: list[dict] | None = None,
        knowledge_only: bool = False,
    ) -> tuple[str, list[str], str, GroundednessResult | None, str]:
        """
        Returns (response, sources, tool_used, groundedness, rewritten_query).

        rewritten_query  — standalone version of *query* after resolving
                           conversational references.  Equals *query* when
                           no rewrite was needed.
        groundedness     — GroundednessResult when tool_used='docs'; else None.
        tool_used        — cache | conversational | calculator | web_search |
                           docs | llm | llm+web
        """
        cache_scope = self.cache_scope + (":private-only-v1" if knowledge_only else "")
        # ── 0. Conversational query rewriting ─────────────────────────────────
        history: list[dict] = conv_history or []
        rewritten_query = QueryRewriter.rewrite(query, history)
        was_rewritten = rewritten_query != query
        if was_rewritten:
            logger.info(
                "Query rewritten:\n  original : %r\n  rewritten: %r",
                query,
                rewritten_query,
            )
        eq = rewritten_query  # effective query used throughout pipeline

        if metrics is not None:
            metrics.original_query = query
            metrics.rewritten_query = rewritten_query
            metrics.was_rewritten = was_rewritten

        # ── helpers to accumulate token counts ────────────────────────────────
        def _add_tokens(pt: int, ct: int) -> None:
            if metrics is not None:
                metrics.prompt_tokens += pt
                metrics.completion_tokens += ct

        # ── 1. Cache lookup (keyed on the rewritten query) ────────────────────
        _cache_t0 = time.perf_counter()
        embedding = np.array(Settings.embed_model.get_text_embedding(eq))
        cached = cache.get(eq, embedding, scope=cache_scope)
        _cache_ms = round((time.perf_counter() - _cache_t0) * 1000, 1)

        _obs_inst = _obs()
        if cached:
            if _obs_inst:
                with _obs_inst.cache_span(True, eq, _cache_ms):
                    pass
            if metrics is not None:
                metrics.cache_hit = True
                metrics.tool_used = "cache"
            return cached[0], cached[1], "cache", None, rewritten_query

        if _obs_inst:
            with _obs_inst.cache_span(False, eq, _cache_ms):
                pass

        classification = QueryClassifier.classify(eq)
        if knowledge_only:
            classification["intent"] = "document_qa"
        if metrics is not None:
            metrics.cache_hit = False
            metrics.query_classification = classification["intent"]
            metrics.query_complexity = round(float(classification["complexity"]), 3)
            metrics.query_word_count = int(classification["word_count"])

        # 2. Greeting / small-talk
        if classification["intent"] == "conversational":
            response = "Hello! How can I help you today? Feel free to ask me anything."
            cache.set(eq, embedding, response, [], scope=cache_scope)
            if metrics is not None:
                metrics.tool_used = "conversational"
                metrics.mode = "fast"
            return response, [], "conversational", None, rewritten_query

        # 3. Math
        if classification["intent"] == "calculator":
            expr = self._extract_math_expr(eq)
            if expr:
                result = self.tools.calculator(expr)
                response = f"Result: {result}"
            else:
                response = "Could not extract a valid math expression from your query."
            cache.set(eq, embedding, response, [], scope=cache_scope)
            if metrics is not None:
                metrics.tool_used = "calculator"
                metrics.mode = "fast"
            return response, [], "calculator", None, rewritten_query

        # 4. Live / temporal data → web first
        if classification["intent"] == "web_search":
            web_ctx = self.tools.web_search(eq)
            if web_ctx:
                llm_t0 = time.perf_counter()
                response, pt, ct = self._answer_with_web(eq, web_ctx)
                _add_tokens(pt, ct)
                if metrics is not None:
                    metrics.llm_latency_ms = round(
                        (time.perf_counter() - llm_t0) * 1000, 1
                    )
                sources: list[str] = ["web"]
            else:
                llm_t0 = time.perf_counter()
                response, pt, ct = self._answer_from_knowledge(eq)
                _add_tokens(pt, ct)
                if metrics is not None:
                    metrics.llm_latency_ms = round(
                        (time.perf_counter() - llm_t0) * 1000, 1
                    )
                sources = []
            cache.set(eq, embedding, response, sources, scope=cache_scope)
            if metrics is not None:
                metrics.tool_used = "web_search"
                metrics.mode = "fast"
            return response, sources, "web_search", None, rewritten_query

        # 5a. Document retrieval (if index exists)
        if self.index is not None:
            ret_t0 = time.perf_counter()
            (
                doc_context,
                doc_sources,
                best_score,
                original_scores,
                doc_names,
                reranker_used,
                reranker_scores,
            ) = self.tools.search_docs(self.index, eq, embedding.tolist())
            retrieval_ms = round((time.perf_counter() - ret_t0) * 1000, 1)

            if metrics is not None:
                metrics.retrieval_latency_ms = retrieval_ms
                metrics.num_docs_retrieved = len(original_scores)
                metrics.original_similarity_scores = original_scores
                metrics.retrieved_doc_names = doc_names
                metrics.reranker_used = reranker_used
                metrics.reranker_scores = reranker_scores
                metrics.num_final_chunks = (
                    K_FINAL if reranker_used else min(K_FINAL, len(original_scores))
                )
                metrics.mode = "advanced"

            # ── Langfuse: retrieval span ───────────────────────────────────────
            if _obs_inst:
                with _obs_inst.retrieval_span(
                    eq,
                    n_results=len(original_scores),
                    top_score=best_score,
                    latency_ms=retrieval_ms,
                    doc_names=doc_names,
                ):
                    pass
                if reranker_used:
                    with _obs_inst.rerank_span(
                        n_in=len(original_scores),
                        n_out=K_FINAL,
                        top_scores=reranker_scores[:5],
                        latency_ms=retrieval_ms,
                    ):
                        pass

            # 5b. Documents are relevant — ground answer in them
            if doc_context and best_score >= DOC_RELEVANCE_THRESHOLD:
                llm_t0 = time.perf_counter()
                response, pt, ct = self._answer_from_docs(eq, doc_context)
                _add_tokens(pt, ct)
                if metrics is not None:
                    metrics.llm_latency_ms = round(
                        (time.perf_counter() - llm_t0) * 1000, 1
                    )

                # ── Groundedness check ─────────────────────────────────────────
                gnd = self._check_groundedness(response, doc_context)
                if metrics is not None:
                    metrics.groundedness_score = gnd.score
                    metrics.supported_claims = gnd.supported_claims
                    metrics.unsupported_claims = gnd.unsupported_claims
                    metrics.groundedness_verdict = gnd.verdict
                    metrics.groundedness_latency_ms = gnd.latency_ms

                # Too unreliable → substitute fallback
                if gnd.score < GROUNDEDNESS_MIN_SCORE:
                    logger.info(
                        "Groundedness %.0f%% below threshold — returning fallback",
                        gnd.score * 100,
                    )
                    cache.set(
                        eq,
                        embedding,
                        GROUNDEDNESS_FALLBACK,
                        doc_sources,
                        scope=cache_scope,
                    )
                    if metrics is not None:
                        metrics.tool_used = "docs"
                    return (
                        GROUNDEDNESS_FALLBACK,
                        doc_sources,
                        "docs",
                        gnd,
                        rewritten_query,
                    )

                if not knowledge_only and self._is_vague(response):
                    web_ctx = self.tools.web_search(eq)
                    if web_ctx:
                        llm_t0 = time.perf_counter()
                        response, pt, ct = self._answer_with_web(eq, web_ctx)
                        _add_tokens(pt, ct)
                        if metrics is not None:
                            metrics.llm_latency_ms += round(
                                (time.perf_counter() - llm_t0) * 1000, 1
                            )
                        doc_sources.append("web (enriched)")
                    else:
                        llm_t0 = time.perf_counter()
                        response, pt, ct = self._answer_from_knowledge(eq)
                        _add_tokens(pt, ct)
                        if metrics is not None:
                            metrics.llm_latency_ms += round(
                                (time.perf_counter() - llm_t0) * 1000, 1
                            )

                cache.set(eq, embedding, response, doc_sources, scope=cache_scope)
                if metrics is not None:
                    metrics.tool_used = "docs"
                return response, doc_sources, "docs", gnd, rewritten_query

            if knowledge_only:
                return "The document does not provide enough relevant evidence to answer this question.", [], "docs", None, rewritten_query

            # 5c. Off-topic → LLM general knowledge
            llm_t0 = time.perf_counter()
            llm_response, pt, ct = self._answer_from_knowledge(eq)
            _add_tokens(pt, ct)
            if metrics is not None:
                metrics.llm_latency_ms = round((time.perf_counter() - llm_t0) * 1000, 1)

            if not knowledge_only and self._is_vague(llm_response):
                web_ctx = self.tools.web_search(eq)
                if web_ctx:
                    llm_t0 = time.perf_counter()
                    llm_response, pt, ct = self._answer_with_web(eq, web_ctx)
                    _add_tokens(pt, ct)
                    if metrics is not None:
                        metrics.llm_latency_ms += round(
                            (time.perf_counter() - llm_t0) * 1000, 1
                        )
                    cache.set(
                        eq, embedding, llm_response, ["web"], scope=cache_scope
                    )
                    if metrics is not None:
                        metrics.tool_used = "llm+web"
                    return llm_response, ["web"], "llm+web", None, rewritten_query

            cache.set(eq, embedding, llm_response, [], scope=cache_scope)
            if metrics is not None:
                metrics.tool_used = "llm"
            return llm_response, [], "llm", None, rewritten_query

        if knowledge_only:
            return "Document knowledge is unavailable.", [], "docs", None, rewritten_query

        # 6. No index → LLM general knowledge
        llm_t0 = time.perf_counter()
        llm_response, pt, ct = self._answer_from_knowledge(eq)
        _add_tokens(pt, ct)
        if metrics is not None:
            metrics.llm_latency_ms = round((time.perf_counter() - llm_t0) * 1000, 1)
            metrics.mode = "fast"

        fallback_sources: list[str] = []
        tool_used = "llm"

        if not knowledge_only and self._is_vague(llm_response):
            web_ctx = self.tools.web_search(eq)
            if web_ctx:
                llm_t0 = time.perf_counter()
                llm_response, pt, ct = self._answer_with_web(eq, web_ctx)
                _add_tokens(pt, ct)
                if metrics is not None:
                    metrics.llm_latency_ms += round(
                        (time.perf_counter() - llm_t0) * 1000, 1
                    )
                fallback_sources = ["web"]
                tool_used = "llm+web"

        cache.set(eq, embedding, llm_response, fallback_sources, scope=cache_scope)
        if metrics is not None:
            metrics.tool_used = tool_used
        return llm_response, fallback_sources, tool_used, None, rewritten_query


# ── Response Formatter ─────────────────────────────────────────────────────────
def format_response(response: str, sources: list[str]) -> str:
    response = response.strip()
    response = re.sub(
        r"^(Answer|Response|Here's|Based on):\s*", "", response, flags=re.IGNORECASE
    )
    response = re.sub(r"\*\*([^*]+)\*\*", r"\1", response)
    response = re.sub(r"\n{3,}", "\n\n", response)
    if sources:
        response += f"  [Source: {', '.join(dict.fromkeys(sources))}]"
    return response


# ── CLI Entry Point ────────────────────────────────────────────────────────────
def main() -> None:
    initialize_settings()

    file_input = input(
        "Enter file paths (comma-separated, or press Enter to skip): "
    ).strip()
    file_paths = [
        f.strip()
        for f in file_input.split(",")
        if f.strip() and os.path.exists(f.strip())
    ]

    index = None
    if file_paths:
        print(f"\nProcessing {len(file_paths)} file(s)…")
        index = create_index(file_paths)
        if index is None:
            print("Warning: index creation failed. Falling back to LLM-only mode.\n")
    else:
        print("No files provided — running in LLM-only mode.\n")

    agent = Agent(index)

    while True:
        query = input("Q: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        t0 = time.time()
        response, sources, tool_used, _gnd, _rewritten = agent.run(query)
        elapsed = (time.time() - t0) * 1000

        print(f"\n{format_response(response, sources)}")
        print(f"[Tool: {tool_used} | {elapsed:.0f}ms]\n")


if __name__ == "__main__":
    main()
