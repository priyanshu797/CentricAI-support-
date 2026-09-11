from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import smtplib
import tempfile
import time
import traceback
import warnings
from collections.abc import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import cast

# Load .env before anything else so os.getenv() picks up all values
from dotenv import load_dotenv

load_dotenv()

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

import prompt
from app.error import (
    AppError,
    ErrorCode,
    flask_error_response,
    log_error,
    sse_error_frame,
)
from app.observability import obs
from app.mcp.client import get_mcp_client
from app.request_analysis import analyze_request
from app.request_context import trusted_context, trusted_session_name, register_document, require_document, owner_id

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose output from 3rd-party libraries that pollute the terminal.
# These loggers emit FutureWarnings, download progress, internal debug traces,
# and HTTP request logs that are irrelevant to the application operator.
_SILENT_LOGGERS = [
    # Hugging Face / Transformers
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    # LlamaIndex
    "llama_index",
    "llama_index.core",
    "llama_index.core.indices",
    "llama_index.core.storage",
    "llama_index.embeddings",
    "llama_index.embeddings.huggingface",
    # HTTP clients
    "httpx",
    "httpcore",
    "urllib3",
    "urllib3.connectionpool",
    # NLTK
    "nltk",
    # PyTorch / Torchvision (loaded by sentence-transformers)
    "torch",
    "torchvision",
    "faiss",
    # pymongo internal connection pool / heartbeat chatter
    "pymongo",
]
# NOTE: werkzeug is intentionally NOT silenced — it prints the
# "Running on http://..." startup banner and per-request logs.

for _lib in _SILENT_LOGGERS:
    logging.getLogger(_lib).setLevel(logging.ERROR)

# Also suppress Python warnings from transformers and related libs
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sentence_transformers"
)
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # silences tokenizer fork warning

# ── Optional heavy dependencies ────────────────────────────────────────────────
try:
    import nltk

    def _safe_nltk_download() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    _safe_nltk_download()
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import importlib.util

    RAG_AVAILABLE = (
        importlib.util.find_spec("llama_index") is not None
        and importlib.util.find_spec("llama_index.embeddings.huggingface") is not None
    )
    if RAG_AVAILABLE:
        logger.info("LlamaIndex + HuggingFace embedding available")
except Exception as exc:  # noqa: BLE001
    RAG_AVAILABLE = False
    logger.warning("RAG availability check failed: %s", exc)

try:
    from groq import Groq
    from groq.types.chat import ChatCompletionMessageParam

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq SDK not installed")

# ── LLM fallback (Groq → Gemini) ──────────────────────────────────────────────
try:
    from app.llm_fallback import GEMINI_MODEL as _GEMINI_MODEL  # noqa: I001
    from app.llm_fallback import GROQ_MODEL as _GROQ_MODEL
    from app.llm_fallback import llm_chat as _llm_chat

    LLM_FALLBACK_AVAILABLE = True
    logger.info("LLM fallback module loaded (primary: Groq, fallback: Gemini)")
except Exception as exc:  # noqa: BLE001
    LLM_FALLBACK_AVAILABLE = False
    _GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    _GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    logger.warning("LLM fallback module unavailable: %s", exc)

# ── Agentic RAG imports ────────────────────────────────────────────────────────
try:
    from app.content_retrieval import (  # noqa: I001 — grouped by logical purpose
        DOC_RELEVANCE_THRESHOLD,  # noqa: F401 — re-exported for tests
        Agent,
        GROQ_API_KEY as RAG_GROQ_API_KEY,  # noqa: F401
        MultiLevelCache,  # noqa: F401
        QueryClassifier,  # noqa: F401
        Tools,  # noqa: F401
        format_response as rag_format_response,  # noqa: F401
        initialize_settings as rag_initialize_settings,
    )
    from app.indexing_service import DocumentIndexingService

    AGENTIC_RAG_AVAILABLE = True
    logger.info("Agentic RAG module loaded")
except Exception as exc:  # noqa: BLE001 — intentional broad catch at import time
    AGENTIC_RAG_AVAILABLE = False
    logger.warning("Agentic RAG module unavailable: %s", exc)

# ── Metrics store ──────────────────────────────────────────────────────────────
try:
    from app.content_retrieval import GroundednessResult  # noqa: F401 – type hint only
    from app.rag_metrics import RagMetrics
    from app.rag_metrics import get_store as _get_metrics_store

    _metrics_store = _get_metrics_store()
    METRICS_AVAILABLE = True
    logger.info("RAG metrics store ready")
except Exception as exc:  # noqa: BLE001
    METRICS_AVAILABLE = False
    _metrics_store = None  # type: ignore[assignment]
    logger.warning("RAG metrics store unavailable: %s", exc)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RAG_CACHE_DIR = os.path.join(BASE_DIR, "rag_cache")

ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "doc",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "gif",
    "tiff",
    "csv",
    "json",
    "xml",
    "html",
    "htm",
    "md",
    "mp3",
    "wav",
    "mp4",
}

# ── API key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)
CORS(app)
import secrets
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

for _d in [UPLOAD_FOLDER, RAG_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── MongoDB conversation store ─────────────────────────────────────────────────
from app.mongo_store import ConversationStore

_MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_MONGO_DB: str = os.getenv("MONGO_DB", "centrix_support")

_conversation_store: ConversationStore | None = None
try:
    _conversation_store = ConversationStore(_MONGO_URI, _MONGO_DB)
    if _conversation_store.ping():
        logger.info("MongoDB connected — conversations stored in '%s'", _MONGO_DB)
    else:
        logger.warning("MongoDB ping failed — conversations will not be persisted")
        _conversation_store = None
except Exception as _mongo_exc:  # noqa: BLE001
    logger.warning(
        "MongoDB unavailable (%s) — conversations will not be persisted", _mongo_exc
    )
    _conversation_store = None

# ── Mood / Task stores (share the same MongoDB client / db) ───────────────────
from app.mood_store import MoodStore, TaskStore

_mood_store: MoodStore | None = None
_task_store: TaskStore | None = None
if _conversation_store is not None:
    try:
        _db = _conversation_store._db  # reuse existing connection — no new client
        _mood_store = MoodStore(_db)
        _task_store = TaskStore(_db)
        logger.info("Mood and Task stores ready")
    except Exception as _ms_exc:  # noqa: BLE001
        logger.warning("Mood/Task stores init failed: %s", _ms_exc)

# ── Groq client ────────────────────────────────────────────────────────────────
groq_client = None
if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialised")
    except Exception as exc:  # noqa: BLE001
        logger.error("Groq init failed: %s", exc)

# Email config
SENDER_EMAIL: str | None = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD: str | None = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL: str | None = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)

# ── Initialise RAG settings once at startup ────────────────────────────────────
_rag_settings_initialised = False
if AGENTIC_RAG_AVAILABLE and RAG_AVAILABLE:
    try:
        rag_initialize_settings()
        _rag_settings_initialised = True
        logger.info("Agentic RAG settings initialised")
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG settings init failed: %s", exc)


_indexing_service: DocumentIndexingService | None = None
if _rag_settings_initialised:
    try:
        _indexing_service = DocumentIndexingService(
            os.path.join(BASE_DIR, "rag_storage")
        )
        logger.info("Asynchronous document indexing service ready")
    except Exception as exc:  # noqa: BLE001 — optional service startup
        logger.error("Document indexing service init failed: %s", exc)


# ── Agentic AI — import tools and runner ──────────────────────────────────────
from app.agent_runner import AgentRunner  # noqa: I001 — runner before tools intentional
from app.agent_tools import init_tool_services


# ── Language / emotion helpers ─────────────────────────────────────────────────
def detect_language(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    hindi_pattern = re.compile(r"[\u0900-\u097F]")
    hinglish_words = {
        "kya",
        "hai",
        "hoon",
        "hain",
        "aur",
        "ki",
        "ka",
        "ke",
        "ko",
        "mein",
        "main",
        "se",
        "par",
        "theek",
        "nahi",
        "haan",
        "bahut",
        "bohot",
    }
    english_words = {
        "the",
        "is",
        "are",
        "was",
        "were",
        "what",
        "when",
        "where",
        "how",
        "feel",
        "feeling",
        "help",
        "need",
    }
    hindi_chars = len(hindi_pattern.findall(text))
    total_chars = len(re.sub(r"\s", "", text))
    if total_chars and (hindi_chars / total_chars) > 0.3:
        return "hindi"
    words = set(cleaned.split())
    h = len(words & hinglish_words)
    e = len(words & english_words)
    if h and e:
        return "hinglish"
    if h > e:
        return "hinglish"
    if hindi_chars:
        return "hindi"
    return "english"


def detect_emotion(text: str) -> tuple[str, float]:
    tl = text.lower()
    groups: dict[str, list[str]] = {
        "stressed": [
            "stressed",
            "stress",
            "stressful",
            "pressure",
            "tense",
            "tension",
            "overwhelmed",
            "burnout",
            "burnt out",
            "exhausted",
            "drained",
            "overloaded",
            "overworked",
            "swamped",
            "can't cope",
            "cannot cope",
        ],
        "anxious": [
            "anxious",
            "anxiety",
            "panic",
            "panicking",
            "worried",
            "worrying",
            "scared",
            "fearful",
            "fear",
            "nervous",
            "nervous wreck",
            "on edge",
            "restless",
            "uneasy",
            "apprehensive",
            "dread",
            "dreading",
            "can't concentrate",
            "cannot concentrate",
            "mind racing",
        ],
        "sad": [
            "sad",
            "depressed",
            "depression",
            "lonely",
            "loneliness",
            "hopeless",
            "crying",
            "cry",
            "tears",
            "heartbroken",
            "grief",
            "grieving",
            "miserable",
            "unhappy",
            "down",
            "blue",
            "empty",
            "numb",
            "lost",
            "worthless",
            "helpless",
        ],
        "angry": [
            "angry",
            "anger",
            "rage",
            "furious",
            "frustrated",
            "frustration",
            "annoyed",
            "irritated",
            "irritable",
            "mad",
            "upset",
            "resentful",
            "bitter",
            "hate",
            "disgusted",
        ],
        "overwhelmed": [
            "overwhelmed",
            "too much",
            "can't handle",
            "cannot handle",
            "falling apart",
            "breaking down",
            "drowning",
            "no way out",
        ],
        "happy": [
            "happy",
            "happiness",
            "joy",
            "joyful",
            "excited",
            "great",
            "wonderful",
            "amazing",
            "fantastic",
            "grateful",
            "thankful",
            "peaceful",
            "calm",
            "content",
            "positive",
            "good mood",
        ],
    }
    # Weight multi-word phrases more heavily than single keywords
    scores: dict[str, float] = {}
    for emotion, kws in groups.items():
        score = 0.0
        for kw in kws:
            if kw in tl:
                score += len(kw.split())  # multi-word phrases score higher
        scores[emotion] = score

    if max(scores.values()) == 0:
        return "neutral", 0.3
    top = max(scores, key=scores.get)  # type: ignore[arg-type]
    # Normalise: max single keyword = 1 word → score of 1 → conf 0.2
    # 5 keywords matched → conf 1.0. Scale is per-emotion match depth.
    raw_conf = min(scores[top] / 5.0, 1.0)
    # Minimum confidence of 0.45 when at least one keyword matched
    confidence = max(raw_conf, 0.45)
    return top, round(confidence, 2)


def emotion_intensity(confidence: float) -> str:
    """Map a confidence score to a human-readable intensity label."""
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


# Contextual action suggestions per emotion
_EMOTION_SUGGESTIONS: dict[str, list[dict[str, str]]] = {
    "stressed": [
        {
            "label": "Try a 2-minute breathing exercise",
            "prompt": "Guide me through a quick 2-minute breathing exercise to relieve stress.",
        },
        {
            "label": "Create a small self-care plan",
            "prompt": "Help me create a simple self-care plan to manage my stress.",
        },
        {
            "label": "Talk about what's causing the stress",
            "prompt": "I want to talk about what's been causing my stress.",
        },
    ],
    "anxious": [
        {
            "label": "Try a grounding exercise",
            "prompt": "Guide me through a grounding exercise to calm my anxiety.",
        },
        {
            "label": "Learn about managing anxiety",
            "prompt": "What are some practical ways to manage anxiety day to day?",
        },
        {
            "label": "Talk through what I'm worried about",
            "prompt": "I'd like to talk through what I'm feeling anxious about.",
        },
    ],
    "sad": [
        {
            "label": "Explore mood-lifting activities",
            "prompt": "What are some gentle activities that can help lift my mood?",
        },
        {
            "label": "Practice a self-compassion exercise",
            "prompt": "Guide me through a self-compassion or kindness exercise.",
        },
        {
            "label": "Just talk — I need to be heard",
            "prompt": "I just need someone to talk to about how I'm feeling.",
        },
    ],
    "angry": [
        {
            "label": "Try a calming technique",
            "prompt": "Give me a calming technique to help with my anger right now.",
        },
        {
            "label": "Understand what's behind the anger",
            "prompt": "Help me understand and process what's driving my anger.",
        },
        {
            "label": "Talk it through",
            "prompt": "I want to talk about why I'm feeling so angry.",
        },
    ],
    "overwhelmed": [
        {
            "label": "Break things into small steps",
            "prompt": "Help me break down what I'm dealing with into small manageable steps.",
        },
        {
            "label": "Try a 2-minute breathing exercise",
            "prompt": "Guide me through a quick 2-minute breathing exercise.",
        },
        {
            "label": "Talk about what's too much right now",
            "prompt": "I want to talk about everything that feels overwhelming.",
        },
    ],
    "happy": [
        {
            "label": "Build on this positive energy",
            "prompt": "How can I build on this positive mood and keep it going?",
        },
        {
            "label": "Set a wellness intention for today",
            "prompt": "Help me set a positive wellness intention for today.",
        },
        {
            "label": "Share what's going well",
            "prompt": "I'd like to share what's been making me feel good lately.",
        },
    ],
    "neutral": [
        {
            "label": "Check in on your wellbeing",
            "prompt": "Can you do a quick wellbeing check-in with me?",
        },
        {
            "label": "Explore self-care ideas",
            "prompt": "What are some simple self-care ideas I can try today?",
        },
        {
            "label": "Start a relaxation exercise",
            "prompt": "Guide me through a short relaxation exercise.",
        },
    ],
}


def detect_high_risk(text: str) -> bool:
    crisis = ["suicidal", "kill myself", "want to die", "end it all", "suicide"]
    return any(kw in text.lower() for kw in crisis)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_conversation(session_name: str) -> list[dict[str, str]]:
    """Load conversation history from MongoDB (falls back to empty list)."""
    if _conversation_store is not None:
        return _conversation_store.load(session_name)
    return []


def _save_conversation(session_name: str, history: list[dict]) -> None:
    """Persist full conversation history to MongoDB."""
    if _conversation_store is not None:
        _conversation_store.save(session_name, history)
    else:
        logger.warning(
            "MongoDB unavailable — conversation not saved for session %r", session_name
        )


# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/learn_more")
def learn_more():
    return render_template("learn_more.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


@app.route("/about")
def about():
    from datetime import datetime, timezone

    return render_template("about.html", year=datetime.now(tz=timezone.utc).year)


@app.route("/resource")
def resource():
    return render_template("resource.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        files = request.files.getlist("file")
        if not files:
            return jsonify({"success": False, "error": "No files provided"}), 400
        if _indexing_service is None:
            return (
                jsonify({"success": False, "error": "RAG pipeline is unavailable"}),
                503,
            )

        saved_files: list[str] = []
        file_info: list[dict] = []
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                logger.warning("Rejected: %s", file.filename)
                continue
            original_name = secure_filename(file.filename)

            # ── Save to a temp path first, compute SHA-256, then decide ──────
            with tempfile.NamedTemporaryFile(
                delete=False, dir=app.config["UPLOAD_FOLDER"], suffix="_tmp"
            ) as _tmp:
                file.save(_tmp)
                tmp_path = _tmp.name

            # Compute SHA-256 of the uploaded bytes
            _digest = hashlib.sha256()
            with open(tmp_path, "rb") as _fh:
                for _blk in iter(lambda: _fh.read(1 << 20), b""):
                    _digest.update(_blk)
            content_hash = _digest.hexdigest()

            # Check whether this exact content is already on disk
            canonical_name = (
                f"{content_hash}{os.path.splitext(original_name)[1].lower()}"
            )
            canonical_path = os.path.join(app.config["UPLOAD_FOLDER"], canonical_name)

            if os.path.exists(canonical_path):
                # Exact duplicate — discard the temp file, reuse existing
                os.remove(tmp_path)
                filepath = canonical_path
                size = os.path.getsize(filepath)
                logger.info("Dedup hit — reusing %s", canonical_name)
            else:
                # New content — rename temp file to canonical name
                os.replace(tmp_path, canonical_path)
                filepath = canonical_path
                size = os.path.getsize(filepath)
                logger.info("Uploaded: %s (%d bytes)", canonical_name, size)

            saved_files.append(filepath)
            file_info.append(
                {
                    "name": canonical_name,
                    "original_name": original_name,
                    "path": filepath,
                    "url": url_for("uploaded_file", filename=canonical_name),
                    "size": size,
                    "type": os.path.splitext(canonical_name)[1].lower(),
                }
            )

        if not saved_files:
            return jsonify({"success": False, "error": "No valid files uploaded"}), 400

        document_id = _indexing_service.submit(
            saved_files, [item["original_name"] for item in file_info]
        )
        register_document(document_id, [os.path.basename(path) for path in saved_files])

        return (
            jsonify(
                {
                    "success": True,
                    "filepaths": saved_files,
                    "filepath": saved_files[0],
                    "count": len(saved_files),
                    "files": file_info,
                    "document_id": document_id,
                    "status": "Uploading",
                    "progress": 0,
                    "message": "Upload received. Background indexing started.",
                }
            ),
            202,
        )
    except Exception as exc:  # noqa: BLE001
        log_error("upload", exc)
        return flask_error_response(exc)


@app.route("/documents/<document_id>/status", methods=["GET"])
def document_status(document_id: str):
    require_document(document_id)
    if _indexing_service is None:
        return jsonify({"success": False, "error": "Indexing unavailable"}), 503
    status = _indexing_service.status(document_id)
    if status is None:
        return jsonify({"success": False, "error": "Document job not found"}), 404
    return jsonify({"success": True, **status})


# ── Shared helpers used by both /search and /search/stream ────────────────────
_CRISIS_MSGS: dict[str, str] = {
    "english": (
        "I can hear that you're going through something very difficult. "
        "You're not alone.\n\nIndia: +91 9152987821\nInternational: 988\n\nHelp is available 24/7."
    ),
    "hindi": "आप कुछ बहुत कठिन से गुज़र रहे हैं। आप अकेले नहीं हैं।\n\nभारत: +91 9152987821",
    "hinglish": "Aap akele nahi hain.\n\nIndia: +91 9152987821\nInternational: 988",
}
_EMOTION_EMOJI: dict[str, str] = {
    "stressed": "😤",
    "overwhelmed": "😰",
    "sad": "😢",
    "angry": "😠",
    "anxious": "😨",
    "neutral": "😌",
    "happy": "😊",
}


def _indexed_agent(document_id: str) -> Agent | None:
    """Return an agent over already-indexed vectors; never process source files."""
    if not document_id or _indexing_service is None:
        return None
    prepared = _indexing_service.prepared_indexes(document_id)
    if prepared is None:
        return None
    indexes, checksums = prepared
    return Agent(
        indexes,
        cache_scope="response-format-v2:documents:" + "|".join(sorted(checksums)),
    )


# All injected callables now exist, so initialise the tool registry once.
try:
    init_tool_services(
        mood_store=_mood_store,
        task_store=_task_store,
        rag_agent_factory=_indexed_agent,
        detect_emotion_fn=detect_emotion,
        llm_chat_fn=_llm_chat if LLM_FALLBACK_AVAILABLE else None,
        conversation_store=_conversation_store,
    )
    logger.info("Agent tool services fully wired")
except Exception as _agent_wire_exc:  # noqa: BLE001
    logger.warning("Agent tool services wiring failed: %s", _agent_wire_exc)


# ── MCP — build client and initialise ─────────────────────────────────────────
# MCPClient() auto-discovers in-process servers via _configured_servers().
# Services (mood_store, task_store, rag_agent_factory) are already accessible
# through agent_tools module globals set by init_tool_services() above.
try:
    from app.mcp import set_mcp_client
    from app.mcp.client import MCPClient as _MCPClient

    _built_client = _MCPClient()   # uses _configured_servers() by default
    _built_client.initialize()
    set_mcp_client(_built_client)
    logger.info(
        "MCP client ready — %d tools across 3 servers",
        _built_client.health()["total_tools"],
    )
except Exception as _mcp_init_exc:  # noqa: BLE001
    logger.warning(
        "MCP initialisation failed — agent will use _NullMCPClient: %s",
        _mcp_init_exc,
    )


_mcp_client = get_mcp_client()


@app.route("/health/mcp", methods=["GET"])
def mcp_health():
    # Token-gate: if MCP_HEALTH_TOKEN is set, require Authorization header.
    # If the env var is empty/unset, the endpoint is open (useful locally).
    import hmac
    token = os.getenv("MCP_HEALTH_TOKEN", "").strip()
    if token:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + token):
            return jsonify({"error": "Not found"}), 404
    if _mcp_client is None:
        return jsonify({"status": "unavailable", "error": "MCP client not initialised"}), 503
    health = _mcp_client.health()
    return jsonify(health), 200 if health["status"] == "ok" else 503


def _used_uploaded_documents(tool_used: str, sources: list[str]) -> bool:
    if tool_used == "docs" or tool_used.startswith("docs+"):
        return True
    if tool_used == "cache":
        return any(not source.lower().startswith("web") for source in sources)
    return False


# ── Provider logging helper ────────────────────────────────────────────────────
_PROVIDER_LABEL: dict[str, str] = {
    "groq": f"{_GROQ_MODEL} (Groq)",
    "gemini": f"{_GEMINI_MODEL} (Google)",
}


def _log_provider(context: str, provider: str, query_snippet: str = "") -> None:
    """Always log which model answered a query — INFO level, always visible."""
    label = _PROVIDER_LABEL.get(provider, provider)
    snippet = f" | query={query_snippet!r}" if query_snippet else ""
    logger.info("🤖  [%s] answered by → %s%s", context, label, snippet)


def _enrich_with_history(
    response: str,
    question: str,
    conv_history: list[dict],
    tool_used: str,
) -> str:
    """Optionally refine response using recent conversation context."""
    if not (
        conv_history
        and LLM_FALLBACK_AVAILABLE
        and tool_used not in ("cache", "conversational", "calculator")
    ):
        return response
    try:
        hist_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in conv_history[-6:]
        )
        enrich_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Preserve accuracy while enriching "
                    "context. Keep the answer in readable Markdown with short paragraphs, "
                    "clear section headings, and one bullet per line when useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{hist_text}\n\n"
                    f"Current question: {question}\n\nBest answer so far:\n{response}\n\n"
                    "Refine using conversation context if helpful. Preserve Markdown "
                    "headings, blank lines, short paragraphs, and one bullet per line. "
                    "For substantial answers, retain or add a concise Summary section. "
                    "If already complete, return it unchanged."
                ),
            },
        ]
        enriched, _, _, provider = _llm_chat(  # type: ignore[misc]
            enrich_messages, temperature=0.4, max_tokens=1000
        )
        _log_provider("_enrich_with_history", provider, question[:60])
        return enriched or response
    except (OSError, RuntimeError) as exc:
        logger.warning("History enrichment failed: %s", exc)
        return response


def _save_agent_metrics(question, session_name, lang, emotion, insights):
    if not METRICS_AVAILABLE or _metrics_store is None:
        return
    try:
        rag, llm = insights["rag"], insights["llm"]
        metrics = RagMetrics(query=question, session_name=session_name, language=lang,
            emotion_detected=emotion, model_name=llm.get("model") or "",
            tool_used="docs" if rag["used"] else "agent", cache_hit=insights["cache"]["hit"],
            num_docs_retrieved=rag.get("chunks_retrieved") or 0,
            total_latency_ms=insights["total_latency_ms"], llm_latency_ms=llm.get("latency_ms") or 0)
        metrics.finalise()
        _metrics_store.save(metrics)
    except Exception as exc:
        logger.warning("Agent metrics unavailable (%s)", type(exc).__name__)


# ── Main search endpoint ───────────────────────────────────────────────────────
@app.route("/search", methods=["POST"])
def search():
    try:
        t0 = time.time()
        data = request.get_json(force=True)
        question: str = data.get("query") or data.get("question", "")
        session_name: str = trusted_session_name(data.get("session_name", "default_session"))
        document_id: str = require_document(data.get("document_id", ""))

        if not question:
            return jsonify({"success": False, "error": "Query required"}), 400

        logger.info("[search] Q=%r session=%s", question[:100], session_name)

        # Compute emotion before entering the trace context
        lang = detect_language(question)
        emotion, confidence = detect_emotion(question)
        _e_t0 = time.time()
        emotion_emoji = _EMOTION_EMOJI.get(emotion, "💭")
        intensity = emotion_intensity(confidence)
        suggestions = _EMOTION_SUGGESTIONS.get(emotion, _EMOTION_SUGGESTIONS["neutral"])
        _emotion_ms = round((time.time() - _e_t0) * 1000, 1)

        # Store final response so it can be set as trace output
        _final_response: str = ""
        _result_json: dict = {}

        with obs.start_trace(
            "search",
            session_id=session_name,
            query=question,
            tags=["search", "non-streaming"],
        ):

            with obs.emotion_span(emotion, confidence, intensity, _emotion_ms):
                pass

            if detect_high_risk(question):
                _final_response = _CRISIS_MSGS.get(lang, _CRISIS_MSGS["english"])
                obs.set_trace_output(_final_response)
                return jsonify(
                    {
                        "success": True,
                        "emotion_detected": emotion,
                        "language": lang,
                        "response": _final_response,
                    }
                )

            conv_history = _load_conversation(session_name)
            indexed_agent = _indexed_agent(document_id)
            if document_id and indexed_agent is None:
                return (
                    jsonify(
                        {"success": False, "error": "Document indexing is not complete"}
                    ),
                    409,
                )

            runner = AgentRunner(_mcp_client)
            response = "".join(runner.stream(question, session_name, document_id, conv_history,
                request_analysis=analyze_request(question), user_id=owner_id()))
            insights = runner.insights
            insights["emotion"] = {"label": emotion, "confidence": confidence}
            conv_history.extend([{"role": "user", "content": question},
                                 {"role": "assistant", "content": response}])
            with obs.span("MongoDB", metadata={"operation": "save_conversation"}):
                _save_conversation(session_name, conv_history)
            obs.set_trace_output(response)
            _save_agent_metrics(question, session_name, lang, emotion, insights)
            return jsonify({"success": True, "response": response, "sources": runner.sources,
                "emotion_detected": emotion, "emotion_emoji": emotion_emoji,
                "emotion_confidence": round(confidence, 2), "emotion_intensity": intensity,
                "emotion_suggestions": suggestions, "language": lang,
                "model_used": insights["llm"]["model"], "rag_used": insights["rag"]["used"],
                "tool_used": "agent", "groundedness": insights["rag"],
                "response_insights": insights, "time": round(time.time() - t0, 2)})

    except Exception as exc:  # noqa: BLE001
        log_error("search", exc)
        return flask_error_response(exc, {"response": "An error occurred."})


# ── Streaming search endpoint ─────────────────────────────────────────────────
@app.route("/search/stream", methods=["GET"])
def search_stream():
    """
    SSE endpoint — now routes every non-crisis request through the
    real tool-calling AgentRunner.

    SSE frames (in order):
      meta        — emotion, language, tool labels
      chunk       — one or more text tokens
      done        — sources list
      insights    — Response Inspector telemetry (safe to ignore on old clients)
    """
    question = request.args.get("q", "").strip()
    session_name = trusted_session_name(request.args.get("session_name", "default_session"))
    document_id = require_document(request.args.get("document_id", "").strip())
    request_analysis = analyze_request(question)
    trusted_user = owner_id()

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate() -> Generator[str, None, None]:
        if not question:
            yield _sse({"type": "error", "message": "Query required"})
            return

        # ── Emotion + language (cheap — always run first) ─────────────────────
        lang = detect_language(question)
        emotion, conf = detect_emotion(question)
        emotion_emoji = _EMOTION_EMOJI.get(emotion, "💭")
        intensity = emotion_intensity(conf)
        suggestions = _EMOTION_SUGGESTIONS.get(emotion, _EMOTION_SUGGESTIONS["neutral"])

        # ── Crisis bypass — skip agent entirely ───────────────────────────────
        if detect_high_risk(question):
            msg = _CRISIS_MSGS.get(lang, _CRISIS_MSGS["english"])
            yield _sse(
                {
                    "type": "meta",
                    "emotion_detected": emotion,
                    "emotion_emoji": emotion_emoji,
                    "emotion_confidence": round(conf, 2),
                    "emotion_intensity": intensity,
                    "emotion_suggestions": suggestions,
                    "language": lang,
                    "tool_used": "crisis",
                    "rag_used": False,
                }
            )
            yield _sse({"type": "chunk", "text": msg})
            yield _sse({"type": "done", "sources": []})
            return

        # ── Document index guard ──────────────────────────────────────────────
        if document_id and _indexed_agent(document_id) is None:
            yield _sse(
                {"type": "error", "message": "Document indexing is not complete"}
            )
            return

        # ── Emit meta frame before agent starts so UI shows emotion pill ──────
        yield _sse(
            {
                "type": "meta",
                "emotion_detected": emotion,
                "emotion_emoji": emotion_emoji,
                "emotion_confidence": round(conf, 2),
                "emotion_intensity": intensity,
                "emotion_suggestions": suggestions,
                "language": lang,
                "tool_used": "agent",
                "rag_used": bool(document_id),
            }
        )

        # ── Agentic loop ──────────────────────────────────────────────────────
        _t0 = time.time()
        conv_history = _load_conversation(session_name)
        full_response = ""

        # Keepalive ping — tells the browser the connection is alive while
        # the agent is planning and executing tools (can take 5–15 seconds
        # for multi-tool chains). Without this the browser closes the SSE
        # connection and shows "No response received."
        yield _sse({"type": "ping"})

        with obs.start_trace(
            "agent-stream",
            session_id=session_name,
            query=question,
            tags=["agent", "streaming"],
        ):
            try:
                runner = AgentRunner(_mcp_client)
                for chunk in runner.stream(
                    question=question,
                    session_name=session_name,
                    document_id=document_id,
                    conv_history=conv_history,
                    request_analysis=request_analysis,
                    user_id=trusted_user,
                ):
                    if chunk:
                        full_response += chunk
                        yield _sse({"type": "chunk", "text": chunk})

                insights = runner.insights
                insights["emotion"] = {"label": emotion, "confidence": conf}

            except Exception as exc:  # noqa: BLE001
                log_error("search/stream[agent]", exc)
                yield sse_error_frame(exc)
                return

            # Guard: if agent produced no text, emit a fallback message so
            # the frontend never hits the "No response received" branch.
            if not full_response.strip():
                fallback = "I wasn't able to get a response right now. Please try again."
                full_response = fallback
                yield _sse({"type": "chunk", "text": fallback})

            # ── Persist conversation to MongoDB ───────────────────────────────
            conv_history.append({"role": "user", "content": question})
            conv_history.append({"role": "assistant", "content": full_response})
            with obs.span("MongoDB", metadata={"operation": "save_conversation"}):
                _save_conversation(session_name, conv_history)

            # ── RAG metrics (if RAG tool was actually used) ───────────────────
            rag_info = insights.get("rag", {})
            if (
                rag_info.get("used")
                and METRICS_AVAILABLE
                and _metrics_store is not None
            ):
                try:
                    from app.rag_metrics import RagMetrics

                    _sm = RagMetrics(
                        query=question,
                        session_name=session_name,
                        language=lang,
                        emotion_detected=emotion,
                        model_name=_GROQ_MODEL,
                        tool_used="docs",
                        cache_hit=insights.get("cache", {}).get("hit", False),
                        num_docs_retrieved=rag_info.get("chunks_retrieved", 0),
                        total_latency_ms=round((time.time() - _t0) * 1000, 1),
                    )
                    _sm.finalise()
                    _metrics_store.save(_sm)
                except Exception as _me:  # noqa: BLE001
                    logger.warning("Agent stream metrics save failed: %s", _me)

            obs.set_trace_output(full_response)

        sources = runner.sources

        yield _sse({"type": "done", "sources": sources})

        # ── Response Inspector telemetry (new frame type — old clients ignore) ─
        yield _sse({"type": "insights", "data": insights})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Emotion Analysis endpoint ─────────────────────────────────────────────────
@app.route("/emotion/analyze", methods=["POST"])
def emotion_analyze():
    """
    Accepts a user message and returns a rich emotion analysis:
      - emotion, confidence, intensity (Low/Medium/High)
      - a short empathetic acknowledgement from the LLM
      - contextual action suggestions
    """
    try:
        data = request.get_json(force=True)
        text: str = (data.get("text") or "").strip()
        if not text:
            return jsonify({"success": False, "error": "text field required"}), 400

        emotion, confidence = detect_emotion(text)
        intensity = emotion_intensity(confidence)
        suggestions = _EMOTION_SUGGESTIONS.get(emotion, _EMOTION_SUGGESTIONS["neutral"])
        emoji = _EMOTION_EMOJI.get(emotion, "💭")

        # LLM-powered empathetic acknowledgement (1–2 sentences, no list)
        acknowledgement = ""
        if LLM_FALLBACK_AVAILABLE:
            try:
                ack_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a warm, empathetic mental wellness assistant. "
                            "Write exactly 1-2 sentences that acknowledge the user's "
                            "emotional state with compassion. Do NOT offer advice, "
                            "lists, or questions. Keep it under 35 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f'The user said: "{text}"\n'
                            f"Detected emotion: {emotion} (intensity: {intensity}).\n"
                            "Write a brief, warm acknowledgement."
                        ),
                    },
                ]
                ack_text, _, _, ack_provider = _llm_chat(  # type: ignore[misc]
                    ack_messages, temperature=0.6, max_tokens=80
                )
                _log_provider("/emotion/analyze", ack_provider)
                acknowledgement = (ack_text or "").strip()
            except (OSError, RuntimeError) as exc:
                logger.warning("Emotion acknowledgement LLM call failed: %s", exc)
                acknowledgement = (
                    "I hear you, and I understand this is weighing on you. "
                    "You're not alone in feeling this way."
                )

        return jsonify(
            {
                "success": True,
                "emotion": emotion,
                "emoji": emoji,
                "confidence": confidence,
                "intensity": intensity,
                "acknowledgement": acknowledgement,
                "suggestions": suggestions,
            }
        )

    except Exception as exc:  # noqa: BLE001
        log_error("emotion/analyze", exc)
        return flask_error_response(exc)


# ── Agentic AI — Task and Mood REST endpoints ─────────────────────────────────


@app.route("/api/tasks", methods=["GET"])
def api_get_tasks():
    """GET /api/tasks?session_name=...&status=pending"""
    try:
        session_name = trusted_session_name(request.args.get("session_name", "default_session"))
        status = request.args.get("status", "pending")
        if _task_store is None:
            return jsonify({"success": False, "error": "Task store unavailable"}), 503
        tasks = _task_store.list_tasks(
            session_name, status=None if status == "all" else status
        )
        return jsonify({"success": True, "tasks": tasks, "count": len(tasks)})
    except Exception as exc:  # noqa: BLE001
        log_error("api/tasks GET", exc)
        return flask_error_response(exc)


@app.route("/api/tasks", methods=["POST"])
def api_create_task():
    """POST /api/tasks  body: {session_name, title, due_date?, priority?}"""
    try:
        data = request.get_json(force=True)
        session_name = trusted_session_name(data.get("session_name", "default_session"))
        title = (data.get("title") or "").strip()
        if not title:
            return jsonify({"success": False, "error": "title is required"}), 400
        if _task_store is None:
            return jsonify({"success": False, "error": "Task store unavailable"}), 503
        task = _task_store.add(
            session_name=session_name,
            title=title,
            due_date=data.get("due_date"),
            priority=data.get("priority", "medium"),
        )
        return jsonify({"success": True, "task": task}), 201
    except Exception as exc:  # noqa: BLE001
        log_error("api/tasks POST", exc)
        return flask_error_response(exc)


@app.route("/api/tasks/<task_id>/complete", methods=["POST"])
def api_complete_task(task_id: str):
    """POST /api/tasks/<id>/complete"""
    try:
        if _task_store is None:
            return jsonify({"success": False, "error": "Task store unavailable"}), 503
        ok = _task_store.complete(task_id, owner_prefix=owner_id())
        return jsonify({"success": ok, "task_id": task_id})
    except Exception as exc:  # noqa: BLE001
        log_error("api/tasks/complete", exc)
        return flask_error_response(exc)


@app.route("/api/mood", methods=["GET"])
def api_get_mood():
    """GET /api/mood?session_name=...&limit=10"""
    try:
        session_name = trusted_session_name(request.args.get("session_name", "default_session"))
        limit = min(int(request.args.get("limit", 10)), 50)
        if _mood_store is None:
            return jsonify({"success": False, "error": "Mood store unavailable"}), 503
        summary = _mood_store.summary(session_name, limit=limit)
        return jsonify({"success": True, **summary})
    except Exception as exc:  # noqa: BLE001
        log_error("api/mood GET", exc)
        return flask_error_response(exc)


@app.route("/api/mood", methods=["POST"])
def api_log_mood():
    """POST /api/mood  body: {session_name, emotion, score?, note?}"""
    try:
        data = request.get_json(force=True)
        session_name = trusted_session_name(data.get("session_name", "default_session"))
        emotion = (data.get("emotion") or "").strip()
        if not emotion:
            return jsonify({"success": False, "error": "emotion is required"}), 400
        if _mood_store is None:
            return jsonify({"success": False, "error": "Mood store unavailable"}), 503
        score = data.get("score")
        if score is not None:
            score = max(1, min(5, int(score)))
        entry_id = _mood_store.log(
            session_name=session_name,
            emotion=emotion,
            score=score,
            note=data.get("note", ""),
        )
        return jsonify({"success": True, "entry_id": entry_id, "emotion": emotion}), 201
    except Exception as exc:  # noqa: BLE001
        log_error("api/mood POST", exc)
        return flask_error_response(exc)


# ── Conversation management ────────────────────────────────────────────────────
@app.route("/conversation/history", methods=["POST"])
def get_conversation_history():
    try:
        data = request.get_json(force=True)
        session_name = trusted_session_name(data.get("session_name", "default_session"))
        history = _load_conversation(session_name)
        return jsonify({"success": True, "history": history, "count": len(history)})
    except Exception as exc:  # noqa: BLE001
        log_error("conversation/history", exc)
        return flask_error_response(exc)


@app.route("/conversation/clear", methods=["POST"])
def clear_conversation_history():
    try:
        data = request.get_json(force=True)
        session_name = trusted_session_name(data.get("session_name", "default_session"))
        if _conversation_store is not None:
            _conversation_store.clear(session_name)
        return jsonify({"success": True, "message": "Conversation history cleared"})
    except Exception as exc:  # noqa: BLE001
        log_error("conversation/clear", exc)
        return flask_error_response(exc)


# ── Email ──────────────────────────────────────────────────────────────────────
@app.route("/send_email", methods=["POST"])
def send_email():
    try:
        if not (SENDER_EMAIL and SENDER_PASSWORD):
            raise ValueError("Email configuration missing")
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = f"Contact Form: {name}"
        msg.attach(
            MIMEText(f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}", "plain")
        )
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        flash("Message sent successfully!", "success")
    except Exception as exc:  # noqa: BLE001
        log_error("send_email", exc)
        flash(
            "Your message could not be sent right now. Please try again later.", "error"
        )
    return redirect(url_for("contact"))


# ── RAG Evaluation Dashboard ───────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    """HTML dashboard showing per-query RAG metrics."""
    return render_template("dashboard.html")


@app.route("/dashboard/api/metrics", methods=["GET"])
def dashboard_api_metrics():
    """Return paginated metrics records + aggregate summary as JSON."""
    if not METRICS_AVAILABLE or _metrics_store is None:
        return jsonify({"success": False, "error": "Metrics store unavailable"}), 503
    try:
        limit = min(int(request.args.get("limit", 100)), 500)
        offset = max(int(request.args.get("offset", 0)), 0)
        session_filter = request.args.get("session") or None
        records = _metrics_store.fetch(
            limit=limit, offset=offset, session_name=session_filter
        )
        summary = _metrics_store.aggregate()
        total = _metrics_store.count()
        return jsonify(
            {
                "success": True,
                "records": records,
                "summary": summary,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )
    except Exception as exc:  # noqa: BLE001
        log_error("dashboard/metrics", exc)
        return flask_error_response(exc)


@app.route("/dashboard/api/clear", methods=["POST"])
def dashboard_api_clear():
    """Delete all stored metrics records."""
    if not METRICS_AVAILABLE or _metrics_store is None:
        return flask_error_response(
            AppError(ErrorCode.METRICS_FAILED, "Metrics store unavailable", 503)
        )
    try:
        deleted = _metrics_store.clear()
        return jsonify({"success": True, "deleted": deleted})
    except Exception as exc:  # noqa: BLE001
        log_error("dashboard/clear", exc)
        return flask_error_response(exc)


# ── Static / health ────────────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import session, abort
    allowed = {name for names in session.get("documents", {}).values() for name in names}
    if filename not in allowed:
        abort(403)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "online",
            "groq_available": groq_client is not None,
            "groq_sdk_installed": GROQ_AVAILABLE,
            "api_key_set": bool(GROQ_API_KEY),
            "rag_available": RAG_AVAILABLE,
            "agentic_rag_available": AGENTIC_RAG_AVAILABLE,
            "rag_settings_initialised": _rag_settings_initialised,
            "metrics_available": METRICS_AVAILABLE,
            "mongo_connected": _conversation_store is not None
            and _conversation_store.ping(),
            "mood_store_ready": _mood_store is not None,
            "task_store_ready": _task_store is not None,
            "agent_enabled": True,
            "supported_formats": sorted(ALLOWED_EXTENSIONS),
        }
    )


# ── Global Flask error handlers ───────────────────────────────────────────────


@app.errorhandler(400)
def bad_request(exc):
    log_error("http/400", exc)
    return flask_error_response(AppError(ErrorCode.BAD_REQUEST, str(exc), 400))


@app.errorhandler(404)
def not_found(exc):
    return (
        jsonify({"success": False, "error": "The requested page could not be found."}),
        404,
    )


@app.errorhandler(413)
def payload_too_large(exc):
    return flask_error_response(AppError(ErrorCode.FILE_TOO_LARGE, str(exc), 413))


@app.errorhandler(500)
def internal_server_error(exc):
    log_error("http/500", exc)
    return flask_error_response(AppError(ErrorCode.INTERNAL_ERROR, str(exc), 500))


if __name__ == "__main__":
    import atexit

    atexit.register(obs.flush)  # flush Langfuse events on clean shutdown
    print("CentrixSupport — Mental Health AI Chatbot (Agentic RAG)")
    server_host = os.environ.get("HOST", "0.0.0.0")
    server_port = int(os.environ.get("PORT", "8000"))
    browser_host = "127.0.0.1" if server_host in {"0.0.0.0", "::"} else server_host

    print(f"Server ready: http://{browser_host}:{server_port}", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    app.run(host=server_host, port=server_port, debug=False, use_reloader=False)
