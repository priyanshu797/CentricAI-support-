"""
app/error.py
─────────────────────────────────────────────────────────────────────────────
Centralised error handling for CentrixSupport.

• USER_MESSAGES  – maps every known failure category to a friendly string
                   that is safe to display directly in the UI.
• AppError       – lightweight exception that carries a category + HTTP code.
• handle()       – converts any exception into a (user_msg, http_status) pair.
• flask_error_response() – wraps handle() into a Flask JSON response.
• log_error()    – logs the real cause internally without leaking it to users.

Usage in a Flask route
──────────────────────
    from app.error import flask_error_response, AppError, log_error

    @app.route("/search", methods=["POST"])
    def search():
        try:
            ...
        except Exception as exc:
            log_error("search", exc)
            return flask_error_response(exc)
"""

from __future__ import annotations

import logging
import traceback
from enum import Enum
from typing import Any

from flask import jsonify

logger = logging.getLogger(__name__)

# ── Error categories ───────────────────────────────────────────────────────────


class ErrorCode(str, Enum):
    # LLM / AI provider
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_TIMEOUT = "llm_timeout"
    LLM_INVALID_KEY = "llm_invalid_key"

    # RAG / indexing
    RAG_UNAVAILABLE = "rag_unavailable"
    INDEXING_FAILED = "indexing_failed"
    INDEXING_INCOMPLETE = "indexing_incomplete"
    DOCUMENT_NOT_FOUND = "document_not_found"

    # File upload
    UPLOAD_FAILED = "upload_failed"
    FILE_TOO_LARGE = "file_too_large"
    FILE_TYPE_INVALID = "file_type_invalid"
    NO_FILE_PROVIDED = "no_file_provided"

    # Network / connectivity
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"

    # Database / storage
    STORAGE_FAILED = "storage_failed"
    METRICS_FAILED = "metrics_failed"

    # Email
    EMAIL_FAILED = "email_failed"

    # Auth / config
    CONFIG_MISSING = "config_missing"

    # General
    BAD_REQUEST = "bad_request"
    INTERNAL_ERROR = "internal_error"


# ── User-facing messages (never expose raw exceptions here) ───────────────────

USER_MESSAGES: dict[ErrorCode, str] = {
    # LLM
    ErrorCode.LLM_UNAVAILABLE: (
        "Our AI assistant is temporarily unavailable. " "Please try again in a moment."
    ),
    ErrorCode.LLM_RATE_LIMIT: (
        "We're experiencing high demand right now. "
        "Please wait a few seconds and try again."
    ),
    ErrorCode.LLM_TIMEOUT: (
        "The AI took too long to respond. "
        "Please try again — shorter questions tend to respond faster."
    ),
    ErrorCode.LLM_INVALID_KEY: (
        "There is a configuration issue on our end. "
        "Our team has been notified. Please try again later."
    ),
    # RAG
    ErrorCode.RAG_UNAVAILABLE: (
        "The document search feature is currently unavailable. "
        "Your question will be answered using general knowledge."
    ),
    ErrorCode.INDEXING_FAILED: (
        "We couldn't process your document. "
        "Please check the file is not corrupted and try uploading again."
    ),
    ErrorCode.INDEXING_INCOMPLETE: (
        "Your document is still being processed. "
        "Please wait a moment before asking a question about it."
    ),
    ErrorCode.DOCUMENT_NOT_FOUND: (
        "The uploaded document could not be found. " "Please upload it again."
    ),
    # Upload
    ErrorCode.UPLOAD_FAILED: (
        "Your file could not be uploaded. "
        "Please check your connection and try again."
    ),
    ErrorCode.FILE_TOO_LARGE: (
        "Your file is too large. " "Please upload a file smaller than 150 MB."
    ),
    ErrorCode.FILE_TYPE_INVALID: (
        "This file type is not supported. "
        "Please upload a PDF, Word document, image, or text file."
    ),
    ErrorCode.NO_FILE_PROVIDED: (
        "No file was selected. Please choose a file and try again."
    ),
    # Network
    ErrorCode.NETWORK_ERROR: (
        "A network error occurred. " "Please check your connection and try again."
    ),
    ErrorCode.TIMEOUT: ("The request timed out. Please try again."),
    # Storage
    ErrorCode.STORAGE_FAILED: (
        "We had trouble saving your data. "
        "Your session may not be preserved, but you can continue chatting."
    ),
    ErrorCode.METRICS_FAILED: (
        "Analytics recording failed silently. " "Your chat experience is unaffected."
    ),
    # Email
    ErrorCode.EMAIL_FAILED: (
        "Your message could not be sent right now. "
        "Please try again later or contact us directly."
    ),
    # Config
    ErrorCode.CONFIG_MISSING: (
        "A required service is not configured. " "Please contact support."
    ),
    # General
    ErrorCode.BAD_REQUEST: (
        "Your request could not be understood. "
        "Please check your input and try again."
    ),
    ErrorCode.INTERNAL_ERROR: (
        "Something went wrong on our end. " "Please try again in a moment."
    ),
}

_DEFAULT_MESSAGE = USER_MESSAGES[ErrorCode.INTERNAL_ERROR]


# ── AppError — raise this anywhere in the codebase ────────────────────────────


class AppError(Exception):
    """
    Structured application error.

    Parameters
    ----------
    code     : ErrorCode that maps to a user-friendly message.
    detail   : Internal detail string (logged only, never sent to the user).
    status   : HTTP status code (default 500).
    """

    def __init__(
        self,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        detail: str = "",
        status: int = 500,
    ) -> None:
        super().__init__(detail or code.value)
        self.code = code
        self.detail = detail
        self.status = status

    def user_message(self) -> str:
        return USER_MESSAGES.get(self.code, _DEFAULT_MESSAGE)


# ── Keyword-based classifier: raw exception → ErrorCode ──────────────────────

_KEYWORD_MAP: list[tuple[tuple[str, ...], ErrorCode, int]] = [
    # (keywords_in_str_lower, ErrorCode, http_status)
    (
        ("rate limit", "rate_limit", "429", "too many requests"),
        ErrorCode.LLM_RATE_LIMIT,
        429,
    ),
    (
        ("invalid api key", "invalid_api_key", "authentication", "401", "api key"),
        ErrorCode.LLM_INVALID_KEY,
        503,
    ),
    (
        ("timeout", "timed out", "read timeout", "connect timeout"),
        ErrorCode.LLM_TIMEOUT,
        504,
    ),
    (
        (
            "connection",
            "network",
            "unreachable",
            "refused",
            "name or service not known",
        ),
        ErrorCode.NETWORK_ERROR,
        503,
    ),
    (
        ("groq", "llm", "model", "completion", "openai", "gemini"),
        ErrorCode.LLM_UNAVAILABLE,
        503,
    ),
    (
        ("index", "vectorstore", "llamaindex", "llama_index", "embed"),
        ErrorCode.RAG_UNAVAILABLE,
        503,
    ),
    (
        ("indexing", "process", "parse", "extract", "document"),
        ErrorCode.INDEXING_FAILED,
        422,
    ),
    (("upload", "file", "save", "write"), ErrorCode.UPLOAD_FAILED, 500),
    (("too large", "content length", "413"), ErrorCode.FILE_TOO_LARGE, 413),
    (
        ("sql", "sqlite", "database", "db error", "storage"),
        ErrorCode.STORAGE_FAILED,
        500,
    ),
    (("smtp", "email", "mail"), ErrorCode.EMAIL_FAILED, 500),
]


def _classify(exc: BaseException) -> tuple[ErrorCode, int]:
    """Infer an ErrorCode from exception type and message keywords."""
    if isinstance(exc, AppError):
        return exc.code, exc.status

    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    combined = f"{exc_type} {msg}"

    for keywords, code, status in _KEYWORD_MAP:
        if any(kw in combined for kw in keywords):
            return code, status

    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return ErrorCode.BAD_REQUEST, 400

    return ErrorCode.INTERNAL_ERROR, 500


# ── Public helpers ────────────────────────────────────────────────────────────


def log_error(context: str, exc: BaseException) -> None:
    """
    Log the real exception details internally.
    Call this before returning any user-facing response.
    """
    logger.error(
        "[%s] %s: %s\n%s",
        context,
        type(exc).__name__,
        exc,
        traceback.format_exc(),
    )


def handle(exc: BaseException) -> tuple[str, int]:
    """
    Convert any exception to (user_friendly_message, http_status).
    Safe to call without logging — does not log internally.
    """
    if isinstance(exc, AppError):
        return exc.user_message(), exc.status
    code, status = _classify(exc)
    return USER_MESSAGES.get(code, _DEFAULT_MESSAGE), status


def flask_error_response(
    exc: BaseException,
    extra: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    """
    Return a Flask JSON error response safe for the browser.

    Response body:
        {
            "success": false,
            "error":   "<user-friendly message>",
            ...extra   (optional additional fields)
        }
    """
    message, status = handle(exc)
    body: dict[str, Any] = {"success": False, "error": message}
    if extra:
        body.update(extra)
    return jsonify(body), status


def sse_error_frame(exc: BaseException) -> str:
    """
    Return a Server-Sent Events error frame with a user-friendly message.
    Used in streaming endpoints.
    """
    import json as _json

    message, _ = handle(exc)
    return f"data: {_json.dumps({'type': 'error', 'message': message})}\n\n"
