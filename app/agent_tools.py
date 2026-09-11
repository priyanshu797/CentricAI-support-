"""
app/agent_tools.py
─────────────────────────────────────────────────────────────────────────────
Nine callable tools that wrap existing CentrixSupport functionality.

Every tool is a plain Python function that accepts a typed dict payload and
returns a typed dict result. The agent_runner calls them directly after the
LLM decides which tool to invoke.

No new business logic is introduced here — each tool is a thin wrapper over
existing services:
  • analyze_emotion        → detect_emotion() from main.py
  • search_knowledge       → Agent.run() from content_retrieval.py
  • create_self_care_plan  → logic from self_care_plan.py (LLM-generated)
  • get_tasks              → TaskStore.list_tasks()
  • create_task            → TaskStore.add()
  • log_mood               → MoodStore.log()
  • get_mood_history       → MoodStore.summary()
  • start_breathing_exercise → LLM-guided text (no timers — chat context)
  • summarize_document     → Agent.run() with summarize prompt

Tool JSON schemas are also defined here so agent_runner can pass them to the
Groq /chat/completions endpoint as tool definitions.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy service references (injected at runtime by agent_runner) ─────────────
# These are set once by init_tool_services() called from main.py on startup.

_mood_store: Any = None  # app.mood_store.MoodStore
_task_store: Any = None  # app.mood_store.TaskStore
_rag_agent_factory: Any = None  # callable(document_id) → Agent | None
_detect_emotion_fn: Any = None  # callable(text) -> (label, confidence)
_llm_chat_fn: Any = None  # callable(messages, **kw) -> (text, pt, ct, provider)
_conversation_store: Any = None


def init_tool_services(
    mood_store: Any,
    task_store: Any,
    rag_agent_factory: Any,
    detect_emotion_fn: Any,
    llm_chat_fn: Any,
    conversation_store: Any = None,
) -> None:
    """Called once from main.py after all services are initialised."""
    global _mood_store, _task_store, _rag_agent_factory
    global _detect_emotion_fn, _llm_chat_fn
    global _conversation_store
    _conversation_store = conversation_store
    _mood_store = mood_store
    _task_store = task_store
    _rag_agent_factory = rag_agent_factory
    _detect_emotion_fn = detect_emotion_fn
    _llm_chat_fn = llm_chat_fn


def analyze_emotion(text: str, session_name: str = "") -> dict[str, Any]:
    """
    Classify the emotional state expressed in *text*.

    Uses the existing keyword-weighted detect_emotion() from main.py.
    Optionally logs the result to MongoDB if session_name is provided.

    Returns
    -------
    {
      "emotion":    str,    # e.g. "stressed"
      "confidence": float,  # 0.0 – 1.0
      "intensity":  str,    # "Low" | "Medium" | "High"
      "logged":     bool
    }
    """
    t0 = time.perf_counter()
    if not _detect_emotion_fn:
        return {"error": "Emotion service not initialised"}

    try:
        label, confidence = _detect_emotion_fn(text)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Emotion detection failed: {exc}"}

    # Map confidence to intensity (mirrors emotion_intensity() in main.py)
    if confidence >= 0.75:
        intensity = "High"
    elif confidence >= 0.50:
        intensity = "Medium"
    else:
        intensity = "Low"

    logged = False
    if session_name and _mood_store:
        try:
            _mood_store.log(
                session_name=session_name,
                emotion=label,
                confidence=confidence,
                note=text[:200],
            )
            logged = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_emotion: mood log failed: %s", exc)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "emotion": label,
        "confidence": round(confidence, 3),
        "intensity": intensity,
        "logged": logged,
        "latency_ms": latency_ms,
    }


def search_knowledge(
    query: str,
    document_id: str = "",
    session_name: str = "",
) -> dict[str, Any]:
    """
    Search the uploaded document knowledge base for information about *query*.

    Delegates to the existing RAG Agent.run() pipeline:
      embedding → vector search → optional reranking → LLM grounding.

    Returns
    -------
    {
      "answer":      str,
      "sources":     list[str],
      "tool_used":   str,   # "docs" | "cache" | "llm" | …
      "cache_hit":   bool,
      "chunks_retrieved": int,
      "top_similarity":   float | None,
      "groundedness_score": float | None
    }
    """
    t0 = time.perf_counter()
    if not _rag_agent_factory:
        return {"error": "RAG service not initialised", "answer": ""}

    agent = _rag_agent_factory(document_id)
    if agent is None and document_id:
        return {
            "error": "Document not indexed yet",
            "answer": "",
            "sources": [],
        }

    if agent is None:
        # No document — answer from general knowledge
        if not _llm_chat_fn:
            return {"error": "LLM service not initialised", "answer": ""}
        try:
            from app.llm_fallback import GROQ_MODEL

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful knowledge assistant. Answer concisely and accurately."
                    ),
                },
                {"role": "user", "content": query},
            ]
            text, pt, ct, provider = _llm_chat_fn(
                messages, temperature=0.5, max_tokens=600
            )
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return {
                "answer": text,
                "sources": [],
                "tool_used": "llm",
                "cache_hit": False,
                "chunks_retrieved": 0,
                "top_similarity": None,
                "groundedness_score": None,
                "provider": provider,
                "latency_ms": latency_ms,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "answer": ""}

    try:
        from app.rag_metrics import RagMetrics

        metrics = RagMetrics(query=query, session_name=session_name)
        answer, sources, tool_used, gnd, _ = agent.run(query, metrics=metrics, knowledge_only=True)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer": answer,
            "sources": sources,
            "tool_used": tool_used,
            "cache_hit": metrics.cache_hit,
            "chunks_retrieved": metrics.num_docs_retrieved,
            "top_similarity": (
                metrics.original_similarity_scores[0]
                if metrics.original_similarity_scores
                else None
            ),
            "groundedness_score": gnd.score if gnd is not None else None,
            "latency_ms": latency_ms,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("search_knowledge error: %s", exc)
        return {"error": str(exc), "answer": ""}


def create_self_care_plan(
    duration_days: int = 3,
    focus_areas: list[str] | None = None,
    context: str = "",
) -> dict[str, Any]:
    """
    Generate a personalised multi-day self-care plan using the LLM.

    The plan is structured around the user's focus areas (e.g. sleep,
    stress, nutrition).  The existing self_care_plan.py keyword logic is
    used as template inspiration; the LLM produces the full plan text.

    Returns
    -------
    {
      "plan":        str,   # formatted plan text
      "duration_days": int,
      "focus_areas": list[str]
    }
    """
    t0 = time.perf_counter()
    if not _llm_chat_fn:
        return {"error": "LLM service not initialised", "plan": ""}

    focus = focus_areas or ["stress management", "sleep hygiene", "mindfulness"]
    focus_str = ", ".join(focus)
    duration_days = max(1, min(duration_days, 14))

    prompt = (
        f"Create a {duration_days}-day personalised self-care plan for someone "
        f"focusing on: {focus_str}.\n"
        + (f"Additional context: {context}\n" if context else "")
        + "\nFormat as a clear day-by-day plan with:\n"
        "- Morning routine (5-15 min)\n"
        "- Midday check-in (5 min)\n"
        "- Evening wind-down (10-15 min)\n"
        "Use encouraging, warm language. Be specific and actionable.\n"
        "Start directly with 'Day 1:' — no preamble."
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a compassionate wellness coach creating personalised self-care plans. "
                    "Be practical, warm, and evidence-based."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        plan_text, pt, ct, provider = _llm_chat_fn(
            messages, temperature=0.65, max_tokens=1200
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "plan": plan_text,
            "duration_days": duration_days,
            "focus_areas": focus,
            "provider": provider,
            "latency_ms": latency_ms,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("create_self_care_plan error: %s", exc)
        return {"error": str(exc), "plan": ""}


def get_tasks(session_name: str, status: str = "pending") -> dict[str, Any]:
    """
    Retrieve tasks for this session from MongoDB.

    Parameters
    ----------
    status : "pending" | "done" | "all"  (default "pending")

    Returns
    -------
    {
      "tasks":  list[dict],
      "count":  int
    }
    """
    if not _task_store:
        return {"error": "Task service not initialised", "tasks": [], "count": 0}

    try:
        effective_status = None if status == "all" else status
        tasks = _task_store.list_tasks(session_name, status=effective_status)
        return {"tasks": tasks, "count": len(tasks)}
    except Exception as exc:  # noqa: BLE001
        logger.error("get_tasks error: %s", exc)
        return {"error": str(exc), "tasks": [], "count": 0}


def create_task(
    session_name: str,
    title: str,
    due_date: str | None = None,
    priority: str = "medium",
) -> dict[str, Any]:
    """
    Create a new task for this session in MongoDB.

    Parameters
    ----------
    due_date : ISO date string "YYYY-MM-DD" or None
    priority : "low" | "medium" | "high"

    Returns
    -------
    {
      "task":    dict,   # the created task
      "success": bool
    }
    """
    if not _task_store:
        return {"error": "Task service not initialised", "success": False}

    if not title or not title.strip():
        return {"error": "Task title is required", "success": False}

    try:
        task = _task_store.add(
            session_name=session_name,
            title=title.strip(),
            due_date=due_date,
            priority=priority,
        )
        return {"task": task, "success": bool(task.get("id"))}
    except Exception as exc:  # noqa: BLE001
        logger.error("create_task error: %s", exc)
        return {"error": str(exc), "success": False}


def log_mood(
    session_name: str,
    emotion: str,
    score: int | None = None,
    note: str = "",
) -> dict[str, Any]:
    """
    Log a mood entry for this session in MongoDB.

    Parameters
    ----------
    score : 1–5 numeric mood rating (optional)
    note  : free-form note (max 500 chars)

    Returns
    -------
    {
      "logged":     bool,
      "entry_id":   str,
      "emotion":    str
    }
    """
    if not _mood_store:
        return {"error": "Mood service not initialised", "logged": False}

    if score is not None:
        score = max(1, min(5, int(score)))

    try:
        entry_id = _mood_store.log(
            session_name=session_name,
            emotion=emotion,
            score=score,
            note=note,
        )
        return {"logged": bool(entry_id), "entry_id": entry_id, "emotion": emotion}
    except Exception as exc:  # noqa: BLE001
        logger.error("log_mood error: %s", exc)
        return {"error": str(exc), "logged": False}


def get_mood_history(session_name: str, limit: int = 10) -> dict[str, Any]:
    """
    Retrieve mood history summary for this session from MongoDB.

    Returns
    -------
    {
      "summary":       dict,   # most_frequent, average_score, emotion_counts
      "recent_entries": list[dict]
    }
    """
    if not _mood_store:
        return {"error": "Mood service not initialised", "summary": {}}

    limit = max(1, min(limit, 50))
    try:
        summary = _mood_store.summary(session_name, limit=limit)
        return {
            "summary": {k: v for k, v in summary.items() if k != "recent"},
            "recent_entries": summary.get("recent", []),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("get_mood_history error: %s", exc)
        return {"error": str(exc), "summary": {}}


def start_breathing_exercise(technique: str = "4-7-8") -> dict[str, Any]:
    """
    Return a text-based guided breathing exercise.

    The existing mental_exercises.py has interactive CLI versions using
    time.sleep() and input() — not suitable for a chat interface.
    This tool generates equivalent guidance as structured text via the LLM.

    Parameters
    ----------
    technique : "4-7-8" | "box" | "diaphragmatic"  (default "4-7-8")

    Returns
    -------
    {
      "exercise": str,   # formatted exercise instructions
      "technique": str
    }
    """
    t0 = time.perf_counter()
    if not _llm_chat_fn:
        return {"error": "LLM service not initialised", "exercise": ""}

    technique_prompts: dict[str, str] = {
        "4-7-8": (
            "Write a calm, guided 4-7-8 breathing exercise in text form for a chat interface. "
            "Include: inhale for 4 seconds, hold for 7, exhale for 8. 3 cycles. "
            "Add a brief intro and closing. Use encouraging, peaceful language. "
            "Format with clear step numbers."
        ),
        "box": (
            "Write a calm, guided box breathing exercise (4-4-4-4 technique) in text form. "
            "4 cycles. Warm, encouraging tone. Clear numbered steps."
        ),
        "diaphragmatic": (
            "Write a diaphragmatic (belly) breathing exercise guide in text form. "
            "5 minutes. Include posture guidance, hand placement, and breath awareness. "
            "Calm, step-by-step format."
        ),
    }

    prompt = technique_prompts.get(
        technique,
        technique_prompts["4-7-8"],
    )

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a mindfulness and breathing coach. "
                    "Create clear, calming, text-based exercises for a wellness chat app."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        exercise_text, _pt, _ct, provider = _llm_chat_fn(
            messages, temperature=0.5, max_tokens=500
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "exercise": exercise_text,
            "technique": technique,
            "provider": provider,
            "latency_ms": latency_ms,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("start_breathing_exercise error: %s", exc)
        return {"error": str(exc), "exercise": ""}


def summarize_document(
    query: str = "Summarize the key points of this document",
    document_id: str = "",
    session_name: str = "",
) -> dict[str, Any]:
    """
    Summarize the content of an uploaded document.

    Delegates to search_knowledge with a summarization-focused query.

    Returns
    -------
    {
      "summary":  str,
      "sources":  list[str],
      "cache_hit": bool
    }
    """
    summarize_query = (
        f"{query} — Please provide a comprehensive summary covering "
        "the main topics, key findings, and important recommendations."
    )
    result = search_knowledge(
        query=summarize_query,
        document_id=document_id,
        session_name=session_name,
    )
    return {
        **result,
        "summary": result.get("answer", ""),
        "sources": result.get("sources", []),
        "cache_hit": result.get("cache_hit", False),
        "error": result.get("error"),
        "latency_ms": result.get("latency_ms", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# web_search tool — real-time web results via DuckDuckGo
# ─────────────────────────────────────────────────────────────────────────────


def web_search(query: str) -> dict[str, Any]:
    """Legacy alias; all search business logic lives in app.web_search."""
    try:
        results = search_web(query)
        return {"results": results, "found": bool(results)}
    except Exception:
        return {"error": "DuckDuckGo unavailable", "results": [], "found": False}


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — used by agent_runner to build Groq tool schemas
# ─────────────────────────────────────────────────────────────────────────────

# Each entry:  (function, groq_tool_schema_dict)
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "analyze_emotion": {
        "type": "function",
        "function": {
            "name": "analyze_emotion",
            "description": (
                "Analyse the emotional content of a text and return the detected emotion, "
                "confidence score, and intensity level. Use when understanding the user's "
                "emotional state is helpful. Optionally logs the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyse for emotional content.",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier for logging (optional).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    "search_knowledge": {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the user's uploaded document knowledge base for information. "
                "Use for any question that might be answered by uploaded documents. "
                "If no document is uploaded, answers from general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The document_id of the uploaded file (leave empty if none).",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    "create_self_care_plan": {
        "type": "function",
        "function": {
            "name": "create_self_care_plan",
            "description": (
                "Generate a personalised multi-day self-care plan tailored to the user's needs. "
                "Use when a user asks for a self-care, wellness, or recovery plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_days": {
                        "type": "integer",
                        "description": "Number of days for the plan (1–14). Default 3.",
                        "minimum": 1,
                        "maximum": 14,
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Areas to focus on, e.g. ['stress management', 'sleep', 'nutrition']. "
                            "Defaults to general wellness."
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the user's situation.",
                    },
                },
                "required": [],
            },
        },
    },
    "get_tasks": {
        "type": "function",
        "function": {
            "name": "get_tasks",
            "description": (
                "Retrieve the user's task list from the database. "
                "Use when the user asks to see, check, or review their tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "done", "all"],
                        "description": "Filter by task status. Default 'pending'.",
                    },
                },
                "required": ["session_name"],
            },
        },
    },
    "create_task": {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a new task and save it to the database. "
                "Use when the user wants to add, schedule, or remember a task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short description of the task.",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format (optional).",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Task priority. Default 'medium'.",
                    },
                },
                "required": ["session_name", "title"],
            },
        },
    },
    "log_mood": {
        "type": "function",
        "function": {
            "name": "log_mood",
            "description": (
                "Log a mood entry for the current session in the database. "
                "Use when the user explicitly rates their mood or when you want to "
                "record a detected emotional state for tracking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                    "emotion": {
                        "type": "string",
                        "description": (
                            "Emotion label: stressed, anxious, sad, angry, "
                            "overwhelmed, happy, or neutral."
                        ),
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Numeric mood score 1 (very bad) to 5 (great).",
                    },
                    "note": {
                        "type": "string",
                        "description": "Brief note about the mood entry.",
                    },
                },
                "required": ["session_name", "emotion"],
            },
        },
    },
    "get_mood_history": {
        "type": "function",
        "function": {
            "name": "get_mood_history",
            "description": (
                "Retrieve the user's recent mood history and summary statistics "
                "from the database. Use when the user asks about their mood patterns, "
                "history, or wants context for a self-care plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to retrieve (default 10, max 50).",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["session_name"],
            },
        },
    },
    "start_breathing_exercise": {
        "type": "function",
        "function": {
            "name": "start_breathing_exercise",
            "description": (
                "Generate a guided breathing exercise for the user to follow in the chat. "
                "Use when the user asks for breathing, relaxation, or calming exercises."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "technique": {
                        "type": "string",
                        "enum": ["4-7-8", "box", "diaphragmatic"],
                        "description": (
                            "Breathing technique: '4-7-8' (relaxation), "
                            "'box' (focus/calm), 'diaphragmatic' (deep breathing). "
                            "Default '4-7-8'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    "summarize_document": {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": (
                "Summarize the content of an uploaded document. "
                "Use when the user wants an overview or summary of their uploaded file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific summary request, e.g. 'summarize key stress-reduction "
                            "techniques'. Defaults to general summary."
                        ),
                    },
                    "document_id": {
                        "type": "string",
                        "description": "The document_id of the uploaded file.",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Session identifier.",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, real-time information. "
                "Use for queries about recent events, latest news, current statistics, "
                "or any question that requires up-to-date information not available in "
                "uploaded documents. Examples: 'mental health trends 2026', "
                "'latest anxiety research', 'current wellness guidelines'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
}


# Map tool name → callable
TOOL_FUNCTIONS: dict[str, Any] = {
    "analyze_emotion": analyze_emotion,
    "search_knowledge": search_knowledge,
    "create_self_care_plan": create_self_care_plan,
    "get_tasks": get_tasks,
    "create_task": create_task,
    "log_mood": log_mood,
    "get_mood_history": get_mood_history,
    "start_breathing_exercise": start_breathing_exercise,
    "summarize_document": summarize_document,
    "web_search": web_search,
}


def get_conversation_history(session_name: str, limit: int = 10) -> dict:
    if _conversation_store is None:
        return {"error": "Conversation service unavailable"}
    history = _conversation_store.load(session_name)[-limit:]
    return {"history": history, "count": len(history)}


def search_web(query: str, num_results: int = 5) -> list[dict]:
    from app.web_search import search_web as _search
    return _search(query, num_results)


TOOL_FUNCTIONS.update(search_web=search_web, search_uploaded_document=search_knowledge,
                      get_conversation_history=get_conversation_history)
TOOL_REGISTRY["search_web"] = {"type": "function", "function": {
    "name": "search_web", "description": "Search the web for current public information using Google Search (SerpApi) with DuckDuckGo as fallback. Use a public topic query; never send private document text or personal mood/history. Returns title, URL and snippet.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 2000},
        "num_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
        "required": ["query"], "additionalProperties": False}}}
TOOL_REGISTRY["get_conversation_history"] = {"type": "function", "function": {
    "name": "get_conversation_history", "description": "Read recent turns in the current conversation.",
    "parameters": {"type": "object", "properties": {
        "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": []}}}
from copy import deepcopy
TOOL_REGISTRY["search_uploaded_document"] = deepcopy(TOOL_REGISTRY["search_knowledge"])
TOOL_REGISTRY["search_uploaded_document"]["function"]["name"] = "search_uploaded_document"


def public_schema(name: str) -> dict:
    """One schema source for MCP and the local emotion tool. Identity is host-only."""
    schema = deepcopy(TOOL_REGISTRY[name]["function"])
    params = schema["parameters"]
    private = {"session_name", "document_id", "user_id"}
    params["properties"] = {k: v for k, v in params["properties"].items() if k not in private}
    params["required"] = [k for k in params.get("required", []) if k not in private]
    params["additionalProperties"] = False
    for key, spec in params["properties"].items():
        if spec.get("type") == "string":
            spec.setdefault("maxLength", 2000)
            if key in params["required"]:
                spec.setdefault("minLength", 1)
        if spec.get("type") == "array":
            spec["maxItems"] = 10
            spec["items"]["maxLength"] = 200
    if name == "create_task":
        params["properties"]["title"]["maxLength"] = 300
        params["properties"]["due_date"].update(type=["string", "null"], format="date")
    if name == "log_mood":
        params["properties"]["note"]["maxLength"] = 500
    return schema
