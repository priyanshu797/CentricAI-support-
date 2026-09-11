"""
app/mood_store.py
─────────────────────────────────────────────────────────────────────────────
MongoDB-backed mood history and task storage for CentrixSupport.

Collections (in database  centrix_support):
  • moods  — per-session mood log entries
  • tasks  — persistent task list per session

Both stores share the single MongoClient that is passed in from main.py
so there is never more than one connection per process.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, ClassVar

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# MoodStore
# ─────────────────────────────────────────────────────────────────────────────


class MoodStore:
    """
    Stores mood log entries and exposes history for a session.

    Mood document schema
    ────────────────────
    {
      session_name : str,
      emotion      : str,          # e.g. "stressed", "happy"
      score        : int | None,   # 1-5 from the Daily Mind Check widget
      note         : str,          # free-form user note (may be "")
      confidence   : float | None, # 0-1 from detect_emotion()
      logged_at    : datetime
    }
    """

    COLLECTION = "moods"

    def __init__(self, db: Any) -> None:
        self._col: Collection = db[self.COLLECTION]
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._col.create_index(
                [("session_name", 1), ("logged_at", -1)],
                background=True,
                name="idx_moods_session_time",
            )
        except PyMongoError as exc:
            logger.warning("MoodStore index skipped: %s", exc)

    # ── Write ──────────────────────────────────────────────────────────────────

    def log(
        self,
        session_name: str,
        emotion: str,
        score: int | None = None,
        note: str = "",
        confidence: float | None = None,
    ) -> str:
        """Insert a mood entry. Returns the inserted document id as str."""
        doc = {
            "session_name": session_name,
            "emotion": emotion,
            "score": score,
            "note": note[:500],
            "confidence": confidence,
            "logged_at": _utcnow(),
        }
        try:
            result = self._col.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as exc:
            logger.warning("MoodStore.log failed: %s", exc)
            return ""

    # ── Read ───────────────────────────────────────────────────────────────────

    def history(
        self,
        session_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent mood entries for a session, newest first.
        Timestamps are returned as ISO strings for JSON serialisability.
        """
        try:
            cursor = (
                self._col.find(
                    {"session_name": session_name},
                    {"_id": 0, "session_name": 0},
                )
                .sort("logged_at", -1)
                .limit(limit)
            )
            entries = []
            for doc in cursor:
                doc["logged_at"] = doc["logged_at"].isoformat()
                entries.append(doc)
            return entries
        except PyMongoError as exc:
            logger.warning("MoodStore.history failed: %s", exc)
            return []

    def summary(self, session_name: str, limit: int = 20) -> dict[str, Any]:
        """
        Return aggregate mood stats for the last *limit* entries:
        most_frequent, average_score (if available), entry_count.
        """
        entries = self.history(session_name, limit=limit)
        if not entries:
            return {"entry_count": 0, "most_frequent": None, "average_score": None}

        emotion_counts: dict[str, int] = {}
        scores = []
        for e in entries:
            em = e.get("emotion", "unknown")
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
            if e.get("score") is not None:
                scores.append(e["score"])

        most_frequent = max(emotion_counts, key=emotion_counts.__getitem__)
        avg_score = round(sum(scores) / len(scores), 1) if scores else None

        return {
            "entry_count": len(entries),
            "most_frequent": most_frequent,
            "average_score": avg_score,
            "emotion_counts": emotion_counts,
            "recent": entries[:5],
        }


# ─────────────────────────────────────────────────────────────────────────────
# TaskStore
# ─────────────────────────────────────────────────────────────────────────────


class TaskStore:
    """
    Persistent task list stored in MongoDB.

    Task document schema
    ────────────────────
    {
      session_name : str,
      title        : str,
      due_date     : str | None,   # ISO date string "YYYY-MM-DD"
      priority     : str,          # "low" | "medium" | "high"
      status       : str,          # "pending" | "done"
      created_at   : datetime,
      updated_at   : datetime
    }
    """

    COLLECTION:       ClassVar[str]       = "tasks"
    VALID_PRIORITIES: ClassVar[set[str]]  = {"low", "medium", "high"}
    VALID_STATUSES:   ClassVar[set[str]]  = {"pending", "done"}

    def __init__(self, db: Any) -> None:
        self._col: Collection = db[self.COLLECTION]
        self._ensure_index()

    def _ensure_index(self) -> None:
        try:
            self._col.create_index(
                [("session_name", 1), ("created_at", -1)],
                background=True,
                name="idx_tasks_session_time",
            )
        except PyMongoError as exc:
            logger.warning("TaskStore index skipped: %s", exc)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        session_name: str,
        title: str,
        due_date: str | None = None,
        priority: str = "medium",
    ) -> dict[str, Any]:
        """
        Create a task.  Returns the new task dict (without MongoDB _id).
        Validates priority; falls back to "medium" on invalid values.
        """
        if priority not in self.VALID_PRIORITIES:
            priority = "medium"

        now = _utcnow()
        doc = {
            "session_name": session_name,
            "title": title[:300],
            "due_date": due_date,
            "priority": priority,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = self._col.insert_one(doc)
            doc["id"] = str(result.inserted_id)
        except PyMongoError as exc:
            logger.warning("TaskStore.add failed: %s", exc)
            doc["id"] = ""

        # Return serialisable form (no ObjectId, ISO datetimes)
        return self._serialise(doc)

    def complete(self, task_id: str, owner_prefix: str | None = None) -> bool:
        """Mark a task as done by its string id. Returns True on success."""
        from bson import ObjectId  # type: ignore[import]

        try:
            filt = {"_id": ObjectId(task_id)}
            if owner_prefix is not None:
                import re
                filt["session_name"] = {"$regex": "^" + re.escape(owner_prefix + ":")}
            res = self._col.update_one(
                filt,
                {"$set": {"status": "done", "updated_at": _utcnow()}},
            )
            return res.modified_count > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("TaskStore.complete failed: %s", exc)
            return False

    # ── Read ───────────────────────────────────────────────────────────────────

    def list_tasks(
        self,
        session_name: str,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return tasks for a session (newest first).
        Optionally filter by status: "pending" | "done".
        """
        filt: dict[str, Any] = {"session_name": session_name}
        if status in self.VALID_STATUSES:
            filt["status"] = status
        try:
            cursor = (
                self._col.find(filt, {"session_name": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            return [self._serialise(doc) for doc in cursor]
        except PyMongoError as exc:
            logger.warning("TaskStore.list_tasks failed: %s", exc)
            return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise(doc: dict) -> dict:
        """Convert ObjectId → str and datetime → ISO string."""
        from bson import ObjectId  # type: ignore[import]

        out = {}
        for k, v in doc.items():
            if k == "_id":
                out["id"] = str(v)
            elif isinstance(v, datetime):
                out[k] = v.isoformat()
            elif isinstance(v, ObjectId):
                out[k] = str(v)
            else:
                out[k] = v
        if "id" not in out and "_id" not in doc:
            pass  # already added by add()
        return out
