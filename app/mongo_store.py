"""
app/mongo_store.py
──────────────────────────────────────────────────────────────────────────────
MongoDB-backed conversation store for CentrixSupport.

Collection layout  (database: centrix_support)
───────────────────────────────────────────────
Collection: conversations
  {
    _id:          ObjectId (auto),
    session_name: str,          # unique session identifier
    messages:     [             # ordered list of turns
      { role: "user"|"assistant", content: str, ts: datetime }
    ],
    created_at:   datetime,
    updated_at:   datetime
  }
  Index: session_name (unique)

Public API
──────────
  ConversationStore(uri, db_name)   — connect, create index
  .load(session_name)  → list[dict] # [{role, content}, ...]
  .save(session_name, messages)     # full replace (upsert)
  .append(session_name, role, content) # add single turn
  .clear(session_name)              # delete session doc
  .ping()              → bool       # True if Mongo is reachable
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class ConversationStore:
    """Thread-safe MongoDB conversation store."""

    COLLECTION = "conversations"

    def __init__(self, uri: str, db_name: str = "centrix_support") -> None:
        self._client: MongoClient = MongoClient(
            uri,
            serverSelectionTimeoutMS=5_000,
            connectTimeoutMS=5_000,
            socketTimeoutMS=10_000,
        )
        self._db = self._client[db_name]
        self._col = self._db[self.COLLECTION]
        self._ensure_index()
        logger.info("MongoDB conversation store connected (db=%s)", db_name)

    # ── Index ──────────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        try:
            self._col.create_index(
                [("session_name", ASCENDING)],
                unique=True,
                background=True,
                name="idx_session_name",
            )
        except PyMongoError as exc:
            logger.warning("Index creation skipped: %s", exc)

    # ── Connectivity ───────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if MongoDB is reachable."""
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    # ── Core operations ────────────────────────────────────────────────────────

    def load(self, session_name: str) -> list[dict[str, str]]:
        """
        Return the message list for a session.
        Each item is {role, content} — timestamps are stripped so the
        existing code that only expects role+content continues to work.
        """
        try:
            doc = self._col.find_one({"session_name": session_name}, {"messages": 1})
            if doc and doc.get("messages"):
                return [
                    {"role": m["role"], "content": m["content"]}
                    for m in doc["messages"]
                ]
        except PyMongoError as exc:
            logger.warning("MongoDB load failed for session %r: %s", session_name, exc)
        return []

    def save(self, session_name: str, messages: list[dict[str, Any]]) -> None:
        """
        Full replace — write the entire message list for a session.
        Accepts both {role, content} and {role, content, ts} formats.
        """
        now = _utcnow()
        mongo_messages = [
            {
                "role": m["role"],
                "content": m["content"],
                "ts": m.get("ts", now),
            }
            for m in messages
        ]
        try:
            self._col.update_one(
                {"session_name": session_name},
                {
                    "$set": {
                        "messages": mongo_messages,
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "session_name": session_name,
                        "created_at": now,
                    },
                },
                upsert=True,
            )
        except PyMongoError as exc:
            logger.warning("MongoDB save failed for session %r: %s", session_name, exc)

    def append(self, session_name: str, role: str, content: str) -> None:
        """Append a single turn to an existing or new session document."""
        now = _utcnow()
        try:
            self._col.update_one(
                {"session_name": session_name},
                {
                    "$push": {
                        "messages": {"role": role, "content": content, "ts": now}
                    },
                    "$set": {"updated_at": now},
                    "$setOnInsert": {
                        "session_name": session_name,
                        "created_at": now,
                    },
                },
                upsert=True,
            )
        except PyMongoError as exc:
            logger.warning(
                "MongoDB append failed for session %r: %s", session_name, exc
            )

    def clear(self, session_name: str) -> None:
        """Delete the session document entirely."""
        try:
            self._col.delete_one({"session_name": session_name})
        except PyMongoError as exc:
            logger.warning("MongoDB clear failed for session %r: %s", session_name, exc)

    def all_sessions(self) -> list[str]:
        """Return a sorted list of all known session names."""
        try:
            return sorted(
                doc["session_name"] for doc in self._col.find({}, {"session_name": 1})
            )
        except PyMongoError as exc:
            logger.warning("MongoDB all_sessions failed: %s", exc)
            return []

    def close(self) -> None:
        """Close the underlying MongoClient."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
