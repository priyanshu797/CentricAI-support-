"""Bind anonymous browser ownership using Flask's signed session cookie.

Named conversations are scoped to the browser owner; they are not identities.
An authentication integration can set session['owner_id'] after login.
"""
import hashlib
import uuid

from flask import abort, session

from app.mcp.context import TrustedContext


def owner_id():
    if not session.get("owner_id"):
        session["owner_id"] = uuid.uuid4().hex
    return session["owner_id"]


def trusted_session_name(label="default_session"):
    if not isinstance(label, str) or len(label) > 200:
        abort(400, description="Invalid conversation name")
    return owner_id() + ":" + hashlib.sha256(label.encode()).hexdigest()[:24]


def register_document(document_id, filenames):
    owner_id()
    documents = dict(session.get("documents", {}))
    documents[document_id] = filenames
    session["documents"] = dict(list(documents.items())[-20:])


def require_document(document_id):
    if document_id and document_id not in session.get("documents", {}):
        abort(403, description="Document is not available in this browser session")
    return document_id or ""


def trusted_context(label="default_session", document_id=""):
    return TrustedContext(trusted_session_name(label), require_document(document_id), owner_id())
