"""Conservative request hints; the LLM chooses tools and their ordering."""
import re
from datetime import date


def analyze_request(question: str) -> dict:
    q = question.lower().strip()
    has = lambda pattern: bool(re.search(pattern, q))  # noqa: E731

    # ── Signal detection ──────────────────────────────────────────────────────
    casual = bool(re.fullmatch(r"(hi|hello|hey|thanks|thank you|how are you)[!.? ]*", q))
    document = has(r"\b(upload\w*|pdf|document|attached|my file)\b")
    explicit_web = has(
        r"\b(search (?:the )?web|look (?:it up |up )?online|"
        r"web research|web search|on the web)\b"
    )
    no_web = has(
        r"\b(don't|do not|without|no)\s+(?:use |search |searching )?(?:the )?(web|internet|online)\b"
    )
    temporal = has(r"\b(latest|current|today|this week|recent\w*|now|breaking|news|updated)\b")
    temporal |= has(rf"\b({date.today().year}|{date.today().year - 1})\b")
    external = has(
        r"\b(research|studies|study|guidelines|developments|happening|happened|"
        r"information|news|evidence|discoveries|advances|updates|status|version|"
        r"release|prices|weather|events)\b"
    )
    personal = (
        has(r"\b(my|me|i|i'm)\b")
        and has(r"\b(mood|tasks?|exams?|feeling|stressed|plan)\b")
    )
    internal_action = has(
        r"\b(create|add|schedule|check|show|list|start|make)\b.{0,45}"
        r"\b(tasks?|reminders?|self-care|self care|plans?|breathing|mood)\b"
    )
    compare = document and has(
        r"\b(compare|comparison|versus|vs|contrast|supported|consistent)\b"
    )

    # ── Web policy ────────────────────────────────────────────────────────────
    web = not no_web and (
        explicit_web or (temporal and external and (not document or compare))
    )

    if (
        no_web
        or casual
        or (document and not web)
        or ((personal or internal_action) and not external and not explicit_web)
    ):
        policy = "never"
    elif web:
        policy = "required"
    else:
        policy = "auto"

    # ── Write authorization ───────────────────────────────────────────────────
    writes: list[str] = []
    if has(r"\b(create|add|schedule|set|make)\b.{0,45}\b(task|reminder|todo)\b") or has(
        r"\bremind me\b"
    ):
        if not has(r"\b(don't|do not|never)\b.{0,30}\b(create|add|schedule|set|make|remind)\b"):
            writes.append("create_task")
    if has(r"\b(log|record|save|track)\b.{0,30}\b(mood|feeling|emotion)\b"):
        if not has(
            r"\b(don't|do not|never)\b.{0,30}\b(log|record|save|track)\b"
        ):
            writes.append("log_mood")

    return {
        "web":          policy,
        "document":     document,
        "combined":     document and web,
        "allowed_writes": writes,
        "casual":       casual,
        "date":         date.today().isoformat(),
    }
