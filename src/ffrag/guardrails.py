from __future__ import annotations

BLOCKED_TOKENS = (
    "personality",
    "성격",
    "motivation",
    "동기",
    "가치관",
)


def guardrail_violation(text: str) -> str | None:
    lowered = text.lower()
    for token in BLOCKED_TOKENS:
        if token in lowered:
            return f"blocked token detected: {token}"
    return None
