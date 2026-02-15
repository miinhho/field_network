from __future__ import annotations

from .guardrails import guardrail_violation
from .models import Answer


def compose_answer(
    query_type: str,
    claims: list[str],
    evidence_ids: list[str],
    metrics_used: dict[str, float],
    uncertainty: float,
) -> Answer:
    combined = " ".join(claims)
    reason = guardrail_violation(combined)
    if reason:
        return Answer(
            query_type=query_type,
            claims=[],
            evidence_ids=evidence_ids,
            metrics_used=metrics_used,
            uncertainty=1.0,
            blocked_by_guardrail=True,
            block_reason=reason,
        )

    return Answer(
        query_type=query_type,
        claims=claims,
        evidence_ids=evidence_ids,
        metrics_used=metrics_used,
        uncertainty=max(0.0, min(1.0, uncertainty)),
    )
