from __future__ import annotations

from .models import Query


class QueryRouter:
    """Simple rule-based router for PoC query classes."""

    DESCRIBE = "describe"
    PREDICT = "predict"
    INTERVENE = "intervene"

    def classify(self, query: Query) -> str:
        if query.query_type:
            return query.query_type

        text = query.text.lower()
        if any(token in text for token in ("predict", "likely", "will", "전이", "예측")):
            return self.PREDICT
        if any(token in text for token in ("intervene", "change", "what if", "개입", "바꾸")):
            return self.INTERVENE
        return self.DESCRIBE
