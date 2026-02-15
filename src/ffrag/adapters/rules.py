from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models import Actant, Interaction, LayeredGraph, Perturbation
from .base import BaseAdapter


@dataclass(slots=True)
class MappingSpec:
    """Declarative mapping for record -> canonical graph conversion."""

    node_fields: dict[str, str]
    # (source_field, target_field, layer, base_weight)
    edge_rules: list[tuple[str, str, str, float]]
    timestamp_field: str
    perturb_target_field: str
    default_intensity: float = 1.0


class GenericMappingAdapter(BaseAdapter):
    """Generic adapter for user-defined domain records via MappingSpec."""

    def __init__(
        self,
        records: list[dict[str, object]],
        spec: MappingSpec,
        graph_id: str = "generic-graph",
        schema_version: str = "0.1",
    ) -> None:
        super().__init__()
        self.records = records
        self.spec = spec
        self.graph_id = graph_id
        self.schema_version = schema_version

    def to_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id=self.graph_id, schema_version=self.schema_version)
        if not self.records:
            return g

        idx = 0
        for rec in self.records:
            ts = self._as_datetime(rec.get(self.spec.timestamp_field))
            if ts is None:
                continue
            for field_name, node_kind in self.spec.node_fields.items():
                value = rec.get(field_name)
                if value is None:
                    continue
                node_id = str(value)
                if node_id not in g.actants:
                    g.actants[node_id] = Actant(actant_id=node_id, kind=node_kind, label=node_id)

            for src_field, dst_field, layer, base_w in self.spec.edge_rules:
                src = rec.get(src_field)
                dst = rec.get(dst_field)
                if src is None or dst is None:
                    continue
                src_id = str(src)
                dst_id = str(dst)
                if src_id not in g.actants:
                    g.actants[src_id] = Actant(actant_id=src_id, kind="entity", label=src_id)
                if dst_id not in g.actants:
                    g.actants[dst_id] = Actant(actant_id=dst_id, kind="entity", label=dst_id)
                g.interactions.append(
                    Interaction(
                        interaction_id=f"{self.graph_id}:e:{idx}",
                        timestamp=ts,
                        source_id=src_id,
                        target_id=dst_id,
                        layer=layer,
                        weight=max(0.0, float(base_w)),
                        metadata={"source_record_idx": str(idx)},
                    )
                )
                idx += 1
        return g

    def default_perturbation(self) -> Perturbation:
        if not self.records:
            return Perturbation(
                perturbation_id=f"{self.graph_id}:p0",
                timestamp=datetime.now(timezone.utc),
                targets=[],
                intensity=max(0.0, self.spec.default_intensity),
                kind="generic",
            )

        counts: dict[str, int] = {}
        latest: datetime | None = None
        for rec in self.records:
            target = rec.get(self.spec.perturb_target_field)
            ts = self._as_datetime(rec.get(self.spec.timestamp_field))
            if target is not None:
                tid = str(target)
                counts[tid] = counts.get(tid, 0) + 1
            if ts is not None:
                if latest is None:
                    latest = ts
                else:
                    ts_key = ts.timestamp() if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc).timestamp()
                    latest_key = (
                        latest.timestamp()
                        if latest.tzinfo is not None
                        else latest.replace(tzinfo=timezone.utc).timestamp()
                    )
                    if ts_key > latest_key:
                        latest = ts
        chosen = max(counts.items(), key=lambda item: item[1])[0] if counts else ""
        return Perturbation(
            perturbation_id=f"{self.graph_id}:p0",
            timestamp=latest if latest is not None else datetime.now(timezone.utc),
            targets=[chosen] if chosen else [],
            intensity=max(0.0, self.spec.default_intensity),
            kind="generic",
        )

    def mapping_report(self) -> dict[str, float]:
        return {
            "record_count": float(len(self.records)),
            "edge_rule_count": float(len(self.spec.edge_rules)),
            "node_field_count": float(len(self.spec.node_fields)),
        }

    def _as_datetime(self, value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        return None
