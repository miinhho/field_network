from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

from ffrag.adapters import BaseAdapter
from ffrag.models import Actant, Interaction, LayeredGraph, Perturbation


@dataclass(slots=True)
class CalendarEvent:
    event_id: str
    start_time: datetime
    end_time: datetime
    organizer_id: str
    attendee_ids: list[str]
    location_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class CalendarEventsAdapter(BaseAdapter):
    """Example domain adapter moved out of core SDK package."""

    def __init__(
        self,
        events: list[CalendarEvent],
        graph_id: str = "calendar-graph",
        schema_version: str = "0.1",
    ) -> None:
        super().__init__()
        self.events = events
        self.graph_id = graph_id
        self.schema_version = schema_version

    def to_graph(self) -> LayeredGraph:
        g = LayeredGraph(graph_id=self.graph_id, schema_version=self.schema_version)
        if not self.events:
            return g
        for ev in self.events:
            self._ensure_actant(g, ev.organizer_id, "person")
            for attendee in ev.attendee_ids:
                self._ensure_actant(g, attendee, "person")
            if ev.location_id:
                self._ensure_actant(g, ev.location_id, "place")
        idx = 0
        for ev in self.events:
            duration_h = max(0.1, (ev.end_time - ev.start_time).total_seconds() / 3600.0)
            participants = [ev.organizer_id] + [a for a in ev.attendee_ids if a != ev.organizer_id]
            for attendee in ev.attendee_ids:
                if attendee == ev.organizer_id:
                    continue
                g.interactions.append(
                    Interaction(
                        interaction_id=f"{ev.event_id}:org:{idx}",
                        timestamp=ev.start_time,
                        source_id=ev.organizer_id,
                        target_id=attendee,
                        layer="social",
                        weight=round(0.8 * duration_h, 6),
                    )
                )
                idx += 1
            for a, b in combinations(sorted(set(participants)), 2):
                g.interactions.append(
                    Interaction(
                        interaction_id=f"{ev.event_id}:co:{idx}",
                        timestamp=ev.start_time,
                        source_id=a,
                        target_id=b,
                        layer="temporal",
                        weight=round(0.45 * duration_h, 6),
                    )
                )
                idx += 1
            if ev.location_id:
                g.interactions.append(
                    Interaction(
                        interaction_id=f"{ev.event_id}:loc:{idx}",
                        timestamp=ev.start_time,
                        source_id=ev.organizer_id,
                        target_id=ev.location_id,
                        layer="spatial",
                        weight=round(0.5 * duration_h, 6),
                    )
                )
                idx += 1
        return g

    def default_perturbation(self) -> Perturbation:
        if not self.events:
            return Perturbation(
                perturbation_id=f"{self.graph_id}:p0",
                timestamp=datetime.now(timezone.utc),
                targets=[],
                intensity=0.5,
                kind="calendar",
            )
        counts: dict[str, int] = {}
        latest = self.events[0].start_time
        for ev in self.events:
            counts[ev.organizer_id] = counts.get(ev.organizer_id, 0) + 1
            latest = max(latest, ev.start_time)
        target = max(counts.items(), key=lambda item: item[1])[0]
        return Perturbation(
            perturbation_id=f"{self.graph_id}:p0",
            timestamp=latest,
            targets=[target],
            intensity=1.0,
            kind="calendar",
        )

    def _ensure_actant(self, graph: LayeredGraph, node_id: str, kind: str) -> None:
        if node_id in graph.actants:
            return
        graph.actants[node_id] = Actant(actant_id=node_id, kind=kind, label=node_id)
