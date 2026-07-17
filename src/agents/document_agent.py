from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentEdge:
    weight: float
    activation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "activation_count": self.activation_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentEdge":
        return cls(
            weight=float(data["weight"]),
            activation_count=int(data.get("activation_count", 0)),
        )


@dataclass
class DocumentAgent:
    doc_id: str
    title: str
    text: str
    edges: dict[str, AgentEdge] = field(default_factory=dict)

    def add_edge(self, target_doc_id: str, weight: float) -> None:
        if target_doc_id in self.edges:
            self.edges[target_doc_id].weight = float(weight)
            return

        self.edges[target_doc_id] = AgentEdge(weight=float(weight))

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "edges": {
                target_doc_id: edge.to_dict()
                for target_doc_id, edge in self.edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentAgent":
        edges = {
            target_doc_id: AgentEdge.from_dict(edge_data)
            for target_doc_id, edge_data in data.get("edges", {}).items()
        }

        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            text=data["text"],
            edges=edges,
        )
