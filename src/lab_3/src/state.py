from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path
from .utils import now_iso, ensure_dir
import json


@dataclass
class AgentState:
    topic: str
    objective: str
    step_id: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    status: str = "running"
    stop_reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


def log_step(
    state: AgentState,
    action: str,
    payload: Dict[str, Any],
    result: str,
    next_reason: str = "",
    latency_ms: float | None = None
) -> None:
    state.history.append({
        "timestamp": now_iso(),
        "step_id": state.step_id,
        "action": action,
        "payload": payload,
        "result_preview": str(result)[:500],
        "n_sources": len(state.sources),
        "next_reason": next_reason,
        "latency_ms": latency_ms
    })
    state.step_id += 1


def save_trace(state: AgentState, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    trace = {
        "topic": state.topic,
        "objective": state.objective,
        "history": state.history,
        "n_sources": len(state.sources),
        "notes": state.notes,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "metrics": state.metrics,
        "final_answer": state.final_answer
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)