"""
실행 결과 타입
- NodeResult:     그래프 탐색 결과 노드
- ExecutionStats: 실행 통계 (레이어별 타이밍, DB 호출 수 등)
- LayerTiming:    레이어별 소요 시간 기록
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeResult:
    id:   str
    type: str
    name: Optional[str]    = None
    text: Optional[str]    = None
    path: str              = ""
    meta: Dict[str, Any]   = field(default_factory=dict)


@dataclass
class LayerTiming:
    label:      str
    elapsed_ms: float
    extra:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStats:
    total_elapsed_s:    float             = 0.0
    hop_counts:         List[int]         = field(default_factory=list)
    path_summary:       str               = ""
    cache_hits:         int               = 0
    db_calls:           int               = 0
    pruned_total:       int               = 0
    layer_timings:      List[LayerTiming] = field(default_factory=list)

    def timing_summary(self) -> str:
        if not self.layer_timings:
            return "(no timing data)"
        lines = ["─── Layer Timing ──────────────────────────────────────────────"]
        for t in self.layer_timings:
            extra_str = ("  " + "  ".join(f"{k}={v}" for k, v in t.extra.items())) if t.extra else ""
            lines.append(f"  {t.label:<50} {t.elapsed_ms:>8.1f} ms{extra_str}")
        lines.append(f"  {'TOTAL':<50} {self.total_elapsed_s * 1000:>8.1f} ms")
        lines.append("────────────────────────────────────────────────────────────")
        return "\n".join(lines)
