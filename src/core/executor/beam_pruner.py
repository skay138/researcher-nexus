"""
Beam Pruner
- hop explosion 방어: Cypher LIMIT(1차) + beam_width 절단(2차)
- 점수 전파(ScorePropagate)로 순위 결정 — 의미 유사도 불필요
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class BeamPruner:
    """
    hop당 유지할 최대 노드 수(beam_width)를 보유.
    실제 pruning은 ExecutionEngine._run_single_hop 의 ScorePropagate 로직이 담당.
    """

    def __init__(self, beam_width: int = 50):
        self.beam_width = beam_width
