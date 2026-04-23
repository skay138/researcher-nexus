"""
Beam Pruner
- hop explosion 2차 방어선 (1차: CypherCompiler의 LIMIT)
- 중간 결과를 의미적 유사도 기준으로 beam_width개로 압축
- 쿼리 문맥과 관련 없는 노드를 조기 제거
"""

from __future__ import annotations
from typing import Callable, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class BeamPruner:
    """
    각 hop 결과 노드를 원본 쿼리와의 의미적 유사도 기준으로 pruning.

    동작:
    1. node_ids가 beam_width 이하면 아무것도 안 함 (fast path)
    2. 초과하면 각 노드의 텍스트를 벡터화하여 쿼리와 코사인 유사도 계산
    3. 상위 beam_width개만 다음 hop으로 전달

    Args:
        vectorizer:  텍스트 → 벡터 변환기 (sentence-transformers 등)
        beam_width:  hop당 유지할 최대 노드 수
        fallback_strategy: 벡터화 실패 시 전략 ("truncate" | "error")
    """

    def __init__(
        self,
        vectorizer=None,
        beam_width: int = 50,
        fallback_strategy: str = "truncate",
    ):
        self.vectorizer = vectorizer
        self.beam_width = beam_width
        self.fallback_strategy = fallback_strategy

    def prune(
        self,
        node_ids: List[str],
        query_context: str,
        node_text_fetcher: Callable[[List[str]], List[str]],
        beam_width: Optional[int] = None,
    ) -> List[str]:
        """
        Args:
            node_ids:          현재 hop 결과 노드 ID 목록
            query_context:     원본 사용자 쿼리 (유사도 기준)
            node_text_fetcher: ID 목록 → 텍스트 목록 변환 함수
            beam_width:        요청별 오버라이드 (None이면 self.beam_width 사용)

        Returns:
            pruning 후 노드 ID 목록 (beam_width개 이하)
        """
        bw = beam_width if beam_width is not None else self.beam_width
        if len(node_ids) <= bw:
            logger.debug("BeamPruner: %d nodes ≤ beam_width(%d), skip", len(node_ids), bw)
            return node_ids

        if self.vectorizer is None:
            return self._fallback(node_ids, bw)

        try:
            return self._semantic_prune(node_ids, query_context, node_text_fetcher, bw)
        except Exception as e:
            logger.warning("BeamPruner semantic prune failed: %s", e)
            return self._fallback(node_ids, bw)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _semantic_prune(
        self,
        node_ids: List[str],
        query_context: str,
        node_text_fetcher: Callable,
        beam_width: int,
    ) -> List[str]:
        import numpy as np

        t0 = time.time()

        # 노드 텍스트 로드
        node_texts = node_text_fetcher(node_ids)

        # 벡터화
        query_vec = self.vectorizer.encode(query_context, normalize_embeddings=True)
        node_vecs = self.vectorizer.encode(node_texts,   normalize_embeddings=True)

        # 코사인 유사도 (정규화된 벡터이므로 내적 = cosine)
        scores: "np.ndarray" = node_vecs @ query_vec

        # 상위 beam_width개 인덱스
        top_k = min(beam_width, len(node_ids))
        top_indices = scores.argsort()[-top_k:][::-1]
        pruned = [node_ids[i] for i in top_indices]

        elapsed = time.time() - t0
        logger.info(
            "BeamPruner: %d → %d nodes (pruned %d) in %.2fs | min_score=%.3f max_score=%.3f",
            len(node_ids), len(pruned), len(node_ids) - len(pruned),
            elapsed, float(scores.min()), float(scores.max()),
        )
        return pruned

    def _fallback(self, node_ids: List[str], beam_width: int) -> List[str]:
        if self.fallback_strategy == "error":
            raise RuntimeError(
                f"Beam pruning failed and fallback is disabled. "
                f"Node count {len(node_ids)} exceeds beam_width {beam_width}."
            )
        logger.warning(
            "BeamPruner fallback: truncating %d → %d (no semantic scoring)",
            len(node_ids), beam_width,
        )
        return node_ids[:beam_width]
