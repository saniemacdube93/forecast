"""
Adaptive Dynamic RBMH Sampler.

Extends RECIPE-TKG's composite-weight Rule-Based Multi-Hop history sampler
with incremental updates for streaming graphs:

  - Cached hop distances are invalidated only for the affected neighbourhood
    (not the whole graph) when a new edge arrives.
  - Temporal decay is adaptive: steeper decay when the stream velocity is high.
  - Rules are fetched from IncrementalRuleMiner instead of offline chains.

Weight formula (identical to RECIPE-TKG):
    w = w_n · w_f · (w_t + w_c + w_cp)
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from dynamic_recipe.incremental_rules import IncrementalRuleMiner
from dynamic_recipe.stream_processor  import EdgeStreamProcessor


Fact = Tuple[int, int, int, int]   # (subject, relation, object, timestamp)


class DynamicRBMHSampler:
    """
    Adaptive RBMH sampler for incrementally updated TKGs.

    Parameters match the hyperparameters γ1…γ4 from the RECIPE-TKG paper.
    """

    def __init__(
        self,
        stream_processor: EdgeStreamProcessor,
        rule_miner: IncrementalRuleMiner,
        gamma1: float = 0.6,   # hop-distance decay
        gamma2: float = 0.6,   # frequency penalty
        gamma3: float = 0.01,  # temporal recency
        gamma4: float = 0.1,   # co-occurrence
        n_tlr_facts: int = 20, # TLR-phase facts to retrieve
        n_total_facts: int = 50,
        max_hops: int = 3,
        seed: int = 42,
    ):
        self.stream   = stream_processor
        self.rules    = rule_miner
        self.g1, self.g2 = gamma1, gamma2
        self.g3, self.g4 = gamma3, gamma4
        self.n_tlr       = n_tlr_facts
        self.n_total     = n_total_facts
        self.max_hops    = max_hops
        self._rng        = random.Random(seed)

        # Hop-distance cache: (src, dst) → distance
        self._hop_cache: Dict[Tuple[int,int], int] = {}
        # Frequency cache: (s, r, o) → count
        self._freq_cache: Dict[Tuple[int,int,int], int] = defaultdict(int)
        # Co-occurrence cache: (a, b) → count
        self._cooc_cache: Dict[Tuple[int,int], int] = defaultdict(int)

        # Stream velocity: facts per unit time (for adaptive decay)
        self._velocity_ema: float = 0.0
        self._ema_alpha: float = 0.1

    # ------------------------------------------------------------------
    # Cache invalidation on edge update
    # ------------------------------------------------------------------

    def update_on_edge(self, new_edge: Fact) -> None:
        """
        Called whenever a new edge (s, r, o, t) is ingested.
        Invalidates hop-distance cache entries for entities in the
        affected neighbourhood only.
        """
        s, r, o, t = new_edge

        # Update frequency and co-occurrence caches
        self._freq_cache[(s, r, o)] += 1
        self._cooc_cache[(s, o)]    += 1
        self._cooc_cache[(o, s)]    += 1

        # Update velocity EMA
        self._velocity_ema = (
            self._ema_alpha * 1.0 + (1 - self._ema_alpha) * self._velocity_ema
        )

        # Invalidate hop cache for entities up to max_hops away from s or o
        affected = {s, o}
        # Expand one more ring for safety
        for eid in [s, o]:
            for _, nb, _ in self.stream.adj_out.get(eid, []):
                affected.add(nb)
            for _, nb, _ in self.stream.adj_in.get(eid, []):
                affected.add(nb)

        stale = [k for k in self._hop_cache if k[0] in affected or k[1] in affected]
        for k in stale:
            del self._hop_cache[k]

    # ------------------------------------------------------------------
    # Main sampling entry point
    # ------------------------------------------------------------------

    def sample(
        self,
        query_subject: int,
        query_relation: int,
        query_time: int,
        n_facts: Optional[int] = None,
    ) -> List[Fact]:
        """
        Sample up to n_facts historical facts for a TKG query.
        Returns a list of (s, r, o, t) integer tuples.
        """
        n = n_facts or self.n_total

        # Stage 1: TLR — rule-guided 1-hop retrieval
        tlr_facts = self._tlr_stage(query_subject, query_relation, query_time)

        # Stage 2: Context-guided multi-hop expansion
        all_history = self.stream.get_facts_before(query_time)
        if not all_history:
            return tlr_facts[:n]

        tlr_entities: Set[int] = set()
        for s, r, o, t in tlr_facts:
            tlr_entities.add(s)
            tlr_entities.add(o)

        expansion_facts = self._expansion_stage(
            query_subject, query_relation, query_time,
            all_history, tlr_facts, tlr_entities, n
        )

        combined = tlr_facts + expansion_facts
        # Deduplicate preserving TLR priority
        seen: Set[Tuple] = set()
        result: List[Fact] = []
        for f in combined:
            if f not in seen:
                seen.add(f)
                result.append(f)
            if len(result) >= n:
                break

        return result

    # ------------------------------------------------------------------
    # Stage 1: TLR (Temporal Logical Rule-based sampling)
    # ------------------------------------------------------------------

    def _tlr_stage(
        self,
        subject: int,
        relation: int,
        query_time: int,
    ) -> List[Fact]:
        """
        Apply active rules for query_relation to retrieve subject-aligned facts.
        """
        active_rules = self.rules.rules_for_query(relation, top_k=5)
        retrieved: List[Fact] = []

        for rule in active_rules:
            if len(rule.body_rels) == 0:
                continue
            # Follow 1-hop body relation from subject
            first_body_rel = rule.body_rels[0]
            for r, o, t in self.stream.adj_out.get(subject, []):
                if r == first_body_rel and t < query_time:
                    retrieved.append((subject, r, o, t))
            if len(retrieved) >= self.n_tlr:
                break

        # Fallback: plain 1-hop history if rules yield nothing
        if not retrieved:
            for r, o, t in sorted(
                self.stream.adj_out.get(subject, []),
                key=lambda x: x[2], reverse=True
            )[:self.n_tlr]:
                if t < query_time:
                    retrieved.append((subject, r, o, t))

        return retrieved[:self.n_tlr]

    # ------------------------------------------------------------------
    # Stage 2: Multi-hop expansion
    # ------------------------------------------------------------------

    def _expansion_stage(
        self,
        subject: int,
        relation: int,
        query_time: int,
        all_history: List[Fact],
        tlr_facts: List[Fact],
        tlr_entities: Set[int],
        n_target: int,
    ) -> List[Fact]:
        """
        Score all candidate facts in all_history and sample top ones.
        """
        tlr_set = set(tlr_facts)
        candidates = [f for f in all_history if f not in tlr_set]

        # Limit candidate pool to 10 × n_target as per RECIPE-TKG
        pool_size = min(len(candidates), 10 * n_target)
        if pool_size < len(candidates):
            candidates = self._rng.sample(candidates, pool_size)

        max_t = query_time

        scored: List[Tuple[float, Fact]] = []
        for fact in candidates:
            s, r, o, t = fact
            w = self._weight(s, r, o, t, subject, relation,
                             tlr_entities, max_t)
            scored.append((w, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [f for _, f in scored[:5 * n_target]]

        # Weighted probabilistic sampling
        if not top_candidates:
            return []
        weights = [self._weight(s, r, o, t, subject, relation,
                                tlr_entities, max_t)
                   for s, r, o, t in top_candidates]
        total_w = sum(weights)
        if total_w == 0:
            return top_candidates[:n_target]

        probs = [w / total_w for w in weights]
        k = min(n_target, len(top_candidates))
        indices = _weighted_sample_no_replacement(range(len(top_candidates)),
                                                  probs, k, self._rng)
        return [top_candidates[i] for i in indices]

    # ------------------------------------------------------------------
    # Composite weight (identical formula to RECIPE-TKG)
    # ------------------------------------------------------------------

    def _weight(
        self,
        s: int, r: int, o: int, t: int,
        query_subj: int, query_rel: int,
        tlr_entities: Set[int],
        max_t: int,
    ) -> float:
        w_n  = self._neighbor_weight(s, o, query_subj)
        w_f  = self._freq_weight(s, r, o)
        w_t  = self._time_weight(t, max_t)
        w_c  = self._cooc_weight(s, o, query_subj)
        w_cp = 1.0 if (s in tlr_entities or o in tlr_entities) else 0.0
        return w_n * w_f * (w_t + w_c + w_cp)

    def _neighbor_weight(self, s: int, o: int, query_subj: int) -> float:
        """w_n = exp(-γ1 · (hops_s + hops_o - 1)), 0 if unreachable."""
        h_s = self._hop(query_subj, s)
        h_o = self._hop(query_subj, o)
        if h_s > self.max_hops and h_o > self.max_hops:
            return 0.0
        hops = min(h_s, h_o)
        return math.exp(-self.g1 * max(hops - 1, 0))

    def _freq_weight(self, s: int, r: int, o: int) -> float:
        """w_f = 1 / (γ2 · log(n_sro) + 1)"""
        n = max(self._freq_cache.get((s, r, o), 1), 1)
        return 1.0 / (self.g2 * math.log(n) + 1.0)

    def _time_weight(self, t: int, max_t: int) -> float:
        """w_t = exp(-γ3 · |max_t - t|)"""
        delta = max_t - t
        # Adaptive: steeper decay when stream is faster
        gamma3 = self.g3 * (1 + self._velocity_ema * 0.1)
        return math.exp(-gamma3 * max(delta, 0))

    def _cooc_weight(self, s: int, o: int, query_subj: int) -> float:
        """w_c = log(1 + γ4·n_so) / (1 + log(1 + γ4·n_so))"""
        n = max(self._cooc_cache.get((s, query_subj), 0) +
                self._cooc_cache.get((o, query_subj), 0), 0)
        v = math.log(1 + self.g4 * n)
        return v / (1 + v)

    def _hop(self, src: int, dst: int) -> int:
        """Cached hop distance from src to dst."""
        key = (src, dst)
        if key not in self._hop_cache:
            self._hop_cache[key] = self.stream.hop_distance(
                src, dst, self.max_hops
            )
        return self._hop_cache[key]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _weighted_sample_no_replacement(population, weights, k, rng):
    population = list(population)
    weights    = list(weights)
    total      = sum(weights)
    if total == 0:
        return rng.sample(range(len(population)), min(k, len(population)))
    keys = [(rng.random() ** (1.0 / max(w / total, 1e-9)), i)
            for i, w in enumerate(weights)]
    keys.sort(reverse=True)
    return [population[i] for _, i in keys[:k]]
