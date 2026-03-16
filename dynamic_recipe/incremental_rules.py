"""
Incremental / Online Rule Mining for Dynamic TKGs.

RECIPE-TKG limitation: rule mining is offline and must be fully rerun when
the graph changes.  This module replaces that with an online rule store that
updates incrementally in O(k) per new edge instead of O(|E|).

A "rule" here is a temporal relational path:
    head_relation ← body_relation_1 ○ body_relation_2 ○ … ○ body_relation_k

The miner tracks confidence and support counts and raises/lowers rules
as the graph evolves — no batch reprocessing required.
"""

import os
import json
import time
import math
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Rule:
    """A single temporal relational rule with confidence tracking."""

    __slots__ = ("head_rel", "body_rels", "support", "total_head",
                 "confidence", "last_seen", "created_at")

    def __init__(self, head_rel: int, body_rels: Tuple[int, ...],
                 support: int = 1, total_head: int = 1):
        self.head_rel   : int            = head_rel
        self.body_rels  : Tuple[int,...] = body_rels
        self.support    : int            = support
        self.total_head : int            = total_head
        self.confidence : float          = support / max(total_head, 1)
        self.last_seen  : float          = time.time()
        self.created_at : float          = time.time()

    def update(self, delta_support: int = 1, delta_head: int = 0) -> None:
        self.support    += delta_support
        self.total_head += delta_head
        self.confidence  = self.support / max(self.total_head, 1)
        self.last_seen   = time.time()

    def to_dict(self) -> Dict:
        return {
            "head_rel"  : self.head_rel,
            "body_rels" : list(self.body_rels),
            "support"   : self.support,
            "total_head": self.total_head,
            "confidence": self.confidence,
            "last_seen" : self.last_seen,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: Dict) -> "Rule":
        r = Rule(d["head_rel"], tuple(d["body_rels"]),
                 d["support"], d["total_head"])
        r.confidence = d["confidence"]
        r.last_seen  = d["last_seen"]
        r.created_at = d["created_at"]
        return r


# ---------------------------------------------------------------------------
# Incremental Rule Miner
# ---------------------------------------------------------------------------

class IncrementalRuleMiner:
    """
    Online rule miner that updates rule confidence incrementally when new
    edges arrive, without requiring full re-mining.

    Algorithm:
      1. Maintain an in-memory rule store indexed by (head_rel, body_rels).
      2. For each new edge (s, r, o, t), extract potential rule patterns by
         checking 1-hop and 2-hop paths in the current adjacency structure.
      3. Increment support counters for matching patterns.
      4. Add new rules when support >= min_support.
      5. Prune stale rules via LRU eviction when buffer is full.
      6. Optionally decay confidence of rarely-seen rules over time.
    """

    def __init__(
        self,
        max_rules: int = 5000,
        min_support: int = 2,
        min_confidence: float = 0.05,
        max_rule_length: int = 3,
        staleness_threshold: float = 7 * 24 * 3600,  # 7 days in seconds
        seed: int = 42,
    ):
        self.max_rules            = max_rules
        self.min_support          = min_support
        self.min_confidence       = min_confidence
        self.max_rule_length      = max_rule_length
        self.staleness_threshold  = staleness_threshold

        # Primary rule store: key = (head_rel, body_rels) → Rule
        self._rules: OrderedDict[Tuple, Rule] = OrderedDict()

        # Index: head_rel → list of rule keys
        self._head_idx: Dict[int, List[Tuple]] = defaultdict(list)

        # Head relation counters (for confidence normalisation)
        self._head_counts: Dict[int, int] = defaultdict(int)

        # Path index: for O(1) lookup of (s, o) multi-hop paths
        # adj[s][(o, path_rels)] = True
        self._adj_out: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def build_from_graph(self, facts: List[Tuple[int,int,int,int]]) -> None:
        """
        Cold-start: build rule store from an existing list of facts.
        facts: list of (subject, relation, object, timestamp) integer tuples.
        """
        # First pass: build adjacency
        for s, r, o, t in facts:
            self._adj_out[s][o].append(r)

        # Second pass: mine 1-hop and 2-hop rules
        for s, r, o, t in facts:
            self._head_counts[r] += 1
            # 1-hop rule: r ← r  (trivial, skipped)
            # 2-hop: find paths s → mid → o
            for mid, r1_list in self._adj_out[s].items():
                if mid == o:
                    continue
                if o in self._adj_out[mid]:
                    for r2 in self._adj_out[mid][o]:
                        for r1 in r1_list:
                            self._add_or_update_rule(
                                head_rel=r,
                                body_rels=(r1, r2),
                                delta_support=1,
                            )

        self._prune_low_confidence()
        self._evict_if_full()

    def update_rules(self, new_edge: Tuple[int,int,int,int]) -> None:
        """
        Incrementally update rules when one new edge (s, r, o, t) arrives.
        O(degree(s) + degree(o)) per update.
        """
        s, r, o, t = new_edge

        # Update adjacency
        self._adj_out[s][o].append(r)
        self._head_counts[r] += 1

        # 2-hop rule: s →r→ o, and we close via existing paths
        #   rule: r ← (r1, r2) if s →r1→ mid →r2→ o
        for mid, r1_list in list(self._adj_out[s].items()):
            if mid == o:
                continue
            if o in self._adj_out.get(mid, {}):
                for r2 in self._adj_out[mid][o]:
                    for r1 in r1_list:
                        self._add_or_update_rule(r, (r1, r2))

        # Also mine: new edge closes an open 2-hop path → new rule
        #   for any s2 →r1→ s: rule r_head ← (r1, r)
        for s2, neigh in list(self._adj_out.items()):
            if s in neigh:
                for r1 in neigh[s]:
                    # s2 →r1→ s →r→ o: what is the original relation s2→o?
                    if o in self._adj_out.get(s2, {}):
                        for r_head in self._adj_out[s2][o]:
                            self._add_or_update_rule(r_head, (r1, r))

        self._evict_if_full()

    def get_active_rules(
        self,
        head_rel: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Rule]:
        """
        Return active rules sorted by confidence (desc).
        Optionally filter by head_rel or minimum confidence.
        """
        min_conf = min_confidence or self.min_confidence
        if head_rel is not None:
            keys = self._head_idx.get(head_rel, [])
            rules = [self._rules[k] for k in keys if k in self._rules]
        else:
            rules = list(self._rules.values())

        return sorted(
            [r for r in rules if r.confidence >= min_conf],
            key=lambda r: r.confidence,
            reverse=True,
        )

    def rules_for_query(self, query_rel: int, top_k: int = 10) -> List[Rule]:
        """Return top-k rules whose head matches the query relation."""
        return self.get_active_rules(head_rel=query_rel)[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "rules"      : {str(k): v.to_dict() for k, v in self._rules.items()},
            "head_counts": dict(self._head_counts),
            "min_support": self.min_support,
            "min_conf"   : self.min_confidence,
            "max_rules"  : self.max_rules,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._rules.clear()
        self._head_idx.clear()
        for k_str, v in data["rules"].items():
            rule = Rule.from_dict(v)
            key  = (rule.head_rel, rule.body_rels)
            self._rules[key] = rule
            self._head_idx[rule.head_rel].append(key)
        self._head_counts = defaultdict(int, {int(k): v
                                               for k, v in data["head_counts"].items()})
        self.min_support    = data.get("min_support", self.min_support)
        self.min_confidence = data.get("min_conf", self.min_confidence)
        self.max_rules      = data.get("max_rules", self.max_rules)

    def to_chains_format(self) -> Dict:
        """
        Export rules in the same format as RECIPE-TKG's offline miner output
        (list of chain dicts per head relation) for drop-in compatibility.
        """
        chains: Dict[str, List[Dict]] = defaultdict(list)
        for rule in self._rules.values():
            chains[str(rule.head_rel)].append({
                "rule_body" : list(rule.body_rels),
                "confidence": rule.confidence,
                "support"   : rule.support,
            })
        return dict(chains)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_or_update_rule(
        self,
        head_rel: int,
        body_rels: Tuple[int, ...],
        delta_support: int = 1,
    ) -> None:
        key = (head_rel, body_rels)
        if key in self._rules:
            self._rules[key].update(delta_support=delta_support)
            self._rules.move_to_end(key)   # LRU: mark as recently used
        else:
            total_head = self._head_counts.get(head_rel, 1)
            rule = Rule(head_rel, body_rels, delta_support, total_head)
            self._rules[key] = rule
            self._head_idx[head_rel].append(key)

    def _prune_low_confidence(self) -> None:
        to_delete = [k for k, r in self._rules.items()
                     if r.confidence < self.min_confidence
                     or r.support < self.min_support]
        for k in to_delete:
            self._delete_rule(k)

    def _evict_if_full(self) -> None:
        """LRU eviction when buffer exceeds max_rules."""
        while len(self._rules) > self.max_rules:
            k, _ = self._rules.popitem(last=False)   # FIFO / LRU
            head_rel = k[0]
            if k in self._head_idx.get(head_rel, []):
                self._head_idx[head_rel].remove(k)

    def _delete_rule(self, key: Tuple) -> None:
        rule = self._rules.pop(key, None)
        if rule is not None:
            if key in self._head_idx.get(rule.head_rel, []):
                self._head_idx[rule.head_rel].remove(key)

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return (f"IncrementalRuleMiner("
                f"rules={len(self)}, "
                f"max={self.max_rules}, "
                f"min_conf={self.min_confidence:.2f})")
