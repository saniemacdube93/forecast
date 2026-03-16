"""
Edge Stream Processor for Dynamic TKGs.

Handles edges arriving as a stream (real-time or simulated) with:
  - O(1) adjacency list updates
  - Time-bucketed snapshots for temporal range queries
  - New entity / relation emergence detection
  - Streaming simulation from static dataset files
"""

import os
import json
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple


Fact = Tuple[int, int, int, int]   # (subject, relation, object, timestamp)


class EdgeStreamProcessor:
    """
    Maintains a dynamic graph that supports streaming edge ingestion.

    Internal structures:
      adj_out[s]  = [(r, o, t), ...]    out-going edges from s
      adj_in[o]   = [(r, s, t), ...]    in-coming edges into o
      buckets[t]  = [Fact, ...]         facts indexed by discrete timestamp
      entity_set  = set of all known entities
      rel_set     = set of all known relations
    """

    def __init__(self, time_granularity: int = 1):
        """
        Parameters
        ----------
        time_granularity : int
            Temporal resolution in the dataset's native units.
            1 = daily (ICEWS), 1 = 15-min (GDELT), 1 = yearly (YAGO).
        """
        self.time_granularity = time_granularity

        # Adjacency (mutable, O(1) insert)
        self.adj_out: Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)
        self.adj_in:  Dict[int, List[Tuple[int,int,int]]] = defaultdict(list)

        # Time-indexed buckets
        self.buckets: Dict[int, List[Fact]] = defaultdict(list)

        # Entity and relation registries
        self.entity_set: Set[int] = set()
        self.rel_set:    Set[int] = set()

        # Stream counters
        self._total_ingested = 0
        self._new_entities_seen = 0

        # All facts in insertion order (for replay / sampling)
        self._all_facts: List[Fact] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_edge(self, s: int, r: int, o: int, t: int) -> bool:
        """
        Ingest one edge.  Returns True if any new entity or relation appeared.
        """
        new_item = False
        for eid in (s, o):
            if eid not in self.entity_set:
                self.entity_set.add(eid)
                self._new_entities_seen += 1
                new_item = True
        if r not in self.rel_set:
            self.rel_set.add(r)
            new_item = True

        self.adj_out[s].append((r, o, t))
        self.adj_in[o].append((r, s, t))
        self.buckets[t].append((s, r, o, t))
        self._all_facts.append((s, r, o, t))
        self._total_ingested += 1
        return new_item

    def ingest_batch(self, edges: List[Fact]) -> int:
        """Ingest a list of (s, r, o, t) tuples; returns count of new entities."""
        before = self._new_entities_seen
        for s, r, o, t in edges:
            self.ingest_edge(s, r, o, t)
        return self._new_entities_seen - before

    def ingest_from_file(self, path: str) -> int:
        """Load a .txt file of whitespace-separated (s, r, o, t) integer rows."""
        facts = _read_facts_file(path)
        return self.ingest_batch(facts)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_snapshot(self, t_start: int, t_end: int) -> List[Fact]:
        """Return all facts with timestamp in [t_start, t_end]."""
        facts = []
        for t in range(t_start, t_end + 1):
            facts.extend(self.buckets.get(t, []))
        return facts

    def get_facts_before(self, t: int, max_facts: Optional[int] = None) -> List[Fact]:
        """Return all facts with timestamp < t (temporal history)."""
        out = []
        for ts, batch in sorted(self.buckets.items()):
            if ts >= t:
                break
            out.extend(batch)
        if max_facts is not None:
            out = out[-max_facts:]   # most recent max_facts
        return out

    def get_neighbors_1hop(self, entity: int, direction: str = "out") -> List[Tuple[int,int,int]]:
        """
        Return 1-hop neighbours.
        direction: 'out' → (r, o, t) list; 'in' → (r, s, t) list; 'both'
        """
        if direction == "out":
            return list(self.adj_out.get(entity, []))
        elif direction == "in":
            return list(self.adj_in.get(entity, []))
        else:
            return (list(self.adj_out.get(entity, [])) +
                    list(self.adj_in.get(entity, [])))

    def get_neighbors_khop(self, entity: int, k: int = 2) -> Set[int]:
        """
        BFS up to k hops; returns set of reachable entity IDs (excluding source).
        """
        visited: Set[int] = {entity}
        frontier: Set[int] = {entity}
        for _ in range(k):
            next_frontier: Set[int] = set()
            for e in frontier:
                for _, nb, _ in self.adj_out.get(e, []):
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
                for _, nb, _ in self.adj_in.get(e, []):
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(entity)
        return visited

    def hop_distance(self, src: int, dst: int, max_hops: int = 4) -> int:
        """
        BFS shortest-hop distance from src to dst.
        Returns max_hops + 1 if unreachable within max_hops.
        """
        if src == dst:
            return 0
        visited = {src}
        frontier = {src}
        for h in range(1, max_hops + 1):
            nxt = set()
            for e in frontier:
                for _, nb, _ in self.adj_out.get(e, []):
                    if nb == dst:
                        return h
                    if nb not in visited:
                        nxt.add(nb)
                        visited.add(nb)
            frontier = nxt
            if not frontier:
                break
        return max_hops + 1

    def facts_involving(self, entity: int) -> List[Fact]:
        """All facts where entity appears as subject or object."""
        out_facts = [(s, r, o, t) for r, o, t in self.adj_out.get(entity, [])
                     for s in [entity]]
        in_facts  = [(s, r, o, t) for r, s, t in self.adj_in.get(entity, [])
                     for o in [entity]]
        return out_facts + in_facts

    # ------------------------------------------------------------------
    # Stream simulation from static dataset
    # ------------------------------------------------------------------

    def simulate_stream(
        self,
        dataset_path: str,
        chunk_size: int = 500,
        shuffle_within_chunk: bool = False,
        seed: int = 42,
    ) -> Iterator[Tuple[int, List[Fact]]]:
        """
        Simulate a stream from a static TKG dataset.

        Reads <dataset_path>/train.txt, sorts by timestamp, and yields
        (chunk_idx, batch_of_facts) tuples.  Caller ingests each batch
        and trains continually.
        """
        import random as _random
        rng = _random.Random(seed)

        facts = _read_facts_file(os.path.join(dataset_path, "train.txt"))
        facts.sort(key=lambda x: x[3])   # sort by timestamp

        for chunk_idx, start in enumerate(range(0, len(facts), chunk_size)):
            batch = facts[start: start + chunk_size]
            if shuffle_within_chunk:
                rng.shuffle(batch)
            yield chunk_idx, batch

    def simulate_stream_from_list(
        self,
        facts: List[Fact],
        chunk_size: int = 500,
    ) -> Iterator[Tuple[int, List[Fact]]]:
        """Same as simulate_stream but takes a pre-loaded list."""
        facts_sorted = sorted(facts, key=lambda x: x[3])
        for chunk_idx, start in enumerate(range(0, len(facts_sorted), chunk_size)):
            yield chunk_idx, facts_sorted[start: start + chunk_size]

    # ------------------------------------------------------------------
    # Snapshot / state management
    # ------------------------------------------------------------------

    def checkpoint(self) -> Dict:
        """Export a minimal checkpoint (entity/rel sets + counts)."""
        return {
            "total_ingested"    : self._total_ingested,
            "new_entities_seen" : self._new_entities_seen,
            "n_entities"        : len(self.entity_set),
            "n_relations"       : len(self.rel_set),
            "timestamps"        : sorted(self.buckets.keys()),
        }

    def get_all_facts(self) -> List[Fact]:
        return list(self._all_facts)

    def stats(self) -> str:
        return (
            f"StreamProcessor: {self._total_ingested} facts ingested, "
            f"{len(self.entity_set)} entities, {len(self.rel_set)} relations, "
            f"{len(self.buckets)} timestamps"
        )

    def __repr__(self) -> str:
        return self.stats()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read_facts_file(path: str) -> List[Fact]:
    """Read whitespace-separated (s r o t) int rows from a .txt file."""
    facts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    facts.append((s, r, o, t))
                except ValueError:
                    continue
    return facts


def load_dataset_splits(
    dataset_path: str,
) -> Tuple[List[Fact], List[Fact], List[Fact]]:
    """Return (train_facts, valid_facts, test_facts) from a dataset directory."""
    train = _read_facts_file(os.path.join(dataset_path, "train.txt"))
    valid = _read_facts_file(os.path.join(dataset_path, "valid.txt"))
    test  = _read_facts_file(os.path.join(dataset_path, "test.txt"))
    return train, valid, test
