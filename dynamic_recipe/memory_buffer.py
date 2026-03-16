"""
Memory Buffer for Continual Learning.

Implements:
  - ReservoirBuffer: Vitter's Algorithm R for uniform random sampling
  - PriorityReplayBuffer: recency + importance weighted replay
"""

import random
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reservoir Buffer (Vitter's Algorithm R)
# ---------------------------------------------------------------------------

class ReservoirBuffer:
    """
    Maintains a fixed-size reservoir that is a uniform random sample of all
    items ever added.  O(1) amortised insert, O(k) sample.

    Reference: Vitter, J. S. (1985). Random sampling with a reservoir.
    ACM Transactions on Mathematical Software, 11(1), 37-57.
    """

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self._n_seen = 0          # total items processed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def add(self, sample: Dict) -> None:
        """Add one sample; maintains the reservoir invariant."""
        self._n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            j = self._rng.randint(0, self._n_seen - 1)
            if j < self.capacity:
                self.buffer[j] = sample

    def add_batch(self, samples: List[Dict]) -> None:
        for s in samples:
            self.add(s)

    # ------------------------------------------------------------------
    def sample(self, n: int) -> List[Dict]:
        """Return up to n items drawn uniformly at random (without replacement)."""
        k = min(n, len(self.buffer))
        return self._rng.sample(self.buffer, k)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int = 1) -> bool:
        return len(self.buffer) >= min_size

    def state_dict(self) -> Dict:
        return {"capacity": self.capacity, "buffer": self.buffer,
                "n_seen": self._n_seen}

    def load_state_dict(self, d: Dict) -> None:
        self.capacity = d["capacity"]
        self.buffer   = d["buffer"]
        self._n_seen  = d["n_seen"]


# ---------------------------------------------------------------------------
# Priority Replay Buffer
# ---------------------------------------------------------------------------

class PriorityReplayBuffer(ReservoirBuffer):
    """
    Extends ReservoirBuffer with importance-weighted sampling.

    Each slot stores (sample, score) where
        score = α · recency_score + (1-α) · importance_score

    High-score samples are less likely to be evicted (soft priority).
    """

    def __init__(self, capacity: int, alpha: float = 0.5, seed: int = 42):
        super().__init__(capacity, seed)
        self.alpha = alpha
        self._scores: List[float] = []  # parallel list of scores
        self._step = 0                  # global step counter

    # ------------------------------------------------------------------
    def add(self, sample: Dict, importance: float = 1.0) -> None:  # type: ignore[override]
        self._n_seen += 1
        self._step   += 1
        recency = 1.0  # newest item gets recency = 1.0
        score = self.alpha * recency + (1 - self.alpha) * importance

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self._scores.append(score)
        else:
            # Replace the slot with lowest score (soft priority eviction)
            # with probability proportional to 1-score of the candidate slot
            min_idx = int(min(range(len(self._scores)),
                               key=lambda i: self._scores[i]))
            if self._scores[min_idx] < score:
                self.buffer[min_idx] = sample
                self._scores[min_idx] = score
            # Decay all recency scores
            self._scores = [s * 0.999 for s in self._scores]

    def add_batch(self, samples: List[Dict],
                  importances: Optional[List[float]] = None) -> None:
        if importances is None:
            importances = [1.0] * len(samples)
        for s, imp in zip(samples, importances):
            self.add(s, imp)

    # ------------------------------------------------------------------
    def sample(self, n: int) -> List[Dict]:
        """Weighted sampling proportional to scores (without replacement)."""
        if len(self.buffer) == 0:
            return []
        k = min(n, len(self.buffer))
        total = sum(self._scores)
        if total == 0 or len(set(self._scores)) == 1:
            return self._rng.sample(self.buffer, k)
        # Weighted sampling without replacement via tournament
        weights = [s / total for s in self._scores]
        indices = _weighted_sample_no_replacement(
            list(range(len(self.buffer))), weights, k, self._rng
        )
        return [self.buffer[i] for i in indices]

    def state_dict(self) -> Dict:
        d = super().state_dict()
        d["scores"] = self._scores
        d["step"]   = self._step
        d["alpha"]  = self.alpha
        return d

    def load_state_dict(self, d: Dict) -> None:
        super().load_state_dict(d)
        self._scores = d.get("scores", [1.0] * len(self.buffer))
        self._step   = d.get("step", 0)
        self.alpha   = d.get("alpha", self.alpha)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _weighted_sample_no_replacement(
    population: List[int],
    weights: List[float],
    k: int,
    rng: random.Random,
) -> List[int]:
    """Weighted sampling without replacement (A-Res algorithm)."""
    keys = [(rng.random() ** (1.0 / max(w, 1e-9)), idx)
            for idx, w in zip(population, weights)]
    keys.sort(reverse=True)
    return [idx for _, idx in keys[:k]]
