"""
Dynamic RECIPE (D-RECIPE): Continual TKG Forecasting
======================================================
Extends RECIPE-TKG to handle dynamic, incrementally updated
Temporal Knowledge Graphs with continual learning.

Core innovations over RECIPE-TKG:
  1. Incremental rule mining — no full re-mining on graph change
  2. Edge stream processor — O(1) adjacency updates
  3. EWC + replay + knowledge distillation — no catastrophic forgetting
  4. Adaptive RBMH sampler — incremental hop recalculation
  5. Full Apple MPS (M-series) compatibility
"""

__version__ = "1.0.0"
__author__  = "D-RECIPE PhD Extension"

# Lazy imports: only load submodules when accessed to avoid torch dependency
# at import time on systems where torch is not yet installed.
try:
    from dynamic_recipe.mps_utils        import get_device, get_dtype, load_model_mps
    from dynamic_recipe.continual_learner import EWCRegularizer, KnowledgeDistiller, ContinualTrainer
    from dynamic_recipe.adaptive_sampler import DynamicRBMHSampler
    from dynamic_recipe.dynamic_evaler   import DynamicEvaler
except ImportError:
    pass  # torch not installed; torch-dependent modules unavailable

# Pure-Python modules always available
from dynamic_recipe.memory_buffer    import ReservoirBuffer, PriorityReplayBuffer
from dynamic_recipe.incremental_rules import IncrementalRuleMiner
from dynamic_recipe.stream_processor import EdgeStreamProcessor

__all__ = [
    "get_device", "get_dtype", "load_model_mps",
    "ReservoirBuffer", "PriorityReplayBuffer",
    "IncrementalRuleMiner",
    "EdgeStreamProcessor",
    "EWCRegularizer", "KnowledgeDistiller", "ContinualTrainer",
    "DynamicRBMHSampler",
    "DynamicEvaler",
]
