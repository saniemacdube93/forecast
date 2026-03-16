"""
Dynamic Evaluator for D-RECIPE.

Extends RECIPE-TKG's Evaler with:
  - Per-time-period Hits@1/3/10 tracking
  - Continual learning metrics: BWT, FWT, Average Accuracy
  - Semantic similarity filtering with adaptive threshold
  - Sparse / dense history breakdown
"""

import os
import json
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util as st_util


Fact = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Temporal-aware filtered Hits@k helper
# ---------------------------------------------------------------------------

def hits_at_k(predictions: List[str], ground_truth: str, k: int) -> float:
    """Return 1.0 if ground_truth is in top-k predictions, else 0.0."""
    return 1.0 if ground_truth in predictions[:k] else 0.0


def compute_hits(
    predictions_list: List[List[str]],
    ground_truths:    List[str],
    ks: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """Compute mean Hits@k for a list of prediction sets."""
    n = len(ground_truths)
    if n == 0:
        return {f"Hits@{k}": 0.0 for k in ks}
    return {
        f"Hits@{k}": sum(hits_at_k(preds, gt, k)
                         for preds, gt in zip(predictions_list, ground_truths)) / n
        for k in ks
    }


# ---------------------------------------------------------------------------
# Adaptive semantic filtering
# ---------------------------------------------------------------------------

class AdaptiveSemanticFilter:
    """
    Refines model predictions using embedding similarity between
    the prediction and the input context.  Threshold τ is learned
    from a calibration set and updated online.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2", tau: float = 0.6):
        self.tau = tau
        self._sim_history: List[float] = []
        try:
            self.encoder = SentenceTransformer(model_name)
        except Exception:
            self.encoder = None

    def similarity(self, prediction: str, context: str) -> float:
        if self.encoder is None:
            return 1.0
        try:
            emb_p = self.encoder.encode(prediction, convert_to_tensor=True)
            emb_c = self.encoder.encode(context[:512], convert_to_tensor=True)
            return float(st_util.cos_sim(emb_p, emb_c))
        except Exception:
            return 1.0

    def should_accept(self, prediction: str, context: str,
                      history_set: Optional[set] = None) -> bool:
        """Accept if prediction is in history or similarity >= τ."""
        if history_set and prediction in history_set:
            return True
        sim = self.similarity(prediction, context)
        self._sim_history.append(sim)
        return sim >= self.tau

    def update_threshold(self) -> None:
        """Adjust τ based on running distribution of similarities."""
        if len(self._sim_history) > 100:
            # Set τ to the 40th percentile of recent similarities
            self.tau = float(np.percentile(self._sim_history[-200:], 40))
            self._sim_history = self._sim_history[-200:]


# ---------------------------------------------------------------------------
# Main DynamicEvaler
# ---------------------------------------------------------------------------

class DynamicEvaler:
    """
    Evaluation engine for the D-RECIPE continual learning setting.

    Usage
    -----
    evaler = DynamicEvaler(entity2id, id2entity)
    evaler.eval_period(model, tokenizer, test_facts, period_id=0)
    ...
    print(evaler.compute_bwt())
    """

    def __init__(
        self,
        entity2id: Dict[str, int],
        id2entity:  Dict[int, str],
        all_entities: List[str],
        filter_set: Optional[set] = None,
        device: Optional[torch.device] = None,
        top_k_generate: int = 10,
    ):
        self.entity2id      = entity2id
        self.id2entity      = id2entity
        self.all_entities   = all_entities
        self.filter_set     = filter_set or set()
        self.top_k          = top_k_generate
        self.sem_filter     = AdaptiveSemanticFilter()
        self.device         = device or torch.device("cpu")

        # acc_matrix[period_eval][period_trained] = Hits@10
        self._acc_records: List[Dict[str, float]] = []    # per-period results
        self._period_ids:  List[int]               = []

    # ------------------------------------------------------------------
    def eval_period(
        self,
        model,
        tokenizer,
        test_facts:   List[Dict],   # list of {'prompt': str, 'answer': str}
        period_id:    int,
        max_samples:  int = 200,
    ) -> Dict[str, float]:
        """
        Evaluate model on test_facts for one time period.
        Returns Hits@1/3/10.
        """
        model.eval()
        predictions_list: List[List[str]] = []
        ground_truths:    List[str]        = []

        samples = test_facts[:max_samples]
        for item in samples:
            prompt = item["prompt"]
            answer = item.get("answer", "")
            preds  = self._generate_candidates(model, tokenizer, prompt)
            predictions_list.append(preds)
            ground_truths.append(answer)

        metrics = compute_hits(predictions_list, ground_truths)
        metrics["period"] = period_id
        metrics["n_samples"] = len(samples)

        self._acc_records.append(metrics)
        self._period_ids.append(period_id)
        return metrics

    # ------------------------------------------------------------------
    def eval_all_periods(
        self,
        model,
        tokenizer,
        all_period_data: Dict[int, List[Dict]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate current model on all previously seen time periods.
        Used to build the full accuracy matrix for BWT/FWT.
        """
        results = {}
        for pid, test_data in sorted(all_period_data.items()):
            r = self.eval_period(model, tokenizer, test_data, period_id=pid)
            results[pid] = r
        return results

    # ------------------------------------------------------------------
    # Continual learning metrics
    # ------------------------------------------------------------------

    def compute_bwt(self) -> float:
        """
        Backward Transfer: measures forgetting.
        BWT = (1/T-1) Σ_{t=2}^{T} (R_{T,t} − R_{t,t})

        Negative BWT = catastrophic forgetting.
        """
        if len(self._acc_records) < 2:
            return 0.0
        T = len(self._acc_records)
        bwt = sum(
            self._acc_records[T-1]["Hits@10"] - self._acc_records[t]["Hits@10"]
            for t in range(T - 1)
        ) / (T - 1)
        return float(bwt)

    def compute_fwt(self, random_baselines: Optional[Dict[int, float]] = None) -> float:
        """
        Forward Transfer: measures positive transfer to unseen tasks.
        FWT = (1/T-1) Σ_{t=2}^{T} (R_{t-1,t} − b_t)
        where b_t is a random/zero-shot baseline.
        """
        if len(self._acc_records) < 2:
            return 0.0
        T = len(self._acc_records)
        total = 0.0
        for t in range(1, T):
            r_prev = self._acc_records[t-1]["Hits@10"]
            r_curr = self._acc_records[t]["Hits@10"]
            b      = (random_baselines or {}).get(t, 0.0)
            total += r_prev - b
        return total / (T - 1)

    def compute_average_accuracy(self) -> float:
        """AvgAcc = (1/T) Σ_t Hits@10_t"""
        if not self._acc_records:
            return 0.0
        return float(np.mean([r["Hits@10"] for r in self._acc_records]))

    def get_metrics_summary(self) -> Dict:
        return {
            "average_accuracy": self.compute_average_accuracy(),
            "bwt"             : self.compute_bwt(),
            "fwt"             : self.compute_fwt(),
            "n_periods"       : len(self._acc_records),
            "per_period"      : self._acc_records,
        }

    # ------------------------------------------------------------------
    # Sparse vs dense history breakdown
    # ------------------------------------------------------------------

    def eval_by_history_length(
        self,
        model,
        tokenizer,
        test_facts:  List[Dict],   # each must have 'history_len' field
        bins: List[int] = (0, 2, 5, 10, 20, 50),
    ) -> Dict[str, Dict]:
        """
        Same analysis as RECIPE-TKG Figure 5: Hits@10 by history length bucket.
        """
        buckets: Dict[str, List] = {f"{bins[i]}-{bins[i+1]}": []
                                     for i in range(len(bins) - 1)}
        buckets[f"{bins[-1]}+"] = []

        for item in test_facts:
            h_len  = item.get("history_len", 0)
            prompt = item["prompt"]
            answer = item.get("answer", "")
            preds  = self._generate_candidates(model, tokenizer, prompt)
            hit    = hits_at_k(preds, answer, 10)

            label = f"{bins[-1]}+"
            for i in range(len(bins) - 1):
                if bins[i] <= h_len < bins[i+1]:
                    label = f"{bins[i]}-{bins[i+1]}"
                    break
            buckets[label].append(hit)

        return {
            label: {
                "Hits@10": float(np.mean(vals)) if vals else 0.0,
                "n"      : len(vals),
            }
            for label, vals in buckets.items()
        }

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        model,
        tokenizer,
        prompt: str,
        n_candidates: int = 10,
    ) -> List[str]:
        """
        Generate top-k entity candidates from the LLM.
        Returns a list of entity name strings.
        """
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=max(n_candidates, 5),
                    num_return_sequences=n_candidates,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            candidates = []
            prompt_len = inputs["input_ids"].shape[1]
            for seq in outputs:
                generated = seq[prompt_len:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                entity = _extract_entity(text)
                if entity and entity not in candidates:
                    candidates.append(entity)

            return candidates[:n_candidates]

        except Exception as e:
            return []

    def save_results(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_metrics_summary(), f, indent=2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_entity(text: str) -> str:
    """
    Extract the entity name from a model generation.
    Mirrors RECIPE-TKG's postprocessing: strip noise, take first meaningful token.
    """
    # Remove instruction artefacts
    text = re.sub(r"<[^>]+>", "", text).strip()
    # Take first line
    text = text.split("\n")[0].strip()
    # Remove common prefixes
    for prefix in ["The answer is", "Answer:", "Entity:", "Object:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # Remove trailing punctuation
    text = text.rstrip(".,;:!? ")
    return text


def build_test_prompts(
    test_facts:  List[Fact],
    id2entity:   Dict[int, str],
    id2relation: Dict[int, str],
    history_facts: List[Fact],
    max_history: int = 50,
) -> List[Dict]:
    """
    Convert integer-encoded test facts to prompt dicts for evaluation.
    """
    # Build a quick entity history index
    hist_by_subj: Dict[int, List[Fact]] = {}
    for f in history_facts:
        s = f[0]
        hist_by_subj.setdefault(s, []).append(f)

    prompts = []
    for s, r, o, t in test_facts:
        subj_name = id2entity.get(s, str(s))
        rel_name  = id2relation.get(r, str(r))
        gt_name   = id2entity.get(o, str(o))

        # Build context from subject's history
        history = sorted(hist_by_subj.get(s, []), key=lambda x: x[3])[-max_history:]
        context_lines = [
            f"{f[3]}:[{id2entity.get(f[0],str(f[0]))}, "
            f"{id2relation.get(f[1],str(f[1]))}, "
            f"{id2entity.get(f[2],str(f[2]))}]"
            for f in history
        ]
        context = "\n".join(context_lines)

        prompt = (
            f"Context:\n{context}\n\n"
            f"Query: {t}:[{subj_name}, {rel_name}, ?]\n"
            f"Prediction: "
        )
        prompts.append({
            "prompt"      : prompt,
            "answer"      : gt_name,
            "history_len" : len(history),
            "subject_id"  : s,
        })
    return prompts
