"""
Continual Learning Module for D-RECIPE.

Implements three complementary mechanisms to prevent catastrophic forgetting
when the model adapts to a growing/changing TKG:

  1. EWC (Elastic Weight Consolidation)
     - Computes diagonal Fisher information matrix over LoRA parameters.
     - Adds a quadratic penalty L_EWC = Σ F_i(θ_i − θ*_i)² that slows
       learning of parameters important for previous tasks.

  2. Knowledge Distillation (KD)
     - Keeps a frozen snapshot (teacher) of the model before adaptation.
     - KD loss = KL(softmax(z_teacher/T) ∥ softmax(z_student/T)) encourages
       the updated model to preserve its previous output distribution.

  3. Experience Replay
     - Maintains a PriorityReplayBuffer of past training samples.
     - New-data batches are mixed with replay samples before each gradient step.

Combined loss:
    L = L_CE  +  α · L_contrastive  +  β · L_EWC  +  γ · L_KD
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dynamic_recipe.memory_buffer import PriorityReplayBuffer
from dynamic_recipe.mps_utils import get_device, empty_cache


# ---------------------------------------------------------------------------
# EWC Regulariser
# ---------------------------------------------------------------------------

class EWCRegularizer:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).
    Computes per-parameter Fisher importance and penalises drift.
    """

    def __init__(self, lambda_ewc: float = 0.1):
        self.lambda_ewc = lambda_ewc
        self.fisher:    Optional[Dict[str, torch.Tensor]] = None
        self.theta_star: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    def compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        n_batches: int = 20,
    ) -> None:
        """
        Estimate diagonal Fisher information matrix over *trainable* params
        using n_batches mini-batches from dataloader.

        F_i ≈ E[(∂ log p(y|x) / ∂θ_i)²]

        Accumulators and the final fisher/theta_star dicts are always stored
        on CPU.  This prevents a second full copy of LoRA weights from sitting
        in MPS memory, which was the main cause of OOM on M-series Macs.
        ewc_loss() already calls .to(param.device) before computing the
        penalty, so the on-CPU storage is transparent to the rest of training.
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}

        # Initialise accumulators on CPU to avoid pinning extra tensors on MPS
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros(
                    param.data.shape, dtype=param.data.dtype, device="cpu"
                )

        n_processed = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch.get("labels", input_ids).to(device)

            model.zero_grad()
            try:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    continue
                loss.backward()
            except Exception:
                continue

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Move grad to CPU before squaring so MPS peak stays low
                    fisher[name] += param.grad.detach().cpu() ** 2

            n_processed += 1

        # Normalise (all on CPU)
        if n_processed > 0:
            for name in fisher:
                fisher[name] /= n_processed

        # Save reference parameters (θ*) on CPU
        self.fisher    = fisher
        self.theta_star = {
            name: param.data.detach().clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        model.train()

    # ------------------------------------------------------------------
    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Return L_EWC = λ · Σ_i F_i (θ_i − θ*_i)²"""
        if self.fisher is None or self.theta_star is None:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                f     = self.fisher[name].to(param.device)
                theta = self.theta_star[name].to(param.device)
                loss  = loss + (f * (param - theta) ** 2).sum()
        return self.lambda_ewc * loss

    def is_ready(self) -> bool:
        return self.fisher is not None

    def state_dict(self) -> Dict:
        return {
            "lambda_ewc": self.lambda_ewc,
            "fisher"    : {k: v.cpu() for k, v in self.fisher.items()}
                          if self.fisher else None,
            "theta_star": {k: v.cpu() for k, v in self.theta_star.items()}
                          if self.theta_star else None,
        }

    def load_state_dict(self, d: Dict) -> None:
        self.lambda_ewc = d["lambda_ewc"]
        self.fisher     = d["fisher"]
        self.theta_star = d["theta_star"]


# ---------------------------------------------------------------------------
# Knowledge Distiller
# ---------------------------------------------------------------------------

class KnowledgeDistiller:
    """
    Knowledge Distillation from a frozen teacher snapshot.
    Teacher is a copy of the model state before continual adaptation.
    """

    def __init__(self, temperature: float = 2.0, gamma_kd: float = 0.3):
        self.temperature = temperature
        self.gamma_kd    = gamma_kd
        self.teacher: Optional[nn.Module] = None

    def snapshot(self, model: nn.Module) -> None:
        """Freeze a CPU copy of the current model as the teacher.

        Keeping the teacher on CPU avoids doubling MPS memory (~14 GB for
        LLaMA-2-7B BF16).  distillation_loss() already routes inputs to
        whichever device the teacher lives on, so no other changes are needed.
        """
        self.teacher = copy.deepcopy(model).cpu()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_KD = KL(softmax(z_teacher/T) ∥ log_softmax(z_student/T))
        Computed only over non-padding tokens.
        """
        if self.teacher is None:
            return torch.tensor(0.0, device=student_logits.device)

        device = student_logits.device
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids.to(next(self.teacher.parameters()).device)
            )
            teacher_logits = teacher_out.logits.to(device)

        T = self.temperature
        # Shift for next-token prediction
        s_shift = student_logits[:, :-1]
        t_shift = teacher_logits[:, :-1]

        s_log_probs = F.log_softmax(s_shift / T, dim=-1)
        t_probs     = F.softmax(t_shift / T, dim=-1)

        # Mask padding (label == -100)
        mask = (labels[:, 1:] != -100).float().to(device)
        kl   = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
        kl   = (kl * mask).sum() / mask.sum().clamp(min=1)

        return self.gamma_kd * (T ** 2) * kl

    def is_ready(self) -> bool:
        return self.teacher is not None


# ---------------------------------------------------------------------------
# Continual Trainer
# ---------------------------------------------------------------------------

class ContinualTrainer:
    """
    Orchestrates one round of continual training on a new data chunk.

    Loss: L = L_CE + α·L_contrastive + β·L_EWC + γ·L_KD
    """

    def __init__(
        self,
        ewc_regularizer: EWCRegularizer,
        distiller: KnowledgeDistiller,
        replay_buffer: PriorityReplayBuffer,
        alpha_contrastive: float = 0.2,
        replay_ratio: float = 0.3,
        fisher_update_freq: int = 5,   # update Fisher every N chunks
        fisher_n_batches: int = 20,    # mini-batches per Fisher estimate
        device: Optional[torch.device] = None,
    ):
        self.ewc              = ewc_regularizer
        self.distiller        = distiller
        self.buffer           = replay_buffer
        self.alpha_cont       = alpha_contrastive
        self.replay_ratio     = replay_ratio
        self.fisher_freq      = fisher_update_freq
        self.fisher_n_batches = fisher_n_batches
        self.device           = device or get_device()
        self._chunk_idx       = 0

    # ------------------------------------------------------------------
    def train_on_chunk(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        new_samples: List[Dict],
        tokenizer,
        n_epochs: int = 3,
        batch_size: int = 4,
        contrastive_fn=None,
    ) -> Dict[str, float]:
        """
        Train model on new_samples mixed with replay.

        Parameters
        ----------
        model          : the LLM with LoRA adapters
        optimizer      : AdamW (MPS-compatible)
        new_samples    : list of dicts with 'input', 'output', 'entities'
        tokenizer      : HuggingFace tokenizer
        n_epochs       : gradient steps per chunk
        batch_size     : per-device batch size
        contrastive_fn : optional function returning L_contrastive from model output

        Returns
        -------
        dict of average losses {'ce', 'ewc', 'kd', 'total'}
        """
        model.train()
        model.to(self.device)

        # Mix new data with replay
        n_replay = int(len(new_samples) * self.replay_ratio / (1 - self.replay_ratio + 1e-9))
        replay_samples = self.buffer.sample(n_replay) if self.buffer.is_ready() else []
        all_samples = new_samples + replay_samples

        if not all_samples:
            return {"ce": 0., "ewc": 0., "kd": 0., "total": 0.}

        dataset = _SimpleQADataset(all_samples, tokenizer,
                                   max_len=512, device=self.device)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=_collate_fn)

        total_loss = total_ce = total_ewc = total_kd = 0.
        n_steps = 0

        for epoch in range(n_epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)

                optimizer.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                l_ce = outputs.loss

                # EWC
                l_ewc = self.ewc.ewc_loss(model) if self.ewc.is_ready() else \
                        torch.tensor(0., device=self.device)

                # Knowledge Distillation
                l_kd = self.distiller.distillation_loss(
                    outputs.logits, input_ids, labels
                ) if self.distiller.is_ready() else \
                    torch.tensor(0., device=self.device)

                # Contrastive (optional, from RECIPE-TKG)
                l_cont = torch.tensor(0., device=self.device)
                if contrastive_fn is not None:
                    try:
                        l_cont = contrastive_fn(model, batch)
                    except Exception:
                        pass

                loss = l_ce + self.alpha_cont * l_cont + l_ewc + l_kd

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()

                total_ce   += l_ce.item()
                total_ewc  += l_ewc.item()
                total_kd   += l_kd.item()
                total_loss += loss.item()
                n_steps    += 1

        # Add new data to replay buffer
        self.buffer.add_batch(new_samples)

        # Update Fisher periodically
        self._chunk_idx += 1
        if self._chunk_idx % self.fisher_freq == 0 and len(loader) > 0:
            self.ewc.compute_fisher(model, loader, self.device,
                                    n_batches=self.fisher_n_batches)
            self.distiller.snapshot(model)   # refresh teacher (stored on CPU)

        empty_cache(self.device)

        denom = max(n_steps, 1)
        return {
            "ce"   : total_ce   / denom,
            "ewc"  : total_ewc  / denom,
            "kd"   : total_kd   / denom,
            "total": total_loss / denom,
        }

    # ------------------------------------------------------------------
    def initialise(self, model: nn.Module, seed_dataloader: DataLoader) -> None:
        """
        Call once after initial (pre-stream) training to capture the Fisher
        matrix and take the first teacher snapshot.
        """
        self.ewc.compute_fisher(model, seed_dataloader, self.device,
                                n_batches=self.fisher_n_batches)
        self.distiller.snapshot(model)   # stored on CPU


# ---------------------------------------------------------------------------
# Simple dataset / collate helpers
# ---------------------------------------------------------------------------

class _SimpleQADataset(Dataset):
    """Tokenise prompt+answer pairs for causal LM training."""

    def __init__(self, samples, tokenizer, max_len=512, device=None):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.device    = device
        self.items     = []
        for s in samples:
            prompt = s.get("input", "")
            answer = s.get("output", "")
            full   = prompt + answer
            self.items.append(full)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.items[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids"     : enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def _collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attn_mask = []
    labels    = []
    for x in batch:
        ids = x["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attn_mask.append(x["attention_mask"] + [0] * pad)
        lab = ids + [-100] * pad       # -100 = ignore padding in CE loss
        labels.append(lab)
    return {
        "input_ids"      : torch.tensor(input_ids,  dtype=torch.long),
        "attention_mask" : torch.tensor(attn_mask,  dtype=torch.long),
        "labels"         : torch.tensor(labels,     dtype=torch.long),
    }
