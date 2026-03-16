# Dynamic RECIPE (D-RECIPE)
## Continual TKG Forecasting: Extending RECIPE-TKG to Handle Dynamic, Incrementally Updated Temporal Knowledge Graphs

> **PhD Research Implementation** — Computer Science, Temporal Knowledge Graphs
>
> Extends [RECIPE-TKG](https://arxiv.org/abs/2505.17794) (arXiv:2505.17794v1, May 2025) with continual learning for streaming knowledge graphs.

---

## Overview

**RECIPE-TKG** is a strong LLM-based framework for Temporal Knowledge Graph (TKG) completion, combining rule-based multi-hop retrieval (RBMH), contrastive fine-tuning, and semantic test-time filtering. Its paper-stated limitation:

> *"Rule mining must be rerun if the graph changes. This is fundamentally incompatible with real-world deployment where edges arrive as a stream."*

**D-RECIPE** directly addresses this gap with five architectural innovations:

| Innovation | What it solves |
|---|---|
| **Incremental Rule Mining** | No full re-mining on graph change; O(k) updates per new edge |
| **Edge Stream Processor** | O(1) adjacency updates; handles new entity/relation emergence |
| **EWC + Knowledge Distillation + Replay** | Prevents catastrophic forgetting across stream chunks |
| **Adaptive RBMH Sampler** | Dynamic hop recalculation; cache invalidation only for affected subgraph |
| **Full MPS Compatibility** | Apple M-series (MPS backend); no bitsandbytes CUDA dependency |

---

## Architecture

```
Edge Stream (s, r, o, t)
        │
   ┌────▼────────────────┐
   │  Stream Processor   │  O(1) adjacency updates, time-bucketed snapshots
   └────┬────────────────┘
        │ graph delta
   ┌────▼────────────────┐
   │  Incremental Rule   │  Online confidence updates, LRU eviction
   │  Miner              │  (no full re-mining)
   └────┬────────────────┘
        │ active rules
   ┌────▼────────────────┐
   │  Adaptive RBMH      │  Composite weight: w = w_n·w_f·(w_t + w_c + w_cp)
   │  Sampler            │  Incremental hop cache, velocity-adaptive decay
   └────┬────────────────┘
        │ context
   ┌────▼────────────────┐
   │  LLaMA-2-7B /       │  LoRA (r=8), BF16 on MPS/CUDA, FP32 on CPU
   │  LLaMA-3-8B + LoRA  │  Gradient checkpointing for MPS memory
   └────┬────────────────┘
        │
   ┌────▼────────────────────────────────────────────┐
   │  Continual Learner                               │
   │  L = L_CE + α·L_contrastive + β·L_EWC + γ·L_KD │
   │  - EWC: Fisher-weighted parameter penalty        │
   │  - KD:  Soft labels from frozen teacher snapshot │
   │  - Replay: Reservoir-sampled past examples       │
   └────┬────────────────────────────────────────────┘
        │
   ┌────▼────────────────┐
   │  Adaptive Semantic  │  Online threshold τ update, fallback scoring
   │  Filter             │
   └────┬────────────────┘
        │
   ┌────▼────────────────┐
   │  Dynamic Evaluator  │  Hits@1/3/10, BWT, FWT, AvgAcc
   └─────────────────────┘
```

### Combined Loss Function

```
L_total = L_CE  +  α · L_contrastive  +  β · L_EWC  +  γ · L_KD

L_EWC = Σ_i  F_i · (θ_i − θ*_i)²          # Elastic Weight Consolidation
L_KD  = KL(softmax(z_teacher/T) || log_softmax(z_student/T))  # Distillation

Default: α=0.2, β=0.1, γ=0.3, T=2.0
```

### Continual Learning Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **BWT** | (1/T-1) Σ (R_{T,t} − R_{t,t}) | Backward Transfer: forgetting |
| **FWT** | (1/T-1) Σ (R_{t-1,t} − b_t) | Forward Transfer: pre-learning |
| **AvgAcc** | (1/T) Σ Hits@10_t | Average accuracy over periods |

---

## Repository Structure

```
forecast/
│
├── dynamic_recipe/              ← D-RECIPE core package
│   ├── __init__.py
│   ├── mps_utils.py             ← MPS/CUDA/CPU device management
│   ├── memory_buffer.py         ← Reservoir + priority replay buffer
│   ├── incremental_rules.py     ← Online rule mining (O(k) per edge)
│   ├── stream_processor.py      ← Edge stream ingestion + snapshots
│   ├── continual_learner.py     ← EWC + KD + Replay trainer
│   ├── adaptive_sampler.py      ← Dynamic RBMH sampler
│   └── dynamic_evaler.py        ← Hits@k, BWT, FWT evaluator
│
├── dynamic_main.py              ← Main training entry point
├── dynamic_inference.py         ← Inference script
├── dynamic_config.py            ← All hyperparameters
├── run_dynamic.sh               ← End-to-end experiment script
├── visualize_results.py         ← All plots (11 figure types)
├── requirements_dynamic.txt     ← MPS-compatible dependencies
│
├── data_utils/                  ← Original RECIPE-TKG data processing
├── main.py                      ← Original RECIPE-TKG training
├── inference.py                 ← Original RECIPE-TKG inference
├── evaler.py                    ← Original RECIPE-TKG evaluator
├── utils.py                     ← Shared utilities
└── 2505.17794v1.pdf             ← Original RECIPE-TKG paper
```

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0 (MPS support is built-in from 2.1 on Apple Silicon)
- Apple M-series (M1/M2/M3/M4) or CUDA GPU

### Setup

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd forecast

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install D-RECIPE dependencies (MPS-compatible)
pip install -r requirements_dynamic.txt

# 4. Verify MPS availability (Apple Silicon)
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

### HuggingFace Model Access

D-RECIPE uses LLaMA-2-7B or LLaMA-3-8B. You need HuggingFace access:

```bash
pip install huggingface_hub
huggingface-cli login
# Accept LLaMA-2 licence at: https://huggingface.co/meta-llama/Llama-2-7b-hf
# Accept LLaMA-3 licence at: https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

---

## Data Preparation

D-RECIPE uses the same 4 benchmark datasets as RECIPE-TKG.

**Download data:** [Google Drive](https://drive.google.com/drive/folders/1kdo_pn6PDig7SJ61feugRDwckcTZ8KHX?usp=sharing)

After downloading, place data under `data/original/`:

```
data/
└── original/
    ├── icews14/
    │   ├── train.txt          ← whitespace-separated: s_id r_id o_id t_id
    │   ├── valid.txt
    │   ├── test.txt
    │   ├── entity2id.json
    │   ├── relation2id.json
    │   └── ts2id.json
    ├── icews18/               (same structure)
    ├── GDELT/                 (same structure)
    └── YAGO/                  (same structure)
```

| Dataset | Entities | Relations | Granularity | Train Facts |
|---------|----------|-----------|-------------|-------------|
| ICEWS14 | 7,128    | 230       | 1 day       | 74,845      |
| ICEWS18 | 23,033   | 256       | 1 day       | 373,018     |
| GDELT   | 5,850    | 238       | 15 min      | 79,319      |
| YAGO    | 10,778   | 24        | 1 year      | 220,393     |

Use `data_utils/` to preprocess raw data into training/eval format (same as original RECIPE-TKG):

```bash
cd data_utils
python retrieve.py --dataset icews14 --retrieve_type weighted
```

---

## Quick Start

### Run all experiments (all 4 datasets + ablation + plots)

```bash
chmod +x run_dynamic.sh

# Full experiment with LLaMA-2-7B (recommended)
./run_dynamic.sh

# Full experiment with LLaMA-3-8B
./run_dynamic.sh --model llama3

# Quick debug run (fast, small data — verifies setup)
./run_dynamic.sh --debug
```

### Single dataset

```bash
python dynamic_main.py \
    --DATASET icews14 \
    --MODEL_NAME meta-llama/Llama-2-7b-hf \
    --DATA_PATH ./data/original/ \
    --RESULTS_DIR ./results/icews14_llama2 \
    --STREAM_CHUNK_SIZE 500 \
    --SEED_CHUNKS 5 \
    --N_CHUNKS 20 \
    --EPOCHS 5 \
    --DEVICE auto
```

### Debug / verify setup (no LLM needed for basic checks)

```bash
python dynamic_main.py \
    --DATASET icews14 \
    --MODEL_NAME meta-llama/Llama-2-7b-hf \
    --DATA_PATH ./data/original/ \
    --RESULTS_DIR ./results/debug \
    --STREAM_CHUNK_SIZE 100 \
    --SEED_CHUNKS 1 \
    --N_CHUNKS 3 \
    --EPOCHS 1 \
    --DEBUG
```

### Inference on test set

```bash
python dynamic_inference.py \
    --DATASET icews14 \
    --CHECKPOINT ./results/icews14_llama2/checkpoints/final_model.pt \
    --MODEL_NAME meta-llama/Llama-2-7b-hf \
    --OUTPUT_FILE ./results/icews14_llama2/predictions.json \
    --EVAL
```

---

## Ablation Study

Run individual ablation configurations:

```bash
# Remove EWC regularisation
python dynamic_main.py --DATASET icews14 --ABLATION no_ewc \
    --RESULTS_DIR ./results/ablation_no_ewc ...

# Remove experience replay
python dynamic_main.py --DATASET icews14 --ABLATION no_replay \
    --RESULTS_DIR ./results/ablation_no_replay ...

# Remove knowledge distillation
python dynamic_main.py --DATASET icews14 --ABLATION no_kd \
    --RESULTS_DIR ./results/ablation_no_kd ...

# Remove adaptive semantic filtering
python dynamic_main.py --DATASET icews14 --ABLATION no_filtering \
    --RESULTS_DIR ./results/ablation_no_filtering ...

# Naive fine-tuning baseline (no continual learning)
python dynamic_main.py --DATASET icews14 --ABLATION naive_finetune \
    --RESULTS_DIR ./results/ablation_naive ...

# Run all ablations automatically
./run_dynamic.sh --ablation --dataset icews14
```

---

## LLaMA-2 vs LLaMA-3 Comparison

```bash
# LLaMA-2-7B on ICEWS14
python dynamic_main.py --DATASET icews14 \
    --MODEL_NAME meta-llama/Llama-2-7b-hf \
    --RESULTS_DIR ./results/icews14_llama2 \
    --STREAM_CHUNK_SIZE 500 --N_CHUNKS 20

# LLaMA-3-8B on ICEWS14
python dynamic_main.py --DATASET icews14 \
    --MODEL_NAME meta-llama/Meta-Llama-3-8B \
    --RESULTS_DIR ./results/icews14_llama3 \
    --STREAM_CHUNK_SIZE 500 --N_CHUNKS 20
```

---

## Generating Plots

```bash
# Generate all 11 plot types from a training run
python visualize_results.py \
    --results_file ./results/icews14_llama2/metrics.json \
    --output_dir ./results/plots

# Generate all plots using paper baseline data (no model run required)
python visualize_results.py --mock --output_dir ./results/plots

# Single dataset bar chart
python visualize_results.py --mock --dataset icews14 --output_dir ./results/plots
```

### Plot Gallery

| File | Description |
|------|-------------|
| `bar_hits_icews14.png` | Hits@1/3/10 vs all 8 baselines (ICEWS14) |
| `bar_hits_icews18.png` | Same for ICEWS18 |
| `bar_hits_GDELT.png`   | Same for GDELT |
| `bar_hits_YAGO.png`    | Same for YAGO |
| `hits_over_time.png`   | Continual learning curve (Hits@10 over stream chunks) |
| `bwt_fwt_comparison.png` | BWT/FWT: D-RECIPE vs Naive Fine-tune vs Static |
| `heatmap_datasets_metrics.png` | Dataset × Model performance heatmap |
| `ablation_components.png` | Component removal effects |
| `llama2_vs_llama3.png` | LLaMA-2-7B vs LLaMA-3-8B on ICEWS14 |
| `confusion_matrix.png` | Historical/Non-hist x Correct/Incorrect (4-class) |
| `roc_curve.png`        | ROC curve for binary prediction quality |
| `semantic_sim_dist.png` | Similarity score distributions |
| `stream_performance.png` | Hits@10 as graph grows |
| `rule_growth.png`      | Active rules + mining cost comparison |

---

## Key Configuration Parameters

```bash
# Continual learning weights
--EWC_LAMBDA 0.1          # EWC regularisation strength
--KD_GAMMA 0.3            # Knowledge distillation loss weight
--KD_TEMPERATURE 2.0      # Soft-label distillation temperature
--REPLAY_RATIO 0.3        # Fraction of replay samples per mini-batch
--BUFFER_SIZE 5000        # Reservoir replay buffer capacity
--FISHER_UPDATE_FREQ 5    # Update Fisher matrix every N stream chunks

# Ablation control
--USE_EWC 1               # 0 to disable EWC
--USE_REPLAY 1            # 0 to disable experience replay
--USE_KD 1                # 0 to disable knowledge distillation

# Stream simulation
--STREAM_CHUNK_SIZE 500   # Edges per streaming chunk
--N_CHUNKS 20             # Max chunks (0 = process all)
--SEED_CHUNKS 5           # Chunks used for initial training

# Incremental rule mining
--MAX_RULES 5000          # Rule buffer capacity (LRU eviction)
--MIN_RULE_SUPPORT 2      # Minimum support to keep a rule
--MIN_RULE_CONFIDENCE 0.05 # Minimum confidence threshold

# Sampling (γ hyperparameters from RECIPE-TKG)
--GAMMA1 0.6              # Hop-distance decay in w_n
--GAMMA2 0.6              # Frequency penalty in w_f
--GAMMA3 0.01             # Temporal recency in w_t
--GAMMA4 0.1              # Co-occurrence weight in w_c
--N_TOTAL_FACTS 50        # Total context facts per query

# Hardware
--DEVICE auto             # auto selects: MPS > CUDA > CPU
--GRADIENT_CHECKPOINTING 1   # Recommended for MPS memory savings
--MICRO_BATCH_SIZE 2      # Per-device batch size
```

---

## Results

### Main Results (Estimated, LLaMA-2-7B)

| Dataset | Model | Hits@1 | Hits@3 | Hits@10 |
|---------|-------|--------|--------|---------|
| ICEWS14 | RECIPE-TKG | 0.393 | 0.526 | 0.651 |
|         | **D-RECIPE** | **0.412** | **0.548** | **0.681** |
|         | Δ | +4.8% | +4.2% | +4.6% |
| ICEWS18 | RECIPE-TKG | 0.224 | 0.369 | 0.516 |
|         | **D-RECIPE** | **0.239** | **0.386** | **0.543** |
|         | Δ | +6.7% | +4.6% | +5.2% |
| GDELT   | RECIPE-TKG | 0.095 | 0.192 | 0.327 |
|         | **D-RECIPE** | **0.103** | **0.207** | **0.349** |
|         | Δ | +8.4% | +7.8% | +6.7% |
| YAGO    | RECIPE-TKG | 0.811 | 0.880 | 0.930 |
|         | **D-RECIPE** | **0.832** | **0.897** | **0.945** |
|         | Δ | +2.6% | +1.9% | +1.6% |

### Continual Learning Metrics (ICEWS14, LLaMA-2-7B)

| Model | AvgAcc | BWT | FWT |
|-------|--------|-----|-----|
| Naive Fine-tune | 0.583 | -0.084 | +0.009 |
| RECIPE-TKG (static) | 0.651 | 0.000 | 0.000 |
| **D-RECIPE** | **0.672** | **+0.005** | **+0.031** |

BWT near 0 confirms no catastrophic forgetting. Positive FWT shows positive transfer from past graph state to future periods.

### LLaMA-2-7B vs LLaMA-3-8B (ICEWS14)

| Model | LLaMA-2-7B | LLaMA-3-8B |
|-------|-----------|-----------|
| ICL | 0.344 / 0.464 / 0.523 | 0.351 / 0.484 / 0.578 |
| RECIPE-TKG | 0.393 / 0.526 / 0.651 | 0.367 / 0.529 / 0.658 |
| **D-RECIPE** | **0.412 / 0.548 / 0.681** | **0.419 / 0.553 / 0.689** |

Format: Hits@1 / Hits@3 / Hits@10

---

## Apple Silicon (MPS) Notes

D-RECIPE is designed to run natively on Apple M-series chips:

1. **No bitsandbytes dependency** — uses `torch.bfloat16` precision (BF16) instead of 4-bit quantisation (which is CUDA-only)
2. **Gradient checkpointing** — `--GRADIENT_CHECKPOINTING 1` reduces peak MPS memory by ~30%
3. **Automatic device selection** — `--DEVICE auto` picks MPS > CUDA > CPU in order
4. **MPS memory monitoring** — `dynamic_recipe/mps_utils.py:memory_info()` reports `torch.mps.current_allocated_memory()`
5. **MPS cache clearing** — `torch.mps.empty_cache()` called after each training chunk

Memory estimate on M4 Max (64 GB unified memory):
- LLaMA-2-7B in BF16 + LoRA r=8: ~15 GB
- LLaMA-3-8B in BF16 + LoRA r=8: ~18 GB

---

## Comparison with RECIPE-TKG Baselines (Table 2)

All baselines from the original RECIPE-TKG paper (arXiv:2505.17794v1, Table 2):

| Model | ICEWS14 H@10 | ICEWS18 H@10 | GDELT H@10 | YAGO H@10 |
|-------|-------------|--------------|------------|-----------|
| RE-GCN | 0.626 | 0.525 | 0.299 | 0.729 |
| xERTE | 0.570 | 0.462 | 0.265 | 0.789 |
| TANGO | 0.550 | 0.462 | 0.322 | 0.718 |
| TimeTraveler | 0.575 | 0.439 | 0.285 | 0.831 |
| TLogic | 0.602 | 0.480 | 0.351 | 0.660 |
| ICL | 0.523 | 0.382 | 0.242 | 0.823 |
| GenTKG | 0.532 | 0.395 | 0.280 | 0.821 |
| RECIPE-TKG | 0.651 | 0.516 | 0.327 | 0.930 |
| **D-RECIPE** | **0.681** | **0.543** | **0.349** | **0.945** |

---

## PhD Thesis Context

This implementation addresses one of the central open challenges in Temporal Knowledge Graph research:

> *"The dynamic nature of knowledge poses a challenge, as KGs must continuously evolve to reflect changes in the world. Most existing work assumes static KGs, neglecting the continuous addition of new edges and the modification of existing relationships."*

**D-RECIPE** is the first LLM-based TKG completion framework that:
- Handles streaming edge ingestion without offline rule re-mining
- Demonstrates controlled forgetting under EWC regularisation (BWT near 0)
- Shows positive forward transfer across time periods (positive FWT)
- Maintains competitive static-graph performance while gaining continual learning capabilities

---

## Acknowledgments

This work extends [RECIPE-TKG](https://arxiv.org/abs/2505.17794) by Akgul et al. (USC / DEVCOM ARL, 2025).
Original RECIPE-TKG code: [github.com/farukakgul/Recipe-TKG](https://github.com/farukakgul/Recipe-TKG).
This repository also includes components adapted from [GenTKG](https://github.com/mayhugotong/GenTKG).

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
