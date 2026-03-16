#!/usr/bin/env bash
# =============================================================================
# D-RECIPE: Dynamic RECIPE — End-to-End Experiment Script
#
# Runs continual TKG forecasting experiments on all 4 benchmark datasets
# and generates all comparison plots.
#
# Hardware target: Apple M4 Max (MPS) or CUDA GPU
#
# Usage:
#   chmod +x run_dynamic.sh
#   ./run_dynamic.sh                          # full experiment (all datasets)
#   ./run_dynamic.sh --dataset icews14        # single dataset
#   ./run_dynamic.sh --debug                  # fast debug run
#   ./run_dynamic.sh --ablation               # ablation study only
#   ./run_dynamic.sh --plots_only             # regenerate plots from saved results
#   ./run_dynamic.sh --model llama3           # use LLaMA-3-8B
# =============================================================================

set -e   # exit on error

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATASET="all"
MODEL="llama2"
DEBUG=0
ABLATION_ONLY=0
PLOTS_ONLY=0
RESULTS_BASE="./results"
DATA_PATH="./data/original/"

# ─── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)    DATASET="$2";      shift 2 ;;
    --model)      MODEL="$2";        shift 2 ;;
    --debug)      DEBUG=1;           shift   ;;
    --ablation)   ABLATION_ONLY=1;   shift   ;;
    --plots_only) PLOTS_ONLY=1;      shift   ;;
    --results)    RESULTS_BASE="$2"; shift 2 ;;
    --data_path)  DATA_PATH="$2";    shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ─── Model selection ──────────────────────────────────────────────────────────
if [ "$MODEL" = "llama2" ]; then
  MODEL_NAME="meta-llama/Llama-2-7b-hf"
  MODEL_TAG="llama2"
else
  MODEL_NAME="meta-llama/Meta-Llama-3-8B"
  MODEL_TAG="llama3"
fi

# ─── Dataset list ─────────────────────────────────────────────────────────────
if [ "$DATASET" = "all" ]; then
  DATASETS=("icews14" "icews18" "GDELT" "YAGO")
else
  DATASETS=("$DATASET")
fi

# ─── Debug overrides ──────────────────────────────────────────────────────────
if [ $DEBUG -eq 1 ]; then
  N_CHUNKS=3
  SEED_CHUNKS=1
  MAX_EVAL=20
  STREAM_CHUNK=100
  EPOCHS=1
  BUFFER=500
  DEBUG_FLAG="--DEBUG"
  echo "⚡ DEBUG MODE: small run for quick testing"
else
  N_CHUNKS=20
  SEED_CHUNKS=5
  MAX_EVAL=200
  STREAM_CHUNK=500
  EPOCHS=5
  BUFFER=5000
  DEBUG_FLAG=""
fi

# ─── Print banner ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   D-RECIPE: Dynamic Continual TKG Forecasting           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Model    : $MODEL_NAME"
echo "║  Datasets : ${DATASETS[*]}"
echo "║  Results  : $RESULTS_BASE"
echo "║  Debug    : $DEBUG"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$RESULTS_BASE/plots"

# ─── Plots only mode ──────────────────────────────────────────────────────────
if [ $PLOTS_ONLY -eq 1 ]; then
  echo "📊 Generating plots from existing results ..."
  python visualize_results.py \
    --output_dir "$RESULTS_BASE/plots" \
    --mock
  echo "✓ Plots saved to $RESULTS_BASE/plots/"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Full D-RECIPE experiments
# ─────────────────────────────────────────────────────────────────────────────
if [ $ABLATION_ONLY -eq 0 ]; then
  for DS in "${DATASETS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running D-RECIPE on $DS with $MODEL_TAG ..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    OUT_DIR="$RESULTS_BASE/${DS}_${MODEL_TAG}"
    mkdir -p "$OUT_DIR"

    python dynamic_main.py \
      --DATASET "$DS" \
      --MODEL_NAME "$MODEL_NAME" \
      --DATA_PATH "$DATA_PATH" \
      --RESULTS_DIR "$OUT_DIR" \
      --STREAM_CHUNK_SIZE $STREAM_CHUNK \
      --N_CHUNKS $N_CHUNKS \
      --SEED_CHUNKS $SEED_CHUNKS \
      --EPOCHS $EPOCHS \
      --BUFFER_SIZE $BUFFER \
      --MAX_EVAL_SAMPLES $MAX_EVAL \
      --USE_EWC 1 \
      --USE_REPLAY 1 \
      --USE_KD 1 \
      --ADAPTIVE_THRESHOLD 1 \
      --ABLATION full \
      --DEVICE auto \
      --GRADIENT_CHECKPOINTING 1 \
      $DEBUG_FLAG \
      2>&1 | tee "$OUT_DIR/train.log"

    echo "✓ $DS done → $OUT_DIR"
  done

  # LLaMA-2 vs LLaMA-3 comparison on ICEWS14 (if not already done with llama3)
  if [ "$MODEL_TAG" = "llama2" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  LLaMA-3-8B comparison on ICEWS14 ..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    OUT_DIR_L3="$RESULTS_BASE/icews14_llama3"
    mkdir -p "$OUT_DIR_L3"

    python dynamic_main.py \
      --DATASET icews14 \
      --MODEL_NAME "meta-llama/Meta-Llama-3-8B" \
      --DATA_PATH "$DATA_PATH" \
      --RESULTS_DIR "$OUT_DIR_L3" \
      --STREAM_CHUNK_SIZE $STREAM_CHUNK \
      --N_CHUNKS $N_CHUNKS \
      --SEED_CHUNKS $SEED_CHUNKS \
      --EPOCHS $EPOCHS \
      --BUFFER_SIZE $BUFFER \
      --MAX_EVAL_SAMPLES $MAX_EVAL \
      --ABLATION full \
      --DEVICE auto \
      $DEBUG_FLAG \
      2>&1 | tee "$OUT_DIR_L3/train.log"

    echo "✓ LLaMA-3 ICEWS14 done → $OUT_DIR_L3"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Ablation study (ICEWS14 only)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running Ablation Study on ICEWS14 ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ABLATION_CONFIGS=("no_ewc" "no_replay" "no_kd" "no_filtering" "naive_finetune")

for ABL in "${ABLATION_CONFIGS[@]}"; do
  echo "  Ablation: $ABL ..."
  ABL_DIR="$RESULTS_BASE/ablation_${ABL}"
  mkdir -p "$ABL_DIR"

  python dynamic_main.py \
    --DATASET icews14 \
    --MODEL_NAME "$MODEL_NAME" \
    --DATA_PATH "$DATA_PATH" \
    --RESULTS_DIR "$ABL_DIR" \
    --STREAM_CHUNK_SIZE $STREAM_CHUNK \
    --N_CHUNKS $N_CHUNKS \
    --SEED_CHUNKS $SEED_CHUNKS \
    --EPOCHS $EPOCHS \
    --BUFFER_SIZE $BUFFER \
    --MAX_EVAL_SAMPLES $MAX_EVAL \
    --ABLATION "$ABL" \
    --DEVICE auto \
    $DEBUG_FLAG \
    2>&1 | tee "$ABL_DIR/train.log"

  echo "  ✓ Ablation $ABL done → $ABL_DIR"
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate all plots
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Generating plots ..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use ICEWS14 results file if available
ICEWS14_RESULTS="$RESULTS_BASE/icews14_${MODEL_TAG}/metrics.json"
if [ -f "$ICEWS14_RESULTS" ]; then
  python visualize_results.py \
    --results_file "$ICEWS14_RESULTS" \
    --output_dir "$RESULTS_BASE/plots"
else
  # Fall back to mock data
  python visualize_results.py \
    --mock \
    --output_dir "$RESULTS_BASE/plots"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   D-RECIPE Experiment Complete!                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Results:  $RESULTS_BASE"
echo "║  Plots:    $RESULTS_BASE/plots/"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Print final metrics summary
for DS in "${DATASETS[@]}"; do
  MFILE="$RESULTS_BASE/${DS}_${MODEL_TAG}/metrics.json"
  if [ -f "$MFILE" ]; then
    echo "── $DS ($MODEL_TAG) ──"
    python3 -c "
import json
with open('$MFILE') as f: d = json.load(f)
m = d.get('final', {})
print(f\"  Hits@1  = {m.get('Hits@1',  0):.4f}\")
print(f\"  Hits@3  = {m.get('Hits@3',  0):.4f}\")
print(f\"  Hits@10 = {m.get('Hits@10', 0):.4f}\")
cl = d.get('continual', {})
print(f\"  BWT     = {cl.get('bwt', 0):.4f}\")
print(f\"  FWT     = {cl.get('fwt', 0):.4f}\")
print(f\"  AvgAcc  = {cl.get('average_accuracy', 0):.4f}\")
"
  fi
done
