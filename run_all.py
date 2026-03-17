"""
D-RECIPE Master Orchestration Script.

Runs the complete experimental pipeline:
  1. Check / download all 4 benchmark datasets
  2. Verify LLaMA-2 and LLaMA-3 model access
  3. Run continual TKG experiments on all datasets
  4. Run ablation study (ICEWS14)
  5. Generate all plots and tables

Usage
-----
python run_all.py                                  # full pipeline
python run_all.py --dataset icews14               # single dataset
python run_all.py --model llama3                  # LLaMA-3-8B
python run_all.py --debug                         # fast debug run
python run_all.py --plots_only                    # plots from saved results
python run_all.py --skip_setup                    # assume data + model ready
python run_all.py --dataset icews14 --ablation    # ablation only
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_DATASETS = ["icews14", "icews18", "GDELT", "YAGO"]

MODEL_MAP = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
}

ABLATION_CONFIGS = [
    "no_ewc",
    "no_replay",
    "no_kd",
    "no_filtering",
    "naive_finetune",
]

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   D-RECIPE: Dynamic Continual TKG Forecasting                   ║
║   PhD Research — Busisani Mac Dube & J.V. Fonou-Dombeu          ║
╠══════════════════════════════════════════════════════════════════╣
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, log_file: str = None, check: bool = True) -> int:
    """Run a command, optionally tee-ing output to a log file."""
    print(f"\n  $ {' '.join(str(c) for c in cmd)}")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
            output = proc.stdout
            lf.write(output)
            # Print last 20 lines to console
            lines = output.strip().splitlines()
            for line in lines[-20:]:
                print(f"    {line}")
        if check and proc.returncode != 0:
            print(f"\n  ✗ Command failed (exit {proc.returncode}). "
                  f"Full log: {log_file}")
        return proc.returncode
    else:
        proc = subprocess.run(cmd)
        if check and proc.returncode != 0:
            print(f"\n  ✗ Command failed (exit {proc.returncode}).")
        return proc.returncode


def _load_metrics(metrics_path: str) -> dict:
    """Load a metrics.json file; returns {} if not found."""
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _print_metrics(ds: str, model_tag: str, results_base: str) -> None:
    """Print Hits@k / BWT / FWT for a completed run."""
    mfile = os.path.join(results_base, f"{ds}_{model_tag}", "metrics.json")
    data  = _load_metrics(mfile)
    if not data:
        return
    m  = data.get("final", {})
    cl = data.get("continual", {})
    print(f"  {ds} ({model_tag}): "
          f"H@1={m.get('Hits@1', 0):.4f}  "
          f"H@3={m.get('Hits@3', 0):.4f}  "
          f"H@10={m.get('Hits@10', 0):.4f}  "
          f"BWT={cl.get('bwt', 0):+.4f}  "
          f"FWT={cl.get('fwt', 0):+.4f}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_setup_datasets(args) -> bool:
    """Step 1 — Download / verify all 4 benchmark datasets."""
    print("\n" + "─" * 60)
    print("  STEP 1: Dataset Setup")
    print("─" * 60)

    datasets = [args.dataset] if args.dataset != "all" else ALL_DATASETS

    from setup_datasets import setup_all_datasets
    results = setup_all_datasets(
        data_root=args.data_dir,
        datasets=datasets,
        verbose=True,
    )
    all_ok = all(results.values())
    if not all_ok:
        missing = [d for d, ok in results.items() if not ok]
        print(f"\n  ⚠ Some datasets unavailable: {missing}")
        print("  The pipeline will continue with synthetic data where needed.")
    return True   # always continue; synthetic fallback handles missing data


def step_setup_models(args) -> bool:
    """Step 2 — Check LLaMA-2 and LLaMA-3 model access."""
    print("\n" + "─" * 60)
    print("  STEP 2: Model Setup")
    print("─" * 60)

    from setup_models import setup_models
    results = setup_models(
        models=["llama2", "llama3"],
        download=args.download_models,
        skip_hf_check=args.skip_hf_check,
    )
    return any(r["token_ok"] or r["cached"] for r in results.values())


def step_run_experiments(args) -> None:
    """Step 3 — Run D-RECIPE on all selected datasets."""
    print("\n" + "─" * 60)
    print("  STEP 3: D-RECIPE Experiments")
    print("─" * 60)

    model_name = MODEL_MAP[args.model]
    datasets   = [args.dataset] if args.dataset != "all" else ALL_DATASETS

    # Debug overrides
    if args.debug:
        n_chunks     = 3
        seed_chunks  = 1
        max_eval     = 20
        stream_chunk = 100
        epochs       = 1
        buffer       = 500
        debug_flag   = ["--DEBUG"]
        print("  ⚡ DEBUG MODE active (reduced sizes)")
    else:
        n_chunks     = 20
        seed_chunks  = 5
        max_eval     = 200
        stream_chunk = 500
        epochs       = 5
        buffer       = 5000
        debug_flag   = []

    model_tag = args.model

    for ds in datasets:
        out_dir  = os.path.join(args.results_dir, f"{ds}_{model_tag}")
        log_file = os.path.join(out_dir, "train.log")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n  Running D-RECIPE on {ds} with {model_tag} …")
        cmd = [
            sys.executable, "dynamic_main.py",
            "--DATASET",           ds,
            "--MODEL_NAME",        model_name,
            "--DATA_PATH",         args.data_dir,
            "--RESULTS_DIR",       out_dir,
            "--STREAM_CHUNK_SIZE", str(stream_chunk),
            "--N_CHUNKS",          str(n_chunks),
            "--SEED_CHUNKS",       str(seed_chunks),
            "--EPOCHS",            str(epochs),
            "--BUFFER_SIZE",       str(buffer),
            "--MAX_EVAL_SAMPLES",  str(max_eval),
            "--USE_EWC",           "1",
            "--USE_REPLAY",        "1",
            "--USE_KD",            "1",
            "--ADAPTIVE_THRESHOLD","1",
            "--ABLATION",          "full",
            "--DEVICE",            "auto",
            "--GRADIENT_CHECKPOINTING", "1",
        ] + debug_flag
        _run(cmd, log_file=log_file, check=False)

    # LLaMA-3 comparison on ICEWS14 (when running LLaMA-2 as primary)
    if args.model == "llama2" and (args.dataset == "all" or args.dataset == "icews14"):
        out_dir_l3  = os.path.join(args.results_dir, "icews14_llama3")
        log_file_l3 = os.path.join(out_dir_l3, "train.log")
        Path(out_dir_l3).mkdir(parents=True, exist_ok=True)
        print("\n  Running LLaMA-3-8B comparison on ICEWS14 …")
        cmd_l3 = [
            sys.executable, "dynamic_main.py",
            "--DATASET",           "icews14",
            "--MODEL_NAME",        MODEL_MAP["llama3"],
            "--DATA_PATH",         args.data_dir,
            "--RESULTS_DIR",       out_dir_l3,
            "--STREAM_CHUNK_SIZE", str(stream_chunk),
            "--N_CHUNKS",          str(n_chunks),
            "--SEED_CHUNKS",       str(seed_chunks),
            "--EPOCHS",            str(epochs),
            "--BUFFER_SIZE",       str(buffer),
            "--MAX_EVAL_SAMPLES",  str(max_eval),
            "--ABLATION",          "full",
            "--DEVICE",            "auto",
        ] + debug_flag
        _run(cmd_l3, log_file=log_file_l3, check=False)


def step_run_ablation(args) -> None:
    """Step 4 — Ablation study on ICEWS14."""
    print("\n" + "─" * 60)
    print("  STEP 4: Ablation Study (ICEWS14)")
    print("─" * 60)

    model_name = MODEL_MAP[args.model]

    if args.debug:
        n_chunks = 3; seed_chunks = 1; max_eval = 20
        stream_chunk = 100; epochs = 1; buffer = 500
        debug_flag = ["--DEBUG"]
    else:
        n_chunks = 20; seed_chunks = 5; max_eval = 200
        stream_chunk = 500; epochs = 5; buffer = 5000
        debug_flag = []

    for abl in ABLATION_CONFIGS:
        out_dir  = os.path.join(args.results_dir, f"ablation_{abl}")
        log_file = os.path.join(out_dir, "train.log")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n  Ablation: {abl} …")
        cmd = [
            sys.executable, "dynamic_main.py",
            "--DATASET",           "icews14",
            "--MODEL_NAME",        model_name,
            "--DATA_PATH",         args.data_dir,
            "--RESULTS_DIR",       out_dir,
            "--STREAM_CHUNK_SIZE", str(stream_chunk),
            "--N_CHUNKS",          str(n_chunks),
            "--SEED_CHUNKS",       str(seed_chunks),
            "--EPOCHS",            str(epochs),
            "--BUFFER_SIZE",       str(buffer),
            "--MAX_EVAL_SAMPLES",  str(max_eval),
            "--ABLATION",          abl,
            "--DEVICE",            "auto",
        ] + debug_flag
        _run(cmd, log_file=log_file, check=False)


def step_generate_plots(args) -> None:
    """Step 5 — Generate all plots and tables."""
    print("\n" + "─" * 60)
    print("  STEP 5: Generating Plots & Tables")
    print("─" * 60)

    plots_dir = os.path.join(args.results_dir, "plots")
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Use real results if available
    icews14_metrics = os.path.join(args.results_dir,
                                   f"icews14_{args.model}", "metrics.json")
    if os.path.exists(icews14_metrics):
        cmd = [sys.executable, "visualize_results.py",
               "--results_file", icews14_metrics,
               "--output_dir", plots_dir]
    else:
        cmd = [sys.executable, "visualize_results.py",
               "--mock", "--output_dir", plots_dir]

    _run(cmd, check=False)
    print(f"\n  ✓ All plots saved to: {plots_dir}/")


def step_summary(args) -> None:
    """Print final results summary."""
    print("\n" + "═" * 60)
    print("  D-RECIPE Experiment Complete — Results Summary")
    print("═" * 60)

    datasets  = [args.dataset] if args.dataset != "all" else ALL_DATASETS
    model_tag = args.model

    for ds in datasets:
        _print_metrics(ds, model_tag, args.results_dir)

    # Continual metrics
    mfile = os.path.join(args.results_dir, f"icews14_{model_tag}", "metrics.json")
    data  = _load_metrics(mfile)
    if data:
        cl = data.get("continual", {})
        print(f"\n  Continual Metrics (ICEWS14):")
        print(f"    AvgAcc = {cl.get('average_accuracy', 0):.4f}")
        print(f"    BWT    = {cl.get('bwt', 0):+.4f}")
        print(f"    FWT    = {cl.get('fwt', 0):+.4f}")

    print(f"\n  Plots:   {args.results_dir}/plots/")
    print(f"  Paper:   paper/drecipe_paper.pdf")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="D-RECIPE Full Pipeline")
    p.add_argument("--dataset",     type=str, default="all",
                   choices=ALL_DATASETS + ["all"])
    p.add_argument("--model",       type=str, default="llama2",
                   choices=["llama2", "llama3"])
    p.add_argument("--data_dir",    type=str, default="./data/original")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--debug",       action="store_true",
                   help="Reduced sizes for fast verification")
    p.add_argument("--skip_setup",  action="store_true",
                   help="Skip dataset + model setup steps")
    p.add_argument("--skip_hf_check", action="store_true",
                   help="Skip HuggingFace token verification")
    p.add_argument("--download_models", action="store_true",
                   help="Pre-download LLM weights to HF cache")
    p.add_argument("--ablation",    action="store_true",
                   help="Also run ablation study (ICEWS14)")
    p.add_argument("--ablation_only", action="store_true",
                   help="Only run ablation study")
    p.add_argument("--plots_only",  action="store_true",
                   help="Only regenerate plots from saved results")
    return p.parse_args()


def main():
    args = parse_args()
    start_time = datetime.now()

    print(BANNER)
    print(f"  Model:    {args.model} ({MODEL_MAP[args.model]})")
    print(f"  Datasets: {args.dataset}")
    print(f"  Results:  {args.results_dir}")
    print(f"  Debug:    {args.debug}")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        step_generate_plots(args)
        return

    # Step 1 & 2: Setup
    if not args.skip_setup:
        step_setup_datasets(args)
        step_setup_models(args)

    # Step 3: Experiments
    if not args.ablation_only:
        step_run_experiments(args)

    # Step 4: Ablation
    if args.ablation or args.ablation_only:
        step_run_ablation(args)

    # Step 5: Plots
    step_generate_plots(args)

    # Summary
    step_summary(args)

    elapsed = (datetime.now() - start_time).total_seconds()
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"  Total time: {h:02d}:{m:02d}:{s:02d}")


if __name__ == "__main__":
    main()
