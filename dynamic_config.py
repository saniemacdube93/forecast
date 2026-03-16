"""
Configuration for Dynamic RECIPE (D-RECIPE).

Extends the base RECIPE-TKG config.py with continual learning,
streaming, and MPS-specific parameters.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="D-RECIPE: Dynamic Continual TKG Forecasting"
    )

    # ------------------------------------------------------------------ #
    # Inherited from RECIPE-TKG (unchanged defaults for reproducibility)
    # ------------------------------------------------------------------ #
    parser.add_argument("--MICRO_BATCH_SIZE", type=int, default=2)
    parser.add_argument("--BATCH_SIZE",       type=int, default=512)
    parser.add_argument("--EPOCHS",           type=int, default=10,
                        help="Epochs per stream chunk (reduced from 50 for speed)")
    parser.add_argument("--WARMUP_STEPS",     type=int, default=50)
    parser.add_argument("--LEARNING_RATE",    type=float, default=3e-4)
    parser.add_argument("--CONTEXT_LEN",      type=int, default=4096)
    parser.add_argument("--TARGET_LEN",       type=int, default=128)
    parser.add_argument("--TEXT_LEN",         type=int, default=256)
    parser.add_argument("--LORA_R",           type=int, default=8)
    parser.add_argument("--LORA_ALPHA",       type=int, default=16)
    parser.add_argument("--LORA_DROPOUT",     type=float, default=0.05)
    parser.add_argument("--MODEL_NAME",       type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        choices=["meta-llama/Llama-2-7b-hf",
                                 "meta-llama/Meta-Llama-3-8B"])
    parser.add_argument("--LOGGING_STEPS",    type=int,  default=1)
    parser.add_argument("--OUTPUT_DIR",       type=str,  default="./output_model")
    parser.add_argument("--DATASET",          type=str,  default="icews14",
                        choices=["icews14", "icews18", "GDELT", "YAGO"])
    parser.add_argument("--DATA_PATH",        type=str,  default="./data/original/")
    parser.add_argument("--CONTRASTIVE",      type=int,  default=1)
    parser.add_argument("--CONTRASTIVE_WEIGHT", type=float, default=0.2)

    # ------------------------------------------------------------------ #
    # Continual learning
    # ------------------------------------------------------------------ #
    parser.add_argument("--EWC_LAMBDA",      type=float, default=0.1,
                        help="EWC regularisation strength")
    parser.add_argument("--KD_GAMMA",        type=float, default=0.3,
                        help="Knowledge distillation loss weight")
    parser.add_argument("--KD_TEMPERATURE",  type=float, default=2.0,
                        help="Soft-label distillation temperature")
    parser.add_argument("--REPLAY_RATIO",    type=float, default=0.3,
                        help="Fraction of replay samples per batch (0=no replay)")
    parser.add_argument("--BUFFER_SIZE",     type=int,   default=5000,
                        help="Replay buffer capacity (reservoir sampling)")
    parser.add_argument("--FISHER_UPDATE_FREQ", type=int, default=5,
                        help="Update Fisher matrix every N stream chunks")
    parser.add_argument("--USE_EWC",         type=int,   default=1,
                        help="Enable EWC regularisation (0=off)")
    parser.add_argument("--USE_REPLAY",      type=int,   default=1,
                        help="Enable experience replay (0=off)")
    parser.add_argument("--USE_KD",          type=int,   default=1,
                        help="Enable knowledge distillation (0=off)")

    # ------------------------------------------------------------------ #
    # Stream / dynamic graph
    # ------------------------------------------------------------------ #
    parser.add_argument("--STREAM_CHUNK_SIZE", type=int, default=500,
                        help="Number of edges per streaming chunk")
    parser.add_argument("--N_CHUNKS",          type=int, default=0,
                        help="Max chunks to process (0=all)")
    parser.add_argument("--SEED_CHUNKS",       type=int, default=5,
                        help="Chunks used for initial (pre-stream) training")

    # ------------------------------------------------------------------ #
    # Incremental rule mining
    # ------------------------------------------------------------------ #
    parser.add_argument("--MAX_RULES",          type=int,   default=5000)
    parser.add_argument("--MIN_RULE_SUPPORT",   type=int,   default=2)
    parser.add_argument("--MIN_RULE_CONFIDENCE",type=float, default=0.05)
    parser.add_argument("--RULE_STATE_PATH",    type=str,   default="")

    # ------------------------------------------------------------------ #
    # Sampling hyperparameters (γ1..γ4 from RECIPE-TKG)
    # ------------------------------------------------------------------ #
    parser.add_argument("--GAMMA1", type=float, default=0.6,
                        help="Hop-distance decay")
    parser.add_argument("--GAMMA2", type=float, default=0.6,
                        help="Frequency penalty")
    parser.add_argument("--GAMMA3", type=float, default=0.01,
                        help="Temporal recency")
    parser.add_argument("--GAMMA4", type=float, default=0.1,
                        help="Co-occurrence weight")
    parser.add_argument("--N_TLR_FACTS",   type=int, default=20)
    parser.add_argument("--N_TOTAL_FACTS", type=int, default=50)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    parser.add_argument("--EVAL_EVERY_N_CHUNKS", type=int, default=2)
    parser.add_argument("--MAX_EVAL_SAMPLES",    type=int, default=200)
    parser.add_argument("--SIM_THRESHOLD",       type=float, default=0.6,
                        help="Initial semantic similarity threshold for filtering")
    parser.add_argument("--ADAPTIVE_THRESHOLD",  type=int, default=1,
                        help="Adapt similarity threshold online (0=fixed)")

    # ------------------------------------------------------------------ #
    # Hardware / MPS
    # ------------------------------------------------------------------ #
    parser.add_argument("--DEVICE",             type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Device selection (auto = best available)")
    parser.add_argument("--GRADIENT_CHECKPOINTING", type=int, default=1,
                        help="Enable gradient checkpointing (reduces MPS memory)")

    # ------------------------------------------------------------------ #
    # Experiment / output
    # ------------------------------------------------------------------ #
    parser.add_argument("--RESULTS_DIR",  type=str, default="./results",
                        help="Directory for saving metrics, checkpoints, plots")
    parser.add_argument("--SAVE_EVERY",   type=int, default=5,
                        help="Save checkpoint every N stream chunks")
    parser.add_argument("--DEBUG",        action="store_true",
                        help="Debug mode: smaller data, more logging")
    parser.add_argument("--REPORT_TO",    type=str, default=None,
                        help="Logging backend (wandb, tensorboard, none)")
    parser.add_argument("--RUN_NAME",     type=str, default="",
                        help="Experiment name for logging")

    # ------------------------------------------------------------------ #
    # Ablation switches
    # ------------------------------------------------------------------ #
    parser.add_argument("--ABLATION",     type=str, default="full",
                        choices=["full", "no_ewc", "no_replay", "no_kd",
                                 "no_incremental_rules", "no_filtering",
                                 "naive_finetune"],
                        help="Ablation configuration")

    return parser.parse_args()


def args_to_dict(args) -> dict:
    return vars(args)


def print_config(args) -> None:
    print("\n" + "=" * 60)
    print("  D-RECIPE Configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<30} {v}")
    print("=" * 60 + "\n")
