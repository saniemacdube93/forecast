"""
Dynamic RECIPE (D-RECIPE) — Main Training Script.

Continual TKG forecasting with:
  1. Edge stream simulation from static datasets
  2. Incremental rule mining (no full re-mining)
  3. EWC + Knowledge Distillation + Experience Replay
  4. Adaptive RBMH sampling with incremental hop computation
  5. Per-period evaluation + BWT / FWT metrics
  6. Full Apple MPS (M-series) compatibility

Usage
-----
python dynamic_main.py --DATASET icews14 --MODEL_NAME meta-llama/Llama-2-7b-hf \
       --STREAM_CHUNK_SIZE 500 --SEED_CHUNKS 5 --N_CHUNKS 20 \
       --RESULTS_DIR ./results/icews14_llama2
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# D-RECIPE modules
from dynamic_config import parse_args, print_config
from dynamic_recipe.mps_utils       import get_device, get_dtype, load_model_mps, \
                                            get_lora_config_mps, get_optimizer, \
                                            empty_cache, memory_info
from dynamic_recipe.memory_buffer   import PriorityReplayBuffer
from dynamic_recipe.incremental_rules import IncrementalRuleMiner
from dynamic_recipe.stream_processor  import EdgeStreamProcessor, load_dataset_splits
from dynamic_recipe.continual_learner import EWCRegularizer, KnowledgeDistiller, \
                                              ContinualTrainer
from dynamic_recipe.adaptive_sampler  import DynamicRBMHSampler
from dynamic_recipe.dynamic_evaler    import DynamicEvaler, build_test_prompts

# Compatibility with original codebase
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "icews14" : {"num_relations": 230, "period": 24, "granularity": 1},
    "icews18" : {"num_relations": 256, "period": 24, "granularity": 1},
    "GDELT"   : {"num_relations": 238, "period": 1,  "granularity": 1},
    "YAGO"    : {"num_relations": 24,  "period": 1,  "granularity": 1},
}


def get_data_path(base_path: str, dataset: str) -> str:
    return os.path.join(base_path, dataset)


def load_id_maps(data_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load entity2id, id2entity, relation2id, id2relation from dataset dir."""
    def load_json(name):
        p = os.path.join(data_path, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    entity2id   = load_json("entity2id.json")
    relation2id = load_json("relation2id.json")
    id2entity   = {int(v): k for k, v in entity2id.items()}
    id2relation = {int(v): k for k, v in relation2id.items()}
    return entity2id, id2entity, relation2id, id2relation


def facts_to_training_samples(
    facts: List[Tuple],
    id2entity: Dict,
    id2relation: Dict,
    sampler: DynamicRBMHSampler,
    max_samples: int = 1024,
) -> List[Dict]:
    """
    Convert raw facts to training dicts {'input': prompt, 'output': entity}.
    Uses the DynamicRBMHSampler to build contextual prompts.
    """
    samples = []
    facts_subset = random.sample(facts, min(max_samples, len(facts)))

    for s, r, o, t in facts_subset:
        subj_name = id2entity.get(s, str(s))
        rel_name  = id2relation.get(r, str(r))
        obj_name  = id2entity.get(o, str(o))

        # Sample historical context
        hist_facts = sampler.sample(s, r, t)
        context_lines = []
        for hs, hr, ho, ht in hist_facts:
            hs_n = id2entity.get(hs, str(hs))
            hr_n = id2relation.get(hr, str(hr))
            ho_n = id2entity.get(ho, str(ho))
            context_lines.append(f"{ht}:[{hs_n}, {hr_n}, {ho_n}]")
        context = "\n".join(context_lines)

        prompt = (
            f"Context:\n{context}\n\n"
            f"Query: {t}:[{subj_name}, {rel_name}, ?]\n"
            f"Prediction: "
        )
        samples.append({
            "input"  : prompt,
            "output" : obj_name,
            "subject": s,
            "relation": r,
            "object" : o,
            "time"   : t,
        })
    return samples


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(42)
    print_config(args)

    # ------------------------------------------------------------------ #
    # Device & paths
    # ------------------------------------------------------------------ #
    if args.DEVICE == "auto":
        device = get_device()
    else:
        device = torch.device(args.DEVICE)
    logger.info(f"Using device: {device}  ({memory_info(device)})")

    results_dir = Path(args.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir    = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    data_path = get_data_path(args.DATA_PATH, args.DATASET)
    logger.info(f"Loading dataset from {data_path}")

    entity2id, id2entity, relation2id, id2relation = load_id_maps(data_path)
    train_facts, valid_facts, test_facts = load_dataset_splits(data_path)

    logger.info(f"Train: {len(train_facts)} | Valid: {len(valid_facts)} | "
                f"Test: {len(test_facts)}")

    # ------------------------------------------------------------------ #
    # Initialise D-RECIPE components
    # ------------------------------------------------------------------ #
    stream = EdgeStreamProcessor(
        time_granularity=DATASET_CONFIG[args.DATASET]["granularity"]
    )
    rule_miner = IncrementalRuleMiner(
        max_rules=args.MAX_RULES,
        min_support=args.MIN_RULE_SUPPORT,
        min_confidence=args.MIN_RULE_CONFIDENCE,
    )
    replay_buffer = PriorityReplayBuffer(capacity=args.BUFFER_SIZE)
    ewc   = EWCRegularizer(lambda_ewc=args.EWC_LAMBDA if args.USE_EWC else 0.0)
    dist  = KnowledgeDistiller(temperature=args.KD_TEMPERATURE,
                               gamma_kd=args.KD_GAMMA if args.USE_KD else 0.0)
    sampler = DynamicRBMHSampler(
        stream_processor=stream,
        rule_miner=rule_miner,
        gamma1=args.GAMMA1, gamma2=args.GAMMA2,
        gamma3=args.GAMMA3, gamma4=args.GAMMA4,
        n_tlr_facts=args.N_TLR_FACTS,
        n_total_facts=args.N_TOTAL_FACTS,
    )

    # ------------------------------------------------------------------ #
    # Load model (MPS-compatible)
    # ------------------------------------------------------------------ #
    logger.info(f"Loading model: {args.MODEL_NAME}")
    lora_config = get_lora_config_mps(args)
    model, tokenizer = load_model_mps(
        args.MODEL_NAME,
        lora_config=lora_config,
        device=device,
        gradient_checkpointing=bool(args.GRADIENT_CHECKPOINTING),
    )
    optimizer = get_optimizer(model, lr=args.LEARNING_RATE)

    # ------------------------------------------------------------------ #
    # Evaluator
    # ------------------------------------------------------------------ #
    all_entity_names = list(entity2id.keys())
    evaler = DynamicEvaler(
        entity2id=entity2id,
        id2entity=id2entity,
        all_entities=all_entity_names,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # Continual trainer
    # ------------------------------------------------------------------ #
    trainer = ContinualTrainer(
        ewc_regularizer=ewc,
        distiller=dist,
        replay_buffer=replay_buffer,
        alpha_contrastive=args.CONTRASTIVE_WEIGHT,
        replay_ratio=args.REPLAY_RATIO if args.USE_REPLAY else 0.0,
        fisher_update_freq=args.FISHER_UPDATE_FREQ,
        fisher_n_batches=args.FISHER_N_BATCHES,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # Apply ablation overrides
    # ------------------------------------------------------------------ #
    if args.ABLATION == "no_ewc":
        ewc.lambda_ewc = 0.0
        logger.info("[Ablation] EWC disabled")
    elif args.ABLATION == "no_replay":
        trainer.replay_ratio = 0.0
        logger.info("[Ablation] Replay disabled")
    elif args.ABLATION == "no_kd":
        dist.gamma_kd = 0.0
        logger.info("[Ablation] Knowledge distillation disabled")
    elif args.ABLATION == "no_filtering":
        evaler.sem_filter.tau = 0.0   # always accept
        logger.info("[Ablation] Semantic filtering disabled")
    elif args.ABLATION == "naive_finetune":
        ewc.lambda_ewc = 0.0
        trainer.replay_ratio = 0.0
        dist.gamma_kd = 0.0
        logger.info("[Ablation] Naive fine-tune (no CL components)")

    # ------------------------------------------------------------------ #
    # Pre-stream seeding: ingest first SEED_CHUNKS batches
    # ------------------------------------------------------------------ #
    logger.info(f"Seeding with first {args.SEED_CHUNKS} chunks "
                f"(chunk_size={args.STREAM_CHUNK_SIZE})")

    stream_gen = stream.simulate_stream(data_path, chunk_size=args.STREAM_CHUNK_SIZE)

    seed_facts: List[Tuple] = []
    for chunk_idx, chunk in stream_gen:
        stream.ingest_batch(chunk)
        for edge in chunk:
            rule_miner.update_rules(edge)
            sampler.update_on_edge(edge)
        seed_facts.extend(chunk)
        if chunk_idx + 1 >= args.SEED_CHUNKS:
            break

    logger.info(f"Seed phase: {len(seed_facts)} facts, "
                f"{len(stream.entity_set)} entities, "
                f"{len(rule_miner)} rules")

    # Initial training on seed data
    seed_samples = facts_to_training_samples(
        seed_facts, id2entity, id2relation, sampler,
        max_samples=1024 if not args.DEBUG else 64,
    )
    logger.info(f"Initial training on {len(seed_samples)} seed samples ...")
    loss_info = trainer.train_on_chunk(
        model, optimizer, seed_samples, tokenizer,
        n_epochs=5 if not args.DEBUG else 1,
        batch_size=args.MICRO_BATCH_SIZE,
    )
    logger.info(f"Seed training loss: {loss_info}")

    # Initialise EWC Fisher and teacher snapshot
    from torch.utils.data import DataLoader
    from dynamic_recipe.continual_learner import _SimpleQADataset, _collate_fn
    seed_dataset = _SimpleQADataset(seed_samples, tokenizer,
                                    max_len=512, device=device)
    seed_loader  = DataLoader(seed_dataset, batch_size=2, shuffle=False,
                              collate_fn=_collate_fn)
    trainer.initialise(model, seed_loader)

    # ------------------------------------------------------------------ #
    # Continual streaming loop
    # ------------------------------------------------------------------ #
    all_metrics: List[Dict] = []
    period_test_data: Dict[int, List[Dict]] = {}
    chunk_count = 0

    # Build test prompts from test_facts
    test_prompts = build_test_prompts(
        test_facts[:500 if not args.DEBUG else 50],
        id2entity, id2relation, seed_facts
    )

    logger.info("Starting continual streaming loop ...")
    for chunk_idx, chunk in stream_gen:
        if args.N_CHUNKS > 0 and chunk_count >= args.N_CHUNKS:
            break

        t_start = time.time()

        # 1. Ingest new edges
        n_new_entities = stream.ingest_batch(chunk)
        for edge in chunk:
            rule_miner.update_rules(edge)
            sampler.update_on_edge(edge)

        # 2. Build training samples from new chunk
        new_samples = facts_to_training_samples(
            chunk, id2entity, id2relation, sampler,
            max_samples=512 if not args.DEBUG else 32,
        )

        # 3. Continual training
        loss_info = trainer.train_on_chunk(
            model, optimizer, new_samples, tokenizer,
            n_epochs=args.EPOCHS,
            batch_size=args.MICRO_BATCH_SIZE,
        )

        chunk_count += 1
        elapsed = time.time() - t_start

        logger.info(
            f"Chunk {chunk_count:03d} | edges={len(chunk)} | "
            f"new_entities={n_new_entities} | "
            f"rules={len(rule_miner)} | "
            f"loss={loss_info['total']:.4f} | {elapsed:.1f}s"
        )

        # 4. Evaluate every N chunks
        if chunk_count % args.EVAL_EVERY_N_CHUNKS == 0:
            logger.info(f"Evaluating at chunk {chunk_count} ...")
            # Update test prompts with growing history
            current_history = stream.get_facts_before(
                max(stream.buckets.keys()) + 1
            )
            period_prompts = build_test_prompts(
                test_facts[:args.MAX_EVAL_SAMPLES],
                id2entity, id2relation, current_history
            )
            period_test_data[chunk_count] = period_prompts
            metrics = evaler.eval_period(
                model, tokenizer, period_prompts,
                period_id=chunk_count,
                max_samples=args.MAX_EVAL_SAMPLES,
            )
            if args.ADAPTIVE_THRESHOLD:
                evaler.sem_filter.update_threshold()

            logger.info(
                f"  Period {chunk_count}: "
                f"Hits@1={metrics['Hits@1']:.4f} "
                f"Hits@3={metrics['Hits@3']:.4f} "
                f"Hits@10={metrics['Hits@10']:.4f}"
            )
            all_metrics.append({
                "chunk": chunk_count,
                "n_entities": len(stream.entity_set),
                "n_rules": len(rule_miner),
                **metrics,
                **loss_info,
            })

        # 5. Save checkpoint
        if chunk_count % args.SAVE_EVERY == 0:
            ckpt_path = ckpt_dir / f"checkpoint_chunk{chunk_count:03d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ewc_state": ewc.state_dict(),
                "chunk_idx": chunk_count,
                "metrics": all_metrics,
            }, ckpt_path)
            rule_miner.save_state(str(ckpt_dir / f"rules_chunk{chunk_count:03d}.json"))
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------ #
    # Final evaluation & continual metrics
    # ------------------------------------------------------------------ #
    logger.info("Running final evaluation ...")
    final_history = stream.get_all_facts()
    final_prompts = build_test_prompts(
        test_facts[:args.MAX_EVAL_SAMPLES],
        id2entity, id2relation, final_history
    )
    final_metrics = evaler.eval_period(
        model, tokenizer, final_prompts,
        period_id=chunk_count + 1,
        max_samples=args.MAX_EVAL_SAMPLES,
    )
    summary = evaler.get_metrics_summary()
    summary.update(final_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("  D-RECIPE Final Results")
    logger.info("=" * 60)
    logger.info(f"  Dataset       : {args.DATASET}")
    logger.info(f"  Model         : {args.MODEL_NAME}")
    logger.info(f"  Ablation      : {args.ABLATION}")
    logger.info(f"  Chunks        : {chunk_count}")
    logger.info(f"  Hits@1        : {final_metrics['Hits@1']:.4f}")
    logger.info(f"  Hits@3        : {final_metrics['Hits@3']:.4f}")
    logger.info(f"  Hits@10       : {final_metrics['Hits@10']:.4f}")
    logger.info(f"  AvgAcc        : {summary['average_accuracy']:.4f}")
    logger.info(f"  BWT           : {summary['bwt']:.4f}")
    logger.info(f"  FWT           : {summary['fwt']:.4f}")
    logger.info("=" * 60)

    # Save results
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config"      : vars(args),
            "final"       : final_metrics,
            "continual"   : summary,
            "per_chunk"   : all_metrics,
        }, f, indent=2)
    logger.info(f"Results saved to {metrics_path}")

    # Save final model
    final_ckpt = ckpt_dir / "final_model.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.info(f"Final model saved to {final_ckpt}")

    return summary


if __name__ == "__main__":
    main()
