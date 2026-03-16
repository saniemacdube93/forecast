"""
Dynamic RECIPE — Inference Script.

Load a saved D-RECIPE checkpoint and run temporal link prediction
on a query dataset, with optional semantic filtering.

Usage
-----
# Single query
python dynamic_inference.py \
    --DATASET icews14 \
    --CHECKPOINT ./results/icews14_llama2/checkpoints/final_model.pt \
    --MODEL_NAME meta-llama/Llama-2-7b-hf \
    --QUERY_FILE ./data/original/icews14/test.txt \
    --OUTPUT_FILE ./results/icews14_llama2/predictions.json

# Evaluate on test set and print Hits@k
python dynamic_inference.py --DATASET icews14 --CHECKPOINT ... --EVAL
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from dynamic_recipe.mps_utils      import get_device, get_dtype, load_model_mps, \
                                           get_lora_config_mps
from dynamic_recipe.dynamic_evaler import DynamicEvaler, build_test_prompts, \
                                           compute_hits
from dynamic_recipe.stream_processor import load_dataset_splits, _read_facts_file

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_inference_args():
    p = argparse.ArgumentParser(description="D-RECIPE Inference")
    p.add_argument("--DATASET",     type=str, required=True,
                   choices=["icews14", "icews18", "GDELT", "YAGO"])
    p.add_argument("--DATA_PATH",   type=str, default="./data/original/")
    p.add_argument("--CHECKPOINT",  type=str, required=True,
                   help="Path to saved model state dict (.pt)")
    p.add_argument("--MODEL_NAME",  type=str,
                   default="meta-llama/Llama-2-7b-hf",
                   choices=["meta-llama/Llama-2-7b-hf",
                            "meta-llama/Meta-Llama-3-8B"])
    p.add_argument("--QUERY_FILE",  type=str, default="",
                   help="Optional: custom query file (tab/space-separated s r o t)")
    p.add_argument("--OUTPUT_FILE", type=str, default="",
                   help="JSON file to write predictions to")
    p.add_argument("--MAX_SAMPLES", type=int, default=500)
    p.add_argument("--TOP_K",       type=int, default=10)
    p.add_argument("--SIM_THRESHOLD", type=float, default=0.6)
    p.add_argument("--EVAL",        action="store_true",
                   help="Print Hits@1/3/10 after inference")
    p.add_argument("--DEVICE",      type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    # LoRA params (must match training config)
    p.add_argument("--LORA_R",       type=int, default=8)
    p.add_argument("--LORA_ALPHA",   type=int, default=16)
    p.add_argument("--LORA_DROPOUT", type=float, default=0.05)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load ID maps
# ---------------------------------------------------------------------------

def load_id_maps(data_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
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


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_inference(args):
    # Device
    device = get_device() if args.DEVICE == "auto" else torch.device(args.DEVICE)
    logger.info(f"Inference device: {device}")

    # Dataset paths
    data_path = os.path.join(args.DATA_PATH, args.DATASET)
    entity2id, id2entity, relation2id, id2relation = load_id_maps(data_path)

    # Load facts
    train_facts, valid_facts, test_facts = load_dataset_splits(data_path)
    if args.QUERY_FILE:
        query_facts = _read_facts_file(args.QUERY_FILE)
    else:
        query_facts = test_facts[:args.MAX_SAMPLES]

    logger.info(f"Running inference on {len(query_facts)} queries")

    # Build prompts using full training history
    all_history = train_facts + valid_facts
    test_prompts = build_test_prompts(
        query_facts, id2entity, id2relation, all_history
    )

    # Load model
    logger.info(f"Loading model: {args.MODEL_NAME}")
    lora_config = get_lora_config_mps(args)
    model, tokenizer = load_model_mps(
        args.MODEL_NAME,
        lora_config=lora_config,
        device=device,
        gradient_checkpointing=False,   # no need during inference
    )

    # Load saved checkpoint
    if os.path.exists(args.CHECKPOINT):
        logger.info(f"Loading checkpoint: {args.CHECKPOINT}")
        state = torch.load(args.CHECKPOINT, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found: {args.CHECKPOINT}. "
                       "Running with base model weights.")

    # Evaluator
    evaler = DynamicEvaler(
        entity2id=entity2id,
        id2entity=id2entity,
        all_entities=list(entity2id.keys()),
        device=device,
        top_k_generate=args.TOP_K,
    )
    evaler.sem_filter.tau = args.SIM_THRESHOLD

    # Run inference
    logger.info("Generating predictions ...")
    model.eval()
    results = []
    predictions_list = []
    ground_truths    = []

    for i, item in enumerate(test_prompts):
        preds = evaler._generate_candidates(model, tokenizer, item["prompt"],
                                            n_candidates=args.TOP_K)
        # Semantic filtering
        filtered = []
        context = item["prompt"]
        for p in preds:
            if evaler.sem_filter.should_accept(p, context):
                filtered.append(p)
        if not filtered:
            filtered = preds   # fallback: no filtering

        results.append({
            "query"      : item["prompt"],
            "answer"     : item["answer"],
            "predictions": filtered[:args.TOP_K],
            "correct_in_top1" : item["answer"] in filtered[:1],
            "correct_in_top3" : item["answer"] in filtered[:3],
            "correct_in_top10": item["answer"] in filtered[:10],
        })
        predictions_list.append(filtered)
        ground_truths.append(item["answer"])

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(test_prompts)}")

    # Metrics
    if args.EVAL:
        metrics = compute_hits(predictions_list, ground_truths)
        logger.info("\n" + "=" * 50)
        logger.info(f"  Dataset:  {args.DATASET}")
        logger.info(f"  Model:    {args.MODEL_NAME}")
        logger.info(f"  Samples:  {len(ground_truths)}")
        logger.info(f"  Hits@1:   {metrics['Hits@1']:.4f}")
        logger.info(f"  Hits@3:   {metrics['Hits@3']:.4f}")
        logger.info(f"  Hits@10:  {metrics['Hits@10']:.4f}")
        logger.info("=" * 50)

    # Save predictions
    if args.OUTPUT_FILE:
        Path(args.OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "dataset"    : args.DATASET,
            "model"      : args.MODEL_NAME,
            "checkpoint" : args.CHECKPOINT,
            "n_queries"  : len(results),
            "metrics"    : compute_hits(predictions_list, ground_truths) if args.EVAL else {},
            "predictions": results,
        }
        with open(args.OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Predictions saved to {args.OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    args = parse_inference_args()
    run_inference(args)
