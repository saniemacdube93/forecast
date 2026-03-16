"""
MPS / CUDA / CPU compatibility utilities for Apple Silicon and beyond.

Key principle: replace bitsandbytes 4-bit/8-bit quantization (CUDA-only)
with bfloat16 / float16 precision on MPS and CUDA, float32 on CPU.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Return the best floating-point dtype for the given device."""
    if device.type == "mps":
        return torch.bfloat16   # MPS supports bfloat16 from PyTorch 2.1+
    elif device.type == "cuda":
        return torch.bfloat16
    else:
        return torch.float32


def is_mps() -> bool:
    return torch.backends.mps.is_available()


def is_cuda() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Model loading — MPS-compatible (no bitsandbytes)
# ---------------------------------------------------------------------------

def load_model_mps(
    model_name: str,
    lora_config: Optional[LoraConfig] = None,
    device: Optional[torch.device] = None,
    gradient_checkpointing: bool = True,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """
    Load a causal LM with LoRA, compatible with MPS / CUDA / CPU.

    Replaces the bitsandbytes-based `get_model_and_tokenizer()` in utils.py
    with a device-agnostic variant using dtype precision instead of quantisation.
    """
    if device is None:
        device = get_device()
    dtype = get_dtype(device)

    # Determine device_map: on MPS use explicit .to(device) instead of "auto"
    if device.type == "mps":
        device_map = None   # we call .to(device) manually
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Move to MPS explicitly (device_map="auto" not supported on MPS)
    if device.type == "mps":
        model = model.to(device)

    # Apply LoRA if config provided
    if lora_config is not None:
        model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing to reduce MPS peak memory
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Make input embeddings trainable (same as utils.py)
    embeddings = model.get_input_embeddings()
    if hasattr(embeddings, "weight"):
        embeddings.weight.requires_grad = True

    model.config.use_cache = False

    # Tokenizer
    if "Llama-3" in model_name or "llama-3" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, pad_token="</s>"
        )

    return model, tokenizer


def get_lora_config_mps(args) -> LoraConfig:
    """Return a LoRA config compatible with any device (same hyperparams as original)."""
    target_modules = (
        ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA-3
        if "Llama-3" in args.MODEL_NAME else
        ["q_proj", "v_proj"]                        # LLaMA-2
    )
    return LoraConfig(
        r=args.LORA_R,
        lora_alpha=args.LORA_ALPHA,
        lora_dropout=args.LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


# ---------------------------------------------------------------------------
# MPS-compatible AdamW (no bnb.optim.AdamW8bit)
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, lr: float = 3e-4) -> torch.optim.Optimizer:
    """Return AdamW — standard PyTorch, works on MPS/CUDA/CPU."""
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=False,  # fused=True is CUDA-only
    )


# ---------------------------------------------------------------------------
# Memory utilities for MPS
# ---------------------------------------------------------------------------

def empty_cache(device: torch.device) -> None:
    """Free cached memory on the given device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def memory_info(device: torch.device) -> str:
    """Return a human-readable memory usage string."""
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        return f"CUDA — allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB"
    elif device.type == "mps":
        # MPS memory stats (PyTorch ≥ 2.1)
        try:
            alloc = torch.mps.current_allocated_memory() / 1e9
            return f"MPS — allocated: {alloc:.2f} GB"
        except Exception:
            return "MPS — memory stats unavailable"
    else:
        import psutil
        vm = psutil.virtual_memory()
        return f"CPU — used: {vm.used/1e9:.2f} GB / {vm.total/1e9:.2f} GB"
