"""
D-RECIPE Model Setup Script.

Verifies that LLaMA-2-7B and LLaMA-3-8B are accessible via HuggingFace,
checks system memory requirements, and optionally pre-downloads model weights.

Usage
-----
python setup_models.py                          # check both models
python setup_models.py --model llama2           # check LLaMA-2 only
python setup_models.py --download               # also cache model weights
python setup_models.py --skip_hf_check         # skip HuggingFace token check
"""

import os
import sys
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "llama2": {
        "hf_id":       "meta-llama/Llama-2-7b-hf",
        "size_gb":     13.0,   # BF16 weights
        "min_ram_gb":  16.0,   # with LoRA + optimizer states
        "license_url": "https://huggingface.co/meta-llama/Llama-2-7b-hf",
        "description": "LLaMA-2-7B — Meta AI, 2023",
    },
    "llama3": {
        "hf_id":       "meta-llama/Meta-Llama-3-8B",
        "size_gb":     16.0,   # BF16 weights
        "min_ram_gb":  20.0,
        "license_url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B",
        "description": "LLaMA-3-8B — Meta AI, 2024",
    },
}

# ---------------------------------------------------------------------------
# System info helpers
# ---------------------------------------------------------------------------

def _get_system_memory_gb() -> float:
    """Return total RAM / unified memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    # Fallback: read /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / 1e6
    except Exception:
        pass
    return 0.0


def _get_device_info() -> Dict:
    """Return dict with device type and memory."""
    info = {
        "platform": platform.system(),
        "chip":     platform.processor(),
        "mps":      False,
        "cuda":     False,
        "cuda_mem_gb": 0.0,
        "mps_mem_gb":  0.0,
        "ram_gb":   _get_system_memory_gb(),
    }
    try:
        import torch
        if torch.backends.mps.is_available():
            info["mps"] = True
            try:
                # Unified memory on Apple Silicon = system RAM
                info["mps_mem_gb"] = info["ram_gb"]
            except Exception:
                pass
        if torch.cuda.is_available():
            info["cuda"] = True
            try:
                mem = torch.cuda.get_device_properties(0).total_memory
                info["cuda_mem_gb"] = mem / 1e9
                info["cuda_name"]   = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except ImportError:
        pass
    return info


def _check_hf_token() -> Tuple[bool, Optional[str]]:
    """Return (has_token, username_or_None)."""
    # Check environment variable first
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return True, "<env-var>"

    # Check ~/.cache/huggingface/token
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        tok = hf_token_path.read_text().strip()
        if tok:
            return True, "<cached>"

    # Try huggingface_hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        return True, user.get("name", "unknown")
    except Exception:
        pass

    return False, None


def _check_model_accessible(hf_id: str) -> Tuple[bool, str]:
    """
    Check whether the model exists and is accessible with current credentials.
    Returns (accessible, reason).
    """
    try:
        from huggingface_hub import model_info
        info = model_info(hf_id)
        return True, f"gated={getattr(info, 'gated', False)}"
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "gated" in err.lower():
            return False, "Model gated — accept licence at huggingface.co"
        if "404" in err:
            return False, "Model not found on HuggingFace Hub"
        return False, f"Network error: {err[:80]}"


def _is_model_cached(hf_id: str) -> Tuple[bool, Optional[str]]:
    """Return (is_cached, cache_dir_or_None)."""
    try:
        from huggingface_hub import snapshot_download, HfFileSystem
        from huggingface_hub.utils import LocalEntryNotFoundError
        # Check HF cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # model cache folder pattern
        safe_name = "models--" + hf_id.replace("/", "--")
        model_cache = cache_dir / safe_name
        if model_cache.exists():
            snapshots = list((model_cache / "snapshots").glob("*"))
            if snapshots:
                return True, str(snapshots[-1])
    except Exception:
        pass
    return False, None


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def check_model(
    model_key: str,
    download: bool = False,
    skip_hf_check: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Check / setup a single model.  Returns status dict.
    """
    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]
    result = {
        "model_key": model_key,
        "hf_id":     hf_id,
        "ready":     False,
        "cached":    False,
        "token_ok":  False,
        "accessible": False,
        "messages":  [],
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Model: {hf_id}")
        print(f"  {cfg['description']}")
        print(f"  Size (BF16): ~{cfg['size_gb']:.0f} GB | "
              f"Min RAM: ~{cfg['min_ram_gb']:.0f} GB")
        print(f"{'─'*60}")

    # 1. Check cache
    cached, cache_path = _is_model_cached(hf_id)
    result["cached"] = cached
    if cached:
        msg = f"  ✓ Model weights cached at: {cache_path}"
        result["ready"] = True
    else:
        msg = "  ○ Model NOT cached (will be downloaded on first run)"
    result["messages"].append(msg)
    if verbose:
        print(msg)

    if skip_hf_check:
        result["token_ok"] = True
        result["accessible"] = True
        if verbose:
            print("  ○ HuggingFace check skipped (--skip_hf_check)")
        return result

    # 2. Check HF token
    has_token, username = _check_hf_token()
    result["token_ok"] = has_token
    if has_token:
        msg = f"  ✓ HuggingFace token found (user: {username})"
    else:
        msg = ("  ✗ No HuggingFace token found.\n"
               "    Run: huggingface-cli login\n"
               "    Or:  export HF_TOKEN=<your_token>")
    result["messages"].append(msg)
    if verbose:
        print(msg)

    if not has_token:
        if verbose:
            print(f"\n  Licence URL: {cfg['license_url']}")
        return result

    # 3. Check model access
    accessible, reason = _check_model_accessible(hf_id)
    result["accessible"] = accessible
    if accessible:
        msg = f"  ✓ Model accessible ({reason})"
        result["ready"] = True
    else:
        msg = (f"  ✗ Model NOT accessible: {reason}\n"
               f"    Accept the licence at: {cfg['license_url']}")
    result["messages"].append(msg)
    if verbose:
        print(msg)

    # 4. Optional download
    if download and accessible and not cached:
        if verbose:
            print(f"\n  Downloading {hf_id} weights (this may take a while) …")
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(hf_id, ignore_patterns=["*.bin"])
            result["cached"] = True
            result["messages"].append(f"  ✓ Downloaded to: {path}")
            if verbose:
                print(f"  ✓ Downloaded to: {path}")
        except Exception as e:
            result["messages"].append(f"  ✗ Download failed: {e}")
            if verbose:
                print(f"  ✗ Download failed: {e}")

    return result


def check_system(verbose: bool = True) -> Dict:
    """Print system hardware report."""
    info = _get_device_info()

    if verbose:
        print("\n" + "="*60)
        print("  System Hardware Report")
        print("="*60)
        print(f"  OS:        {info['platform']}")
        print(f"  Processor: {info['chip']}")
        print(f"  RAM:       {info['ram_gb']:.1f} GB")
        if info["mps"]:
            print(f"  MPS:       ✓ Available (Apple Silicon)")
            print(f"  MPS RAM:   {info['mps_mem_gb']:.1f} GB (unified memory)")
        if info["cuda"]:
            print(f"  CUDA:      ✓ Available — {info.get('cuda_name', 'unknown')}")
            print(f"  VRAM:      {info['cuda_mem_gb']:.1f} GB")
        if not info["mps"] and not info["cuda"]:
            print("  ⚠  No GPU accelerator detected; training will be VERY slow on CPU")
        print("="*60)

    return info


def setup_models(
    models: Optional[list] = None,
    download: bool = False,
    skip_hf_check: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Check/setup all requested models.
    Returns {model_key: status_dict}.
    """
    if models is None:
        models = list(MODELS.keys())

    sys_info = check_system(verbose=verbose)
    results = {}

    for mk in models:
        if mk not in MODELS:
            print(f"  Unknown model key: {mk}. Choose from {list(MODELS.keys())}")
            continue
        cfg = MODELS[mk]
        req = cfg["min_ram_gb"]
        avail = (sys_info.get("mps_mem_gb") or
                 sys_info.get("cuda_mem_gb") or
                 sys_info.get("ram_gb", 0))
        if avail > 0 and avail < req:
            print(f"\n  ⚠ Warning: {mk} requires ~{req:.0f} GB; "
                  f"detected {avail:.1f} GB. Performance may be degraded.")
        results[mk] = check_model(
            mk,
            download=download,
            skip_hf_check=skip_hf_check,
            verbose=verbose,
        )

    if verbose:
        print("\n" + "="*60)
        print("  Model Setup Summary")
        print("="*60)
        for mk, res in results.items():
            status = "✓ Ready" if res["ready"] else "○ Will download on first run"
            cached = "(cached)" if res["cached"] else "(not cached)"
            print(f"  {mk:<10} {status:<35} {cached}")
        print("="*60)

        print("\n  To train with LLaMA-2-7B:")
        print("    python dynamic_main.py --MODEL_NAME meta-llama/Llama-2-7b-hf ...")
        print("\n  To train with LLaMA-3-8B:")
        print("    python dynamic_main.py --MODEL_NAME meta-llama/Meta-Llama-3-8B ...")
        print("\n  Full experiment:")
        print("    python run_all.py")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="D-RECIPE Model Setup")
    p.add_argument("--model", type=str, default="all",
                   choices=list(MODELS.keys()) + ["all"],
                   help="Which model to check (default: all)")
    p.add_argument("--download", action="store_true",
                   help="Download model weights to HuggingFace cache")
    p.add_argument("--skip_hf_check", action="store_true",
                   help="Skip HuggingFace token and access checks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = setup_models(
        models=models,
        download=args.download,
        skip_hf_check=args.skip_hf_check,
    )
    all_ready = all(r["token_ok"] for r in results.values())
    sys.exit(0 if all_ready else 1)
