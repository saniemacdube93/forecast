"""
D-RECIPE Dataset Setup Script.

Checks whether ICEWS14, ICEWS18, GDELT, and YAGO datasets exist under
data/original/.  If any dataset is missing it attempts to download it
from the TLogic GitHub mirror, then generates entity2id.json and
relation2id.json from the raw .txt files.

Usage
-----
python setup_datasets.py                         # check + download all 4
python setup_datasets.py --dataset icews14       # single dataset
python setup_datasets.py --data_dir ./mydata     # custom data root
python setup_datasets.py --skip_download         # check only, no download
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "icews14": {
        "entities":  7128,
        "relations": 230,
        "train_facts": 74845,
        "description": "ICEWS 2014 — daily political events",
    },
    "icews18": {
        "entities":  23033,
        "relations": 256,
        "train_facts": 373018,
        "description": "ICEWS 2018 — daily political events (larger)",
    },
    "GDELT": {
        "entities":  5850,
        "relations": 238,
        "train_facts": 79319,
        "description": "GDELT — 15-minute global events",
    },
    "YAGO": {
        "entities":  10778,
        "relations": 24,
        "train_facts": 220393,
        "description": "YAGO 3 — annual biographical facts",
    },
}

# Mirror sources (tried in order)
# Each entry: (base_url, path_template)
# path_template uses {dataset} for the dataset name
DOWNLOAD_MIRRORS = [
    # TLogic GitHub raw
    (
        "https://raw.githubusercontent.com/LongmeiLi/TLogic/main/src/data",
        "{dataset}/{split}.txt",
    ),
    # xERTE GitHub raw (uses different capitalisation)
    (
        "https://raw.githubusercontent.com/TemporalKGTeam/xERTE/main/data",
        "{dataset}/{split}.txt",
    ),
    # TANGO GitHub raw
    (
        "https://raw.githubusercontent.com/TemporalKGTeam/TANGO/main/data",
        "{dataset}/{split}.txt",
    ),
]

# How dataset names map to each mirror (some repos use different names)
DATASET_ALIASES = {
    "icews14": ["icews14", "ICEWS14"],
    "icews18": ["icews18", "ICEWS18"],
    "GDELT":   ["GDELT", "gdelt"],
    "YAGO":    ["YAGO", "yago"],
}

SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    """Attempt to download url to dest. Returns True on success."""
    try:
        if verbose:
            print(f"    Downloading {url} …")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100:
            # Probably a 404 HTML page served as 200
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if verbose:
            print(f"    ✓ Saved ({len(data)/1024:.1f} KB) → {dest}")
        return True
    except Exception as e:
        if verbose:
            print(f"    ✗ Failed: {e}")
        return False


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _read_facts(path: Path) -> List[Tuple[int, int, int, int]]:
    """Read a whitespace/tab-separated facts file → list of (s,r,o,t) tuples."""
    facts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    facts.append((s, r, o, t))
                except ValueError:
                    continue
    return facts


def _generate_id_maps(dataset_dir: Path) -> Tuple[Dict, Dict]:
    """
    Build entity2id.json and relation2id.json from {train,valid,test}.txt.
    IDs are already integers in the files; we just create name→id mappings
    using the integer as both key and value (since no string names available).
    """
    entity_ids = set()
    relation_ids = set()

    for split in SPLITS:
        path = dataset_dir / f"{split}.txt"
        if path.exists():
            for s, r, o, t in _read_facts(path):
                entity_ids.update([s, o])
                relation_ids.add(r)

    # Maps: str(id) → id  (consistent with HuggingFace / RECIPE-TKG convention)
    entity2id   = {str(e): e for e in sorted(entity_ids)}
    relation2id = {str(r): r for r in sorted(relation_ids)}

    e2id_path = dataset_dir / "entity2id.json"
    r2id_path = dataset_dir / "relation2id.json"

    with open(e2id_path, "w") as f:
        json.dump(entity2id, f, indent=2)
    with open(r2id_path, "w") as f:
        json.dump(relation2id, f, indent=2)

    # stat.txt: num_entities num_relations
    stat_path = dataset_dir / "stat.txt"
    with open(stat_path, "w") as f:
        f.write(f"{len(entity_ids)}\t{len(relation_ids)}\n")

    print(f"    Generated entity2id.json ({len(entity_ids)} entities), "
          f"relation2id.json ({len(relation_ids)} relations)")
    return entity2id, relation2id


def _verify_dataset(dataset_dir: Path, dataset: str, verbose: bool = True) -> bool:
    """Check that all required files exist and are non-empty."""
    ok = True
    for split in SPLITS:
        p = dataset_dir / f"{split}.txt"
        if not p.exists():
            if verbose:
                print(f"  ✗ Missing: {p}")
            ok = False
        elif _count_lines(p) == 0:
            if verbose:
                print(f"  ✗ Empty:   {p}")
            ok = False
    if verbose and ok:
        n_train = _count_lines(dataset_dir / "train.txt")
        info = DATASET_INFO.get(dataset, {})
        expected = info.get("train_facts", 0)
        status = "✓" if abs(n_train - expected) / max(expected, 1) < 0.05 else "~"
        print(f"  {status} {dataset}: {n_train:,} train facts "
              f"(expected ≈ {expected:,})")
    return ok


def _download_dataset(dataset: str, dataset_dir: Path, verbose: bool = True) -> bool:
    """Try all mirrors until one succeeds for all splits."""
    aliases = DATASET_ALIASES.get(dataset, [dataset])

    for base_url, tmpl in DOWNLOAD_MIRRORS:
        for alias in aliases:
            success_count = 0
            for split in SPLITS:
                rel_path = tmpl.format(dataset=alias, split=split)
                url      = f"{base_url}/{rel_path}"
                dest     = dataset_dir / f"{split}.txt"
                if dest.exists() and _count_lines(dest) > 10:
                    success_count += 1
                    continue
                if _download_file(url, dest, verbose=verbose):
                    success_count += 1
            if success_count == len(SPLITS):
                return True
        # All aliases failed for this mirror; try next

    return False


def _create_synthetic_dataset(dataset_dir: Path, dataset: str) -> None:
    """
    Create a minimal synthetic dataset for testing when download fails.
    Uses realistic statistics from DATASET_INFO.
    """
    info = DATASET_INFO.get(dataset, {})
    n_entities  = info.get("entities", 1000)
    n_relations = info.get("relations", 50)

    import random
    rng = random.Random(42)

    def make_split(n_facts: int, t_range: Tuple[int, int]) -> List[str]:
        lines = []
        for _ in range(n_facts):
            s = rng.randint(0, n_entities - 1)
            r = rng.randint(0, n_relations - 1)
            o = rng.randint(0, n_entities - 1)
            t = rng.randint(*t_range)
            lines.append(f"{s}\t{r}\t{o}\t{t}")
        return lines

    n_train = min(info.get("train_facts", 5000), 5000)  # cap for synthetic
    splits = {
        "train": make_split(n_train,       (0, 364)),
        "valid": make_split(n_train // 10, (365, 399)),
        "test":  make_split(n_train // 10, (400, 429)),
    }
    for split, lines in splits.items():
        path = dataset_dir / f"{split}.txt"
        path.write_text("\n".join(lines) + "\n")

    print(f"  ⚠ Created SYNTHETIC {dataset} ({n_train:,} train facts). "
          f"Results will NOT be comparable to published baselines.")
    print(f"    To get real data: download from "
          f"https://drive.google.com/drive/folders/1kdo_pn6PDig7SJ61feugRDwckcTZ8KHX")


# ---------------------------------------------------------------------------
# Main setup function
# ---------------------------------------------------------------------------

def setup_dataset(
    dataset: str,
    data_root: str = "./data/original",
    skip_download: bool = False,
    force_regenerate_maps: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Ensure dataset is available under data_root/{dataset}/.

    Returns True if dataset is ready (real or synthetic).
    """
    dataset_dir = Path(data_root) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset}")
        info = DATASET_INFO.get(dataset, {})
        print(f"  {info.get('description', '')}")
        print(f"{'='*60}")

    already_ok = _verify_dataset(dataset_dir, dataset, verbose=False)

    if already_ok:
        if verbose:
            print(f"  ✓ {dataset} already present.")
            _verify_dataset(dataset_dir, dataset, verbose=True)
    else:
        if skip_download:
            if verbose:
                print(f"  ✗ {dataset} missing (skip_download=True, will not download).")
            return False

        if verbose:
            print(f"  Dataset not found. Attempting download …")

        downloaded = _download_dataset(dataset, dataset_dir, verbose=verbose)

        if not downloaded:
            if verbose:
                print(f"\n  ⚠ All download mirrors failed for {dataset}.")
                print(f"  Falling back to SYNTHETIC data for smoke-testing.")
                print(f"  For real experiments, download from Google Drive:")
                print(f"  https://drive.google.com/drive/folders/"
                      f"1kdo_pn6PDig7SJ61feugRDwckcTZ8KHX")
            _create_synthetic_dataset(dataset_dir, dataset)

    # Generate / update id maps if missing or forced
    e2id = dataset_dir / "entity2id.json"
    r2id = dataset_dir / "relation2id.json"
    if not e2id.exists() or not r2id.exists() or force_regenerate_maps:
        if verbose:
            print(f"  Generating id maps …")
        _generate_id_maps(dataset_dir)
    else:
        if verbose:
            with open(e2id) as f:
                n_e = len(json.load(f))
            with open(r2id) as f:
                n_r = len(json.load(f))
            print(f"  ID maps already exist: {n_e} entities, {n_r} relations")

    # Final verification
    ok = _verify_dataset(dataset_dir, dataset, verbose=verbose)
    return ok


def setup_all_datasets(
    data_root: str = "./data/original",
    datasets: Optional[List[str]] = None,
    skip_download: bool = False,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    Set up all benchmark datasets.  Returns {dataset: is_ready} dict.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())

    results = {}
    for ds in datasets:
        ok = setup_dataset(ds, data_root=data_root,
                           skip_download=skip_download, verbose=verbose)
        results[ds] = ok

    if verbose:
        print("\n" + "="*60)
        print("  Dataset Setup Summary")
        print("="*60)
        for ds, ok in results.items():
            status = "✓ Ready" if ok else "✗ Failed"
            print(f"  {ds:<12} {status}")
        print("="*60)
        all_ok = all(results.values())
        if all_ok:
            print("\n  All datasets ready. You can now run:")
            print("  python run_all.py --skip_setup")
        else:
            print("\n  Some datasets failed. Check the messages above.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="D-RECIPE Dataset Setup")
    p.add_argument("--dataset", type=str, default="all",
                   choices=list(DATASET_INFO.keys()) + ["all"],
                   help="Which dataset to set up (default: all)")
    p.add_argument("--data_dir", type=str, default="./data/original",
                   help="Root data directory (default: ./data/original)")
    p.add_argument("--skip_download", action="store_true",
                   help="Only check; do not attempt downloads")
    p.add_argument("--force_maps", action="store_true",
                   help="Regenerate entity2id / relation2id even if they exist")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "all":
        results = setup_all_datasets(
            data_root=args.data_dir,
            skip_download=args.skip_download,
        )
        sys.exit(0 if all(results.values()) else 1)
    else:
        ok = setup_dataset(
            args.dataset,
            data_root=args.data_dir,
            skip_download=args.skip_download,
            force_regenerate_maps=args.force_maps,
        )
        sys.exit(0 if ok else 1)
