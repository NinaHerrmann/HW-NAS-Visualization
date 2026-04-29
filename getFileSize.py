import os
import re
import csv
import bz2
import pickle
from pathlib import Path

import pandas as pd

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
import torch
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models

MODELS_DIR = Path("/scratch/tmp/n_herr03/NATS_Benchmark/models/espdl")
OUT_DIR = Path("/scratch/tmp/n_herr03/hwnas/result")
OUT_DIR.mkdir(parents=True, exist_ok=True)
hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")

CSV_PATH = Path("model_sizes.csv")

DATASET = "cifar10"
WEIGHTPATH = "/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full"  # weightpath in your snippet

# Regex for model{idx}_{seed}.espdl (or .onnx)
PATTERN = re.compile(r"^model(?P<idx>\d+)_(?P<seed>\d+)\.(?P<ext>espdl|onnx)$")

def filesize_bytes(p: Path) -> int:
    return p.stat().st_size if p and p.exists() else 0

def load_state_dict_for_seed(idx: int, seed: int, dataset: str, weightpath: str):
    """
    Loads the net_state_dict corresponding to (dataset, seed) for a given idx.
    """
    weights_path = Path(weightpath) / f"{idx:06d}.pickle.pbz2"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    with bz2.BZ2File(weights_path, "rb") as f:
        data = pickle.load(f)

    # Your original logic tries keys and picks max(validkey)
    validkey = [k for k in data.keys() if k in data]
    if not validkey:
        raise RuntimeError(f"No valid keys found in {weights_path}")

    key = max(validkey)

    # Direct lookup is simpler if the structure matches:
    all_results = data[key]["all_results"]
    lookup_key = (dataset, seed)
    if lookup_key not in all_results:
        # If the structure is inconsistent, show available seeds:
        available = [k for k in all_results.keys() if isinstance(k, tuple) and len(k) == 2 and k[0] == dataset]
        raise KeyError(f"{lookup_key} not in all_results. Available: {available[:10]} ...")

    return all_results[lookup_key]["net_state_dict"], key

def export_torch_model(idx: int, seed: int, dataset: str):
    """
    Reconstructs the model, loads weights, and saves to OUT_DIR as .pt.
    Returns path to saved file.
    """
    # Get config for this architecture index
    netconfig = hw_api.get_net_config(idx, dataset)

    # Rebuild and load weights for this seed
    state_dict, key = load_state_dict_for_seed(idx, seed, dataset, WEIGHTPATH)
    network = get_cell_based_tiny_net(netconfig)
    network.load_state_dict(state_dict)
    network.eval()

    out_path = OUT_DIR / f"model{idx}_{seed}.pt"
    #scripted = torch.jit.script(network)
    example_input = torch.randn(1, 3, 32, 32)  # adjust to your expected input
    traced = torch.jit.trace(network, example_input)
    traced.save(out_path)
    #scripted.save(out_path)

    return out_path, key

def scan_folder(models_dir: Path):
    """
    Returns dict keyed by (idx, seed) with paths for espdl/onnx if present.
    """
    found = {}
    for p in models_dir.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.match(p.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        seed = int(m.group("seed"))
        ext = m.group("ext")

        key = (idx, seed)
        found.setdefault(key, {})
        found[key][ext] = p
    return found

def main():
    found = scan_folder(MODELS_DIR)
    df_done = pd.read_csv(CSV_PATH)
    done_set = set(zip(df_done['idx'], df_done['seed']))
    fieldnames = [
        "idx", "seed", "key",
        "espdl_size_bytes",
        "torch_size_bytes",
    ]
    rows_between_sync = 200
    rows_written = 0
    with open(CSV_PATH, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (idx, seed), paths in sorted(found.items()):
            key = (idx, seed)
            if key in done_set:
                continue
            espdl_path = paths.get("espdl")

            espdl_size = filesize_bytes(espdl_path) if espdl_path else 0

            # Export torch model for this (idx, seed)
            try:
                torch_path = OUT_DIR / f"model{idx}_{seed}.pt"
                if not torch_path.exists():
                    torch_path, key = export_torch_model(idx, seed, DATASET)
                torch_size = filesize_bytes(torch_path)
            except Exception as e:
                torch_size = 0
                key = 0
                print(f"[WARN] Failed idx={idx} seed={seed}: {e}")

            writer.writerow({
                "idx": idx,
                "seed": seed,
                "key": key,
                "espdl_size_bytes": espdl_size,
                "torch_size_bytes": torch_size,
            })
            rows_written += 1
            if rows_written % rows_between_sync == 0:
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    print(f"[WARN] fsync failed: {e}")

    print(f"Wrote {CSV_PATH} with {len(found)} rows.")

if __name__ == "__main__":
    main()
