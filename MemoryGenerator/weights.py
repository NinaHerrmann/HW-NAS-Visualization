import bz2
import pickle
from pathlib import Path

import torch


def load_state_dicts(weightpath, idx, dataset):
    """Return {seed: state_dict}. Prefers extracted .pth, falls back to .pickle.pbz2."""
    pth = Path(weightpath) / f"{idx:06d}.pth"
    if pth.exists():
        per_seed = torch.load(pth, map_location="cpu")["weights"].get(dataset, {})
        return {s: {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                    for k, v in sd.items()}
                for s, sd in per_seed.items()}

    pbz2 = Path(weightpath) / f"{idx:06d}.pickle.pbz2"
    if not pbz2.exists():
        return {}
    with bz2.BZ2File(pbz2, "rb") as f:
        data = pickle.load(f)
    all_results = data[max(data.keys())]["all_results"]
    return {k[1]: r["net_state_dict"] for k, r in all_results.items()
            if isinstance(k, tuple) and len(k) == 2 and k[0] == dataset}