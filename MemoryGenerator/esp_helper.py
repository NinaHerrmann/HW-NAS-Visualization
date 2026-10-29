from esp_ppq.autoquant.interface import AutoQuantSearchSetting
import copy
import itertools
import math

def make_fast_autoquant_setting(run_dir: str) -> AutoQuantSearchSetting:
    s = AutoQuantSearchSetting(
        search_mode="exhaustive",   # space is tiny -> deterministic, bounded
        num_of_candidates=1,
        run_dir=run_dir,
    )
    # deepcopy: never mutate (possibly shared) default dicts
    strat = copy.deepcopy(s.strategy_space)
    params = copy.deepcopy(s.param_space)

    strat.update({
        "num_of_bits":              [True],
        "calib_algorithm":          [True],
        "weight_equalization":      [True, False],   # cheap, often helps on CNNs
        "horizontal_layer_split":   [False],
        "bias_correction":          [True, False],   # cheap (32 steps)
        "mixed_precision":          [False],
        "tqt":                      [False],         # 500-2000 training steps -> off
        "fusion_alignment":         [True],
        "blockwise_reconstruction": [False],         # 1000 training steps -> off
    })

    params["num_of_bits"] = {"value": [8]}
    params["calib_algorithm"] = {"method": ["percentile", "mse"]}   # drop kl
    params["weight_equalization"].update({
        "iterations": [10], "value_threshold": [0.5], "opt_level": [2],
    })
    params["bias_correction"].update({
        "block_size": [4], "steps": [32], "topk": [-1],
    })
    # NATS-201 has no Concat / Resize -> these choices are irrelevant; fix them.
    params["fusion_alignment"].update({
        "elementwise_to": ["Align to Large"],
        "concat_to":      ["Align to Output"],
        "avgpooling_to":  ["Align to Output"],
        "resize_to":      ["Align to Output"],
        "force_overlap":  [False],
    })

    s.strategy_space = strat
    s.param_space = params
    s.top_strategy = 2
    s.early_stop_patience = 1
    s.score_direction = "maximize"
    s.resume = False
    return s
def estimate_space(s) -> int:
    """Upper bound of exhaustive candidates (structural knobs x pass combos)."""
    total = 0
    passes = list(s.strategy_space.keys())
    for combo in itertools.product(*[s.strategy_space[p] for p in passes]):
        n = 1
        for p, on in zip(passes, combo):
            if on:
                n *= math.prod(len(v) for v in s.param_space[p].values())
        total += n
    return total