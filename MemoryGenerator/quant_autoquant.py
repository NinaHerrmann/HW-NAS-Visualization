import copy

from esp_ppq.api import espdl_auto_quantize_onnx
from esp_ppq.autoquant.interface import AutoQuantSearchSetting

from data import collate_x_only
from evaluation import make_autoquant_evaluate_fn


def make_fast_autoquant_setting(run_dir):
    s = AutoQuantSearchSetting(search_mode="exhaustive", num_of_candidates=1, run_dir=run_dir)
    strat, params = copy.deepcopy(s.strategy_space), copy.deepcopy(s.param_space)
    strat.update({
        "num_of_bits": [True], "calib_algorithm": [True],
        "weight_equalization": [True, False], "bias_correction": [True, False],
        "horizontal_layer_split": [False], "mixed_precision": [False],
        "tqt": [False], "blockwise_reconstruction": [False], "fusion_alignment": [True],
    })
    params["num_of_bits"] = {"value": [8]}
    params["calib_algorithm"] = {"method": ["percentile", "mse"]}
    params["weight_equalization"].update({"iterations": [10], "value_threshold": [0.5], "opt_level": [2]})
    params["bias_correction"].update({"block_size": [4], "steps": [32], "topk": [-1]})
    params["fusion_alignment"].update({
        "elementwise_to": ["Align to Large"], "concat_to": ["Align to Output"],
        "avgpooling_to": ["Align to Output"], "resize_to": ["Align to Output"],
        "force_overlap": [False],
    })
    s.strategy_space, s.param_space = strat, params
    s.top_strategy, s.early_stop_patience = 2, 1
    s.score_direction, s.resume = "maximize", True
    return s


def run_autoquant(onnx_file, espdl_file, loaders, target, run_dir):
    """Returns (search_acc, test_acc) of the best candidate."""
    track = {"eval_acc": -1.0, "test_acc": float("nan")}
    espdl_auto_quantize_onnx(
        onnx_import_file=onnx_file,
        espdl_export_file=espdl_file,
        calib_dataloader=loaders.calib,
        calib_steps=len(loaders.calib),
        input_shape=loaders.input_shape[1:],
        evaluate_fn=make_autoquant_evaluate_fn(loaders.search, loaders.test, track),
        target=target,
        collate_fn=collate_x_only,
        setting=make_fast_autoquant_setting(run_dir),
        device="cpu",
        verbose=0,
    )
    return track["eval_acc"], track["test_acc"]