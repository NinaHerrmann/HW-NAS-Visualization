"""
Baselines:
  int8  : plain 8-bit ESP-DL quantization
  mixed : 8-bit, but the top-k layers with the highest quantization error run in 16 bit

Procedure for 'mixed':
  1. quantize to int8                          (reuses the int8 run)
  2. layer-wise error analysis (SNR per layer, each layer quantized in isolation)
  3. promote the worst k Conv/Gemm layers to int16 via the dispatching table
  4. quantize again and export
"""
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.quantization.analyse import layerwise_error_analyse

from data import collate_x_only

PROMOTABLE_OPS = {"Conv", "Gemm", "MatMul"}


def _quantize(onnx_file, espdl_file, loaders, target, setting, export=True):
    return espdl_quantize_onnx(
        onnx_import_file=onnx_file,
        espdl_export_file=espdl_file,
        calib_dataloader=loaders.calib,
        calib_steps=len(loaders.calib),
        input_shape=loaders.input_shape,
        target=target,
        num_of_bits=8,
        collate_fn=collate_x_only,
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=not export,
        export_test_values=False,
        verbose=0,
    )


def layer_errors(graph, loaders, steps=4):
    """{op_name: snr_error} for computing layers, sorted worst first.
    PPQ's 'snr' is noise/signal power, so higher means worse."""
    raw = layerwise_error_analyse(
        graph=graph, running_device="cpu", dataloader=loaders.calib,
        collate_fn=collate_x_only, method="snr", steps=steps, verbose=False,
    )
    errors = {}
    for name, err in raw.items():
        op = graph.operations.get(name)
        if op is not None and op.type in PROMOTABLE_OPS:
            errors[name] = float(err[0] if isinstance(err, (tuple, list)) else err)
    return dict(sorted(errors.items(), key=lambda kv: kv[1], reverse=True))


def run_int8(onnx_file, espdl_file, loaders, target, export=True):
    setting = QuantizationSettingFactory.espdl_setting()
    return _quantize(onnx_file, espdl_file, loaders, target, setting, export)


def run_mixed(onnx_file, espdl_file, loaders, target, int8_graph, topk):
    """Returns (graph, promoted_layer_names, all_layer_errors)."""
    errors = layer_errors(int8_graph, loaders)
    worst = list(errors)[:topk]

    setting = QuantizationSettingFactory.espdl_setting()   # fresh setting every time
    for name in worst:
        setting.dispatching_table.append(name, get_target_platform(target, 16))

    graph = _quantize(onnx_file, espdl_file, loaders, target, setting)
    return graph, worst, errors