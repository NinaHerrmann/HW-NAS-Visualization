#!/usr/bin/env python3
import os
import time
import traceback

import torch
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from xautodl.models import get_cell_based_tiny_net

from config import TARGET, parse_args
from data import build_loaders
from evaluation import graph_top1, top1
from export import export_onnx
from quant_autoquant import run_autoquant
from quant_baseline import run_int8, run_mixed
from results import ResultWriter
from weights import load_state_dicts


def process_seed(idx, seed, state_dict, netconfig, loaders, args, writer):
    todo = [m for m in args.methods if not writer.is_done(idx, seed, m)]
    if not todo:
        return
    name = f"model{idx}_{seed}"
    path = lambda sub, ext: f"{args.modelpath}/{sub}/{name}{ext}"

    network = get_cell_based_tiny_net(netconfig)
    network.load_state_dict(state_dict)
    network.eval()

    def log(method, t0, eval_acc, test_acc, info=""):
        writer.write(idx, seed, args.dataset, method, eval_acc, test_acc, time.time() - t0, info)

    # ---- fp32
    if "fp32" in todo:
        t0 = time.time()
        log("fp32", t0, top1(network, loaders.search), top1(network, loaders.test))

    if not ({"int8", "mixed", "autoquant"} & set(todo)):
        return
    onnx_file = path("onnx", ".onnx")
    export_onnx(network, onnx_file, loaders.input_shape)

    # ---- int8 (also needed as the starting point for 'mixed')
    int8_graph = None
    if "int8" in todo or "mixed" in todo:
        t0 = time.time()
        int8_graph = run_int8(onnx_file, path("espdl", "_int8.espdl"), loaders, TARGET,
                              export="int8" in todo)
        if "int8" in todo:
            log("int8", t0, graph_top1(int8_graph, loaders.search),
                graph_top1(int8_graph, loaders.test))

    # ---- int8 + int16 for the worst layers
    if "mixed" in todo:
        t0 = time.time()
        graph, worst, _ = run_mixed(onnx_file, path("espdl", "_mixed.espdl"),
                                    loaders, TARGET, int8_graph, args.topk)
        log("mixed", t0, graph_top1(graph, loaders.search),
            graph_top1(graph, loaders.test), info=";".join(worst))

    # ---- AutoQuant
    if "autoquant" in todo:
        t0 = time.time()
        eval_acc, test_acc = run_autoquant(onnx_file, path("espdl", "_autoquant.espdl"),
                                           loaders, TARGET, path("autoquant", ""))
        log("autoquant", t0, eval_acc, test_acc)


def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    for sub in ("onnx", "espdl", "autoquant"):
        os.makedirs(f"{args.modelpath}/{sub}", exist_ok=True)

    writer = ResultWriter(args.resultpath)
    hw_api = HWAPI(args.hwapi, search_space="nasbench201")
    loaders = build_loaders(args)

    for idx in args.indices:
        state_dicts = load_state_dicts(args.weightpath, idx, args.dataset)
        if not state_dicts:
            print(f"[skip] no weights for {idx}")
            continue
        netconfig = hw_api.get_net_config(idx, args.dataset)
        for seed, sd in state_dicts.items():
            try:
                process_seed(idx, seed, sd, netconfig, loaders, args, writer)
            except Exception:
                print(f"[ERROR] idx={idx} seed={seed}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()