from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import moocore

nas_api = API('NAS-Bench-201-v1_1-096897.pth')
hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")

all_data = []

for idx in range(5**6):

    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        HW_metrics = hw_api.query_by_index(idx, dataset)
        netconfig = hw_api.get_net_config(idx, dataset)
        nasindex = nas_api.archstr2index[netconfig['arch_str']]

        if dataset == "cifar10":
            max_epochs = 12
        else:
            max_epochs = 200

        metrics = nas_api.query_by_index(idx, dataset, hp = str(max_epochs))
        metrics_seed = metrics[next(iter(metrics))]
        all_metrics = HW_metrics
        all_metrics["flop"] = metrics_seed.flop
        all_metrics["test_acc"] = metrics_seed.eval_acc1es[f"ori-test@{max_epochs-1}"]
        all_metrics["dataset"] = dataset
        all_archs = metrics_seed.arch_config["arch_str"].split("|")

        split_arch = [arch.split("~")[0] for arch in all_archs if arch != "" and arch != "+"]

        for i, arch in enumerate(split_arch):
            all_metrics[f"arch_{i}"] = arch

        all_data.append(all_metrics)

df = pd.DataFrame(all_data)
