from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API

nas_api = API('NAS-Bench-201-v1_0-e61699.pth')
hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
for idx in range(3):
    for dataset in ["cifar10"]: #"cifar100", "ImageNet16-120"]:
        HW_metrics = hw_api.query_by_index(idx, dataset)
        netconfig = hw_api.get_net_config(idx, dataset)
        nasindex = nas_api.archstr2index[netconfig['arch_str']]
        if 'cifar10' == dataset:
            metrics = nas_api.query_by_index(idx, 'cifar10-valid')

        print("The HW_metrics (type: {}) for No.{} @ {} under NAS-Bench-201: {}".format(type(HW_metrics),   idx,
                                                                               dataset,
                                                                               HW_metrics))
for idx in range(3):
    for dataset in ["cifar10-valid"]:
        print("helo")

