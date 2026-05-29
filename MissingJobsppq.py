import bz2
import os
import pickle
import time
import torch
import torchvision
import torchvision.transforms as transforms
from esp_ppq import TorchExecutor, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch
from torch.utils.data import DataLoader
import argparse
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models

target = "esp32s3"

quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.equalization = True
# quant_setting.equalization_setting.iterations = 5
# quant_setting.equalization_setting.value_threshold = 0.5
# quant_setting.equalization_setting.opt_level = 2
def build_parser():
    p = argparse.ArgumentParser(description="Example: accept a list of integers")
    p.add_argument('-file',
                   type=str,
                   help='file that has combination as lines')
    return p

def parse_args():
    p = build_parser()
    args = p.parse_args()
    return args

def evaluate_top1(executor, loader):
    correct = 0
    total = 0

    for images, labels in loader:
        # run quantized graph
        out = executor(images)   # sometimes executor(*[images]) is needed

        # TorchExecutor may return list/tuple
        if isinstance(out, (list, tuple)):
            logits = out[1]
        else:
            logits = out

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()
    #print(f"labeled {total} correct: {correct} acc {100 * correct / total}")
    return correct / total



hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
all_data = []
args = parse_args()
file = args.file
model_path = "/scratch/tmp/n_herr03/NATS_Benchmark/models/espdl"
resultpath = "/scratch/tmp/n_herr03/hwnas/result"
weightpath = "/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full"

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])
testset = torchvision.datasets.CIFAR10(root=f'{resultpath}/data', train=False, download=True, transform=transform)

batchsize = 1
calib_loader = DataLoader(
    testset, batch_size=batchsize, shuffle=False, drop_last=True
)

def collate_x_only(input):
    return input[0]

if not os.path.exists(f"{model_path}/onnx"):
    os.makedirs(f"{model_path}/onnx")
if not os.path.exists(f"{model_path}/espdl"):
    os.makedirs(f"{model_path}/espdl")
counter = 0
with open(file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        idx = int(line.split()[0])
        print(counter)
        counter = counter+1
        for dataset in ["cifar10"]:
            init_start = time.process_time()
            HW_metrics = hw_api.query_by_index(idx, dataset)
            netconfig = hw_api.get_net_config(idx, dataset)
            #print(f"read {weightpath}/{idx:06d}.pickle.pbz2")
            
            weights_path = f'{weightpath}/{idx:06d}.pickle.pbz2'  # or .pkl
            try:
                with bz2.BZ2File(weights_path, "rb") as f:
                    data = pickle.load(f)
            except FileNotFoundError:
                with open("/home/n/n_herr03/HW-NAS-Visualization/noweight.txt", "a") as f:
                    f.write(f"{idx}\n")
                continue
            validkey = []
            for key in data.keys():
                if key in data: validkey.append(key)

            if not validkey:
                with open("/home/n/n_herr03/HW-NAS-Visualization/novalidkey.txt", "a") as f:
                    f.write(f"{idx}\n")
                continue

            key = max(validkey)
            for innerkey in data[key]["all_results"]:
                if isinstance(innerkey[0], str) and (innerkey[0] == 'cifar10'):
                    _, checkseed = innerkey
                else:
                    with open("/home/n/n_herr03/HW-NAS-Visualization/noseed.txt", "a") as f:
                        f.write(f"{idx}\n")
                    continue
                with open(f'{resultpath}/shouldwork.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{idx},{innerkey}\n")
