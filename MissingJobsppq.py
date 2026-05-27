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
    p.add_argument('--modelpath',
                   type=str,
                   help='file to store models')
    p.add_argument('--weightpath',
                   type=str,
                   help='folder where weights are stored')
    p.add_argument('--resultpath',
                   type=str,
                   help='resultfile for accuracy')
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
model_path = args.modelpath
resultpath = args.resultpath
weightpath = args.weightpath

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])
#print(f"read data {resultpath}/data")
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

with open(file, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        idx = int(line.split()[0])

        for dataset in ["cifar10"]:
            init_start = time.process_time()
            HW_metrics = hw_api.query_by_index(idx, dataset)
            netconfig = hw_api.get_net_config(idx, dataset)
            print(f"read {weightpath}/{idx:06d}.pickle.pbz2")
            weights_path = f'{weightpath}/{idx:06d}.pickle.pbz2'  # or .pkl
            try:
                with bz2.BZ2File(weights_path, "rb") as f:
                    data = pickle.load(f)
            except FileNotFoundError:
                with open("/home/n/n_herr03/HW-NAS-Visualization/doesnotwork.txt", "a") as f:
                    f.write(f"{idx}\n")
                continue
            validkey = []
            for key in data.keys():
                if key in data: validkey.append(key)

            if not validkey:
                print("No valid data found")
                exit()

            key = max(validkey)
            for innerkey in data[key]["all_results"]:
                if isinstance(innerkey[0], str) and (innerkey[0] == 'cifar10'):
                    _, checkseed = innerkey
                else:
                    with open("/home/n/n_herr03/HW-NAS-Visualization/noseed.txt", "a") as f:
                        f.write(f"{idx}\n")


                ourdict = data[key]["all_results"][('cifar10', checkseed)]["net_state_dict"]
                print(ourdict)
                network = get_cell_based_tiny_net(netconfig)
                network.load_state_dict(ourdict)
                x = torch.rand([1, 3, 32, 32], dtype=torch.float32)
                network.eval()
                init_end = time.process_time()
                acc = evaluate_top1(network, calib_loader)
                transform_start = time.process_time()
                quant_ppq_graph = espdl_quantize_torch(network, f"{model_path}/espdl/model{idx}_{seed}.espdl",
                                                    collate_fn=collate_x_only, calib_dataloader=calib_loader, calib_steps=32,
                                                    error_report=False, verbose=0,
                                                    input_shape=[batchsize, 3, 32, 32])  # setting=quant_setting)
                executor = TorchExecutor(quant_ppq_graph, device='cpu')
                dataset = calib_loader.dataset
                transform_end = time.process_time()
                accqu = evaluate_top1(executor, calib_loader)
                if not os.path.exists(f'{resultpath}/result.csv'):
                    with open('result.csv', 'a', encoding='utf-8') as f:
                        f.write("idx,seed,dataset,recorded_test_acc,quant_test_acc\n")
                line = f"{idx},{checkseed },cifar10,{acc * 100:.2f},{accqu * 100:.2f}\n"
                print(line)
                with open(f'{resultpath}/result.csv', 'a', encoding='utf-8') as f:
                    f.write(line)