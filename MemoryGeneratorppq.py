import bz2
import onnx
import os
import pickle
import time
import torch
import torchvision
import onnxruntime as ort
import pandas as pd
import torchvision.transforms as transforms
from esp_ppq import TorchExecutor, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader
import argparse
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models

# nas_api = API('NAS-Bench-201-v1_0-e61699.pth', )


target = "esp32s3"

quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.equalization = True
# quant_setting.equalization_setting.iterations = 5
# quant_setting.equalization_setting.value_threshold = 0.5
# quant_setting.equalization_setting.opt_level = 2
def build_parser():
    p = argparse.ArgumentParser(description="Example: accept a list of integers")
    p.add_argument('-n', '--nums',
                   nargs='+',            # one or more; use '*' to allow zero
                   type=int,
                   metavar='N',
                   help='list of integers (e.g. --nums 1 2 3)')
    return p
def parse_args():
    p = build_parser()
    args = p.parse_args()

    nums = args.nums if args.nums is not None else []
    if args.nums is not None:
        nums = args.nums
    args.nums_parsed = nums
    return args.nums_parsed
def evaluate_top1(executor, loader):
    correct = 0
    total = 0

    for images, labels in loader:
        # run quantized graph
        out = executor(images)   # sometimes executor(*[images]) is needed

        # TorchExecutor may return list/tuple
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out

        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / total

def convert_tflite_to_header(tflite_content, output_header_path, float16=False):
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in
                 range(0, len(tflite_content), 12)]

    hex_array = ',\n  '.join(hex_lines)

    with open(output_header_path, 'w') as header_file:
        if float16:
            header_file.write('alignas(16) const unsigned char model[] = {\n  ')
        else:
            header_file.write('const unsigned char model[] = {\n  ')
        header_file.write(f'{hex_array}\n')
        header_file.write('};\n\n')


hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
collectedhwnas = pd.read_csv("all_hwnas.csv")
all_data = []

if not os.path.exists("models/torch"):
    os.makedirs("models/torch")
if not os.path.exists("models/onnx"):
    os.makedirs("models/onnx")
if not os.path.exists("models/tf"):
    os.makedirs("models/tf")
if not os.path.exists("models/espdl"):
    os.makedirs("models/espdl")

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batchsize = 1
calib_loader = DataLoader(
    testset, batch_size=batchsize, shuffle=False, drop_last=True
)


def collate_x_only(input):
    return input[0]

recordedacc = pd.read_csv("all_hwnas.csv")

idxs = parse_args()
for idx in idxs:
    for dataset in ["cifar10"]:
        init_start = time.process_time()
        HW_metrics = hw_api.query_by_index(idx, dataset)
        netconfig = hw_api.get_net_config(idx, dataset)
        weights_path = f'./data/NATS-tss-v1_0-3ffb9-full/{idx:06d}.pickle.pbz2'  # or .pkl
        with bz2.BZ2File(weights_path, "rb") as f:
            data = pickle.load(f)
        validkey = []
        seed = 999
        for key in data.keys():
            if key in data: validkey.append(key)

        if not validkey:
            print("No valid data found")
            exit()

        key = max(validkey)
        for innerkey in data[key]["all_results"]:
            if isinstance(innerkey[0], str) and (innerkey[0] == 'cifar10'):
                _, seed = innerkey
            else:
                continue

            ourdict = data[key]["all_results"][('cifar10', seed)]["net_state_dict"]
            network = get_cell_based_tiny_net(netconfig)
            network.load_state_dict(ourdict)
            x = torch.rand([1, 3, 32, 32], dtype=torch.float32)
            network.eval()
            init_end = time.process_time()
            acc = evaluate_top1(network, calib_loader)
            transform_start = time.process_time()
            mask = (
                    (recordedacc['seed'] == seed) &
                    (recordedacc['arch_index'] == idx) &
                    (recordedacc['dataset'] == 'cifar10')
            )
            shouldbe = recordedacc.loc[mask, 'test_acc']
            torch.onnx.export(network.eval(), x, f"models/onnx/model{idx}_{seed}.onnx", opset_version=18, verbose=0)
            onnxmodel = onnx.load(f"models/onnx/model{idx}_{seed}.onnx")
            onnx.checker.check_model(onnxmodel)
            sess = ort.InferenceSession(f"models/onnx/model{idx}_{seed}.onnx", providers=["CPUExecutionProvider"])
            quant_ppq_graph = espdl_quantize_onnx(f"models/onnx/model{idx}_{seed}.onnx", f"models/espdl/model{idx}_{seed}.espdl",
                                                collate_fn=collate_x_only, calib_dataloader=calib_loader, calib_steps=32,
                                                error_report=False, verbose=0,
                                                input_shape=[batchsize, 3, 32, 32])  # setting=quant_setting)
            executor = TorchExecutor(quant_ppq_graph, device='cpu')
            dataset = calib_loader.dataset
            transform_end = time.process_time()
            accqu = evaluate_top1(executor, calib_loader)
            if not os.path.exists('result.csv'):
                with open('result.csv', 'a', encoding='utf-8') as f:
                    f.write("idx,seed,dataset,test_acc,recorded_test_acc,quant_test_acc,rec_memory,rec_flops\n")
            line = f"{idx},{seed},cifar10,{shouldbe.iloc[0]},{acc * 100:.2f},{accqu * 100:.2f},TODO,TODO\n"
            with open('result.csv', 'a', encoding='utf-8') as f:
                f.write(line)