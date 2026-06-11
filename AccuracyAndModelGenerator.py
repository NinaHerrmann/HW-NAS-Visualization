import bz2
import os
import pickle
import time
import torch
import torchvision
import torchvision.transforms as transforms
from esp_ppq import TorchExecutor, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import pandas as pd
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
    p.add_argument('--file',
                   type=str,
                   help='file with indexes')
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

    indxdf = pd.read_csv(args.file, header=None)
    nums = indxdf[0].tolist()
    if nums is None:
        exit("nums == None")

    args.nums_parsed = nums
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

def load_imagenet16(root, split='train', transform=None):
    """
    root  – folder that contains the *.pkl files (train_data_batch_*, val_data)
    split – 'train' or 'val'
    transform – optional torchvision transform applied on‑the‑fly
    """
    # ---------- 1) read the pickle files ----------
    if split == 'train':
        # there are 9 training batches
        data_lst, label_lst = [], []
        for i in range(1, 10):
            p = os.path.join(root, f'train_data_batch_{i}')
            with open(p, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data_lst.append(batch['data'])      # shape (N, 3072)
                label_lst.append(batch['labels'])   # list of ints
        data  = np.concatenate(data_lst, axis=0)   # (15625, 3072)
        label = np.concatenate(label_lst, axis=0) # (15625,)
    else:  # validation
        p = os.path.join(root, 'val_data')
        with open(p, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data  = batch['data']    # (2500, 3072)
            label = np.array(batch['labels'])

    # ---------- 2) reshape to (N, C, H, W) ----------
    # each image is stored row‑major as R‑G‑B channels concatenated
    N = data.shape[0]
    data = data.reshape(N, 3, 16, 16).astype(np.float32) / 255.0   # [0,1] float

    # ---------- 3) torch tensors ----------
    imgs   = torch.from_numpy(data)                # (N,3,16,16)
    targets = torch.from_numpy(label).long()      # (N,)

    # ---------- 4) optional torchvision transform ----------
    if transform is not None:
        # wrap in a tiny Dataset that applies the transform per‑sample
        class ImgTransformDataset(torch.utils.data.Dataset):
            def __len__(self):   return N
            def __getitem__(self, idx):
                img = imgs[idx]
                img = transform(img)          # expects PIL or Tensor
                return img, targets[idx]
        dataset = ImgTransformDataset()
    else:
        dataset = TensorDataset(imgs, targets)

    return dataset

hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
all_data = []
args = parse_args()
idxs = args.nums_parsed
model_path = args.modelpath
resultpath = args.resultpath
weightpath = args.weightpath
string_dataset = 'ImageNet16-120'
if (string_dataset == "cifar 10"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    testset = torchvision.datasets.CIFAR10(root=f'{resultpath}/data', train=False, download=True, transform=transform)
if (string_dataset == "ImageNet16-120"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    testset = load_imagenet16(root=f'{resultpath}/data/ImageNet16', split='val', transform=transform)

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

for idx in idxs:
    for wip_dataset in [string_dataset]:
        init_start = time.process_time()
        HW_metrics = hw_api.query_by_index(idx, wip_dataset)
        netconfig = hw_api.get_net_config(idx, wip_dataset)
        print(f"read {weightpath}/{idx:06d}.pickle.pbz2")
        weights_path = f'{weightpath}/{idx:06d}.pickle.pbz2'  # or .pkl
        if not os.path.exists(weights_path):
            continue
        with bz2.BZ2File(weights_path, "rb") as f:
            data = pickle.load(f)
        validkey = []
        for key in data.keys():
            if key in data: validkey.append(key)

        if not validkey:
            print("No valid data found")
            exit()

        key = max(validkey)
        for innerkey in data[key]["all_results"]:
            if isinstance(innerkey[0], str) and (innerkey[0] == wip_dataset):
                _, seed = innerkey
            else:
                continue


            ourdict = data[key]["all_results"][(wip_dataset, seed)]["net_state_dict"]
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
            line = f"{idx},{seed},{wip_dataset},{acc * 100:.2f},{accqu * 100:.2f}\n"
            print(line)
            with open(f'{resultpath}/result.csv', 'a', encoding='utf-8') as f:
                f.write(line)
