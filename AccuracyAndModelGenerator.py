import bz2
import copy
import os
import pickle
import time
import torch
import torchvision
import torchvision.transforms as transforms
from esp_ppq import TorchExecutor, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, espdl_quantize_torch, espdl_auto_quantize_onnx, AutoQuantSearchSetting
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models
from xautodl.datasets import get_datasets
target = "esp32s3"

quant_setting = QuantizationSettingFactory.espdl_setting()

# quant_setting.equalization = True
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
    p.add_argument('--datapath',
                   type=str,
                   help='datapath for accuracy')
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
def load_imagenet16_120(root, split='val', num_classes=120):
    files = ([os.path.join(root, f'train_data_batch_{i}') for i in range(1, 11)]
             if split == 'train' else [os.path.join(root, 'val_data')])

    data_lst, label_lst = [], []
    for p in files:
        with open(p, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        lbl = np.array(batch['labels'])
        mask = (lbl >= 1) & (lbl <= num_classes)          # filter per file
        data_lst.append(batch['data'][mask])
        label_lst.append(lbl[mask] - 1)                   # 1-based -> 0-based
        del batch
    data, label = np.concatenate(data_lst), np.concatenate(label_lst)

    imgs = torch.from_numpy(data.reshape(-1, 3, 16, 16).astype(np.float32) / 255.0)
    mean = torch.tensor([122.68, 116.66, 104.01]).view(1, 3, 1, 1) / 255
    std  = torch.tensor([63.22, 61.26, 65.09]).view(1, 3, 1, 1) / 255
    imgs = (imgs - mean) / std
    return TensorDataset(imgs, torch.from_numpy(label).long())
def make_subset_loader(dataset, n, batch_size, seed=0):
    g = torch.Generator().manual_seed(seed)              # same subset for every model
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return DataLoader(Subset(dataset, idx), batch_size=batch_size,
                      shuffle=False, drop_last=True)

hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
all_data = []
args = parse_args()
idxs = args.nums_parsed
model_path = args.modelpath
resultpath = args.resultpath
weightpath = args.weightpath
datapath = args.datapath
string_dataset = 'ImageNet16-120'
batchsize = 1

if (string_dataset == "cifar 10"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    root = f'{datapath}/cifar10'

    train_data, valid_data, xshape, class_num = get_datasets('cifar10', root, cutout=0)
    train_noaug = copy.deepcopy(train_data)
    train_noaug.transform = valid_data.transform

    calib_loader = make_subset_loader(train_noaug, n=512, batch_size=batchsize, seed=0)
    eval_loader = make_subset_loader(train_noaug, n=1000, batch_size=batchsize, seed=1)
    test_loader = DataLoader(valid_data, batch_size=batchsize, shuffle=False, drop_last=True)
elif (string_dataset == "ImageNet16-120"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    root = f'{datapath}/ImageNet16'
    train_data, valid_data, xshape, class_num = get_datasets('ImageNet16-120', root, cutout=0)
    train_noaug = copy.deepcopy(train_data)
    train_noaug.transform = valid_data.transform

    calib_loader = make_subset_loader(train_noaug, n=512, batch_size=batchsize, seed=0)
    eval_loader = make_subset_loader(train_noaug, n=1000, batch_size=batchsize, seed=1)
    test_loader = DataLoader(valid_data, batch_size=batchsize, shuffle=False, drop_last=True)
else:
    print('dataset not supported')
    exit()


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
            network.eval()
            acc = evaluate_top1(network, test_loader)

            onnx_file = f"{model_path}/onnx/model{idx}_{seed}.onnx"
            espdl_file = f"{model_path}/espdl/model{idx}_{seed}.espdl"
            INPUT_SHAPE = [3, 16, 16]  # ImageNet16 is 16x16, not 32x32!
            EXPORT_SHAPE = [batchsize] + INPUT_SHAPE

            # 1) export torch -> onnx
            dummy = torch.randn(EXPORT_SHAPE)
            torch.onnx.export(
                network, dummy, onnx_file,
                opset_version=13,
                dynamo=False,
                input_names=["input"],
                output_names=["features", "logits"],  # tiny_net returns (features, logits)
                do_constant_folding=True,
            )

            # 2) auto quantization
            best = {"eval_acc": -1.0, "test_acc": None}


            def make_evaluate_fn(eval_loader, test_loader):
                def evaluate_fn(graph):
                    executor = TorchExecutor(graph, device="cpu")
                    acc = evaluate_top1(executor, eval_loader)

                    # new best candidate -> also measure the real test accuracy now
                    if acc > best["eval_acc"]:
                        best["eval_acc"] = acc
                        best["test_acc"] = evaluate_top1(executor, test_loader)

                    return acc, {"top1": acc}

                return evaluate_fn


            setting = AutoQuantSearchSetting()
            setting.score_direction = "maximize"  # accuracy: higher is better

            results = espdl_auto_quantize_onnx(
                onnx_import_file=onnx_file,
                espdl_export_file=espdl_file,
                calib_dataloader=calib_loader,
                calib_steps=32,
                input_shape=INPUT_SHAPE,  # [1, 3, 16, 16]
                evaluate_fn=make_evaluate_fn(eval_loader, test_loader),
                target=target,  # "esp32s3"
                collate_fn=collate_x_only,
                setting=setting,
                device="cpu",
                verbose=0,
            )

            accqu = best["test_acc"]

            # 3) write results (no `continue` anymore)
            result_file = f'{resultpath}/result.csv'
            if not os.path.exists(result_file):
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write("idx,seed,dataset,float_test_acc,quant_eval_acc,quant_test_acc\n")

            line = (f"{idx},{seed},{wip_dataset},{acc * 100:.2f},"
                    f"{best['eval_acc'] * 100:.2f},{accqu * 100:.2f}\n")
            print(line)
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(line)
