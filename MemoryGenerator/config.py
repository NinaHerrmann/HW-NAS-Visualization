import argparse
import pandas as pd

TARGET = "esp32s3"
DATA_ROOTS = {"cifar10": "cifar10", "ImageNet16-120": "ImageNet16"}
METHODS = ("fp32", "int8", "mixed", "autoquant")


def parse_args():
    p = argparse.ArgumentParser(description="NATS-Bench -> ESP-DL quantization")
    p.add_argument("--file", required=True, help="CSV file with architecture indices")
    p.add_argument("--modelpath", required=True, help="output folder for onnx/espdl/autoquant")
    p.add_argument("--weightpath", required=True, help="folder with *.pth or *.pickle.pbz2")
    p.add_argument("--resultpath", required=True, help="folder for result.csv")
    p.add_argument("--datapath", required=True, help="dataset root folder")
    p.add_argument("--dataset", default="ImageNet16-120", choices=list(DATA_ROOTS))
    p.add_argument("--hwapi", default="../HW-NAS-Bench-v1_0.pickle")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--topk", type=int, default=3, help="#layers promoted to int16 in 'mixed'")
    p.add_argument("--n-calib", type=int, default=1024)
    p.add_argument("--n-search", type=int, default=1000)
    p.add_argument("--calib-bs", type=int, default=128)
    p.add_argument("--eval-bs", type=int, default=256)
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = p.parse_args()
    args.indices = pd.read_csv(args.file, header=None)[0].astype(int).tolist()
    return args