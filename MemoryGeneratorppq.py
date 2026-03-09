from esp_ppq import QuantizationSettingFactory, TorchExecutor, BaseGraph
import onnxruntime as ort
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API
import pandas as pd
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models
#nas_api = API('NAS-Bench-201-v1_1-096897.pth')
import os
import numpy as np
import onnx
from esp_ppq.api import get_target_platform, espdl_quantize_torch, espdl_quantize_onnx
from torch.utils.data import DataLoader, TensorDataset
import torch
import torchvision
import torchvision.transforms as transforms

target="esp32s3"
num_of_bits=8
quant_setting = QuantizationSettingFactory.espdl_setting()
quant_setting.equalization = True
quant_setting.equalization_setting.iterations = 4
quant_setting.equalization_setting.value_threshold = .4
quant_setting.equalization_setting.opt_level = 2
quant_setting.equalization_setting.interested_layers = None

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
import subprocess

if not os.path.exists("models/torch"):
    os.makedirs("models/torch")
if not os.path.exists("models/onnx"):
    os.makedirs("models/onnx")
if not os.path.exists("models/tf"):
    os.makedirs("models/tf")
if not os.path.exists("models/espdl"):
    os.makedirs("models/espdl")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
batchsize = 1
calib_loader = DataLoader(
    testset, batch_size=batchsize, shuffle=False, drop_last=True
)
def collate_x_only(input):
    return input[0]

idx=4
for dataset in ["cifar10"]:
    HW_metrics = hw_api.query_by_index(idx, dataset)
    netconfig = hw_api.get_net_config(idx, dataset)
    network = get_cell_based_tiny_net(netconfig)  # create the network from configuration
    x = torch.rand([1, 3, 32, 32], dtype=torch.float32)
    network.eval()
    out_data = network(x)
    torch.onnx.export(network.eval(), x, f"models/onnx/model{idx}.onnx", opset_version=18)

    onnxmodel = onnx.load(f"models/onnx/model{idx}.onnx")
    onnx.checker.check_model(onnxmodel)
    sess = ort.InferenceSession(f"models/onnx/model{idx}.onnx", providers=["CPUExecutionProvider"])
    input0 = sess.get_inputs()[0]
    print("Input name:", input0.name, "shape:", input0.shape, "dtype:", input0.type)
    print("Outputs:", [o.name for o in sess.get_outputs()])

    # run prediction
    t = x.detach().cpu().numpy().astype(np.float32)
    outputs = sess.run(None, {input0.name: t})
    y = outputs[0]
    print("Output shape:", y.shape, "dtype:", y.dtype) #collate_fn=collate_x_only,
    quant_ppq_graph = espdl_quantize_onnx(f"models/onnx/model{idx}.onnx", f"models/espdl/model{idx}.espdl", collate_fn=collate_x_only, calib_dataloader=calib_loader, calib_steps=32, error_report=True, verbose=0, input_shape=[batchsize, 3, 32, 32])
    executor = TorchExecutor(quant_ppq_graph, device='cpu')
    print(testset.data[0])
    dataset = calib_loader.dataset
    #for data in iter(calib_loader):

    results = executor(next(iter(calib_loader))[0])
    print("Results:", results)

