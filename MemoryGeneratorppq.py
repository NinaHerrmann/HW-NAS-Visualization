from esp_ppq import QuantizationSettingFactory

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API
import pandas as pd
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models
#nas_api = API('NAS-Bench-201-v1_1-096897.pth')
import os
from esp_ppq.api import get_target_platform, espdl_quantize_torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torchvision
import torchvision.transforms as transforms

target="esp32s3"
num_of_bits=8
batch_size=32
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
def collate_x_only(batch):
    xs = [x for x, _ in batch]
    return torch.stack(xs, 0)

calib_loader = DataLoader(
testset, batch_size=8, shuffle=False, collate_fn=collate_x_only
)
for idx in range(5):
    for dataset in ["cifar10"]:
        HW_metrics = hw_api.query_by_index(idx, dataset)
        netconfig = hw_api.get_net_config(idx, dataset)
        network = get_cell_based_tiny_net(netconfig)  # create the network from configurration
        x = torch.rand(1, 3, 32, 32)
        quant_ppq_graph = espdl_quantize_torch(network.eval(), f"models/espdl/model{idx}.espdl", calib_dataloader=calib_loader, calib_steps=8, input_shape=[8, 3, 32, 32])
