from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API
import pandas as pd
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models
import numpy as np
import torch
import onnx
import tensorflow as tf
#nas_api = API('NAS-Bench-201-v1_1-096897.pth')
import os
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

indxdf = pd.read_csv("./restjobs/rmdup/noweight.csv", header=None)
nums = indxdf.row[0].tolist()
if nums is None:
    exit("nums == None")

print(nums)
exit()

for idx in idxs:
    for dataset in ["cifar10"]:
        HW_metrics = hw_api.query_by_index(idx, dataset)
        netconfig = hw_api.get_net_config(idx, dataset)
        network = get_cell_based_tiny_net(netconfig)  # create the network from configurration
        x = torch.rand(1, 3, 32, 32)
        #torch.save(network.eval(), f"models/torch/torch{idx}.pth")
        torch.onnx.export(network.eval(), x, f"models/onnx/model{idx}.onnx", opset_version=25)
        subprocess.run([f"onnx2tf -i models/onnx/model{idx}.onnx -o models/tf/model{idx}"], shell=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(f"models/tf/model{idx}")
        tflite_model = converter.convert()
        with open(f"models/tf/model{idx}/model{idx}_float32.tflite", 'rb') as tflite_file:
            tflite_content = tflite_file.read()

        out_file = f'tflitetemplate/model{idx}.h'
        convert_tflite_to_header(tflite_content, out_file, True)