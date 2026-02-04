from hw_nas_bench_api import HWNASBenchAPI as HWAPI
from nas_201_api import NASBench201API as API
import pandas as pd
from xautodl.models import get_cell_based_tiny_net  # this module is in AutoDL-Projects/lib/models
import numpy as np
import torch
import onnx
import tensorflow as tf
#nas_api = API('NAS-Bench-201-v1_1-096897.pth')

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

for idx in range(10):
    for dataset in ["cifar10"]:
        try:
            HW_metrics = hw_api.query_by_index(idx, dataset)
            netconfig = hw_api.get_net_config(idx, dataset)
            #nasindex = nas_api.archstr2index[netconfig['arch_str']]
            all_archs = netconfig["arch_str"].split("|")
            split_arch = [arch.split("~")[0] for arch in all_archs if arch != "" and arch != "+"]
            network = get_cell_based_tiny_net(netconfig)  # create the network from configurration
            #x = np.random.rand(8, 32, 32, 3).astype('float32')
            x = torch.randn(8, 3, 32, 32)
            torch.onnx.export(network.eval(), x, f"model{idx}.onnx", opset_version=25)
            subprocess.run([f"onnx2tf -i model{idx}.onnx -o saved_model{idx}"], shell=True)  # doesn't capture output
            with open(f"saved_model{idx}/model{idx}_float16.tflite", 'rb') as tflite_file:
                tflite_content = tflite_file.read()

            out_file = f'tflitetemplate/model{idx}.h'
            convert_tflite_to_header(tflite_content, out_file, True)

            print(network)  # show the structure of this architecture
        except FileNotFoundError:
            with open("MemoryLog", 'a')  as logfile:
                logfile.write(f"{idx}:FileNotFoundError\n")
        except Exception as e:
            with open("MemoryLog", 'a')  as logfile:
                logfile.write(f"{idx}:e\n")



