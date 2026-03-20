#!/bin/bash
ESP_EXPORT="/scratch/tmp/n_herr03/esp/esp-idf/export.sh"
EXAMPLE="espressif/esp-dl=3.1.3:how_to_run_model"
PROJECT_DIR="./how_to_run_model"
PROGRAM_PATH="./how_to_run_model/main/models/s3"

if [ ! -d "$PROJECT_DIR" ]; then
    if [ -f "$ESP_EXPORT" ]; then
        # shellcheck disable=SC1090
        . "$ESP_EXPORT"
    else
        echo "Error: esp-idf export script not found at $ESP_EXPORT" >&2
        exit 1
    fi
    idf.py create-project-from-example "$EXAMPLE"
fi

"/scratch/tmp/n_herr03/esp/esp-idf/install.sh"
. $ESP_EXPORT

for f in /scratch/tmp/n_herr03/NATS_Benchmark/models/testmodels/*.espdl; do
    [ -e "$f" ] || continue   # skip literal pattern if no match

    if [[ $f =~ model([0-9]+)_([0-9]+)\.espdl ]]; then
        idx="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"
        echo "First number: $idx"
        echo "Second number: $seed"
    fi
    echo "Processing: /scratch/tmp/n_herr03/NATS_Benchmark/models/espdl/model${idx}_${seed}.espdl"

    espmodelpath="/scratch/tmp/n_herr03/NATS_Benchmark/models/testmodels/model${idx}_${seed}.espdl"
    dest="${PROGRAM_PATH}/model.espdl"
    echo "Copying $espmodelpath -> $dest"
    rm -f "$dest"
    cp "$espmodelpath" "$dest"
    # shellcheck disable=SC2164
    cd "how_to_run_model"
    idf.py set-target esp32s3
    idf.py build
    /home/n/n_herr03/.espressif/python_env/idf6.1_py3.10_env/bin/python "/scratch/tmp/n_herr03/esp/esp-idf/tools/idf_size.py" "./build/model_in_flash_rodata.map" --format "json2" > ../output.json
    cd ".."
    pip install pandas
    python3 convert_to_csv.py "output.json" --output "memory_results.csv" --idx ${idx} --seed ${seed} --dataset "cifar10"
done
