#!/bin/bash

TMPDIR="~/tmp"
EXAMPLE="espressif/esp-dl=3.3.14:how_to_run_model"
echo "Arg1: $1"
echo "Arg2: $2"
PROJECT_DIR="$TMPDIR/how_to_run_model${2}"
PROGRAM_PATH="$TMPDIR/how_to_run_model${2}/main/models/s3"

if [ ! -d "$PROJECT_DIR" ]; then
    if [ -f "$ESP_EXPORT" ]; then
        # shellcheck disable=SC1090
        . "$ESP_EXPORT"
    else
        echo "Error: esp-idf export script not found at $ESP_EXPORT" >&2
        exit 1
    fi
    # idf.py create-project-from-example "$EXAMPLE" ${PROJECT_DIR}
    cp -r /home/nherrmann/tmp/how_to_run_model ${PROJECT_DIR}
fi


for f in /scratch/tmp/n_herr03/NATS_Benchmark/models/espdl/model${1}_*.espdl; do
    [ -e "$f" ] || continue
    echo "File: $f"
    # capture the two numeric parts (model index and the trailing number)
    if [[ $f =~ model([0-9]+)_([0-9]+)\.espdl$ ]]; then
        idx="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"  # the number after the underscore
        echo "First number (idx): $idx"
        echo "Second number (seed): $seed"
    else
        echo "No numeric suffix found in $f"
    fi
    echo "Processing: ${3}/espdl/model${idx}_${seed}.espdl"

    espmodelpath="${3}/espdl/model${idx}_${seed}.espdl"
    dest="${PROGRAM_PATH}/model.espdl"
    echo "Copying $espmodelpath -> $dest"
    rm -f "$dest"
    cp -f "$espmodelpath" "$dest"
    # shellcheck disable=SC2164
    cd $PROJECT_DIR
    idf.py clean
    #rm -rf build
    idf.py set-target esp32s3
    idf.py build || { echo "$idx $seed \n " > failed.txt; exit 1; }
    /home/nherrmann/.espressif/tools/python/v6.1/venv/bin/python "/home/nherrmann/.espressif/tools/python/v6.1/esp-idf/tools/idf_size.py" "./build/model_in_flash_rodata.map" --format "json" > output.json
    cd ~/MemoryGenerator
    python3 convert_to_csv.py "${PROJECT_DIR}/output.json" --output "memory_results.csv" --idx ${idx} --seed ${seed} --dataset "imagenet"
done

rm -rf "$TMPDIR/how_to_run_model${2}"