#!/bin/bash
ESP_EXPORT="/Users/ninaherrmann/esp/esp-idf/export.sh"
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

"/Users/ninaherrmann/esp/esp-idf/install.sh"
. $ESP_EXPORT

idxs=(26)
for idx in "${idxs[@]}"; do
  espdlmodelpath="models/espdl/model${idx}.espdl"
  dest="${PROGRAM_PATH}/model.espdl"
  echo "Copying $espdlmodelpath -> $dest"
  rm -f "$dest"
  cp "$espdlmodelpath" "$dest"
  # shellcheck disable=SC2164
  cd "how_to_run_model"
  idf.py set-target esp32s3
  idf.py build
  /Users/ninaherrmann/.espressif/python_env/idf5.5_py3.9_env/bin/python "/Users/ninaherrmann/esp/v5.5.2/esp-idf/tools/idf_size.py" "./build/model_in_flash_rodata.map" > output.txt
  cd ".."
done