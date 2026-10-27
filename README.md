## How to use - Generate Memory Numbers yourself

1. Download the pth file for NAS 201  ([.pth file](https://drive.usercontent.google.com/download?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_&export=download&authuser=0))
2. xautodl
   - `git clone git@github.com:D-X-Y/AutoDL-Projects.git AutoDL-Projects`
   - `mv AutoDL-Projects/xautodl ./`
   - `rm -rf AutoDL-Projects`
3. install the esp-idf toolkit [documentation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/linux-setup.html#debian-based-linux-installation-via-apt)
4. Generate the combinations you want to process as chunk files. E.g. generate 32 chunk files with each a equal share of the 15625 model architectures, randomly shuffeled (so hopefully every job takes roughly the same time).
```bash
seq 0 15624 | shuf | split -n r/32 -d -a 2 - chunk_
```
5. all .sh files are not generic! - adapt to your needs!
6. clone the default project ` idf.py create-project-from-example "espressif/esp-dl=3.3.14:how_to_run_model"` in a temporary directory so you can copy paste it