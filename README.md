## How to use - Generate Memory Numbers yourself

1. Download the pth file for NAS 201 (Tested AI-Slop:)
   - Open https://developers.google.com/oauthplayground
   - Click the gear icon (top right) → make sure nothing unusual is set, then close it.
   - In Step 1, scroll to Drive API v3 and select only https://www.googleapis.com/auth/drive.readonly. Don't use drive.file, which only covers files created by the app.
   - Click Authorize APIs and log in again. You must re-authorize, because the old token keeps its old scope.
   - Click Exchange authorization code for tokens and copy the new access token.
```bash
TOKEN="[YOUR_TOKEN]"
FOLDER="17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB"
curl -s -G -H "Authorization: Bearer $TOKEN" \
  "https://www.googleapis.com/drive/v3/files" \
  --data-urlencode "q='${FOLDER}' in parents and trashed=false" \
  --data-urlencode "fields=files(id,name,size,mimeType)" \
  --data-urlencode "pageSize=1000" \
  --data-urlencode "supportsAllDrives=true" \
  --data-urlencode "includeItemsFromAllDrives=true" \
| jq -r '.files[] | select(.mimeType != "application/vnd.google-apps.folder") | "\(.id)\t\(.size)\t\(.name)"' \
| while IFS=$'\t' read -r id size name; do
    if [ -f "$name" ] && [ "$(stat -c %s "$name")" = "$size" ]; then
      echo "=== Skipping $name (complete) ==="
      continue
    fi
    echo "=== Downloading $name ==="
    curl -L -C - --fail -H "Authorization: Bearer $TOKEN" \
      "https://www.googleapis.com/drive/v3/files/${id}?alt=media&supportsAllDrives=true" \
      -o "$name"
  done
```
- this script will fail after 1 h as the token expires (or you have a crazy good internet connection). You can generate a new one and restart the script, it skips files that are already downloaded.
2. xautodl
   - `git clone git@github.com:D-X-Y/AutoDL-Projects.git AutoDL-Projects`
   - `mv AutoDL-Projects/xautodl ./`
   - `rm -rf AutoDL-Projects`
3. install the esp-idf toolkit [documentation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/linux-setup.html#debian-based-linux-installation-via-apt)
4. clone the default project ` idf.py create-project-from-example "espressif/esp-dl=3.3.14:how_to_run_model"` in a temporary directory so you can copy paste it
5. Generate the combinations you want to process as chunk files. E.g. generate 32 chunk files with each a equal share of the 15625 model architectures, randomly shuffeled (so hopefully every job takes roughly the same time).
```bash
seq 0 15624 | shuf | split -n r/32 -d -a 2 - chunk_
```
5. Unfortunatly, esp-ppq is merely python 3.10-3.12 - I downgraded to python 3.11 but this is not really a sustainable solution 
6. all .sh files are not generic! - adapt to your needs! (e.g. python version as mentioned above, paths and other things.)
