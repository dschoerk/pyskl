#!/usr/bin/env bash
# Download pre-processed NTU RGB+D skeleton pkl files from OpenMMLab CDN.
# Run from the repo root: bash tools/data/download_nturgbd.sh

set -euo pipefail

DATA_DIR="data/nturgbd"
mkdir -p "$DATA_DIR"

BASE="https://download.openmmlab.com/mmaction/pyskl/data/nturgbd"

download() {
    local url="$1"
    local dest="$DATA_DIR/$(basename "$url")"
    if [ -f "$dest" ]; then
        echo "Already exists, skipping: $dest"
    else
        echo "Downloading $(basename "$url") ..."
        wget -q --show-progress -O "$dest" "$url"
    fi
}

# NTU60
download "$BASE/ntu60_hrnet.pkl"    # 2D HRNet skeletons (~1.1 GB)
download "$BASE/ntu60_3danno.pkl"   # 3D skeletons    (~250 MB)

# NTU120 (comment out if not needed)
# download "$BASE/ntu120_hrnet.pkl"
# download "$BASE/ntu120_3danno.pkl"

echo "Done. Files are in $DATA_DIR/"
