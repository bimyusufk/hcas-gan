#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_and_extract_camo.sh <zip_url> [output_dir]

Arguments:
  zip_url     Direct URL to camo.zip (HTTP/HTTPS or a pre-signed download URL)
  output_dir  Target folder to extract into (default: camo)

Examples:
  bash scripts/download_and_extract_camo.sh "https://example.com/camo.zip"
  bash scripts/download_and_extract_camo.sh "https://example.com/camo.zip" camo
EOF
}

if [[ ${1:-} == "" || ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

ZIP_URL="$1"
OUTPUT_DIR="${2:-camo}"
TMP_ZIP="/tmp/camo.zip"

cleanup() {
  rm -f "$TMP_ZIP"
}
trap cleanup EXIT

echo "[HCAS-GAN] Downloading camo.zip from: $ZIP_URL"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 --retry-delay 2 -o "$TMP_ZIP" "$ZIP_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_ZIP" "$ZIP_URL"
else
  echo "[HCAS-GAN] Error: curl or wget is required." >&2
  exit 1
fi

echo "[HCAS-GAN] Extracting to: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

if command -v unzip >/dev/null 2>&1; then
  unzip -o "$TMP_ZIP" -d "$OUTPUT_DIR"
else
  python3 - <<PY
from pathlib import Path
import zipfile

zip_path = Path(r"$TMP_ZIP")
out_dir = Path(r"$OUTPUT_DIR")
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(out_dir)
print(f"[HCAS-GAN] Extracted {len(zf.namelist())} entries to {out_dir}")
PY
fi

echo "[HCAS-GAN] Done."
