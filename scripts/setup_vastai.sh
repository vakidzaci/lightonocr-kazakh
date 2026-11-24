#!/usr/bin/env bash
# One-time setup on a fresh vast.ai instance (PyTorch template, CUDA >= 12.8 image recommended).
#   bash scripts/setup_vastai.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> system packages (fonts that cover Kazakh letters, tmux)"
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  # installed one by one: package names differ slightly between Ubuntu/Debian releases
  for pkg in tmux fontconfig fonts-dejavu fonts-liberation2 fonts-noto-core fonts-paratype fonts-roboto \
             fonts-open-sans fonts-ibm-plex fonts-linuxlibertine fonts-freefont-ttf fonts-urw-base35 \
             fonts-crosextra-carlito fonts-crosextra-caladea fonts-ubuntu fonts-firacode; do
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null 2>&1 \
      && echo "   installed $pkg" || echo "   (skipped $pkg - not available)"
  done
fi

echo "==> python packages"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

echo "==> checks"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available - pick a GPU instance / CUDA image"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {p.name}, {p.total_memory / 1024**3:.0f} GB")
print(f"   torch {torch.__version__}, CUDA {torch.version.cuda}, bf16: {torch.cuda.is_bf16_supported()}")
import transformers, peft
print(f"   transformers {transformers.__version__}, peft {peft.__version__}")
from generator.fonts import find_fonts, FontPicker
fonts = find_fonts(refresh=True)
print(f"   {len(fonts)} fonts in {len(FontPicker(fonts).families)} families can render Kazakh")
EOF

if [ -n "${HF_TOKEN:-}" ]; then
  echo "==> HF_TOKEN is set (needed only to push models to the Hub)"
else
  echo "==> HF_TOKEN not set: downloads work without it; set it to push models: export HF_TOKEN=hf_..."
fi
echo "done. Next: bash scripts/download_all.sh   (or the full pipeline: bash scripts/run_all.sh)"
