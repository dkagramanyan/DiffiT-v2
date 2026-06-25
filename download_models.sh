#!/bin/bash
# Pure-shell prefetch of every pretrained weight DiffiT training + the combra
# metrics need, straight into the caches the libraries look in -- no GPU
# required. Run on a node WITH internet (e.g. a login node); the weights cache
# under $HOME, shared with the offline compute nodes, so the training jobs then
# need no network.
#
# Two cache families are populated:
#   * torch.hub / clip caches  -- via plain wget/curl/git (combra backbones +
#     the torchvision InceptionV3 used for inline FID/IS).
#   * HuggingFace cache        -- the two SD-VAE decoders, via huggingface-cli
#     (falls back to `python scripts/download_models.py` if the CLI is absent).
#
# URLs and on-disk filenames are pinned to what the installed libraries expect
# (pytorch-fid, open_clip 'openai', torch.hub dinov2, torchvision inception).
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh
set -u

MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
HUB_CKPT="${MODEL_CACHE}/torch/hub/checkpoints"   # torchvision + pytorch-fid + dinov2 weights
HUB_DIR="${MODEL_CACHE}/torch/hub"                # torch.hub repo code (dinov2)
CLIP_DIR="${MODEL_CACHE}/clip"                    # open_clip 'openai' weights
mkdir -p "$HUB_CKPT" "$HUB_DIR" "$CLIP_DIR"

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: need wget or curl on PATH." >&2
    exit 1
fi

status=0
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  exists: ${dest##*/}"
        return 0
    fi
    echo "  downloading: ${url##*/}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    else
        curl -fL -o "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    fi
}

echo "Caching pretrained models under: $MODEL_CACHE"

echo; echo "[1/4] torchvision InceptionV3 (inline FID/IS during training) -> $HUB_CKPT"
fetch "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" "$HUB_CKPT/inception_v3_google-0cc3c7bd.pth"

echo; echo "[2/4] InceptionV3 FID weights (combra fid) -> $HUB_CKPT"
fetch "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" "$HUB_CKPT/pt_inception-2015-12-05-6726825d.pth"

echo; echo "[3/4] CLIP ViT-L-14-336 'openai' (combra cmmd) -> $CLIP_DIR"
fetch "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt" "$CLIP_DIR/ViT-L-14-336px.pt"

echo; echo "[4/4] DINOv2 dinov2_vitb14 (combra fd_dinov2) -> $HUB_DIR"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" "$HUB_CKPT/dinov2_vitb14_pretrain.pth"
# torch.hub also needs the dinov2 model code (it normally fetches the repo itself).
if [[ -d "$HUB_DIR/facebookresearch_dinov2_main" ]]; then
    echo "  exists: facebookresearch_dinov2_main/"
elif command -v git >/dev/null 2>&1; then
    echo "  cloning facebookresearch/dinov2"
    git clone --depth 1 https://github.com/facebookresearch/dinov2 "$HUB_DIR/facebookresearch_dinov2_main" \
        || { echo "  FAILED: git clone dinov2"; status=1; }
else
    echo "  SKIPPED dinov2 repo: git not on PATH"; status=1
fi

# --- SD-VAE decoders (HuggingFace cache) ---------------------------------
# These live in the HF cache (snapshots/blobs layout), so use the HF CLI when
# available; otherwise fall back to the Python downloader which uses the
# diffusers/HF hub client directly. Newer huggingface_hub ships `hf` and turns
# `huggingface-cli` into a deprecation stub that exits non-zero, so prefer `hf`.
echo; echo "[VAE] stabilityai/sd-vae-ft-ema + sd-vae-ft-mse -> HuggingFace cache"
if command -v hf >/dev/null 2>&1; then
    hf_cmd="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_cmd="huggingface-cli"
else
    hf_cmd=""
fi
if [[ -n "$hf_cmd" ]]; then
    for repo in stabilityai/sd-vae-ft-ema stabilityai/sd-vae-ft-mse; do
        echo "  downloading: $repo"
        "$hf_cmd" download "$repo" >/dev/null \
            || { echo "  FAILED: $repo"; status=1; }
    done
else
    echo "  hf / huggingface-cli not found -- fetch the VAEs with:"
    echo "      python scripts/download_models.py"
fi

echo
if [[ $status -eq 0 ]]; then
    echo "Done. All weights cached under $MODEL_CACHE (+ HuggingFace cache for the VAEs)."
else
    echo "Some downloads failed (see above)."
fi
exit $status
