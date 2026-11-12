#!/usr/bin/env bash
# ==============================================================================
# Convert merged HuggingFace FP16/BF16 checkpoint to quantized GGUF (Q4_K_M)
# and generate SHA256 integrity manifest for Edge deployment.
# ==============================================================================
set -euo pipefail

MERGED_DIR="${1:?Usage: $0 <merged_checkpoint_dir> <output_gguf_path>}"
OUTPUT_GGUF="${2:?Usage: $0 <merged_checkpoint_dir> <output_gguf_path>}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"

mkdir -p "$(dirname "$OUTPUT_GGUF")"
FP16_GGUF="${OUTPUT_GGUF%.gguf}_fp16_tmp.gguf"

echo "[Quantize] Converting Safetensors in '$MERGED_DIR' to FP16 GGUF..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$FP16_GGUF" \
    --outtype f16

echo "[Quantize] Quantizing FP16 GGUF -> $QUANT_TYPE..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$FP16_GGUF" "$OUTPUT_GGUF" "$QUANT_TYPE"

rm -f "$FP16_GGUF"

echo "[Quantize] Computing SHA256 checksum..."
(cd "$(dirname "$OUTPUT_GGUF")" && sha256sum "$(basename "$OUTPUT_GGUF")" > "$(basename "$OUTPUT_GGUF").sha256")
echo "[Quantize] Artifact ready: $OUTPUT_GGUF"
