#!/bin/bash
set -e
cd "$(dirname "$0")"

# The Nix gcc wrapper crashes (stack overflow from hardening flags + CUDA headers).
# Work around by invoking .nvcc-wrapped directly in a clean environment using system gcc-13.
# NVCC_WRAPPED, NVCC_HOST_COMPILER, and NVCC_EXTRA_FLAGS are exported by the flake.nix shellHook.
if [[ -z "$NVCC_WRAPPED" || -z "$NVCC_HOST_COMPILER" || -z "$NVCC_EXTRA_FLAGS" ]]; then
  echo "error: NVCC_WRAPPED, NVCC_HOST_COMPILER, and NVCC_EXTRA_FLAGS must be set (enter the nix dev shell first)" >&2
  exit 1
fi

env -i HOME="$HOME" PATH=/usr/bin:/bin \
  "$NVCC_WRAPPED" \
  --compiler-bindir "$NVCC_HOST_COMPILER" \
  $NVCC_EXTRA_FLAGS \
  -ptx kernel.cu -o kernel.ptx
