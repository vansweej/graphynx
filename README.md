# graphynx

graphynx is a graph-based execution engine that unifies data processing and GPU
computation into a single optimized pipeline. From dirty data to GPU execution —
in one graph.

It models real-world workloads — from messy data cleaning to high-performance GPU
kernels — as a single dataflow graph, enabling efficient execution across CPU and
GPU with minimal data movement.

## Overview

graphynx provides a **backend-agnostic dataflow graph execution engine** in Rust.
Users define computation as a directed graph of typed nodes. The engine schedules
and executes nodes in dependency order, dispatching work to whichever backend a
node targets — raw compute kernels on GPUs, primitive ML operations, or entire
pre-trained model inference.

Backends plug in through a single unified trait:

| Backend | Kind | Status |
|---|---|---|
| CPU | Compute | planned |
| CUDA | Compute | in progress |
| OpenCL | Compute | planned |
| Vulkan / wgpu | Compute | planned |
| ONNX Runtime | ML inference | planned |
| libtorch | ML ops + inference | planned |
| candle / burn | ML ops + inference | planned |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Getting Started

### Prerequisites

- [Nix](https://nixos.org/) with flakes enabled, **or** a manual CUDA SDK install
- NVIDIA GPU with a compatible driver

### Build

```bash
# Enter the reproducible dev shell (sets CUDA_PATH, RUSTFLAGS, etc.)
nix develop

# Compile the CUDA kernel to PTX (required before running)
./compile-kernel.sh

# Build the crate
cargo build
```

### Run the demo

```bash
cargo run
```

The demo launches a CUDA kernel that doubles every element of a 10-integer array:

```
Input:  [3, 7, 1, 9, 4, 6, 2, 8, 5, 10]
Output: [6, 14, 2, 18, 8, 12, 4, 16, 10, 20]
```

### Test

```bash
cargo test               # run all tests
cargo test <test_name>   # run a single test by name
cargo tarpaulin          # code coverage
```

### Lint

```bash
cargo clippy
cargo fmt --check
cargo deny check
```

## Project Structure

```
src/
  lib.rs                        # crate root — run_kernel<T>, public API re-exports
  main.rs                       # standalone CUDA demo
  backends/
    mod.rs                      # Backend trait, DeviceBuffer, KernelDescriptor, BackendError, DeviceId
    compute/
      mod.rs
      cuda.rs                   # CudaBackend, CudaBuffer, CudaKernelDesc
    ml/
      mod.rs                    # (ML runtime backends — planned)
  core/
    mod.rs
    types/
      mod.rs
      dtype.rs                  # DType scalar element type enum, DTypeError
      dim.rs                    # Dim (Fixed / Dynamic / Symbolic)
      tensor_type.rs            # TensorType, TensorTypeBuilder, TensorTypeError, Layout
      layout.rs                 # Layout enum
      shape/
        mod.rs                  # Shape struct, ShapeError, constructors, strides, reshape
        ops.rs                  # Broadcasting and compatibility logic
    ops/
      mod.rs                    # MlOp enum, MlOpError, safe constructors
      params.rs                 # per-op parameter structs
build.rs                        # emits CUDA linker search paths for cargo
kernel.cu                       # CUDA C source for the hello_kernel
compile-kernel.sh               # compiles kernel.cu → kernel.ptx via nvcc
ARCHITECTURE.md                 # full long-term architecture plan
AGENTS.md                       # build/lint/style reference for agentic coding tools
docs/                           # detailed module documentation
```

## License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.
