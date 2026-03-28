# Getting Started

## Prerequisites

- [Nix](https://nixos.org/download.html) with flakes enabled
- NVIDIA GPU with installed driver (kernel module)
- Host NVIDIA libraries (`libcuda.so.1`) accessible to the Nix environment

## Environment Setup

All commands must be run inside the Nix development shell:

```bash
nix develop
```

The shell hook automatically sets:
- `CUDA_PATH` — path to CUDA toolkit (headers, nvcc, stub libraries)
- `NVRTC_PATH` — path to NVRTC runtime compilation library
- `RUSTFLAGS` — rpaths for linking against host NVIDIA driver libraries

## Build

```bash
cargo build              # Debug build
cargo build --release    # Release build
```

## Compile the CUDA Kernel

Before running the binary, compile the PTX kernel:

```bash
./compile-kernel.sh
```

This compiles `kernel.cu` into `kernel.ptx` using `nvcc`. The script requires the Nix shell environment variables (`NVCC_WRAPPED`, `NVCC_HOST_COMPILER`, `NVCC_EXTRA_FLAGS`).

## Run the Demo

```bash
cargo run
```

Expected output:

```
Input:  [3, 7, 1, 9, 4, 6, 2, 8, 5, 10]
Output: [6, 14, 2, 18, 8, 12, 4, 16, 10, 20]
```

The demo loads `kernel.ptx`, creates a CUDA backend on device 0, and runs a kernel that doubles each integer element.

## Run Tests

```bash
cargo test                # Run all tests
cargo test <name>         # Run tests matching <name>
cargo tarpaulin           # Code coverage report
```

## Lint and Format

```bash
cargo clippy              # Lint checks
cargo fmt                 # Format code
cargo fmt --check         # Check formatting without writing
cargo deny check          # License, advisory, and ban checks
```

## Project Structure

```
src/
  lib.rs              # Crate root, run_kernel<T> convenience function
  backend.rs          # Core traits: Backend, DeviceBuffer, KernelDescriptor
  cuda_backend.rs     # CUDA implementation of the Backend trait
  dtype.rs            # DType scalar element type enum
  tensor_type.rs      # Tensor type system: Dim, Layout, TensorType, TensorTypeBuilder
  main.rs             # Binary entry point (CUDA demo)
build.rs              # Build script (CUDA linker paths)
kernel.cu             # CUDA kernel source
compile-kernel.sh     # PTX compilation script
```

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
graphynx = { path = "../path/to/graphynx" }
```

### Running a CUDA kernel

```rust
use graphynx::cuda_backend::{CudaBackend, CudaKernelDesc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ptx = include_str!("path/to/kernel.ptx");
    let backend = CudaBackend::new(0, ptx, "module_name")?;

    let desc = CudaKernelDesc::new("kernel_func", [1, 1, 1], [N as u32, 1, 1]);
    let input: Vec<i32> = vec![1, 2, 3, 4, 5];
    let output: Vec<i32> = graphynx::run_kernel(&backend, &desc, &input)?;

    println!("{:?}", output);
    Ok(())
}
```

### Describing tensor types for graph edges

```rust
use graphynx::tensor_type::{Dim, Layout, TensorType};
use graphynx::backend::DeviceId;
use graphynx::dtype::DType;

// Common shapes use the short constructors:
let scalar   = TensorType::scalar(DType::F32);
let vector   = TensorType::vector(DType::I32, 1024)?;
let matrix   = TensorType::matrix(DType::F64, 512, 512)?;

// Complex shapes use the builder:
let image = TensorType::builder(DType::F32)
    .shape(vec![
        Dim::Symbolic("batch".into()),
        Dim::Fixed(3),
        Dim::Fixed(224),
        Dim::Fixed(224),
    ])
    .layout(Layout::NCHW)
    .dim_names(vec![
        "batch".into(), "channels".into(), "height".into(), "width".into(),
    ])
    .device(DeviceId::new("cuda:0"))
    .build()?;

println!("{}", image); // f32[batch, 3, 224, 224] NCHW @ cuda:0
```

## Rust Toolchain

The Rust toolchain is pinned in `rust-toolchain.toml` to `stable 1.94.1`. Do not change this without following the upgrade procedure documented in `AGENTS.md`.

## Further Reading

- [Architecture Overview](architecture.md) — layered design, data flow, and design principles
- [Backend Trait System](backend-trait.md) — how the `Backend`, `DeviceBuffer`, and `KernelDescriptor` traits work
- [CUDA Backend](cuda-backend.md) — CUDA-specific implementation details
- [DType](dtype.md) — scalar element type system
- [Tensor Type System](tensor-type.md) — `Dim`, `Layout`, `TensorType`, construction, compatibility, and display
- [ARCHITECTURE.md](../ARCHITECTURE.md) — full long-term design plan
