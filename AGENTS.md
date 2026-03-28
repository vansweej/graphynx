# AGENTS.md — Developer & Agent Reference

This file documents the conventions, commands, and code style for the `rustycuda` project. All agentic coding agents operating in this repository must follow these guidelines.

---

## Environment Setup

This project uses a **Nix flake** for reproducible development environments. All build, test, and lint commands must be run inside the Nix dev shell:

```bash
nix develop
```

The shell hook sets `CUDA_PATH`, `NVRTC_PATH`, and the necessary `RUSTFLAGS` rpaths automatically. Do not attempt to build outside of `nix develop` unless the CUDA SDK and NVIDIA driver libraries are available on the host.

---

## Build Commands

```bash
cargo build              # Debug build
cargo build --release    # Release build
```

**CUDA kernel compilation** (must be run inside `nix develop`, required before running the binary):

```bash
./compile-kernel.sh      # Compiles kernel.cu → kernel.ptx via nvcc
```

The `build.rs` script emits `cargo:rustc-link-search=native=` directives for CUDA stub libraries based on the `CUDA_PATH` and `NVRTC_PATH` environment variables set by the Nix shell.

---

## Test Commands

```bash
cargo test                          # Run all tests
cargo test <test_name>              # Run a single test by name (substring match)
cargo test <module>::<test_name>    # Run a specific test in a specific module
cargo tarpaulin                     # Code coverage report
```

There are no tests in the codebase yet. When adding tests, place unit tests in `mod tests` blocks at the bottom of each source file, and integration tests in a `tests/` directory.

---

## Lint and Format Commands

```bash
cargo clippy                # Run Clippy (default settings — no clippy.toml exists)
cargo fmt                   # Format code (default rustfmt settings — no rustfmt.toml exists)
cargo fmt --check           # Check formatting without writing changes
cargo deny check            # Check licenses, banned crates, advisories, and sources
cargo outdated              # Check for outdated dependencies
```

There is no `clippy.toml` or `rustfmt.toml`. Do not introduce `#![allow(...)]` suppressions without a clear justification in a comment.

---

## Toolchain

The Rust toolchain is pinned in `rust-toolchain.toml`:

```
stable 1.94.1
```

Do not change this file without following the documented upgrade procedure (change version → `nix flake update` → `cargo build && cargo test && cargo tarpaulin`).

---

## Project Structure

```
src/
  lib.rs            # Crate root — declares modules, exposes run_kernel<T>
  main.rs           # Binary entry point — standalone CUDA demo
  backend.rs        # Core trait definitions (Backend, DeviceBuffer, KernelDescriptor, BackendError)
  cuda_backend.rs   # CUDA implementation of the Backend trait
build.rs            # Build script — emits CUDA linker search paths
kernel.cu           # CUDA C source for the hello_kernel
compile-kernel.sh   # Compiles kernel.cu → kernel.ptx
deny.toml           # cargo-deny license/ban/advisory config
rust-toolchain.toml # Pinned Rust toolchain
ARCHITECTURE.md     # Detailed long-term architecture plan — read before making structural changes
```

The `ARCHITECTURE.md` is the authoritative design reference. Read it before making structural changes.

---

## Module Organization

- Modules are currently **flat** — all source files live in `src/` with no subdirectories.
- `lib.rs` declares `pub mod backend` and `pub mod cuda_backend`. Nothing is re-exported at the crate root except the `run_kernel` function.
- All `unsafe` code is **confined to backend implementations** (`cuda_backend.rs`). The core library must remain 100% safe Rust.
- Planned future layout (see `ARCHITECTURE.md`) introduces `src/core/`, `src/execution/`, `src/backends/` subdirectories. New backend implementations go under `src/backends/`.

---

## Code Style

### Imports

Use a two-tier grouping with a blank line between tiers:

1. `std` imports and external crate imports
2. Local crate imports (`use crate::...`)

```rust
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::backend::{Backend, BackendError, DeviceBuffer, KernelDescriptor};
```

Do not mix `std`/external and local imports in the same group. Within each group, no further sub-sorting is required but alphabetical order is preferred.

### Naming Conventions

| Category | Convention | Examples |
|---|---|---|
| Types, structs, enums | `PascalCase` | `CudaBackend`, `BackendError`, `CudaBuffer` |
| Traits | `PascalCase` | `Backend`, `DeviceBuffer`, `KernelDescriptor` |
| Functions and methods | `snake_case` | `run_kernel`, `upload`, `download`, `size_bytes` |
| Local variables | `snake_case` | `input_bytes`, `output_size_bytes`, `num_elements` |
| Constants | `UPPER_SNAKE_CASE` | `N`, `MAX_BLOCKS` |
| Modules | `snake_case` | `backend`, `cuda_backend` |
| Error enum variants | `PascalCase` | `Cuda`, `InvalidKernel`, `Buffer` |

### Error Handling

- Use `thiserror` for all error type definitions. Do not implement `std::error::Error` manually.
- All error enums derive `#[derive(Debug, Error)]`.
- Error variants carry a `String` message: `Cuda(String)`, formatted with `#[error("... {0}")]`.
- Convert foreign errors at FFI/crate boundaries using `.map_err(|e| BackendError::Cuda(e.to_string()))` — always convert to `String` at the boundary.
- Use `ok_or_else(|| BackendError::InvalidKernel("...".to_string()))` for `Option` → `Result` conversions.
- Use `?` for error propagation throughout. Avoid `unwrap()` except in `main.rs` where the panic is guarded by a documented precondition.
- `main()` returns `Result<(), Box<dyn std::error::Error>>` to allow `?` throughout.

### Type Annotations

Add explicit type annotations on local variables when the type is non-obvious from context, especially for CUDA and generic types:

```rust
let input: Vec<i32> = vec![1, 2, 3];
let d_input: CudaSlice<i32> = dev.htod_sync_copy(&input)?;
let mut d_output: CudaSlice<i32> = dev.alloc_zeros(N)?;
```

Express generic bounds inline on function signatures:

```rust
pub fn run_kernel<T: Pod>(backend: &dyn Backend, ...) -> Result<Vec<T>, BackendError>
```

Use `Box<dyn Trait>` for owned trait objects and `&dyn Trait` for borrowed ones. Use `Arc<T>` for shared ownership across thread/ownership boundaries (e.g., `Arc<CudaDevice>`).

### Documentation Comments

- Use `///` doc comments on all public items (types, traits, functions, methods).
- First line: a concise summary sentence.
- Use `# Flow`, `# Input`, `# Output`, `# Safety`, `# Errors` sections with bullet points or numbered steps for complex items.
- Use `//` inline comments liberally — every non-trivial statement should explain *why*, not just *what*.
- Use `// ---` visual separators to group logical sections within long functions (e.g., `// --- Input ---`, `// --- Output ---`).
- Precede every `unsafe` block with a `// Safety:` comment explaining the precondition:

```rust
// Safety: kernel arguments match the PTX signature (const int*, int*).
unsafe { kernel.launch(cfg, (&d_input, &mut d_output)) }?;
```

### Unsafe Code

- All `unsafe` must be accompanied by a `// Safety:` comment.
- Confine `unsafe` to backend implementations. The core library layer must never contain `unsafe`.
- Use `bytemuck::cast_slice` for safe `&[T]` ↔ `&[u8]` reinterpretation instead of manual unsafe casts.

### Formatting

- 4-space indentation (rustfmt default).
- Opening braces on the same line.
- Trailing commas in multi-line function calls and struct literals.
- No alignment padding of struct fields in production code (alignment is acceptable in documentation examples).

---

## Key Architectural Constraints

These are non-negotiable design principles (see `ARCHITECTURE.md` for rationale):

1. **Zero backend dependencies in the core layer** — core and execution code must compile and test without any GPU SDK installed.
2. **All `unsafe` confined to backend implementations** — the core engine is 100% safe Rust.
3. **All backends are feature-gated** — `cargo build` with no features must give a working CPU-only build.
4. **`KernelDescriptor` is a trait, not an enum** — extend via new types, never by adding variants to a core enum.
5. **Graph is immutable after `build()`** — no structural mutations during execution.

---

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `cudarc` | `0.9` | CUDA driver API bindings (`driver`, `std` features only) |
| `thiserror` | `2` | Error type derivation |
| `bytemuck` | `1` | Safe byte-level type reinterpretation (`derive` feature) |

When adding new dependencies:
- Use `default-features = false` and enable only the features actually needed.
- Run `cargo deny check` after any `Cargo.toml` change to verify license and advisory compliance.
- Do not add dependencies that pull in GPU SDK crates into any non-backend module.
