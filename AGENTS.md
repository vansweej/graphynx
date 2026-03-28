# AGENTS.md — graphynx agent guide

This file is the repo-local guide for coding agents working in `graphynx`.
Follow it as the source of truth for commands, validation steps, and style.

## Environment

- This repo uses a Nix flake. Run builds, tests, lint, and coverage inside the
  dev shell.
- Preferred one-off form:

```bash
nix develop --command <cmd>
```

- The shell sets `CUDA_PATH`, `NVRTC_PATH`, `RUSTFLAGS`, `NVCC_WRAPPED`,
  `NVCC_HOST_COMPILER`, and related linker/runtime settings.
- Do not try to recreate the CUDA environment manually.

## Rule files

- No repo-local Cursor rules were found in `.cursor/rules/` or `.cursorrules`.
- No repo-local Copilot rules were found in `.github/copilot-instructions.md`.
- Also read `ARCHITECTURE.md` before making structural changes.

## Current codebase shape

- Public modules in `src/lib.rs` are:
  - `backend`
  - `cuda_backend`
  - `dtype`
  - `ml_op`
  - `shape`
  - `tensor_type`
- `run_kernel<T>` is the current crate-level convenience API.
- The codebase is still flat under `src/`; future subdirectories are only plans.

## Design constraints

- Keep the core library backend-agnostic.
- Keep `unsafe` confined to backend implementations.
- Prefer validated constructors for public types.
- Preserve a working default build without requiring CUDA execution paths.
- Treat `KernelDescriptor` as an extensible trait, not an enum.
- Avoid patterns that fight the planned immutable-graph architecture.

## Build commands

Run inside the Nix shell:

```bash
nix develop --command cargo build
nix develop --command cargo build --release
```

Build the demo CUDA kernel with:

```bash
nix develop --command ./compile-kernel.sh
```

Notes:

- `compile-kernel.sh` requires the NVCC environment exported by the flake.
- `build.rs` emits CUDA linker search paths from `CUDA_PATH` and `NVRTC_PATH`.

## Test commands

Primary commands:

```bash
nix develop --command cargo test
nix develop --command cargo test --doc
nix develop --command cargo tarpaulin
```

Single-test workflows:

```bash
nix develop --command cargo test some_test_name
nix develop --command cargo test module::tests::some_test_name
nix develop --command cargo test module::tests::some_test_name -- --exact
nix develop --command cargo test ml_op::tests::conv2d_new_valid -- --exact
```

Useful scoped runs:

```bash
nix develop --command cargo test ml_op
nix develop --command cargo test tensor_type::tests
nix develop --command cargo test --doc ml_op
```

Testing expectations:

- Run `cargo test` after code changes.
- Run `cargo tarpaulin` when adding or changing logic; target roughly 90%+.
- CUDA-specific paths may use `#[cfg(not(tarpaulin_include))]` when coverage is
  not portable or not meaningful.
- Prefer unit tests in `mod tests` within the same source file.

## Lint / format / maintenance

```bash
nix develop --command cargo fmt
nix develop --command cargo fmt --check
nix develop --command cargo clippy
nix develop --command cargo deny check
nix develop --command cargo outdated
```

- Fix clippy warnings instead of suppressing them when practical.
- Do not add `#![allow(...)]` or `#[allow(...)]` without a real reason.

## Toolchain and dependencies

- Rust is pinned in `rust-toolchain.toml` to `1.94.1`.
- If you update the toolchain, also run:

```bash
nix flake update
nix develop --command cargo build
nix develop --command cargo test
nix develop --command cargo tarpaulin
```

- Key dependencies today:
  - `cudarc = 0.9` with `default-features = false`, `driver`, `std`
  - `thiserror = 2`
  - `bytemuck = 1` with `derive`
- New dependencies should minimize features and must not pull GPU runtime
  dependencies into core modules.

## Imports and module structure

- Use two import groups with one blank line between them:
  1. `std` + external crates
  2. `crate::...`
- Preferred shape:

```rust
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use thiserror::Error;

use crate::backend::{Backend, BackendError};
```

- Keep module names `snake_case`.
- In larger files, use `// ── ... ──` section headings.

## Naming and formatting

- Types, enums, traits: `PascalCase`
- Functions, methods, modules, locals: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Error variants: `PascalCase`
- Prefer descriptive names over ad hoc abbreviations.
- Follow rustfmt defaults, 4-space indentation, same-line braces, and trailing
  commas in multiline literals/calls.

## Types and API design

- Prefer small validated constructors for public types.
- Fallible constructors should return `Result<Self, ErrorType>` with a dedicated
  error enum.
- Keep unchecked constructors only for compatibility or deliberate convenience.
- Add explicit type annotations when types are not obvious, especially around
  CUDA slices, trait objects, and generic code.
- Use `&dyn Trait` for borrowed trait objects and `Box<dyn Trait>` for owned
  ones.
- Use `Arc<T>` only when shared ownership is actually needed.

## Error handling

- Use `thiserror::Error` for error enums.
- Prefer dedicated errors such as `BackendError`, `ShapeError`,
  `TensorTypeError`, `DTypeError`, and `MlOpError`.
- Convert foreign errors at boundaries with `.to_string()`.
- Prefer `?` and `ok_or_else(...)` over manual `match` boilerplate.
- Avoid `unwrap()` in library code; it is acceptable in tests and narrowly
  documented demo-only paths.

## Documentation, comments, and unsafe

- Public items should have `///` docs with a concise opening summary.
- Add `# Errors`, `# Safety`, and `# Examples` when useful.
- Keep doc examples runnable where practical.
- Use comments to explain why, not just what.
- Every `unsafe` block must have a preceding `// Safety:` comment.
- Core modules must remain safe Rust.
- Prefer `bytemuck::cast_slice` over manual pointer casts.
- Downcasting via `Any` is part of the backend abstraction; keep failures
  specific and actionable.

## Change hygiene

- Match the repo's style of broad unit-test coverage within each file.
- Add both success-path and failure-path tests for validated constructors.
- If you change examples in docs, ensure doc-tests still pass.
- After editing `Cargo.toml`, run `cargo deny check`.
- Before finishing substantial work, run at least:

```bash
nix develop --command cargo fmt --check
nix develop --command cargo clippy
nix develop --command cargo test
```
