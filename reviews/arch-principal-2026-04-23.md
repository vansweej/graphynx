# Architecture Review — graphynx workspace

**Date:** 2026-04-23  
**Reviewer:** Principal Software Architect  
**Scope:** Full workspace — `graph-core`, `backends`, `backends-cuda`, `runtime`, `playground`  
**Commit:** HEAD (workspace as-read)

---

## 1. Executive Summary

The graphynx workspace aspires to be a **backend-agnostic dataflow graph execution engine** for ML workloads. The ARCHITECTURE.md describes a rich three-layer design (Core → Execution → Backend) with a graph IR, scheduler, executor, buffer manager, and pluggable backends.

**In practice, only the bottom half of the Core layer and a single CUDA backend exist.** The type system (`DType`, `Dim`, `Shape`, `Layout`, `TensorType`) and ML-op catalog are well-executed. The Backend trait is structurally sound. Everything else described in the architecture document — Graph IR, builder API, scheduler, executor, buffer manager — does not exist.

The project is at the **foundation-laying stage**. The foundations are mostly well-built, but several structural decisions need correction before the execution layer is implemented, or they will become costly to fix later.

**Verdict:** The architecture direction is sound. The core type system is production-quality. Three issues require immediate attention before further build-out.

---

## 2. What Works Well

### 2.1 Clean Layered Dependency Graph

```
graph-core  (zero hardware deps)
    ↑
backends    (depends on graph-core only)
    ↑
backends-cuda (depends on backends only)
    ↑
runtime     (depends on all three)
```

The DAG is strict and well-enforced via Cargo workspace dependencies. `graph-core` depends only on `std`, `thiserror`, and `bytemuck`. `unsafe` is confined to backend crates. This is the right structure.

### 2.2 Type System Design

- **`DType`** — Complete scalar type coverage with `Custom(&'static str)` escape hatch. Category helpers (`is_float`, `is_signed`, etc.) are useful. The `&'static str` lifetime for Custom avoids allocation while still being descriptive.

- **`Dim`** — The three-variant design (`Fixed`, `Dynamic`, `Symbolic`) is superior to the `Dynamic(Option<String>)` originally in the architecture doc. Symbolic dim matching rules are correctly implemented and well-tested.

- **`Shape`** — Broadcasting, reshape validation, and stride computation are correctly implemented. The `Shape` type encapsulates all dimensionality logic in one place.

- **`TensorType`** — Private fields with validated constructors, fluent builder, consuming transforms. This is textbook defensive API design.

- **`MlOp` catalog** — Curated enum with `Custom` escape hatch. Parameter structs use validated constructors. Comprehensive error types.

All types use `thiserror` for ergonomic errors. All public constructors validate invariants. Test coverage appears thorough across success and failure paths.

### 2.3 Backend Trait Design

The unified `Backend` trait with `BackendCaps` / `MemoryModel` capability flags is a pragmatic choice for the current project size. The trait's decision to use default implementations returning `Err(UnsupportedNodeKind)` for optional dispatch methods keeps the implementor experience simple.

`KernelDescriptor` as a trait (not an enum) is extensible — new backends never touch core code.

### 2.4 Build Environment

The Nix flake is thorough. CUDA toolchain, NVIDIA driver library symlinks, glibc version management, and tarpaulin compatibility are all handled. The `build.rs` correctly avoids polluting rpath with system library directories.

### 2.5 Test Infrastructure

Unit tests colocated in source files. Integration tests in `runtime/tests/`. Mock backends for testing the dispatch pipeline without CUDA hardware. Tarpaulin-aware `#[cfg(not(tarpaulin_include))]` annotations on hardware-dependent code.

---

## 3. Issues Requiring Immediate Attention

### 3.1 CRITICAL: `runtime` has a hard dependency on `backends-cuda`

**Problem:** The `runtime` crate lists `backends-cuda` as an unconditional dependency in `Cargo.toml`. This means `runtime` cannot compile without the CUDA toolchain. The `playground` crate has the same issue.

This directly violates the architectural invariant documented in ARCHITECTURE.md §8:

> **Key invariant:** Core and Execution layers have zero backend deps.

And §9 decision #8:

> Core and Execution layers have zero backend deps. Compiles and tests without any hardware SDK.

And §9 decision #9:

> `cargo build` gives CPU-only. `--features cuda,onnx` adds CUDA + ONNX.

**Impact:** Anyone who checks out this repo without a CUDA-capable Nix environment cannot build the runtime or run tests. CI on non-GPU nodes cannot validate runtime logic. Every future backend-agnostic feature (graph IR, scheduler, executor) will inherit this hard CUDA dependency if built in the `runtime` crate.

**Recommendation:** The `runtime` crate should depend only on `graph-core` and `backends`. CUDA should be a Cargo feature:

```toml
[dependencies]
graph-core = { workspace = true }
backends   = { workspace = true }

[features]
cuda = ["dep:backends-cuda"]

[dependencies.backends-cuda]
workspace = true
optional = true
```

The `demo` binary should be feature-gated: `[[bin]] name = "demo" required-features = ["cuda"]`. The `playground` crate should similarly make `backends-cuda` optional.

### 3.2 CRITICAL: Execution layer is entirely absent

**Problem:** The ARCHITECTURE.md describes five major execution-layer components:

| Component | Status |
|---|---|
| Graph IR (nodes, edges, builder) | **Not implemented** |
| Graph validation (cycles, types, backends) | **Not implemented** |
| Scheduler (topological sort) | **Not implemented** |
| Executor (dispatch loop) | **Not implemented** |
| Buffer Manager (alloc, transfer, tracking) | **Not implemented** |

The only runtime functionality is `run_kernel()`, which is a flat alloc→upload→dispatch→download sequence with no graph concept. This is a convenience wrapper, not a graph engine.

**Impact:** This is a maturity gap, not a design flaw. However, the architecture document reads as if these components exist. Anyone evaluating this project against the architecture doc will be misled.

**Recommendation:** ARCHITECTURE.md should clearly separate "implemented" from "planned" sections. Alternatively, add a status table at the top of the document. Do not let the architecture doc and the codebase diverge silently — this erodes trust in both.

### 3.3 HIGH: Backend trait dispatch signatures diverge from architecture

**Problem:** The implemented `Backend` trait has:

```rust
fn dispatch_ml_op(&self, op_name: &str, inputs: &[&[u8]], outputs: &mut [Vec<u8>]) -> Result<(), BackendError>;
fn dispatch_ml_model(&self, model_name: &str, inputs: &[&[u8]], outputs: &mut [Vec<u8>]) -> Result<(), BackendError>;
```

The architecture document specifies:

```rust
fn dispatch_ml_op(&self, op: &MlOp, inputs: &[&[u8]], outputs: &mut [Vec<u8>]) -> Result<(), BackendError>;
fn dispatch_ml_model(&self, desc: &MlModelDescriptor, inputs: &[&[u8]], outputs: &mut [Vec<u8>]) -> Result<(), BackendError>;
```

The implementation passes a `&str` name instead of the typed `&MlOp` or `&MlModelDescriptor`. This loses all parameter information. A backend receiving `"Conv2d"` as a string has no access to kernel size, stride, padding, etc. — making the entire `MlOp` parameter system unusable from the dispatch path.

**Impact:** When MlOp dispatch is actually implemented, this signature will need to change, breaking all backend implementations.

**Recommendation:** Fix now while there are zero callers. Change `dispatch_ml_op` to accept `&MlOp` and `dispatch_ml_model` to accept a descriptor type. The `backends` crate already depends on `graph-core`, so `MlOp` is available.

---

## 4. Structural Concerns (Medium Priority)

### 4.1 `CudaBackend` and `CudaKernelDesc` are demo-only

`CudaBackend::new()` hardcodes the function name `"hello_kernel"` in the PTX registration. `CudaKernelDesc::new()` hardcodes `module_name: "hello"`. These are demo artifacts that make the CUDA backend unusable for any other kernel.

Before the CUDA backend can be used as a general compute backend:
- `CudaBackend` needs a module registration API (load arbitrary PTX modules with arbitrary exported functions).
- `CudaKernelDesc` needs to accept an arbitrary module name.
- The single-input/single-output constraint in `dispatch_compute` needs to be lifted.

**This is expected for the current stage**, but these should be tracked as blockers for the first real workload.

### 4.2 `run_kernel` assumes output_size == input_size

```rust
let output_size_bytes = input_bytes.len(); // always matches input
```

This only works for element-wise kernels like the "double each value" demo. Any real kernel (matmul, convolution, reduction) will have a different output size. This function signature needs to either accept an explicit output size or be replaced by the buffer manager described in the architecture.

### 4.3 `dispatch_compute` uses `&mut [&mut dyn DeviceBuffer]` (double indirection)

The output buffer parameter `outputs: &mut [&mut dyn DeviceBuffer]` requires the caller to have mutable references to trait objects in a mutable slice. This is awkward to use and forces the CUDA backend into a pattern where it clones the `CudaSlice`, runs the kernel on the clone, then copies back with `dtod_copy`:

```rust
let mut output_slice = output.slice.clone();
unsafe { kernel.launch(cfg, (&input.slice, &mut output_slice))? };
self.device.dtod_copy(&output_slice, &mut output.slice)?;
```

This is an unnecessary device-to-device copy on every kernel launch. Consider whether `outputs: &mut [Box<dyn DeviceBuffer>]` or a different ownership model would allow direct kernel execution into the output buffer.

### 4.4 No `BackendRegistry`

The architecture describes a registry for mapping `DeviceId`s to backend instances. This doesn't exist. When the executor is implemented, it will need a way to look up backends. Consider designing the registry API before the executor, so the executor's interface is correct from the start.

### 4.5 Mock test infrastructure is duplicated

`MemBackend`, `MemBuffer`, `DoubleKernelDesc`, and `FailAllocBackend` are defined in both:
- `runtime/src/lib.rs` (unit tests)
- `runtime/tests/common/mod.rs` (integration tests)

These are nearly identical. Consider extracting them into a `dev-dependencies`-only test utilities crate or a shared `cfg(test)` module.

---

## 5. Observations (Low Priority / For Awareness)

### 5.1 `DeviceId` is an unstructured string

`DeviceId` wraps a `String` with no format validation beyond non-emptiness. The architecture doc implies a `<backend>:<index>` convention, but it's not enforced. When multi-device scheduling is implemented, the executor will need to parse device IDs to route to backends. Consider a structured `DeviceId` with backend-name and ordinal fields, or at least a `DeviceId::parse()` method.

### 5.2 `MemBuffer` uses `RefCell<Vec<u8>>` with `unsafe impl Send + Sync`

The mock buffer in tests uses interior mutability via `RefCell` and then declares `Send + Sync` unsafely. This works because tests are single-threaded, but if tests ever run concurrently or if this mock is reused in async code, it will be unsound. Using `Mutex<Vec<u8>>` would be safe and negligible in test performance.

### 5.3 `backends/src/ml/mod.rs` is a placeholder

The `ml/` module contains a single comment. If it's not yet needed, remove it to avoid suggesting that ML runtime backend infrastructure exists. Add it back when actual ML backend work begins.

### 5.4 `workspace.members` includes `playground`

The `playground` crate is a scratch space with a hard dependency on `backends-cuda`. Its presence in the workspace means `cargo test` and `cargo build` in the workspace root will try to compile it, which fails without CUDA.

### 5.5 Edition 2021 could be upgraded

The workspace uses `edition = "2021"`. Rust 2024 edition (available since 1.85.0, and this project uses 1.94.1) offers improved `unsafe` block checking and other ergonomic improvements. Not urgent, but worth considering.

---

## 6. Failure Mode Analysis

| Scenario | Current Behavior | Risk |
|---|---|---|
| Build without CUDA toolchain | `cargo build` fails at `backends-cuda` link step | **Blocks all non-CUDA development** |
| CUDA device not available at runtime | `CudaBackend::new()` returns `Err(BackendError::Device)` | Handled correctly |
| PTX module invalid | `load_ptx` returns `Err` → `BackendError::Device` | Handled correctly |
| Kernel function name mismatch | `get_func` returns `None` → `BackendError::InvalidKernel` | Handled correctly |
| Buffer size mismatch on download | `host.copy_from_slice(&result)` panics if lengths differ | **Bug**: should return `Err` |
| Empty input to `run_kernel` | `bytemuck::cast_slice` panics on zero-length | **Documented as known limitation**, but should return `Err` |
| Backend receives wrong buffer type | `downcast_ref` returns `None` → `BackendError::Buffer` | Handled correctly |
| Concurrent backend access | Backend trait requires `Send + Sync`; CudaBackend uses `Arc<CudaDevice>` | Correct for single-stream; unclear for multi-stream |

---

## 7. Scalability Assessment

| Dimension | Assessment |
|---|---|
| **Adding new backends** | Good. `KernelDescriptor` as trait + `Backend` trait with defaults means new backends need only implement what they support. No core code changes. |
| **Growing the MlOp catalog** | Good. Adding a new variant to `MlOp` is additive. Backends handle unknown ops via `UnsupportedOp`. |
| **Multi-device execution** | Not yet addressable. Requires the graph IR, scheduler, and buffer manager. The `DeviceId` + `BackendCaps` foundation is adequate. |
| **Large graphs (1000s of nodes)** | Unknown. No graph IR exists to evaluate. The topological sort described in ARCHITECTURE.md (Kahn's algorithm) is O(V+E), which is fine. |
| **Team scaling** | Good crate boundaries. Core, backends, and runtime can be developed in parallel by different people. |

---

## 8. Recommendations — Priority Order

| # | Priority | Action |
|---|---|---|
| 1 | **Critical** | Make `backends-cuda` an optional/feature-gated dependency in `runtime` and `playground`. Ensure `cargo build` and `cargo test` work without CUDA. |
| 2 | **Critical** | Change `dispatch_ml_op` signature from `&str` to `&MlOp`. Change `dispatch_ml_model` to accept a typed descriptor. Do this before any backend implements these methods. |
| 3 | **High** | Add a status section to ARCHITECTURE.md distinguishing implemented vs. planned components. |
| 4 | **High** | Fix `download` in `CudaBackend` to check `host.len() == src.size_bytes()` instead of relying on `copy_from_slice` panicking. |
| 5 | **Medium** | Remove the `dtod_copy` workaround in `dispatch_compute` by redesigning the output buffer ownership model. |
| 6 | **Medium** | Generalize `CudaBackend::new` and `CudaKernelDesc::new` to accept arbitrary modules and function names. |
| 7 | **Medium** | Handle empty input in `run_kernel` gracefully (return `Err` instead of panicking). |
| 8 | **Low** | Deduplicate mock test infrastructure into a shared location. |
| 9 | **Low** | Remove the empty `backends/src/ml/` module until ML runtime work begins. |

---

## 9. Assumptions

- I assume the "planned immutable-graph architecture" described in AGENTS.md and ARCHITECTURE.md is the intended target. If the project has pivoted to a different execution model, this review should be re-evaluated.
- I assume CUDA is the only hardware backend that will exist short-term. The review would change if multiple explicit-memory backends (e.g., Vulkan, OpenCL) were imminent.
- I assume the project is in active early development and the architecture doc was written as a forward-looking design, not a description of current state. If it was intended as a specification that should already be implemented, the gap is more concerning.

---

*End of review.*
