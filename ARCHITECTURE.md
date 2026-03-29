# Dataflow Graph Execution System — Architecture Plan

## 1. Vision

A **backend-agnostic dataflow graph execution engine** in Rust. Users define
computation as a directed graph of typed nodes. The engine schedules and
executes nodes in dependency order, dispatching work to whichever backend a
node targets — raw compute kernels on GPUs/FPGAs, primitive ML operations,
or entire pre-trained model inference. CUDA, ONNX Runtime, PyTorch, and
every other backend plug in through a single unified trait.

### Design choices (from requirements)

| Decision             | Choice                                                                      |
|----------------------|-----------------------------------------------------------------------------|
| Execution targets    | **Pluggable backends** — CPU, CUDA, OpenCL, Vulkan, wgpu, FPGA, ONNX RT, libtorch, TFLite, candle, burn, custom |
| Graph topology       | **DAG first**, streaming/cycles as a future extension                       |
| Edge data            | **Typed tensors** with dtype, shape (static + dynamic dims), layout, named dims |
| Node kinds           | **Compute** (raw kernels), **MlOp** (primitive ops), **MlModel** (whole-model inference) |
| ML op catalog        | **Curated enum** of common ops + `Custom(String)` escape hatch              |
| Graph definition API | **Rust builder pattern** (programmatic)                                     |
| Scale                | **Single machine** initially                                                |

---

## 2. Architectural layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Backend Layer                                 │
│                                                                         │
│  Compute backends              ML runtime backends                      │
│  ┌─────┐ ┌──────┐ ┌───────┐   ┌──────┐ ┌────────┐ ┌──────┐ ┌───────┐  │
│  │ CPU │ │ CUDA │ │OpenCL │   │ ONNX │ │libtorch│ │TFLite│ │candle │  │
│  └──┬──┘ └──┬───┘ └──┬────┘   └──┬───┘ └───┬────┘ └──┬───┘ └──┬────┘  │
│     │       │        │           │         │         │        │        │
│  ┌──┴┐ ┌───┴┐ ┌─────┴┐ ┌──┐    │         │         │        │        │
│  │wgpu│ │Vkn │ │ FPGA ││..│    │         │         │        │        │
│  └──┬─┘ └──┬─┘ └──┬───┘└┬─┘    │         │         │        │        │
│     └───┬───┴──────┴─────┴──────┴────┬────┴─────────┴────────┘        │
│                                      │                                 │
│                           impl Backend trait                           │
│                       (unified, capability-based)                      │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴─────────────────────────────────┐
│                         Execution Layer                                │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────────┐ │
│  │  Scheduler   │  │  Executor    │  │  Buffer Manager               │ │
│  │  (topo sort, │  │  (dispatch   │  │  (allocate, transfer, track   │ │
│  │   readiness) │  │   loop)      │  │   location; defers to backend │ │
│  │              │  │              │  │   for managed-memory runtimes)│ │
│  └──────────────┘  └──────────────┘  └───────────────────────────────┘ │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴─────────────────────────────────┐
│                           Core Layer                                   │
│                                                                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Graph IR    │  │  Tensor Type     │  │  ML Op   │  │  Error     │ │
│  │  (nodes,     │  │  System (dtype,  │  │  Catalog │  │  Types     │ │
│  │   edges,     │  │   shape, layout, │  │          │  │            │ │
│  │   builder)   │  │   named dims)    │  │          │  │            │ │
│  └──────────────┘  └──────────────────┘  └──────────┘  └────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

**Key invariant:** The Core and Execution layers have zero dependencies on any
backend crate. They depend only on `std` and lightweight utility crates.

---

## 3. Core layer

### 3.1 Tensor type system

The type system describes the data flowing along every edge. It must be rich
enough to validate ML pipelines at graph-build time.

```
src/core/
  types/
    mod.rs
    dtype.rs         -- DType enum
    dim.rs           -- Dim (Fixed / Dynamic / Symbolic)
    tensor_type.rs   -- TensorType
    layout.rs        -- Layout enum
    buffer.rs        -- AnyBuffer trait, Buffer<T>
```

```rust
/// Scalar element type.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    Bool,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F16, BF16, F32, F64,
    /// Escape hatch for backend-specific types (e.g. quantised types).
    Custom(&'static str),
}

impl DType {
    /// Size of one element in bytes. Returns None for Custom.
    pub fn size_bytes(&self) -> Option<usize> { ... }
}

/// A single dimension — either a compile-time constant or a runtime-
/// determined value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dim {
    /// Known at graph-build time.
    Static(usize),
    /// Unknown until execution (e.g. batch size).
    /// The optional string is a symbolic name for grouping
    /// (all dims named "batch" must resolve to the same value).
    Dynamic(Option<String>),
}

/// Standard memory layouts for multi-dimensional tensors.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Layout {
    /// Row-major / C-contiguous (last dim varies fastest).
    RowMajor,
    /// Column-major / Fortran-contiguous (first dim varies fastest).
    ColMajor,
    /// Batch × Channels × Height × Width (PyTorch default).
    NCHW,
    /// Batch × Height × Width × Channels (TensorFlow default).
    NHWC,
    /// No specific layout constraint.
    Any,
}

/// Complete description of a tensor flowing along a graph edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    pub dtype:     DType,
    pub shape:     Vec<Dim>,
    pub layout:    Layout,
    /// Optional human-readable names for each dimension.
    /// e.g. ["batch", "channels", "height", "width"]
    pub dim_names: Option<Vec<String>>,
}
```

**Validation rules** (enforced at `build()` time):

- An edge's producer `TensorType` must be compatible with the consumer's.
- `Static` dims must match exactly.
- `Dynamic` dims match any value, but two `Dynamic("batch")` dims across
  the graph must resolve to the same runtime value.
- Layouts must be equal, or one side must be `Layout::Any`.

### 3.2 ML op catalog

A curated set of common ML operations. Backends that support `MlOp` nodes
inspect the `MlOp` variant to decide how to execute it. Each op carries an
associated params struct.

```
src/core/
  ops/
    mod.rs           -- MlOp enum, re-exports
    params.rs        -- per-op parameter structs
```

```rust
/// Curated catalog of primitive ML operations.
///
/// Each variant maps to a well-known operation that multiple backends
/// can implement. The engine validates input/output tensor types
/// against the op's signature at graph-build time.
pub enum MlOp {
    // ── Linear algebra ──────────────────────────────────────────────
    MatMul(MatMulParams),
    Linear(LinearParams),

    // ── Convolution ─────────────────────────────────────────────────
    Conv2d(Conv2dParams),

    // ── Activation ──────────────────────────────────────────────────
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Softmax(SoftmaxParams),

    // ── Normalisation ───────────────────────────────────────────────
    BatchNorm(BatchNormParams),
    LayerNorm(LayerNormParams),

    // ── Pooling ─────────────────────────────────────────────────────
    MaxPool2d(PoolParams),
    AvgPool2d(PoolParams),

    // ── Shape manipulation ──────────────────────────────────────────
    Reshape(ReshapeParams),
    Transpose(TransposeParams),
    Concat(ConcatParams),
    Flatten(FlattenParams),

    // ── Regularisation ──────────────────────────────────────────────
    Dropout(DropoutParams),

    // ── Element-wise arithmetic ─────────────────────────────────────
    Add,
    Mul,

    // ── Escape hatch ────────────────────────────────────────────────
    /// For any operation not in the catalog. The string is a
    /// backend-interpreted identifier. The `Vec<u8>` carries
    /// serialised parameters.
    Custom { name: String, params: Vec<u8> },
}

// ── Example param structs ───────────────────────────────────────────

pub struct Conv2dParams {
    pub kernel_size: [usize; 2],
    pub stride:      [usize; 2],
    pub padding:     [usize; 2],
    pub dilation:    [usize; 2],
    pub groups:      usize,
}

pub struct SoftmaxParams {
    pub axis: i32,
}

pub struct MatMulParams {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct LinearParams {
    pub in_features:  usize,
    pub out_features: usize,
    pub bias:         bool,
}

pub struct PoolParams {
    pub kernel_size: [usize; 2],
    pub stride:      [usize; 2],
    pub padding:     [usize; 2],
}

pub struct BatchNormParams {
    pub num_features: usize,
    pub eps:          f64,
    pub momentum:     f64,
}

pub struct LayerNormParams {
    pub normalized_shape: Vec<usize>,
    pub eps:              f64,
}

pub struct ReshapeParams {
    pub target_shape: Vec<Dim>,
}

pub struct TransposeParams {
    pub perm: Vec<usize>,
}

pub struct ConcatParams {
    pub axis: i32,
}

pub struct FlattenParams {
    pub start_dim: i32,
    pub end_dim:   i32,
}

pub struct DropoutParams {
    pub p: f64,
}
```

### 3.3 Graph IR and node kinds

```
src/core/
  graph/
    mod.rs
    node.rs          -- Node, NodeId, NodeKind
    edge.rs          -- Edge, Port
    builder.rs       -- GraphBuilder (fluent API)
    ir.rs            -- Graph struct (adjacency list)
    validate.rs      -- cycle detection, type checks, backend checks
```

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct NodeId(pub usize);

/// What kind of computation this node performs.
pub enum NodeKind {
    /// Raw compute kernel (CUDA PTX, SPIR-V, WGSL, FPGA bitstream,
    /// or a native Rust function). The kernel descriptor is backend-
    /// specific and opaque to the engine.
    Compute(Box<dyn KernelDescriptor>),

    /// A single primitive ML operation from the curated catalog.
    /// Multiple backends may support the same MlOp — the engine
    /// picks the one matching the node's target device.
    MlOp(MlOp),

    /// An opaque, pre-trained model loaded from a serialised format
    /// (ONNX, TorchScript, TFLite, SafeTensors, etc.).
    /// The backend handles loading, memory, and execution internally.
    MlModel(MlModelDescriptor),
}

/// Describes a pre-trained model for MlModel nodes.
pub struct MlModelDescriptor {
    /// Format identifier: "onnx", "torchscript", "tflite",
    /// "safetensors", etc.
    pub format:     String,
    /// Path or embedded bytes of the serialised model.
    pub source:     ModelSource,
    /// Named model inputs and their expected tensor types.
    pub inputs:     Vec<(String, TensorType)>,
    /// Named model outputs and their tensor types.
    pub outputs:    Vec<(String, TensorType)>,
}

pub enum ModelSource {
    /// Path to a model file on disk.
    File(PathBuf),
    /// Model bytes embedded in the binary.
    Bytes(Vec<u8>),
}

pub struct Node {
    pub id:       NodeId,
    pub name:     String,
    /// Which backend this node targets (by DeviceId).
    pub device:   DeviceId,
    /// What this node does.
    pub kind:     NodeKind,
    /// Input port type signatures.
    pub inputs:   Vec<TensorType>,
    /// Output port type signatures.
    pub outputs:  Vec<TensorType>,
}

pub struct Edge {
    pub from: (NodeId, usize),   // (source node, output port)
    pub to:   (NodeId, usize),   // (target node, input port)
    pub tag:  TensorType,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
```

### 3.4 Builder API

```rust
let mut builder = GraphBuilder::new();

builder.register_backend("cpu", cpu_backend);
builder.register_backend("cuda:0", cuda_backend);
builder.register_backend("onnx:cpu", onnx_backend);

let t_input = TensorType {
    dtype: DType::F32,
    shape: vec![Dim::Dynamic(Some("batch".into())), Dim::Static(3),
                Dim::Static(224), Dim::Static(224)],
    layout: Layout::NCHW,
    dim_names: Some(vec!["batch".into(), "channels".into(),
                         "height".into(), "width".into()]),
};
let t_features = TensorType {
    dtype: DType::F32,
    shape: vec![Dim::Dynamic(Some("batch".into())), Dim::Static(1000)],
    layout: Layout::RowMajor,
    dim_names: Some(vec!["batch".into(), "classes".into()]),
};

let graph = builder
    // Load raw image data on CPU.
    .add_node("load")
        .device("cpu")
        .compute(load_images_kernel)
        .output(0, t_input.clone())
        .done()

    // Run a conv2d on the GPU via a hand-written CUDA kernel.
    .add_node("conv")
        .device("cuda:0")
        .ml_op(MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3], stride: [1, 1],
            padding: [1, 1], dilation: [1, 1], groups: 1,
        }))
        .input(0, t_input.clone())
        .output(0, t_input.clone())   // same shape after padding
        .done()

    // Run a full ONNX model for classification.
    .add_node("classify")
        .device("onnx:cpu")
        .ml_model(MlModelDescriptor {
            format:  "onnx".into(),
            source:  ModelSource::File("models/resnet50.onnx".into()),
            inputs:  vec![("input".into(), t_input.clone())],
            outputs: vec![("output".into(), t_features.clone())],
        })
        .input(0, t_input.clone())
        .output(0, t_features.clone())
        .done()

    // Print results on CPU.
    .add_node("print")
        .device("cpu")
        .compute(print_results_kernel)
        .input(0, t_features.clone())
        .done()

    .edge("load", 0, "conv", 0)?
    .edge("conv", 0, "classify", 0)?
    .edge("classify", 0, "print", 0)?
    .build()?;
```

---

## 4. The Backend trait (unified, capability-based)

A single trait that all backends implement — whether they manage raw device
memory (CUDA, OpenCL) or handle memory internally (ONNX Runtime, libtorch).
The `capabilities()` method tells the executor how to interact with this
backend.

```
src/backends/
  mod.rs             -- Backend trait, DeviceId, BackendCaps, BackendRegistry
```

```rust
/// Identifies a backend instance at runtime.
/// Examples: "cpu", "cuda:0", "opencl:1", "onnx:cpu", "libtorch:cuda:0",
///           "fpga:my-bitstream"
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId(pub String);

/// Declares how the executor should interact with this backend.
pub struct BackendCaps {
    /// Does the engine manage memory for this backend (alloc, upload,
    /// download), or does the backend handle it internally?
    pub memory: MemoryModel,

    /// Which node kinds this backend can execute.
    pub supported_kinds: Vec<NodeKindTag>,
}

pub enum MemoryModel {
    /// The engine calls alloc/upload/download explicitly.
    /// Used by: CPU, CUDA, OpenCL, Vulkan, wgpu, FPGA.
    Explicit,

    /// The backend manages its own memory. The engine passes raw
    /// host bytes in and gets raw host bytes out.
    /// Used by: ONNX Runtime, libtorch, TFLite, candle, burn.
    Managed,
}

/// Tags for what kinds of nodes a backend supports (without carrying
/// the full node payload).
pub enum NodeKindTag {
    Compute,
    MlOp,
    MlModel,
}

/// A handle to memory allocated by an explicit-memory backend.
/// Opaque to the engine.
pub trait DeviceBuffer: Send + Sync {
    fn size_bytes(&self) -> usize;
    fn device_id(&self) -> &DeviceId;
}

/// Backend-specific kernel/program description. Each backend defines
/// its own concrete type and downcasts from `dyn KernelDescriptor`.
pub trait KernelDescriptor: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// The unified backend interface.
pub trait Backend: Send + Sync {
    /// Human-readable name ("cuda", "onnx", "libtorch", "cpu", ...).
    fn name(&self) -> &str;

    /// Device identifier for this backend instance.
    fn device_id(&self) -> &DeviceId;

    /// What this backend can do.
    fn capabilities(&self) -> BackendCaps;

    // ── Explicit-memory operations ──────────────────────────────
    // These are called only when capabilities().memory == Explicit.
    // Backends with Managed memory may return BackendError::NotApplicable.

    /// Allocate a zeroed buffer of `size_bytes` on this device.
    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError>;

    /// Copy host memory → device buffer.
    fn upload(&self, host: &[u8], dst: &dyn DeviceBuffer) -> Result<(), BackendError>;

    /// Copy device buffer → host memory.
    fn download(&self, src: &dyn DeviceBuffer, host: &mut [u8]) -> Result<(), BackendError>;

    // ── Dispatch ────────────────────────────────────────────────

    /// Execute a raw compute kernel.
    /// Called for NodeKind::Compute nodes.
    fn dispatch_compute(
        &self,
        desc:    &dyn KernelDescriptor,
        inputs:  &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedNodeKind)
    }

    /// Execute a primitive ML operation.
    /// Called for NodeKind::MlOp nodes.
    fn dispatch_ml_op(
        &self,
        op:      &MlOp,
        inputs:  &[&[u8]],     // host-side tensor data
        outputs: &mut [Vec<u8>],
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedNodeKind)
    }

    /// Run whole-model inference.
    /// Called for NodeKind::MlModel nodes.
    fn dispatch_ml_model(
        &self,
        desc:    &MlModelDescriptor,
        inputs:  &[&[u8]],     // host-side tensor data
        outputs: &mut [Vec<u8>],
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedNodeKind)
    }
}
```

### 4.1 Future consideration: splitting the Backend trait

The current unified `Backend` trait spans three distinct abstraction layers:

1. **Low-level compute** — explicit device memory management (`alloc`,
   `upload`, `download`) and raw kernel dispatch (`dispatch_compute`).
2. **Mid-level ML ops** — primitive operation dispatch (`dispatch_ml_op`)
   where the backend maps a curated op to its own implementation (cuBLAS,
   cuDNN, hand-written kernels, etc.).
3. **High-level inference** — whole-model execution (`dispatch_ml_model`)
   where the backend owns the runtime session and memory entirely.

As the codebase grows, these responsibilities may pull in different
directions: a CUDA backend only needs layer 1 (and optionally 2), an ONNX
Runtime backend only needs layer 3, and a framework like candle or burn may
sit at layer 2. Forcing every backend to carry default-returning stubs for
layers it does not participate in adds surface area and makes the trait
harder to reason about.

If this tension becomes concrete — for example, when a second managed-memory
backend is added, or when the `dispatch_ml_op` surface grows beyond a
handful of operations — consider splitting `Backend` into layered traits:

```
ComputeBackend        — alloc, upload, download, dispatch_compute
MlOpBackend           — dispatch_ml_op
InferenceBackend      — dispatch_ml_model
```

Each backend would implement only the traits matching its capabilities, and
the executor would accept `&dyn ComputeBackend`, `&dyn MlOpBackend`, or
`&dyn InferenceBackend` depending on the `NodeKind` being dispatched. The
existing `BackendCaps` / `NodeKindTag` mechanism could be replaced by trait
bounds, making unsupported operations a compile-time error rather than a
runtime `Err(UnsupportedNodeKind)`.

**Do not split preemptively.** The unified trait is simpler while only one or
two backends exist. Split when the cost of the monolithic trait (unused
default stubs, confusing implementor experience, mixed memory models in one
trait) outweighs the cost of the extra trait hierarchy.

---

**How the executor uses capabilities:**

| `MemoryModel` | Before dispatch                     | Dispatch call                           | After dispatch                  |
|---------------|-------------------------------------|-----------------------------------------|---------------------------------|
| `Explicit`    | BufferManager allocs + uploads      | `dispatch_compute(DeviceBuffers)`       | BufferManager downloads if needed |
| `Managed`     | BufferManager serialises to `&[u8]` | `dispatch_ml_op/model(&[u8], Vec<u8>)` | BufferManager deserialises output |

---

## 5. Execution layer

### 5.1 Scheduler

```
src/execution/
  scheduler/
    mod.rs
    topo.rs          -- topological sort (Kahn's algorithm)
```

Produces an ordered list of `NodeId`s from the graph. Pure function over the
graph IR — no backend awareness.

### 5.2 Buffer manager

```
src/execution/
  buffer/
    mod.rs
    manager.rs       -- BufferManager
    location.rs      -- LocationTracker
```

```rust
/// Tracks a logical tensor and its physical copies.
pub struct ManagedBuffer {
    pub tensor_type: TensorType,
    /// Authoritative host-side copy (always present).
    pub host:        Vec<u8>,
    /// Physical device copies, keyed by DeviceId.
    /// Only populated for Explicit-memory backends.
    pub copies:      HashMap<DeviceId, (Box<dyn DeviceBuffer>, bool /*dirty*/)>,
}
```

The buffer manager's logic branches on the target backend's `MemoryModel`:

- **Explicit:** Allocate device buffers, upload before dispatch, download
  after dispatch if a downstream node needs host data.
- **Managed:** Ensure host bytes are up-to-date, pass `&[u8]` slices to the
  backend, collect `Vec<u8>` output. No device buffers involved.

### 5.3 Executor

```
src/execution/
  executor.rs        -- Executor run loop
```

```
Executor::run(graph, backends):
  1.  Validate graph (cycles, type mismatches, missing backends,
      backend capability vs node kind).
  2.  Schedule: topologically sort nodes.
  3.  For each node in order:
        a.  Resolve input buffers from upstream outputs.
        b.  Match on backend capabilities:
              Explicit memory:
                - BufferManager: ensure inputs are on the target device.
                - Allocate output device buffers.
                - Call backend.dispatch_compute(desc, dev_inputs, dev_outputs).
                - Register output buffers with BufferManager.
              Managed memory:
                - BufferManager: ensure inputs are on host.
                - Serialise inputs to &[u8].
                - Call backend.dispatch_ml_op/model(inputs, outputs).
                - Store output Vec<u8> in BufferManager as host buffers.
        c.  Mark outputs as available for downstream consumers.
  4.  Return final output buffers.
```

---

## 6. Backend layer — Pluggable implementations

All backends implement the unified `Backend` trait. They are grouped by
nature for code organisation but share the same interface.

```
src/backends/
  mod.rs                 -- Backend trait, BackendRegistry, re-exports

  compute/               -- Explicit-memory, raw-kernel backends
    mod.rs
    cpu.rs               -- CpuBackend              (always available)
    cuda.rs              -- CudaBackend              (feature = "cuda")
    opencl.rs            -- OpenClBackend            (feature = "opencl")
    vulkan.rs            -- VulkanBackend            (feature = "vulkan")
    wgpu.rs              -- WgpuBackend              (feature = "wgpu")
    fpga.rs              -- FpgaBackend              (feature = "fpga")

  ml/                    -- Managed-memory, ML runtime backends
    mod.rs
    onnx.rs              -- OnnxBackend              (feature = "onnx")
    torch.rs             -- TorchBackend             (feature = "torch")
    tflite.rs            -- TfLiteBackend            (feature = "tflite")
    candle.rs            -- CandleBackend            (feature = "candle")
    burn.rs              -- BurnBackend              (feature = "burn")
```

### 6.1 Compute backend example: CUDA

```rust
/// CUDA-specific kernel descriptor.
pub struct CudaKernelDesc {
    pub ptx:      String,
    pub function: String,
    pub grid:     [u32; 3],
    pub block:    [u32; 3],
}

impl KernelDescriptor for CudaKernelDesc {
    fn as_any(&self) -> &dyn Any { self }
}

pub struct CudaBackend {
    device:         Arc<CudaDevice>,
    loaded_modules: HashMap<String, ()>,
}

impl Backend for CudaBackend {
    fn name(&self) -> &str { "cuda" }
    fn device_id(&self) -> &DeviceId { &DeviceId("cuda:0".into()) }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            memory: MemoryModel::Explicit,
            supported_kinds: vec![NodeKindTag::Compute, NodeKindTag::MlOp],
        }
    }

    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> { ... }
    fn upload(&self, host: &[u8], dst: &dyn DeviceBuffer) -> Result<(), BackendError> { ... }
    fn download(&self, src: &dyn DeviceBuffer, host: &mut [u8]) -> Result<(), BackendError> { ... }

    fn dispatch_compute(&self, desc, inputs, outputs) -> Result<(), BackendError> {
        let cuda_desc = desc.as_any().downcast_ref::<CudaKernelDesc>()
            .ok_or(BackendError::UnsupportedKernel)?;
        // load PTX if not cached, set up launch config, launch kernel
        ...
    }

    fn dispatch_ml_op(&self, op, inputs, outputs) -> Result<(), BackendError> {
        // For ops like MatMul, Conv2d — could use cuBLAS, cuDNN, or
        // hand-written CUDA kernels.
        match op {
            MlOp::MatMul(params) => { /* cuBLAS sgemm */ }
            MlOp::Conv2d(params) => { /* cuDNN conv forward */ }
            MlOp::Relu            => { /* element-wise kernel */ }
            _ => Err(BackendError::UnsupportedOp),
        }
    }
}
```

### 6.2 ML runtime backend example: ONNX Runtime

```rust
pub struct OnnxBackend {
    session: ort::Session,
}

impl Backend for OnnxBackend {
    fn name(&self) -> &str { "onnx" }
    fn device_id(&self) -> &DeviceId { &DeviceId("onnx:cpu".into()) }
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            memory: MemoryModel::Managed,
            supported_kinds: vec![NodeKindTag::MlModel, NodeKindTag::MlOp],
        }
    }

    // Explicit memory ops — not applicable.
    fn alloc(&self, _: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        Err(BackendError::NotApplicable)
    }
    fn upload(&self, _: &[u8], _: &dyn DeviceBuffer) -> Result<(), BackendError> {
        Err(BackendError::NotApplicable)
    }
    fn download(&self, _: &dyn DeviceBuffer, _: &mut [u8]) -> Result<(), BackendError> {
        Err(BackendError::NotApplicable)
    }

    fn dispatch_ml_model(&self, desc, inputs, outputs) -> Result<(), BackendError> {
        // Wrap input &[u8] slices as ORT tensors.
        // Run the ONNX session.
        // Copy output tensors into Vec<u8>.
        ...
    }

    fn dispatch_ml_op(&self, op, inputs, outputs) -> Result<(), BackendError> {
        // Build a single-op ONNX graph on the fly, or use ORT's
        // operator API. Useful for running individual ops like MatMul
        // through ORT without a full model.
        ...
    }
}
```

---

## 7. Data flow example — mixed pipeline

```
 User code                    Engine internals
 ─────────                    ────────────────

 builder.build()  ─────────>  Graph IR
                                      │
 executor.run()   ─────────>  Scheduler: [load, preprocess, classify, print]
                                      │
                              ┌───────┴────────┐
                              ▼                │
                        ┌───────────┐          │
                        │  load     │          │
                        │  (cpu)    │ Compute  │
                        │  Explicit │          │
                        └─────┬─────┘          │
                              │ host bytes     │
                              ▼                │
                        ┌───────────┐          │
                        │ preprocess│          │
      auto-transfer     │ (cuda:0)  │ MlOp:    │
      host → device     │ Explicit  │ Conv2d   │
                        └─────┬─────┘          │
                              │ device buf     │
                              ▼                │
                       BufferManager:          │
                       download to host        │
                              │                │
                              ▼                │
                        ┌───────────┐          │
                        │ classify  │          │
                        │ (onnx:cpu)│ MlModel  │
                        │ Managed   │          │
                        └─────┬─────┘          │
                              │ host bytes     │
                              ▼                │
                        ┌───────────┐          │
                        │  print    │          │
                        │  (cpu)    │ Compute  │
                        │  Explicit │          │
                        └───────────┘          │
```

---

## 8. Source tree

The repository is a **Cargo workspace** with four member crates. All crates
are rooted directly in the repository root. There is a single `Cargo.lock` at
the workspace root.

```
Cargo.toml                    -- workspace manifest (resolver = "2")
Cargo.lock

core/                         -- crate: graph-core
  Cargo.toml
  src/
    lib.rs                    -- public API re-exports
    types/
      mod.rs
      device_id.rs            -- DeviceId, DeviceIdError
      dtype.rs                -- DType
      dim.rs                  -- Dim (Static / Dynamic)
      tensor_type.rs          -- TensorType
      layout.rs               -- Layout
      shape/
        mod.rs                -- Shape, ShapeError, strides, reshape
        ops.rs                -- broadcasting, compatibility
    ops/
      mod.rs                  -- MlOp enum, MlOpError
      params.rs               -- per-op parameter structs

backends/                     -- crate: backends
  Cargo.toml
  src/
    lib.rs                    -- Backend trait, BackendError, KernelDescriptor,
                              -- pub re-exports of DeviceId / DeviceIdError
    ml/
      mod.rs                  -- ML runtime backend stubs / helpers

backends-cuda/                -- crate: backends-cuda
  Cargo.toml
  build.rs                    -- emits CUDA linker search paths
  compile-kernel.sh           -- compiles kernel.cu → kernel.ptx via NVCC
  kernel.cu                   -- CUDA C source for the demo kernel
  kernel.ptx                  -- pre-compiled PTX (checked in)
  src/
    lib.rs                    -- CudaBackend, CudaBuffer, CudaKernelDesc

runtime/                      -- crate: runtime
  Cargo.toml
  src/
    lib.rs                    -- run_kernel convenience API + unit tests
    main.rs                   -- binary: demo  (cfg-gated for tarpaulin)
  tests/
    common/
      mod.rs                  -- shared mock infrastructure
    run_kernel_toy.rs         -- integration tests for run_kernel
    type_system_toy.rs        -- integration tests for graph-core types
```

**Dependency graph:**

```
graph-core
    ↑
backends      (depends on graph-core)
    ↑
backends-cuda (depends on backends)
    ↑
runtime       (depends on graph-core + backends + backends-cuda)
```

**Key invariant:** `graph-core` is backend-agnostic — it depends only on
`std`, `thiserror`, and `bytemuck`. All CUDA and hardware-specific code lives
in `backends-cuda`. `unsafe` is confined to backend crates.

---

## 9. Key design decisions & rationale

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Unified `Backend` trait with capability flags** | Avoids two parallel trait hierarchies. The executor branches on `MemoryModel` — one code path for explicit-memory backends, one for managed-memory runtimes. A backend can support multiple node kinds (e.g. CUDA handles both `Compute` and `MlOp` via cuBLAS/cuDNN). |
| 2 | **`KernelDescriptor` is a trait, not an enum** | Open for extension. Each backend defines its own descriptor struct and downcasts. Adding a new backend never touches core code. |
| 3 | **Rich `TensorType` with dynamic dims and layouts** | Enables graph-build-time validation of ML pipelines. Dynamic dims with symbolic names ensure consistency across the graph (all "batch" dims resolve to the same value at runtime). |
| 4 | **Curated `MlOp` enum + `Custom` escape hatch** | Gives the engine a shared vocabulary for common ML ops so multiple backends can support the same op. The `Custom` variant ensures the catalog is never a bottleneck. |
| 5 | **`MlModel` nodes with `MlModelDescriptor`** | Allows wrapping entire pre-trained models (ONNX, TorchScript, TFLite) as single opaque nodes. The backend handles loading, session management, and execution — the engine just routes data. |
| 6 | **Managed-memory backends use `&[u8]` / `Vec<u8>`** | ML runtimes manage their own device memory. Passing host bytes avoids forcing them into the engine's buffer management. The engine serialises/deserialises at the boundary. |
| 7 | **BufferManager branches on `MemoryModel`** | Explicit backends get full alloc/upload/download lifecycle. Managed backends get pass-through of host bytes. Same executor loop, different data handling. |
| 8 | **Core and Execution layers have zero backend deps** | Compiles and tests without any hardware SDK or ML framework installed. |
| 9 | **All backends are feature-gated** | `cargo build` gives CPU-only. `--features cuda,onnx` adds CUDA + ONNX. Users never pay for backends they don't use. |
| 10 | **`unsafe` confined to backend implementations** | All FFI (CUDA, OpenCL, Vulkan, libtorch C++) lives inside `backends/`. The core engine is 100% safe Rust. |
| 11 | **Graph is immutable after `build()`** | Simplifies scheduling — no mutations during execution. |
| 12 | **DAG scheduling first, streaming later** | Topological sort is correct for acyclic graphs. Streaming/cycles layer on top without changing core abstractions. |

---

## 10. Extension points (not in initial scope)

| Extension | How the architecture supports it |
|-----------|----------------------------------|
| **Multi-device** | `BackendRegistry` maps multiple `DeviceId`s. Scheduler assigns nodes to devices. BufferManager handles cross-device transfers. |
| **Async / pipelined execution** | Executor swaps blocking dispatch for async. Backends expose `dispatch_*_async` returning a future. |
| **Streaming / cycles** | Replace topo-sort with a readiness-queue that re-enqueues nodes whose inputs refresh. |
| **Config-file graphs** | YAML/JSON/TOML front-end constructs a `Graph` via the builder API. |
| **Profiling** | Instrument executor with timing callbacks around dispatch and transfer. |
| **Device-to-device transfer** | Add `Backend::transfer(src, dst, buffer)` for direct peer copies (CUDA P2P, DMA). |
| **Training** | `MlOp` nodes gain a `backward()` path. The scheduler runs the graph forward, then in reverse for gradient computation. |
| **Graph optimisation** | Fusion passes (merge adjacent ops), constant folding, layout transposition insertion — all operate on the immutable `Graph` IR and produce a new optimised `Graph`. |
| **Subgraphs** | A node can contain a nested `Graph`, enabling hierarchical composition and reuse. |

---

## 11. Dependency budget

| Crate          | Layer     | Purpose                          | Required |
|----------------|-----------|----------------------------------|----------|
| `thiserror`    | core      | Ergonomic error types            | yes      |
| `log`          | execution | Structured logging               | yes      |
| `cudarc`       | backends  | CUDA driver API                  | feature  |
| `opencl3`      | backends  | OpenCL bindings                  | feature  |
| `vulkano`/`ash`| backends  | Vulkan compute                   | feature  |
| `wgpu`         | backends  | WebGPU compute                   | feature  |
| `ort`          | backends  | ONNX Runtime bindings            | feature  |
| `tch`          | backends  | PyTorch (libtorch) bindings      | feature  |
| `tflitec`      | backends  | TensorFlow Lite bindings         | feature  |
| `candle-core`  | backends  | Rust-native ML (Hugging Face)    | feature  |
| `burn`         | backends  | Rust-native ML framework         | feature  |

Core and execution layers depend only on `std`, `thiserror`, and `log`. All
hardware and ML framework crates are behind Cargo features in the backend layer.
