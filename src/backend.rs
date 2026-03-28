use std::any::Any;

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that any backend implementation may return.
///
/// Variants are backend-agnostic — use the message string to carry
/// backend-specific context (e.g. "CUDA: invalid PTX").
#[derive(Debug, Error)]
pub enum BackendError {
    /// A device or driver-level error (CUDA, OpenCL, Vulkan, …).
    #[error("Device error: {0}")]
    Device(String),

    /// The supplied `KernelDescriptor` was not recognised or was malformed.
    #[error("Invalid kernel descriptor: {0}")]
    InvalidKernel(String),

    /// A buffer allocation or transfer error.
    #[error("Buffer error: {0}")]
    Buffer(String),

    /// The requested operation is not applicable for this backend's
    /// `MemoryModel` (e.g. calling `alloc` on a managed-memory backend).
    #[error("Operation not applicable for this backend")]
    NotApplicable,

    /// The backend does not support the requested `NodeKind`.
    #[error("Unsupported node kind")]
    UnsupportedNodeKind,

    /// The backend does not support the requested `MlOp` variant.
    #[error("Unsupported ML op")]
    UnsupportedOp,
}

// ── DeviceId ──────────────────────────────────────────────────────────────────

/// Identifies a backend instance at runtime.
///
/// By convention, use the form `"<backend>:<index>"` for hardware backends
/// and `"<runtime>:<device>"` for ML runtime backends.
///
/// # Examples
/// `"cpu"`, `"cuda:0"`, `"opencl:1"`, `"onnx:cpu"`, `"libtorch:cuda:0"`
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId(pub String);

impl DeviceId {
    /// Construct a `DeviceId` from any string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Capabilities ──────────────────────────────────────────────────────────────

/// Declares how the executor should interact with a backend.
///
/// The executor inspects `BackendCaps` at dispatch time to decide whether to
/// manage device memory explicitly or hand raw host bytes to the backend.
pub struct BackendCaps {
    /// Whether the engine manages memory for this backend, or whether the
    /// backend handles it internally.
    pub memory: MemoryModel,

    /// The node kinds this backend can execute.
    pub supported_kinds: Vec<NodeKindTag>,
}

/// How a backend handles tensor memory.
pub enum MemoryModel {
    /// The engine calls `alloc` / `upload` / `download` explicitly.
    ///
    /// Used by: CPU, CUDA, OpenCL, Vulkan, wgpu, FPGA.
    Explicit,

    /// The backend manages its own device memory. The engine passes raw host
    /// bytes in and receives raw host bytes out.
    ///
    /// Used by: ONNX Runtime, libtorch, TFLite, candle, burn.
    Managed,
}

/// Tags the kinds of graph nodes a backend can execute.
///
/// Used in `BackendCaps::supported_kinds` without carrying a full node payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum NodeKindTag {
    /// Raw compute kernel (CUDA PTX, SPIR-V, WGSL, native Rust, …).
    Compute,
    /// Primitive ML operation from the curated `MlOp` catalog.
    MlOp,
    /// Opaque pre-trained model (ONNX, TorchScript, TFLite, …).
    MlModel,
}

// ── DeviceBuffer ──────────────────────────────────────────────────────────────

/// A handle to memory allocated on a device by an explicit-memory backend.
///
/// The concrete type is defined by each backend and is opaque to the engine.
/// Backends recover their concrete buffer type via `as_any` / `as_any_mut`
/// downcasting inside their own dispatch implementations.
pub trait DeviceBuffer: Send + Sync {
    /// Size of the allocation in bytes.
    fn size_bytes(&self) -> usize;

    /// The device that owns this buffer.
    fn device_id(&self) -> &DeviceId;

    /// Returns `self` as `&dyn Any` to enable backend-internal downcasting.
    ///
    /// This is an implementation detail for backend use only — the engine
    /// does not call this method.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns `self` as `&mut dyn Any` to enable backend-internal downcasting.
    ///
    /// This is an implementation detail for backend use only — the engine
    /// does not call this method.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// ── KernelDescriptor ─────────────────────────────────────────────────────────

/// Backend-specific description of a compute kernel to execute.
///
/// Each backend defines its own concrete descriptor struct (e.g.
/// `CudaKernelDesc`) and downcasts from `&dyn KernelDescriptor` inside its
/// `dispatch_compute` implementation. The `Any` supertrait bound ensures the
/// concrete type is `'static`, which is required for safe downcasting.
pub trait KernelDescriptor: Any + Send + Sync {
    /// Returns `self` as `&dyn Any` to enable downcasting.
    fn as_any(&self) -> &dyn Any;
}

// ── Backend ───────────────────────────────────────────────────────────────────

/// The unified interface that all compute and ML-runtime backends implement.
///
/// A backend represents a single logical device or runtime (e.g. one CUDA GPU,
/// one ONNX Runtime session). The executor uses `capabilities()` to determine
/// which dispatch path and memory management strategy to apply.
///
/// # Explicit-memory backends (CUDA, CPU, OpenCL, …)
///
/// The executor calls `alloc`, `upload`, and `download` to manage device
/// buffers. `dispatch_compute` receives `&dyn DeviceBuffer` handles.
///
/// # Managed-memory backends (ONNX Runtime, libtorch, …)
///
/// The executor passes host bytes directly. `alloc`, `upload`, and `download`
/// should return `Err(BackendError::NotApplicable)`.
pub trait Backend: Send + Sync {
    /// Human-readable backend name (e.g. `"cuda"`, `"onnx"`, `"cpu"`).
    fn name(&self) -> &str;

    /// Device identifier for this backend instance (e.g. `"cuda:0"`).
    fn device_id(&self) -> &DeviceId;

    /// Declares what this backend supports and how the executor interacts
    /// with it.
    fn capabilities(&self) -> BackendCaps;

    // ── Explicit-memory operations ────────────────────────────────────────
    // Called only when `capabilities().memory == MemoryModel::Explicit`.
    // Managed-memory backends must return `Err(BackendError::NotApplicable)`.

    /// Allocate a zeroed device buffer of `size_bytes`.
    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError>;

    /// Copy `host` bytes into an existing device buffer `dst`.
    ///
    /// `dst` must have been allocated by this backend and
    /// `dst.size_bytes() >= host.len()`.
    fn upload(&self, host: &[u8], dst: &dyn DeviceBuffer) -> Result<(), BackendError>;

    /// Copy the contents of device buffer `src` into the caller-provided
    /// `host` slice.
    ///
    /// `host.len()` must equal `src.size_bytes()`.
    fn download(&self, src: &dyn DeviceBuffer, host: &mut [u8]) -> Result<(), BackendError>;

    // ── Dispatch ──────────────────────────────────────────────────────────

    /// Execute a raw compute kernel described by `desc`.
    ///
    /// Called for `NodeKind::Compute` nodes on explicit-memory backends.
    /// The default implementation returns `Err(BackendError::UnsupportedNodeKind)`.
    fn dispatch_compute(
        &self,
        desc: &dyn KernelDescriptor,
        inputs: &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
    ) -> Result<(), BackendError> {
        let _ = (desc, inputs, outputs);
        Err(BackendError::UnsupportedNodeKind)
    }

    /// Execute a primitive ML operation from the curated catalog.
    ///
    /// Called for `NodeKind::MlOp` nodes. `inputs` and `outputs` carry
    /// raw host-side tensor bytes (used by managed-memory backends).
    /// The default implementation returns `Err(BackendError::UnsupportedNodeKind)`.
    fn dispatch_ml_op(
        &self,
        op_name: &str,
        inputs: &[&[u8]],
        outputs: &mut [Vec<u8>],
    ) -> Result<(), BackendError> {
        let _ = (op_name, inputs, outputs);
        Err(BackendError::UnsupportedNodeKind)
    }

    /// Run whole-model inference for a pre-trained model.
    ///
    /// Called for `NodeKind::MlModel` nodes. `inputs` and `outputs` carry
    /// raw host-side tensor bytes. The default implementation returns
    /// `Err(BackendError::UnsupportedNodeKind)`.
    fn dispatch_ml_model(
        &self,
        model_name: &str,
        inputs: &[&[u8]],
        outputs: &mut [Vec<u8>],
    ) -> Result<(), BackendError> {
        let _ = (model_name, inputs, outputs);
        Err(BackendError::UnsupportedNodeKind)
    }
}
