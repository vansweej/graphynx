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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    // ── BackendError ─────────────────────────────────────────────────────

    #[test]
    fn backend_error_device_display() {
        let err = BackendError::Device("GPU overheated".to_string());
        assert_eq!(format!("{err}"), "Device error: GPU overheated");
    }

    #[test]
    fn backend_error_invalid_kernel_display() {
        let err = BackendError::InvalidKernel("bad PTX".to_string());
        assert_eq!(format!("{err}"), "Invalid kernel descriptor: bad PTX");
    }

    #[test]
    fn backend_error_buffer_display() {
        let err = BackendError::Buffer("out of memory".to_string());
        assert_eq!(format!("{err}"), "Buffer error: out of memory");
    }

    #[test]
    fn backend_error_not_applicable_display() {
        let err = BackendError::NotApplicable;
        assert_eq!(
            format!("{err}"),
            "Operation not applicable for this backend"
        );
    }

    #[test]
    fn backend_error_unsupported_node_kind_display() {
        let err = BackendError::UnsupportedNodeKind;
        assert_eq!(format!("{err}"), "Unsupported node kind");
    }

    #[test]
    fn backend_error_unsupported_op_display() {
        let err = BackendError::UnsupportedOp;
        assert_eq!(format!("{err}"), "Unsupported ML op");
    }

    #[test]
    fn backend_error_debug_format() {
        let err = BackendError::Device("test".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Device"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn backend_error_implements_std_error() {
        let err = BackendError::Device("test".to_string());
        // Verify the error can be used as &dyn std::error::Error.
        let _dyn_err: &dyn std::error::Error = &err;
    }

    // ── DeviceId ─────────────────────────────────────────────────────────

    #[test]
    fn device_id_new_from_str() {
        let id = DeviceId::new("cuda:0");
        assert_eq!(id.0, "cuda:0");
    }

    #[test]
    fn device_id_new_from_string() {
        let id = DeviceId::new(String::from("opencl:1"));
        assert_eq!(id.0, "opencl:1");
    }

    #[test]
    fn device_id_display() {
        let id = DeviceId::new("cuda:0");
        assert_eq!(format!("{id}"), "cuda:0");
    }

    #[test]
    fn device_id_debug() {
        let id = DeviceId::new("cpu");
        let debug = format!("{id:?}");
        assert!(debug.contains("cpu"));
    }

    #[test]
    fn device_id_clone() {
        let id = DeviceId::new("cuda:0");
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn device_id_equality() {
        let a = DeviceId::new("cuda:0");
        let b = DeviceId::new("cuda:0");
        let c = DeviceId::new("cuda:1");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn device_id_hash_as_map_key() {
        let mut map: HashMap<DeviceId, &str> = HashMap::new();
        map.insert(DeviceId::new("cuda:0"), "gpu0");
        map.insert(DeviceId::new("cpu"), "host");

        assert_eq!(map.get(&DeviceId::new("cuda:0")), Some(&"gpu0"));
        assert_eq!(map.get(&DeviceId::new("cpu")), Some(&"host"));
        assert_eq!(map.get(&DeviceId::new("cuda:1")), None);
    }

    #[test]
    fn device_id_empty_string() {
        let id = DeviceId::new("");
        assert_eq!(format!("{id}"), "");
    }

    // ── NodeKindTag ──────────────────────────────────────────────────────

    #[test]
    fn node_kind_tag_equality() {
        assert_eq!(NodeKindTag::Compute, NodeKindTag::Compute);
        assert_eq!(NodeKindTag::MlOp, NodeKindTag::MlOp);
        assert_eq!(NodeKindTag::MlModel, NodeKindTag::MlModel);
        assert_ne!(NodeKindTag::Compute, NodeKindTag::MlOp);
        assert_ne!(NodeKindTag::MlOp, NodeKindTag::MlModel);
    }

    #[test]
    fn node_kind_tag_clone() {
        let tag = NodeKindTag::Compute;
        let cloned = tag.clone();
        assert_eq!(tag, cloned);
    }

    #[test]
    fn node_kind_tag_debug() {
        assert_eq!(format!("{:?}", NodeKindTag::Compute), "Compute");
        assert_eq!(format!("{:?}", NodeKindTag::MlOp), "MlOp");
        assert_eq!(format!("{:?}", NodeKindTag::MlModel), "MlModel");
    }

    // ── BackendCaps & MemoryModel ────────────────────────────────────────

    #[test]
    fn backend_caps_explicit_memory() {
        let caps = BackendCaps {
            memory: MemoryModel::Explicit,
            supported_kinds: vec![NodeKindTag::Compute],
        };
        assert!(matches!(caps.memory, MemoryModel::Explicit));
        assert_eq!(caps.supported_kinds.len(), 1);
        assert_eq!(caps.supported_kinds[0], NodeKindTag::Compute);
    }

    #[test]
    fn backend_caps_managed_memory() {
        let caps = BackendCaps {
            memory: MemoryModel::Managed,
            supported_kinds: vec![NodeKindTag::MlOp, NodeKindTag::MlModel],
        };
        assert!(matches!(caps.memory, MemoryModel::Managed));
        assert_eq!(caps.supported_kinds.len(), 2);
    }

    #[test]
    fn backend_caps_empty_kinds() {
        let caps = BackendCaps {
            memory: MemoryModel::Explicit,
            supported_kinds: vec![],
        };
        assert!(caps.supported_kinds.is_empty());
    }

    // ── Mock backend for testing default trait methods ────────────────────

    /// A minimal mock backend that only implements the required trait methods.
    /// Uses default implementations for `dispatch_compute`, `dispatch_ml_op`,
    /// and `dispatch_ml_model` to verify they return `UnsupportedNodeKind`.
    struct MockManagedBackend {
        device_id: DeviceId,
    }

    impl MockManagedBackend {
        fn new() -> Self {
            Self {
                device_id: DeviceId::new("mock:0"),
            }
        }
    }

    /// A simple in-memory buffer for testing the `DeviceBuffer` trait.
    struct MockBuffer {
        data: Vec<u8>,
        device_id: DeviceId,
    }

    impl MockBuffer {
        fn new(size: usize, device_id: DeviceId) -> Self {
            Self {
                data: vec![0u8; size],
                device_id,
            }
        }
    }

    impl DeviceBuffer for MockBuffer {
        fn size_bytes(&self) -> usize {
            self.data.len()
        }

        fn device_id(&self) -> &DeviceId {
            &self.device_id
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Backend for MockManagedBackend {
        fn name(&self) -> &str {
            "mock"
        }

        fn device_id(&self) -> &DeviceId {
            &self.device_id
        }

        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                memory: MemoryModel::Managed,
                supported_kinds: vec![],
            }
        }

        fn alloc(&self, _size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn upload(&self, _host: &[u8], _dst: &dyn DeviceBuffer) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn download(&self, _src: &dyn DeviceBuffer, _host: &mut [u8]) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }

        // dispatch_compute, dispatch_ml_op, dispatch_ml_model use defaults.
    }

    #[test]
    fn default_dispatch_compute_returns_unsupported() {
        let backend = MockManagedBackend::new();
        let desc = MockKernelDesc;
        let input_buf = MockBuffer::new(16, DeviceId::new("mock"));
        let mut output_buf = MockBuffer::new(16, DeviceId::new("mock"));
        let result = backend.dispatch_compute(
            &desc,
            &[&input_buf as &dyn DeviceBuffer],
            &mut [&mut output_buf],
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BackendError::UnsupportedNodeKind
        ));
    }

    #[test]
    fn default_dispatch_ml_op_returns_unsupported() {
        let backend = MockManagedBackend::new();
        let input: &[u8] = &[1, 2, 3];
        let mut output = vec![vec![0u8; 3]];
        let result = backend.dispatch_ml_op("relu", &[input], &mut output);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BackendError::UnsupportedNodeKind
        ));
    }

    #[test]
    fn default_dispatch_ml_model_returns_unsupported() {
        let backend = MockManagedBackend::new();
        let input: &[u8] = &[1, 2, 3];
        let mut output = vec![vec![0u8; 3]];
        let result = backend.dispatch_ml_model("resnet50", &[input], &mut output);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BackendError::UnsupportedNodeKind
        ));
    }

    // ── DeviceBuffer trait (via MockBuffer) ──────────────────────────────

    #[test]
    fn mock_buffer_size_bytes() {
        let buf = MockBuffer::new(64, DeviceId::new("test"));
        assert_eq!(buf.size_bytes(), 64);
    }

    #[test]
    fn mock_buffer_device_id() {
        let buf = MockBuffer::new(32, DeviceId::new("cuda:0"));
        assert_eq!(buf.device_id(), &DeviceId::new("cuda:0"));
    }

    #[test]
    fn mock_buffer_as_any_downcast() {
        let buf = MockBuffer::new(16, DeviceId::new("test"));
        let any_ref = buf.as_any();
        assert!(any_ref.downcast_ref::<MockBuffer>().is_some());
    }

    #[test]
    fn mock_buffer_as_any_mut_downcast() {
        let mut buf = MockBuffer::new(16, DeviceId::new("test"));
        let any_mut = buf.as_any_mut();
        assert!(any_mut.downcast_mut::<MockBuffer>().is_some());
    }

    #[test]
    fn mock_buffer_zero_size() {
        let buf = MockBuffer::new(0, DeviceId::new("test"));
        assert_eq!(buf.size_bytes(), 0);
    }

    // ── KernelDescriptor trait ───────────────────────────────────────────

    /// A minimal kernel descriptor for testing.
    struct MockKernelDesc;

    impl KernelDescriptor for MockKernelDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn kernel_descriptor_as_any_downcast() {
        let desc = MockKernelDesc;
        let any_ref: &dyn Any = desc.as_any();
        assert!(any_ref.downcast_ref::<MockKernelDesc>().is_some());
    }

    #[test]
    fn kernel_descriptor_trait_object() {
        let desc = MockKernelDesc;
        let trait_obj: &dyn KernelDescriptor = &desc;
        assert!(trait_obj
            .as_any()
            .downcast_ref::<MockKernelDesc>()
            .is_some());
    }

    // ── Backend trait (via MockManagedBackend) ───────────────────────────

    #[test]
    fn mock_backend_name() {
        let backend = MockManagedBackend::new();
        assert_eq!(backend.name(), "mock");
    }

    #[test]
    fn mock_backend_alloc_returns_not_applicable() {
        let backend = MockManagedBackend::new();
        let result = backend.alloc(64);
        assert!(result.is_err());
        match result {
            Err(BackendError::NotApplicable) => {} // expected
            Err(other) => panic!("expected NotApplicable, got {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    #[test]
    fn mock_backend_upload_returns_not_applicable() {
        let backend = MockManagedBackend::new();
        let buf = MockBuffer::new(16, DeviceId::new("mock"));
        let result = backend.upload(&[1, 2, 3], &buf);
        assert!(matches!(result.unwrap_err(), BackendError::NotApplicable));
    }

    #[test]
    fn mock_backend_download_returns_not_applicable() {
        let backend = MockManagedBackend::new();
        let buf = MockBuffer::new(16, DeviceId::new("mock"));
        let mut host = vec![0u8; 16];
        let result = backend.download(&buf, &mut host);
        assert!(matches!(result.unwrap_err(), BackendError::NotApplicable));
    }

    #[test]
    fn mock_backend_capabilities() {
        let backend = MockManagedBackend::new();
        let caps = backend.capabilities();
        assert!(matches!(caps.memory, MemoryModel::Managed));
        assert!(caps.supported_kinds.is_empty());
    }

    // ── Backend as trait object ──────────────────────────────────────────

    #[test]
    fn backend_can_be_used_as_trait_object() {
        let backend = MockManagedBackend::new();
        let trait_obj: &dyn Backend = &backend;
        assert_eq!(trait_obj.name(), "mock");
    }
}
