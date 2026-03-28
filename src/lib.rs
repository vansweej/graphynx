//! Graphynx — graph-based runtime for heterogeneous CPU/GPU computation.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`backend`] | Core traits (`Backend`, `DeviceBuffer`, `KernelDescriptor`), `DeviceId`, `BackendError` |
//! | [`cuda_backend`] | CUDA implementation of `Backend` via `cudarc` |
//! | [`dtype`] | `DType` — scalar element type enum (`F32`, `I32`, …) |
//! | [`shape`] | `Shape` — validated tensor shape with broadcasting, reshape validation, and stride computation |
//! | [`tensor_type`] | `TensorType`, `Dim`, `Layout` — validated tensor metadata for graph edges |
//! | [`ml_op`] | `MlOp` — curated catalog of primitive ML operations and their parameter structs |
//!
//! # Quick start
//!
//! ```ignore
//! use graphynx::cuda_backend::{CudaBackend, CudaKernelDesc};
//!
//! let ptx = include_str!("../kernel.ptx");
//! let backend = CudaBackend::new(0, ptx, "hello")?;
//! let desc    = CudaKernelDesc::new("hello_kernel", [1, 1, 1], [10, 1, 1]);
//! let input   = vec![1i32, 2, 3, 4, 5];
//! let output: Vec<i32> = graphynx::run_kernel(&backend, &desc, &input)?;
//! ```

pub mod backend;
pub mod cuda_backend;
pub mod dtype;
pub mod ml_op;
pub mod shape;
pub mod tensor_type;

use bytemuck::Pod;
use log::debug;

use backend::{Backend, BackendError, KernelDescriptor};

/// Upload `input`, dispatch `desc`, download the result, and return it as `Vec<T>`.
///
/// This is a convenience wrapper over the explicit-memory backend operations.
/// It allocates a device input buffer, uploads the input bytes, allocates a
/// device output buffer, calls `dispatch_compute`, downloads the result into a
/// host `Vec<u8>`, and reinterprets the bytes as `Vec<T>`.
///
/// # Errors
/// Propagates any `BackendError` returned by `alloc`, `upload`, `dispatch_compute`,
/// or `download`.
pub fn run_kernel<T: Pod>(
    backend: &dyn Backend,
    desc: &dyn KernelDescriptor,
    input: &[T],
) -> Result<Vec<T>, BackendError> {
    let input_bytes: &[u8] = bytemuck::cast_slice(input);
    let output_size_bytes = input_bytes.len();

    debug!(
        "run_kernel: backend='{}', input_bytes={}, elements={}",
        backend.name(),
        input_bytes.len(),
        input.len()
    );

    // --- Input ---
    // Allocate a device buffer and upload the input bytes into it.
    let input_buf = backend.alloc(input_bytes.len())?;
    backend.upload(input_bytes, input_buf.as_ref())?;

    // --- Output ---
    // Allocate a zeroed output buffer of the same size.
    let mut output_buf = backend.alloc(output_size_bytes)?;

    backend.dispatch_compute(desc, &[input_buf.as_ref()], &mut [output_buf.as_mut()])?;

    // Download result into a host buffer of the exact right size.
    let mut output_bytes: Vec<u8> = vec![0u8; output_size_bytes];
    backend.download(output_buf.as_ref(), &mut output_bytes)?;

    let output: Vec<T> = bytemuck::cast_slice(&output_bytes).to_vec();

    debug!("run_kernel: completed, output_elements={}", output.len());
    Ok(output)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::cell::RefCell;

    use super::*;
    use backend::{BackendCaps, DeviceBuffer, DeviceId, MemoryModel, NodeKindTag};

    // ── Mock infrastructure ──────────────────────────────────────────────

    /// A simple in-memory buffer that simulates a device buffer on the host.
    struct MemBuffer {
        data: RefCell<Vec<u8>>,
        device_id: DeviceId,
    }

    impl MemBuffer {
        fn new(size: usize) -> Self {
            Self {
                data: RefCell::new(vec![0u8; size]),
                device_id: DeviceId::new("mem:0"),
            }
        }
    }

    // Safety: MemBuffer is only used in single-threaded tests.
    unsafe impl Send for MemBuffer {}
    unsafe impl Sync for MemBuffer {}

    impl DeviceBuffer for MemBuffer {
        fn size_bytes(&self) -> usize {
            self.data.borrow().len()
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

    /// A minimal kernel descriptor for tests.
    struct DoubleKernelDesc;

    impl KernelDescriptor for DoubleKernelDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// A mock backend that operates entirely in host memory.
    ///
    /// `dispatch_compute` doubles each `i32` element in the input and writes
    /// to the output, simulating the `hello_kernel` from `kernel.cu`.
    struct MemBackend {
        device_id: DeviceId,
    }

    impl MemBackend {
        fn new() -> Self {
            Self {
                device_id: DeviceId::new("mem:0"),
            }
        }
    }

    impl Backend for MemBackend {
        fn name(&self) -> &str {
            "mem"
        }

        fn device_id(&self) -> &DeviceId {
            &self.device_id
        }

        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                memory: MemoryModel::Explicit,
                supported_kinds: vec![NodeKindTag::Compute],
            }
        }

        fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Ok(Box::new(MemBuffer::new(size_bytes)))
        }

        fn upload(&self, host: &[u8], dst: &dyn DeviceBuffer) -> Result<(), BackendError> {
            let mem_dst = dst
                .as_any()
                .downcast_ref::<MemBuffer>()
                .ok_or_else(|| BackendError::Buffer("not a MemBuffer".to_string()))?;
            mem_dst.data.borrow_mut().copy_from_slice(host);
            Ok(())
        }

        fn download(&self, src: &dyn DeviceBuffer, host: &mut [u8]) -> Result<(), BackendError> {
            let mem_src = src
                .as_any()
                .downcast_ref::<MemBuffer>()
                .ok_or_else(|| BackendError::Buffer("not a MemBuffer".to_string()))?;
            host.copy_from_slice(&mem_src.data.borrow());
            Ok(())
        }

        fn dispatch_compute(
            &self,
            _desc: &dyn KernelDescriptor,
            inputs: &[&dyn DeviceBuffer],
            outputs: &mut [&mut dyn DeviceBuffer],
        ) -> Result<(), BackendError> {
            let input = inputs[0]
                .as_any()
                .downcast_ref::<MemBuffer>()
                .ok_or_else(|| BackendError::Buffer("not a MemBuffer".to_string()))?;

            let input_data = input.data.borrow();
            // Interpret as i32 and double each element.
            let ints: &[i32] = bytemuck::cast_slice(&input_data);
            let doubled: Vec<i32> = ints.iter().map(|x| x * 2).collect();
            let doubled_bytes: &[u8] = bytemuck::cast_slice(&doubled);

            let output = outputs[0]
                .as_any_mut()
                .downcast_mut::<MemBuffer>()
                .ok_or_else(|| BackendError::Buffer("not a MemBuffer".to_string()))?;
            output.data.borrow_mut().copy_from_slice(doubled_bytes);
            Ok(())
        }
    }

    /// A mock backend whose `alloc` always fails.
    struct FailAllocBackend {
        device_id: DeviceId,
    }

    impl FailAllocBackend {
        fn new() -> Self {
            Self {
                device_id: DeviceId::new("fail:0"),
            }
        }
    }

    impl Backend for FailAllocBackend {
        fn name(&self) -> &str {
            "fail"
        }

        fn device_id(&self) -> &DeviceId {
            &self.device_id
        }

        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                memory: MemoryModel::Explicit,
                supported_kinds: vec![],
            }
        }

        fn alloc(&self, _size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Err(BackendError::Buffer("alloc failed".to_string()))
        }

        fn upload(&self, _host: &[u8], _dst: &dyn DeviceBuffer) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn download(&self, _src: &dyn DeviceBuffer, _host: &mut [u8]) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }
    }

    // ── run_kernel tests ─────────────────────────────────────────────────

    #[test]
    fn run_kernel_doubles_i32_elements() {
        let backend = MemBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![1, 2, 3, 4, 5];
        let output = run_kernel(&backend, &desc, &input).expect("run_kernel should succeed");
        assert_eq!(output, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    #[should_panic(expected = "cast_slice")]
    fn run_kernel_empty_input_panics() {
        // Empty input produces an empty byte slice. bytemuck::cast_slice
        // panics on zero-length slices due to alignment checks when
        // converting the output back to &[T]. This documents the current
        // limitation.
        let backend = MemBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![];
        let _ = run_kernel(&backend, &desc, &input);
    }

    #[test]
    fn run_kernel_single_element() {
        let backend = MemBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![42];
        let output = run_kernel(&backend, &desc, &input).expect("run_kernel should succeed");
        assert_eq!(output, vec![84]);
    }

    #[test]
    fn run_kernel_preserves_negative_values() {
        let backend = MemBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![-5, -10, 0, 15];
        let output = run_kernel(&backend, &desc, &input).expect("run_kernel should succeed");
        assert_eq!(output, vec![-10, -20, 0, 30]);
    }

    #[test]
    fn run_kernel_output_length_matches_input() {
        let backend = MemBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let output = run_kernel(&backend, &desc, &input).expect("run_kernel should succeed");
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn run_kernel_alloc_failure_propagates() {
        let backend = FailAllocBackend::new();
        let desc = DoubleKernelDesc;
        let input: Vec<i32> = vec![1, 2, 3];
        let result = run_kernel(&backend, &desc, &input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BackendError::Buffer(_)));
    }

    #[test]
    fn run_kernel_uses_backend_trait_object() {
        let backend = MemBackend::new();
        let backend_ref: &dyn Backend = &backend;
        let desc: &dyn KernelDescriptor = &DoubleKernelDesc;
        let input: Vec<i32> = vec![100];
        let output = run_kernel(backend_ref, desc, &input).expect("run_kernel should succeed");
        assert_eq!(output, vec![200]);
    }
}
