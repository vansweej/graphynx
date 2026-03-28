use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::backend::{
    Backend, BackendCaps, BackendError, DeviceBuffer, DeviceId, KernelDescriptor, MemoryModel,
    NodeKindTag,
};

// ── CudaBuffer ────────────────────────────────────────────────────────────────

/// A device-side byte buffer allocated by `CudaBackend`.
pub struct CudaBuffer {
    slice: CudaSlice<u8>,
    device_id: DeviceId,
}

impl DeviceBuffer for CudaBuffer {
    fn size_bytes(&self) -> usize {
        // CudaSlice<u8>: each element is exactly one byte.
        self.slice.len()
    }

    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        Self {
            slice: self.slice.clone(),
            device_id: self.device_id.clone(),
        }
    }
}

// ── KernelDescriptor impl ─────────────────────────────────────────────────────

/// Describes a CUDA PTX kernel and its launch geometry.
#[derive(Clone, Debug)]
pub struct CudaKernelDesc {
    /// PTX module name (used to look up the function with `get_func`).
    pub module_name: String,
    /// Name of the kernel function inside the PTX module.
    pub func_name: String,
    /// Grid dimensions `(x, y, z)`.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)`.
    pub block_dim: (u32, u32, u32),
}

impl CudaKernelDesc {
    /// Construct a descriptor targeting a kernel in the `"hello"` module.
    pub fn new(func_name: &str, grid_dim: [u32; 3], block_dim: [u32; 3]) -> Self {
        Self {
            module_name: "hello".to_string(),
            func_name: func_name.to_string(),
            grid_dim: (grid_dim[0], grid_dim[1], grid_dim[2]),
            block_dim: (block_dim[0], block_dim[1], block_dim[2]),
        }
    }
}

impl KernelDescriptor for CudaKernelDesc {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── CudaBackend ───────────────────────────────────────────────────────────────

/// CUDA compute backend.
///
/// Wraps a single CUDA device and a set of pre-loaded PTX modules.
/// Implements `MemoryModel::Explicit` — the executor manages allocation,
/// upload, and download of device buffers.
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    device_id: DeviceId,
}

impl CudaBackend {
    /// Open CUDA device `device_ordinal`, load `ptx` as module `module_name`,
    /// and register `"hello_kernel"` as the exported function.
    pub fn new(device_ordinal: usize, ptx: &str, module_name: &str) -> Result<Self, BackendError> {
        let device = CudaDevice::new(device_ordinal)
            .map_err(|e| BackendError::Device(format!("CUDA: {e}")))?;

        device
            .load_ptx(ptx.into(), module_name, &["hello_kernel"])
            .map_err(|e| BackendError::Device(format!("CUDA: {e}")))?;

        let device_id = DeviceId::new(format!("cuda:{device_ordinal}"));

        Ok(Self { device, device_id })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            memory: MemoryModel::Explicit,
            // MlOp support (cuBLAS, cuDNN) is planned but not yet implemented.
            supported_kinds: vec![NodeKindTag::Compute],
        }
    }

    // ── Explicit-memory operations ────────────────────────────────────────

    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let slice: CudaSlice<u8> = self
            .device
            .alloc_zeros(size_bytes)
            .map_err(|e| BackendError::Buffer(format!("CUDA alloc: {e}")))?;

        Ok(Box::new(CudaBuffer {
            slice,
            device_id: self.device_id.clone(),
        }))
    }

    fn upload(&self, host: &[u8], dst: &dyn DeviceBuffer) -> Result<(), BackendError> {
        let cuda_dst = dst
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::Buffer("upload: dst is not a CudaBuffer".to_string()))?;

        // Clone the slice handle to get a mutable reference into the same
        // device allocation — htod_sync_copy_into requires &mut CudaSlice.
        let mut dst_slice = cuda_dst.slice.clone();
        self.device
            .htod_sync_copy_into(host, &mut dst_slice)
            .map_err(|e| BackendError::Device(format!("CUDA upload: {e}")))?;

        Ok(())
    }

    fn download(&self, src: &dyn DeviceBuffer, host: &mut [u8]) -> Result<(), BackendError> {
        let cuda_src = src
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::Buffer("download: src is not a CudaBuffer".to_string()))?;

        let result: Vec<u8> = self
            .device
            .dtoh_sync_copy(&cuda_src.slice)
            .map_err(|e| BackendError::Device(format!("CUDA download: {e}")))?;

        host.copy_from_slice(&result);
        Ok(())
    }

    // ── Dispatch ──────────────────────────────────────────────────────────

    fn dispatch_compute(
        &self,
        desc: &dyn KernelDescriptor,
        inputs: &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
    ) -> Result<(), BackendError> {
        let kernel_desc = desc
            .as_any()
            .downcast_ref::<CudaKernelDesc>()
            .ok_or_else(|| {
                BackendError::InvalidKernel("dispatch_compute: not a CudaKernelDesc".to_string())
            })?;

        let kernel = self
            .device
            .get_func(&kernel_desc.module_name, &kernel_desc.func_name)
            .ok_or_else(|| {
                BackendError::InvalidKernel(format!(
                    "CUDA: function '{}' not found in module '{}'",
                    kernel_desc.func_name, kernel_desc.module_name
                ))
            })?;

        let cfg = LaunchConfig {
            grid_dim: kernel_desc.grid_dim,
            block_dim: kernel_desc.block_dim,
            shared_mem_bytes: 0,
        };

        // Only single-input / single-output kernels are supported for now.
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(BackendError::InvalidKernel(
                "CUDA: only single input/output kernels are currently supported".to_string(),
            ));
        }

        let input = inputs[0]
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| {
                BackendError::InvalidKernel(
                    "dispatch_compute: input is not a CudaBuffer".to_string(),
                )
            })?;

        let output = outputs[0]
            .as_any_mut()
            .downcast_mut::<CudaBuffer>()
            .ok_or_else(|| {
                BackendError::InvalidKernel(
                    "dispatch_compute: output is not a CudaBuffer".to_string(),
                )
            })?;

        // Clone the output slice handle so we can pass it mutably to the
        // kernel while still holding `output` for the dtod_copy below.
        let mut output_slice = output.slice.clone();

        // Safety: kernel arguments match the PTX signature (const u8*, u8*).
        unsafe {
            kernel
                .launch(cfg, (&input.slice, &mut output_slice))
                .map_err(|e| BackendError::Device(format!("CUDA launch: {e}")))?;
        }

        // Copy the kernel result back into the canonical output buffer.
        self.device
            .dtod_copy(&output_slice, &mut output.slice)
            .map_err(|e| BackendError::Device(format!("CUDA dtod_copy: {e}")))?;

        Ok(())
    }
}
