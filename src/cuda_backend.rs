use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::backend::{Backend, BackendError, DeviceBuffer, KernelDescriptor};

pub struct CudaBuffer {
    slice: CudaSlice<u8>,
}

impl DeviceBuffer for CudaBuffer {
    fn size_bytes(&self) -> usize {
        self.slice.len() * std::mem::size_of::<u8>()
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaKernelDesc {
    pub module_name: String,
    pub func_name: String,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}

impl CudaKernelDesc {
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

pub struct CudaBackend {
    device: Arc<CudaDevice>,
}

impl CudaBackend {
    pub fn new(device_ordinal: usize, ptx: &str, module_name: &str) -> Result<Self, BackendError> {
        let device =
            CudaDevice::new(device_ordinal).map_err(|e| BackendError::Cuda(e.to_string()))?;

        device
            .load_ptx(ptx.into(), module_name, &["hello_kernel"])
            .map_err(|e| BackendError::Cuda(e.to_string()))?;

        Ok(Self { device })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let num_elements = size_bytes;
        let slice: CudaSlice<u8> = self
            .device
            .alloc_zeros(num_elements)
            .map_err(|e| BackendError::Buffer(e.to_string()))?;

        Ok(Box::new(CudaBuffer { slice }))
    }

    fn upload(&self, data: &[u8]) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let slice: CudaSlice<u8> = self
            .device
            .htod_sync_copy(data)
            .map_err(|e| BackendError::Cuda(e.to_string()))?;

        Ok(Box::new(CudaBuffer { slice }))
    }

    fn download(&self, buf: &dyn DeviceBuffer) -> Result<Vec<u8>, BackendError> {
        let cuda_buf = buf
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| BackendError::InvalidKernel("Not a CudaBuffer".to_string()))?;

        let result: Vec<u8> = self
            .device
            .dtoh_sync_copy(&cuda_buf.slice)
            .map_err(|e| BackendError::Cuda(e.to_string()))?;

        Ok(result)
    }

    fn dispatch(
        &self,
        desc: &dyn KernelDescriptor,
        inputs: &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
    ) -> Result<(), BackendError> {
        let kernel_desc = desc
            .as_any()
            .downcast_ref::<CudaKernelDesc>()
            .ok_or_else(|| BackendError::InvalidKernel("Not a CudaKernelDesc".to_string()))?;

        let kernel = self
            .device
            .get_func(&kernel_desc.module_name, &kernel_desc.func_name)
            .ok_or_else(|| {
                BackendError::InvalidKernel(format!(
                    "Function '{}' not found in module '{}'",
                    kernel_desc.func_name, kernel_desc.module_name
                ))
            })?;

        let cfg = LaunchConfig {
            grid_dim: kernel_desc.grid_dim,
            block_dim: kernel_desc.block_dim,
            shared_mem_bytes: 0,
        };

        if inputs.len() == 1 && outputs.len() == 1 {
            let input = inputs[0]
                .as_any()
                .downcast_ref::<CudaBuffer>()
                .ok_or_else(|| {
                    BackendError::InvalidKernel("Invalid input buffer type".to_string())
                })?;

            let output = outputs[0]
                .as_any_mut()
                .downcast_mut::<CudaBuffer>()
                .ok_or_else(|| {
                    BackendError::InvalidKernel("Invalid output buffer type".to_string())
                })?;

            let mut output_slice = output.slice.clone();

            unsafe {
                kernel
                    .launch(cfg, (&input.slice, &mut output_slice))
                    .map_err(|e| BackendError::Cuda(e.to_string()))?;
            }

            self.device
                .dtod_copy(&output_slice, &mut output.slice)
                .map_err(|e| BackendError::Cuda(e.to_string()))?;
        } else {
            return Err(BackendError::InvalidKernel(
                "Only single input/output supported".to_string(),
            ));
        }

        Ok(())
    }
}
