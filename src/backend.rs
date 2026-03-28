use std::any::Any;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("Invalid kernel descriptor: {0}")]
    InvalidKernel(String),
    #[error("Buffer error: {0}")]
    Buffer(String),
}

pub trait DeviceBuffer: Send + Sync {
    fn size_bytes(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait KernelDescriptor: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn alloc(&self, size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError>;
    fn upload(&self, data: &[u8]) -> Result<Box<dyn DeviceBuffer>, BackendError>;
    fn download(&self, buf: &dyn DeviceBuffer) -> Result<Vec<u8>, BackendError>;
    fn dispatch(
        &self,
        desc: &dyn KernelDescriptor,
        inputs: &[&dyn DeviceBuffer],
        outputs: &mut [&mut dyn DeviceBuffer],
    ) -> Result<(), BackendError>;
}
