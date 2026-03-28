use std::any::Any;
use std::cell::RefCell;

use graphynx::backend::{
    Backend, BackendCaps, BackendError, DeviceBuffer, DeviceId, KernelDescriptor, MemoryModel,
    NodeKindTag,
};

pub fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

pub struct MemBuffer {
    data: RefCell<Vec<u8>>,
    device_id: DeviceId,
}

impl MemBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: RefCell::new(vec![0u8; size]),
            device_id: DeviceId::new("mem:0"),
        }
    }
}

// Safety: MemBuffer is only used in single-threaded test code.
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

pub struct DoubleKernelDesc;

impl KernelDescriptor for DoubleKernelDesc {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct MemBackend {
    device_id: DeviceId,
}

impl MemBackend {
    pub fn new() -> Self {
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

pub struct FailAllocBackend {
    device_id: DeviceId,
}

impl FailAllocBackend {
    pub fn new() -> Self {
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
