pub mod backend;
pub mod cuda_backend;

use bytemuck::Pod;

use backend::{Backend, BackendError, KernelDescriptor};

pub fn run_kernel<T: Pod>(
    backend: &dyn Backend,
    desc: &dyn KernelDescriptor,
    input: &[T],
) -> Result<Vec<T>, BackendError> {
    let input_bytes: &[u8] = bytemuck::cast_slice(input);
    let output_size_bytes = input_bytes.len();

    let input_buf = backend.upload(input_bytes)?;
    let mut output_buf = backend.alloc(output_size_bytes)?;

    backend.dispatch(desc, &[input_buf.as_ref()], &mut [output_buf.as_mut()])?;

    let output_bytes = backend.download(output_buf.as_ref())?;

    let output: Vec<T> = bytemuck::cast_slice(&output_bytes).to_vec();
    Ok(output)
}
