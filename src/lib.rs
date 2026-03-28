pub mod backend;
pub mod cuda_backend;

use bytemuck::Pod;

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
    Ok(output)
}
