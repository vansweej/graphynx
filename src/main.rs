use rustycuda::cuda_backend::{CudaBackend, CudaKernelDesc};

/// Number of elements in the input/output arrays.
const N: usize = 10;

/// Demonstrates launching a CUDA kernel that doubles every element of an array.
///
/// # Flow
/// 1. Initialise the `CudaBackend` (opens GPU device 0, loads PTX, registers kernel).
/// 2. Describe the kernel launch via `CudaKernelDesc`.
/// 3. Call `run_kernel` — upload, dispatch, and download are handled by the backend.
/// 4. Print the result.
///
/// # Input
/// A fixed array of 10 integers: `[3, 7, 1, 9, 4, 6, 2, 8, 5, 10]`
///
/// # Output
/// Each element doubled: `[6, 14, 2, 18, 8, 12, 4, 16, 10, 20]`
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the pre-compiled PTX; the backend handles device init and PTX registration.
    let ptx = include_str!("../kernel.ptx");
    let backend = CudaBackend::new(0, ptx, "hello")?;

    // Describe which kernel to run and how to partition the work:
    //   - 1 block with N threads (one thread per element).
    let desc = CudaKernelDesc::new("hello_kernel", [1, 1, 1], [N as u32, 1, 1]);

    // --- Input ---
    // 10 arbitrary integers that will be processed by the GPU kernel.
    let input: Vec<i32> = vec![3, 7, 1, 9, 4, 6, 2, 8, 5, 10];
    println!("Input:  {:?}", input);

    // --- Output ---
    // Upload, dispatch, and download are all handled inside run_kernel.
    let output: Vec<i32> = rustycuda::run_kernel(&backend, &desc, &input)?;
    println!("Output: {:?}", output);

    Ok(())
}
