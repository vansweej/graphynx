/**
 * hello_kernel - doubles every element of an array on the GPU.
 *
 * Each CUDA thread handles one element. The thread's global index (idx)
 * is computed from the block and thread indices, then used to read from
 * the input array and write the doubled value to the output array.
 *
 * @param in   Input array  (read-only).  Element i is the i-th integer to process.
 * @param out  Output array (write-only). Element i receives in[i] * 2.
 *
 * Launch configuration: 1 block x N threads, where N == number of elements.
 */
extern "C" __global__ void hello_kernel(const int* in, int* out) {
    // Global thread index — one thread per array element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx] * 2;
}
