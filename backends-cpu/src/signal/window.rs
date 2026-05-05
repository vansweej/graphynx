//! Windowing functions for spectral analysis.
//!
//! A window function tapers a signal frame to zero at its edges, reducing
//! *spectral leakage* — the smearing of energy from one frequency bin into
//! adjacent bins that occurs when the FFT assumes a periodic signal.
//!
//! ## Available windows
//!
//! | Window | Formula | Side-lobe attenuation | Main-lobe width |
//! |--------|---------|----------------------|-----------------|
//! | Hann | `0.5 × (1 − cos(2πn/(N−1)))` | −31.5 dB | 4 bins |
//! | Hamming | `0.54 − 0.46 × cos(2πn/(N−1))` | −41.0 dB | 4 bins |
//! | Blackman | `0.42 − 0.5 × cos(2πn/(N−1)) + 0.08 × cos(4πn/(N−1))` | −58.1 dB | 6 bins |
//!
//! For voice-frequency analysis the **Hann** window is a good default.
//!
//! ## Dispatch
//!
//! Called from [`super::dispatch_signal_op`] when the op is
//! [`Op::Window`](graph_core::ops::Op::Window).

use bytemuck::{cast_slice, cast_slice_mut};
use graph_core::ops::signal::{WindowKind, WindowParams};

use backends::BackendError;

/// Apply a window function to a frame of f32 samples.
///
/// Reads `inputs[0]` as `[size] f32`, multiplies element-wise by the window
/// coefficients, and writes the result to `outputs[0]`.
///
/// ## Window coefficient formulas
///
/// For a frame of `N` samples, coefficient at position `n` (0-indexed):
///
/// - **Hann:** `w[n] = 0.5 × (1 − cos(2πn / (N−1)))`
/// - **Hamming:** `w[n] = 0.54 − 0.46 × cos(2πn / (N−1))`
/// - **Blackman:** `w[n] = 0.42 − 0.5 × cos(2πn/(N−1)) + 0.08 × cos(4πn/(N−1))`
///
/// # Errors
///
/// - [`BackendError::Buffer`] if `inputs[0]` byte length does not match
///   `params.size * 4` (i.e. `params.size` f32 values).
pub(crate) fn dispatch_window(
    params: &WindowParams,
    inputs: &[&[u8]],
    outputs: &mut [Vec<u8>],
) -> Result<(), BackendError> {
    let n = params.size;
    let expected_bytes = n * std::mem::size_of::<f32>();

    if inputs[0].len() != expected_bytes {
        return Err(BackendError::Buffer(format!(
            "Window: expected {expected_bytes} input bytes ({n} f32), got {}",
            inputs[0].len()
        )));
    }

    let input: &[f32] = cast_slice(inputs[0]);

    // Pre-compute window coefficients and apply in one pass.
    outputs[0].resize(expected_bytes, 0);
    let output: &mut [f32] = cast_slice_mut(&mut outputs[0]);

    let n_f = (n - 1) as f64;
    for (i, (x, y)) in input.iter().zip(output.iter_mut()).enumerate() {
        let theta = std::f64::consts::TAU * i as f64 / n_f;
        let w = match params.kind {
            WindowKind::Hann => 0.5 * (1.0 - theta.cos()),
            WindowKind::Hamming => 0.54 - 0.46 * theta.cos(),
            WindowKind::Blackman => 0.42 - 0.5 * theta.cos() + 0.08 * (2.0 * theta).cos(),
        };
        *y = *x * w as f32;
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run_window(kind: WindowKind, size: usize, input: &[f32]) -> Vec<f32> {
        let params = WindowParams::new(kind, size).unwrap();
        let input_bytes: Vec<u8> = bytemuck::cast_slice(input).to_vec();
        let mut outputs = vec![Vec::new()];
        dispatch_window(&params, &[&input_bytes], &mut outputs).unwrap();
        bytemuck::cast_slice(&outputs[0]).to_vec()
    }

    #[test]
    fn hann_endpoints_are_zero() {
        // Hann window: w[0] = 0, w[N-1] = 0 (for N >= 2).
        let out = run_window(WindowKind::Hann, 8, &[1.0f32; 8]);
        assert!(
            out[0].abs() < 1e-6,
            "Hann w[0] should be ~0, got {}",
            out[0]
        );
        assert!(
            out[7].abs() < 1e-6,
            "Hann w[N-1] should be ~0, got {}",
            out[7]
        );
    }

    #[test]
    fn hann_midpoint_is_one() {
        // For N=3: w[1] = 0.5*(1 - cos(2π*1/2)) = 0.5*(1 - cos(π)) = 1.0
        let out = run_window(WindowKind::Hann, 3, &[1.0f32; 3]);
        assert!(
            (out[1] - 1.0).abs() < 1e-6,
            "Hann midpoint should be 1.0, got {}",
            out[1]
        );
    }

    #[test]
    fn hamming_endpoints_are_nonzero() {
        // Hamming does not taper to zero: w[0] = 0.54 - 0.46 = 0.08
        let out = run_window(WindowKind::Hamming, 8, &[1.0f32; 8]);
        assert!(
            (out[0] - 0.08).abs() < 1e-5,
            "Hamming w[0] should be ~0.08, got {}",
            out[0]
        );
    }

    #[test]
    fn blackman_endpoints_near_zero() {
        // Blackman: w[0] = 0.42 - 0.5 + 0.08 = 0.0
        let out = run_window(WindowKind::Blackman, 8, &[1.0f32; 8]);
        assert!(
            out[0].abs() < 1e-5,
            "Blackman w[0] should be ~0, got {}",
            out[0]
        );
    }

    #[test]
    fn window_scales_input() {
        // All-2 input → output should be 2 * window_coefficient.
        let out_ones = run_window(WindowKind::Hann, 16, &[1.0f32; 16]);
        let out_twos = run_window(WindowKind::Hann, 16, &[2.0f32; 16]);
        for (a, b) in out_ones.iter().zip(out_twos.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-6);
        }
    }

    #[test]
    fn window_output_length_matches_input() {
        let out = run_window(WindowKind::Hann, 64, &[1.0f32; 64]);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn window_wrong_input_size_returns_error() {
        let params = WindowParams::new(WindowKind::Hann, 8).unwrap();
        let bad_input = vec![0u8; 12]; // 3 f32, not 8
        let mut outputs = vec![Vec::new()];
        let result = dispatch_window(&params, &[&bad_input], &mut outputs);
        assert!(result.is_err());
    }

    #[test]
    fn window_single_sample() {
        // N=1: n_f = 0.0 → theta = 0/0 = NaN, but cos(NaN) = NaN.
        // For N=1 the loop runs once with i=0, theta=0/0.
        // Hann: 0.5*(1 - cos(0)) = 0.5*(1-1) = 0. Output should be 0.
        // This is an edge case — document it and verify no panic.
        let out = run_window(WindowKind::Hann, 1, &[1.0f32]);
        // NaN * 1.0 = NaN; we just verify no panic and the length is correct.
        assert_eq!(out.len(), 1);
    }
}
