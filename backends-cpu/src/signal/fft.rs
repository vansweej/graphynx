//! Fast Fourier Transform dispatch.
//!
//! Wraps [`rustfft`] to compute forward and inverse DFTs on f32 data.
//!
//! ## Implementation strategy
//!
//! `rustfft` operates on [`num_complex::Complex<f32>`] values. For a real
//! input frame we zero-pad the imaginary part before calling the forward FFT:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │  Real input [N] f32                                                  │
//! │      │                                                               │
//! │      ▼  zero-pad imaginary                                           │
//! │  Complex input [N] Complex<f32>  (im = 0)                           │
//! │      │                                                               │
//! │      ▼  rustfft forward                                              │
//! │  Complex output [N] Complex<f32>                                     │
//! │      │                                                               │
//! │      ├─── FftOutput::Complex   → [N] f32 interleaved (re, im, …)    │
//! │      ├─── FftOutput::Magnitude → [N/2+1] f32  √(re² + im²)          │
//! │      └─── FftOutput::Power     → [N/2+1] f32  re² + im²             │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## FftPlanner caching
//!
//! The [`rustfft::FftPlanner`] is held in the parent [`CpuBackend`] behind a
//! `Mutex`. It caches twiddle-factor tables across calls, so repeated FFTs of
//! the same size are fast. The planner is passed in by reference.
//!
//! ## Performance note
//!
//! Power-of-two sizes (512, 1024, 2048, …) use the highly-optimised
//! Cooley-Tukey algorithm. Non-power-of-two sizes fall back to a mixed-radix
//! plan that is correct but slower.
//!
//! [`CpuBackend`]: crate::CpuBackend

use num_complex::Complex;
use rustfft::FftPlanner;

use backends::BackendError;
use graph_core::ops::signal::{FftDirection, FftOutput, FftParams};

/// Compute an FFT using the provided planner.
///
/// Reads `inputs[0]` as `[params.size] f32` real samples, runs the FFT, and
/// writes the result to `outputs[0]` in the format specified by
/// [`FftParams::output`].
///
/// ## Output formats
///
/// | `FftOutput` | Output length | Content |
/// |-------------|---------------|---------|
/// | `Complex` | `size` f32 pairs (interleaved) | Full complex spectrum |
/// | `Magnitude` | `size/2 + 1` f32 values | `√(re² + im²)` |
/// | `Power` | `size/2 + 1` f32 values | `re² + im²` |
///
/// ## Algorithm
///
/// 1. Cast input bytes to `&[f32]`.
/// 2. Build `[Complex<f32>; N]` with `im = 0.0`.
/// 3. Call `planner.plan_fft_forward(N)` (or `plan_fft_inverse(N)`).
/// 4. Run the FFT in-place.
/// 5. Extract the requested output format from the complex result.
///
/// # Errors
///
/// Returns [`BackendError::Buffer`] if `inputs[0]` byte length does not equal
/// `params.size * 4`.
pub(crate) fn dispatch_fft(
    params: &FftParams,
    inputs: &[&[u8]],
    outputs: &mut [Vec<u8>],
    planner: &mut FftPlanner<f32>,
) -> Result<(), BackendError> {
    let n = params.size;
    let expected_bytes = n * std::mem::size_of::<f32>();

    if inputs[0].len() != expected_bytes {
        return Err(BackendError::Buffer(format!(
            "Fft: expected {expected_bytes} input bytes ({n} f32), got {}",
            inputs[0].len()
        )));
    }

    let input: &[f32] = bytemuck::cast_slice(inputs[0]);

    // Build complex buffer: real input, zero imaginary.
    let mut buf: Vec<Complex<f32>> = input.iter().map(|&re| Complex { re, im: 0.0 }).collect();

    // Plan and execute the FFT.
    match params.direction {
        FftDirection::Forward => {
            let fft = planner.plan_fft_forward(n);
            fft.process(&mut buf);
        }
        FftDirection::Inverse => {
            let fft = planner.plan_fft_inverse(n);
            fft.process(&mut buf);
        }
    }

    // Write output in the requested format.
    match params.output {
        FftOutput::Complex => {
            // Interleave re and im as f32 pairs.
            outputs[0].resize(n * 2 * std::mem::size_of::<f32>(), 0);
            let out: &mut [f32] = bytemuck::cast_slice_mut(&mut outputs[0]);
            for (i, c) in buf.iter().enumerate() {
                out[2 * i] = c.re;
                out[2 * i + 1] = c.im;
            }
        }
        FftOutput::Magnitude => {
            let one_sided = n / 2 + 1;
            outputs[0].resize(one_sided * std::mem::size_of::<f32>(), 0);
            let out: &mut [f32] = bytemuck::cast_slice_mut(&mut outputs[0]);
            for (i, c) in buf[..one_sided].iter().enumerate() {
                out[i] = c.norm();
            }
        }
        FftOutput::Power => {
            let one_sided = n / 2 + 1;
            outputs[0].resize(one_sided * std::mem::size_of::<f32>(), 0);
            let out: &mut [f32] = bytemuck::cast_slice_mut(&mut outputs[0]);
            for (i, c) in buf[..one_sided].iter().enumerate() {
                out[i] = c.norm_sqr();
            }
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use rustfft::FftPlanner;

    use graph_core::ops::signal::{FftDirection, FftOutput, FftParams};

    use super::*;

    fn run_fft(params: &FftParams, input: &[f32]) -> Vec<u8> {
        let input_bytes: Vec<u8> = bytemuck::cast_slice(input).to_vec();
        let mut outputs = vec![Vec::new()];
        let mut planner = FftPlanner::new();
        dispatch_fft(params, &[&input_bytes], &mut outputs, &mut planner).unwrap();
        outputs.remove(0)
    }

    fn fft_magnitude(size: usize, input: &[f32]) -> Vec<f32> {
        let params = FftParams::new(size, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        let bytes = run_fft(&params, input);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    fn fft_power(size: usize, input: &[f32]) -> Vec<f32> {
        let params = FftParams::new(size, FftDirection::Forward, FftOutput::Power).unwrap();
        let bytes = run_fft(&params, input);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    fn fft_complex(size: usize, input: &[f32]) -> Vec<f32> {
        let params = FftParams::new(size, FftDirection::Forward, FftOutput::Complex).unwrap();
        let bytes = run_fft(&params, input);
        bytemuck::cast_slice(&bytes).to_vec()
    }

    // ── DC component ─────────────────────────────────────────────────────

    #[test]
    fn dc_signal_energy_at_bin_zero() {
        // A constant signal has all energy at bin 0 (DC).
        let n = 64;
        let dc = vec![1.0f32; n];
        let mag = fft_magnitude(n, &dc);
        // Bin 0 magnitude = N (sum of N ones).
        assert!(
            (mag[0] - n as f32).abs() < 1e-3,
            "DC bin 0 magnitude should be {n}, got {}",
            mag[0]
        );
        // All other bins should be near zero.
        for (i, &m) in mag[1..].iter().enumerate() {
            assert!(
                m < 1e-3,
                "Non-DC bin {} has unexpected magnitude {m}",
                i + 1
            );
        }
    }

    // ── Sine wave peak at expected bin ────────────────────────────────────

    #[test]
    fn sine_peak_at_correct_bin() {
        // A pure sine at frequency k has a peak at bin k.
        let n = 64;
        let k = 5usize; // target bin
        let sine: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32).sin())
            .collect();
        let mag = fft_magnitude(n, &sine);
        let peak_bin = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak_bin, k, "Peak should be at bin {k}, got {peak_bin}");
    }

    // ── Output length ─────────────────────────────────────────────────────

    #[test]
    fn magnitude_output_length_is_one_sided() {
        let n = 128;
        let mag = fft_magnitude(n, &vec![1.0f32; n]);
        assert_eq!(mag.len(), n / 2 + 1);
    }

    #[test]
    fn power_output_length_is_one_sided() {
        let n = 128;
        let pow = fft_power(n, &vec![1.0f32; n]);
        assert_eq!(pow.len(), n / 2 + 1);
    }

    #[test]
    fn complex_output_length_is_full_spectrum() {
        let n = 64;
        let cx = fft_complex(n, &vec![1.0f32; n]);
        // Complex output: n pairs of (re, im) → 2n f32 values.
        assert_eq!(cx.len(), 2 * n);
    }

    // ── Power = magnitude² ────────────────────────────────────────────────

    #[test]
    fn power_equals_magnitude_squared() {
        let n = 32;
        let input: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mag = fft_magnitude(n, &input);
        let pow = fft_power(n, &input);
        for (m, p) in mag.iter().zip(pow.iter()) {
            assert!((m * m - p).abs() < 1e-4, "mag²={} ≠ pow={}", m * m, p);
        }
    }

    // ── Wrong input size ──────────────────────────────────────────────────

    #[test]
    fn wrong_input_size_returns_error() {
        let params = FftParams::new(64, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        let bad = vec![0u8; 10]; // not 64*4 bytes
        let mut outputs = vec![Vec::new()];
        let mut planner = FftPlanner::new();
        let result = dispatch_fft(&params, &[&bad], &mut outputs, &mut planner);
        assert!(result.is_err());
    }

    // ── Inverse FFT ───────────────────────────────────────────────────────

    #[test]
    fn inverse_fft_does_not_panic() {
        // Inverse FFT is complex→complex; we just verify it runs without panic.
        let params = FftParams::new(32, FftDirection::Inverse, FftOutput::Complex).unwrap();
        let input = vec![0.0f32; 32];
        let bytes = run_fft(&params, &input);
        // Output should be 2*32 f32 values (complex interleaved).
        assert_eq!(bytes.len(), 32 * 2 * 4);
    }
}
