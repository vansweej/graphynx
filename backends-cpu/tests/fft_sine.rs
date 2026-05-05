//! Integration test: 440 Hz sine wave → FFT → peak bin check.
//!
//! Generates a synthetic 440 Hz sine wave at 44 100 Hz sample rate,
//! runs it through the CPU backend FFT, and asserts that the dominant
//! magnitude bin corresponds to 440 Hz.

use backends::Backend;
use backends_cpu::CpuBackend;
use graph_core::ops::{FftDirection, FftOutput, FftParams, Op};

const SAMPLE_RATE: f32 = 44_100.0;
const FFT_SIZE: usize = 4096;
const FREQ_HZ: f32 = 440.0;

/// Build a 440 Hz sine wave as raw `f32` bytes.
fn sine_bytes(freq: f32, sr: f32, n: usize) -> Vec<u8> {
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();
    bytemuck::cast_slice(&samples).to_vec()
}

#[test]
fn fft_sine_peak_at_440hz() {
    let backend = CpuBackend::new("cpu");
    let params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();

    let input = sine_bytes(FREQ_HZ, SAMPLE_RATE, FFT_SIZE);
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::Fft(params), &[&input], &mut outputs)
        .unwrap();

    let magnitudes: &[f32] = bytemuck::cast_slice(&outputs[0]);
    // One-sided spectrum: FFT_SIZE/2 + 1 bins.
    assert_eq!(magnitudes.len(), FFT_SIZE / 2 + 1);

    // Find the bin with the maximum magnitude (skip DC bin 0).
    let peak_bin = magnitudes[1..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i + 1) // offset back for skipped DC
        .unwrap();

    // Expected bin: round(freq * N / sr)
    let expected_bin = (FREQ_HZ * FFT_SIZE as f32 / SAMPLE_RATE).round() as usize;

    // Allow ±1 bin tolerance for rounding.
    assert!(
        peak_bin.abs_diff(expected_bin) <= 1,
        "Peak bin {peak_bin} is not within 1 of expected bin {expected_bin} for {FREQ_HZ} Hz"
    );
}

#[test]
fn fft_zero_input_gives_zero_magnitudes() {
    let backend = CpuBackend::new("cpu");
    let params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    let input: Vec<u8> = bytemuck::cast_slice(&vec![0.0f32; FFT_SIZE]).to_vec();
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::Fft(params), &[&input], &mut outputs)
        .unwrap();
    let magnitudes: &[f32] = bytemuck::cast_slice(&outputs[0]);
    for &m in magnitudes {
        assert!(m.abs() < 1e-6, "Expected ~0, got {m}");
    }
}

#[test]
fn fft_output_length_is_one_sided() {
    let backend = CpuBackend::new("cpu");
    for &size in &[64usize, 256, 1024] {
        let params = FftParams::new(size, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        let input: Vec<u8> = bytemuck::cast_slice(&vec![1.0f32; size]).to_vec();
        let mut outputs = vec![Vec::new()];
        backend
            .dispatch_op(&Op::Fft(params), &[&input], &mut outputs)
            .unwrap();
        let magnitudes: &[f32] = bytemuck::cast_slice(&outputs[0]);
        assert_eq!(
            magnitudes.len(),
            size / 2 + 1,
            "Wrong output length for FFT size {size}"
        );
    }
}
