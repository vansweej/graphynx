//! Integration test: band energy extraction from a real spectrum.
//!
//! Generates a 440 Hz sine wave, computes its FFT via the CPU backend,
//! then runs `BandExtract` and asserts that the "mid" band (250–2000 Hz)
//! contains the dominant energy.

use backends::Backend;
use backends_cpu::CpuBackend;
use graph_core::ops::{BandDef, BandExtractParams, FftDirection, FftOutput, FftParams, Op};

const SAMPLE_RATE: f32 = 8_000.0;
const FFT_SIZE: usize = 1024;

fn sine_bytes(freq: f32, sr: f32, n: usize) -> Vec<u8> {
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect();
    bytemuck::cast_slice(&samples).to_vec()
}

fn compute_spectrum(backend: &CpuBackend, freq: f32) -> Vec<u8> {
    let params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    let input = sine_bytes(freq, SAMPLE_RATE, FFT_SIZE);
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::Fft(params), &[&input], &mut outputs)
        .unwrap();
    outputs.remove(0)
}

#[test]
fn band_extract_440hz_energy_in_mid_band() {
    let backend = CpuBackend::new("cpu");
    let spectrum = compute_spectrum(&backend, 440.0);

    let bands = vec![
        BandDef::new(0.0, 250.0, "low").unwrap(),
        BandDef::new(250.0, 2000.0, "mid").unwrap(),
        BandDef::new(2000.0, 4000.0, "high").unwrap(),
    ];
    let params = BandExtractParams::new(bands, SAMPLE_RATE, 0.0).unwrap();
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::BandExtract(params), &[&spectrum], &mut outputs)
        .unwrap();

    let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
    assert_eq!(energies.len(), 3);

    // 440 Hz is in the mid band; it should have the highest energy.
    let (low, mid, high) = (energies[0], energies[1], energies[2]);
    assert!(
        mid > low && mid > high,
        "Mid band ({mid}) should dominate: low={low}, high={high}"
    );
}

#[test]
fn band_extract_output_length_equals_band_count() {
    let backend = CpuBackend::new("cpu");
    let spectrum = compute_spectrum(&backend, 440.0);

    let bands = vec![
        BandDef::new(0.0, 1000.0, "a").unwrap(),
        BandDef::new(1000.0, 2000.0, "b").unwrap(),
        BandDef::new(2000.0, 3000.0, "c").unwrap(),
        BandDef::new(3000.0, 4000.0, "d").unwrap(),
    ];
    let params = BandExtractParams::new(bands, SAMPLE_RATE, 0.0).unwrap();
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::BandExtract(params), &[&spectrum], &mut outputs)
        .unwrap();
    let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
    assert_eq!(energies.len(), 4);
}

#[test]
fn band_extract_all_energies_non_negative() {
    let backend = CpuBackend::new("cpu");
    let spectrum = compute_spectrum(&backend, 440.0);

    let bands = vec![
        BandDef::new(0.0, 1000.0, "low").unwrap(),
        BandDef::new(1000.0, 4000.0, "high").unwrap(),
    ];
    let params = BandExtractParams::new(bands, SAMPLE_RATE, 0.0).unwrap();
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::BandExtract(params), &[&spectrum], &mut outputs)
        .unwrap();
    let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
    for &e in energies {
        assert!(e >= 0.0, "Energy should be non-negative, got {e}");
    }
}
