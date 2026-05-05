//! Integration test: windowing function coefficients.
//!
//! Applies each supported window kind to a unit-valued frame and checks
//! that the output coefficients match the expected analytical formula.

use backends::Backend;
use backends_cpu::CpuBackend;
use graph_core::ops::{Op, WindowKind, WindowParams};

fn unit_frame_bytes(n: usize) -> Vec<u8> {
    bytemuck::cast_slice(&vec![1.0f32; n]).to_vec()
}

fn dispatch_window(kind: WindowKind, n: usize) -> Vec<f32> {
    let backend = CpuBackend::new("cpu");
    let params = WindowParams::new(kind, n).unwrap();
    let input = unit_frame_bytes(n);
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::Window(params), &[&input], &mut outputs)
        .unwrap();
    bytemuck::cast_slice::<u8, f32>(&outputs[0]).to_vec()
}

// ── Hann ──────────────────────────────────────────────────────────────────────

#[test]
fn hann_endpoints_are_zero() {
    let coeffs = dispatch_window(WindowKind::Hann, 64);
    assert!(
        coeffs[0].abs() < 1e-6,
        "Hann[0] should be ~0, got {}",
        coeffs[0]
    );
    assert!(
        coeffs[63].abs() < 1e-6,
        "Hann[N-1] should be ~0, got {}",
        coeffs[63]
    );
}

#[test]
fn hann_centre_is_one() {
    // For N=3 (odd), the midpoint bin 1 gives exactly:
    //   w[1] = 0.5 * (1 - cos(2π * 1 / 2)) = 0.5 * (1 - cos(π)) = 1.0
    let coeffs = dispatch_window(WindowKind::Hann, 3);
    let peak = coeffs[1];
    assert!(
        (peak - 1.0).abs() < 1e-5,
        "Hann centre should be ~1.0, got {peak}"
    );
}

#[test]
fn hann_matches_formula() {
    let n = 16usize;
    let coeffs = dispatch_window(WindowKind::Hann, n);
    for (i, &c) in coeffs.iter().enumerate() {
        let expected = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
        assert!(
            (c - expected).abs() < 1e-5,
            "Hann[{i}]: expected {expected}, got {c}"
        );
    }
}

// ── Hamming ───────────────────────────────────────────────────────────────────

#[test]
fn hamming_endpoints_near_008() {
    let coeffs = dispatch_window(WindowKind::Hamming, 64);
    // Hamming endpoints ≈ 0.08 (not zero).
    assert!(
        (coeffs[0] - 0.08).abs() < 0.01,
        "Hamming[0] ≈ 0.08, got {}",
        coeffs[0]
    );
}

#[test]
fn hamming_matches_formula() {
    let n = 16usize;
    let coeffs = dispatch_window(WindowKind::Hamming, n);
    for (i, &c) in coeffs.iter().enumerate() {
        let expected = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
        assert!(
            (c - expected).abs() < 1e-5,
            "Hamming[{i}]: expected {expected}, got {c}"
        );
    }
}

// ── Blackman ──────────────────────────────────────────────────────────────────

#[test]
fn blackman_endpoints_near_zero() {
    let coeffs = dispatch_window(WindowKind::Blackman, 64);
    assert!(
        coeffs[0].abs() < 1e-5,
        "Blackman[0] should be ~0, got {}",
        coeffs[0]
    );
}

#[test]
fn blackman_matches_formula() {
    let n = 16usize;
    let coeffs = dispatch_window(WindowKind::Blackman, n);
    for (i, &c) in coeffs.iter().enumerate() {
        let t = 2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32;
        let expected = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
        assert!(
            (c - expected).abs() < 1e-5,
            "Blackman[{i}]: expected {expected}, got {c}"
        );
    }
}

// ── General ───────────────────────────────────────────────────────────────────

#[test]
fn window_output_length_matches_input() {
    for &n in &[32usize, 128, 512] {
        let coeffs = dispatch_window(WindowKind::Hann, n);
        assert_eq!(coeffs.len(), n, "Window output length mismatch for n={n}");
    }
}

#[test]
fn window_scales_input_values() {
    // Input = 2.0 everywhere; output should be 2.0 * window_coeff.
    let n = 16usize;
    let backend = CpuBackend::new("cpu");
    let params = WindowParams::new(WindowKind::Hann, n).unwrap();
    let input: Vec<u8> = bytemuck::cast_slice(&vec![2.0f32; n]).to_vec();
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::Window(params), &[&input], &mut outputs)
        .unwrap();
    let out: &[f32] = bytemuck::cast_slice(&outputs[0]);

    let unit = dispatch_window(WindowKind::Hann, n);
    for (i, (&o, &u)) in out.iter().zip(unit.iter()).enumerate() {
        assert!(
            (o - 2.0 * u).abs() < 1e-5,
            "Window scaling failed at [{i}]: {o} ≠ 2 × {u}"
        );
    }
}
