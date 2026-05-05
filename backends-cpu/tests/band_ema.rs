//! Integration test: EMA smoothing across multiple ticks.
//!
//! Verifies that the EMA state output from one tick feeds correctly into the
//! next tick, and that the smoothed values converge toward the raw energies
//! after many iterations.

use backends::Backend;
use backends_cpu::CpuBackend;
use graph_core::ops::{BandDef, BandExtractParams, Op};

const SAMPLE_RATE: f32 = 8_000.0;
const FFT_SIZE: usize = 512;

fn flat_spectrum_bytes(n_bins: usize, value: f32) -> Vec<u8> {
    bytemuck::cast_slice(&vec![value; n_bins]).to_vec()
}

fn make_params(smoothing: f32) -> BandExtractParams {
    let bands = vec![
        BandDef::new(0.0, 1000.0, "low").unwrap(),
        BandDef::new(1000.0, 4000.0, "mid").unwrap(),
    ];
    BandExtractParams::new(bands, SAMPLE_RATE, smoothing).unwrap()
}

/// Run one stateful band-extract tick; returns (energies, new_state).
fn tick(
    backend: &CpuBackend,
    params: &BandExtractParams,
    state: &[f32],
    spectrum: &[u8],
) -> (Vec<f32>, Vec<f32>) {
    let state_bytes: Vec<u8> = bytemuck::cast_slice(state).to_vec();
    let mut outputs = vec![Vec::new(), Vec::new()];
    backend
        .dispatch_op(
            &Op::BandExtract(params.clone()),
            &[&state_bytes, spectrum],
            &mut outputs,
        )
        .unwrap();
    let energies = bytemuck::cast_slice::<u8, f32>(&outputs[0]).to_vec();
    let new_state = bytemuck::cast_slice::<u8, f32>(&outputs[1]).to_vec();
    (energies, new_state)
}

#[test]
fn ema_first_tick_zero_state_scales_by_alpha() {
    let backend = CpuBackend::new("cpu");
    let alpha = 0.5_f32;
    let params = make_params(alpha);

    // Flat spectrum of 1.0 everywhere.
    let n_bins = FFT_SIZE / 2 + 1;
    let spectrum = flat_spectrum_bytes(n_bins, 1.0);

    // First tick: state = zeros.
    let zero_state = vec![0.0f32; 2];
    let (energies, new_state) = tick(&backend, &params, &zero_state, &spectrum);

    // EMA: y = alpha * x + (1-alpha) * 0 = alpha * x
    // Energies and new_state must be identical.
    assert_eq!(energies, new_state);
    // All energies must be positive.
    for &e in &energies {
        assert!(
            e > 0.0,
            "Expected positive energy after first tick, got {e}"
        );
    }
}

#[test]
fn ema_second_tick_blends_previous_state() {
    let backend = CpuBackend::new("cpu");
    let alpha = 0.5_f32;
    let params = make_params(alpha);

    let n_bins = FFT_SIZE / 2 + 1;
    let spectrum = flat_spectrum_bytes(n_bins, 1.0);
    let zero_state = vec![0.0f32; 2];

    // Tick 1.
    let (_, state1) = tick(&backend, &params, &zero_state, &spectrum);
    // Tick 2.
    let (energies2, state2) = tick(&backend, &params, &state1, &spectrum);

    // After two ticks the state should be closer to raw than after one tick.
    // Specifically: state2 = alpha*raw + (1-alpha)*state1 > state1 (since raw > state1).
    for (s1, s2) in state1.iter().zip(state2.iter()) {
        assert!(s2 > s1, "State should increase toward raw: {s2} > {s1}");
    }
    // Energies equal new state.
    assert_eq!(energies2, state2);
}

#[test]
fn ema_converges_to_raw_after_many_ticks() {
    let backend = CpuBackend::new("cpu");
    let params = make_params(0.5);

    let n_bins = FFT_SIZE / 2 + 1;
    let spectrum = flat_spectrum_bytes(n_bins, 1.0);

    // Compute raw (stateless) energies for comparison.
    let stateless = make_params(0.0);
    let zero_state_sl = vec![0.0f32; 2];
    // Stateless: inputs[0] = spectrum directly.
    let mut raw_out = vec![Vec::new()];
    backend
        .dispatch_op(&Op::BandExtract(stateless), &[&spectrum], &mut raw_out)
        .unwrap();
    let raw: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&raw_out[0]).to_vec();

    // Run 40 ticks with smoothing=0.5.
    let mut state = zero_state_sl;
    for _ in 0..40 {
        let (_, new_state) = tick(&backend, &params, &state, &spectrum);
        state = new_state;
    }

    for (s, r) in state.iter().zip(raw.iter()) {
        assert!(
            (s - r).abs() < 0.01,
            "EMA state {s} should converge to raw {r} after 40 ticks"
        );
    }
}

#[test]
fn ema_stateful_outputs_two_non_empty_buffers() {
    let backend = CpuBackend::new("cpu");
    let params = make_params(0.3);
    let n_bins = FFT_SIZE / 2 + 1;
    let spectrum = flat_spectrum_bytes(n_bins, 1.0);
    let state = vec![0.0f32; 2];
    let (energies, new_state) = tick(&backend, &params, &state, &spectrum);
    assert!(!energies.is_empty());
    assert!(!new_state.is_empty());
    assert_eq!(energies.len(), new_state.len());
}

#[test]
fn ema_alpha_zero_is_stateless_equivalent() {
    // smoothing=0.0 → is_stateful() == false → stateless path.
    let backend = CpuBackend::new("cpu");
    let params = make_params(0.0);
    let n_bins = FFT_SIZE / 2 + 1;
    let spectrum = flat_spectrum_bytes(n_bins, 1.0);
    let mut outputs = vec![Vec::new()];
    backend
        .dispatch_op(&Op::BandExtract(params), &[&spectrum], &mut outputs)
        .unwrap();
    let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
    assert_eq!(energies.len(), 2);
    for &e in energies {
        assert!(e > 0.0);
    }
}
