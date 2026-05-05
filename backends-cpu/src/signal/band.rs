//! Frequency band energy extraction.
//!
//! Converts a magnitude or power spectrum (from [`Op::Fft`]) into a compact
//! vector of per-band energy values, with optional exponential moving-average
//! (EMA) smoothing.
//!
//! ## Bin-to-frequency mapping
//!
//! For an FFT of size `N` at sample rate `sr` Hz, the centre frequency of
//! bin `b` is:
//!
//! ```text
//! f(b) = b × sr / N
//! ```
//!
//! A band covering `[low_hz, high_hz)` maps to the inclusive bin range:
//!
//! ```text
//! bin_low  = floor(low_hz  × N / sr)
//! bin_high = ceil (high_hz × N / sr)
//! included = bin_low ..= min(bin_high, N/2)
//! ```
//!
//! If no bins fall within a band the output for that band is `0.0`.
//!
//! ## EMA smoothing
//!
//! When `smoothing α > 0.0` the op is stateful. The EMA update is:
//!
//! ```text
//! y[t] = α × x[t] + (1 − α) × y[t−1]
//! ```
//!
//! ## State convention
//!
//! The executor prepends/appends state according to the following protocol:
//!
//! ```text
//! Stateful (smoothing > 0):
//!   inputs[0]  = EMA state  [B] f32   (zeros on first tick)
//!   inputs[1]  = spectrum   [N/2+1] f32
//!   outputs[0] = energies   [B] f32
//!   outputs[1] = new state  [B] f32
//!
//! Stateless (smoothing == 0):
//!   inputs[0]  = spectrum   [N/2+1] f32
//!   outputs[0] = energies   [B] f32
//! ```
//!
//! [`Op::Fft`]: graph_core::ops::Op::Fft

use bytemuck::{cast_slice, cast_slice_mut};
use graph_core::ops::signal::BandExtractParams;

use backends::BackendError;

/// Extract per-band energy from a spectrum, with optional EMA smoothing.
///
/// See the [module-level documentation](self) for the full algorithm and
/// state convention.
///
/// # Errors
///
/// Returns [`BackendError::Buffer`] if the spectrum input slice has a byte
/// length that is not a multiple of 4 (i.e. not a valid `f32` slice).
pub(crate) fn dispatch_band_extract(
    params: &BandExtractParams,
    inputs: &[&[u8]],
    outputs: &mut [Vec<u8>],
) -> Result<(), BackendError> {
    let n_bands = params.bands.len();
    let f32_size = std::mem::size_of::<f32>();

    // Determine which input slot holds the spectrum (state-aware).
    let (spectrum_bytes, state_bytes) = if params.is_stateful() {
        // inputs[0] = EMA state, inputs[1] = spectrum
        (inputs[1], Some(inputs[0]))
    } else {
        // inputs[0] = spectrum
        (inputs[0], None)
    };

    if spectrum_bytes.len() % f32_size != 0 {
        return Err(BackendError::Buffer(format!(
            "BandExtract: spectrum byte length {} is not a multiple of 4",
            spectrum_bytes.len()
        )));
    }

    let spectrum: &[f32] = cast_slice(spectrum_bytes);
    let spectrum_len = spectrum.len();

    // Compute raw band energies by summing spectrum bins within each band.
    let raw_energies: Vec<f32> = params
        .bands
        .iter()
        .map(|band| {
            let sr = params.sample_rate_hz;
            // Reconstruct the full FFT size from the one-sided spectrum length.
            // spectrum_len = N/2 + 1  →  N = 2*(spectrum_len - 1)
            let n_full = 2 * (spectrum_len.saturating_sub(1)).max(1);
            let bin_low = ((band.low_hz * n_full as f32 / sr).floor() as usize)
                .min(spectrum_len.saturating_sub(1));
            let bin_high = ((band.high_hz * n_full as f32 / sr).ceil() as usize)
                .min(spectrum_len.saturating_sub(1));

            if bin_low > bin_high {
                return 0.0;
            }
            spectrum[bin_low..=bin_high].iter().sum()
        })
        .collect();

    // Apply EMA smoothing if stateful.
    let smoothed: Vec<f32> = if let Some(state_bytes) = state_bytes {
        let state: &[f32] = cast_slice(state_bytes);
        let alpha = params.smoothing;
        raw_energies
            .iter()
            .zip(state.iter())
            .map(|(&x, &s)| alpha * x + (1.0 - alpha) * s)
            .collect()
    } else {
        raw_energies
    };

    // Write band energies to outputs[0].
    outputs[0].resize(n_bands * f32_size, 0);
    let out: &mut [f32] = cast_slice_mut(&mut outputs[0]);
    out.copy_from_slice(&smoothed);

    // Write updated EMA state to outputs[1] (stateful only).
    if params.is_stateful() {
        outputs[1].resize(n_bands * f32_size, 0);
        let state_out: &mut [f32] = cast_slice_mut(&mut outputs[1]);
        state_out.copy_from_slice(&smoothed);
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::ops::signal::{BandDef, BandExtractParams};

    use super::*;

    fn make_params(smoothing: f32) -> BandExtractParams {
        let bands = vec![
            BandDef::new(0.0, 1000.0, "low").unwrap(),
            BandDef::new(1000.0, 4000.0, "mid").unwrap(),
            BandDef::new(4000.0, 8000.0, "high").unwrap(),
        ];
        BandExtractParams::new(bands, 8000.0, smoothing).unwrap()
    }

    fn flat_spectrum_bytes(len: usize, value: f32) -> Vec<u8> {
        let v: Vec<f32> = vec![value; len];
        bytemuck::cast_slice(&v).to_vec()
    }

    #[test]
    fn stateless_flat_spectrum_distributes_energy() {
        // 513 bins (FFT size 1024, sr=8000 Hz).
        // All bins = 1.0 → each band gets the number of bins it covers.
        let params = make_params(0.0);
        let spectrum = flat_spectrum_bytes(513, 1.0);
        let mut outputs = vec![Vec::new()];
        dispatch_band_extract(&params, &[&spectrum], &mut outputs).unwrap();
        let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
        assert_eq!(energies.len(), 3);
        for &e in energies {
            assert!(e > 0.0, "Expected positive energy, got {e}");
        }
    }

    #[test]
    fn stateless_output_has_correct_length() {
        let params = make_params(0.0);
        let spectrum = flat_spectrum_bytes(257, 1.0);
        let mut outputs = vec![Vec::new()];
        dispatch_band_extract(&params, &[&spectrum], &mut outputs).unwrap();
        let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
        assert_eq!(energies.len(), 3);
    }

    #[test]
    fn stateful_first_tick_with_zero_state() {
        // First tick: state = zeros → smoothed = alpha * raw + (1-alpha) * 0 = alpha * raw.
        let params = make_params(0.5);
        let spectrum = flat_spectrum_bytes(513, 2.0);
        let zero_state: Vec<u8> = vec![0u8; 3 * 4];
        let mut outputs = vec![Vec::new(), Vec::new()];
        dispatch_band_extract(&params, &[&zero_state, &spectrum], &mut outputs).unwrap();
        let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
        let new_state: &[f32] = bytemuck::cast_slice(&outputs[1]);
        // Energies and new_state must be identical.
        assert_eq!(energies, new_state);
        for &e in energies {
            assert!(e > 0.0);
        }
    }

    #[test]
    fn stateful_ema_converges_toward_raw() {
        // After many ticks with the same input, EMA state converges to raw.
        let params = make_params(0.5);
        let spectrum = flat_spectrum_bytes(513, 1.0);

        let stateless = make_params(0.0);
        let mut raw_out = vec![Vec::new()];
        dispatch_band_extract(&stateless, &[&spectrum], &mut raw_out).unwrap();
        let raw: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&raw_out[0]).to_vec();

        let mut state: Vec<f32> = vec![0.0; 3];
        for _ in 0..30 {
            let state_bytes: Vec<u8> = bytemuck::cast_slice(&state).to_vec();
            let mut outputs = vec![Vec::new(), Vec::new()];
            dispatch_band_extract(&params, &[&state_bytes, &spectrum], &mut outputs).unwrap();
            state = bytemuck::cast_slice::<u8, f32>(&outputs[1]).to_vec();
        }
        for (s, r) in state.iter().zip(raw.iter()) {
            assert!((s - r).abs() < 0.01, "State {s} should converge to raw {r}");
        }
    }

    #[test]
    fn stateful_outputs_two_buffers() {
        let params = make_params(0.5);
        let spectrum = flat_spectrum_bytes(257, 1.0);
        let state = vec![0u8; 3 * 4];
        let mut outputs = vec![Vec::new(), Vec::new()];
        dispatch_band_extract(&params, &[&state, &spectrum], &mut outputs).unwrap();
        assert!(!outputs[0].is_empty());
        assert!(!outputs[1].is_empty());
    }

    #[test]
    fn zero_spectrum_gives_zero_energies() {
        let params = make_params(0.0);
        let spectrum = flat_spectrum_bytes(257, 0.0);
        let mut outputs = vec![Vec::new()];
        dispatch_band_extract(&params, &[&spectrum], &mut outputs).unwrap();
        let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
        for &e in energies {
            assert_eq!(e, 0.0);
        }
    }

    #[test]
    fn invalid_spectrum_bytes_returns_error() {
        let params = make_params(0.0);
        let bad = vec![0u8; 7]; // not a multiple of 4
        let mut outputs = vec![Vec::new()];
        let result = dispatch_band_extract(&params, &[&bad], &mut outputs);
        assert!(result.is_err());
    }

    #[test]
    fn single_band_full_spectrum() {
        let bands = vec![BandDef::new(0.0, 22050.0, "full").unwrap()];
        let params = BandExtractParams::new(bands, 44100.0, 0.0).unwrap();
        let spectrum = flat_spectrum_bytes(1025, 1.0);
        let mut outputs = vec![Vec::new()];
        dispatch_band_extract(&params, &[&spectrum], &mut outputs).unwrap();
        let energies: &[f32] = bytemuck::cast_slice(&outputs[0]);
        assert_eq!(energies.len(), 1);
        assert!(energies[0] > 0.0);
    }
}
