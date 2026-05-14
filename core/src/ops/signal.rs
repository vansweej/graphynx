//! Signal processing operation parameters.
//!
//! This module provides the parameter types for the three signal-processing
//! operations in the graphynx op catalog:
//!
//! | Op | Purpose |
//! |----|---------|
//! | [`WindowParams`] | Apply a windowing function to a raw audio frame before FFT |
//! | [`FftParams`] | Compute a Fast Fourier Transform (forward or inverse) |
//! | [`BandExtractParams`] | Extract per-band energy from a magnitude/power spectrum |
//!
//! ## Audio pipeline
//!
//! The three ops compose into a standard short-time spectral analysis pipeline:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                   Audio analysis pipeline                    │
//! │                                                              │
//! │  [N] f32          [N] f32        [N/2+1] f32    [B] f32     │
//! │  Audio frame ──► Window ──────► FFT ──────────► BandExtract │
//! │  (raw PCM)        (Hann etc.)   (Magnitude)     (EMA)       │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Stateful operation
//!
//! [`BandExtractParams`] with `smoothing > 0.0` is the first *stateful* op in
//! the catalog. The executor threads an exponential moving-average (EMA) state
//! vector through it across ticks:
//!
//! ```text
//! Tick t:
//!   inputs  = [ema_state_t-1, spectrum_t]
//!   outputs = [band_energies_t, ema_state_t]
//!
//! EMA formula:  y[t] = α × x[t] + (1 − α) × y[t−1]
//!   where α = smoothing ∈ (0.0, 1.0)
//! ```
//!
//! When `smoothing == 0.0` the op is stateless and the executor does not
//! prepend/append state buffers.

use super::OpError;

// ── Window ────────────────────────────────────────────────────────────────────

/// Which windowing function to apply to a frame before FFT.
///
/// Windowing reduces spectral leakage by tapering the frame edges to zero.
/// The choice of window trades off frequency resolution against side-lobe
/// attenuation:
///
/// | Window | Side-lobe attenuation | Main-lobe width |
/// |--------|-----------------------|-----------------|
/// | Hann | −31.5 dB | 4 bins |
/// | Hamming | −41.0 dB | 4 bins |
/// | Blackman | −58.1 dB | 6 bins |
///
/// For voice-frequency analysis the **Hann** window is a good default.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// Hann (von Hann) window: `w[n] = 0.5 × (1 − cos(2πn / (N−1)))`.
    ///
    /// Good general-purpose choice. Widely used for audio analysis.
    Hann,
    /// Hamming window: `w[n] = 0.54 − 0.46 × cos(2πn / (N−1))`.
    ///
    /// Slightly better side-lobe attenuation than Hann at the cost of
    /// non-zero endpoints (does not taper fully to zero).
    Hamming,
    /// Blackman window: `w[n] = 0.42 − 0.5 × cos(2πn/(N−1)) + 0.08 × cos(4πn/(N−1))`.
    ///
    /// Highest side-lobe attenuation of the three; wider main lobe.
    Blackman,
}

/// Parameters for [`Op::Window`](super::super::ops::Op::Window).
///
/// Applies a windowing function element-wise to a 1-D frame of `size` f32
/// samples. The output has the same shape as the input.
///
/// # Errors
///
/// [`WindowParams::new`] returns [`OpError::ZeroWindowSize`] if `size == 0`.
///
/// # Examples
///
/// ```
/// use graph_core::ops::signal::{WindowKind, WindowParams};
///
/// let p = WindowParams::new(WindowKind::Hann, 2048).unwrap();
/// assert_eq!(p.size, 2048);
/// assert_eq!(p.kind, WindowKind::Hann);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowParams {
    /// Which windowing function to apply.
    pub kind: WindowKind,
    /// Frame length in samples. Must be > 0.
    pub size: usize,
}

impl WindowParams {
    /// Construct validated window parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::ZeroWindowSize`] if `size == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::{WindowKind, WindowParams};
    ///
    /// assert!(WindowParams::new(WindowKind::Hann, 2048).is_ok());
    /// assert!(WindowParams::new(WindowKind::Hann, 0).is_err());
    /// ```
    pub fn new(kind: WindowKind, size: usize) -> Result<Self, OpError> {
        if size == 0 {
            return Err(OpError::ZeroWindowSize);
        }
        Ok(Self { kind, size })
    }
}

// ── FFT ───────────────────────────────────────────────────────────────────────

/// Direction of the Fourier transform.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward DFT: time domain → frequency domain.
    Forward,
    /// Inverse DFT: frequency domain → time domain.
    ///
    /// Note: the output is complex-valued. Real-output optimisation
    /// (via a real-FFT plan) is deferred to a future phase.
    Inverse,
}

/// What the FFT output tensor contains.
///
/// The three modes differ in output shape and dtype:
///
/// ```text
/// ┌──────────────────────────────────────────────────────────────┐
/// │  Real input [N] f32  ──►  rustfft  ──►  Complex [N]         │
/// │                                                              │
/// │  Complex  →  [N] f32 interleaved  (re₀, im₀, re₁, im₁, …)  │
/// │  Magnitude →  [N/2+1] f32         √(re² + im²)              │
/// │  Power     →  [N/2+1] f32         re² + im²                 │
/// └──────────────────────────────────────────────────────────────┘
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FftOutput {
    /// Full complex spectrum stored as interleaved f32 pairs.
    ///
    /// Output shape: `[size]`, dtype: f32 (re, im interleaved).
    /// Byte length: `size * 2 * 4`.
    Complex,
    /// One-sided magnitude spectrum: `√(re² + im²)`.
    ///
    /// Output shape: `[size / 2 + 1]`, dtype: f32.
    Magnitude,
    /// One-sided power spectrum: `re² + im²`.
    ///
    /// Output shape: `[size / 2 + 1]`, dtype: f32.
    Power,
}

/// Parameters for [`Op::Fft`](super::super::ops::Op::Fft).
///
/// Computes a Fast Fourier Transform of a 1-D real input frame.
///
/// ## Bin-to-frequency mapping
///
/// For a forward FFT of size `N` at sample rate `sr`:
///
/// ```text
/// frequency(bin) = bin × sr / N
/// ```
///
/// The one-sided spectrum contains bins `0 … N/2` (inclusive), giving
/// `N/2 + 1` output values.
///
/// ## Performance note
///
/// Power-of-two sizes (512, 1024, 2048, 4096, …) are processed most
/// efficiently by `rustfft`. Non-power-of-two sizes are correct but slower.
///
/// # Errors
///
/// [`FftParams::new`] returns [`OpError::ZeroFftSize`] if `size == 0`.
///
/// # Examples
///
/// ```
/// use graph_core::ops::signal::{FftDirection, FftOutput, FftParams};
///
/// let p = FftParams::new(2048, FftDirection::Forward, FftOutput::Magnitude).unwrap();
/// assert_eq!(p.size, 2048);
/// assert_eq!(p.one_sided_len(), 1025);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FftParams {
    /// Number of samples in the input frame. Must be > 0.
    ///
    /// Power-of-two values (512, 1024, 2048, …) are recommended for
    /// performance.
    pub size: usize,
    /// Transform direction.
    pub direction: FftDirection,
    /// What the output tensor contains.
    pub output: FftOutput,
}

impl FftParams {
    /// Construct validated FFT parameters.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::ZeroFftSize`] if `size == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::{FftDirection, FftOutput, FftParams};
    ///
    /// assert!(FftParams::new(1024, FftDirection::Forward, FftOutput::Magnitude).is_ok());
    /// assert!(FftParams::new(0, FftDirection::Forward, FftOutput::Magnitude).is_err());
    /// ```
    pub fn new(size: usize, direction: FftDirection, output: FftOutput) -> Result<Self, OpError> {
        if size == 0 {
            return Err(OpError::ZeroFftSize);
        }
        Ok(Self {
            size,
            direction,
            output,
        })
    }

    /// Length of the one-sided (real-input) spectrum: `size / 2 + 1`.
    ///
    /// This is the output length for [`FftOutput::Magnitude`] and
    /// [`FftOutput::Power`] modes.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::{FftDirection, FftOutput, FftParams};
    ///
    /// let p = FftParams::new(2048, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    /// assert_eq!(p.one_sided_len(), 1025);
    ///
    /// let p2 = FftParams::new(512, FftDirection::Forward, FftOutput::Power).unwrap();
    /// assert_eq!(p2.one_sided_len(), 257);
    /// ```
    pub fn one_sided_len(&self) -> usize {
        self.size / 2 + 1
    }
}

// ── BandExtract ───────────────────────────────────────────────────────────────

/// Definition of a single frequency band for energy extraction.
///
/// A band covers the half-open interval `[low_hz, high_hz)` in the frequency
/// domain. Any FFT bin whose centre frequency falls within this interval
/// contributes to the band's energy sum.
///
/// ## Bin inclusion rule
///
/// ```text
/// bin_low  = floor(low_hz  × fft_size / sample_rate)
/// bin_high = ceil (high_hz × fft_size / sample_rate)
/// included bins: bin_low ..= min(bin_high, fft_size/2)
/// ```
///
/// If no bins fall within a band (e.g. the band is narrower than one bin),
/// the output for that band is `0.0`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BandDef {
    /// Lower bound of the band in Hz. Must be ≥ 0 and < `high_hz`.
    pub low_hz: f32,
    /// Upper bound of the band in Hz. Must be > `low_hz`.
    pub high_hz: f32,
    /// Human-readable label (e.g. `"low"`, `"mid"`, `"high"`).
    ///
    /// Used for logging and visualisation; not interpreted by the backend.
    pub label: String,
}

impl BandDef {
    /// Construct a validated band definition.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::InvalidBandRange`] if `low_hz >= high_hz` or
    /// `low_hz < 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::BandDef;
    ///
    /// assert!(BandDef::new(80.0, 300.0, "low").is_ok());
    /// assert!(BandDef::new(300.0, 80.0, "bad").is_err()); // low >= high
    /// assert!(BandDef::new(-1.0, 300.0, "bad").is_err()); // negative low
    /// ```
    pub fn new(low_hz: f32, high_hz: f32, label: impl Into<String>) -> Result<Self, OpError> {
        if low_hz < 0.0 || low_hz >= high_hz {
            return Err(OpError::InvalidBandRange {
                low: low_hz,
                high: high_hz,
            });
        }
        Ok(Self {
            low_hz,
            high_hz,
            label: label.into(),
        })
    }
}

/// Parameters for [`Op::BandExtract`](super::super::ops::Op::BandExtract).
///
/// Extracts per-band energy from a magnitude or power spectrum produced by
/// [`Op::Fft`](super::super::ops::Op::Fft) with [`FftOutput::Magnitude`] or
/// [`FftOutput::Power`].
///
/// ## Smoothing (EMA)
///
/// When `smoothing > 0.0` the op is **stateful**. The executor threads an EMA
/// state vector (one f32 per band) through the node across ticks:
///
/// ```text
/// y[t] = α × x[t] + (1 − α) × y[t−1]
/// ```
///
/// where `α = smoothing`. Higher values respond faster to changes; lower
/// values produce a smoother, more sluggish output.
///
/// When `smoothing == 0.0` the op is **stateless** and outputs raw bin sums.
///
/// ## State convention (executor protocol)
///
/// ```text
/// Stateful (smoothing > 0):
///   inputs[0]  = EMA state  [bands.len()] f32  (zeros on first tick)
///   inputs[1]  = spectrum   [fft_size/2+1] f32
///   outputs[0] = energies   [bands.len()] f32
///   outputs[1] = new state  [bands.len()] f32
///
/// Stateless (smoothing == 0):
///   inputs[0]  = spectrum   [fft_size/2+1] f32
///   outputs[0] = energies   [bands.len()] f32
/// ```
///
/// # Errors
///
/// [`BandExtractParams::new`] returns:
/// - [`OpError::EmptyBands`] if `bands` is empty.
/// - [`OpError::InvalidSampleRate`] if `sample_rate_hz <= 0.0`.
/// - [`OpError::InvalidSmoothing`] if `smoothing` is not in `[0.0, 1.0)`.
///
/// # Examples
///
/// ```
/// use graph_core::ops::signal::{BandDef, BandExtractParams};
///
/// let bands = vec![
///     BandDef::new(80.0,  300.0, "low").unwrap(),
///     BandDef::new(300.0, 2000.0, "mid").unwrap(),
///     BandDef::new(2000.0, 8000.0, "high").unwrap(),
/// ];
/// let p = BandExtractParams::new(bands, 44100.0, 0.6).unwrap();
/// assert_eq!(p.bands.len(), 3);
/// assert!(p.is_stateful());
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BandExtractParams {
    /// Frequency bands to extract. Must not be empty.
    pub bands: Vec<BandDef>,
    /// Sample rate of the original audio in Hz. Must be > 0.
    pub sample_rate_hz: f32,
    /// EMA smoothing factor `α ∈ [0.0, 1.0)`.
    ///
    /// `0.0` = no smoothing (stateless); values closer to `1.0` respond
    /// faster to transients. Typical voice-reactive values: `0.4`–`0.7`.
    pub smoothing: f32,
}

impl BandExtractParams {
    /// Construct validated band-extraction parameters.
    ///
    /// # Errors
    ///
    /// - [`OpError::EmptyBands`] — `bands` is empty.
    /// - [`OpError::InvalidSampleRate`] — `sample_rate_hz <= 0.0`.
    /// - [`OpError::InvalidSmoothing`] — `smoothing` not in `[0.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::{BandDef, BandExtractParams};
    ///
    /// let bands = vec![BandDef::new(80.0, 8000.0, "all").unwrap()];
    ///
    /// // Valid — stateful
    /// assert!(BandExtractParams::new(bands.clone(), 44100.0, 0.5).is_ok());
    ///
    /// // Valid — stateless (smoothing == 0)
    /// assert!(BandExtractParams::new(bands.clone(), 44100.0, 0.0).is_ok());
    ///
    /// // Invalid — smoothing must be < 1.0
    /// assert!(BandExtractParams::new(bands.clone(), 44100.0, 1.0).is_err());
    ///
    /// // Invalid — empty bands
    /// assert!(BandExtractParams::new(vec![], 44100.0, 0.5).is_err());
    /// ```
    pub fn new(bands: Vec<BandDef>, sample_rate_hz: f32, smoothing: f32) -> Result<Self, OpError> {
        if bands.is_empty() {
            return Err(OpError::EmptyBands);
        }
        if sample_rate_hz <= 0.0 {
            return Err(OpError::InvalidSampleRate(sample_rate_hz));
        }
        if !(0.0..1.0).contains(&smoothing) {
            return Err(OpError::InvalidSmoothing(smoothing));
        }
        Ok(Self {
            bands,
            sample_rate_hz,
            smoothing,
        })
    }

    /// Returns `true` when `smoothing > 0.0`, meaning the op maintains EMA
    /// state across executor ticks.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::ops::signal::{BandDef, BandExtractParams};
    ///
    /// let band = vec![BandDef::new(80.0, 8000.0, "all").unwrap()];
    /// let stateful  = BandExtractParams::new(band.clone(), 44100.0, 0.5).unwrap();
    /// let stateless = BandExtractParams::new(band,         44100.0, 0.0).unwrap();
    ///
    /// assert!(stateful.is_stateful());
    /// assert!(!stateless.is_stateful());
    /// ```
    pub fn is_stateful(&self) -> bool {
        self.smoothing > 0.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WindowParams ─────────────────────────────────────────────────────

    #[test]
    fn window_params_valid() {
        let p = WindowParams::new(WindowKind::Hann, 2048).unwrap();
        assert_eq!(p.size, 2048);
        assert_eq!(p.kind, WindowKind::Hann);
    }

    #[test]
    fn window_params_zero_size_error() {
        assert!(matches!(
            WindowParams::new(WindowKind::Hann, 0),
            Err(OpError::ZeroWindowSize)
        ));
    }

    #[test]
    fn window_params_all_kinds() {
        for kind in [WindowKind::Hann, WindowKind::Hamming, WindowKind::Blackman] {
            assert!(WindowParams::new(kind, 512).is_ok());
        }
    }

    #[test]
    fn window_params_clone_eq() {
        let a = WindowParams::new(WindowKind::Hamming, 1024).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── FftParams ────────────────────────────────────────────────────────

    #[test]
    fn fft_params_valid() {
        let p = FftParams::new(2048, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        assert_eq!(p.size, 2048);
        assert_eq!(p.direction, FftDirection::Forward);
        assert_eq!(p.output, FftOutput::Magnitude);
    }

    #[test]
    fn fft_params_zero_size_error() {
        assert!(matches!(
            FftParams::new(0, FftDirection::Forward, FftOutput::Magnitude),
            Err(OpError::ZeroFftSize)
        ));
    }

    #[test]
    fn fft_params_one_sided_len_power_of_two() {
        let p = FftParams::new(2048, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        assert_eq!(p.one_sided_len(), 1025);
    }

    #[test]
    fn fft_params_one_sided_len_odd() {
        let p = FftParams::new(513, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        assert_eq!(p.one_sided_len(), 257);
    }

    #[test]
    fn fft_params_all_output_modes() {
        for output in [FftOutput::Complex, FftOutput::Magnitude, FftOutput::Power] {
            assert!(FftParams::new(1024, FftDirection::Forward, output).is_ok());
        }
    }

    #[test]
    fn fft_params_inverse_direction() {
        let p = FftParams::new(512, FftDirection::Inverse, FftOutput::Complex).unwrap();
        assert_eq!(p.direction, FftDirection::Inverse);
    }

    #[test]
    fn fft_params_clone_eq() {
        let a = FftParams::new(1024, FftDirection::Forward, FftOutput::Power).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── BandDef ──────────────────────────────────────────────────────────

    #[test]
    fn band_def_valid() {
        let b = BandDef::new(80.0, 300.0, "low").unwrap();
        assert_eq!(b.low_hz, 80.0);
        assert_eq!(b.high_hz, 300.0);
        assert_eq!(b.label, "low");
    }

    #[test]
    fn band_def_low_equals_high_error() {
        assert!(matches!(
            BandDef::new(300.0, 300.0, "bad"),
            Err(OpError::InvalidBandRange { .. })
        ));
    }

    #[test]
    fn band_def_low_greater_than_high_error() {
        assert!(matches!(
            BandDef::new(500.0, 100.0, "bad"),
            Err(OpError::InvalidBandRange { .. })
        ));
    }

    #[test]
    fn band_def_negative_low_error() {
        assert!(matches!(
            BandDef::new(-1.0, 300.0, "bad"),
            Err(OpError::InvalidBandRange { .. })
        ));
    }

    #[test]
    fn band_def_zero_low_valid() {
        assert!(BandDef::new(0.0, 100.0, "sub").is_ok());
    }

    #[test]
    fn band_def_clone_eq() {
        let a = BandDef::new(80.0, 300.0, "low").unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── BandExtractParams ────────────────────────────────────────────────

    fn three_bands() -> Vec<BandDef> {
        vec![
            BandDef::new(80.0, 300.0, "low").unwrap(),
            BandDef::new(300.0, 2000.0, "mid").unwrap(),
            BandDef::new(2000.0, 8000.0, "high").unwrap(),
        ]
    }

    #[test]
    fn band_extract_valid_stateful() {
        let p = BandExtractParams::new(three_bands(), 44100.0, 0.6).unwrap();
        assert_eq!(p.bands.len(), 3);
        assert_eq!(p.sample_rate_hz, 44100.0);
        assert_eq!(p.smoothing, 0.6);
        assert!(p.is_stateful());
    }

    #[test]
    fn band_extract_valid_stateless() {
        let p = BandExtractParams::new(three_bands(), 44100.0, 0.0).unwrap();
        assert!(!p.is_stateful());
    }

    #[test]
    fn band_extract_empty_bands_error() {
        assert!(matches!(
            BandExtractParams::new(vec![], 44100.0, 0.5),
            Err(OpError::EmptyBands)
        ));
    }

    #[test]
    fn band_extract_zero_sample_rate_error() {
        assert!(matches!(
            BandExtractParams::new(three_bands(), 0.0, 0.5),
            Err(OpError::InvalidSampleRate(_))
        ));
    }

    #[test]
    fn band_extract_negative_sample_rate_error() {
        assert!(matches!(
            BandExtractParams::new(three_bands(), -44100.0, 0.5),
            Err(OpError::InvalidSampleRate(_))
        ));
    }

    #[test]
    fn band_extract_smoothing_one_error() {
        assert!(matches!(
            BandExtractParams::new(three_bands(), 44100.0, 1.0),
            Err(OpError::InvalidSmoothing(_))
        ));
    }

    #[test]
    fn band_extract_smoothing_greater_than_one_error() {
        assert!(matches!(
            BandExtractParams::new(three_bands(), 44100.0, 1.5),
            Err(OpError::InvalidSmoothing(_))
        ));
    }

    #[test]
    fn band_extract_smoothing_negative_error() {
        assert!(matches!(
            BandExtractParams::new(three_bands(), 44100.0, -0.1),
            Err(OpError::InvalidSmoothing(_))
        ));
    }

    #[test]
    fn band_extract_clone_eq() {
        let a = BandExtractParams::new(three_bands(), 44100.0, 0.5).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn band_extract_single_band_valid() {
        let bands = vec![BandDef::new(0.0, 22050.0, "full").unwrap()];
        assert!(BandExtractParams::new(bands, 44100.0, 0.0).is_ok());
    }
}
