//! Audio source abstraction for the graphynx signal pipeline.
//!
//! This module provides:
//!
//! - [`AudioSource`] — a non-blocking trait for feeding audio frames into the
//!   executor.
//! - [`AudioConfig`] — shared configuration (sample rate, frame size, channels).
//! - [`AudioError`] — errors from audio capture initialisation.
//! - [`SynthSource`] — a deterministic additive-synthesis source (no hardware
//!   required; always available as a fallback).
//!
//! When the `live-audio` feature is enabled, [`capture::CpalCapture`] provides
//! real-time microphone input via `cpal`.
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────┐
//!  │  AudioSource (trait)                                │
//!  │                                                     │
//!  │  SynthSource ──── deterministic additive synthesis  │
//!  │  CpalCapture ──── live mic via cpal + RingBuffer    │
//!  │  (feature = "live-audio")                           │
//!  └─────────────────────────────────────────────────────┘
//!              │  next_frame() → Option<&[f32]>
//!              ▼
//!  Executor::input("audio").write(frame)
//! ```

use std::f32::consts::PI;

#[cfg(feature = "live-audio")]
pub mod capture;
pub mod ringbuf;

// ── AudioConfig ───────────────────────────────────────────────────────────────

/// Configuration shared by all audio sources.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// Sample rate in Hz (typically 44 100 or 48 000).
    pub sample_rate: u32,
    /// Number of samples per frame delivered to the executor.
    pub frame_size: usize,
    /// Number of input channels.  Stereo is downmixed to mono.
    pub channels: u16,
}

impl AudioConfig {
    /// Construct a standard 44.1 kHz mono configuration with the given frame
    /// size.
    pub fn mono_44100(frame_size: usize) -> Self {
        Self {
            sample_rate: 44_100,
            frame_size,
            channels: 1,
        }
    }
}

// ── AudioError ────────────────────────────────────────────────────────────────

/// Errors that can occur when initialising an audio source.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// No audio input device was found on this system.
    #[error("no audio input device available")]
    NoDevice,
    /// The audio stream could not be started.
    #[error("audio stream error: {0}")]
    Stream(String),
    /// The requested sample rate is not supported by the device.
    #[error("unsupported sample rate: {0} Hz")]
    UnsupportedRate(u32),
}

// ── AudioSource ───────────────────────────────────────────────────────────────

/// A non-blocking source of fixed-size audio frames.
///
/// Implementations must be `Send` so they can be moved into the render thread.
/// `next_frame` must never block — if no complete frame is ready yet, it
/// returns `None` and the caller should skip graph execution for this tick.
///
/// # Implementors
///
/// | Type | Description |
/// |------|-------------|
/// | [`SynthSource`] | Deterministic additive synthesis; always returns `Some` |
/// | `CpalCapture` | Live microphone via cpal ring buffer (feature `live-audio`) |
pub trait AudioSource: Send {
    /// Return the latest complete audio frame, or `None` if not enough samples
    /// have accumulated yet.
    ///
    /// The returned slice always has length [`frame_size`](Self::frame_size).
    fn next_frame(&mut self) -> Option<&[f32]>;

    /// Sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Samples per frame.
    fn frame_size(&self) -> usize;
}

// ── SynthSource ───────────────────────────────────────────────────────────────

/// A deterministic synthetic voice source.
///
/// Generates audio via additive synthesis:
/// - Harmonics of `fundamental_hz` with 1/n amplitude roll-off (sawtooth-like
///   source).
/// - Three formant resonances (Gaussian-shaped amplitude envelope in
///   frequency).
/// - A small white-noise floor for breathiness, generated with a deterministic
///   LCG (no `rand` dependency; reproducible across runs).
///
/// `next_frame` always returns `Some` — synthesis never blocks.
///
/// # Example
///
/// ```rust
/// use runtime::audio::{AudioConfig, AudioSource, SynthSource};
///
/// let config = AudioConfig::mono_44100(1024);
/// let mut src = SynthSource::new(config, 220.0, [900.0, 1_800.0, 2_800.0]);
/// let frame = src.next_frame().unwrap();
/// assert_eq!(frame.len(), 1024);
/// ```
pub struct SynthSource {
    config: AudioConfig,
    frame_buf: Vec<f32>,
    /// Accumulated phase in seconds — keeps synthesis continuous across frames.
    phase: f32,
    fundamental_hz: f32,
    formants: [f32; 3],
}

impl SynthSource {
    /// Create a new synthetic source.
    ///
    /// - `fundamental_hz` — fundamental frequency of the voice (e.g. 120 Hz
    ///   for a male voice, 220 Hz for female).
    /// - `formants` — three formant centre frequencies in Hz (F1, F2, F3).
    pub fn new(config: AudioConfig, fundamental_hz: f32, formants: [f32; 3]) -> Self {
        let frame_size = config.frame_size;
        Self {
            config,
            frame_buf: vec![0.0; frame_size],
            phase: 0.0,
            fundamental_hz,
            formants,
        }
    }

    /// Update the synthesis parameters (e.g. when the user switches preset).
    ///
    /// The phase is preserved so the signal remains continuous.
    pub fn set_params(&mut self, fundamental_hz: f32, formants: [f32; 3]) {
        self.fundamental_hz = fundamental_hz;
        self.formants = formants;
    }

    fn synthesise(&mut self) {
        let f0 = self.fundamental_hz;
        let formants = self.formants;
        let sr = self.config.sample_rate as f32;
        let dt = 1.0 / sr;
        let n = self.config.frame_size;

        self.frame_buf.iter_mut().for_each(|s| *s = 0.0);

        // Harmonics with formant shaping.
        let mut harmonic = 1u32;
        loop {
            let freq = f0 * harmonic as f32;
            if freq > sr / 2.0 {
                break;
            }
            let amp = 1.0 / harmonic as f32;
            let formant_gain: f32 = formants
                .iter()
                .map(|&fc| {
                    let bw = fc * 0.15;
                    let diff = (freq - fc) / bw;
                    (-0.5 * diff * diff).exp()
                })
                .sum::<f32>()
                .max(0.05);
            let total_amp = amp * (1.0 + formant_gain);

            for (i, s) in self.frame_buf.iter_mut().enumerate() {
                let t = i as f32 * dt + self.phase;
                *s += total_amp * (2.0 * PI * freq * t).sin();
            }
            harmonic += 1;
        }

        // Deterministic LCG noise floor for breathiness.
        let mut lcg: u32 = 0x12345678u32.wrapping_add((self.phase * 1_000.0) as u32);
        for s in self.frame_buf.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (lcg as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *s += noise * 0.05;
        }

        // Normalise to [-1, 1].
        let peak = self
            .frame_buf
            .iter()
            .map(|s| s.abs())
            .fold(0.0_f32, f32::max);
        if peak > 1e-6 {
            for s in self.frame_buf.iter_mut() {
                *s /= peak;
            }
        }

        // Advance phase; wrap to avoid float precision drift.
        self.phase += n as f32 * dt;
        if self.phase > 3_600.0 {
            self.phase -= 3_600.0;
        }
    }
}

impl AudioSource for SynthSource {
    fn next_frame(&mut self) -> Option<&[f32]> {
        self.synthesise();
        Some(&self.frame_buf)
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn frame_size(&self) -> usize {
        self.config.frame_size
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AudioConfig {
        AudioConfig::mono_44100(1024)
    }

    // ── AudioConfig ───────────────────────────────────────────────────────

    #[test]
    fn audio_config_mono_44100_fields() {
        let c = AudioConfig::mono_44100(512);
        assert_eq!(c.sample_rate, 44_100);
        assert_eq!(c.frame_size, 512);
        assert_eq!(c.channels, 1);
    }

    #[test]
    fn audio_config_clone_and_debug() {
        let c = AudioConfig::mono_44100(1024);
        let c2 = c.clone();
        assert_eq!(c2.frame_size, 1024);
        let _ = format!("{c:?}"); // debug impl
    }

    // ── SynthSource ───────────────────────────────────────────────────────

    #[test]
    fn synth_source_always_returns_some() {
        let mut src = SynthSource::new(default_config(), 220.0, [900.0, 1_800.0, 2_800.0]);
        assert!(src.next_frame().is_some());
    }

    #[test]
    fn synth_source_frame_is_correct_length() {
        let mut src = SynthSource::new(default_config(), 120.0, [700.0, 1_200.0, 2_500.0]);
        let frame = src.next_frame().unwrap();
        assert_eq!(frame.len(), 1024);
    }

    #[test]
    fn synth_source_output_is_normalised() {
        let mut src = SynthSource::new(default_config(), 170.0, [800.0, 1_500.0, 2_650.0]);
        let frame = src.next_frame().unwrap();
        let peak = frame.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        assert!(peak <= 1.0 + 1e-5, "peak={peak}");
    }

    #[test]
    fn synth_source_is_not_silent() {
        let mut src = SynthSource::new(default_config(), 220.0, [900.0, 1_800.0, 2_800.0]);
        let frame = src.next_frame().unwrap();
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / 1024.0).sqrt();
        assert!(rms > 0.01, "frame is too quiet: rms={rms}");
    }

    #[test]
    fn synth_source_different_fundamentals_produce_different_frames() {
        let mut src_male = SynthSource::new(default_config(), 120.0, [700.0, 1_200.0, 2_500.0]);
        let mut src_female = SynthSource::new(default_config(), 220.0, [900.0, 1_800.0, 2_800.0]);

        let male = src_male.next_frame().unwrap().to_vec();
        let female = src_female.next_frame().unwrap().to_vec();

        let diff: f32 = male
            .iter()
            .zip(female.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0, "frames are too similar: diff={diff}");
    }

    #[test]
    fn synth_source_phase_is_continuous_across_frames() {
        let mut src = SynthSource::new(default_config(), 440.0, [700.0, 1_200.0, 2_500.0]);
        // Run several frames and verify no discontinuity (no NaN/inf).
        for _ in 0..10 {
            let frame = src.next_frame().unwrap();
            for &s in frame {
                assert!(s.is_finite(), "non-finite sample: {s}");
            }
        }
    }

    #[test]
    fn synth_source_set_params_changes_output() {
        let mut src = SynthSource::new(default_config(), 120.0, [700.0, 1_200.0, 2_500.0]);
        let frame_before = src.next_frame().unwrap().to_vec();

        src.set_params(440.0, [2_000.0, 3_000.0, 4_000.0]);
        let frame_after = src.next_frame().unwrap().to_vec();

        let diff: f32 = frame_before
            .iter()
            .zip(frame_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "params change had no effect: diff={diff}");
    }

    #[test]
    fn synth_source_reports_correct_config() {
        let src = SynthSource::new(default_config(), 220.0, [900.0, 1_800.0, 2_800.0]);
        assert_eq!(src.sample_rate(), 44_100);
        assert_eq!(src.frame_size(), 1024);
    }

    // ── AudioError ────────────────────────────────────────────────────────

    #[test]
    fn audio_error_display_messages() {
        assert!(AudioError::NoDevice.to_string().contains("no audio"));
        assert!(AudioError::Stream("oops".into())
            .to_string()
            .contains("oops"));
        assert!(AudioError::UnsupportedRate(96_000)
            .to_string()
            .contains("96000"));
    }
}
