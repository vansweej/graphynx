//! Live microphone capture via `cpal`.
//!
//! [`CpalCapture`] opens the default audio input device, starts a non-blocking
//! input stream, and accumulates samples in a lock-free [`RingBuffer`].  The
//! render thread calls [`AudioSource::next_frame`] to drain a complete frame
//! whenever enough samples have accumulated.
//!
//! # Threading model
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  cpal audio thread                                           │
//!  │  cpal_callback(data) → downmix → RingBuffer::push()         │
//!  └───────────────────────────────┬──────────────────────────────┘
//!                                  │  Arc<RingBuffer<f32>>
//!  ┌───────────────────────────────▼──────────────────────────────┐
//!  │  render / game-loop thread                                   │
//!  │  CpalCapture::next_frame() → RingBuffer::pop_into()          │
//!  └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! The ring buffer is sized to `frame_size * 8` to absorb jitter between the
//! cpal callback cadence and the render cadence.

use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::{ringbuf::RingBuffer, AudioConfig, AudioError, AudioSource};

// ── CpalCapture ───────────────────────────────────────────────────────────────

/// A live audio source backed by the system's default microphone.
///
/// Construct with [`CpalCapture::new`].  If no input device is available,
/// `new` returns [`AudioError::NoDevice`] and the caller should fall back to
/// [`SynthSource`](super::SynthSource).
///
/// # Send safety
///
/// `cpal::Stream` is not `Send` on all platforms (notably Linux/ALSA uses a
/// raw pointer internally).  `CpalCapture` is constructed on the render thread
/// and must only ever be used from that same thread — it is never moved across
/// thread boundaries after construction.  The `Arc<RingBuffer>` that the cpal
/// callback shares is `Send + Sync` independently.
pub struct CpalCapture {
    /// Kept alive so the stream is not dropped.
    _stream: cpal::Stream,
    ring: Arc<RingBuffer<f32>>,
    frame_buf: Vec<f32>,
    config: AudioConfig,
}

// SAFETY: CpalCapture is constructed and used exclusively on the render thread.
// The cpal::Stream is never moved to another thread after construction.
// The shared RingBuffer is Send + Sync via Arc + AtomicUsize.
unsafe impl Send for CpalCapture {}

impl CpalCapture {
    /// Open the default audio input device and start capturing.
    ///
    /// # Errors
    ///
    /// - [`AudioError::NoDevice`] — no input device found.
    /// - [`AudioError::UnsupportedRate`] — the device does not support the
    ///   requested sample rate.
    /// - [`AudioError::Stream`] — the stream could not be built or started.
    #[cfg(not(tarpaulin_include))]
    pub fn new(config: AudioConfig) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        // Find a supported f32 stream config at the requested sample rate.
        let supported = device
            .supported_input_configs()
            .map_err(|e| AudioError::Stream(e.to_string()))?
            .find(|c| {
                c.sample_format() == cpal::SampleFormat::F32
                    && c.min_sample_rate().0 <= config.sample_rate
                    && c.max_sample_rate().0 >= config.sample_rate
            })
            .ok_or(AudioError::UnsupportedRate(config.sample_rate))?
            .with_sample_rate(cpal::SampleRate(config.sample_rate));

        let channels = supported.channels() as usize;
        let ring = RingBuffer::<f32>::new(config.frame_size * 8);
        let ring_cb = Arc::clone(&ring);

        let stream = device
            .build_input_stream(
                &supported.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Downmix to mono and push each sample.
                    for chunk in data.chunks(channels.max(1)) {
                        let mono = chunk.iter().sum::<f32>() / channels as f32;
                        ring_cb.push(mono);
                    }
                },
                |err| {
                    log::error!("cpal input stream error: {err}");
                },
                None,
            )
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::Stream(e.to_string()))?;

        let frame_buf = vec![0.0_f32; config.frame_size];
        Ok(Self {
            _stream: stream,
            ring,
            frame_buf,
            config,
        })
    }
}

impl AudioSource for CpalCapture {
    /// Return the next complete frame, or `None` if fewer than `frame_size`
    /// samples have accumulated since the last call.
    fn next_frame(&mut self) -> Option<&[f32]> {
        if self.ring.available() >= self.config.frame_size {
            self.ring.pop_into(&mut self.frame_buf);
            Some(&self.frame_buf)
        } else {
            None
        }
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
    use crate::audio::AudioConfig;

    /// Verify that `CpalCapture::new` returns `AudioError::NoDevice` (or
    /// succeeds) without panicking.  In headless CI there is no input device,
    /// so we only assert it does not panic.
    ///
    /// This test is excluded from tarpaulin because it depends on hardware.
    #[test]
    #[cfg(not(tarpaulin_include))]
    fn new_does_not_panic_in_any_environment() {
        let config = AudioConfig::mono_44100(1024);
        let _ = CpalCapture::new(config); // Ok or Err — both are fine
    }
}
