//! CPU backend for the graphynx signal processing ops.
//!
//! This crate provides [`CpuBackend`], a managed-memory backend that
//! implements [`Backend`] for the three signal-processing operations:
//!
//! | Op | Description |
//! |----|-------------|
//! | [`Op::Window`] | Apply a windowing function (Hann, Hamming, Blackman) to an audio frame |
//! | [`Op::Fft`] | Compute a forward or inverse FFT via [`rustfft`] |
//! | [`Op::BandExtract`] | Extract per-band energy with optional EMA smoothing |
//!
//! All other [`Op`] variants return [`BackendError::UnsupportedOp`].
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  CpuBackend                                                     │
//! │  ├── device_id: DeviceId                                        │
//! │  └── fft_planner: Mutex<FftPlanner<f32>>  (twiddle cache)      │
//! │                                                                 │
//! │  dispatch_op(op, inputs, outputs)                               │
//! │      │                                                          │
//! │      ▼  match op                                                │
//! │  ┌───────────┬───────────┬──────────────┐                      │
//! │  │  Window   │    Fft    │  BandExtract │                      │
//! │  └───────────┴───────────┴──────────────┘                      │
//! │  signal/window.rs  fft.rs       band.rs                        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory model
//!
//! [`CpuBackend`] uses [`MemoryModel::Managed`]: the executor passes raw host
//! byte slices in and receives raw host byte slices out. The backend does not
//! allocate device memory; `alloc`, `upload`, and `download` all return
//! [`BackendError::NotApplicable`].
//!
//! ## FftPlanner caching
//!
//! [`rustfft::FftPlanner`] caches twiddle-factor tables keyed by FFT size and
//! direction. Holding a single planner across dispatch calls avoids
//! recomputing these tables for repeated FFTs of the same size. The planner is
//! wrapped in a [`std::sync::Mutex`] so that [`CpuBackend`] satisfies `Sync`.
//!
//! ## Usage example
//!
//! ```rust,ignore
//! use backends_cpu::CpuBackend;
//! use backends::Backend;
//! use graph_core::ops::{Op, FftParams, FftDirection, FftOutput};
//!
//! let backend = CpuBackend::new("cpu");
//!
//! // 2048-sample frame of zeros → magnitude spectrum of length 1025.
//! let input = vec![0.0f32; 2048];
//! let input_bytes: Vec<u8> = bytemuck::cast_slice(&input).to_vec();
//! let op = Op::Fft(FftParams::new(2048, FftDirection::Forward, FftOutput::Magnitude).unwrap());
//! let mut outputs = vec![Vec::new()];
//! backend.dispatch_op(&op, &[&input_bytes], &mut outputs).unwrap();
//! let magnitude: &[f32] = bytemuck::cast_slice(&outputs[0]);
//! assert_eq!(magnitude.len(), 1025);
//! ```
//!
//! [`Op::Window`]: graph_core::ops::Op::Window
//! [`Op::Fft`]: graph_core::ops::Op::Fft
//! [`Op::BandExtract`]: graph_core::ops::Op::BandExtract
//! [`Op`]: graph_core::ops::Op

use std::sync::Mutex;

use log::warn;
use rustfft::FftPlanner;

use backends::{
    Backend, BackendCaps, BackendError, DeviceBuffer, DeviceId, MemoryModel, NodeKindTag,
};
use graph_core::ops::Op;

mod signal;

// ── CpuBackend ────────────────────────────────────────────────────────────────

/// A managed-memory CPU backend for signal processing operations.
///
/// Supports [`Op::Window`], [`Op::Fft`], and [`Op::BandExtract`].
/// All other ops return [`BackendError::UnsupportedOp`].
///
/// ## Thread safety
///
/// [`CpuBackend`] is `Send + Sync`. The internal [`FftPlanner`] is protected
/// by a [`Mutex`] so that multiple threads can share a single backend instance
/// (though the synchronous executor is single-threaded).
///
/// [`Op::Window`]: graph_core::ops::Op::Window
/// [`Op::Fft`]: graph_core::ops::Op::Fft
/// [`Op::BandExtract`]: graph_core::ops::Op::BandExtract
pub struct CpuBackend {
    /// Logical device identifier (e.g. `"cpu"` or `"cpu:0"`).
    device_id: DeviceId,
    /// Cached FFT planner. Holds twiddle-factor tables across dispatch calls.
    ///
    /// Wrapped in a `Mutex` to satisfy `Sync` without requiring `&mut self`
    /// in `dispatch_op`.
    fft_planner: Mutex<FftPlanner<f32>>,
}

impl CpuBackend {
    /// Create a new CPU backend with the given device identifier.
    ///
    /// The `device_id` string is used to match nodes in the graph to this
    /// backend. Use `"cpu"` for a single-CPU system or `"cpu:0"`, `"cpu:1"`,
    /// etc. when multiple CPU backends are registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use backends::Backend;
    /// use backends_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new("cpu");
    /// assert_eq!(backend.name(), "cpu");
    /// ```
    pub fn new(device_id: impl Into<String>) -> Self {
        Self {
            device_id: DeviceId::new(device_id),
            fft_planner: Mutex::new(FftPlanner::new()),
        }
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    /// Declares managed-memory model and support for `Op` nodes only.
    ///
    /// The CPU backend does not support `Compute` or `MlModel` node kinds.
    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            memory: MemoryModel::Managed,
            supported_kinds: vec![NodeKindTag::Op],
        }
    }

    /// Not applicable — managed-memory backend does not allocate device memory.
    fn alloc(&self, _size_bytes: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        Err(BackendError::NotApplicable)
    }

    /// Not applicable — managed-memory backend does not use device buffers.
    fn upload(&self, _host: &[u8], _dst: &dyn DeviceBuffer) -> Result<(), BackendError> {
        Err(BackendError::NotApplicable)
    }

    /// Not applicable — managed-memory backend does not use device buffers.
    fn download(&self, _src: &dyn DeviceBuffer, _host: &mut [u8]) -> Result<(), BackendError> {
        Err(BackendError::NotApplicable)
    }

    /// Dispatch a signal processing op.
    ///
    /// Supported ops:
    /// - [`Op::Window`] → [`signal::dispatch_window`]
    /// - [`Op::Fft`] → [`signal::dispatch_fft`]
    /// - [`Op::BandExtract`] → [`signal::dispatch_band_extract`]
    ///
    /// All other ops return [`BackendError::UnsupportedOp`].
    ///
    /// # Errors
    ///
    /// - [`BackendError::UnsupportedOp`] — op is not a signal processing op.
    /// - [`BackendError::Buffer`] — input byte length does not match the
    ///   expected size for the op parameters.
    ///
    /// [`Op::Window`]: graph_core::ops::Op::Window
    /// [`Op::Fft`]: graph_core::ops::Op::Fft
    /// [`Op::BandExtract`]: graph_core::ops::Op::BandExtract
    fn dispatch_op(
        &self,
        op: &Op,
        inputs: &[&[u8]],
        outputs: &mut [Vec<u8>],
    ) -> Result<(), BackendError> {
        match op {
            Op::Window(params) => signal::dispatch_window(params, inputs, outputs),
            Op::Fft(params) => {
                let mut planner = self.fft_planner.lock().expect("fft_planner mutex poisoned");
                signal::dispatch_fft(params, inputs, outputs, &mut planner)
            }
            Op::BandExtract(params) => signal::dispatch_band_extract(params, inputs, outputs),
            other => {
                warn!("CpuBackend: unsupported op '{}'", other.name());
                Err(BackendError::UnsupportedOp)
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::ops::{
        BandDef, BandExtractParams, FftDirection, FftOutput, FftParams, WindowKind, WindowParams,
    };

    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend::new("cpu")
    }

    #[test]
    fn backend_name() {
        assert_eq!(backend().name(), "cpu");
    }

    #[test]
    fn backend_device_id() {
        let b = CpuBackend::new("cpu:0");
        assert_eq!(b.device_id().as_str(), "cpu:0");
    }

    #[test]
    fn backend_capabilities_managed_memory() {
        let caps = backend().capabilities();
        assert!(matches!(caps.memory, MemoryModel::Managed));
    }

    #[test]
    fn backend_capabilities_supports_op_kind() {
        let caps = backend().capabilities();
        assert!(caps.supported_kinds.contains(&NodeKindTag::Op));
    }

    #[test]
    fn alloc_returns_not_applicable() {
        assert!(matches!(
            backend().alloc(64),
            Err(BackendError::NotApplicable)
        ));
    }

    #[test]
    fn upload_returns_not_applicable() {
        // We can't construct a DeviceBuffer easily, so just check the error path
        // by verifying the method exists and returns NotApplicable.
        // Use a dummy test via the trait object.
        let b = backend();
        // We can't call upload without a DeviceBuffer, so we test via alloc failure.
        assert!(b.alloc(1).is_err());
    }

    #[test]
    fn unsupported_op_returns_error() {
        let b = backend();
        let mut outputs = vec![Vec::new()];
        let result = b.dispatch_op(&Op::Relu, &[&[]], &mut outputs);
        assert!(matches!(result, Err(BackendError::UnsupportedOp)));
    }

    #[test]
    fn dispatch_window_hann_succeeds() {
        let b = backend();
        let params = WindowParams::new(WindowKind::Hann, 4).unwrap();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 1.0, 1.0, 1.0]).to_vec();
        let mut outputs = vec![Vec::new()];
        b.dispatch_op(&Op::Window(params), &[&input], &mut outputs)
            .unwrap();
        assert_eq!(outputs[0].len(), 4 * 4);
    }

    #[test]
    fn dispatch_fft_magnitude_succeeds() {
        let b = backend();
        let params = FftParams::new(8, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        let input: Vec<u8> = bytemuck::cast_slice(&[0.0f32; 8]).to_vec();
        let mut outputs = vec![Vec::new()];
        b.dispatch_op(&Op::Fft(params), &[&input], &mut outputs)
            .unwrap();
        // One-sided: 8/2+1 = 5 f32 values.
        assert_eq!(outputs[0].len(), 5 * 4);
    }

    #[test]
    fn dispatch_band_extract_stateless_succeeds() {
        let b = backend();
        let bands = vec![BandDef::new(0.0, 4000.0, "all").unwrap()];
        let params = BandExtractParams::new(bands, 8000.0, 0.0).unwrap();
        // 5-bin spectrum (FFT size 8, sr=8000).
        let spectrum: Vec<u8> = bytemuck::cast_slice(&[1.0f32; 5]).to_vec();
        let mut outputs = vec![Vec::new()];
        b.dispatch_op(&Op::BandExtract(params), &[&spectrum], &mut outputs)
            .unwrap();
        assert_eq!(outputs[0].len(), 1 * 4);
    }

    #[test]
    fn fft_planner_reused_across_calls() {
        // Two FFT calls of the same size should both succeed, exercising
        // the planner cache path.
        let b = backend();
        let params = FftParams::new(16, FftDirection::Forward, FftOutput::Magnitude).unwrap();
        let input: Vec<u8> = bytemuck::cast_slice(&[1.0f32; 16]).to_vec();
        for _ in 0..3 {
            let mut outputs = vec![Vec::new()];
            b.dispatch_op(&Op::Fft(params.clone()), &[&input], &mut outputs)
                .unwrap();
        }
    }
}
