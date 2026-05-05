//! Signal processing dispatch routing.
//!
//! This module routes incoming [`Op`] calls from [`CpuBackend::dispatch_op`]
//! to the appropriate sub-module:
//!
//! ```text
//! dispatch_op(op, inputs, outputs)
//!     │
//!     ▼  match op
//! ┌───────────┬──────────────┬──────────────┐
//! │  Window   │     Fft      │  BandExtract │
//! └───────────┴──────────────┴──────────────┘
//!  window.rs     fft.rs          band.rs
//! ```
//!
//! Each sub-module is responsible for one op family and is independently
//! testable.
//!
//! [`Op`]: graph_core::ops::Op
//! [`CpuBackend::dispatch_op`]: crate::CpuBackend::dispatch_op

mod band;
mod fft;
mod window;

pub(crate) use band::dispatch_band_extract;
pub(crate) use fft::dispatch_fft;
pub(crate) use window::dispatch_window;
