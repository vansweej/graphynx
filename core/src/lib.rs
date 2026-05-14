//! Core — backend-agnostic types and operation catalog for graphynx.

pub mod graph;
pub mod ops;
pub mod types;

#[cfg(feature = "serde")]
pub use graph::persist;
