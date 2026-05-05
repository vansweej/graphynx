use backends::BackendError;
use thiserror::Error;

// ── ExecutorError ─────────────────────────────────────────────────────────────

/// Errors that the executor can produce during construction or execution.
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// No backend was registered for the device ID required by a node.
    #[error("No backend registered for device '{0}'")]
    NoBackend(String),

    /// A source name passed to [`Executor::input`] does not exist in the graph.
    #[error("Input '{name}' not found in graph sources")]
    UnknownInput {
        /// The name that was looked up.
        name: String,
    },

    /// A sink name passed to [`Executor::output`] does not exist in the graph.
    #[error("Output '{name}' not found in graph sinks")]
    UnknownOutput {
        /// The name that was looked up.
        name: String,
    },

    /// The byte slice written to an input handle has the wrong length.
    #[error("Input '{name}' has wrong byte length: expected {expected}, got {got}")]
    InputSizeMismatch {
        /// Name of the source.
        name: String,
        /// How many bytes the tensor type requires.
        expected: usize,
        /// How many bytes were actually written.
        got: usize,
    },

    /// A node's output port has a dynamic or symbolic dimension, so its buffer
    /// size cannot be determined at construction time.
    ///
    /// Phase 2 only supports graphs where every output port has fully fixed
    /// dimensions (i.e. `TensorType::size_bytes()` returns `Some`).
    #[error(
        "Dynamic dimensions not supported: node '{node}' output port {port} has no fixed size"
    )]
    DynamicSize {
        /// Name of the node.
        node: String,
        /// Zero-based output port index.
        port: usize,
    },

    /// `Executor::run()` was called but an input handle was not written.
    #[error("Input '{name}' was not written before run()")]
    InputNotWritten {
        /// Name of the source.
        name: String,
    },

    /// A backend returned an error while dispatching a node.
    #[error("Backend error on node '{node}': {source}")]
    Backend {
        /// Name of the node that failed.
        node: String,
        /// The underlying backend error.
        source: BackendError,
    },
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use backends::BackendError;

    use super::*;

    #[test]
    fn no_backend_display() {
        let e = ExecutorError::NoBackend("cuda:0".into());
        assert!(e.to_string().contains("cuda:0"));
    }

    #[test]
    fn unknown_input_display() {
        let e = ExecutorError::UnknownInput {
            name: "audio".into(),
        };
        assert!(e.to_string().contains("audio"));
    }

    #[test]
    fn unknown_output_display() {
        let e = ExecutorError::UnknownOutput { name: "out".into() };
        assert!(e.to_string().contains("out"));
    }

    #[test]
    fn input_size_mismatch_display() {
        let e = ExecutorError::InputSizeMismatch {
            name: "x".into(),
            expected: 16,
            got: 8,
        };
        let s = e.to_string();
        assert!(s.contains("16"));
        assert!(s.contains('8'.to_string().as_str()));
    }

    #[test]
    fn dynamic_size_display() {
        let e = ExecutorError::DynamicSize {
            node: "fft".into(),
            port: 0,
        };
        let s = e.to_string();
        assert!(s.contains("fft"));
        assert!(s.contains('0'.to_string().as_str()));
    }

    #[test]
    fn input_not_written_display() {
        let e = ExecutorError::InputNotWritten { name: "mic".into() };
        assert!(e.to_string().contains("mic"));
    }

    #[test]
    fn backend_error_display() {
        let e = ExecutorError::Backend {
            node: "relu".into(),
            source: BackendError::UnsupportedOp,
        };
        let s = e.to_string();
        assert!(s.contains("relu"));
        assert!(s.contains("Unsupported op"));
    }

    #[test]
    fn executor_error_implements_std_error() {
        let e = ExecutorError::NoBackend("x".into());
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn backend_error_source_chain() {
        let e = ExecutorError::Backend {
            node: "n".into(),
            source: BackendError::Device("gpu fault".into()),
        };
        // thiserror wires #[source] so std::error::Error::source() returns Some
        use std::error::Error;
        assert!(e.source().is_some());
    }
}
