use bytemuck::Pod;

use graph_core::types::TensorType;

use super::error::ExecutorError;

// ── InputHandle ───────────────────────────────────────────────────────────────

/// A typed write-only handle for feeding data into a graph source port.
///
/// Obtain an `InputHandle` via [`Executor::input`]. Write data with
/// [`write`](InputHandle::write) or [`write_bytes`](InputHandle::write_bytes),
/// then call [`Executor::run`] to execute the graph.
///
/// [`Executor::input`]: super::Executor::input
/// [`Executor::run`]: super::Executor::run
#[derive(Debug)]
pub struct InputHandle {
    data: Option<Vec<u8>>,
    tensor_type: TensorType,
    expected_bytes: usize,
}

impl InputHandle {
    pub(crate) fn new(tensor_type: TensorType, expected_bytes: usize) -> Self {
        Self {
            data: None,
            tensor_type,
            expected_bytes,
        }
    }

    /// Write typed data into this input handle.
    ///
    /// The data is reinterpreted as raw bytes via [`bytemuck::cast_slice`].
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::InputSizeMismatch`] if the byte length of
    /// `data` does not match the tensor type's expected size.
    pub fn write<T: Pod>(&mut self, source_name: &str, data: &[T]) -> Result<(), ExecutorError> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.write_bytes(source_name, bytes)
    }

    /// Write raw bytes into this input handle.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::InputSizeMismatch`] if `bytes.len()` does not
    /// match the tensor type's expected size.
    pub fn write_bytes(&mut self, source_name: &str, bytes: &[u8]) -> Result<(), ExecutorError> {
        if bytes.len() != self.expected_bytes {
            return Err(ExecutorError::InputSizeMismatch {
                name: source_name.to_string(),
                expected: self.expected_bytes,
                got: bytes.len(),
            });
        }
        self.data = Some(bytes.to_vec());
        Ok(())
    }

    /// The declared tensor type for this source port.
    pub fn tensor_type(&self) -> &TensorType {
        &self.tensor_type
    }

    /// The expected number of bytes for one write.
    pub fn expected_bytes(&self) -> usize {
        self.expected_bytes
    }

    /// Returns `true` if data has been written since the last `run()`.
    pub fn is_written(&self) -> bool {
        self.data.is_some()
    }

    /// Take the written bytes, leaving the handle empty.
    ///
    /// Called by the executor at the start of each `run()`.
    pub(crate) fn take(&mut self) -> Option<Vec<u8>> {
        self.data.take()
    }
}

// ── OutputHandle ──────────────────────────────────────────────────────────────

/// A typed read-only handle for reading data from a graph sink port.
///
/// Obtain an `OutputHandle` via [`Executor::output`] after calling
/// [`Executor::run`].
///
/// [`Executor::output`]: super::Executor::output
/// [`Executor::run`]: super::Executor::run
#[derive(Debug)]
pub struct OutputHandle {
    data: Option<Vec<u8>>,
    tensor_type: TensorType,
}

impl OutputHandle {
    pub(crate) fn new(tensor_type: TensorType) -> Self {
        Self {
            data: None,
            tensor_type,
        }
    }

    /// Read the output as a typed slice.
    ///
    /// Returns `None` if no data has been produced yet (i.e. `run()` has not
    /// been called, or the last `run()` failed before this sink was reached).
    pub fn read<T: Pod>(&self) -> Option<&[T]> {
        let bytes = self.data.as_deref()?;
        Some(bytemuck::cast_slice(bytes))
    }

    /// Read the output as raw bytes.
    ///
    /// Returns `None` if no data has been produced yet.
    pub fn read_bytes(&self) -> Option<&[u8]> {
        self.data.as_deref()
    }

    /// The declared tensor type for this sink port.
    pub fn tensor_type(&self) -> &TensorType {
        &self.tensor_type
    }

    /// Store output bytes. Called by the executor after each `run()`.
    pub(crate) fn set(&mut self, bytes: Vec<u8>) {
        self.data = Some(bytes);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::types::{dim::Dim, DType, Layout, TensorType};

    use super::*;

    fn f32_type(n: usize) -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(n)], Layout::RowMajor).unwrap()
    }

    // ── InputHandle ───────────────────────────────────────────────────────

    #[test]
    fn input_handle_write_typed() {
        let mut h = InputHandle::new(f32_type(4), 16);
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        h.write("in", &data).unwrap();
        assert!(h.is_written());
        let bytes = h.take().unwrap();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn input_handle_write_bytes() {
        let mut h = InputHandle::new(f32_type(2), 8);
        h.write_bytes("in", &[0u8; 8]).unwrap();
        assert!(h.is_written());
    }

    #[test]
    fn input_handle_wrong_size_returns_error() {
        let mut h = InputHandle::new(f32_type(4), 16);
        let result = h.write_bytes("in", &[0u8; 8]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutorError::InputSizeMismatch { .. }
        ));
    }

    #[test]
    fn input_handle_take_clears_written() {
        let mut h = InputHandle::new(f32_type(1), 4);
        h.write_bytes("in", &[0u8; 4]).unwrap();
        assert!(h.is_written());
        let _ = h.take();
        assert!(!h.is_written());
    }

    #[test]
    fn input_handle_take_empty_returns_none() {
        let mut h = InputHandle::new(f32_type(1), 4);
        assert!(h.take().is_none());
    }

    #[test]
    fn input_handle_accessors() {
        let t = f32_type(8);
        let h = InputHandle::new(t.clone(), 32);
        assert_eq!(h.expected_bytes(), 32);
        assert_eq!(h.tensor_type(), &t);
        assert!(!h.is_written());
    }

    // ── OutputHandle ──────────────────────────────────────────────────────

    #[test]
    fn output_handle_read_before_set_returns_none() {
        let h = OutputHandle::new(f32_type(4));
        assert!(h.read::<f32>().is_none());
        assert!(h.read_bytes().is_none());
    }

    #[test]
    fn output_handle_set_and_read_typed() {
        let mut h = OutputHandle::new(f32_type(2));
        let data: [f32; 2] = [3.14, 2.72];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        h.set(bytes.to_vec());
        let out: &[f32] = h.read().unwrap();
        assert!((out[0] - 3.14_f32).abs() < 1e-6);
        assert!((out[1] - 2.72_f32).abs() < 1e-6);
    }

    #[test]
    fn output_handle_read_bytes() {
        let mut h = OutputHandle::new(f32_type(1));
        h.set(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(h.read_bytes().unwrap(), &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn output_handle_tensor_type() {
        let t = f32_type(3);
        let h = OutputHandle::new(t.clone());
        assert_eq!(h.tensor_type(), &t);
    }
}
