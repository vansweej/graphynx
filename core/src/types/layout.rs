use std::fmt;

// ── Layout ────────────────────────────────────────────────────────────────────

/// Standard memory layout for a contiguous multi-dimensional tensor.
///
/// The variant identifies both the traversal order and — for image layouts —
/// the expected rank. [`Layout::Any`] acts as a wildcard in compatibility
/// checks, meaning "this side imposes no layout constraint".
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Layout {
    /// Row-major / C-contiguous. Last dimension varies fastest.
    ///
    /// Default for most linear algebra operations.
    RowMajor,

    /// Column-major / Fortran-contiguous. First dimension varies fastest.
    ColMajor,

    /// Batch × Channels × Height × Width. Requires rank 4.
    ///
    /// PyTorch's default image layout.
    NCHW,

    /// Batch × Height × Width × Channels. Requires rank 4.
    ///
    /// TensorFlow's default image layout.
    NHWC,

    /// No layout constraint. Compatible with every other layout.
    ///
    /// Used for scalars and for tensors where the layout is unknown or
    /// irrelevant (e.g. the output of a model inference backend).
    Any,
}

impl Layout {
    // ── Compatibility ────────────────────────────────────────────────────

    /// Returns `true` if `self` and `other` are compatible for a graph edge.
    ///
    /// [`Layout::Any`] matches every layout. All other pairs require equality.
    pub fn is_compatible_with(&self, other: &Layout) -> bool {
        matches!(self, Layout::Any) || matches!(other, Layout::Any) || self == other
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Returns `true` for image-channel layouts ([`NCHW`][Layout::NCHW] and
    /// [`NHWC`][Layout::NHWC]).
    pub fn is_image_layout(&self) -> bool {
        matches!(self, Layout::NCHW | Layout::NHWC)
    }

    /// The rank a tensor must have to use this layout, if any.
    ///
    /// Returns `Some(4)` for [`NCHW`][Layout::NCHW] and
    /// [`NHWC`][Layout::NHWC]. Returns `None` for all other layouts, meaning
    /// any rank is acceptable.
    pub fn expected_rank(&self) -> Option<usize> {
        match self {
            Layout::NCHW | Layout::NHWC => Some(4),
            _ => None,
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Layout::RowMajor => "RowMajor",
            Layout::ColMajor => "ColMajor",
            Layout::NCHW => "NCHW",
            Layout::NHWC => "NHWC",
            Layout::Any => "Any",
        };
        f.write_str(s)
    }
}
