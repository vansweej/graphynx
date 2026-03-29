use std::fmt;

use thiserror::Error;

use crate::backends::DeviceId;
use crate::core::types::dim::Dim;
use crate::core::types::dtype::DType;
use crate::core::types::layout::Layout;
use crate::core::types::shape::{Shape, ShapeError};

// ── TensorTypeError ───────────────────────────────────────────────────────────

/// Errors produced when constructing or transforming a [`TensorType`].
///
/// Every variant carries enough context for the caller to understand exactly
/// what invariant was violated and how to fix it.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum TensorTypeError {
    /// A `Fixed` dimension value of zero was supplied. All fixed dimensions
    /// must be ≥ 1.
    #[error("Dimension must be > 0, got 0")]
    ZeroDimension,

    /// A `Symbolic` dimension was given an empty name. Symbolic names must be
    /// non-empty so they can be matched across graph edges.
    #[error("Symbolic dimension name must not be empty")]
    EmptySymbol,

    /// A scalar tensor (rank 0) cannot carry a concrete layout. Scalars must
    /// use [`Layout::Any`].
    #[error("Scalar tensor (rank 0) cannot use layout {0}; use Layout::Any")]
    ScalarWithLayout(Layout),

    /// The chosen layout requires a specific rank that the shape does not
    /// satisfy.
    #[error("Layout {layout} requires rank {expected}, got {actual}")]
    LayoutRankMismatch {
        /// The layout that has a rank constraint.
        layout: Layout,
        /// The rank the layout requires.
        expected: usize,
        /// The rank of the supplied shape.
        actual: usize,
    },

    /// The number of dim-name strings does not equal the number of dimensions
    /// in the shape.
    #[error("dim_names length ({names}) does not match shape length ({shape})")]
    DimNamesMismatch {
        /// Number of name strings supplied.
        names: usize,
        /// Number of dimensions in the shape.
        shape: usize,
    },
}

// ── TensorType ────────────────────────────────────────────────────────────────

/// Complete description of a tensor flowing along a graph edge.
///
/// Combines a scalar element type ([`DType`]), a [`Shape`], a memory
/// [`Layout`], optional human-readable dimension names, and an optional
/// [`DeviceId`] indicating where the tensor lives.
///
/// # Invariants (always upheld)
///
/// - No [`Dim::Fixed(0)`] in `shape`.
/// - No [`Dim::Symbolic("")`] in `shape`.
/// - If `dim_names` is `Some`, its length equals `shape.rank()`.
/// - [`Layout::NCHW`] and [`Layout::NHWC`] require rank 4.
/// - Rank-0 (scalar) tensors must use [`Layout::Any`].
///
/// All fields are private. Use [`TensorType::new`], [`TensorType::scalar`],
/// [`TensorType::vector`], [`TensorType::matrix`], or the
/// [`TensorTypeBuilder`] to construct instances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    // All fields are private to uphold the invariants above.
    dtype: DType,
    shape: Shape,
    layout: Layout,
    dim_names: Option<Vec<String>>,
    device: Option<DeviceId>,
}

impl TensorType {
    // ── Validated constructors ────────────────────────────────────────────

    /// Fully-specified constructor.
    ///
    /// Validates all invariants before returning. `dim_names` and `device`
    /// default to `None`; use [`TensorType::builder`] if you need them.
    ///
    /// # Errors
    ///
    /// Returns a [`TensorTypeError`] if any invariant is violated.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let t = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(224), Dim::Fixed(224)],
    ///     Layout::NCHW,
    /// ).unwrap();
    /// assert_eq!(t.rank(), 4);
    /// ```
    pub fn new(dtype: DType, shape: Vec<Dim>, layout: Layout) -> Result<Self, TensorTypeError> {
        Self::validate_shape_and_layout(&shape, layout)?;
        // Shape::new validates dims (no Fixed(0), no Symbolic("")).
        // We already validated in validate_shape_and_layout, so construct directly.
        let shape = Shape::new(shape).map_err(|e| match e {
            ShapeError::ZeroDimension => TensorTypeError::ZeroDimension,
            ShapeError::EmptySymbol => TensorTypeError::EmptySymbol,
            _ => unreachable!("Shape::new only returns ZeroDimension or EmptySymbol"),
        })?;
        Ok(TensorType {
            dtype,
            shape,
            layout,
            dim_names: None,
            device: None,
        })
    }

    /// Rank-0 scalar tensor.
    ///
    /// Layout is always [`Layout::Any`]. No dim names. This constructor is
    /// infallible because a scalar has no dims to validate.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let s = TensorType::scalar(DType::F32);
    /// assert!(s.is_scalar());
    /// assert_eq!(s.layout(), Layout::Any);
    /// ```
    pub fn scalar(dtype: DType) -> Self {
        TensorType {
            dtype,
            shape: Shape::scalar(),
            layout: Layout::Any,
            dim_names: None,
            device: None,
        }
    }

    /// Rank-1 tensor with one [`Dim::Fixed`] dimension.
    ///
    /// Layout defaults to [`Layout::RowMajor`].
    ///
    /// # Errors
    ///
    /// Returns [`TensorTypeError::ZeroDimension`] if `len == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let v = TensorType::vector(DType::I32, 1024).unwrap();
    /// assert_eq!(v.rank(), 1);
    /// assert_eq!(v.layout(), Layout::RowMajor);
    /// ```
    pub fn vector(dtype: DType, len: usize) -> Result<Self, TensorTypeError> {
        if len == 0 {
            return Err(TensorTypeError::ZeroDimension);
        }
        Ok(TensorType {
            dtype,
            shape: Shape::vector(len).map_err(|_| TensorTypeError::ZeroDimension)?,
            layout: Layout::RowMajor,
            dim_names: None,
            device: None,
        })
    }

    /// Rank-2 tensor with shape `[rows, cols]`.
    ///
    /// Layout defaults to [`Layout::RowMajor`].
    ///
    /// # Errors
    ///
    /// Returns [`TensorTypeError::ZeroDimension`] if `rows == 0` or
    /// `cols == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let m = TensorType::matrix(DType::F64, 3, 4).unwrap();
    /// assert_eq!(m.rank(), 2);
    /// assert_eq!(m.num_elements(), Some(12));
    /// ```
    pub fn matrix(dtype: DType, rows: usize, cols: usize) -> Result<Self, TensorTypeError> {
        if rows == 0 || cols == 0 {
            return Err(TensorTypeError::ZeroDimension);
        }
        Ok(TensorType {
            dtype,
            shape: Shape::matrix(rows, cols).map_err(|_| TensorTypeError::ZeroDimension)?,
            layout: Layout::RowMajor,
            dim_names: None,
            device: None,
        })
    }

    /// Start a fluent builder for complex tensor type specifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::backend::DeviceId;
    /// use graphynx::dtype::DType;
    ///
    /// let t = TensorType::builder(DType::F32)
    ///     .shape(vec![
    ///         Dim::Symbolic("batch".into()),
    ///         Dim::Fixed(3),
    ///         Dim::Fixed(224),
    ///         Dim::Fixed(224),
    ///     ])
    ///     .layout(Layout::NCHW)
    ///     .dim_names(vec![
    ///         "batch".into(), "channels".into(), "height".into(), "width".into(),
    ///     ])
    ///     .device(DeviceId::new("cuda:0"))
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(t.rank(), 4);
    /// assert_eq!(t.device().unwrap().to_string(), "cuda:0");
    /// ```
    pub fn builder(dtype: DType) -> TensorTypeBuilder {
        TensorTypeBuilder::new(dtype)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// The scalar element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The memory layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Optional per-dimension human-readable names.
    ///
    /// If present, the slice has the same length as [`shape().rank()`][Shape::rank].
    pub fn dim_names(&self) -> Option<&[String]> {
        self.dim_names.as_deref()
    }

    /// The device this tensor is placed on, or `None` if unplaced.
    pub fn device(&self) -> Option<&DeviceId> {
        self.device.as_ref()
    }

    // ── Derived properties ────────────────────────────────────────────────

    /// Number of dimensions (0 for scalars).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns `true` for rank-0 tensors.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Total number of elements, if every dimension is [`Dim::Fixed`].
    ///
    /// Returns `None` if any dimension is [`Dim::Dynamic`] or
    /// [`Dim::Symbolic`], or if the tensor is a scalar (returns `Some(1)` for
    /// scalars — one element with no dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let m = TensorType::matrix(DType::F32, 4, 8).unwrap();
    /// assert_eq!(m.num_elements(), Some(32));
    ///
    /// let t = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Dynamic, Dim::Fixed(256)],
    ///     Layout::RowMajor,
    /// ).unwrap();
    /// assert_eq!(t.num_elements(), None);
    /// ```
    pub fn num_elements(&self) -> Option<usize> {
        self.shape.num_elements()
    }

    /// Total size in bytes, if every dimension is fixed and `dtype` has a
    /// known element size.
    ///
    /// Returns `None` if [`num_elements`][Self::num_elements] is `None` or if
    /// the dtype is [`DType::Custom`].
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::TensorType;
    /// use graphynx::dtype::DType;
    ///
    /// let v = TensorType::vector(DType::F32, 1024).unwrap();
    /// assert_eq!(v.size_bytes(), Some(4096));
    /// ```
    pub fn size_bytes(&self) -> Option<usize> {
        let n = self.num_elements()?;
        let elem = self.dtype.size_bytes()?;
        Some(n * elem)
    }

    // ── Transforms ────────────────────────────────────────────────────────
    //
    // Every transform consumes `self` and returns a new, validated TensorType.
    // There are no `&mut self` setters — partial mutation can leave an object
    // in an invalid state, which these consuming transforms prevent.

    /// Return a new `TensorType` with the layout replaced.
    ///
    /// Re-validates layout vs. rank so the invariant is preserved.
    ///
    /// # Errors
    ///
    /// Returns [`TensorTypeError`] if the new layout is incompatible with the
    /// current rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let t = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(224), Dim::Fixed(224)],
    ///     Layout::RowMajor,
    /// ).unwrap();
    /// let t_nchw = t.with_layout(Layout::NCHW).unwrap();
    /// assert_eq!(t_nchw.layout(), Layout::NCHW);
    /// ```
    pub fn with_layout(mut self, layout: Layout) -> Result<Self, TensorTypeError> {
        Self::validate_shape_and_layout(self.shape.dims(), layout)?;
        self.layout = layout;
        Ok(self)
    }

    /// Return a new `TensorType` placed on the given device.
    ///
    /// Infallible — device placement does not affect tensor validity.
    pub fn with_device(mut self, device: DeviceId) -> Self {
        self.device = Some(device);
        self
    }

    /// Return a new `TensorType` with the device placement cleared.
    ///
    /// Infallible — removing device placement does not affect tensor validity.
    pub fn unplaced(mut self) -> Self {
        self.device = None;
        self
    }

    /// Return a new `TensorType` with dimension names added.
    ///
    /// # Errors
    ///
    /// Returns [`TensorTypeError::DimNamesMismatch`] if `names.len() !=
    /// self.rank()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let t = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Fixed(1), Dim::Fixed(3), Dim::Fixed(224), Dim::Fixed(224)],
    ///     Layout::NCHW,
    /// ).unwrap();
    /// let named = t.with_dim_names(vec![
    ///     "batch".into(), "channels".into(), "height".into(), "width".into(),
    /// ]).unwrap();
    /// assert_eq!(named.dim_names().unwrap()[0], "batch");
    /// ```
    pub fn with_dim_names(mut self, names: Vec<String>) -> Result<Self, TensorTypeError> {
        if names.len() != self.shape.rank() {
            return Err(TensorTypeError::DimNamesMismatch {
                names: names.len(),
                shape: self.shape.rank(),
            });
        }
        self.dim_names = Some(names);
        Ok(self)
    }

    // ── Compatibility ─────────────────────────────────────────────────────

    /// Returns `true` if `self` and `other` could be connected by a graph edge.
    ///
    /// # Rules
    ///
    /// - `dtype` must be equal.
    /// - `rank` must be equal.
    /// - Each dimension pair must satisfy [`Dim::is_compatible_with`].
    /// - Layouts must satisfy [`Layout::is_compatible_with`].
    /// - `device` is **not** checked; device placement is the scheduler's
    ///   responsibility.
    /// - `dim_names` are **not** checked; names are advisory only.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::tensor_type::{Dim, Layout, TensorType};
    /// use graphynx::dtype::DType;
    ///
    /// let producer = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Fixed(3), Dim::Fixed(256)],
    ///     Layout::RowMajor,
    /// ).unwrap();
    /// let consumer = TensorType::new(
    ///     DType::F32,
    ///     vec![Dim::Dynamic, Dim::Fixed(256)],
    ///     Layout::Any,
    /// ).unwrap();
    /// assert!(producer.is_compatible_with(&consumer));
    /// ```
    pub fn is_compatible_with(&self, other: &TensorType) -> bool {
        // dtype must match.
        if self.dtype != other.dtype {
            return false;
        }
        // Shape compatibility (rank + per-dim checks).
        if !self.shape.is_compatible_with(&other.shape) {
            return false;
        }
        // Layouts must be compatible.
        self.layout.is_compatible_with(&other.layout)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    /// Validate shape and layout together, enforcing:
    /// - No Fixed(0) dims.
    /// - No Symbolic("") dims.
    /// - Rank-0 must use Layout::Any.
    /// - Image layouts require rank 4.
    fn validate_shape_and_layout(shape: &[Dim], layout: Layout) -> Result<(), TensorTypeError> {
        // --- Per-dim validation ---
        for dim in shape {
            match dim {
                Dim::Fixed(0) => return Err(TensorTypeError::ZeroDimension),
                Dim::Symbolic(s) if s.is_empty() => return Err(TensorTypeError::EmptySymbol),
                _ => {}
            }
        }

        let rank = shape.len();

        // --- Scalar constraint ---
        if rank == 0 && layout != Layout::Any {
            return Err(TensorTypeError::ScalarWithLayout(layout));
        }

        // --- Layout rank constraint ---
        if let Some(expected) = layout.expected_rank() {
            if rank != expected {
                return Err(TensorTypeError::LayoutRankMismatch {
                    layout,
                    expected,
                    actual: rank,
                });
            }
        }

        Ok(())
    }
}

impl fmt::Display for TensorType {
    /// Formats the tensor type in a concise human-readable form.
    ///
    /// Format: `dtype[dim, dim, ...] Layout` and optionally `@ device`.
    ///
    /// # Examples
    ///
    /// ```text
    /// f32[] Any
    /// i32[1024] RowMajor
    /// f32[batch, 3, 224, 224] NCHW @ cuda:0
    /// f64[?, 256] RowMajor
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // dtype
        write!(f, "{}", self.dtype)?;

        // shape (Shape's Display produces [dim, dim, ...])
        write!(f, "{}", self.shape)?;

        // layout
        write!(f, " {}", self.layout)?;

        // optional device
        if let Some(dev) = &self.device {
            write!(f, " @ {dev}")?;
        }

        Ok(())
    }
}

// ── TensorTypeBuilder ─────────────────────────────────────────────────────────

/// Fluent builder for [`TensorType`].
///
/// Construct via [`TensorType::builder`]. Call methods to set optional fields,
/// then call [`build`][TensorTypeBuilder::build] to validate and produce the
/// final `TensorType`.
///
/// Defaults:
/// - `shape`: `[]` (scalar)
/// - `layout`: [`Layout::Any`]
/// - `dim_names`: `None`
/// - `device`: `None`
pub struct TensorTypeBuilder {
    dtype: DType,
    shape: Vec<Dim>,
    layout: Layout,
    dim_names: Option<Vec<String>>,
    device: Option<DeviceId>,
}

impl TensorTypeBuilder {
    /// Create a new builder with the given dtype and all other fields at their
    /// defaults.
    pub fn new(dtype: DType) -> Self {
        TensorTypeBuilder {
            dtype,
            shape: vec![],
            layout: Layout::Any,
            dim_names: None,
            device: None,
        }
    }

    /// Set the shape.
    pub fn shape(mut self, shape: Vec<Dim>) -> Self {
        self.shape = shape;
        self
    }

    /// Set the memory layout.
    pub fn layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Set per-dimension names. Length must match `shape` at `build()` time.
    pub fn dim_names(mut self, names: Vec<String>) -> Self {
        self.dim_names = Some(names);
        self
    }

    /// Set the device placement.
    pub fn device(mut self, device: DeviceId) -> Self {
        self.device = Some(device);
        self
    }

    /// Validate all fields and produce a [`TensorType`].
    ///
    /// # Errors
    ///
    /// Returns the first [`TensorTypeError`] encountered:
    /// shape/layout validation runs first, then dim_names length check.
    pub fn build(self) -> Result<TensorType, TensorTypeError> {
        // Validate shape + layout invariants.
        TensorType::validate_shape_and_layout(&self.shape, self.layout)?;

        // Build the Shape (validates dims internally).
        let shape = Shape::new(self.shape).map_err(|e| match e {
            ShapeError::ZeroDimension => TensorTypeError::ZeroDimension,
            ShapeError::EmptySymbol => TensorTypeError::EmptySymbol,
            _ => unreachable!("Shape::new only returns ZeroDimension or EmptySymbol"),
        })?;

        // Validate dim_names length.
        if let Some(ref names) = self.dim_names {
            if names.len() != shape.rank() {
                return Err(TensorTypeError::DimNamesMismatch {
                    names: names.len(),
                    shape: shape.rank(),
                });
            }
        }

        Ok(TensorType {
            dtype: self.dtype,
            shape,
            layout: self.layout,
            dim_names: self.dim_names,
            device: self.device,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::backends::DeviceId;
    use crate::core::types::dtype::DType;

    use super::*;

    // ── Dim tests ─────────────────────────────────────────────────────────

    mod dim_tests {
        use super::*;

        // --- constructors ---

        #[test]
        fn fixed_zero_is_error() {
            assert_eq!(
                Dim::fixed(0),
                Err(crate::core::types::dim::DimError::ZeroDimension)
            );
        }

        #[test]
        fn fixed_one_is_ok() {
            assert_eq!(Dim::fixed(1), Ok(Dim::Fixed(1)));
        }

        #[test]
        fn fixed_large_is_ok() {
            assert_eq!(Dim::fixed(usize::MAX), Ok(Dim::Fixed(usize::MAX)));
        }

        #[test]
        fn symbolic_nonempty_is_ok() {
            assert_eq!(
                Dim::symbolic("batch"),
                Ok(Dim::Symbolic("batch".to_string()))
            );
        }

        #[test]
        fn symbolic_empty_is_error() {
            assert_eq!(
                Dim::symbolic(""),
                Err(crate::core::types::dim::DimError::EmptySymbol)
            );
        }

        #[test]
        fn dynamic_construction() {
            // Direct variant construction — no validation needed.
            let d = Dim::Dynamic;
            assert!(d.is_dynamic());
        }

        // --- is_* predicates ---

        #[test]
        fn is_fixed_truth_table() {
            assert!(Dim::Fixed(5).is_fixed());
            assert!(!Dim::Dynamic.is_fixed());
            assert!(!Dim::Symbolic("x".into()).is_fixed());
        }

        #[test]
        fn is_dynamic_truth_table() {
            assert!(!Dim::Fixed(5).is_dynamic());
            assert!(Dim::Dynamic.is_dynamic());
            assert!(Dim::Symbolic("x".into()).is_dynamic());
        }

        #[test]
        fn is_symbolic_truth_table() {
            assert!(!Dim::Fixed(5).is_symbolic());
            assert!(!Dim::Dynamic.is_symbolic());
            assert!(Dim::Symbolic("x".into()).is_symbolic());
        }

        // --- accessor methods ---

        #[test]
        fn fixed_value_returns_some_for_fixed() {
            assert_eq!(Dim::Fixed(42).fixed_value(), Some(42));
        }

        #[test]
        fn fixed_value_returns_none_for_dynamic() {
            assert_eq!(Dim::Dynamic.fixed_value(), None);
        }

        #[test]
        fn fixed_value_returns_none_for_symbolic() {
            assert_eq!(Dim::Symbolic("n".into()).fixed_value(), None);
        }

        #[test]
        fn symbol_returns_some_for_symbolic() {
            assert_eq!(Dim::Symbolic("batch".into()).symbol(), Some("batch"));
        }

        #[test]
        fn symbol_returns_none_for_fixed() {
            assert_eq!(Dim::Fixed(3).symbol(), None);
        }

        #[test]
        fn symbol_returns_none_for_dynamic() {
            assert_eq!(Dim::Dynamic.symbol(), None);
        }

        // --- compatibility matrix ---

        #[test]
        fn compat_fixed_same() {
            assert!(Dim::Fixed(3).is_compatible_with(&Dim::Fixed(3)));
        }

        #[test]
        fn compat_fixed_different() {
            assert!(!Dim::Fixed(3).is_compatible_with(&Dim::Fixed(4)));
        }

        #[test]
        fn compat_fixed_vs_dynamic() {
            assert!(Dim::Fixed(3).is_compatible_with(&Dim::Dynamic));
            assert!(Dim::Dynamic.is_compatible_with(&Dim::Fixed(3)));
        }

        #[test]
        fn compat_fixed_vs_symbolic() {
            assert!(Dim::Fixed(3).is_compatible_with(&Dim::Symbolic("n".into())));
            assert!(Dim::Symbolic("n".into()).is_compatible_with(&Dim::Fixed(3)));
        }

        #[test]
        fn compat_dynamic_vs_dynamic() {
            assert!(Dim::Dynamic.is_compatible_with(&Dim::Dynamic));
        }

        #[test]
        fn compat_dynamic_vs_symbolic() {
            assert!(Dim::Dynamic.is_compatible_with(&Dim::Symbolic("batch".into())));
            assert!(Dim::Symbolic("batch".into()).is_compatible_with(&Dim::Dynamic));
        }

        #[test]
        fn compat_symbolic_same_name() {
            assert!(
                Dim::Symbolic("batch".into()).is_compatible_with(&Dim::Symbolic("batch".into()))
            );
        }

        #[test]
        fn compat_symbolic_different_names() {
            assert!(!Dim::Symbolic("batch".into()).is_compatible_with(&Dim::Symbolic("seq".into())));
        }

        // --- derives ---

        #[test]
        fn dim_clone_eq_hash() {
            let d = Dim::Symbolic("x".into());
            let d2 = d.clone();
            assert_eq!(d, d2);
            let mut map = HashMap::new();
            map.insert(d, 1u32);
            assert_eq!(map[&d2], 1);
        }

        #[test]
        fn dim_debug() {
            let s = format!("{:?}", Dim::Fixed(7));
            assert!(s.contains("Fixed"));
        }

        // --- Display ---

        #[test]
        fn display_fixed() {
            assert_eq!(Dim::Fixed(42).to_string(), "42");
        }

        #[test]
        fn display_dynamic() {
            assert_eq!(Dim::Dynamic.to_string(), "?");
        }

        #[test]
        fn display_symbolic() {
            assert_eq!(Dim::Symbolic("batch".into()).to_string(), "batch");
        }
    }

    // ── Layout tests ──────────────────────────────────────────────────────

    mod layout_tests {
        use super::*;

        // --- is_compatible_with ---

        #[test]
        fn any_is_compatible_with_all() {
            for layout in [
                Layout::RowMajor,
                Layout::ColMajor,
                Layout::NCHW,
                Layout::NHWC,
                Layout::Any,
            ] {
                assert!(
                    Layout::Any.is_compatible_with(&layout),
                    "Any should be compatible with {layout}"
                );
                assert!(
                    layout.is_compatible_with(&Layout::Any),
                    "{layout} should be compatible with Any"
                );
            }
        }

        #[test]
        fn same_layout_compatible() {
            assert!(Layout::RowMajor.is_compatible_with(&Layout::RowMajor));
            assert!(Layout::NCHW.is_compatible_with(&Layout::NCHW));
        }

        #[test]
        fn different_concrete_layouts_incompatible() {
            assert!(!Layout::RowMajor.is_compatible_with(&Layout::ColMajor));
            assert!(!Layout::NCHW.is_compatible_with(&Layout::NHWC));
            assert!(!Layout::RowMajor.is_compatible_with(&Layout::NCHW));
        }

        // --- expected_rank ---

        #[test]
        fn nchw_requires_rank_4() {
            assert_eq!(Layout::NCHW.expected_rank(), Some(4));
        }

        #[test]
        fn nhwc_requires_rank_4() {
            assert_eq!(Layout::NHWC.expected_rank(), Some(4));
        }

        #[test]
        fn other_layouts_have_no_rank_constraint() {
            assert_eq!(Layout::RowMajor.expected_rank(), None);
            assert_eq!(Layout::ColMajor.expected_rank(), None);
            assert_eq!(Layout::Any.expected_rank(), None);
        }

        // --- is_image_layout ---

        #[test]
        fn image_layout_truth_table() {
            assert!(Layout::NCHW.is_image_layout());
            assert!(Layout::NHWC.is_image_layout());
            assert!(!Layout::RowMajor.is_image_layout());
            assert!(!Layout::ColMajor.is_image_layout());
            assert!(!Layout::Any.is_image_layout());
        }

        // --- derives ---

        #[test]
        fn layout_copy_clone_eq_hash() {
            let l = Layout::NCHW;
            let l2 = l; // Copy
            assert_eq!(l, l2);
            let mut map = HashMap::new();
            map.insert(l, "nchw");
            assert_eq!(map[&l2], "nchw");
        }

        #[test]
        fn layout_debug() {
            assert!(format!("{:?}", Layout::RowMajor).contains("RowMajor"));
        }

        // --- Display ---

        #[test]
        fn display_all_variants() {
            assert_eq!(Layout::RowMajor.to_string(), "RowMajor");
            assert_eq!(Layout::ColMajor.to_string(), "ColMajor");
            assert_eq!(Layout::NCHW.to_string(), "NCHW");
            assert_eq!(Layout::NHWC.to_string(), "NHWC");
            assert_eq!(Layout::Any.to_string(), "Any");
        }
    }

    // ── TensorTypeError tests ─────────────────────────────────────────────

    mod error_tests {
        use super::*;

        #[test]
        fn zero_dimension_message() {
            let msg = TensorTypeError::ZeroDimension.to_string();
            assert!(msg.contains("0"), "msg was: {msg}");
        }

        #[test]
        fn empty_symbol_message() {
            let msg = TensorTypeError::EmptySymbol.to_string();
            assert!(msg.to_lowercase().contains("empty"), "msg was: {msg}");
        }

        #[test]
        fn scalar_with_layout_message() {
            let msg = TensorTypeError::ScalarWithLayout(Layout::NCHW).to_string();
            assert!(msg.contains("NCHW"), "msg was: {msg}");
        }

        #[test]
        fn layout_rank_mismatch_message() {
            let msg = TensorTypeError::LayoutRankMismatch {
                layout: Layout::NCHW,
                expected: 4,
                actual: 3,
            }
            .to_string();
            assert!(
                msg.contains("NCHW") && msg.contains('4') && msg.contains('3'),
                "msg was: {msg}"
            );
        }

        #[test]
        fn dim_names_mismatch_message() {
            let msg = TensorTypeError::DimNamesMismatch { names: 3, shape: 4 }.to_string();
            assert!(msg.contains('3') && msg.contains('4'), "msg was: {msg}");
        }

        #[test]
        fn errors_clone_eq() {
            let e = TensorTypeError::ZeroDimension;
            assert_eq!(e.clone(), e);
        }
    }

    // ── TensorType construction tests ─────────────────────────────────────

    mod construction_tests {
        use super::*;

        // --- scalar ---

        #[test]
        fn scalar_rank_zero() {
            let s = TensorType::scalar(DType::F32);
            assert_eq!(s.rank(), 0);
        }

        #[test]
        fn scalar_layout_any() {
            assert_eq!(TensorType::scalar(DType::I32).layout(), Layout::Any);
        }

        #[test]
        fn scalar_is_scalar() {
            assert!(TensorType::scalar(DType::F64).is_scalar());
        }

        #[test]
        fn scalar_no_dim_names() {
            assert!(TensorType::scalar(DType::U8).dim_names().is_none());
        }

        #[test]
        fn scalar_no_device() {
            assert!(TensorType::scalar(DType::Bool).device().is_none());
        }

        // --- vector ---

        #[test]
        fn vector_rank_one() {
            assert_eq!(TensorType::vector(DType::F32, 1024).unwrap().rank(), 1);
        }

        #[test]
        fn vector_layout_row_major() {
            assert_eq!(
                TensorType::vector(DType::F32, 512).unwrap().layout(),
                Layout::RowMajor
            );
        }

        #[test]
        fn vector_zero_len_error() {
            assert_eq!(
                TensorType::vector(DType::F32, 0),
                Err(TensorTypeError::ZeroDimension)
            );
        }

        // --- matrix ---

        #[test]
        fn matrix_rank_two() {
            assert_eq!(TensorType::matrix(DType::F64, 3, 4).unwrap().rank(), 2);
        }

        #[test]
        fn matrix_num_elements() {
            assert_eq!(
                TensorType::matrix(DType::F32, 4, 8).unwrap().num_elements(),
                Some(32)
            );
        }

        #[test]
        fn matrix_zero_rows_error() {
            assert_eq!(
                TensorType::matrix(DType::F32, 0, 4),
                Err(TensorTypeError::ZeroDimension)
            );
        }

        #[test]
        fn matrix_zero_cols_error() {
            assert_eq!(
                TensorType::matrix(DType::F32, 4, 0),
                Err(TensorTypeError::ZeroDimension)
            );
        }

        // --- new ---

        #[test]
        fn new_nchw_rank4_ok() {
            let t = TensorType::new(
                DType::F32,
                vec![
                    Dim::Fixed(1),
                    Dim::Fixed(3),
                    Dim::Fixed(224),
                    Dim::Fixed(224),
                ],
                Layout::NCHW,
            );
            assert!(t.is_ok());
        }

        #[test]
        fn new_nchw_rank3_error() {
            let err = TensorType::new(
                DType::F32,
                vec![Dim::Fixed(3), Dim::Fixed(224), Dim::Fixed(224)],
                Layout::NCHW,
            )
            .unwrap_err();
            assert_eq!(
                err,
                TensorTypeError::LayoutRankMismatch {
                    layout: Layout::NCHW,
                    expected: 4,
                    actual: 3,
                }
            );
        }

        #[test]
        fn new_scalar_rowmajor_error() {
            let err = TensorType::new(DType::F32, vec![], Layout::RowMajor).unwrap_err();
            assert_eq!(err, TensorTypeError::ScalarWithLayout(Layout::RowMajor));
        }

        #[test]
        fn new_scalar_any_ok() {
            assert!(TensorType::new(DType::F32, vec![], Layout::Any).is_ok());
        }

        #[test]
        fn new_fixed_zero_in_shape_error() {
            let err = TensorType::new(
                DType::F32,
                vec![Dim::Fixed(0), Dim::Fixed(4)],
                Layout::RowMajor,
            )
            .unwrap_err();
            assert_eq!(err, TensorTypeError::ZeroDimension);
        }

        #[test]
        fn new_empty_symbolic_in_shape_error() {
            let err = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("".into()), Dim::Fixed(4)],
                Layout::RowMajor,
            )
            .unwrap_err();
            assert_eq!(err, TensorTypeError::EmptySymbol);
        }

        // --- builder ---

        #[test]
        fn builder_dtype_only_builds_scalar() {
            let t = TensorType::builder(DType::F32).build().unwrap();
            assert!(t.is_scalar());
            assert_eq!(t.layout(), Layout::Any);
        }

        #[test]
        fn builder_full_spec() {
            let t = TensorType::builder(DType::F32)
                .shape(vec![
                    Dim::Symbolic("batch".into()),
                    Dim::Fixed(3),
                    Dim::Fixed(224),
                    Dim::Fixed(224),
                ])
                .layout(Layout::NCHW)
                .dim_names(vec![
                    "batch".into(),
                    "channels".into(),
                    "height".into(),
                    "width".into(),
                ])
                .device(DeviceId::new("cuda:0"))
                .build()
                .unwrap();

            assert_eq!(t.rank(), 4);
            assert_eq!(t.layout(), Layout::NCHW);
            assert_eq!(t.dim_names().unwrap().len(), 4);
            assert_eq!(t.device().unwrap(), &DeviceId::new("cuda:0"));
        }

        #[test]
        fn builder_dim_names_length_mismatch() {
            let err = TensorType::builder(DType::F32)
                .shape(vec![Dim::Fixed(4), Dim::Fixed(4)])
                .dim_names(vec!["only_one".into()])
                .build()
                .unwrap_err();
            assert_eq!(
                err,
                TensorTypeError::DimNamesMismatch { names: 1, shape: 2 }
            );
        }

        #[test]
        fn builder_layout_validation_before_dim_names() {
            // Layout error should surface before dim_names mismatch.
            let err = TensorType::builder(DType::F32)
                .shape(vec![Dim::Fixed(3)])
                .layout(Layout::NCHW)
                .dim_names(vec!["only_one".into()])
                .build()
                .unwrap_err();
            assert_eq!(
                err,
                TensorTypeError::LayoutRankMismatch {
                    layout: Layout::NCHW,
                    expected: 4,
                    actual: 1,
                }
            );
        }
    }

    // ── TensorType accessor tests ─────────────────────────────────────────

    mod accessor_tests {
        use super::*;

        fn sample() -> TensorType {
            TensorType::builder(DType::F32)
                .shape(vec![Dim::Fixed(2), Dim::Fixed(3)])
                .layout(Layout::RowMajor)
                .dim_names(vec!["rows".into(), "cols".into()])
                .device(DeviceId::new("cpu"))
                .build()
                .unwrap()
        }

        #[test]
        fn dtype_accessor() {
            assert_eq!(sample().dtype(), DType::F32);
        }

        #[test]
        fn shape_accessor() {
            assert_eq!(sample().shape().dims(), &[Dim::Fixed(2), Dim::Fixed(3)]);
        }

        #[test]
        fn layout_accessor() {
            assert_eq!(sample().layout(), Layout::RowMajor);
        }

        #[test]
        fn dim_names_accessor() {
            let t = sample();
            let names = t.dim_names().unwrap();
            assert_eq!(names, ["rows", "cols"]);
        }

        #[test]
        fn device_accessor() {
            assert_eq!(sample().device(), Some(&DeviceId::new("cpu")));
        }

        #[test]
        fn rank_matches_shape_len() {
            assert_eq!(sample().rank(), 2);
            assert_eq!(TensorType::scalar(DType::U8).rank(), 0);
        }

        #[test]
        fn is_scalar_for_rank_zero() {
            assert!(TensorType::scalar(DType::F32).is_scalar());
            assert!(!sample().is_scalar());
        }

        #[test]
        fn num_elements_all_fixed() {
            assert_eq!(sample().num_elements(), Some(6));
        }

        #[test]
        fn num_elements_scalar() {
            assert_eq!(TensorType::scalar(DType::F32).num_elements(), Some(1));
        }

        #[test]
        fn num_elements_with_dynamic() {
            let t = TensorType::new(
                DType::F32,
                vec![Dim::Dynamic, Dim::Fixed(256)],
                Layout::RowMajor,
            )
            .unwrap();
            assert_eq!(t.num_elements(), None);
        }

        #[test]
        fn num_elements_with_symbolic() {
            let t = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("n".into()), Dim::Fixed(64)],
                Layout::RowMajor,
            )
            .unwrap();
            assert_eq!(t.num_elements(), None);
        }

        #[test]
        fn size_bytes_known() {
            // vector of 1024 f32 → 4096 bytes
            let v = TensorType::vector(DType::F32, 1024).unwrap();
            assert_eq!(v.size_bytes(), Some(4096));
        }

        #[test]
        fn size_bytes_unknown_dynamic() {
            let t = TensorType::new(
                DType::F32,
                vec![Dim::Dynamic, Dim::Fixed(4)],
                Layout::RowMajor,
            )
            .unwrap();
            assert_eq!(t.size_bytes(), None);
        }

        #[test]
        fn size_bytes_custom_dtype() {
            let v = TensorType::new(DType::Custom("q4"), vec![Dim::Fixed(16)], Layout::RowMajor)
                .unwrap();
            assert_eq!(v.size_bytes(), None);
        }
    }

    // ── TensorType transform tests ────────────────────────────────────────

    mod transform_tests {
        use super::*;

        fn rank4_tensor() -> TensorType {
            TensorType::new(
                DType::F32,
                vec![
                    Dim::Fixed(1),
                    Dim::Fixed(3),
                    Dim::Fixed(224),
                    Dim::Fixed(224),
                ],
                Layout::RowMajor,
            )
            .unwrap()
        }

        #[test]
        fn with_layout_valid() {
            let t = rank4_tensor().with_layout(Layout::NCHW).unwrap();
            assert_eq!(t.layout(), Layout::NCHW);
        }

        #[test]
        fn with_layout_invalid_rank() {
            let t = TensorType::matrix(DType::F32, 4, 4).unwrap();
            let err = t.with_layout(Layout::NCHW).unwrap_err();
            assert_eq!(
                err,
                TensorTypeError::LayoutRankMismatch {
                    layout: Layout::NCHW,
                    expected: 4,
                    actual: 2,
                }
            );
        }

        #[test]
        fn with_device_sets_device() {
            let t = TensorType::scalar(DType::F32).with_device(DeviceId::new("cuda:0"));
            assert_eq!(t.device(), Some(&DeviceId::new("cuda:0")));
        }

        #[test]
        fn unplaced_clears_device() {
            let t = TensorType::scalar(DType::F32)
                .with_device(DeviceId::new("cuda:0"))
                .unplaced();
            assert!(t.device().is_none());
        }

        #[test]
        fn with_dim_names_valid() {
            let t = TensorType::matrix(DType::F32, 3, 4)
                .unwrap()
                .with_dim_names(vec!["rows".into(), "cols".into()])
                .unwrap();
            assert_eq!(t.dim_names().unwrap(), ["rows", "cols"]);
        }

        #[test]
        fn with_dim_names_wrong_length() {
            let err = TensorType::matrix(DType::F32, 3, 4)
                .unwrap()
                .with_dim_names(vec!["only_one".into()])
                .unwrap_err();
            assert_eq!(
                err,
                TensorTypeError::DimNamesMismatch { names: 1, shape: 2 }
            );
        }

        #[test]
        fn chaining_with_device_then_unplaced() {
            let t = TensorType::scalar(DType::F32)
                .with_device(DeviceId::new("cpu"))
                .unplaced();
            assert!(t.device().is_none());
            assert!(t.is_scalar());
        }

        #[test]
        fn transforms_do_not_mutate_original_dtype() {
            // with_layout consumes self but dtype should be unchanged.
            let t = rank4_tensor().with_layout(Layout::NCHW).unwrap();
            assert_eq!(t.dtype(), DType::F32);
        }
    }

    // ── TensorType compatibility tests ────────────────────────────────────

    mod compatibility_tests {
        use super::*;

        fn f32_row(shape: Vec<Dim>) -> TensorType {
            TensorType::new(DType::F32, shape, Layout::RowMajor).unwrap()
        }

        #[test]
        fn identical_tensors_are_compatible() {
            let a = TensorType::matrix(DType::F32, 3, 4).unwrap();
            let b = TensorType::matrix(DType::F32, 3, 4).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn different_dtypes_incompatible() {
            let a = TensorType::vector(DType::F32, 4).unwrap();
            let b = TensorType::vector(DType::F64, 4).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn different_ranks_incompatible() {
            let a = TensorType::vector(DType::F32, 4).unwrap();
            let b = TensorType::matrix(DType::F32, 2, 2).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn fixed_same_compatible() {
            let a = f32_row(vec![Dim::Fixed(3), Dim::Fixed(4)]);
            let b = f32_row(vec![Dim::Fixed(3), Dim::Fixed(4)]);
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn fixed_different_incompatible() {
            let a = f32_row(vec![Dim::Fixed(3), Dim::Fixed(4)]);
            let b = f32_row(vec![Dim::Fixed(3), Dim::Fixed(5)]);
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn fixed_vs_dynamic_compatible() {
            let a = f32_row(vec![Dim::Fixed(3), Dim::Fixed(256)]);
            let b = TensorType::new(
                DType::F32,
                vec![Dim::Dynamic, Dim::Fixed(256)],
                Layout::RowMajor,
            )
            .unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn fixed_vs_symbolic_compatible() {
            let a = f32_row(vec![Dim::Fixed(3)]);
            let b = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("n".into())],
                Layout::RowMajor,
            )
            .unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn dynamic_vs_dynamic_compatible() {
            let a = TensorType::new(DType::F32, vec![Dim::Dynamic], Layout::RowMajor).unwrap();
            let b = TensorType::new(DType::F32, vec![Dim::Dynamic], Layout::RowMajor).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn symbolic_same_name_compatible() {
            let a = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("batch".into())],
                Layout::RowMajor,
            )
            .unwrap();
            let b = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("batch".into())],
                Layout::RowMajor,
            )
            .unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn symbolic_different_names_incompatible() {
            let a = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("batch".into())],
                Layout::RowMajor,
            )
            .unwrap();
            let b = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("seq".into())],
                Layout::RowMajor,
            )
            .unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn layout_any_vs_rowmajor_compatible() {
            let a = TensorType::new(DType::F32, vec![Dim::Fixed(4)], Layout::Any).unwrap();
            let b = TensorType::new(DType::F32, vec![Dim::Fixed(4)], Layout::RowMajor).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn rowmajor_vs_colmajor_incompatible() {
            let a = TensorType::new(DType::F32, vec![Dim::Fixed(4)], Layout::RowMajor).unwrap();
            let b = TensorType::new(DType::F32, vec![Dim::Fixed(4)], Layout::ColMajor).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn device_is_ignored_in_compatibility() {
            let a = TensorType::scalar(DType::F32).with_device(DeviceId::new("cuda:0"));
            let b = TensorType::scalar(DType::F32).with_device(DeviceId::new("cpu"));
            // Different devices but both are F32 scalars with Any layout.
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn compatibility_is_symmetric() {
            let a = TensorType::new(
                DType::F32,
                vec![Dim::Fixed(3), Dim::Dynamic],
                Layout::RowMajor,
            )
            .unwrap();
            let b = TensorType::new(DType::F32, vec![Dim::Fixed(3), Dim::Fixed(64)], Layout::Any)
                .unwrap();
            assert_eq!(a.is_compatible_with(&b), b.is_compatible_with(&a));
        }
    }

    // ── Display tests ─────────────────────────────────────────────────────

    mod display_tests {
        use super::*;

        #[test]
        fn display_scalar() {
            let s = TensorType::scalar(DType::F32).to_string();
            assert_eq!(s, "f32[] Any");
        }

        #[test]
        fn display_vector() {
            let v = TensorType::vector(DType::I32, 1024).unwrap().to_string();
            assert_eq!(v, "i32[1024] RowMajor");
        }

        #[test]
        fn display_with_device() {
            let t = TensorType::new(
                DType::F32,
                vec![
                    Dim::Symbolic("batch".into()),
                    Dim::Fixed(3),
                    Dim::Fixed(224),
                    Dim::Fixed(224),
                ],
                Layout::NCHW,
            )
            .unwrap()
            .with_device(DeviceId::new("cuda:0"));
            assert_eq!(t.to_string(), "f32[batch, 3, 224, 224] NCHW @ cuda:0");
        }

        #[test]
        fn display_dynamic_dim() {
            let t = TensorType::new(
                DType::F64,
                vec![Dim::Dynamic, Dim::Fixed(256)],
                Layout::RowMajor,
            )
            .unwrap();
            assert_eq!(t.to_string(), "f64[?, 256] RowMajor");
        }

        #[test]
        fn display_mixed_dims() {
            let t = TensorType::new(
                DType::F32,
                vec![Dim::Symbolic("batch".into()), Dim::Dynamic, Dim::Fixed(3)],
                Layout::Any,
            )
            .unwrap();
            assert_eq!(t.to_string(), "f32[batch, ?, 3] Any");
        }
    }

    // ── TensorType clone / eq / debug ─────────────────────────────────────

    mod derive_tests {
        use super::*;

        #[test]
        fn tensor_type_clone_eq() {
            let a = TensorType::matrix(DType::F32, 4, 4).unwrap();
            let b = a.clone();
            assert_eq!(a, b);
        }

        #[test]
        fn tensor_type_debug() {
            let s = format!("{:?}", TensorType::scalar(DType::F32));
            assert!(s.contains("TensorType"));
        }
    }
}
