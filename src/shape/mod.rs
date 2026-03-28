mod ops;

use std::fmt;

use thiserror::Error;

use crate::tensor_type::Dim;

// ── ShapeError ───────────────────────────────────────────────────────────────

/// Errors produced when constructing or transforming a [`Shape`].
///
/// Every variant carries enough context for the caller to understand exactly
/// what invariant was violated and how to fix it.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum ShapeError {
    /// A `Fixed(0)` dimension was supplied. All fixed dimensions must be >= 1.
    #[error("Dimension must be > 0, got 0")]
    ZeroDimension,

    /// A `Symbolic("")` dimension was supplied. Symbolic names must be
    /// non-empty so they can be matched across graph edges.
    #[error("Symbolic dimension name must not be empty")]
    EmptySymbol,

    /// Two shapes cannot be broadcast together under NumPy-style rules.
    #[error("Cannot broadcast shapes {left} and {right}")]
    IncompatibleBroadcast {
        /// The left-hand shape.
        left: Shape,
        /// The right-hand shape.
        right: Shape,
    },

    /// A reshape would change the total element count.
    #[error("Reshape element count mismatch: source has {from} elements, target has {to}")]
    ReshapeElementMismatch {
        /// Element count of the source shape.
        from: usize,
        /// Element count of the target shape.
        to: usize,
    },

    /// A reshape involves dynamic/symbolic dims on both sides, making it
    /// impossible to verify element-count preservation at build time.
    #[error(
        "Cannot verify reshape: both source and target contain dynamic or symbolic dimensions"
    )]
    ReshapeDynamicAmbiguous,

    /// An operation expected a specific rank but received a different one.
    #[error("Expected rank {expected}, got {actual}")]
    RankMismatch {
        /// The rank the caller expected.
        expected: usize,
        /// The rank that was actually provided.
        actual: usize,
    },
}

// ── Shape ────────────────────────────────────────────────────────────────────

/// A validated tensor shape — an ordered list of [`Dim`]s.
///
/// `Shape` encapsulates all shape-related logic: rank, element count,
/// compatibility checks, broadcasting, reshape validation, and stride
/// computation. It is the canonical way to describe the dimensionality of
/// a tensor in the graphynx type system.
///
/// # Invariants (always upheld)
///
/// - No [`Dim::Fixed(0)`] in dims.
/// - No [`Dim::Symbolic("")`] in dims.
///
/// All fields are private. Use [`Shape::new`], [`Shape::scalar`],
/// [`Shape::vector`], [`Shape::matrix`], or [`Shape::from_fixed`] to
/// construct instances.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Shape {
    dims: Vec<Dim>,
}

impl Shape {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Construct a shape from a vec of [`Dim`]s, validating all invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::ZeroDimension`] if any dim is `Fixed(0)`.
    /// Returns [`ShapeError::EmptySymbol`] if any dim is `Symbolic("")`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    /// use graphynx::tensor_type::Dim;
    ///
    /// let s = Shape::new(vec![Dim::Fixed(3), Dim::Fixed(224)]).unwrap();
    /// assert_eq!(s.rank(), 2);
    /// ```
    pub fn new(dims: Vec<Dim>) -> Result<Self, ShapeError> {
        Self::validate_dims(&dims)?;
        Ok(Shape { dims })
    }

    /// Rank-0 (scalar) shape. Contains no dimensions.
    ///
    /// Infallible — a scalar has no dims to validate.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::scalar();
    /// assert!(s.is_scalar());
    /// assert_eq!(s.rank(), 0);
    /// ```
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    /// Rank-1 shape with a single [`Dim::Fixed`] dimension.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::ZeroDimension`] if `len == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::vector(1024).unwrap();
    /// assert_eq!(s.rank(), 1);
    /// assert_eq!(s.num_elements(), Some(1024));
    /// ```
    pub fn vector(len: usize) -> Result<Self, ShapeError> {
        if len == 0 {
            return Err(ShapeError::ZeroDimension);
        }
        Ok(Shape {
            dims: vec![Dim::Fixed(len)],
        })
    }

    /// Rank-2 shape with two [`Dim::Fixed`] dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::ZeroDimension`] if `rows == 0` or `cols == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::matrix(3, 4).unwrap();
    /// assert_eq!(s.rank(), 2);
    /// assert_eq!(s.num_elements(), Some(12));
    /// ```
    pub fn matrix(rows: usize, cols: usize) -> Result<Self, ShapeError> {
        if rows == 0 || cols == 0 {
            return Err(ShapeError::ZeroDimension);
        }
        Ok(Shape {
            dims: vec![Dim::Fixed(rows), Dim::Fixed(cols)],
        })
    }

    /// Construct a shape from a slice of `usize` values, all as [`Dim::Fixed`].
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::ZeroDimension`] if any value is `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::from_fixed(&[1, 3, 224, 224]).unwrap();
    /// assert_eq!(s.rank(), 4);
    /// assert_eq!(s.num_elements(), Some(1 * 3 * 224 * 224));
    /// ```
    pub fn from_fixed(sizes: &[usize]) -> Result<Self, ShapeError> {
        let dims: Vec<Dim> = sizes.iter().map(|&n| Dim::Fixed(n)).collect();
        Self::validate_dims(&dims)?;
        Ok(Shape { dims })
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// The dimensions as a slice of [`Dim`]s.
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// Number of dimensions (0 for scalars).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns `true` for rank-0 shapes.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Total number of elements, if every dimension is [`Dim::Fixed`].
    ///
    /// Returns `None` if any dimension is [`Dim::Dynamic`] or
    /// [`Dim::Symbolic`]. Returns `Some(1)` for scalars (one element, no
    /// dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    /// use graphynx::tensor_type::Dim;
    ///
    /// let s = Shape::from_fixed(&[2, 3, 4]).unwrap();
    /// assert_eq!(s.num_elements(), Some(24));
    ///
    /// let d = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
    /// assert_eq!(d.num_elements(), None);
    /// ```
    pub fn num_elements(&self) -> Option<usize> {
        if self.dims.is_empty() {
            return Some(1);
        }
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| dim.fixed_value().map(|n| acc * n))
    }

    // ── Reshape validation ───────────────────────────────────────────────

    /// Check whether `self` can be reshaped to `target` while preserving the
    /// total element count.
    ///
    /// # Rules
    ///
    /// - If both shapes are fully fixed, their element counts must match.
    /// - If one or both shapes contain dynamic/symbolic dims, the check
    ///   returns [`ShapeError::ReshapeDynamicAmbiguous`] because we cannot
    ///   verify element-count preservation at build time.
    /// - Scalars can be reshaped to any shape with exactly 1 element and
    ///   vice versa.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::ReshapeElementMismatch`] if both shapes are
    /// fully fixed but have different element counts.
    /// Returns [`ShapeError::ReshapeDynamicAmbiguous`] if verification is
    /// impossible due to dynamic dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let a = Shape::from_fixed(&[2, 3, 4]).unwrap();
    /// let b = Shape::from_fixed(&[6, 4]).unwrap();
    /// assert!(a.can_reshape_to(&b).is_ok());
    ///
    /// let c = Shape::from_fixed(&[5, 5]).unwrap();
    /// assert!(a.can_reshape_to(&c).is_err());
    /// ```
    pub fn can_reshape_to(&self, target: &Shape) -> Result<(), ShapeError> {
        let src_elems = self.num_elements();
        let tgt_elems = target.num_elements();

        match (src_elems, tgt_elems) {
            (Some(s), Some(t)) => {
                if s == t {
                    Ok(())
                } else {
                    Err(ShapeError::ReshapeElementMismatch { from: s, to: t })
                }
            }
            _ => Err(ShapeError::ReshapeDynamicAmbiguous),
        }
    }

    // ── Stride computation ───────────────────────────────────────────────

    /// Compute row-major (C-contiguous) strides for this shape.
    ///
    /// Row-major means the last dimension varies fastest. Stride values
    /// are in units of elements (not bytes).
    ///
    /// Returns `None` if any dimension is not [`Dim::Fixed`].
    /// Returns an empty vec for scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::from_fixed(&[2, 3, 4]).unwrap();
    /// assert_eq!(s.row_major_strides(), Some(vec![12, 4, 1]));
    ///
    /// let scalar = Shape::scalar();
    /// assert_eq!(scalar.row_major_strides(), Some(vec![]));
    /// ```
    pub fn row_major_strides(&self) -> Option<Vec<usize>> {
        if self.dims.is_empty() {
            return Some(vec![]);
        }
        let sizes = self.fixed_sizes()?;
        let mut strides = vec![0usize; sizes.len()];
        let mut stride = 1usize;
        // Walk right to left, accumulating the product.
        for i in (0..sizes.len()).rev() {
            strides[i] = stride;
            stride *= sizes[i];
        }
        Some(strides)
    }

    /// Compute column-major (Fortran-contiguous) strides for this shape.
    ///
    /// Column-major means the first dimension varies fastest. Stride values
    /// are in units of elements (not bytes).
    ///
    /// Returns `None` if any dimension is not [`Dim::Fixed`].
    /// Returns an empty vec for scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let s = Shape::from_fixed(&[2, 3, 4]).unwrap();
    /// assert_eq!(s.col_major_strides(), Some(vec![1, 2, 6]));
    ///
    /// let scalar = Shape::scalar();
    /// assert_eq!(scalar.col_major_strides(), Some(vec![]));
    /// ```
    pub fn col_major_strides(&self) -> Option<Vec<usize>> {
        if self.dims.is_empty() {
            return Some(vec![]);
        }
        let sizes = self.fixed_sizes()?;
        let mut strides = vec![0usize; sizes.len()];
        let mut stride = 1usize;
        // Walk left to right, accumulating the product.
        for i in 0..sizes.len() {
            strides[i] = stride;
            stride *= sizes[i];
        }
        Some(strides)
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Validate that no dim is Fixed(0) or Symbolic("").
    fn validate_dims(dims: &[Dim]) -> Result<(), ShapeError> {
        for dim in dims {
            match dim {
                Dim::Fixed(0) => return Err(ShapeError::ZeroDimension),
                Dim::Symbolic(s) if s.is_empty() => return Err(ShapeError::EmptySymbol),
                _ => {}
            }
        }
        Ok(())
    }

    /// Extract all dims as fixed sizes, returning `None` if any is
    /// dynamic/symbolic.
    fn fixed_sizes(&self) -> Option<Vec<usize>> {
        self.dims.iter().map(|d| d.fixed_value()).collect()
    }
}

impl fmt::Display for Shape {
    /// Formats the shape as `[dim, dim, ...]`.
    ///
    /// # Examples
    ///
    /// ```text
    /// []
    /// [1024]
    /// [batch, 3, 224, 224]
    /// [?, 256]
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShapeError tests ─────────────────────────────────────────────────

    mod error_tests {
        use super::*;

        #[test]
        fn zero_dimension_message() {
            let msg = ShapeError::ZeroDimension.to_string();
            assert!(msg.contains("0"), "msg was: {msg}");
        }

        #[test]
        fn empty_symbol_message() {
            let msg = ShapeError::EmptySymbol.to_string();
            assert!(msg.to_lowercase().contains("empty"), "msg was: {msg}");
        }

        #[test]
        fn incompatible_broadcast_message() {
            let a = Shape::from_fixed(&[3]).unwrap();
            let b = Shape::from_fixed(&[4]).unwrap();
            let msg = ShapeError::IncompatibleBroadcast { left: a, right: b }.to_string();
            assert!(msg.contains("broadcast"), "msg was: {msg}");
        }

        #[test]
        fn reshape_element_mismatch_message() {
            let msg = ShapeError::ReshapeElementMismatch { from: 12, to: 10 }.to_string();
            assert!(msg.contains("12") && msg.contains("10"), "msg was: {msg}");
        }

        #[test]
        fn reshape_dynamic_ambiguous_message() {
            let msg = ShapeError::ReshapeDynamicAmbiguous.to_string();
            assert!(msg.to_lowercase().contains("dynamic"), "msg was: {msg}");
        }

        #[test]
        fn rank_mismatch_message() {
            let msg = ShapeError::RankMismatch {
                expected: 4,
                actual: 3,
            }
            .to_string();
            assert!(msg.contains('4') && msg.contains('3'), "msg was: {msg}");
        }

        #[test]
        fn errors_clone_eq() {
            let e = ShapeError::ZeroDimension;
            assert_eq!(e.clone(), e);
        }

        #[test]
        fn errors_debug() {
            let s = format!("{:?}", ShapeError::EmptySymbol);
            assert!(s.contains("EmptySymbol"));
        }
    }

    // ── Constructor tests ────────────────────────────────────────────────

    mod constructor_tests {
        use super::*;

        #[test]
        fn new_valid() {
            let s = Shape::new(vec![Dim::Fixed(3), Dim::Fixed(4)]).unwrap();
            assert_eq!(s.rank(), 2);
        }

        #[test]
        fn new_empty_is_scalar() {
            let s = Shape::new(vec![]).unwrap();
            assert!(s.is_scalar());
        }

        #[test]
        fn new_zero_dim_error() {
            let err = Shape::new(vec![Dim::Fixed(0), Dim::Fixed(4)]).unwrap_err();
            assert_eq!(err, ShapeError::ZeroDimension);
        }

        #[test]
        fn new_empty_symbolic_error() {
            let err = Shape::new(vec![Dim::Symbolic("".into())]).unwrap_err();
            assert_eq!(err, ShapeError::EmptySymbol);
        }

        #[test]
        fn new_with_dynamic() {
            let s = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            assert_eq!(s.rank(), 2);
        }

        #[test]
        fn new_with_symbolic() {
            let s = Shape::new(vec![Dim::Symbolic("batch".into()), Dim::Fixed(256)]).unwrap();
            assert_eq!(s.rank(), 2);
        }

        #[test]
        fn scalar() {
            let s = Shape::scalar();
            assert!(s.is_scalar());
            assert_eq!(s.rank(), 0);
        }

        #[test]
        fn vector_valid() {
            let s = Shape::vector(512).unwrap();
            assert_eq!(s.rank(), 1);
            assert_eq!(s.dims(), &[Dim::Fixed(512)]);
        }

        #[test]
        fn vector_zero_error() {
            assert_eq!(Shape::vector(0), Err(ShapeError::ZeroDimension));
        }

        #[test]
        fn matrix_valid() {
            let s = Shape::matrix(3, 4).unwrap();
            assert_eq!(s.rank(), 2);
            assert_eq!(s.dims(), &[Dim::Fixed(3), Dim::Fixed(4)]);
        }

        #[test]
        fn matrix_zero_rows_error() {
            assert_eq!(Shape::matrix(0, 4), Err(ShapeError::ZeroDimension));
        }

        #[test]
        fn matrix_zero_cols_error() {
            assert_eq!(Shape::matrix(4, 0), Err(ShapeError::ZeroDimension));
        }

        #[test]
        fn from_fixed_valid() {
            let s = Shape::from_fixed(&[1, 3, 224, 224]).unwrap();
            assert_eq!(s.rank(), 4);
        }

        #[test]
        fn from_fixed_empty_is_scalar() {
            let s = Shape::from_fixed(&[]).unwrap();
            assert!(s.is_scalar());
        }

        #[test]
        fn from_fixed_zero_error() {
            let err = Shape::from_fixed(&[3, 0, 224]).unwrap_err();
            assert_eq!(err, ShapeError::ZeroDimension);
        }
    }

    // ── Accessor tests ───────────────────────────────────────────────────

    mod accessor_tests {
        use super::*;

        #[test]
        fn dims_returns_slice() {
            let s = Shape::from_fixed(&[2, 3]).unwrap();
            assert_eq!(s.dims(), &[Dim::Fixed(2), Dim::Fixed(3)]);
        }

        #[test]
        fn rank_scalar() {
            assert_eq!(Shape::scalar().rank(), 0);
        }

        #[test]
        fn rank_vector() {
            assert_eq!(Shape::vector(10).unwrap().rank(), 1);
        }

        #[test]
        fn is_scalar_true() {
            assert!(Shape::scalar().is_scalar());
        }

        #[test]
        fn is_scalar_false() {
            assert!(!Shape::vector(5).unwrap().is_scalar());
        }

        #[test]
        fn num_elements_fixed() {
            assert_eq!(
                Shape::from_fixed(&[2, 3, 4]).unwrap().num_elements(),
                Some(24)
            );
        }

        #[test]
        fn num_elements_scalar() {
            assert_eq!(Shape::scalar().num_elements(), Some(1));
        }

        #[test]
        fn num_elements_dynamic_returns_none() {
            let s = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            assert_eq!(s.num_elements(), None);
        }

        #[test]
        fn num_elements_symbolic_returns_none() {
            let s = Shape::new(vec![Dim::Symbolic("n".into()), Dim::Fixed(4)]).unwrap();
            assert_eq!(s.num_elements(), None);
        }

        #[test]
        fn num_elements_single_dim() {
            assert_eq!(Shape::vector(7).unwrap().num_elements(), Some(7));
        }
    }

    // ── Reshape validation tests ─────────────────────────────────────────

    mod reshape_tests {
        use super::*;

        #[test]
        fn reshape_same_elements_ok() {
            let a = Shape::from_fixed(&[2, 3, 4]).unwrap();
            let b = Shape::from_fixed(&[6, 4]).unwrap();
            assert!(a.can_reshape_to(&b).is_ok());
        }

        #[test]
        fn reshape_different_elements_error() {
            let a = Shape::from_fixed(&[2, 3, 4]).unwrap();
            let b = Shape::from_fixed(&[5, 5]).unwrap();
            let err = a.can_reshape_to(&b).unwrap_err();
            assert_eq!(err, ShapeError::ReshapeElementMismatch { from: 24, to: 25 });
        }

        #[test]
        fn reshape_to_flat() {
            let a = Shape::from_fixed(&[2, 3, 4]).unwrap();
            let b = Shape::from_fixed(&[24]).unwrap();
            assert!(a.can_reshape_to(&b).is_ok());
        }

        #[test]
        fn reshape_scalar_to_single_element() {
            let a = Shape::scalar();
            let b = Shape::from_fixed(&[1, 1, 1]).unwrap();
            assert!(a.can_reshape_to(&b).is_ok());
        }

        #[test]
        fn reshape_single_element_to_scalar() {
            let a = Shape::from_fixed(&[1]).unwrap();
            let b = Shape::scalar();
            assert!(a.can_reshape_to(&b).is_ok());
        }

        #[test]
        fn reshape_dynamic_source_error() {
            let a = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            let b = Shape::from_fixed(&[16]).unwrap();
            let err = a.can_reshape_to(&b).unwrap_err();
            assert_eq!(err, ShapeError::ReshapeDynamicAmbiguous);
        }

        #[test]
        fn reshape_dynamic_target_error() {
            let a = Shape::from_fixed(&[24]).unwrap();
            let b = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            let err = a.can_reshape_to(&b).unwrap_err();
            assert_eq!(err, ShapeError::ReshapeDynamicAmbiguous);
        }

        #[test]
        fn reshape_both_dynamic_error() {
            let a = Shape::new(vec![Dim::Dynamic]).unwrap();
            let b = Shape::new(vec![Dim::Dynamic, Dim::Fixed(3)]).unwrap();
            assert_eq!(
                a.can_reshape_to(&b).unwrap_err(),
                ShapeError::ReshapeDynamicAmbiguous
            );
        }

        #[test]
        fn reshape_scalar_to_scalar() {
            assert!(Shape::scalar().can_reshape_to(&Shape::scalar()).is_ok());
        }
    }

    // ── Stride tests ─────────────────────────────────────────────────────

    mod stride_tests {
        use super::*;

        #[test]
        fn row_major_3d() {
            let s = Shape::from_fixed(&[2, 3, 4]).unwrap();
            assert_eq!(s.row_major_strides(), Some(vec![12, 4, 1]));
        }

        #[test]
        fn row_major_2d() {
            let s = Shape::from_fixed(&[4, 5]).unwrap();
            assert_eq!(s.row_major_strides(), Some(vec![5, 1]));
        }

        #[test]
        fn row_major_1d() {
            let s = Shape::from_fixed(&[10]).unwrap();
            assert_eq!(s.row_major_strides(), Some(vec![1]));
        }

        #[test]
        fn row_major_scalar() {
            assert_eq!(Shape::scalar().row_major_strides(), Some(vec![]));
        }

        #[test]
        fn row_major_dynamic_returns_none() {
            let s = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            assert_eq!(s.row_major_strides(), None);
        }

        #[test]
        fn col_major_3d() {
            let s = Shape::from_fixed(&[2, 3, 4]).unwrap();
            assert_eq!(s.col_major_strides(), Some(vec![1, 2, 6]));
        }

        #[test]
        fn col_major_2d() {
            let s = Shape::from_fixed(&[4, 5]).unwrap();
            assert_eq!(s.col_major_strides(), Some(vec![1, 4]));
        }

        #[test]
        fn col_major_1d() {
            let s = Shape::from_fixed(&[10]).unwrap();
            assert_eq!(s.col_major_strides(), Some(vec![1]));
        }

        #[test]
        fn col_major_scalar() {
            assert_eq!(Shape::scalar().col_major_strides(), Some(vec![]));
        }

        #[test]
        fn col_major_dynamic_returns_none() {
            let s = Shape::new(vec![Dim::Dynamic, Dim::Fixed(8)]).unwrap();
            assert_eq!(s.col_major_strides(), None);
        }

        #[test]
        fn row_major_4d() {
            // [1, 3, 224, 224]
            let s = Shape::from_fixed(&[1, 3, 224, 224]).unwrap();
            // strides: 3*224*224=150528, 224*224=50176, 224, 1
            assert_eq!(s.row_major_strides(), Some(vec![150528, 50176, 224, 1]));
        }

        #[test]
        fn col_major_4d() {
            let s = Shape::from_fixed(&[1, 3, 224, 224]).unwrap();
            // strides: 1, 1*1=1, 1*3=3, 3*224=672
            assert_eq!(s.col_major_strides(), Some(vec![1, 1, 3, 672]));
        }
    }

    // ── Display tests ────────────────────────────────────────────────────

    mod display_tests {
        use super::*;

        #[test]
        fn display_scalar() {
            assert_eq!(Shape::scalar().to_string(), "[]");
        }

        #[test]
        fn display_vector() {
            assert_eq!(Shape::vector(1024).unwrap().to_string(), "[1024]");
        }

        #[test]
        fn display_matrix() {
            assert_eq!(Shape::matrix(3, 4).unwrap().to_string(), "[3, 4]");
        }

        #[test]
        fn display_with_dynamic() {
            let s = Shape::new(vec![Dim::Dynamic, Dim::Fixed(256)]).unwrap();
            assert_eq!(s.to_string(), "[?, 256]");
        }

        #[test]
        fn display_with_symbolic() {
            let s = Shape::new(vec![
                Dim::Symbolic("batch".into()),
                Dim::Fixed(3),
                Dim::Fixed(224),
                Dim::Fixed(224),
            ])
            .unwrap();
            assert_eq!(s.to_string(), "[batch, 3, 224, 224]");
        }

        #[test]
        fn display_mixed() {
            let s = Shape::new(vec![
                Dim::Symbolic("batch".into()),
                Dim::Dynamic,
                Dim::Fixed(3),
            ])
            .unwrap();
            assert_eq!(s.to_string(), "[batch, ?, 3]");
        }
    }

    // ── Derive tests ─────────────────────────────────────────────────────

    mod derive_tests {
        use std::collections::HashMap;

        use super::*;

        #[test]
        fn shape_clone_eq() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = a.clone();
            assert_eq!(a, b);
        }

        #[test]
        fn shape_hash() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = a.clone();
            let mut map = HashMap::new();
            map.insert(a, "val");
            assert_eq!(map[&b], "val");
        }

        #[test]
        fn shape_debug() {
            let s = format!("{:?}", Shape::from_fixed(&[2, 3]).unwrap());
            assert!(s.contains("Shape"));
        }

        #[test]
        fn shape_ne() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = Shape::from_fixed(&[3, 5]).unwrap();
            assert_ne!(a, b);
        }
    }
}
