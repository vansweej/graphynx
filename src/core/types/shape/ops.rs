//! Broadcasting rules and compatibility logic for [`Shape`].
//!
//! This module contains the computational operations that determine how two
//! shapes relate to each other: dimension-level compatibility checks and
//! NumPy-style broadcasting.  The pure data definitions (struct, error,
//! constructors, accessors, Display) live in the parent [`super`] module.

use crate::core::types::dim::Dim;

use super::{Shape, ShapeError};

impl Shape {
    // ── Compatibility ────────────────────────────────────────────────────

    /// Returns `true` if `self` and `other` have the same rank and each
    /// dimension pair satisfies [`Dim::is_compatible_with`].
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    /// use graphynx::tensor_type::Dim;
    ///
    /// let a = Shape::from_fixed(&[3, 256]).unwrap();
    /// let b = Shape::new(vec![Dim::Dynamic, Dim::Fixed(256)]).unwrap();
    /// assert!(a.is_compatible_with(&b));
    /// ```
    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() {
            return false;
        }
        self.dims()
            .iter()
            .zip(other.dims().iter())
            .all(|(a, b)| a.is_compatible_with(b))
    }

    // ── Broadcasting ─────────────────────────────────────────────────────

    /// Compute the broadcast shape of `self` and `other` using NumPy-style
    /// broadcasting rules.
    ///
    /// # Rules
    ///
    /// 1. Shapes are right-aligned (shorter shape is left-padded with 1s).
    /// 2. For each pair of dimensions (from the right):
    ///    - `Fixed(a)` and `Fixed(b)`: one must be 1, or `a == b`.
    ///    - `Fixed(1)` and anything: the other dim wins.
    ///    - `Dynamic`/`Symbolic` vs `Fixed(n)` where `n > 1`: the fixed dim
    ///      wins (the dynamic dim is assumed broadcastable).
    ///    - `Dynamic` vs `Dynamic`: result is `Dynamic`.
    ///    - `Symbolic(a)` vs `Symbolic(b)`: must share the same name.
    ///    - `Dynamic` vs `Symbolic(s)`: result is `Symbolic(s)` (dynamic
    ///      matches the named constraint).
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError::IncompatibleBroadcast`] if the shapes cannot be
    /// broadcast together.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::shape::Shape;
    ///
    /// let a = Shape::from_fixed(&[4, 1, 3]).unwrap();
    /// let b = Shape::from_fixed(&[5, 3]).unwrap();
    /// let c = a.broadcast_with(&b).unwrap();
    /// assert_eq!(c, Shape::from_fixed(&[4, 5, 3]).unwrap());
    /// ```
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape, ShapeError> {
        let max_rank = self.rank().max(other.rank());
        let mut result_dims: Vec<Dim> = Vec::with_capacity(max_rank);

        // Iterate from the rightmost dimension to the left, right-aligning
        // the two shapes.
        for i in 0..max_rank {
            let a = if i < self.rank() {
                &self.dims()[self.rank() - 1 - i]
            } else {
                // Left-pad with 1.
                &Dim::Fixed(1)
            };
            let b = if i < other.rank() {
                &other.dims()[other.rank() - 1 - i]
            } else {
                // Left-pad with 1.
                &Dim::Fixed(1)
            };

            let merged = broadcast_dim(a, b).ok_or_else(|| ShapeError::IncompatibleBroadcast {
                left: self.clone(),
                right: other.clone(),
            })?;
            result_dims.push(merged);
        }

        // We built dims right-to-left, so reverse.
        result_dims.reverse();
        Ok(Shape::new(result_dims).expect("broadcast produces only valid dims"))
    }
}

/// Broadcast two individual dims together.
///
/// Returns `None` if incompatible.
fn broadcast_dim(a: &Dim, b: &Dim) -> Option<Dim> {
    match (a, b) {
        // Fixed vs Fixed: equal or one is 1.
        (Dim::Fixed(x), Dim::Fixed(y)) => {
            if x == y {
                Some(Dim::Fixed(*x))
            } else if *x == 1 {
                Some(Dim::Fixed(*y))
            } else if *y == 1 {
                Some(Dim::Fixed(*x))
            } else {
                None
            }
        }
        // Fixed(1) broadcasts to anything.
        (Dim::Fixed(1), other) | (other, Dim::Fixed(1)) => Some(other.clone()),
        // Fixed(n > 1) vs Dynamic: fixed wins (dynamic is assumed
        // broadcastable at runtime).
        (Dim::Fixed(n), Dim::Dynamic) | (Dim::Dynamic, Dim::Fixed(n)) => Some(Dim::Fixed(*n)),
        // Fixed(n > 1) vs Symbolic: fixed wins.
        (Dim::Fixed(n), Dim::Symbolic(_)) | (Dim::Symbolic(_), Dim::Fixed(n)) => {
            Some(Dim::Fixed(*n))
        }
        // Dynamic vs Dynamic.
        (Dim::Dynamic, Dim::Dynamic) => Some(Dim::Dynamic),
        // Dynamic vs Symbolic: symbolic wins (preserves the constraint).
        (Dim::Dynamic, Dim::Symbolic(s)) | (Dim::Symbolic(s), Dim::Dynamic) => {
            Some(Dim::Symbolic(s.clone()))
        }
        // Symbolic vs Symbolic: must share the same name.
        (Dim::Symbolic(a_name), Dim::Symbolic(b_name)) => {
            if a_name == b_name {
                Some(Dim::Symbolic(a_name.clone()))
            } else {
                None
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Compatibility tests ──────────────────────────────────────────────

    mod compatibility_tests {
        use super::*;

        #[test]
        fn identical_shapes_compatible() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = Shape::from_fixed(&[3, 4]).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn different_fixed_incompatible() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = Shape::from_fixed(&[3, 5]).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn different_ranks_incompatible() {
            let a = Shape::vector(4).unwrap();
            let b = Shape::matrix(2, 2).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn fixed_vs_dynamic_compatible() {
            let a = Shape::from_fixed(&[3, 256]).unwrap();
            let b = Shape::new(vec![Dim::Dynamic, Dim::Fixed(256)]).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn symbolic_same_name_compatible() {
            let a = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            assert!(a.is_compatible_with(&b));
        }

        #[test]
        fn symbolic_different_names_incompatible() {
            let a = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("seq".into())]).unwrap();
            assert!(!a.is_compatible_with(&b));
        }

        #[test]
        fn compatibility_is_symmetric() {
            let a = Shape::new(vec![Dim::Fixed(3), Dim::Dynamic]).unwrap();
            let b = Shape::from_fixed(&[3, 64]).unwrap();
            assert_eq!(a.is_compatible_with(&b), b.is_compatible_with(&a));
        }

        #[test]
        fn scalar_vs_scalar_compatible() {
            assert!(Shape::scalar().is_compatible_with(&Shape::scalar()));
        }
    }

    // ── Broadcasting tests ───────────────────────────────────────────────

    mod broadcast_tests {
        use super::*;

        #[test]
        fn same_shape_broadcast() {
            let a = Shape::from_fixed(&[3, 4]).unwrap();
            let b = Shape::from_fixed(&[3, 4]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[3, 4]).unwrap());
        }

        #[test]
        fn broadcast_with_one() {
            let a = Shape::from_fixed(&[3, 1]).unwrap();
            let b = Shape::from_fixed(&[3, 4]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[3, 4]).unwrap());
        }

        #[test]
        fn broadcast_one_reversed() {
            let a = Shape::from_fixed(&[1, 4]).unwrap();
            let b = Shape::from_fixed(&[3, 4]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[3, 4]).unwrap());
        }

        #[test]
        fn broadcast_different_ranks() {
            // [4, 1, 3] broadcast with [5, 3] => [4, 5, 3]
            let a = Shape::from_fixed(&[4, 1, 3]).unwrap();
            let b = Shape::from_fixed(&[5, 3]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[4, 5, 3]).unwrap());
        }

        #[test]
        fn broadcast_scalar_with_any() {
            let a = Shape::scalar();
            let b = Shape::from_fixed(&[3, 4]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[3, 4]).unwrap());
        }

        #[test]
        fn broadcast_rank1_with_rank3() {
            // [3] broadcast with [2, 4, 3] => [2, 4, 3]
            let a = Shape::from_fixed(&[3]).unwrap();
            let b = Shape::from_fixed(&[2, 4, 3]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[2, 4, 3]).unwrap());
        }

        #[test]
        fn broadcast_ones_both_sides() {
            // [1, 3] broadcast with [4, 1] => [4, 3]
            let a = Shape::from_fixed(&[1, 3]).unwrap();
            let b = Shape::from_fixed(&[4, 1]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[4, 3]).unwrap());
        }

        #[test]
        fn broadcast_incompatible() {
            let a = Shape::from_fixed(&[3]).unwrap();
            let b = Shape::from_fixed(&[4]).unwrap();
            let err = a.broadcast_with(&b).unwrap_err();
            assert!(matches!(err, ShapeError::IncompatibleBroadcast { .. }));
        }

        #[test]
        fn broadcast_incompatible_inner_dim() {
            let a = Shape::from_fixed(&[2, 3]).unwrap();
            let b = Shape::from_fixed(&[2, 4]).unwrap();
            assert!(a.broadcast_with(&b).is_err());
        }

        #[test]
        fn broadcast_dynamic_with_dynamic() {
            let a = Shape::new(vec![Dim::Dynamic]).unwrap();
            let b = Shape::new(vec![Dim::Dynamic]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Dynamic]);
        }

        #[test]
        fn broadcast_dynamic_with_fixed() {
            let a = Shape::new(vec![Dim::Dynamic]).unwrap();
            let b = Shape::from_fixed(&[5]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Fixed(5)]);
        }

        #[test]
        fn broadcast_symbolic_same_name() {
            let a = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Symbolic("batch".into())]);
        }

        #[test]
        fn broadcast_symbolic_different_names_fails() {
            let a = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("seq".into())]).unwrap();
            assert!(a.broadcast_with(&b).is_err());
        }

        #[test]
        fn broadcast_dynamic_with_symbolic() {
            let a = Shape::new(vec![Dim::Dynamic]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("batch".into())]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Symbolic("batch".into())]);
        }

        #[test]
        fn broadcast_fixed_one_with_symbolic() {
            let a = Shape::new(vec![Dim::Fixed(1)]).unwrap();
            let b = Shape::new(vec![Dim::Symbolic("n".into())]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Symbolic("n".into())]);
        }

        #[test]
        fn broadcast_fixed_one_with_dynamic() {
            let a = Shape::new(vec![Dim::Fixed(1)]).unwrap();
            let b = Shape::new(vec![Dim::Dynamic]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Dynamic]);
        }

        #[test]
        fn broadcast_is_commutative_fixed() {
            let a = Shape::from_fixed(&[4, 1, 3]).unwrap();
            let b = Shape::from_fixed(&[5, 3]).unwrap();
            assert_eq!(a.broadcast_with(&b).unwrap(), b.broadcast_with(&a).unwrap());
        }

        #[test]
        fn broadcast_both_scalars() {
            let a = Shape::scalar();
            let b = Shape::scalar();
            let c = a.broadcast_with(&b).unwrap();
            assert!(c.is_scalar());
        }

        #[test]
        fn broadcast_high_rank() {
            // [8, 1, 6, 1] broadcast with [7, 1, 5] => [8, 7, 6, 5]
            let a = Shape::from_fixed(&[8, 1, 6, 1]).unwrap();
            let b = Shape::from_fixed(&[7, 1, 5]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c, Shape::from_fixed(&[8, 7, 6, 5]).unwrap());
        }

        #[test]
        fn broadcast_symbolic_with_fixed_gt_one() {
            // Symbolic("n") vs Fixed(5): fixed wins.
            let a = Shape::new(vec![Dim::Symbolic("n".into())]).unwrap();
            let b = Shape::from_fixed(&[5]).unwrap();
            let c = a.broadcast_with(&b).unwrap();
            assert_eq!(c.dims(), &[Dim::Fixed(5)]);
        }
    }
}
