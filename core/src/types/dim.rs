use std::fmt;

use thiserror::Error;

// ── DimError ─────────────────────────────────────────────────────────────────

/// Errors produced when constructing a [`Dim`] through a safe constructor.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum DimError {
    /// A `Fixed` dimension value of zero was supplied. All fixed dimensions
    /// must be ≥ 1.
    #[error("Dimension must be > 0, got 0")]
    ZeroDimension,

    /// A `Symbolic` dimension was given an empty name. Symbolic names must be
    /// non-empty so they can be matched across graph edges.
    #[error("Symbolic dimension name must not be empty")]
    EmptySymbol,
}

// ── Dim ───────────────────────────────────────────────────────────────────────

/// A single tensor dimension — either a known constant or a runtime value.
///
/// Splitting the "unknown" case into [`Dim::Dynamic`] and [`Dim::Symbolic`]
/// avoids the footgun of `Dynamic(Option<String>)`, where `None` and `Some("")`
/// are both valid spellings of subtly different things.
///
/// # Compatibility
///
/// Two dims are compatible for a graph edge if the values they could represent
/// at runtime are always the same. See [`Dim::is_compatible_with`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dim {
    /// Known at graph-build time. Always ≥ 1 when constructed through
    /// [`Dim::fixed`]; the bare variant can still be created directly for
    /// tests or internal use, but `TensorType` constructors validate it.
    Fixed(usize),

    /// Unknown until execution. Matches any other dimension value.
    Dynamic,

    /// A named unknown dimension. All `Symbolic` dims with the same name in a
    /// graph must resolve to the same runtime value, enabling consistency
    /// checks (e.g. every `"batch"` dim must be the same integer at runtime).
    Symbolic(String),
}

impl Dim {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Construct a [`Dim::Fixed`] dimension, rejecting zero.
    ///
    /// # Errors
    ///
    /// Returns [`DimError::ZeroDimension`] if `n == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dim::Dim;
    ///
    /// assert!(Dim::fixed(3).is_ok());
    /// assert!(Dim::fixed(0).is_err());
    /// ```
    pub fn fixed(n: usize) -> Result<Self, DimError> {
        if n == 0 {
            Err(DimError::ZeroDimension)
        } else {
            Ok(Dim::Fixed(n))
        }
    }

    /// Construct a [`Dim::Symbolic`] dimension, rejecting empty names.
    ///
    /// # Errors
    ///
    /// Returns [`DimError::EmptySymbol`] if `name` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dim::Dim;
    ///
    /// assert!(Dim::symbolic("batch").is_ok());
    /// assert!(Dim::symbolic("").is_err());
    /// ```
    pub fn symbolic(name: impl Into<String>) -> Result<Self, DimError> {
        let s = name.into();
        if s.is_empty() {
            Err(DimError::EmptySymbol)
        } else {
            Ok(Dim::Symbolic(s))
        }
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Returns `true` if this is a [`Dim::Fixed`] variant.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    /// Returns `true` if this is [`Dim::Dynamic`] or [`Dim::Symbolic`].
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Dim::Dynamic | Dim::Symbolic(_))
    }

    /// Returns `true` if this is a [`Dim::Symbolic`] variant.
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Dim::Symbolic(_))
    }

    /// Returns the fixed value if this is [`Dim::Fixed`], otherwise `None`.
    pub fn fixed_value(&self) -> Option<usize> {
        match self {
            Dim::Fixed(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the symbolic name if this is [`Dim::Symbolic`], otherwise `None`.
    pub fn symbol(&self) -> Option<&str> {
        match self {
            Dim::Symbolic(s) => Some(s.as_str()),
            _ => None,
        }
    }

    // ── Compatibility ────────────────────────────────────────────────────

    /// Returns `true` if `self` and `other` are compatible for a graph edge.
    ///
    /// Compatibility is symmetric. The rules are:
    ///
    /// | LHS | RHS | Result |
    /// |---|---|---|
    /// | `Fixed(a)` | `Fixed(b)` | `a == b` |
    /// | `Fixed(_)` | `Dynamic` | `true` |
    /// | `Fixed(_)` | `Symbolic(_)` | `true` |
    /// | `Dynamic` | `Dynamic` | `true` |
    /// | `Dynamic` | `Symbolic(_)` | `true` |
    /// | `Symbolic(a)` | `Symbolic(b)` | `a == b` |
    ///
    /// Two `Symbolic` dims with **different** names are incompatible: they may
    /// resolve to different values at runtime.
    pub fn is_compatible_with(&self, other: &Dim) -> bool {
        match (self, other) {
            // Fixed vs Fixed: both must be the same value.
            (Dim::Fixed(a), Dim::Fixed(b)) => a == b,
            // Fixed is always compatible with Dynamic or Symbolic (unknown dims
            // can match any concrete size).
            (Dim::Fixed(_), Dim::Dynamic | Dim::Symbolic(_)) => true,
            (Dim::Dynamic | Dim::Symbolic(_), Dim::Fixed(_)) => true,
            // Two unnamed unknowns are always compatible.
            (Dim::Dynamic, Dim::Dynamic) => true,
            // An unnamed unknown is compatible with a named unknown.
            (Dim::Dynamic, Dim::Symbolic(_)) | (Dim::Symbolic(_), Dim::Dynamic) => true,
            // Two named unknowns must share the same name.
            (Dim::Symbolic(a), Dim::Symbolic(b)) => a == b,
        }
    }
}

impl fmt::Display for Dim {
    /// Formats the dimension as its value (`"3"`), `"?"` for dynamic, or the
    /// symbolic name (`"batch"`) for symbolic dims.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Dynamic => f.write_str("?"),
            Dim::Symbolic(s) => f.write_str(s),
        }
    }
}
