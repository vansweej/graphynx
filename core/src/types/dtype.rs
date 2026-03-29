use std::fmt;

use thiserror::Error;

// ── DTypeError ───────────────────────────────────────────────────────────────

/// Errors produced when constructing a [`DType`] through a safe constructor.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum DTypeError {
    /// A `Custom` label was empty. Labels must be non-empty so they can
    /// meaningfully identify a backend-specific type.
    #[error("Custom dtype label must not be empty")]
    EmptyCustomLabel,
}

// ── DType ────────────────────────────────────────────────────────────────────

/// Scalar element type for tensors and buffers.
///
/// Represents the data type of individual elements flowing through the
/// computation graph. Every edge in the graph carries a `DType` as part of
/// its type metadata, enabling compile-time validation and size calculations.
///
/// # Variants
///
/// - Boolean and unsigned integers: `Bool`, `U8`, `U16`, `U32`, `U64`
/// - Signed integers: `I8`, `I16`, `I32`, `I64`
/// - Floating point: `F16`, `BF16`, `F32`, `F64`
/// - Backend-specific escape hatch: `Custom(&'static str)`
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    /// Boolean (1 byte).
    Bool,

    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,

    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,

    /// IEEE 754 half-precision float (16-bit).
    F16,
    /// Brain floating point (16-bit, same exponent range as F32).
    BF16,
    /// IEEE 754 single-precision float (32-bit).
    F32,
    /// IEEE 754 double-precision float (64-bit).
    F64,

    /// Escape hatch for backend-specific types (e.g. quantised formats).
    ///
    /// The string label identifies the type within the backend that
    /// defines it. Size and alignment are unknown to the core layer.
    Custom(&'static str),
}

impl DType {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Construct a [`DType::Custom`] variant, rejecting empty labels.
    ///
    /// # Errors
    ///
    /// Returns [`DTypeError::EmptyCustomLabel`] if `label` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::custom("q4").is_ok());
    /// assert!(DType::custom("").is_err());
    /// ```
    pub fn custom(label: &'static str) -> Result<Self, DTypeError> {
        if label.is_empty() {
            Err(DTypeError::EmptyCustomLabel)
        } else {
            Ok(DType::Custom(label))
        }
    }

    // ── Size ─────────────────────────────────────────────────────────────

    /// Size of one element in bytes.
    ///
    /// Returns `None` for `Custom` types whose size is unknown to the
    /// core layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert_eq!(DType::F32.size_bytes(), Some(4));
    /// assert_eq!(DType::Bool.size_bytes(), Some(1));
    /// assert_eq!(DType::Custom("q4").size_bytes(), None);
    /// ```
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            DType::Bool | DType::U8 | DType::I8 => Some(1),
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => Some(2),
            DType::U32 | DType::I32 | DType::F32 => Some(4),
            DType::U64 | DType::I64 | DType::F64 => Some(8),
            DType::Custom(_) => None,
        }
    }

    // ── Alignment ────────────────────────────────────────────────────────

    /// Natural alignment of one element in bytes.
    ///
    /// For all standard types this equals `size_bytes()` (natural
    /// alignment). Returns `None` for `Custom` types.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert_eq!(DType::F64.alignment(), Some(8));
    /// assert_eq!(DType::F16.alignment(), Some(2));
    /// assert_eq!(DType::Custom("q8").alignment(), None);
    /// ```
    pub fn alignment(&self) -> Option<usize> {
        // Natural alignment equals type size for all standard scalar types.
        self.size_bytes()
    }

    // ── Human-readable name ──────────────────────────────────────────────

    /// Human-readable name for this data type.
    ///
    /// Returns lowercase names matching common ML framework conventions:
    /// `"bool"`, `"u8"`, `"f32"`, `"bf16"`, etc. For `Custom` types the
    /// backend-provided label is returned directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert_eq!(DType::F32.name(), "f32");
    /// assert_eq!(DType::BF16.name(), "bf16");
    /// assert_eq!(DType::Custom("q4_0").name(), "q4_0");
    /// ```
    pub fn name(&self) -> &str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::Custom(s) => s,
        }
    }

    // ── Category helpers ─────────────────────────────────────────────────

    /// Returns `true` if this is a floating-point type (F16, BF16, F32, F64).
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::F32.is_float());
    /// assert!(!DType::I32.is_float());
    /// ```
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Returns `true` if this is an integer type (signed or unsigned).
    ///
    /// Includes U8..U64 and I8..I64. Does *not* include `Bool`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::I32.is_int());
    /// assert!(DType::U8.is_int());
    /// assert!(!DType::Bool.is_int());
    /// ```
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
        )
    }

    /// Returns `true` if this is a signed type (I8..I64 or any float).
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::I32.is_signed());
    /// assert!(DType::F32.is_signed());
    /// assert!(!DType::U32.is_signed());
    /// ```
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::F16
                | DType::BF16
                | DType::F32
                | DType::F64
        )
    }

    /// Returns `true` if this is an unsigned type (Bool, U8..U64).
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::U32.is_unsigned());
    /// assert!(DType::Bool.is_unsigned());
    /// assert!(!DType::I32.is_unsigned());
    /// ```
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            DType::Bool | DType::U8 | DType::U16 | DType::U32 | DType::U64
        )
    }

    /// Returns `true` if this is a `Custom` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::dtype::DType;
    ///
    /// assert!(DType::Custom("q4").is_custom());
    /// assert!(!DType::F32.is_custom());
    /// ```
    pub fn is_custom(&self) -> bool {
        matches!(self, DType::Custom(_))
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for DType {
    /// Formats the data type using its human-readable name.
    ///
    /// Delegates to `name()` so that `format!("{}", DType::F32)` produces
    /// `"f32"` rather than the debug representation `"F32"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
