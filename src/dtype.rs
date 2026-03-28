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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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
    /// use graphynx::dtype::DType;
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

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// All concrete (non-Custom) variants for exhaustive testing.
    const ALL_CONCRETE: [DType; 13] = [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];

    // ── custom constructor ──────────────────────────────────────────────

    #[test]
    fn custom_nonempty_is_ok() {
        assert_eq!(DType::custom("q4"), Ok(DType::Custom("q4")));
    }

    #[test]
    fn custom_empty_is_error() {
        assert_eq!(DType::custom(""), Err(DTypeError::EmptyCustomLabel));
    }

    #[test]
    fn custom_constructor_returns_custom_variant() {
        let dt = DType::custom("fp8_e4m3").unwrap();
        assert!(dt.is_custom());
        assert_eq!(dt.name(), "fp8_e4m3");
    }

    // ── DTypeError ──────────────────────────────────────────────────────

    #[test]
    fn dtype_error_display() {
        let err = DTypeError::EmptyCustomLabel;
        assert!(err.to_string().to_lowercase().contains("empty"));
    }

    #[test]
    fn dtype_error_clone_eq() {
        let err = DTypeError::EmptyCustomLabel;
        assert_eq!(err.clone(), err);
    }

    // ── size_bytes ───────────────────────────────────────────────────────

    #[test]
    fn size_bytes_one_byte_types() {
        assert_eq!(DType::Bool.size_bytes(), Some(1));
        assert_eq!(DType::U8.size_bytes(), Some(1));
        assert_eq!(DType::I8.size_bytes(), Some(1));
    }

    #[test]
    fn size_bytes_two_byte_types() {
        assert_eq!(DType::U16.size_bytes(), Some(2));
        assert_eq!(DType::I16.size_bytes(), Some(2));
        assert_eq!(DType::F16.size_bytes(), Some(2));
        assert_eq!(DType::BF16.size_bytes(), Some(2));
    }

    #[test]
    fn size_bytes_four_byte_types() {
        assert_eq!(DType::U32.size_bytes(), Some(4));
        assert_eq!(DType::I32.size_bytes(), Some(4));
        assert_eq!(DType::F32.size_bytes(), Some(4));
    }

    #[test]
    fn size_bytes_eight_byte_types() {
        assert_eq!(DType::U64.size_bytes(), Some(8));
        assert_eq!(DType::I64.size_bytes(), Some(8));
        assert_eq!(DType::F64.size_bytes(), Some(8));
    }

    #[test]
    fn size_bytes_custom_returns_none() {
        assert_eq!(DType::Custom("q4_0").size_bytes(), None);
        assert_eq!(DType::Custom("int4").size_bytes(), None);
    }

    #[test]
    fn size_bytes_all_concrete_are_some() {
        for dt in &ALL_CONCRETE {
            assert!(
                dt.size_bytes().is_some(),
                "{:?} should have a known size",
                dt
            );
        }
    }

    // ── alignment ────────────────────────────────────────────────────────

    #[test]
    fn alignment_equals_size_for_concrete_types() {
        for dt in &ALL_CONCRETE {
            assert_eq!(
                dt.alignment(),
                dt.size_bytes(),
                "{:?} alignment should equal size_bytes",
                dt
            );
        }
    }

    #[test]
    fn alignment_custom_returns_none() {
        assert_eq!(DType::Custom("fp8").alignment(), None);
    }

    #[test]
    fn alignment_specific_values() {
        assert_eq!(DType::F16.alignment(), Some(2));
        assert_eq!(DType::BF16.alignment(), Some(2));
        assert_eq!(DType::F32.alignment(), Some(4));
        assert_eq!(DType::F64.alignment(), Some(8));
        assert_eq!(DType::U8.alignment(), Some(1));
    }

    // ── name ─────────────────────────────────────────────────────────────

    #[test]
    fn name_returns_expected_strings() {
        let expected = [
            (DType::Bool, "bool"),
            (DType::U8, "u8"),
            (DType::U16, "u16"),
            (DType::U32, "u32"),
            (DType::U64, "u64"),
            (DType::I8, "i8"),
            (DType::I16, "i16"),
            (DType::I32, "i32"),
            (DType::I64, "i64"),
            (DType::F16, "f16"),
            (DType::BF16, "bf16"),
            (DType::F32, "f32"),
            (DType::F64, "f64"),
        ];
        for (dt, name) in &expected {
            assert_eq!(dt.name(), *name, "{:?} name mismatch", dt);
        }
    }

    #[test]
    fn name_custom_returns_label() {
        assert_eq!(DType::Custom("q4_0").name(), "q4_0");
        assert_eq!(DType::Custom("fp8_e4m3").name(), "fp8_e4m3");
    }

    // ── Display ──────────────────────────────────────────────────────────

    #[test]
    fn display_matches_name() {
        for dt in &ALL_CONCRETE {
            assert_eq!(format!("{}", dt), dt.name(), "{:?} Display mismatch", dt);
        }
    }

    #[test]
    fn display_custom() {
        let dt = DType::Custom("q8_1");
        assert_eq!(format!("{}", dt), "q8_1");
    }

    // ── is_float ─────────────────────────────────────────────────────────

    #[test]
    fn is_float_true_for_floats() {
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
    }

    #[test]
    fn is_float_false_for_non_floats() {
        assert!(!DType::Bool.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::U16.is_float());
        assert!(!DType::U32.is_float());
        assert!(!DType::U64.is_float());
        assert!(!DType::I8.is_float());
        assert!(!DType::I16.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
        assert!(!DType::Custom("float_custom").is_float());
    }

    // ── is_int ───────────────────────────────────────────────────────────

    #[test]
    fn is_int_true_for_integers() {
        assert!(DType::U8.is_int());
        assert!(DType::U16.is_int());
        assert!(DType::U32.is_int());
        assert!(DType::U64.is_int());
        assert!(DType::I8.is_int());
        assert!(DType::I16.is_int());
        assert!(DType::I32.is_int());
        assert!(DType::I64.is_int());
    }

    #[test]
    fn is_int_false_for_non_integers() {
        assert!(!DType::Bool.is_int());
        assert!(!DType::F16.is_int());
        assert!(!DType::BF16.is_int());
        assert!(!DType::F32.is_int());
        assert!(!DType::F64.is_int());
        assert!(!DType::Custom("int_custom").is_int());
    }

    // ── is_signed ────────────────────────────────────────────────────────

    #[test]
    fn is_signed_true_for_signed_types() {
        assert!(DType::I8.is_signed());
        assert!(DType::I16.is_signed());
        assert!(DType::I32.is_signed());
        assert!(DType::I64.is_signed());
        assert!(DType::F16.is_signed());
        assert!(DType::BF16.is_signed());
        assert!(DType::F32.is_signed());
        assert!(DType::F64.is_signed());
    }

    #[test]
    fn is_signed_false_for_unsigned_types() {
        assert!(!DType::Bool.is_signed());
        assert!(!DType::U8.is_signed());
        assert!(!DType::U16.is_signed());
        assert!(!DType::U32.is_signed());
        assert!(!DType::U64.is_signed());
        assert!(!DType::Custom("signed_custom").is_signed());
    }

    // ── is_unsigned ──────────────────────────────────────────────────────

    #[test]
    fn is_unsigned_true_for_unsigned_types() {
        assert!(DType::Bool.is_unsigned());
        assert!(DType::U8.is_unsigned());
        assert!(DType::U16.is_unsigned());
        assert!(DType::U32.is_unsigned());
        assert!(DType::U64.is_unsigned());
    }

    #[test]
    fn is_unsigned_false_for_signed_types() {
        assert!(!DType::I8.is_unsigned());
        assert!(!DType::I16.is_unsigned());
        assert!(!DType::I32.is_unsigned());
        assert!(!DType::I64.is_unsigned());
        assert!(!DType::F16.is_unsigned());
        assert!(!DType::BF16.is_unsigned());
        assert!(!DType::F32.is_unsigned());
        assert!(!DType::F64.is_unsigned());
        assert!(!DType::Custom("unsigned_custom").is_unsigned());
    }

    // ── is_custom ────────────────────────────────────────────────────────

    #[test]
    fn is_custom_true_for_custom() {
        assert!(DType::Custom("q4").is_custom());
        assert!(DType::Custom("").is_custom());
    }

    #[test]
    fn is_custom_false_for_concrete() {
        for dt in &ALL_CONCRETE {
            assert!(!dt.is_custom(), "{:?} should not be custom", dt);
        }
    }

    // ── Hash (works as HashMap key) ──────────────────────────────────────

    #[test]
    fn hash_usable_as_map_key() {
        let mut map: HashMap<DType, &str> = HashMap::new();
        map.insert(DType::F32, "single precision");
        map.insert(DType::F64, "double precision");
        map.insert(DType::Custom("q4"), "quantised");

        assert_eq!(map.get(&DType::F32), Some(&"single precision"));
        assert_eq!(map.get(&DType::F64), Some(&"double precision"));
        assert_eq!(map.get(&DType::Custom("q4")), Some(&"quantised"));
        assert_eq!(map.get(&DType::I32), None);
    }

    #[test]
    fn hash_distinct_custom_labels_are_distinct_keys() {
        let mut map: HashMap<DType, u32> = HashMap::new();
        map.insert(DType::Custom("q4"), 1);
        map.insert(DType::Custom("q8"), 2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&DType::Custom("q4")), Some(&1));
        assert_eq!(map.get(&DType::Custom("q8")), Some(&2));
    }

    // ── PartialEq ────────────────────────────────────────────────────────

    #[test]
    fn eq_same_variants() {
        for dt in &ALL_CONCRETE {
            assert_eq!(*dt, *dt);
        }
        assert_eq!(DType::Custom("q4"), DType::Custom("q4"));
    }

    #[test]
    fn ne_different_variants() {
        assert_ne!(DType::F32, DType::F64);
        assert_ne!(DType::I32, DType::U32);
        assert_ne!(DType::Custom("q4"), DType::Custom("q8"));
        assert_ne!(DType::Custom("f32"), DType::F32);
    }

    // ── Copy semantics ───────────────────────────────────────────────────

    #[test]
    fn copy_semantics() {
        let a = DType::F32;
        let b = a; // Copy, not move
        assert_eq!(a, b); // `a` is still usable
    }

    #[test]
    fn clone_equals_original() {
        for dt in &ALL_CONCRETE {
            assert_eq!(*dt, dt.clone());
        }
        let custom = DType::Custom("q4");
        assert_eq!(custom, custom.clone());
    }

    // ── Category mutual exclusion ────────────────────────────────────────

    #[test]
    fn float_and_int_are_mutually_exclusive() {
        for dt in &ALL_CONCRETE {
            assert!(
                !(dt.is_float() && dt.is_int()),
                "{:?} is both float and int",
                dt
            );
        }
    }

    #[test]
    fn signed_and_unsigned_are_mutually_exclusive() {
        for dt in &ALL_CONCRETE {
            assert!(
                !(dt.is_signed() && dt.is_unsigned()),
                "{:?} is both signed and unsigned",
                dt
            );
        }
    }

    #[test]
    fn every_concrete_type_is_categorised() {
        // Every concrete type is either float, int, or bool.
        for dt in &ALL_CONCRETE {
            let categorised = dt.is_float() || dt.is_int() || matches!(dt, DType::Bool);
            assert!(categorised, "{:?} has no category", dt);
        }
    }

    #[test]
    fn every_concrete_type_has_signedness() {
        // Every concrete type is either signed or unsigned.
        for dt in &ALL_CONCRETE {
            assert!(
                dt.is_signed() || dt.is_unsigned(),
                "{:?} has no signedness",
                dt
            );
        }
    }

    // ── Custom has no category ───────────────────────────────────────────

    #[test]
    fn custom_has_no_category_flags() {
        let dt = DType::Custom("mystery");
        assert!(!dt.is_float());
        assert!(!dt.is_int());
        assert!(!dt.is_signed());
        assert!(!dt.is_unsigned());
        assert!(dt.is_custom());
    }

    // ── Debug ────────────────────────────────────────────────────────────

    #[test]
    fn debug_format_is_distinct_from_display() {
        // Debug uses the enum variant name, Display uses the human-readable name.
        let dt = DType::F32;
        let debug = format!("{:?}", dt);
        let display = format!("{}", dt);
        assert_eq!(debug, "F32");
        assert_eq!(display, "f32");
    }
}
