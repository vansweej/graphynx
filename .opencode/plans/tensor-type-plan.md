# TensorType Design Plan — Zero-Footgun Tensor Metadata

## Goal

Implement the `Dim`, `Layout`, and `TensorType` types from ARCHITECTURE.md as a
core-layer module with **no backend dependencies**, **no invalid states**, and
**ownership semantics that eliminate common footguns**.

The type describes tensor metadata flowing along graph edges — it does _not_ own
data.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Location | `src/tensor_type.rs` | Flat module structure, consistent with existing `dtype.rs` / `backend.rs` |
| Shape repr | `Vec<Dim>` (dynamic) | Runtime-checked, all ranks supported, validated at construction |
| Layout | Contiguous-only (RowMajor, ColMajor, NCHW, NHWC, Any) | No stride arithmetic initially; extensible later |
| Construction | Fully opaque — private fields, validated constructors + builder | Invalid states are unrepresentable |
| Device | `Option<DeviceId>` | `None` = unplaced, `Some` = placed; useful for graph planning |
| Compatibility | `is_compatible_with()` method | Graph builder can validate edges at build time |

---

## Types

### 1. `Dim`

```rust
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dim {
    /// Known at graph-build time. Must be > 0.
    Fixed(usize),
    /// Unknown until execution.
    Dynamic,
    /// Symbolic named dimension. Two dims with the same name must resolve
    /// to the same runtime value.
    Symbolic(String),
}
```

**Separation of `Dynamic` and `Symbolic`** instead of `Dynamic(Option<String>)`:

- Makes pattern matching clearer — no `Dynamic(None)` vs `Dynamic(Some(...))`.
- `Symbolic("batch")` is self-documenting at the call site.
- The ARCHITECTURE.md uses `Dynamic(Option<String>)`, but splitting it into two
  variants is a strictly better API for the user while remaining semantically
  equivalent.

**Key methods:**

| Method | Signature | Behavior |
|---|---|---|
| `fixed` | `fn fixed(n: usize) -> Result<Self, TensorTypeError>` | Rejects 0 |
| `is_fixed` | `fn is_fixed(&self) -> bool` | |
| `is_dynamic` | `fn is_dynamic(&self) -> bool` | True for both `Dynamic` and `Symbolic` |
| `is_symbolic` | `fn is_symbolic(&self) -> bool` | |
| `fixed_value` | `fn fixed_value(&self) -> Option<usize>` | Returns `Some(n)` for `Fixed(n)` |
| `symbol` | `fn symbol(&self) -> Option<&str>` | Returns the name for `Symbolic` |
| `is_compatible_with` | `fn is_compatible_with(&self, other: &Dim) -> bool` | See compatibility rules below |

**`Dim` compatibility rules:**

| LHS | RHS | Compatible? |
|---|---|---|
| `Fixed(a)` | `Fixed(b)` | `a == b` |
| `Fixed(_)` | `Dynamic` | Yes |
| `Fixed(_)` | `Symbolic(_)` | Yes |
| `Dynamic` | `Dynamic` | Yes |
| `Dynamic` | `Symbolic(_)` | Yes |
| `Symbolic(a)` | `Symbolic(b)` | `a == b` |

Symbolic-to-Symbolic requires name equality — two different symbolic names
represent potentially different runtime values and must not silently match.

### 2. `Layout`

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Layout {
    /// Row-major / C-contiguous (last dim varies fastest).
    RowMajor,
    /// Column-major / Fortran-contiguous (first dim varies fastest).
    ColMajor,
    /// Batch × Channels × Height × Width (PyTorch convention).
    NCHW,
    /// Batch × Height × Width × Channels (TensorFlow convention).
    NHWC,
    /// No specific layout constraint — compatible with all layouts.
    Any,
}
```

**Key methods:**

| Method | Signature | Behavior |
|---|---|---|
| `is_compatible_with` | `fn is_compatible_with(&self, other: &Layout) -> bool` | `Any` matches everything; otherwise strict equality |
| `is_image_layout` | `fn is_image_layout(&self) -> bool` | True for NCHW, NHWC |
| `expected_rank` | `fn expected_rank(&self) -> Option<usize>` | `NCHW`/`NHWC` → `Some(4)`, others → `None` |

### 3. `TensorTypeError`

```rust
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum TensorTypeError {
    #[error("Dimension must be > 0, got 0")]
    ZeroDimension,

    #[error("Scalar tensor (rank 0) cannot use layout {0}")]
    ScalarWithLayout(Layout),

    #[error("Layout {layout} requires rank {expected}, got {actual}")]
    LayoutRankMismatch {
        layout: Layout,
        expected: usize,
        actual: usize,
    },

    #[error("dim_names length ({names}) does not match shape length ({shape})")]
    DimNamesMismatch { names: usize, shape: usize },

    #[error("Empty symbolic dimension name")]
    EmptySymbol,

    #[error("Custom DType cannot compute element size")]
    CustomDTypeSize,
}
```

All construction errors are specific and actionable — the user knows exactly
what they did wrong and how to fix it.

### 4. `TensorType`

```rust
/// Complete description of a tensor flowing along a graph edge.
///
/// All fields are private. Construction goes through validated methods
/// (`new`, `scalar`, `vector`, `matrix`, `builder`) that enforce invariants.
/// Once constructed, a `TensorType` is always valid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorType {
    dtype:     DType,
    shape:     Vec<Dim>,
    layout:    Layout,
    dim_names: Option<Vec<String>>,
    device:    Option<DeviceId>,
}
```

**Invariants enforced at construction (never violated):**

1. No `Fixed(0)` dimensions in `shape`.
2. No `Symbolic("")` in `shape`.
3. If `dim_names` is `Some`, its length equals `shape.len()`.
4. If `layout` is `NCHW` or `NHWC`, rank must be 4.
5. Scalars (rank 0) must use `Layout::Any`.

---

## Construction API

### Direct constructors (common cases, zero ceremony)

```rust
impl TensorType {
    /// Fully specified constructor with validation.
    pub fn new(
        dtype: DType,
        shape: Vec<Dim>,
        layout: Layout,
    ) -> Result<Self, TensorTypeError>;

    /// Rank-0 tensor (scalar). Layout is always `Any`, no dim names.
    pub fn scalar(dtype: DType) -> Self;

    /// Rank-1 tensor with a single Fixed dimension. Layout is `RowMajor`.
    pub fn vector(dtype: DType, len: usize) -> Result<Self, TensorTypeError>;

    /// Rank-2 tensor [rows, cols]. Layout is `RowMajor`.
    pub fn matrix(
        dtype: DType,
        rows: usize,
        cols: usize,
    ) -> Result<Self, TensorTypeError>;
}
```

### Builder (complex cases with optional fields)

```rust
pub struct TensorTypeBuilder { ... }

impl TensorTypeBuilder {
    pub fn new(dtype: DType) -> Self;
    pub fn shape(self, shape: Vec<Dim>) -> Self;
    pub fn layout(self, layout: Layout) -> Self;
    pub fn dim_names(self, names: Vec<String>) -> Self;
    pub fn device(self, device: DeviceId) -> Self;
    pub fn build(self) -> Result<TensorType, TensorTypeError>;
}

// Entry point on TensorType:
impl TensorType {
    pub fn builder(dtype: DType) -> TensorTypeBuilder;
}
```

The builder defaults to `shape: vec![]` (scalar), `layout: Layout::Any`,
`dim_names: None`, `device: None`. Validation runs in `build()`.

---

## Accessor API (read-only, no mutation footguns)

```rust
impl TensorType {
    // ── Core properties ─────────────────────────────────────────────
    pub fn dtype(&self) -> DType;
    pub fn shape(&self) -> &[Dim];
    pub fn layout(&self) -> Layout;
    pub fn dim_names(&self) -> Option<&[String]>;
    pub fn device(&self) -> Option<&DeviceId>;

    // ── Derived properties ──────────────────────────────────────────
    pub fn rank(&self) -> usize;
    pub fn is_scalar(&self) -> bool;

    /// Total number of elements, if all dims are Fixed.
    /// Returns None if any dim is Dynamic or Symbolic.
    pub fn num_elements(&self) -> Option<usize>;

    /// Total size in bytes, if all dims are Fixed and dtype is not Custom.
    /// Returns None otherwise.
    pub fn size_bytes(&self) -> Option<usize>;

    // ── Transforms (return new TensorType, never mutate) ────────────
    /// Return a new TensorType with different layout.
    /// Validates layout against current rank.
    pub fn with_layout(self, layout: Layout) -> Result<Self, TensorTypeError>;

    /// Return a new TensorType placed on a device.
    pub fn with_device(self, device: DeviceId) -> Self;

    /// Return a new TensorType with device cleared.
    pub fn unplaced(self) -> Self;

    /// Return a new TensorType with dim names.
    pub fn with_dim_names(self, names: Vec<String>) -> Result<Self, TensorTypeError>;

    // ── Compatibility ───────────────────────────────────────────────
    /// Check if this TensorType is compatible with `other` for a graph edge.
    ///
    /// Rules:
    /// - DTypes must be equal
    /// - Ranks must be equal
    /// - Each dim pair must be compatible (see Dim::is_compatible_with)
    /// - Layouts must be compatible (see Layout::is_compatible_with)
    /// - Device is NOT checked (placement is the scheduler's job)
    pub fn is_compatible_with(&self, other: &TensorType) -> bool;
}
```

**Why no mutation methods?** Every `with_*` method consumes `self` and returns a
new value. This is the same pattern as `PathBuf::with_extension` and
`String::replace`. It eliminates:

- Accidentally mutating a shared `TensorType` reference.
- Leaving an object in a partially-updated invalid state.
- The need for `&mut self` methods that require exclusive ownership.

---

## Ownership Design — Zero Footguns

### What we avoid

| Footgun | How we prevent it |
|---|---|
| Invalid `Fixed(0)` dimension | Validated at construction; `Dim::fixed()` returns `Result` |
| Mismatched `dim_names` length | Validated at construction |
| Wrong rank for NCHW/NHWC layout | Validated at construction |
| Scalar with non-Any layout | Validated at construction |
| Empty symbolic name `Symbolic("")` | Validated at construction |
| Mutating shared TensorType | All fields private; transforms consume + return |
| Partial update leaving invalid state | No setter methods; transforms re-validate |
| Silent type mismatch on graph edges | `is_compatible_with()` with clear rules |
| Accidental clone where borrow suffices | `Clone` is derived but accessors return `&` references |
| Confusing `Dynamic(None)` vs `Dynamic(Some(""))` | Separate `Dynamic` and `Symbolic` variants |

### Clone semantics

`TensorType` derives `Clone`. This is intentional: tensor metadata is small
(a DType + a few Dims + an enum + optional names) and is expected to be cloned
when building graph nodes/edges. The cost is negligible.

### Display

`TensorType` implements `Display` with a clear format:

```
f32[batch, 3, 224, 224] NCHW @ cuda:0
f32[] Any                              // scalar
i32[1024] RowMajor                     // vector, no device
f64[?, 256] RowMajor                   // Dynamic dim shown as ?
```

`Dim` implements `Display`: `Fixed(n)` → `"n"`, `Dynamic` → `"?"`,
`Symbolic(s)` → `"s"`.

`Layout` implements `Display`: variant name as-is.

---

## File Structure

### New file: `src/tensor_type.rs`

```
src/tensor_type.rs
├── TensorTypeError (enum, thiserror)
├── Dim (enum + methods + Display)
├── Layout (enum + methods + Display)
├── TensorType (struct + validated constructors + accessors + transforms)
├── TensorTypeBuilder (struct + fluent builder)
└── mod tests
    ├── dim_tests (Fixed/Dynamic/Symbolic construction, compatibility matrix)
    ├── layout_tests (compatibility, expected_rank)
    ├── tensor_type_construction_tests (new, scalar, vector, matrix, builder)
    ├── tensor_type_validation_tests (all error cases)
    ├── tensor_type_accessor_tests (all getters, derived properties)
    ├── tensor_type_transform_tests (with_layout, with_device, with_dim_names)
    ├── tensor_type_compatibility_tests (full compatibility matrix)
    └── display_tests (Dim, Layout, TensorType formatting)
```

### Modified file: `src/lib.rs`

Add `pub mod tensor_type;` to the module declarations.

---

## Dependencies

**None new.** Only uses:
- `std::fmt` (Display)
- `thiserror` (already in Cargo.toml)
- `crate::dtype::DType` (existing)
- `crate::backend::DeviceId` (existing)

This keeps the module at the **core layer** with zero backend dependencies, as
required by the architecture.

---

## Test Plan

Target: **≥90% line coverage** via `cargo tarpaulin`.

### `Dim` tests (~20 tests)

- `Dim::fixed(0)` returns `Err(ZeroDimension)`
- `Dim::fixed(1)` through large values succeed
- `Dynamic` and `Symbolic("batch")` construction
- `Symbolic("")` is caught (either at construction or via TensorType validation)
- `is_fixed`, `is_dynamic`, `is_symbolic` truth table
- `fixed_value()` returns `Some`/`None` correctly
- `symbol()` returns `Some`/`None` correctly
- Full compatibility matrix (6×6 = 36 cases, grouped)
- `Clone`, `Debug`, `PartialEq`, `Hash` derive verification
- `Display` formatting for all three variants

### `Layout` tests (~12 tests)

- All 5 variants construct correctly
- `is_compatible_with` matrix: `Any` matches everything, others match self only
- `expected_rank()`: NCHW/NHWC → 4, others → None
- `is_image_layout()` truth table
- `Copy`, `Clone`, `Debug`, `PartialEq`, `Hash` derive verification
- `Display` formatting

### `TensorTypeError` tests (~6 tests)

- Each variant's Display message is correct and actionable
- `Clone`, `Debug`, `PartialEq` work

### `TensorType` construction tests (~20 tests)

- `scalar(F32)` produces rank-0, Layout::Any, no dim_names
- `vector(F32, 1024)` produces rank-1, RowMajor
- `vector(F32, 0)` returns `Err(ZeroDimension)`
- `matrix(F32, 3, 3)` produces rank-2, RowMajor
- `matrix(F32, 0, 5)` returns error
- `new(F32, [Fixed(3), Fixed(224), Fixed(224)], NCHW)` returns `LayoutRankMismatch` (rank 3 ≠ 4)
- `new(F32, [Fixed(1), Fixed(3), Fixed(224), Fixed(224)], NCHW)` succeeds
- `new(F32, [], RowMajor)` returns `ScalarWithLayout` (scalar must be Any)
- Builder: dtype-only builds scalar
- Builder: full specification with dim_names and device
- Builder: dim_names length mismatch
- Builder: multiple validation errors (first one wins)

### `TensorType` accessor tests (~10 tests)

- `dtype()`, `shape()`, `layout()`, `dim_names()`, `device()` return correct values
- `rank()` matches shape length
- `is_scalar()` for rank 0 vs rank > 0
- `num_elements()` with all-Fixed, with some Dynamic, with empty shape
- `size_bytes()` with known dtype and all Fixed dims
- `size_bytes()` returns None for Custom dtype

### `TensorType` transform tests (~8 tests)

- `with_layout(NCHW)` on rank-4 tensor succeeds
- `with_layout(NCHW)` on rank-3 tensor fails
- `with_device(DeviceId::new("cuda:0"))` sets device
- `unplaced()` clears device
- `with_dim_names()` with correct length succeeds
- `with_dim_names()` with wrong length fails
- Chaining: `scalar(F32).with_device(...).unplaced()` round-trips

### `TensorType` compatibility tests (~15 tests)

- Identical TensorTypes are compatible
- Different dtypes are incompatible
- Different ranks are incompatible
- Fixed(3) vs Fixed(3) — compatible
- Fixed(3) vs Fixed(4) — incompatible
- Fixed(3) vs Dynamic — compatible
- Fixed(3) vs Symbolic("n") — compatible
- Dynamic vs Dynamic — compatible
- Symbolic("a") vs Symbolic("a") — compatible
- Symbolic("a") vs Symbolic("b") — incompatible
- Layout::Any vs Layout::RowMajor — compatible
- Layout::RowMajor vs Layout::ColMajor — incompatible
- Device is ignored in compatibility checks

### `Display` tests (~5 tests)

- Scalar: `"f32[] Any"`
- Vector: `"i32[1024] RowMajor"`
- With device: `"f32[batch, 3, 224, 224] NCHW @ cuda:0"`
- Dynamic dims: `"f64[?, 256] RowMajor"`
- Mixed: `"f32[batch, ?, 3] Any"`

**Estimated total: ~96 tests, covering all branches.**

---

## Implementation Order

1. **Branch**: Create feature branch `feat/tensor-type` from `main`
2. **`TensorTypeError`**: Define the error enum with `thiserror`
3. **`Dim`**: Enum + methods + `Display` + tests
4. **`Layout`**: Enum + methods + `Display` + tests
5. **`TensorType`**: Struct + private fields + validated `new`/`scalar`/`vector`/`matrix` + tests
6. **`TensorTypeBuilder`**: Builder struct + `build()` validation + tests
7. **Accessors**: `dtype()`, `shape()`, etc. + derived properties + tests
8. **Transforms**: `with_layout`, `with_device`, `unplaced`, `with_dim_names` + tests
9. **Compatibility**: `is_compatible_with` on all three types + tests
10. **Display**: `TensorType` Display impl + tests
11. **Wire up**: Add `pub mod tensor_type;` to `lib.rs`
12. **Quality gate**: `cargo fmt`, `cargo clippy`, `cargo test`, `cargo tarpaulin`

---

## Future Extensions (not in this PR)

- **Stride support**: Add `strides: Option<Vec<usize>>` for non-contiguous views
- **Broadcasting rules**: `broadcast_shape(&TensorType, &TensorType) -> Result<TensorType>`
- **Op signature validation**: Each `MlOp` declares expected input/output `TensorType` patterns
- **Named dimension tracking**: Graph-wide resolution of `Symbolic` dims to concrete values
- **Serialization**: `serde` support behind a feature gate
