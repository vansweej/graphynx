# ML Op Catalog

The `ml_op` module (`src/ml_op.rs`) provides a curated catalog of primitive ML operations and their parameter structs. It has no GPU SDK dependencies and is 100% safe Rust.

## Overview

`MlOp` is the vocabulary the engine uses to describe computation at nodes. Multiple backends can implement the same `MlOp`, allowing the executor to route an operation to whichever backend targets the node's device.

```mermaid
graph LR
    MlOp --> LinearAlgebra["Linear algebra<br/>MatMul · Linear"]
    MlOp --> Convolution["Convolution<br/>Conv2d"]
    MlOp --> Activation["Activation<br/>Relu · Sigmoid · Tanh · Gelu · Softmax"]
    MlOp --> Normalisation["Normalisation<br/>BatchNorm · LayerNorm"]
    MlOp --> Pooling["Pooling<br/>MaxPool2d · AvgPool2d"]
    MlOp --> Shape["Shape<br/>Reshape · Transpose · Concat · Flatten"]
    MlOp --> Regularisation["Regularisation<br/>Dropout"]
    MlOp --> ElementWise["Element-wise<br/>Add · Mul"]
    MlOp --> EscapeHatch["Escape hatch<br/>Custom { name, params }"]
```

## `MlOpError`

Errors produced when constructing ML operation parameters through safe constructors. Every variant carries enough context for the caller to understand exactly what invariant was violated.

```mermaid
graph TD
    MlOpError --> ZeroSpatialParam["ZeroSpatialParam { param }<br/>kernel_size, stride, or dilation was 0"]
    MlOpError --> ZeroGroups["ZeroGroups<br/>groups was 0"]
    MlOpError --> ZeroFeatures["ZeroFeatures { param }<br/>in_features or out_features was 0"]
    MlOpError --> ZeroNumFeatures["ZeroNumFeatures<br/>num_features was 0"]
    MlOpError --> NonPositiveEps["NonPositiveEps(f64)<br/>eps was ≤ 0"]
    MlOpError --> InvalidMomentum["InvalidMomentum(f64)<br/>momentum outside [0.0, 1.0)"]
    MlOpError --> InvalidDropoutP["InvalidDropoutP(f64)<br/>p outside [0.0, 1.0)"]
    MlOpError --> InvalidNormalizedShape["InvalidNormalizedShape<br/>empty or contains zeros"]
    MlOpError --> InvalidPermutation["InvalidPermutation { perm, expected_len }<br/>not a permutation of 0..rank"]
    MlOpError --> EmptyCustomName["EmptyCustomName<br/>Custom op name was empty"]
```

Derives: `Debug`, `Error`, `Clone`, `PartialEq`.

## `MlOp` Enum

### Linear algebra

| Variant | Params struct | Description |
|---|---|---|
| `MatMul(MatMulParams)` | `transpose_a`, `transpose_b` | `C = op(A) · op(B)` |
| `Linear(LinearParams)` | `in_features`, `out_features`, `bias` | Fully-connected layer `y = x W^T + b` |

### Convolution

| Variant | Params struct | Description |
|---|---|---|
| `Conv2d(Conv2dParams)` | `kernel_size`, `stride`, `padding`, `dilation`, `groups` | 2-D spatial convolution |

### Activation

| Variant | Params struct | Description |
|---|---|---|
| `Relu` | — | `max(0, x)` |
| `Sigmoid` | — | `1 / (1 + exp(-x))` |
| `Tanh` | — | Hyperbolic tangent |
| `Gelu` | — | Gaussian error linear unit |
| `Softmax(SoftmaxParams)` | `axis` | Softmax along an axis |

### Normalisation

| Variant | Params struct | Description |
|---|---|---|
| `BatchNorm(BatchNormParams)` | `num_features`, `eps`, `momentum` | Batch normalisation |
| `LayerNorm(LayerNormParams)` | `normalized_shape`, `eps` | Layer normalisation |

### Pooling

| Variant | Params struct | Description |
|---|---|---|
| `MaxPool2d(PoolParams)` | `kernel_size`, `stride`, `padding` | 2-D max pooling |
| `AvgPool2d(PoolParams)` | `kernel_size`, `stride`, `padding` | 2-D average pooling |

### Shape manipulation

| Variant | Params struct | Description |
|---|---|---|
| `Reshape(ReshapeParams)` | `target_shape: Shape` | Change shape, preserve element count |
| `Transpose(TransposeParams)` | `perm: Vec<usize>` | Permute axes |
| `Concat(ConcatParams)` | `axis: i32` | Concatenate tensors along an axis |
| `Flatten(FlattenParams)` | `start_dim`, `end_dim` | Flatten a range of axes into one |

### Regularisation

| Variant | Params struct | Description |
|---|---|---|
| `Dropout(DropoutParams)` | `p: f64` | Zero elements with probability `p` during training |

### Element-wise arithmetic

| Variant | Params struct | Description |
|---|---|---|
| `Add` | — | Element-wise addition |
| `Mul` | — | Element-wise multiplication |

### Escape hatch

| Variant | Fields | Description |
|---|---|---|
| `Custom { name, params }` | `name: String`, `params: Vec<u8>` | Any operation not in the catalog |

`name` is a backend-interpreted identifier. `params` carries serialised operation parameters in any format the backend expects (JSON, protobuf, raw bytes, etc.).

Use `MlOp::custom(name, params)` for validated construction — it rejects empty names. Direct construction via `MlOp::Custom { name, params }` is also possible but does not validate.

## Query Methods

| Method | Returns | Description |
|---|---|---|
| `name()` | `&str` | Human-readable operation name. For `Custom`, returns the user-supplied `name`. |
| `is_parameterless()` | `bool` | `true` for `Relu`, `Sigmoid`, `Tanh`, `Gelu`, `Add`, `Mul`. |
| `is_custom()` | `bool` | `true` for `Custom { .. }`. |
| `is_spatial_2d()` | `bool` | `true` for `Conv2d`, `MaxPool2d`, `AvgPool2d` — operations that require a 4-D input. |

## `Display`

`MlOp` formats as its name string:

```
Relu         → "Relu"
Conv2d(…)    → "Conv2d"
Custom{…}    → "<user-supplied name>"
```

## Param Structs Reference

All param structs with non-trivial invariants provide a `new()` safe constructor that returns `Result<Self, MlOpError>`. Direct struct construction is also possible but bypasses validation.

### `Conv2dParams`

```rust
pub struct Conv2dParams {
    pub kernel_size: [usize; 2],   // [kh, kw]
    pub stride:      [usize; 2],   // [sh, sw]
    pub padding:     [usize; 2],   // [ph, pw]
    pub dilation:    [usize; 2],   // [dh, dw]
    pub groups:      usize,
}
```

All spatial parameters are `[height, width]` ordered. `groups = 1` is a standard convolution; `groups = in_channels` gives a depth-wise convolution.

**`Conv2dParams::new(kernel_size, stride, padding, dilation, groups)`** — Returns `Err` if `kernel_size`, `stride`, or `dilation` contain zeros, or if `groups == 0`.

### `MatMulParams`

```rust
pub struct MatMulParams {
    pub transpose_a: bool,
    pub transpose_b: bool,
}
```

**`MatMulParams::new(transpose_a, transpose_b)`** — Infallible constructor.

### `LinearParams`

```rust
pub struct LinearParams {
    pub in_features:  usize,
    pub out_features: usize,
    pub bias:         bool,
}
```

**`LinearParams::new(in_features, out_features, bias)`** — Returns `Err(MlOpError::ZeroFeatures)` if either feature count is 0.

### `PoolParams`

Shared by `MaxPool2d` and `AvgPool2d`:

```rust
pub struct PoolParams {
    pub kernel_size: [usize; 2],
    pub stride:      [usize; 2],
    pub padding:     [usize; 2],
}
```

**`PoolParams::new(kernel_size, stride, padding)`** — Returns `Err` if `kernel_size` or `stride` contain zeros.

### `BatchNormParams`

```rust
pub struct BatchNormParams {
    pub num_features: usize,
    pub eps:          f64,
    /// None = cumulative moving average. Some(0.1) is a common default.
    pub momentum:     Option<f64>,
}
```

**`BatchNormParams::new(num_features, eps, momentum)`** — Returns `Err` if `num_features == 0`, `eps <= 0`, or `momentum` is outside `[0.0, 1.0)`.

### `LayerNormParams`

```rust
pub struct LayerNormParams {
    pub normalized_shape: Vec<usize>,
    pub eps:              f64,
}
```

**`LayerNormParams::new(normalized_shape, eps)`** — Returns `Err` if `normalized_shape` is empty, contains zeros, or `eps <= 0`.

### `SoftmaxParams`

```rust
pub struct SoftmaxParams {
    pub axis: i32,   // negative values index from the end
}
```

**`SoftmaxParams::new(axis)`** — Infallible constructor.

### `ReshapeParams`

```rust
pub struct ReshapeParams {
    pub target_shape: Shape,  // validated Shape (see shape.md)
}
```

`ReshapeParams::new(shape)` takes a pre-validated [`Shape`](shape.md) and is infallible.

### `TransposeParams`

```rust
pub struct TransposeParams {
    pub perm: Vec<usize>,   // must be a permutation of 0..rank
}
```

**`TransposeParams::new(perm)`** — Returns `Err(MlOpError::InvalidPermutation)` if `perm` is empty or is not a valid permutation of `0..perm.len()`.

### `ConcatParams`

```rust
pub struct ConcatParams {
    pub axis: i32,   // negative values index from the end
}
```

**`ConcatParams::new(axis)`** — Infallible constructor.

### `FlattenParams`

```rust
pub struct FlattenParams {
    pub start_dim: i32,   // inclusive
    pub end_dim:   i32,   // inclusive; -1 means last dim
}
```

**`FlattenParams::new(start_dim, end_dim)`** — Infallible constructor.

### `DropoutParams`

```rust
pub struct DropoutParams {
    pub p: f64,   // probability in [0.0, 1.0)
}
```

**`DropoutParams::new(p)`** — Returns `Err(MlOpError::InvalidDropoutP)` if `p` is outside `[0.0, 1.0)`.

## Usage Examples

### Using safe constructors (recommended)

```rust
use graphynx::ml_op::{
    MlOp, Conv2dParams, LinearParams, MatMulParams,
    SoftmaxParams, BatchNormParams, PoolParams,
    DropoutParams, TransposeParams,
};

// Standard 3×3 convolution — validated constructor
let conv = MlOp::Conv2d(Conv2dParams::new(
    [3, 3],   // kernel_size
    [1, 1],   // stride
    [1, 1],   // padding
    [1, 1],   // dilation
    1,        // groups
).unwrap());
assert_eq!(conv.name(), "Conv2d");
assert!(conv.is_spatial_2d());

// Fully-connected layer — validated constructor
let fc = MlOp::Linear(LinearParams::new(1024, 256, true).unwrap());
assert_eq!(fc.name(), "Linear");

// Batch normalisation — validated constructor
let bn = MlOp::BatchNorm(BatchNormParams::new(64, 1e-5, Some(0.1)).unwrap());

// Transpose — validated permutation
let t = MlOp::Transpose(TransposeParams::new(vec![0, 2, 1]).unwrap());

// Dropout — validated probability
let drop = MlOp::Dropout(DropoutParams::new(0.5).unwrap());

// Parameterless activations (no constructor needed)
let relu = MlOp::Relu;
assert!(relu.is_parameterless());
println!("{}", relu); // "Relu"
```

### Using `MlOp::custom()` (safe constructor)

```rust
use graphynx::ml_op::MlOp;

// Safe custom op constructor — rejects empty names
let custom = MlOp::custom("my_fused_op", vec![/* serialised params */]).unwrap();
assert!(custom.is_custom());

// Empty name is rejected
assert!(MlOp::custom("", vec![]).is_err());
```

### Direct struct construction (unchecked)

```rust
use graphynx::ml_op::{MlOp, Conv2dParams};

// Direct construction bypasses validation — use only with known-good values
let conv = MlOp::Conv2d(Conv2dParams {
    kernel_size: [3, 3],
    stride:      [1, 1],
    padding:     [1, 1],
    dilation:    [1, 1],
    groups:      1,
});
```

### Validation errors

```rust
use graphynx::ml_op::{Conv2dParams, LinearParams, DropoutParams, MlOpError};

// Zero kernel size is rejected
let err = Conv2dParams::new([0, 3], [1, 1], [0, 0], [1, 1], 1).unwrap_err();
assert!(matches!(err, MlOpError::ZeroSpatialParam { .. }));

// Zero features rejected
let err = LinearParams::new(0, 256, true).unwrap_err();
assert!(matches!(err, MlOpError::ZeroFeatures { .. }));

// Dropout p out of range
let err = DropoutParams::new(1.0).unwrap_err();
assert!(matches!(err, MlOpError::InvalidDropoutP(_)));
```

## Extension Pattern

When a backend receives a node with a `Custom` op, it should inspect `name` and deserialise `params`:

```rust
fn dispatch_ml_op(&self, op: &MlOp, ...) -> Result<(), BackendError> {
    match op {
        MlOp::Relu    => { /* element-wise max(0, x) */ }
        MlOp::Conv2d(p) => { /* cuDNN conv forward */ }
        MlOp::Custom { name, params } if name == "my_fused_op" => {
            let config: MyOpConfig = serde_json::from_slice(params)
                .map_err(|e| BackendError::InvalidKernel(e.to_string()))?;
            // ... execute fused op
        }
        _ => return Err(BackendError::UnsupportedOp),
    }
    Ok(())
}
```

## Further Reading

- [Shape Module](shape.md) — `Shape` type used in `ReshapeParams`
- [Tensor Type System](tensor-type.md) — `TensorType`, `Dim`, `Layout` used in `ReshapeParams`
- [Backend Trait System](backend-trait.md) — `dispatch_ml_op` and `dispatch_ml_model`
- [Architecture Overview](architecture.md) — where `MlOp` sits in the layered design
- [ARCHITECTURE.md](../ARCHITECTURE.md) — full long-term plan including the Graph IR that hosts `MlOp` nodes
