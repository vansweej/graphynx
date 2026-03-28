use std::fmt;

use thiserror::Error;

use crate::shape::Shape;

// ── MlOpError ─────────────────────────────────────────────────────────────────

/// Errors produced when constructing ML operation parameters through safe
/// constructors.
///
/// Every variant carries enough context for the caller to understand exactly
/// what invariant was violated.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MlOpError {
    /// A spatial dimension parameter (kernel size, stride, dilation) was zero.
    /// All spatial parameters must be >= 1.
    #[error("{param} must be > 0, got 0")]
    ZeroSpatialParam {
        /// Name of the parameter that was zero.
        param: String,
    },

    /// The groups parameter was zero. Must be >= 1.
    #[error("groups must be > 0, got 0")]
    ZeroGroups,

    /// A feature count was zero. Must be >= 1.
    #[error("{param} must be > 0, got 0")]
    ZeroFeatures {
        /// Name of the parameter that was zero.
        param: String,
    },

    /// Batch normalisation `num_features` was zero.
    #[error("num_features must be > 0, got 0")]
    ZeroNumFeatures,

    /// Epsilon was non-positive. Must be > 0.
    #[error("eps must be > 0, got {0}")]
    NonPositiveEps(f64),

    /// Momentum was outside the valid range `[0.0, 1.0)`.
    #[error("momentum must be in [0.0, 1.0), got {0}")]
    InvalidMomentum(f64),

    /// Dropout probability was outside the valid range `[0.0, 1.0)`.
    #[error("dropout p must be in [0.0, 1.0), got {0}")]
    InvalidDropoutP(f64),

    /// Normalised shape contained a zero dimension or was empty.
    #[error("normalized_shape must be non-empty with all values > 0")]
    InvalidNormalizedShape,

    /// Permutation is not a valid permutation of `0..rank`.
    #[error("perm must be a permutation of 0..{expected_len}, got {perm:?}")]
    InvalidPermutation {
        /// The permutation that was supplied.
        perm: Vec<usize>,
        /// Expected length (i.e. tensor rank).
        expected_len: usize,
    },

    /// Custom operation name was empty.
    #[error("Custom op name must not be empty")]
    EmptyCustomName,
}

// ── Parameter structs ─────────────────────────────────────────────────────────

/// Parameters for a 2-D convolution operation.
///
/// All spatial parameters are `[height, width]` ordered arrays.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Conv2dParams {
    /// Size of the convolution kernel: `[kernel_h, kernel_w]`.
    pub kernel_size: [usize; 2],
    /// Stride along each spatial dimension: `[stride_h, stride_w]`.
    pub stride: [usize; 2],
    /// Zero-padding along each spatial dimension: `[pad_h, pad_w]`.
    pub padding: [usize; 2],
    /// Dilation factor along each spatial dimension: `[dil_h, dil_w]`.
    pub dilation: [usize; 2],
    /// Number of blocked connections from input to output channels.
    /// `1` is a standard convolution; equal to the channel count gives a
    /// depth-wise convolution.
    pub groups: usize,
}

impl Conv2dParams {
    /// Construct validated convolution parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError`] if `kernel_size`, `stride`, or `dilation` contain
    /// a zero, or if `groups` is zero. `padding` may contain zeros (no padding
    /// is valid).
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::Conv2dParams;
    ///
    /// let p = Conv2dParams::new([3, 3], [1, 1], [1, 1], [1, 1], 1).unwrap();
    /// assert_eq!(p.kernel_size, [3, 3]);
    ///
    /// assert!(Conv2dParams::new([0, 3], [1, 1], [0, 0], [1, 1], 1).is_err());
    /// ```
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self, MlOpError> {
        if kernel_size[0] == 0 || kernel_size[1] == 0 {
            return Err(MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            });
        }
        if stride[0] == 0 || stride[1] == 0 {
            return Err(MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            });
        }
        if dilation[0] == 0 || dilation[1] == 0 {
            return Err(MlOpError::ZeroSpatialParam {
                param: "dilation".to_string(),
            });
        }
        if groups == 0 {
            return Err(MlOpError::ZeroGroups);
        }
        Ok(Conv2dParams {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        })
    }
}

/// Parameters for a general matrix multiplication.
///
/// Computes `C = op(A) · op(B)`, where `op` is optionally a transpose.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatMulParams {
    /// If `true`, transpose matrix A before multiplying.
    pub transpose_a: bool,
    /// If `true`, transpose matrix B before multiplying.
    pub transpose_b: bool,
}

impl MatMulParams {
    /// Construct matrix multiplication parameters.
    ///
    /// This constructor is infallible — boolean flags have no invalid states.
    pub fn new(transpose_a: bool, transpose_b: bool) -> Self {
        MatMulParams {
            transpose_a,
            transpose_b,
        }
    }
}

/// Parameters for a fully-connected (dense) linear layer.
///
/// Computes `y = x · W^T + b` (when `bias` is `true`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LinearParams {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Whether to include a bias term.
    pub bias: bool,
}

impl LinearParams {
    /// Construct validated linear layer parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError::ZeroFeatures`] if `in_features` or `out_features`
    /// is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::LinearParams;
    ///
    /// let p = LinearParams::new(512, 256, true).unwrap();
    /// assert_eq!(p.in_features, 512);
    ///
    /// assert!(LinearParams::new(0, 256, true).is_err());
    /// ```
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self, MlOpError> {
        if in_features == 0 {
            return Err(MlOpError::ZeroFeatures {
                param: "in_features".to_string(),
            });
        }
        if out_features == 0 {
            return Err(MlOpError::ZeroFeatures {
                param: "out_features".to_string(),
            });
        }
        Ok(LinearParams {
            in_features,
            out_features,
            bias,
        })
    }
}

/// Parameters shared by pooling operations (max-pool and avg-pool).
///
/// All spatial parameters are `[height, width]` ordered arrays.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PoolParams {
    /// Spatial size of the pooling window: `[kh, kw]`.
    pub kernel_size: [usize; 2],
    /// Stride of the pooling window: `[sh, sw]`.
    pub stride: [usize; 2],
    /// Zero-padding around the input: `[ph, pw]`.
    pub padding: [usize; 2],
}

impl PoolParams {
    /// Construct validated pooling parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError::ZeroSpatialParam`] if `kernel_size` or `stride`
    /// contain a zero. `padding` may contain zeros (no padding is valid).
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::PoolParams;
    ///
    /// let p = PoolParams::new([2, 2], [2, 2], [0, 0]).unwrap();
    /// assert_eq!(p.kernel_size, [2, 2]);
    ///
    /// assert!(PoolParams::new([0, 2], [2, 2], [0, 0]).is_err());
    /// ```
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self, MlOpError> {
        if kernel_size[0] == 0 || kernel_size[1] == 0 {
            return Err(MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            });
        }
        if stride[0] == 0 || stride[1] == 0 {
            return Err(MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            });
        }
        Ok(PoolParams {
            kernel_size,
            stride,
            padding,
        })
    }
}

/// Parameters for batch normalisation.
///
/// See [Batch Normalization: Accelerating Deep Network Training](
/// https://arxiv.org/abs/1502.03167).
#[derive(Clone, Debug, PartialEq)]
pub struct BatchNormParams {
    /// Number of features / channels in the input.
    pub num_features: usize,
    /// Small constant added to the variance for numerical stability.
    pub eps: f64,
    /// Momentum for the running mean/variance update. `None` uses cumulative
    /// moving average; a value like `0.1` is a common default.
    pub momentum: Option<f64>,
}

impl BatchNormParams {
    /// Construct validated batch normalisation parameters.
    ///
    /// # Errors
    ///
    /// - [`MlOpError::ZeroNumFeatures`] if `num_features == 0`.
    /// - [`MlOpError::NonPositiveEps`] if `eps <= 0.0`.
    /// - [`MlOpError::InvalidMomentum`] if `momentum` is outside `[0.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::BatchNormParams;
    ///
    /// let p = BatchNormParams::new(64, 1e-5, Some(0.1)).unwrap();
    /// assert_eq!(p.num_features, 64);
    ///
    /// assert!(BatchNormParams::new(0, 1e-5, None).is_err());
    /// assert!(BatchNormParams::new(64, -1.0, None).is_err());
    /// assert!(BatchNormParams::new(64, 1e-5, Some(1.0)).is_err());
    /// ```
    pub fn new(num_features: usize, eps: f64, momentum: Option<f64>) -> Result<Self, MlOpError> {
        if num_features == 0 {
            return Err(MlOpError::ZeroNumFeatures);
        }
        if eps <= 0.0 {
            return Err(MlOpError::NonPositiveEps(eps));
        }
        if let Some(m) = momentum {
            if !(0.0..1.0).contains(&m) {
                return Err(MlOpError::InvalidMomentum(m));
            }
        }
        Ok(BatchNormParams {
            num_features,
            eps,
            momentum,
        })
    }
}

/// Parameters for layer normalisation.
///
/// See [Layer Normalization](https://arxiv.org/abs/1607.06450).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormParams {
    /// Shape of the normalised sub-tensor (the trailing dimensions).
    pub normalized_shape: Vec<usize>,
    /// Small constant added to the variance for numerical stability.
    pub eps: f64,
}

impl LayerNormParams {
    /// Construct validated layer normalisation parameters.
    ///
    /// # Errors
    ///
    /// - [`MlOpError::InvalidNormalizedShape`] if `normalized_shape` is empty
    ///   or contains a zero.
    /// - [`MlOpError::NonPositiveEps`] if `eps <= 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::LayerNormParams;
    ///
    /// let p = LayerNormParams::new(vec![768], 1e-12).unwrap();
    /// assert_eq!(p.normalized_shape, vec![768]);
    ///
    /// assert!(LayerNormParams::new(vec![], 1e-12).is_err());
    /// assert!(LayerNormParams::new(vec![0], 1e-12).is_err());
    /// assert!(LayerNormParams::new(vec![768], -1.0).is_err());
    /// ```
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Result<Self, MlOpError> {
        if normalized_shape.is_empty() || normalized_shape.contains(&0) {
            return Err(MlOpError::InvalidNormalizedShape);
        }
        if eps <= 0.0 {
            return Err(MlOpError::NonPositiveEps(eps));
        }
        Ok(LayerNormParams {
            normalized_shape,
            eps,
        })
    }
}

/// Parameters for softmax activation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SoftmaxParams {
    /// The axis along which softmax is computed. Negative values index from
    /// the end (e.g., `-1` is the last axis).
    pub axis: i32,
}

impl SoftmaxParams {
    /// Construct softmax parameters.
    ///
    /// This constructor is infallible — axis validation requires knowing the
    /// tensor rank, which is checked at graph-build time rather than at
    /// parameter construction time.
    pub fn new(axis: i32) -> Self {
        SoftmaxParams { axis }
    }
}

/// Parameters for a reshape operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReshapeParams {
    /// Target shape. May contain [`Dim::Dynamic`] or [`Dim::Symbolic`]
    /// entries to propagate runtime dimensions.
    pub target_shape: Shape,
}

impl ReshapeParams {
    /// Construct reshape parameters from a validated [`Shape`].
    ///
    /// This constructor is infallible — the [`Shape`] is already validated at
    /// construction time.
    pub fn new(target_shape: Shape) -> Self {
        ReshapeParams { target_shape }
    }
}

/// Parameters for a tensor transpose / permutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransposeParams {
    /// Permutation of axes. Must be a permutation of `0..rank`.
    pub perm: Vec<usize>,
}

impl TransposeParams {
    /// Construct validated transpose parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError::InvalidPermutation`] if `perm` is empty or is not
    /// a valid permutation of `0..perm.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::TransposeParams;
    ///
    /// let p = TransposeParams::new(vec![1, 0]).unwrap();
    /// assert_eq!(p.perm, vec![1, 0]);
    ///
    /// assert!(TransposeParams::new(vec![]).is_err());
    /// assert!(TransposeParams::new(vec![0, 0]).is_err());
    /// assert!(TransposeParams::new(vec![0, 2]).is_err());
    /// ```
    pub fn new(perm: Vec<usize>) -> Result<Self, MlOpError> {
        let len = perm.len();
        if len == 0 {
            return Err(MlOpError::InvalidPermutation {
                perm,
                expected_len: 0,
            });
        }
        // Check that perm is a valid permutation of 0..len.
        let mut seen = vec![false; len];
        for &idx in &perm {
            if idx >= len || seen[idx] {
                return Err(MlOpError::InvalidPermutation {
                    perm,
                    expected_len: len,
                });
            }
            seen[idx] = true;
        }
        Ok(TransposeParams { perm })
    }
}

/// Parameters for concatenating tensors along an axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ConcatParams {
    /// Axis along which to concatenate. Negative values index from the end.
    pub axis: i32,
}

impl ConcatParams {
    /// Construct concatenation parameters.
    ///
    /// This constructor is infallible — axis validation requires knowing the
    /// tensor rank, which is checked at graph-build time.
    pub fn new(axis: i32) -> Self {
        ConcatParams { axis }
    }
}

/// Parameters for flattening a range of dimensions into one.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FlattenParams {
    /// First dimension to flatten (inclusive). Negative values index from end.
    pub start_dim: i32,
    /// Last dimension to flatten (inclusive). Negative values index from end.
    pub end_dim: i32,
}

impl FlattenParams {
    /// Construct flatten parameters.
    ///
    /// This constructor is infallible — axis validation requires knowing the
    /// tensor rank, which is checked at graph-build time.
    pub fn new(start_dim: i32, end_dim: i32) -> Self {
        FlattenParams { start_dim, end_dim }
    }
}

/// Parameters for dropout regularisation.
#[derive(Clone, Debug, PartialEq)]
pub struct DropoutParams {
    /// Probability of an element being zeroed. Must be in `[0.0, 1.0)`.
    pub p: f64,
}

impl DropoutParams {
    /// Construct validated dropout parameters.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError::InvalidDropoutP`] if `p` is outside `[0.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::DropoutParams;
    ///
    /// let p = DropoutParams::new(0.5).unwrap();
    /// assert_eq!(p.p, 0.5);
    ///
    /// assert!(DropoutParams::new(-0.1).is_err());
    /// assert!(DropoutParams::new(1.0).is_err());
    /// ```
    pub fn new(p: f64) -> Result<Self, MlOpError> {
        if !(0.0..1.0).contains(&p) {
            return Err(MlOpError::InvalidDropoutP(p));
        }
        Ok(DropoutParams { p })
    }
}

// ── MlOp ─────────────────────────────────────────────────────────────────────

/// Curated catalog of primitive ML operations.
///
/// Each variant maps to a well-known operation that multiple backends can
/// implement. The engine validates input/output [`crate::tensor_type::TensorType`]
/// against an op's signature at graph-build time.
///
/// # Categories
///
/// | Category | Variants |
/// |---|---|
/// | Linear algebra | [`MatMul`](MlOp::MatMul), [`Linear`](MlOp::Linear) |
/// | Convolution | [`Conv2d`](MlOp::Conv2d) |
/// | Activation | [`Relu`](MlOp::Relu), [`Sigmoid`](MlOp::Sigmoid), [`Tanh`](MlOp::Tanh), [`Gelu`](MlOp::Gelu), [`Softmax`](MlOp::Softmax) |
/// | Normalisation | [`BatchNorm`](MlOp::BatchNorm), [`LayerNorm`](MlOp::LayerNorm) |
/// | Pooling | [`MaxPool2d`](MlOp::MaxPool2d), [`AvgPool2d`](MlOp::AvgPool2d) |
/// | Shape | [`Reshape`](MlOp::Reshape), [`Transpose`](MlOp::Transpose), [`Concat`](MlOp::Concat), [`Flatten`](MlOp::Flatten) |
/// | Regularisation | [`Dropout`](MlOp::Dropout) |
/// | Element-wise | [`Add`](MlOp::Add), [`Mul`](MlOp::Mul) |
/// | Escape hatch | [`Custom`](MlOp::Custom) |
///
/// # Extension
///
/// For any operation not covered by the catalog use [`MlOp::Custom`]. The
/// `name` string is a backend-interpreted identifier and `params` carries
/// serialised (e.g. JSON or binary) operation parameters.
///
/// # Examples
///
/// ```
/// use graphynx::ml_op::{MlOp, Conv2dParams, SoftmaxParams};
///
/// let conv = MlOp::Conv2d(Conv2dParams {
///     kernel_size: [3, 3],
///     stride:      [1, 1],
///     padding:     [1, 1],
///     dilation:    [1, 1],
///     groups:      1,
/// });
/// assert_eq!(conv.name(), "Conv2d");
/// assert!(!conv.is_parameterless());
///
/// let relu = MlOp::Relu;
/// assert_eq!(relu.name(), "Relu");
/// assert!(relu.is_parameterless());
///
/// let softmax = MlOp::Softmax(SoftmaxParams { axis: -1 });
/// assert_eq!(softmax.name(), "Softmax");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum MlOp {
    // ── Linear algebra ───────────────────────────────────────────────────
    /// General matrix multiplication: `C = op(A) · op(B)`.
    MatMul(MatMulParams),
    /// Fully-connected linear layer: `y = x · W^T + b`.
    Linear(LinearParams),

    // ── Convolution ──────────────────────────────────────────────────────
    /// 2-D spatial convolution.
    Conv2d(Conv2dParams),

    // ── Activation ───────────────────────────────────────────────────────
    /// Rectified linear unit: `max(0, x)`.
    Relu,
    /// Sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Gaussian error linear unit.
    Gelu,
    /// Softmax along a given axis.
    Softmax(SoftmaxParams),

    // ── Normalisation ────────────────────────────────────────────────────
    /// Batch normalisation over a mini-batch of inputs.
    BatchNorm(BatchNormParams),
    /// Layer normalisation over the last N dimensions.
    LayerNorm(LayerNormParams),

    // ── Pooling ──────────────────────────────────────────────────────────
    /// 2-D max pooling.
    MaxPool2d(PoolParams),
    /// 2-D average pooling.
    AvgPool2d(PoolParams),

    // ── Shape manipulation ───────────────────────────────────────────────
    /// Reshape to a new shape (total element count must be preserved).
    Reshape(ReshapeParams),
    /// Permute the axes of a tensor.
    Transpose(TransposeParams),
    /// Concatenate tensors along an axis.
    Concat(ConcatParams),
    /// Flatten a range of axes into a single dimension.
    Flatten(FlattenParams),

    // ── Regularisation ───────────────────────────────────────────────────
    /// Drop elements randomly during training.
    Dropout(DropoutParams),

    // ── Element-wise arithmetic ──────────────────────────────────────────
    /// Element-wise addition of two tensors with matching shapes.
    Add,
    /// Element-wise multiplication of two tensors with matching shapes.
    Mul,

    // ── Escape hatch ─────────────────────────────────────────────────────
    /// Any operation not covered by the catalog.
    ///
    /// `name` is a backend-interpreted identifier (e.g. `"my_custom_op"`).
    /// `params` carries serialised parameters in whatever format the backend
    /// expects (JSON, protobuf, raw bytes, etc.).
    Custom {
        /// Backend-interpreted operation name.
        name: String,
        /// Serialised operation parameters.
        params: Vec<u8>,
    },
}

impl MlOp {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Construct a [`MlOp::Custom`] operation, rejecting empty names.
    ///
    /// # Errors
    ///
    /// Returns [`MlOpError::EmptyCustomName`] if `name` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::MlOp;
    ///
    /// assert!(MlOp::custom("my_op", vec![]).is_ok());
    /// assert!(MlOp::custom("", vec![]).is_err());
    /// ```
    pub fn custom(name: impl Into<String>, params: Vec<u8>) -> Result<Self, MlOpError> {
        let n = name.into();
        if n.is_empty() {
            Err(MlOpError::EmptyCustomName)
        } else {
            Ok(MlOp::Custom { name: n, params })
        }
    }

    /// Human-readable name for the operation.
    ///
    /// For [`MlOp::Custom`] this returns the user-supplied `name` string.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::{MlOp, MatMulParams};
    ///
    /// assert_eq!(MlOp::Relu.name(), "Relu");
    /// assert_eq!(MlOp::MatMul(MatMulParams { transpose_a: false, transpose_b: false }).name(), "MatMul");
    /// assert_eq!(MlOp::Custom { name: "my_op".into(), params: vec![] }.name(), "my_op");
    /// ```
    pub fn name(&self) -> &str {
        match self {
            MlOp::MatMul(_) => "MatMul",
            MlOp::Linear(_) => "Linear",
            MlOp::Conv2d(_) => "Conv2d",
            MlOp::Relu => "Relu",
            MlOp::Sigmoid => "Sigmoid",
            MlOp::Tanh => "Tanh",
            MlOp::Gelu => "Gelu",
            MlOp::Softmax(_) => "Softmax",
            MlOp::BatchNorm(_) => "BatchNorm",
            MlOp::LayerNorm(_) => "LayerNorm",
            MlOp::MaxPool2d(_) => "MaxPool2d",
            MlOp::AvgPool2d(_) => "AvgPool2d",
            MlOp::Reshape(_) => "Reshape",
            MlOp::Transpose(_) => "Transpose",
            MlOp::Concat(_) => "Concat",
            MlOp::Flatten(_) => "Flatten",
            MlOp::Dropout(_) => "Dropout",
            MlOp::Add => "Add",
            MlOp::Mul => "Mul",
            MlOp::Custom { name, .. } => name.as_str(),
        }
    }

    /// Returns `true` if this operation carries no parameters.
    ///
    /// The parameterless variants are: [`Relu`](MlOp::Relu),
    /// [`Sigmoid`](MlOp::Sigmoid), [`Tanh`](MlOp::Tanh),
    /// [`Gelu`](MlOp::Gelu), [`Add`](MlOp::Add), and [`Mul`](MlOp::Mul).
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::MlOp;
    ///
    /// assert!(MlOp::Relu.is_parameterless());
    /// assert!(MlOp::Add.is_parameterless());
    /// assert!(!MlOp::Custom { name: "x".into(), params: vec![] }.is_parameterless());
    /// ```
    pub fn is_parameterless(&self) -> bool {
        matches!(
            self,
            MlOp::Relu | MlOp::Sigmoid | MlOp::Tanh | MlOp::Gelu | MlOp::Add | MlOp::Mul
        )
    }

    /// Returns `true` if this is a [`MlOp::Custom`] operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::MlOp;
    ///
    /// assert!(MlOp::Custom { name: "foo".into(), params: vec![] }.is_custom());
    /// assert!(!MlOp::Relu.is_custom());
    /// ```
    pub fn is_custom(&self) -> bool {
        matches!(self, MlOp::Custom { .. })
    }

    /// Returns `true` if this operation is a 2-D spatial operation
    /// (convolution or pooling), which requires a 4-D input tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::{MlOp, Conv2dParams, PoolParams};
    ///
    /// assert!(MlOp::Conv2d(Conv2dParams {
    ///     kernel_size: [3, 3], stride: [1, 1],
    ///     padding: [0, 0], dilation: [1, 1], groups: 1,
    /// }).is_spatial_2d());
    ///
    /// assert!(MlOp::MaxPool2d(PoolParams {
    ///     kernel_size: [2, 2], stride: [2, 2], padding: [0, 0],
    /// }).is_spatial_2d());
    ///
    /// assert!(!MlOp::Relu.is_spatial_2d());
    /// ```
    pub fn is_spatial_2d(&self) -> bool {
        matches!(
            self,
            MlOp::Conv2d(_) | MlOp::MaxPool2d(_) | MlOp::AvgPool2d(_)
        )
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl fmt::Display for MlOp {
    /// Formats the op as its name, e.g. `"Relu"`, `"Conv2d"`, `"my_custom_op"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphynx::ml_op::MlOp;
    ///
    /// assert_eq!(MlOp::Relu.to_string(), "Relu");
    /// assert_eq!(MlOp::Custom { name: "bar".into(), params: vec![] }.to_string(), "bar");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::tensor_type::Dim;

    use super::*;

    // ── MlOp::name() ─────────────────────────────────────────────────────

    #[test]
    fn name_matmul() {
        let op = MlOp::MatMul(MatMulParams {
            transpose_a: false,
            transpose_b: true,
        });
        assert_eq!(op.name(), "MatMul");
    }

    #[test]
    fn name_linear() {
        let op = MlOp::Linear(LinearParams {
            in_features: 512,
            out_features: 256,
            bias: true,
        });
        assert_eq!(op.name(), "Linear");
    }

    #[test]
    fn name_conv2d() {
        let op = MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        });
        assert_eq!(op.name(), "Conv2d");
    }

    #[test]
    fn name_relu() {
        assert_eq!(MlOp::Relu.name(), "Relu");
    }

    #[test]
    fn name_sigmoid() {
        assert_eq!(MlOp::Sigmoid.name(), "Sigmoid");
    }

    #[test]
    fn name_tanh() {
        assert_eq!(MlOp::Tanh.name(), "Tanh");
    }

    #[test]
    fn name_gelu() {
        assert_eq!(MlOp::Gelu.name(), "Gelu");
    }

    #[test]
    fn name_softmax() {
        let op = MlOp::Softmax(SoftmaxParams { axis: -1 });
        assert_eq!(op.name(), "Softmax");
    }

    #[test]
    fn name_batchnorm() {
        let op = MlOp::BatchNorm(BatchNormParams {
            num_features: 64,
            eps: 1e-5,
            momentum: Some(0.1),
        });
        assert_eq!(op.name(), "BatchNorm");
    }

    #[test]
    fn name_layernorm() {
        let op = MlOp::LayerNorm(LayerNormParams {
            normalized_shape: vec![768],
            eps: 1e-12,
        });
        assert_eq!(op.name(), "LayerNorm");
    }

    #[test]
    fn name_maxpool2d() {
        let op = MlOp::MaxPool2d(PoolParams {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        });
        assert_eq!(op.name(), "MaxPool2d");
    }

    #[test]
    fn name_avgpool2d() {
        let op = MlOp::AvgPool2d(PoolParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
        });
        assert_eq!(op.name(), "AvgPool2d");
    }

    #[test]
    fn name_reshape() {
        let op = MlOp::Reshape(ReshapeParams {
            target_shape: Shape::new(vec![Dim::fixed(4).unwrap(), Dim::Dynamic]).unwrap(),
        });
        assert_eq!(op.name(), "Reshape");
    }

    #[test]
    fn name_transpose() {
        let op = MlOp::Transpose(TransposeParams {
            perm: vec![0, 2, 1, 3],
        });
        assert_eq!(op.name(), "Transpose");
    }

    #[test]
    fn name_concat() {
        let op = MlOp::Concat(ConcatParams { axis: 1 });
        assert_eq!(op.name(), "Concat");
    }

    #[test]
    fn name_flatten() {
        let op = MlOp::Flatten(FlattenParams {
            start_dim: 1,
            end_dim: -1,
        });
        assert_eq!(op.name(), "Flatten");
    }

    #[test]
    fn name_dropout() {
        let op = MlOp::Dropout(DropoutParams { p: 0.5 });
        assert_eq!(op.name(), "Dropout");
    }

    #[test]
    fn name_add() {
        assert_eq!(MlOp::Add.name(), "Add");
    }

    #[test]
    fn name_mul() {
        assert_eq!(MlOp::Mul.name(), "Mul");
    }

    #[test]
    fn name_custom_uses_supplied_name() {
        let op = MlOp::Custom {
            name: "my_special_op".to_string(),
            params: vec![1, 2, 3],
        };
        assert_eq!(op.name(), "my_special_op");
    }

    #[test]
    fn name_custom_empty_params() {
        let op = MlOp::Custom {
            name: "no_params".to_string(),
            params: vec![],
        };
        assert_eq!(op.name(), "no_params");
    }

    // ── MlOp::is_parameterless() ─────────────────────────────────────────

    #[test]
    fn is_parameterless_relu() {
        assert!(MlOp::Relu.is_parameterless());
    }

    #[test]
    fn is_parameterless_sigmoid() {
        assert!(MlOp::Sigmoid.is_parameterless());
    }

    #[test]
    fn is_parameterless_tanh() {
        assert!(MlOp::Tanh.is_parameterless());
    }

    #[test]
    fn is_parameterless_gelu() {
        assert!(MlOp::Gelu.is_parameterless());
    }

    #[test]
    fn is_parameterless_add() {
        assert!(MlOp::Add.is_parameterless());
    }

    #[test]
    fn is_parameterless_mul() {
        assert!(MlOp::Mul.is_parameterless());
    }

    #[test]
    fn not_parameterless_conv2d() {
        let op = MlOp::Conv2d(Conv2dParams {
            kernel_size: [1, 1],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        });
        assert!(!op.is_parameterless());
    }

    #[test]
    fn not_parameterless_softmax() {
        assert!(!MlOp::Softmax(SoftmaxParams { axis: 0 }).is_parameterless());
    }

    #[test]
    fn not_parameterless_batchnorm() {
        assert!(!MlOp::BatchNorm(BatchNormParams {
            num_features: 32,
            eps: 1e-5,
            momentum: None,
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_layernorm() {
        assert!(!MlOp::LayerNorm(LayerNormParams {
            normalized_shape: vec![512],
            eps: 1e-5,
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_matmul() {
        assert!(!MlOp::MatMul(MatMulParams {
            transpose_a: false,
            transpose_b: false,
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_linear() {
        assert!(!MlOp::Linear(LinearParams {
            in_features: 10,
            out_features: 5,
            bias: false,
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_maxpool() {
        assert!(!MlOp::MaxPool2d(PoolParams {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_avgpool() {
        assert!(!MlOp::AvgPool2d(PoolParams {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_reshape() {
        assert!(!MlOp::Reshape(ReshapeParams {
            target_shape: Shape::scalar(),
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_transpose() {
        assert!(!MlOp::Transpose(TransposeParams { perm: vec![1, 0] }).is_parameterless());
    }

    #[test]
    fn not_parameterless_concat() {
        assert!(!MlOp::Concat(ConcatParams { axis: 0 }).is_parameterless());
    }

    #[test]
    fn not_parameterless_flatten() {
        assert!(!MlOp::Flatten(FlattenParams {
            start_dim: 0,
            end_dim: -1,
        })
        .is_parameterless());
    }

    #[test]
    fn not_parameterless_dropout() {
        assert!(!MlOp::Dropout(DropoutParams { p: 0.1 }).is_parameterless());
    }

    #[test]
    fn not_parameterless_custom() {
        assert!(!MlOp::Custom {
            name: "x".into(),
            params: vec![],
        }
        .is_parameterless());
    }

    // ── MlOp::is_custom() ────────────────────────────────────────────────

    #[test]
    fn is_custom_true_for_custom() {
        assert!(MlOp::Custom {
            name: "foo".into(),
            params: vec![42],
        }
        .is_custom());
    }

    #[test]
    fn is_custom_false_for_relu() {
        assert!(!MlOp::Relu.is_custom());
    }

    #[test]
    fn is_custom_false_for_conv2d() {
        assert!(!MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        })
        .is_custom());
    }

    // ── MlOp::is_spatial_2d() ────────────────────────────────────────────

    #[test]
    fn is_spatial_2d_conv2d() {
        assert!(MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
        })
        .is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_maxpool() {
        assert!(MlOp::MaxPool2d(PoolParams {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        })
        .is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_avgpool() {
        assert!(MlOp::AvgPool2d(PoolParams {
            kernel_size: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        })
        .is_spatial_2d());
    }

    #[test]
    fn not_spatial_2d_relu() {
        assert!(!MlOp::Relu.is_spatial_2d());
    }

    #[test]
    fn not_spatial_2d_matmul() {
        assert!(!MlOp::MatMul(MatMulParams {
            transpose_a: false,
            transpose_b: false,
        })
        .is_spatial_2d());
    }

    #[test]
    fn not_spatial_2d_add() {
        assert!(!MlOp::Add.is_spatial_2d());
    }

    #[test]
    fn not_spatial_2d_custom() {
        assert!(!MlOp::Custom {
            name: "special".into(),
            params: vec![],
        }
        .is_spatial_2d());
    }

    // ── Display ───────────────────────────────────────────────────────────

    #[test]
    fn display_relu() {
        assert_eq!(MlOp::Relu.to_string(), "Relu");
    }

    #[test]
    fn display_conv2d() {
        let op = MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        });
        assert_eq!(op.to_string(), "Conv2d");
    }

    #[test]
    fn display_custom() {
        let op = MlOp::Custom {
            name: "my_op".to_string(),
            params: vec![],
        };
        assert_eq!(op.to_string(), "my_op");
    }

    #[test]
    fn display_add() {
        assert_eq!(MlOp::Add.to_string(), "Add");
    }

    // ── Clone / Debug ─────────────────────────────────────────────────────

    #[test]
    fn clone_and_eq_relu() {
        let op = MlOp::Relu;
        assert_eq!(op.clone(), op);
    }

    #[test]
    fn clone_and_eq_conv2d() {
        let op = MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [2, 2],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 4,
        });
        assert_eq!(op.clone(), op);
    }

    #[test]
    fn clone_and_eq_custom_with_params() {
        let op = MlOp::Custom {
            name: "test".into(),
            params: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        assert_eq!(op.clone(), op);
    }

    #[test]
    fn debug_relu() {
        let s = format!("{:?}", MlOp::Relu);
        assert!(s.contains("Relu"));
    }

    #[test]
    fn debug_custom() {
        let op = MlOp::Custom {
            name: "my_custom".into(),
            params: vec![1, 2],
        };
        let s = format!("{:?}", op);
        assert!(s.contains("my_custom"));
    }

    // ── Param struct field checks ─────────────────────────────────────────

    #[test]
    fn conv2d_params_fields() {
        let p = Conv2dParams {
            kernel_size: [5, 5],
            stride: [2, 2],
            padding: [2, 2],
            dilation: [1, 1],
            groups: 32,
        };
        assert_eq!(p.kernel_size, [5, 5]);
        assert_eq!(p.stride, [2, 2]);
        assert_eq!(p.padding, [2, 2]);
        assert_eq!(p.dilation, [1, 1]);
        assert_eq!(p.groups, 32);
    }

    #[test]
    fn pool_params_fields() {
        let p = PoolParams {
            kernel_size: [3, 3],
            stride: [2, 2],
            padding: [1, 1],
        };
        assert_eq!(p.kernel_size, [3, 3]);
        assert_eq!(p.stride, [2, 2]);
        assert_eq!(p.padding, [1, 1]);
    }

    #[test]
    fn linear_params_no_bias() {
        let p = LinearParams {
            in_features: 100,
            out_features: 50,
            bias: false,
        };
        assert!(!p.bias);
    }

    #[test]
    fn matmul_params_both_transpose() {
        let p = MatMulParams {
            transpose_a: true,
            transpose_b: true,
        };
        assert!(p.transpose_a);
        assert!(p.transpose_b);
    }

    #[test]
    fn softmax_negative_axis() {
        let p = SoftmaxParams { axis: -1 };
        assert_eq!(p.axis, -1);
    }

    #[test]
    fn batchnorm_none_momentum() {
        let p = BatchNormParams {
            num_features: 128,
            eps: 1e-5,
            momentum: None,
        };
        assert!(p.momentum.is_none());
    }

    #[test]
    fn batchnorm_some_momentum() {
        let p = BatchNormParams {
            num_features: 128,
            eps: 1e-5,
            momentum: Some(0.1),
        };
        assert_eq!(p.momentum, Some(0.1));
    }

    #[test]
    fn layernorm_params() {
        let p = LayerNormParams {
            normalized_shape: vec![512, 256],
            eps: 1e-12,
        };
        assert_eq!(p.normalized_shape, vec![512, 256]);
        assert_eq!(p.eps, 1e-12);
    }

    #[test]
    fn reshape_params_with_dynamic_dim() {
        let p = ReshapeParams {
            target_shape: Shape::new(vec![Dim::fixed(8).unwrap(), Dim::Dynamic]).unwrap(),
        };
        assert_eq!(p.target_shape.dims().len(), 2);
        assert!(matches!(p.target_shape.dims()[0], Dim::Fixed(8)));
        assert!(matches!(p.target_shape.dims()[1], Dim::Dynamic));
    }

    #[test]
    fn transpose_params_identity_permutation() {
        let p = TransposeParams {
            perm: vec![0, 1, 2, 3],
        };
        assert_eq!(p.perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn concat_negative_axis() {
        let p = ConcatParams { axis: -1 };
        assert_eq!(p.axis, -1);
    }

    #[test]
    fn flatten_params_full_range() {
        let p = FlattenParams {
            start_dim: 0,
            end_dim: -1,
        };
        assert_eq!(p.start_dim, 0);
        assert_eq!(p.end_dim, -1);
    }

    #[test]
    fn dropout_params_p_value() {
        let p = DropoutParams { p: 0.3 };
        assert_eq!(p.p, 0.3);
    }

    // ── Equality ─────────────────────────────────────────────────────────

    #[test]
    fn eq_two_identical_conv2d() {
        let a = MlOp::Conv2d(Conv2dParams {
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        });
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_ops() {
        assert_ne!(MlOp::Relu, MlOp::Sigmoid);
    }

    #[test]
    fn ne_different_custom_names() {
        let a = MlOp::Custom {
            name: "op_a".into(),
            params: vec![],
        };
        let b = MlOp::Custom {
            name: "op_b".into(),
            params: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn eq_batchnorm_none_vs_some_momentum_differ() {
        let a = MlOp::BatchNorm(BatchNormParams {
            num_features: 64,
            eps: 1e-5,
            momentum: None,
        });
        let b = MlOp::BatchNorm(BatchNormParams {
            num_features: 64,
            eps: 1e-5,
            momentum: Some(0.1),
        });
        assert_ne!(a, b);
    }

    // ── MlOpError Display ────────────────────────────────────────────────

    #[test]
    fn error_display_zero_spatial_param() {
        let e = MlOpError::ZeroSpatialParam {
            param: "kernel_size".to_string(),
        };
        assert_eq!(e.to_string(), "kernel_size must be > 0, got 0");
    }

    #[test]
    fn error_display_zero_groups() {
        assert_eq!(
            MlOpError::ZeroGroups.to_string(),
            "groups must be > 0, got 0"
        );
    }

    #[test]
    fn error_display_zero_features() {
        let e = MlOpError::ZeroFeatures {
            param: "in_features".to_string(),
        };
        assert_eq!(e.to_string(), "in_features must be > 0, got 0");
    }

    #[test]
    fn error_display_zero_num_features() {
        assert_eq!(
            MlOpError::ZeroNumFeatures.to_string(),
            "num_features must be > 0, got 0"
        );
    }

    #[test]
    fn error_display_non_positive_eps() {
        let e = MlOpError::NonPositiveEps(-1.0);
        assert_eq!(e.to_string(), "eps must be > 0, got -1");
    }

    #[test]
    fn error_display_invalid_momentum() {
        let e = MlOpError::InvalidMomentum(1.5);
        assert_eq!(e.to_string(), "momentum must be in [0.0, 1.0), got 1.5");
    }

    #[test]
    fn error_display_invalid_dropout_p() {
        let e = MlOpError::InvalidDropoutP(-0.1);
        assert_eq!(e.to_string(), "dropout p must be in [0.0, 1.0), got -0.1");
    }

    #[test]
    fn error_display_invalid_normalized_shape() {
        assert_eq!(
            MlOpError::InvalidNormalizedShape.to_string(),
            "normalized_shape must be non-empty with all values > 0"
        );
    }

    #[test]
    fn error_display_invalid_permutation() {
        let e = MlOpError::InvalidPermutation {
            perm: vec![0, 0],
            expected_len: 2,
        };
        assert_eq!(
            e.to_string(),
            "perm must be a permutation of 0..2, got [0, 0]"
        );
    }

    #[test]
    fn error_display_empty_custom_name() {
        assert_eq!(
            MlOpError::EmptyCustomName.to_string(),
            "Custom op name must not be empty"
        );
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = MlOpError::ZeroGroups;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_debug_format() {
        let e = MlOpError::ZeroGroups;
        let s = format!("{:?}", e);
        assert!(s.contains("ZeroGroups"));
    }

    // ── Conv2dParams::new() ──────────────────────────────────────────────

    #[test]
    fn conv2d_new_valid() {
        let p = Conv2dParams::new([3, 3], [1, 1], [1, 1], [1, 1], 1).unwrap();
        assert_eq!(p.kernel_size, [3, 3]);
        assert_eq!(p.stride, [1, 1]);
        assert_eq!(p.padding, [1, 1]);
        assert_eq!(p.dilation, [1, 1]);
        assert_eq!(p.groups, 1);
    }

    #[test]
    fn conv2d_new_valid_zero_padding() {
        let p = Conv2dParams::new([5, 5], [2, 2], [0, 0], [1, 1], 32).unwrap();
        assert_eq!(p.padding, [0, 0]);
        assert_eq!(p.groups, 32);
    }

    #[test]
    fn conv2d_new_valid_asymmetric() {
        let p = Conv2dParams::new([3, 5], [1, 2], [1, 2], [1, 3], 4).unwrap();
        assert_eq!(p.kernel_size, [3, 5]);
        assert_eq!(p.stride, [1, 2]);
        assert_eq!(p.dilation, [1, 3]);
    }

    #[test]
    fn conv2d_new_zero_kernel_h() {
        let err = Conv2dParams::new([0, 3], [1, 1], [0, 0], [1, 1], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_kernel_w() {
        let err = Conv2dParams::new([3, 0], [1, 1], [0, 0], [1, 1], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_stride_h() {
        let err = Conv2dParams::new([3, 3], [0, 1], [0, 0], [1, 1], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_stride_w() {
        let err = Conv2dParams::new([3, 3], [1, 0], [0, 0], [1, 1], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_dilation_h() {
        let err = Conv2dParams::new([3, 3], [1, 1], [0, 0], [0, 1], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "dilation".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_dilation_w() {
        let err = Conv2dParams::new([3, 3], [1, 1], [0, 0], [1, 0], 1).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "dilation".to_string(),
            }
        );
    }

    #[test]
    fn conv2d_new_zero_groups() {
        let err = Conv2dParams::new([3, 3], [1, 1], [0, 0], [1, 1], 0).unwrap_err();
        assert_eq!(err, MlOpError::ZeroGroups);
    }

    // ── PoolParams::new() ────────────────────────────────────────────────

    #[test]
    fn pool_new_valid() {
        let p = PoolParams::new([2, 2], [2, 2], [0, 0]).unwrap();
        assert_eq!(p.kernel_size, [2, 2]);
        assert_eq!(p.stride, [2, 2]);
        assert_eq!(p.padding, [0, 0]);
    }

    #[test]
    fn pool_new_valid_with_padding() {
        let p = PoolParams::new([3, 3], [1, 1], [1, 1]).unwrap();
        assert_eq!(p.padding, [1, 1]);
    }

    #[test]
    fn pool_new_zero_kernel_h() {
        let err = PoolParams::new([0, 2], [2, 2], [0, 0]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            }
        );
    }

    #[test]
    fn pool_new_zero_kernel_w() {
        let err = PoolParams::new([2, 0], [2, 2], [0, 0]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "kernel_size".to_string(),
            }
        );
    }

    #[test]
    fn pool_new_zero_stride_h() {
        let err = PoolParams::new([2, 2], [0, 2], [0, 0]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            }
        );
    }

    #[test]
    fn pool_new_zero_stride_w() {
        let err = PoolParams::new([2, 2], [2, 0], [0, 0]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroSpatialParam {
                param: "stride".to_string(),
            }
        );
    }

    // ── LinearParams::new() ──────────────────────────────────────────────

    #[test]
    fn linear_new_valid_with_bias() {
        let p = LinearParams::new(512, 256, true).unwrap();
        assert_eq!(p.in_features, 512);
        assert_eq!(p.out_features, 256);
        assert!(p.bias);
    }

    #[test]
    fn linear_new_valid_without_bias() {
        let p = LinearParams::new(100, 50, false).unwrap();
        assert!(!p.bias);
    }

    #[test]
    fn linear_new_zero_in_features() {
        let err = LinearParams::new(0, 256, true).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroFeatures {
                param: "in_features".to_string(),
            }
        );
    }

    #[test]
    fn linear_new_zero_out_features() {
        let err = LinearParams::new(512, 0, true).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroFeatures {
                param: "out_features".to_string(),
            }
        );
    }

    #[test]
    fn linear_new_both_zero() {
        // First check hit is in_features
        let err = LinearParams::new(0, 0, false).unwrap_err();
        assert_eq!(
            err,
            MlOpError::ZeroFeatures {
                param: "in_features".to_string(),
            }
        );
    }

    // ── BatchNormParams::new() ───────────────────────────────────────────

    #[test]
    fn batchnorm_new_valid_with_momentum() {
        let p = BatchNormParams::new(64, 1e-5, Some(0.1)).unwrap();
        assert_eq!(p.num_features, 64);
        assert_eq!(p.eps, 1e-5);
        assert_eq!(p.momentum, Some(0.1));
    }

    #[test]
    fn batchnorm_new_valid_no_momentum() {
        let p = BatchNormParams::new(128, 1e-5, None).unwrap();
        assert!(p.momentum.is_none());
    }

    #[test]
    fn batchnorm_new_valid_zero_momentum() {
        let p = BatchNormParams::new(32, 1e-5, Some(0.0)).unwrap();
        assert_eq!(p.momentum, Some(0.0));
    }

    #[test]
    fn batchnorm_new_zero_num_features() {
        let err = BatchNormParams::new(0, 1e-5, None).unwrap_err();
        assert_eq!(err, MlOpError::ZeroNumFeatures);
    }

    #[test]
    fn batchnorm_new_zero_eps() {
        let err = BatchNormParams::new(64, 0.0, None).unwrap_err();
        assert_eq!(err, MlOpError::NonPositiveEps(0.0));
    }

    #[test]
    fn batchnorm_new_negative_eps() {
        let err = BatchNormParams::new(64, -1e-5, None).unwrap_err();
        assert_eq!(err, MlOpError::NonPositiveEps(-1e-5));
    }

    #[test]
    fn batchnorm_new_momentum_one() {
        let err = BatchNormParams::new(64, 1e-5, Some(1.0)).unwrap_err();
        assert_eq!(err, MlOpError::InvalidMomentum(1.0));
    }

    #[test]
    fn batchnorm_new_momentum_negative() {
        let err = BatchNormParams::new(64, 1e-5, Some(-0.1)).unwrap_err();
        assert_eq!(err, MlOpError::InvalidMomentum(-0.1));
    }

    #[test]
    fn batchnorm_new_momentum_greater_than_one() {
        let err = BatchNormParams::new(64, 1e-5, Some(1.5)).unwrap_err();
        assert_eq!(err, MlOpError::InvalidMomentum(1.5));
    }

    // ── LayerNormParams::new() ───────────────────────────────────────────

    #[test]
    fn layernorm_new_valid_single() {
        let p = LayerNormParams::new(vec![768], 1e-12).unwrap();
        assert_eq!(p.normalized_shape, vec![768]);
        assert_eq!(p.eps, 1e-12);
    }

    #[test]
    fn layernorm_new_valid_multi_dim() {
        let p = LayerNormParams::new(vec![512, 256], 1e-5).unwrap();
        assert_eq!(p.normalized_shape, vec![512, 256]);
    }

    #[test]
    fn layernorm_new_empty_shape() {
        let err = LayerNormParams::new(vec![], 1e-12).unwrap_err();
        assert_eq!(err, MlOpError::InvalidNormalizedShape);
    }

    #[test]
    fn layernorm_new_zero_in_shape() {
        let err = LayerNormParams::new(vec![768, 0], 1e-12).unwrap_err();
        assert_eq!(err, MlOpError::InvalidNormalizedShape);
    }

    #[test]
    fn layernorm_new_all_zeros_in_shape() {
        let err = LayerNormParams::new(vec![0], 1e-12).unwrap_err();
        assert_eq!(err, MlOpError::InvalidNormalizedShape);
    }

    #[test]
    fn layernorm_new_zero_eps() {
        let err = LayerNormParams::new(vec![768], 0.0).unwrap_err();
        assert_eq!(err, MlOpError::NonPositiveEps(0.0));
    }

    #[test]
    fn layernorm_new_negative_eps() {
        let err = LayerNormParams::new(vec![768], -1.0).unwrap_err();
        assert_eq!(err, MlOpError::NonPositiveEps(-1.0));
    }

    // ── DropoutParams::new() ─────────────────────────────────────────────

    #[test]
    fn dropout_new_valid_zero() {
        let p = DropoutParams::new(0.0).unwrap();
        assert_eq!(p.p, 0.0);
    }

    #[test]
    fn dropout_new_valid_half() {
        let p = DropoutParams::new(0.5).unwrap();
        assert_eq!(p.p, 0.5);
    }

    #[test]
    fn dropout_new_valid_near_one() {
        let p = DropoutParams::new(0.999).unwrap();
        assert_eq!(p.p, 0.999);
    }

    #[test]
    fn dropout_new_one_is_invalid() {
        let err = DropoutParams::new(1.0).unwrap_err();
        assert_eq!(err, MlOpError::InvalidDropoutP(1.0));
    }

    #[test]
    fn dropout_new_negative() {
        let err = DropoutParams::new(-0.1).unwrap_err();
        assert_eq!(err, MlOpError::InvalidDropoutP(-0.1));
    }

    #[test]
    fn dropout_new_greater_than_one() {
        let err = DropoutParams::new(1.5).unwrap_err();
        assert_eq!(err, MlOpError::InvalidDropoutP(1.5));
    }

    // ── TransposeParams::new() ───────────────────────────────────────────

    #[test]
    fn transpose_new_valid_swap() {
        let p = TransposeParams::new(vec![1, 0]).unwrap();
        assert_eq!(p.perm, vec![1, 0]);
    }

    #[test]
    fn transpose_new_valid_identity() {
        let p = TransposeParams::new(vec![0, 1, 2]).unwrap();
        assert_eq!(p.perm, vec![0, 1, 2]);
    }

    #[test]
    fn transpose_new_valid_4d() {
        let p = TransposeParams::new(vec![0, 2, 3, 1]).unwrap();
        assert_eq!(p.perm, vec![0, 2, 3, 1]);
    }

    #[test]
    fn transpose_new_valid_single() {
        let p = TransposeParams::new(vec![0]).unwrap();
        assert_eq!(p.perm, vec![0]);
    }

    #[test]
    fn transpose_new_empty() {
        let err = TransposeParams::new(vec![]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::InvalidPermutation {
                perm: vec![],
                expected_len: 0,
            }
        );
    }

    #[test]
    fn transpose_new_duplicate() {
        let err = TransposeParams::new(vec![0, 0]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::InvalidPermutation {
                perm: vec![0, 0],
                expected_len: 2,
            }
        );
    }

    #[test]
    fn transpose_new_out_of_range() {
        let err = TransposeParams::new(vec![0, 2]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::InvalidPermutation {
                perm: vec![0, 2],
                expected_len: 2,
            }
        );
    }

    #[test]
    fn transpose_new_missing_index() {
        // [0, 2, 2] — index 1 missing, index 2 duplicated
        let err = TransposeParams::new(vec![0, 2, 2]).unwrap_err();
        assert_eq!(
            err,
            MlOpError::InvalidPermutation {
                perm: vec![0, 2, 2],
                expected_len: 3,
            }
        );
    }

    // ── Infallible constructors ──────────────────────────────────────────

    #[test]
    fn matmul_new() {
        let p = MatMulParams::new(true, false);
        assert!(p.transpose_a);
        assert!(!p.transpose_b);
    }

    #[test]
    fn matmul_new_both_false() {
        let p = MatMulParams::new(false, false);
        assert!(!p.transpose_a);
        assert!(!p.transpose_b);
    }

    #[test]
    fn softmax_new() {
        let p = SoftmaxParams::new(-1);
        assert_eq!(p.axis, -1);
    }

    #[test]
    fn softmax_new_positive_axis() {
        let p = SoftmaxParams::new(2);
        assert_eq!(p.axis, 2);
    }

    #[test]
    fn concat_new() {
        let p = ConcatParams::new(1);
        assert_eq!(p.axis, 1);
    }

    #[test]
    fn concat_new_negative() {
        let p = ConcatParams::new(-2);
        assert_eq!(p.axis, -2);
    }

    #[test]
    fn flatten_new() {
        let p = FlattenParams::new(1, -1);
        assert_eq!(p.start_dim, 1);
        assert_eq!(p.end_dim, -1);
    }

    #[test]
    fn flatten_new_same_dim() {
        let p = FlattenParams::new(0, 0);
        assert_eq!(p.start_dim, 0);
        assert_eq!(p.end_dim, 0);
    }

    #[test]
    fn reshape_new() {
        let shape = Shape::scalar();
        let p = ReshapeParams::new(shape.clone());
        assert_eq!(p.target_shape, shape);
    }

    #[test]
    fn reshape_new_with_dims() {
        let shape = Shape::from_fixed(&[3, 4]).unwrap();
        let p = ReshapeParams::new(shape.clone());
        assert_eq!(p.target_shape, shape);
    }

    // ── MlOp::custom() ──────────────────────────────────────────────────

    #[test]
    fn custom_new_valid() {
        let op = MlOp::custom("my_op", vec![1, 2, 3]).unwrap();
        assert_eq!(op.name(), "my_op");
        assert!(op.is_custom());
    }

    #[test]
    fn custom_new_valid_empty_params() {
        let op = MlOp::custom("simple_op", vec![]).unwrap();
        assert_eq!(op.name(), "simple_op");
    }

    #[test]
    fn custom_new_empty_name() {
        let err = MlOp::custom("", vec![]).unwrap_err();
        assert_eq!(err, MlOpError::EmptyCustomName);
    }

    #[test]
    fn custom_new_accepts_string() {
        let name = String::from("dynamic_name");
        let op = MlOp::custom(name, vec![]).unwrap();
        assert_eq!(op.name(), "dynamic_name");
    }

    // ── Constructors produce correct MlOp variants ───────────────────────

    #[test]
    fn conv2d_new_wraps_in_mlop() {
        let p = Conv2dParams::new([3, 3], [1, 1], [0, 0], [1, 1], 1).unwrap();
        let op = MlOp::Conv2d(p);
        assert_eq!(op.name(), "Conv2d");
        assert!(op.is_spatial_2d());
    }

    #[test]
    fn pool_new_wraps_in_maxpool() {
        let p = PoolParams::new([2, 2], [2, 2], [0, 0]).unwrap();
        let op = MlOp::MaxPool2d(p);
        assert_eq!(op.name(), "MaxPool2d");
        assert!(op.is_spatial_2d());
    }

    #[test]
    fn pool_new_wraps_in_avgpool() {
        let p = PoolParams::new([2, 2], [2, 2], [0, 0]).unwrap();
        let op = MlOp::AvgPool2d(p);
        assert_eq!(op.name(), "AvgPool2d");
    }

    #[test]
    fn linear_new_wraps_in_mlop() {
        let p = LinearParams::new(784, 128, true).unwrap();
        let op = MlOp::Linear(p);
        assert_eq!(op.name(), "Linear");
        assert!(!op.is_parameterless());
    }

    #[test]
    fn batchnorm_new_wraps_in_mlop() {
        let p = BatchNormParams::new(64, 1e-5, Some(0.1)).unwrap();
        let op = MlOp::BatchNorm(p);
        assert_eq!(op.name(), "BatchNorm");
    }

    #[test]
    fn layernorm_new_wraps_in_mlop() {
        let p = LayerNormParams::new(vec![768], 1e-12).unwrap();
        let op = MlOp::LayerNorm(p);
        assert_eq!(op.name(), "LayerNorm");
    }

    #[test]
    fn dropout_new_wraps_in_mlop() {
        let p = DropoutParams::new(0.5).unwrap();
        let op = MlOp::Dropout(p);
        assert_eq!(op.name(), "Dropout");
        assert!(!op.is_parameterless());
    }

    #[test]
    fn transpose_new_wraps_in_mlop() {
        let p = TransposeParams::new(vec![1, 0, 2]).unwrap();
        let op = MlOp::Transpose(p);
        assert_eq!(op.name(), "Transpose");
    }
}
