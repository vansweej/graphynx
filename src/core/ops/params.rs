use crate::core::types::shape::Shape;

use super::MlOpError;

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
