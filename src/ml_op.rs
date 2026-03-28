use std::fmt;

use crate::tensor_type::Dim;

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

/// Parameters for softmax activation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SoftmaxParams {
    /// The axis along which softmax is computed. Negative values index from
    /// the end (e.g., `-1` is the last axis).
    pub axis: i32,
}

/// Parameters for a reshape operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReshapeParams {
    /// Target shape. May contain [`Dim::Dynamic`] or [`Dim::Symbolic`]
    /// entries to propagate runtime dimensions.
    pub target_shape: Vec<Dim>,
}

/// Parameters for a tensor transpose / permutation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransposeParams {
    /// Permutation of axes. Must be a permutation of `0..rank`.
    pub perm: Vec<usize>,
}

/// Parameters for concatenating tensors along an axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ConcatParams {
    /// Axis along which to concatenate. Negative values index from the end.
    pub axis: i32,
}

/// Parameters for flattening a range of dimensions into one.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FlattenParams {
    /// First dimension to flatten (inclusive). Negative values index from end.
    pub start_dim: i32,
    /// Last dimension to flatten (inclusive). Negative values index from end.
    pub end_dim: i32,
}

/// Parameters for dropout regularisation.
#[derive(Clone, Debug, PartialEq)]
pub struct DropoutParams {
    /// Probability of an element being zeroed. Must be in `[0.0, 1.0)`.
    pub p: f64,
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
            target_shape: vec![Dim::fixed(4).unwrap(), Dim::Dynamic],
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
            target_shape: vec![],
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
            target_shape: vec![Dim::fixed(8).unwrap(), Dim::Dynamic],
        };
        assert_eq!(p.target_shape.len(), 2);
        assert!(matches!(p.target_shape[0], Dim::Fixed(8)));
        assert!(matches!(p.target_shape[1], Dim::Dynamic));
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
}
