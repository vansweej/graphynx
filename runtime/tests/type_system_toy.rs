use graph_core::ops::{Conv2dParams, LinearParams, MlOp};
use graph_core::types::dtype::DType;
use graph_core::types::shape::Shape;
use graph_core::types::{Dim, Layout, TensorType};

#[test]
fn toy_dtype_shape_and_tensor_type_work_together() {
    let dtype = DType::F32;
    let shape = Shape::new(vec![
        Dim::symbolic("batch").unwrap(),
        Dim::fixed(3).unwrap(),
        Dim::fixed(224).unwrap(),
        Dim::fixed(224).unwrap(),
    ])
    .unwrap();

    let tensor = TensorType::new(dtype, shape.dims().to_vec(), Layout::NCHW).unwrap();

    assert_eq!(tensor.dtype(), DType::F32);
    assert_eq!(tensor.rank(), 4);
    assert_eq!(tensor.layout(), Layout::NCHW);
    assert_eq!(tensor.num_elements(), None);
}

#[test]
fn toy_tensor_types_can_be_checked_for_compatibility() {
    let produced = TensorType::new(
        DType::F32,
        vec![Dim::symbolic("batch").unwrap(), Dim::fixed(128).unwrap()],
        Layout::RowMajor,
    )
    .unwrap();
    let consumed = TensorType::new(
        DType::F32,
        vec![Dim::Dynamic, Dim::fixed(128).unwrap()],
        Layout::Any,
    )
    .unwrap();

    assert!(produced.is_compatible_with(&consumed));
    assert!(consumed.is_compatible_with(&produced));
}

#[test]
fn toy_shape_supports_broadcast_and_reshape_checks() {
    let batch_features = Shape::new(vec![
        Dim::symbolic("batch").unwrap(),
        Dim::fixed(1).unwrap(),
    ])
    .unwrap();
    let classes = Shape::from_fixed(&[1, 10]).unwrap();
    let broadcast = batch_features.broadcast_with(&classes).unwrap();

    assert_eq!(broadcast.rank(), 2);
    assert_eq!(broadcast.dims()[1], Dim::Fixed(10));

    let flat = Shape::from_fixed(&[2, 3, 4]).unwrap();
    let matrix = Shape::matrix(4, 6).unwrap();
    assert!(flat.can_reshape_to(&matrix).is_ok());
}

#[test]
fn toy_ml_ops_can_be_constructed_as_user_facing_descriptors() {
    let conv = MlOp::Conv2d(Conv2dParams::new([3, 3], [1, 1], [1, 1], [1, 1], 1).unwrap());
    let linear = MlOp::Linear(LinearParams::new(128, 10, true).unwrap());
    let custom = MlOp::custom("toy.normalize", vec![1, 2, 3]).unwrap();

    assert_eq!(conv.name(), "Conv2d");
    assert_eq!(linear.name(), "Linear");
    assert!(custom.is_custom());
}
