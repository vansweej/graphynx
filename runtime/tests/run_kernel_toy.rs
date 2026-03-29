mod common;

use backends::BackendError;

use common::{init_logger, DoubleKernelDesc, FailAllocBackend, MemBackend};

#[test]
fn toy_run_kernel_doubles_values_through_public_api() {
    init_logger();

    let backend = MemBackend::new();
    let desc = DoubleKernelDesc;
    let input = vec![1i32, 2, 3, 4];

    let output = runtime::run_kernel(&backend, &desc, &input).unwrap();

    assert_eq!(output, vec![2, 4, 6, 8]);
}

#[test]
fn toy_run_kernel_accepts_trait_objects() {
    init_logger();

    let backend = MemBackend::new();
    let backend_ref: &dyn backends::Backend = &backend;
    let desc: &dyn backends::KernelDescriptor = &DoubleKernelDesc;

    let output = runtime::run_kernel(backend_ref, desc, &[21i32]).unwrap();

    assert_eq!(output, vec![42]);
}

#[test]
fn toy_run_kernel_propagates_backend_errors() {
    init_logger();

    let backend = FailAllocBackend::new();
    let desc = DoubleKernelDesc;

    let error = runtime::run_kernel(&backend, &desc, &[1i32, 2, 3]).unwrap_err();

    assert!(matches!(error, BackendError::Buffer(_)));
}
