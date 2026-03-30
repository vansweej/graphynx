//! Scratch binary — free to modify or replace.
//!
//! Run with:
//!   nix develop --command cargo run -p playground --bin scratch

use graph_core::ops::params::{Conv2dParams, MatMulParams};
use graph_core::ops::MlOp;
use graph_core::types::dim::Dim;
use graph_core::types::{DType, Layout, TensorType};

fn main() {
    // ── Tensor types ─────────────────────────────────────────────────────────
    let input = TensorType::new(
        DType::F32,
        vec![
            Dim::fixed(16).unwrap(),
            Dim::fixed(3).unwrap(),
            Dim::fixed(224).unwrap(),
            Dim::fixed(224).unwrap(),
        ],
        Layout::NCHW,
    )
    .expect("valid NCHW tensor");

    let scalar = TensorType::scalar(DType::F32);

    println!("input  : {input}");
    println!("scalar : {scalar}");
    println!("compatible: {}", input.is_compatible_with(&input));

    // ── ML ops ───────────────────────────────────────────────────────────────
    let conv = MlOp::Conv2d(
        Conv2dParams::new([3, 3], [1, 1], [1, 1], [1, 1], 1).expect("valid conv2d params"),
    );
    let matmul = MlOp::MatMul(MatMulParams {
        transpose_a: false,
        transpose_b: true,
    });

    println!("op: {conv}");
    println!("op: {}", matmul.name());
}
