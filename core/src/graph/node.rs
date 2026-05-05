use std::fmt;

use crate::ops::Op;
use crate::types::{DeviceId, TensorType};

use super::KernelDescriptor;

// ── NodeId ────────────────────────────────────────────────────────────────────

/// Unique, stable identifier for a node within a [`Graph`](super::Graph).
///
/// Indices are assigned in insertion order by the builder and remain stable
/// after the graph is constructed.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(pub(crate) usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node_{}", self.0)
    }
}

// ── NodeKind ──────────────────────────────────────────────────────────────────

/// What computation a node performs.
///
/// - [`NodeKind::Op`] — a primitive from the curated [`Op`] catalog.
/// - [`NodeKind::Compute`] — an opaque kernel described by a
///   [`KernelDescriptor`] (e.g. a CUDA PTX function).
pub enum NodeKind {
    /// Primitive operation from the curated `Op` catalog.
    Op(Op),
    /// Raw compute kernel (CUDA PTX, SPIR-V, native Rust fn, …).
    Compute(Box<dyn KernelDescriptor>),
}

impl fmt::Debug for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Op(op) => write!(f, "Op({op:?})"),
            NodeKind::Compute(_) => write!(f, "Compute(<dyn KernelDescriptor>)"),
        }
    }
}

// ── Node ──────────────────────────────────────────────────────────────────────

/// A single computation node in a [`Graph`](super::Graph).
///
/// Nodes are immutable after the graph is built. All fields are accessible
/// through read-only accessor methods.
pub struct Node {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    pub(crate) device: DeviceId,
    pub(crate) kind: NodeKind,
    /// Expected [`TensorType`] for each input port (in order).
    pub(crate) inputs: Vec<TensorType>,
    /// Declared [`TensorType`] for each output port (in order).
    pub(crate) outputs: Vec<TensorType>,
    /// Whether this node holds mutable state across invocations (e.g. RNN cell).
    pub(crate) stateful: bool,
}

impl Node {
    /// Unique identifier for this node within its graph.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Human-readable name assigned at build time.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The device this node is scheduled to run on.
    pub fn device(&self) -> &DeviceId {
        &self.device
    }

    /// The kind of computation this node performs.
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    /// Expected tensor types for each input port, in port order.
    pub fn inputs(&self) -> &[TensorType] {
        &self.inputs
    }

    /// Declared tensor types for each output port, in port order.
    pub fn outputs(&self) -> &[TensorType] {
        &self.outputs
    }

    /// Returns `true` if this node holds mutable state across invocations.
    pub fn stateful(&self) -> bool {
        self.stateful
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("device", &self.device)
            .field("kind", &self.kind)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("stateful", &self.stateful)
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::ops::Op;
    use crate::types::{dim::Dim, DType, DeviceId, Layout, TensorType};

    use super::super::KernelDescriptor;
    use super::*;

    struct MockDesc;
    impl KernelDescriptor for MockDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn t() -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(1)], Layout::RowMajor).unwrap()
    }

    fn make_node_op() -> Node {
        Node {
            id: NodeId(0),
            name: "relu".to_string(),
            device: DeviceId::new("cpu"),
            kind: NodeKind::Op(Op::Relu),
            inputs: vec![t()],
            outputs: vec![t()],
            stateful: false,
        }
    }

    fn make_node_compute() -> Node {
        Node {
            id: NodeId(1),
            name: "custom".to_string(),
            device: DeviceId::new("cuda:0"),
            kind: NodeKind::Compute(Box::new(MockDesc)),
            inputs: vec![],
            outputs: vec![t()],
            stateful: true,
        }
    }

    #[test]
    fn node_id_display() {
        assert_eq!(format!("{}", NodeId(42)), "node_42");
    }

    #[test]
    fn node_id_equality() {
        assert_eq!(NodeId(0), NodeId(0));
        assert_ne!(NodeId(0), NodeId(1));
    }

    #[test]
    fn node_id_copy() {
        let a = NodeId(5);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn node_accessors_op() {
        let n = make_node_op();
        assert_eq!(n.id(), NodeId(0));
        assert_eq!(n.name(), "relu");
        assert_eq!(n.device(), &DeviceId::new("cpu"));
        assert!(!n.stateful());
        assert_eq!(n.inputs().len(), 1);
        assert_eq!(n.outputs().len(), 1);
        assert!(matches!(n.kind(), NodeKind::Op(Op::Relu)));
    }

    #[test]
    fn node_accessors_compute() {
        let n = make_node_compute();
        assert_eq!(n.id(), NodeId(1));
        assert_eq!(n.name(), "custom");
        assert_eq!(n.device(), &DeviceId::new("cuda:0"));
        assert!(n.stateful());
        assert!(matches!(n.kind(), NodeKind::Compute(_)));
    }

    #[test]
    fn node_kind_debug_op() {
        let kind = NodeKind::Op(Op::Relu);
        let s = format!("{kind:?}");
        assert!(s.contains("Op"));
        assert!(s.contains("Relu"));
    }

    #[test]
    fn node_kind_debug_compute() {
        let kind = NodeKind::Compute(Box::new(MockDesc));
        let s = format!("{kind:?}");
        assert!(s.contains("Compute"));
    }

    #[test]
    fn node_debug_format() {
        let n = make_node_op();
        let s = format!("{n:?}");
        assert!(s.contains("relu"));
        assert!(s.contains("cpu"));
    }
}
