use std::any::Any;
use std::fmt;

use thiserror::Error;

use crate::types::TensorType;

pub mod builder;
pub mod edge;
pub mod node;
pub mod validate;

pub use builder::GraphBuilder;
pub use edge::{Edge, EdgeSource, PortRef};
pub use node::{Node, NodeId, NodeKind};

// ── KernelDescriptor ──────────────────────────────────────────────────────────

/// Backend-specific description of a compute kernel to execute.
///
/// Each backend defines its own concrete descriptor struct (e.g.
/// `CudaKernelDesc`) and downcasts from `&dyn KernelDescriptor` inside its
/// `dispatch_compute` implementation. The [`Any`] supertrait bound ensures the
/// concrete type is `'static`, which is required for safe downcasting.
///
/// # Example
///
/// ```rust
/// use std::any::Any;
/// use graph_core::graph::KernelDescriptor;
///
/// struct MyKernel { ptx: String }
///
/// impl KernelDescriptor for MyKernel {
///     fn as_any(&self) -> &dyn Any { self }
/// }
///
/// let k: Box<dyn KernelDescriptor> = Box::new(MyKernel { ptx: "...".into() });
/// assert!(k.as_any().downcast_ref::<MyKernel>().is_some());
/// ```
pub trait KernelDescriptor: Any + Send + Sync {
    /// Returns `self` as `&dyn Any` to enable backend-internal downcasting.
    fn as_any(&self) -> &dyn Any;
}

// ── Boundary ports ────────────────────────────────────────────────────────────

/// A named external input port on a [`Graph`].
///
/// Sources represent data that flows into the graph from the outside world
/// (e.g. a microphone buffer, a camera frame, a host tensor).
#[derive(Clone, Debug)]
pub struct SourcePort {
    /// Human-readable name used to reference this source in the builder.
    pub name: String,
    /// Expected tensor type for data arriving at this port.
    pub tensor_type: TensorType,
}

/// A named external output port on a [`Graph`].
///
/// Sinks represent data that the graph produces for the outside world
/// (e.g. a rendered frame, a classification score, a control signal).
#[derive(Clone, Debug)]
pub struct SinkPort {
    /// Human-readable name used to reference this sink in the builder.
    pub name: String,
    /// Expected tensor type for data leaving at this port.
    pub tensor_type: TensorType,
}

/// Wires a node output port to a graph sink.
#[derive(Clone, Debug)]
pub struct SinkConnection {
    /// The node output port that produces the data.
    pub from: PortRef,
    /// Index into [`Graph::sinks`].
    pub sink: usize,
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// An immutable, validated directed acyclic computation graph.
///
/// A `Graph` represents a typed function:
/// `(Source₁, Source₂, …) → (Sink₁, Sink₂, …)`.
///
/// Graphs are constructed via [`GraphBuilder`] and are immutable after
/// [`GraphBuilder::build`] succeeds. All accessors return shared references
/// to the internal data.
///
/// # Example
///
/// ```rust
/// use graph_core::graph::GraphBuilder;
/// use graph_core::ops::Op;
/// use graph_core::types::{dim::Dim, DType, Layout, TensorType};
///
/// let t = TensorType::new(DType::F32, vec![Dim::Fixed(1), Dim::Fixed(8)], Layout::RowMajor).unwrap();
///
/// let graph = GraphBuilder::new()
///     .source("audio", t.clone())
///     .add_node("relu")
///         .device("cpu")
///         .op(Op::Relu)
///         .input_from_source("audio")
///         .output(t.clone())
///         .done()
///     .sink("out", t.clone())
///         .from("relu", 0)
///         .done()
///     .build()
///     .unwrap();
///
/// assert_eq!(graph.node_count(), 1);
/// assert_eq!(graph.sources()[0].name, "audio");
/// assert_eq!(graph.sinks()[0].name, "out");
/// ```
pub struct Graph {
    pub(crate) sources: Vec<SourcePort>,
    pub(crate) sinks: Vec<SinkPort>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
    pub(crate) sink_connections: Vec<SinkConnection>,
}

impl Graph {
    /// The external input boundary ports, in declaration order.
    pub fn sources(&self) -> &[SourcePort] {
        &self.sources
    }

    /// The external output boundary ports, in declaration order.
    pub fn sinks(&self) -> &[SinkPort] {
        &self.sinks
    }

    /// All computation nodes, in insertion order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// All data edges connecting sources and nodes.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Connections from node output ports to graph sinks.
    pub fn sink_connections(&self) -> &[SinkConnection] {
        &self.sink_connections
    }

    /// Total number of computation nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Find a node by its human-readable name.
    ///
    /// Returns `None` if no node with that name exists.
    pub fn find_node(&self, name: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.name == name)
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field(
                "sources",
                &self.sources.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field(
                "sinks",
                &self.sinks.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field("node_count", &self.nodes.len())
            .field("edge_count", &self.edges.len())
            .finish()
    }
}

// ── GraphError ────────────────────────────────────────────────────────────────

/// Errors produced during [`GraphBuilder::build`] validation.
///
/// Multiple errors may be returned in a single `Vec<GraphError>` so the caller
/// can fix all issues in one round.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GraphError {
    /// The graph contains a directed cycle.
    #[error("Cycle detected in graph")]
    Cycle,

    /// An edge connects tensors with incompatible types.
    #[error("Type mismatch on edge to node '{node}' port {port}: {reason}")]
    TypeMismatch {
        /// Name of the destination node.
        node: String,
        /// Zero-based port index on the destination node.
        port: usize,
        /// Human-readable explanation of the mismatch.
        reason: String,
    },

    /// A node input port has no incoming edge.
    #[error("Node '{node}' input port {port} is not connected")]
    UnconnectedPort {
        /// Name of the node.
        node: String,
        /// Zero-based port index.
        port: usize,
    },

    /// A sink has no connection from any node output.
    #[error("Sink '{sink}' has no connection")]
    UnconnectedSink {
        /// Name of the sink.
        sink: String,
    },

    /// A source is declared but never referenced by any edge.
    #[error("Source '{source_name}' is never used")]
    UnusedSource {
        /// Name of the source.
        source_name: String,
    },

    /// A node's device ID is the empty string.
    #[error("Node '{node}' has an empty device ID")]
    EmptyDevice {
        /// Name of the node.
        node: String,
    },

    /// Two nodes share the same name.
    #[error("Duplicate node name '{name}'")]
    DuplicateNodeName {
        /// The duplicated name.
        name: String,
    },

    /// Two sources share the same name.
    #[error("Duplicate source name '{name}'")]
    DuplicateSourceName {
        /// The duplicated name.
        name: String,
    },

    /// Two sinks share the same name.
    #[error("Duplicate sink name '{name}'")]
    DuplicateSinkName {
        /// The duplicated name.
        name: String,
    },

    /// An edge references a source name that was never declared.
    #[error("Unknown source '{name}' referenced in node '{node}'")]
    UnknownSource {
        /// The unknown source name.
        name: String,
        /// The node that referenced it.
        node: String,
    },

    /// An edge references a node that was never declared.
    #[error("Unknown node '{name}'")]
    UnknownNode {
        /// The unknown node identifier.
        name: String,
    },

    /// An edge or sink connection references a port index that is out of range.
    #[error("Node '{node}' port {port} out of range (node has {count} ports)")]
    PortOutOfRange {
        /// Name of the node.
        node: String,
        /// The out-of-range port index.
        port: usize,
        /// How many ports the node actually has.
        count: usize,
    },

    /// A node declares no output ports.
    #[error("Node '{node}' has no output ports declared")]
    NoOutputs {
        /// Name of the node.
        node: String,
    },
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_error_cycle_display() {
        let e = GraphError::Cycle;
        assert_eq!(format!("{e}"), "Cycle detected in graph");
    }

    #[test]
    fn graph_error_type_mismatch_display() {
        let e = GraphError::TypeMismatch {
            node: "relu".into(),
            port: 0,
            reason: "F32 vs I32".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("relu"));
        assert!(s.contains('0'.to_string().as_str()));
        assert!(s.contains("F32 vs I32"));
    }

    #[test]
    fn graph_error_unconnected_port_display() {
        let e = GraphError::UnconnectedPort {
            node: "add".into(),
            port: 1,
        };
        let s = format!("{e}");
        assert!(s.contains("add"));
        assert!(s.contains('1'.to_string().as_str()));
    }

    #[test]
    fn graph_error_unconnected_sink_display() {
        let e = GraphError::UnconnectedSink {
            sink: "output".into(),
        };
        assert!(format!("{e}").contains("output"));
    }

    #[test]
    fn graph_error_unused_source_display() {
        let e = GraphError::UnusedSource {
            source_name: "mic".into(),
        };
        assert!(format!("{e}").contains("mic"));
    }

    #[test]
    fn graph_error_empty_device_display() {
        let e = GraphError::EmptyDevice {
            node: "conv".into(),
        };
        assert!(format!("{e}").contains("conv"));
    }

    #[test]
    fn graph_error_duplicate_names_display() {
        assert!(format!("{}", GraphError::DuplicateNodeName { name: "n".into() }).contains('n'));
        assert!(format!("{}", GraphError::DuplicateSourceName { name: "s".into() }).contains('s'));
        assert!(format!("{}", GraphError::DuplicateSinkName { name: "k".into() }).contains('k'));
    }

    #[test]
    fn graph_error_unknown_node_display() {
        let e = GraphError::UnknownNode {
            name: "ghost".into(),
        };
        assert!(format!("{e}").contains("ghost"));
    }

    #[test]
    fn graph_error_port_out_of_range_display() {
        let e = GraphError::PortOutOfRange {
            node: "n".into(),
            port: 3,
            count: 1,
        };
        let s = format!("{e}");
        assert!(s.contains('3'.to_string().as_str()));
        assert!(s.contains('1'.to_string().as_str()));
    }

    #[test]
    fn graph_error_no_outputs_display() {
        let e = GraphError::NoOutputs {
            node: "sink_node".into(),
        };
        assert!(format!("{e}").contains("sink_node"));
    }

    #[test]
    fn graph_error_clone_and_eq() {
        let e = GraphError::Cycle;
        assert_eq!(e.clone(), e);
        let e2 = GraphError::EmptyDevice { node: "x".into() };
        assert_ne!(e, e2);
    }

    #[test]
    fn graph_error_implements_std_error() {
        let e = GraphError::Cycle;
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn kernel_descriptor_downcast() {
        use std::any::Any;

        struct Desc;
        impl KernelDescriptor for Desc {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let d: Box<dyn KernelDescriptor> = Box::new(Desc);
        assert!(d.as_any().downcast_ref::<Desc>().is_some());
    }

    #[test]
    fn source_port_clone() {
        use crate::types::{dim::Dim, DType, Layout};
        let t = TensorType::new(DType::F32, vec![Dim::Fixed(1)], Layout::RowMajor).unwrap();
        let s = SourcePort {
            name: "x".into(),
            tensor_type: t,
        };
        let c = s.clone();
        assert_eq!(c.name, "x");
    }

    #[test]
    fn sink_port_clone() {
        use crate::types::{dim::Dim, DType, Layout};
        let t = TensorType::new(DType::F32, vec![Dim::Fixed(1)], Layout::RowMajor).unwrap();
        let s = SinkPort {
            name: "y".into(),
            tensor_type: t,
        };
        let c = s.clone();
        assert_eq!(c.name, "y");
    }

    #[test]
    fn graph_debug_format() {
        use crate::graph::GraphBuilder;
        use crate::ops::Op;
        use crate::types::{dim::Dim, DType, Layout};

        let t = TensorType::new(DType::F32, vec![Dim::Fixed(1)], Layout::RowMajor).unwrap();
        let g = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("n")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("n", 0)
            .done()
            .build()
            .unwrap();

        let s = format!("{g:?}");
        assert!(s.contains("Graph"));
        assert!(s.contains("in"));
        assert!(s.contains("out"));
    }
}
