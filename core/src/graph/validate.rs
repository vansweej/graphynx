use std::collections::{HashMap, HashSet, VecDeque};

use crate::types::TensorType;

use super::edge::{Edge, EdgeSource};
use super::node::{Node, NodeId};
use super::{GraphError, SinkConnection, SinkPort, SourcePort};

// ── Internal intermediate representation used by the builder ──────────────────

/// Flattened view of the graph passed to each validation pass.
pub(crate) struct GraphSpec<'a> {
    pub sources: &'a [SourcePort],
    pub sinks: &'a [SinkPort],
    pub nodes: &'a [Node],
    pub edges: &'a [Edge],
    pub sink_connections: &'a [SinkConnection],
}

// ── Top-level entry point ─────────────────────────────────────────────────────

/// Run all validation passes in order and collect every error found.
///
/// Passes are ordered so that later passes can assume earlier invariants hold
/// (e.g. the DAG check is only meaningful after structural validity is
/// confirmed). However, all errors from all passes are collected and returned
/// together so the caller can fix multiple issues in one round.
///
/// # Errors
///
/// Returns a non-empty `Vec<GraphError>` if any validation fails.
pub(crate) fn validate(spec: &GraphSpec<'_>) -> Vec<GraphError> {
    let mut errors: Vec<GraphError> = Vec::new();

    pass_structural(spec, &mut errors);
    pass_completeness(spec, &mut errors);

    // Only run DAG and downstream passes if structural + completeness are clean,
    // to avoid misleading errors from broken references.
    if errors.is_empty() {
        pass_dag(spec, &mut errors);
        pass_type_compat(spec, &mut errors);
        pass_port_coverage(spec, &mut errors);
        pass_boundary_coverage(spec, &mut errors);
    }

    errors
}

// ── Pass 1: Structural ────────────────────────────────────────────────────────

fn pass_structural(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    // Duplicate source names
    let mut seen_sources: HashSet<&str> = HashSet::new();
    for s in spec.sources {
        if !seen_sources.insert(s.name.as_str()) {
            errors.push(GraphError::DuplicateSourceName {
                name: s.name.clone(),
            });
        }
    }

    // Duplicate sink names
    let mut seen_sinks: HashSet<&str> = HashSet::new();
    for s in spec.sinks {
        if !seen_sinks.insert(s.name.as_str()) {
            errors.push(GraphError::DuplicateSinkName {
                name: s.name.clone(),
            });
        }
    }

    // Duplicate node names
    let mut seen_nodes: HashSet<&str> = HashSet::new();
    for n in spec.nodes {
        if !seen_nodes.insert(n.name.as_str()) {
            errors.push(GraphError::DuplicateNodeName {
                name: n.name.clone(),
            });
        }
    }

    // Build lookup maps for reference checks
    let node_by_id: HashMap<NodeId, &Node> = spec.nodes.iter().map(|n| (n.id, n)).collect();

    // Validate every edge's source and destination
    for edge in spec.edges {
        // Validate destination
        match node_by_id.get(&edge.to.node) {
            None => {
                errors.push(GraphError::UnknownNode {
                    name: format!("{}", edge.to.node),
                });
                continue;
            }
            Some(dest_node) => {
                if edge.to.port >= dest_node.inputs.len() {
                    errors.push(GraphError::PortOutOfRange {
                        node: dest_node.name.clone(),
                        port: edge.to.port,
                        count: dest_node.inputs.len(),
                    });
                }
            }
        }

        // Validate source
        match &edge.from {
            EdgeSource::Source(idx) => {
                if *idx >= spec.sources.len() {
                    errors.push(GraphError::UnknownSource {
                        name: format!("source[{idx}]"),
                        node: format!("{}", edge.to.node),
                    });
                }
            }
            EdgeSource::Node(pr) => match node_by_id.get(&pr.node) {
                None => {
                    errors.push(GraphError::UnknownNode {
                        name: format!("{}", pr.node),
                    });
                }
                Some(src_node) => {
                    if pr.port >= src_node.outputs.len() {
                        errors.push(GraphError::PortOutOfRange {
                            node: src_node.name.clone(),
                            port: pr.port,
                            count: src_node.outputs.len(),
                        });
                    }
                }
            },
        }
    }

    // Validate sink connections
    for sc in spec.sink_connections {
        if sc.sink >= spec.sinks.len() {
            errors.push(GraphError::UnknownNode {
                name: format!("sink[{}]", sc.sink),
            });
            continue;
        }
        match node_by_id.get(&sc.from.node) {
            None => {
                errors.push(GraphError::UnknownNode {
                    name: format!("{}", sc.from.node),
                });
            }
            Some(src_node) => {
                if sc.from.port >= src_node.outputs.len() {
                    errors.push(GraphError::PortOutOfRange {
                        node: src_node.name.clone(),
                        port: sc.from.port,
                        count: src_node.outputs.len(),
                    });
                }
            }
        }
    }
}

// ── Pass 2: Completeness ──────────────────────────────────────────────────────

fn pass_completeness(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    for node in spec.nodes {
        // Every node must have a non-empty device ID
        if node.device.as_str().is_empty() {
            errors.push(GraphError::EmptyDevice {
                node: node.name.clone(),
            });
        }

        // Every node must have at least one output
        if node.outputs.is_empty() {
            errors.push(GraphError::NoOutputs {
                node: node.name.clone(),
            });
        }
    }
}

// ── Pass 3: DAG check (Kahn's algorithm) ─────────────────────────────────────

fn pass_dag(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    // Build adjacency: for each node, which nodes does it feed into?
    let n = spec.nodes.len();
    let id_to_idx: HashMap<NodeId, usize> = spec
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.id, i))
        .collect();

    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for edge in spec.edges {
        if let EdgeSource::Node(pr) = &edge.from {
            if let (Some(&src), Some(&dst)) =
                (id_to_idx.get(&pr.node), id_to_idx.get(&edge.to.node))
            {
                adj[src].push(dst);
                in_degree[dst] += 1;
            }
        }
    }

    // Kahn's BFS
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut visited = 0usize;
    while let Some(u) = queue.pop_front() {
        visited += 1;
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if visited < n {
        errors.push(GraphError::Cycle);
    }
}

// ── Pass 4: Type compatibility ────────────────────────────────────────────────

fn pass_type_compat(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    let node_by_id: HashMap<NodeId, &Node> = spec.nodes.iter().map(|n| (n.id, n)).collect();

    for edge in spec.edges {
        // Determine the source tensor type
        let src_type: Option<&TensorType> = match &edge.from {
            EdgeSource::Source(idx) => spec.sources.get(*idx).map(|s| &s.tensor_type),
            EdgeSource::Node(pr) => node_by_id
                .get(&pr.node)
                .and_then(|n| n.outputs.get(pr.port)),
        };

        // Determine the destination tensor type
        let dst_node = node_by_id.get(&edge.to.node);
        let dst_type: Option<&TensorType> = dst_node.and_then(|n| n.inputs.get(edge.to.port));

        if let (Some(src), Some(dst)) = (src_type, dst_type) {
            if !src.is_compatible_with(dst) {
                let node_name = dst_node
                    .map(|n| n.name.as_str())
                    .unwrap_or("<unknown>")
                    .to_string();
                errors.push(GraphError::TypeMismatch {
                    node: node_name,
                    port: edge.to.port,
                    reason: format!("source type {src:?} incompatible with dest type {dst:?}"),
                });
            }
        }
    }
}

// ── Pass 5: Port coverage ─────────────────────────────────────────────────────

fn pass_port_coverage(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    // Count how many edges connect to each (node, port) pair
    let mut coverage: HashMap<(NodeId, usize), usize> = HashMap::new();

    for edge in spec.edges {
        *coverage.entry((edge.to.node, edge.to.port)).or_insert(0) += 1;
    }

    for node in spec.nodes {
        for port in 0..node.inputs.len() {
            let count = coverage.get(&(node.id, port)).copied().unwrap_or(0);
            if count == 0 {
                errors.push(GraphError::UnconnectedPort {
                    node: node.name.clone(),
                    port,
                });
            }
        }
    }
}

// ── Pass 6: Boundary coverage ─────────────────────────────────────────────────

fn pass_boundary_coverage(spec: &GraphSpec<'_>, errors: &mut Vec<GraphError>) {
    // Every source must be used at least once
    let mut source_used = vec![false; spec.sources.len()];
    for edge in spec.edges {
        if let EdgeSource::Source(idx) = edge.from {
            if idx < source_used.len() {
                source_used[idx] = true;
            }
        }
    }
    for (i, used) in source_used.iter().enumerate() {
        if !used {
            errors.push(GraphError::UnusedSource {
                source_name: spec.sources[i].name.clone(),
            });
        }
    }

    // Every sink must have exactly one connection
    let mut sink_connection_count = vec![0usize; spec.sinks.len()];
    for sc in spec.sink_connections {
        if sc.sink < sink_connection_count.len() {
            sink_connection_count[sc.sink] += 1;
        }
    }
    for (i, count) in sink_connection_count.iter().enumerate() {
        if *count == 0 {
            errors.push(GraphError::UnconnectedSink {
                sink: spec.sinks[i].name.clone(),
            });
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::ops::Op;
    use crate::types::{dim::Dim, DType, DeviceId, Layout, TensorType};

    use super::super::edge::{Edge, EdgeSource, PortRef};
    use super::super::node::{Node, NodeId, NodeKind};
    use super::super::{GraphError, SinkConnection, SinkPort, SourcePort};
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn f32_type() -> TensorType {
        TensorType::new(
            DType::F32,
            vec![Dim::Fixed(1), Dim::Fixed(8)],
            Layout::RowMajor,
        )
        .unwrap()
    }

    fn i32_type() -> TensorType {
        TensorType::new(
            DType::I32,
            vec![Dim::Fixed(1), Dim::Fixed(8)],
            Layout::RowMajor,
        )
        .unwrap()
    }

    fn source(name: &str) -> SourcePort {
        SourcePort {
            name: name.to_string(),
            tensor_type: f32_type(),
        }
    }

    fn sink(name: &str) -> SinkPort {
        SinkPort {
            name: name.to_string(),
            tensor_type: f32_type(),
        }
    }

    fn simple_node(id: usize, name: &str, n_inputs: usize, n_outputs: usize) -> Node {
        Node {
            id: NodeId(id),
            name: name.to_string(),
            device: DeviceId::new("cpu"),
            kind: NodeKind::Op(Op::Relu),
            inputs: vec![f32_type(); n_inputs],
            outputs: vec![f32_type(); n_outputs],
            stateful: false,
        }
    }

    fn edge_from_source(src_idx: usize, to_node: usize, to_port: usize) -> Edge {
        Edge {
            from: EdgeSource::Source(src_idx),
            to: PortRef {
                node: NodeId(to_node),
                port: to_port,
            },
        }
    }

    fn edge_from_node(from_node: usize, from_port: usize, to_node: usize, to_port: usize) -> Edge {
        Edge {
            from: EdgeSource::Node(PortRef {
                node: NodeId(from_node),
                port: from_port,
            }),
            to: PortRef {
                node: NodeId(to_node),
                port: to_port,
            },
        }
    }

    fn sink_conn(from_node: usize, from_port: usize, sink_idx: usize) -> SinkConnection {
        SinkConnection {
            from: PortRef {
                node: NodeId(from_node),
                port: from_port,
            },
            sink: sink_idx,
        }
    }

    // ── Valid graph ───────────────────────────────────────────────────────

    #[test]
    fn valid_linear_graph_passes_all() {
        // source → node0 → sink
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "relu", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
    }

    #[test]
    fn valid_two_node_chain() {
        // source → node0 → node1 → sink
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "a", 1, 1), simple_node(1, "b", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0), edge_from_node(0, 0, 1, 0)];
        let sink_connections = vec![sink_conn(1, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        assert!(validate(&spec).is_empty());
    }

    // ── Structural errors ─────────────────────────────────────────────────

    #[test]
    fn duplicate_source_name() {
        let sources = vec![source("in"), source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::DuplicateSourceName { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn duplicate_sink_name() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out"), sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0), sink_conn(0, 0, 1)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::DuplicateSinkName { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn duplicate_node_name() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1), simple_node(1, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0), edge_from_node(0, 0, 1, 0)];
        let sink_connections = vec![sink_conn(1, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::DuplicateNodeName { .. })),
            "{errs:?}"
        );
    }

    // ── Completeness errors ───────────────────────────────────────────────

    #[test]
    fn empty_device_id() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let mut node = simple_node(0, "n", 1, 1);
        node.device = DeviceId::new("");
        let nodes = vec![node];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::EmptyDevice { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn no_outputs_on_node() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let node = simple_node(0, "n", 1, 0); // 0 outputs
        let nodes = vec![node];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::NoOutputs { .. })),
            "{errs:?}"
        );
    }

    // ── DAG errors ────────────────────────────────────────────────────────

    #[test]
    fn cycle_detected() {
        // node0 → node1 → node0 (cycle)
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "a", 2, 1), simple_node(1, "b", 1, 1)];
        let edges = vec![
            edge_from_source(0, 0, 0),
            edge_from_node(0, 0, 1, 0),
            edge_from_node(1, 0, 0, 1), // cycle
        ];
        let sink_connections = vec![sink_conn(1, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter().any(|e| matches!(e, GraphError::Cycle)),
            "{errs:?}"
        );
    }

    // ── Type compatibility errors ─────────────────────────────────────────

    #[test]
    fn type_mismatch_on_edge() {
        // source (F32) → node with I32 input
        let sources = vec![source("in")]; // f32
        let sinks = vec![sink("out")];
        let mut node = simple_node(0, "n", 1, 1);
        node.inputs = vec![i32_type()]; // expects i32
        let nodes = vec![node];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::TypeMismatch { .. })),
            "{errs:?}"
        );
    }

    // ── Port coverage errors ──────────────────────────────────────────────

    #[test]
    fn unconnected_input_port() {
        // node has 1 input but no edge connects to it
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![]; // no edges
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnconnectedPort { .. })),
            "{errs:?}"
        );
    }

    // ── Boundary coverage errors ──────────────────────────────────────────

    #[test]
    fn unused_source() {
        let sources = vec![source("in"), source("unused")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)]; // only source[0] used
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter().any(
                |e| matches!(e, GraphError::UnusedSource { source_name } if source_name == "unused")
            ),
            "{errs:?}"
        );
    }

    #[test]
    fn unconnected_sink() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![]; // sink not connected

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnconnectedSink { sink } if sink == "out")),
            "{errs:?}"
        );
    }

    // ── Port out of range ─────────────────────────────────────────────────

    #[test]
    fn port_out_of_range_on_edge() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        // edge targets port 5 but node only has 1 input
        let edges = vec![Edge {
            from: EdgeSource::Source(0),
            to: PortRef {
                node: NodeId(0),
                port: 5,
            },
        }];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::PortOutOfRange { .. })),
            "{errs:?}"
        );
    }

    // ── KernelDescriptor in Compute node ─────────────────────────────────

    struct MockDesc;
    impl super::super::KernelDescriptor for MockDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn compute_node_passes_validation() {
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let node = Node {
            id: NodeId(0),
            name: "kern".to_string(),
            device: DeviceId::new("cuda:0"),
            kind: NodeKind::Compute(Box::new(MockDesc)),
            inputs: vec![f32_type()],
            outputs: vec![f32_type()],
            stateful: false,
        };
        let nodes = vec![node];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        assert!(validate(&spec).is_empty());
    }

    // ── Additional structural error paths ─────────────────────────────────

    #[test]
    fn edge_to_unknown_node_id() {
        // Edge whose destination NodeId doesn't exist in the node list
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![Edge {
            from: EdgeSource::Source(0),
            to: PortRef {
                node: NodeId(99), // doesn't exist
                port: 0,
            },
        }];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnknownNode { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn edge_from_out_of_range_source_index() {
        // Edge whose source index is beyond the sources list
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![Edge {
            from: EdgeSource::Source(99), // out of range
            to: PortRef {
                node: NodeId(0),
                port: 0,
            },
        }];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnknownSource { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn edge_from_unknown_node_id() {
        // Edge whose source NodeId doesn't exist
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![
            edge_from_source(0, 0, 0),
            Edge {
                from: EdgeSource::Node(PortRef {
                    node: NodeId(99), // doesn't exist
                    port: 0,
                }),
                to: PortRef {
                    node: NodeId(0),
                    port: 0,
                },
            },
        ];
        let sink_connections = vec![sink_conn(0, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnknownNode { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn edge_from_node_port_out_of_range() {
        // Edge whose source port index is beyond the node's output count
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![
            simple_node(0, "a", 1, 1), // only 1 output
            simple_node(1, "b", 2, 1),
        ];
        let edges = vec![
            edge_from_source(0, 0, 0),
            Edge {
                from: EdgeSource::Node(PortRef {
                    node: NodeId(0),
                    port: 5, // out of range
                }),
                to: PortRef {
                    node: NodeId(1),
                    port: 1,
                },
            },
        ];
        let sink_connections = vec![sink_conn(1, 0, 0)];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::PortOutOfRange { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn sink_connection_with_out_of_range_sink_index() {
        // SinkConnection whose sink index is beyond the sinks list
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![SinkConnection {
            from: PortRef {
                node: NodeId(0),
                port: 0,
            },
            sink: 99, // out of range
        }];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnknownNode { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn sink_connection_from_unknown_node() {
        // SinkConnection whose source NodeId doesn't exist
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)];
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![SinkConnection {
            from: PortRef {
                node: NodeId(99), // doesn't exist
                port: 0,
            },
            sink: 0,
        }];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::UnknownNode { .. })),
            "{errs:?}"
        );
    }

    #[test]
    fn sink_connection_port_out_of_range() {
        // SinkConnection whose port index is beyond the node's output count
        let sources = vec![source("in")];
        let sinks = vec![sink("out")];
        let nodes = vec![simple_node(0, "n", 1, 1)]; // 1 output
        let edges = vec![edge_from_source(0, 0, 0)];
        let sink_connections = vec![SinkConnection {
            from: PortRef {
                node: NodeId(0),
                port: 5, // out of range
            },
            sink: 0,
        }];

        let spec = GraphSpec {
            sources: &sources,
            sinks: &sinks,
            nodes: &nodes,
            edges: &edges,
            sink_connections: &sink_connections,
        };
        let errs = validate(&spec);
        assert!(
            errs.iter()
                .any(|e| matches!(e, GraphError::PortOutOfRange { .. })),
            "{errs:?}"
        );
    }
}
