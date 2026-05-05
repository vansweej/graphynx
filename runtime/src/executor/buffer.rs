use std::collections::HashMap;

use graph_core::graph::{EdgeSource, Graph, NodeId};

use super::error::ExecutorError;

// ── BufferArena ───────────────────────────────────────────────────────────────

/// Pre-allocated byte buffers for all inter-node data transfers.
///
/// The arena is built once at [`Executor::new`] time from the graph's declared
/// `TensorType` sizes. Each `run()` call reuses these allocations — no heap
/// allocation occurs during execution.
///
/// Buffers are keyed by `(NodeId, port_index)` for node outputs and by source
/// index for graph source ports.
///
/// [`Executor::new`]: super::Executor::new
#[derive(Debug)]
pub(crate) struct BufferArena {
    /// Output buffers keyed by (NodeId, output_port_index).
    node_outputs: HashMap<(NodeId, usize), Vec<u8>>,
    /// Source buffers keyed by source index (into Graph::sources).
    source_buffers: Vec<Vec<u8>>,
}

impl BufferArena {
    /// Allocate buffers for all node output ports and all source ports in
    /// `graph`.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::DynamicSize`] if any node output port has a
    /// tensor type whose size cannot be determined statically (i.e.
    /// `TensorType::size_bytes()` returns `None`).
    pub fn new(graph: &Graph) -> Result<Self, ExecutorError> {
        let mut node_outputs = HashMap::new();

        for node in graph.nodes() {
            for (port, tensor_type) in node.outputs().iter().enumerate() {
                let size = tensor_type
                    .size_bytes()
                    .ok_or_else(|| ExecutorError::DynamicSize {
                        node: node.name().to_string(),
                        port,
                    })?;
                node_outputs.insert((node.id(), port), vec![0u8; size]);
            }
        }

        // Source buffers are sized from the declared source TensorType.
        // If a source has a dynamic size, we allocate an empty buffer and let
        // the InputHandle validation catch the mismatch at write time.
        let source_buffers = graph
            .sources()
            .iter()
            .map(|s| {
                let size = s.tensor_type.size_bytes().unwrap_or(0);
                vec![0u8; size]
            })
            .collect();

        Ok(Self {
            node_outputs,
            source_buffers,
        })
    }

    /// Get a shared reference to the output buffer for `(node, port)`.
    ///
    /// # Panics
    ///
    /// Panics if the `(node, port)` pair was not pre-allocated (i.e. it does
    /// not exist in the graph). This is a programming error — the executor
    /// only accesses ports that exist in the validated graph.
    pub fn get_output(&self, node: NodeId, port: usize) -> &[u8] {
        self.node_outputs
            .get(&(node, port))
            .expect("BufferArena: (node, port) not found — graph validation invariant violated")
    }

    /// Get a mutable reference to the output buffer for `(node, port)`.
    ///
    /// # Panics
    ///
    /// Panics if the `(node, port)` pair was not pre-allocated.
    pub fn get_output_mut(&mut self, node: NodeId, port: usize) -> &mut Vec<u8> {
        self.node_outputs
            .get_mut(&(node, port))
            .expect("BufferArena: (node, port) not found — graph validation invariant violated")
    }

    /// Copy `data` into the source buffer at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= source_buffers.len()` or if `data.len()` does not
    /// match the pre-allocated buffer size.
    pub fn set_source(&mut self, idx: usize, data: &[u8]) {
        let buf = &mut self.source_buffers[idx];
        buf.clear();
        buf.extend_from_slice(data);
    }

    /// Get a shared reference to the source buffer at `idx`.
    pub fn get_source(&self, idx: usize) -> &[u8] {
        &self.source_buffers[idx]
    }

    /// Resolve an [`EdgeSource`] to the byte slice it currently holds.
    pub fn resolve_edge_source(&self, _graph: &Graph, from: &EdgeSource) -> &[u8] {
        match from {
            EdgeSource::Source(idx) => self.get_source(*idx),
            EdgeSource::Node(pr) => self.get_output(pr.node, pr.port),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::graph::GraphBuilder;
    use graph_core::ops::Op;
    use graph_core::types::{dim::Dim, DType, Dim as DimType, Layout, TensorType};

    use super::*;

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(n)], Layout::RowMajor).unwrap()
    }

    fn dynamic_type() -> TensorType {
        TensorType::new(DType::F32, vec![DimType::Dynamic], Layout::RowMajor).unwrap()
    }

    fn simple_graph(n_outputs: usize) -> graph_core::graph::Graph {
        let t = f32_vec(n_outputs);
        GraphBuilder::new()
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
            .unwrap()
    }

    #[test]
    fn arena_allocates_correct_sizes() {
        let graph = simple_graph(4); // 4 f32 = 16 bytes
        let arena = BufferArena::new(&graph).unwrap();
        let node_id = graph.find_node("n").unwrap().id();
        assert_eq!(arena.get_output(node_id, 0).len(), 16);
    }

    #[test]
    fn arena_source_buffer_sized_correctly() {
        let graph = simple_graph(2); // 2 f32 = 8 bytes
        let arena = BufferArena::new(&graph).unwrap();
        assert_eq!(arena.get_source(0).len(), 8);
    }

    #[test]
    fn set_source_and_get_source() {
        let graph = simple_graph(4);
        let mut arena = BufferArena::new(&graph).unwrap();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        arena.set_source(0, bytes);
        assert_eq!(arena.get_source(0), bytes);
    }

    #[test]
    fn get_output_mut_allows_write() {
        let graph = simple_graph(2);
        let mut arena = BufferArena::new(&graph).unwrap();
        let node_id = graph.find_node("n").unwrap().id();
        {
            let buf = arena.get_output_mut(node_id, 0);
            buf[0] = 0xFF;
        }
        assert_eq!(arena.get_output(node_id, 0)[0], 0xFF);
    }

    #[test]
    fn dynamic_output_returns_error() {
        let t_dyn = dynamic_type();
        let t_in = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t_in.clone())
            .add_node("n")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t_dyn) // dynamic output
            .done()
            .sink("out", t_in.clone())
            .from("n", 0)
            .done()
            .build()
            .unwrap();

        let result = BufferArena::new(&graph);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutorError::DynamicSize { .. }
        ));
    }

    #[test]
    fn resolve_edge_source_from_source() {
        let graph = simple_graph(4);
        let mut arena = BufferArena::new(&graph).unwrap();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        arena.set_source(0, bytes);

        let edge = &graph.edges()[0];
        let resolved = arena.resolve_edge_source(&graph, &edge.from);
        assert_eq!(resolved, bytes);
    }

    #[test]
    fn resolve_edge_source_from_node() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("a")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .add_node("b")
            .device("cpu")
            .op(Op::Relu)
            .input_from("a", 0)
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("b", 0)
            .done()
            .build()
            .unwrap();

        let mut arena = BufferArena::new(&graph).unwrap();
        let a_id = graph.find_node("a").unwrap().id();
        // Write something into a's output
        {
            let buf = arena.get_output_mut(a_id, 0);
            buf[0] = 42;
        }

        // Find the edge from a → b
        let edge_a_to_b = graph
            .edges()
            .iter()
            .find(|e| matches!(&e.from, graph_core::graph::EdgeSource::Node(pr) if pr.node == a_id))
            .unwrap();

        let resolved = arena.resolve_edge_source(&graph, &edge_a_to_b.from);
        assert_eq!(resolved[0], 42);
    }
}
