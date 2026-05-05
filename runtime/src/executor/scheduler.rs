use std::collections::{HashMap, VecDeque};

use graph_core::graph::{EdgeSource, Graph, NodeId};

// ── topological_sort ──────────────────────────────────────────────────────────

/// Compute a topological execution order for the nodes in `graph`.
///
/// Uses Kahn's BFS algorithm. The graph is guaranteed to be a DAG by
/// [`GraphBuilder::build`] — this function assumes that invariant holds and
/// will return a partial order if a cycle somehow exists (the caller should
/// not rely on this for correctness; validation is the guard).
///
/// Nodes with no incoming node-to-node edges are processed first. The returned
/// order is deterministic for a given graph (insertion order is the tiebreaker).
///
/// [`GraphBuilder::build`]: graph_core::graph::GraphBuilder::build
pub(crate) fn topological_sort(graph: &Graph) -> Vec<NodeId> {
    let n = graph.nodes().len();
    if n == 0 {
        return Vec::new();
    }

    // Map NodeId → index in graph.nodes() for O(1) lookup.
    let id_to_idx: HashMap<NodeId, usize> = graph
        .nodes()
        .iter()
        .enumerate()
        .map(|(i, node)| (node.id(), i))
        .collect();

    // Build in-degree and adjacency list (node → downstream nodes).
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for edge in graph.edges() {
        if let EdgeSource::Node(pr) = &edge.from {
            if let (Some(&src), Some(&dst)) =
                (id_to_idx.get(&pr.node), id_to_idx.get(&edge.to.node))
            {
                adj[src].push(dst);
                in_degree[dst] += 1;
            }
        }
    }

    // Kahn's BFS: start with all zero-in-degree nodes.
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(graph.nodes()[u].id());
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    order
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::graph::GraphBuilder;
    use graph_core::ops::Op;
    use graph_core::types::{dim::Dim, DType, Layout, TensorType};

    use super::*;

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(n)], Layout::RowMajor).unwrap()
    }

    fn build_linear(n_nodes: usize) -> Graph {
        let t = f32_vec(4);
        let mut b = GraphBuilder::new().source("in", t.clone());
        let mut prev = "in";
        let names: Vec<String> = (0..n_nodes).map(|i| format!("n{i}")).collect();
        for (i, name) in names.iter().enumerate() {
            let nb = b.add_node(name.as_str()).device("cpu").op(Op::Relu);
            let nb = if i == 0 {
                nb.input_from_source(prev)
            } else {
                nb.input_from(prev, 0)
            };
            b = nb.output(t.clone()).done();
            prev = name.as_str();
        }
        b.sink("out", t.clone())
            .from(prev, 0)
            .done()
            .build()
            .unwrap()
    }

    #[test]
    fn empty_graph_returns_empty_order() {
        let graph = GraphBuilder::new().build().unwrap();
        assert!(topological_sort(&graph).is_empty());
    }

    #[test]
    fn single_node_order() {
        let graph = build_linear(1);
        let order = topological_sort(&graph);
        assert_eq!(order.len(), 1);
    }

    #[test]
    fn two_node_chain_order_is_correct() {
        let graph = build_linear(2);
        let order = topological_sort(&graph);
        assert_eq!(order.len(), 2);
        // n0 must come before n1
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n0 = graph.find_node("n0").unwrap().id();
        let n1 = graph.find_node("n1").unwrap().id();
        assert!(pos[&n0] < pos[&n1]);
    }

    #[test]
    fn three_node_chain_order_is_correct() {
        let graph = build_linear(3);
        let order = topological_sort(&graph);
        assert_eq!(order.len(), 3);
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let n0 = graph.find_node("n0").unwrap().id();
        let n1 = graph.find_node("n1").unwrap().id();
        let n2 = graph.find_node("n2").unwrap().id();
        assert!(pos[&n0] < pos[&n1]);
        assert!(pos[&n1] < pos[&n2]);
    }

    #[test]
    fn diamond_graph_order_is_valid() {
        // source → a → b
        //              ↘
        //          a → c → sink
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
            .add_node("c")
            .device("cpu")
            .op(Op::Add)
            .input_from("a", 0)
            .input_from("b", 0)
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("c", 0)
            .done()
            .build()
            .unwrap();

        let order = topological_sort(&graph);
        assert_eq!(order.len(), 3);
        let pos: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let a = graph.find_node("a").unwrap().id();
        let b = graph.find_node("b").unwrap().id();
        let c = graph.find_node("c").unwrap().id();
        // a must come before b and c; b must come before c
        assert!(pos[&a] < pos[&b]);
        assert!(pos[&a] < pos[&c]);
        assert!(pos[&b] < pos[&c]);
    }
}
