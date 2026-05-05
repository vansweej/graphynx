use std::any::Any;
use std::collections::HashMap;

use graph_core::graph::NodeId;

// ── ExecutionState ────────────────────────────────────────────────────────────

/// Persistent state storage for stateful graph nodes.
///
/// Each stateful node (where [`Node::stateful()`] is `true` and
/// [`Op::state_shape()`] returns `Some`) gets a slot in this map. The executor
/// stores and retrieves state bytes across `run()` calls.
///
/// State is stored as `Box<dyn Any + Send>` so the concrete type is erased.
/// In practice the executor stores `Vec<u8>` (raw state bytes), but the
/// type-erased interface allows future backends to store typed state directly.
///
/// [`Node::stateful()`]: graph_core::graph::Node::stateful
/// [`Op::state_shape()`]: graph_core::ops::Op::state_shape
#[derive(Default)]
pub struct ExecutionState {
    slots: HashMap<NodeId, Box<dyn Any + Send>>,
}

impl ExecutionState {
    /// Create a new, empty execution state.
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    /// Returns a shared reference to the state value for `node`, if present.
    ///
    /// Returns `None` if no state has been stored for this node, or if the
    /// stored value is not of type `T`.
    pub fn get<T: 'static + Send>(&self, node: NodeId) -> Option<&T> {
        self.slots.get(&node)?.downcast_ref::<T>()
    }

    /// Returns a mutable reference to the state value for `node`, if present.
    ///
    /// Returns `None` if no state has been stored for this node, or if the
    /// stored value is not of type `T`.
    pub fn get_mut<T: 'static + Send>(&mut self, node: NodeId) -> Option<&mut T> {
        self.slots.get_mut(&node)?.downcast_mut::<T>()
    }

    /// Insert or replace the state value for `node`.
    pub fn insert<T: Send + 'static>(&mut self, node: NodeId, state: T) {
        self.slots.insert(node, Box::new(state));
    }

    /// Returns `true` if a state value has been stored for `node`.
    pub fn contains(&self, node: NodeId) -> bool {
        self.slots.contains_key(&node)
    }

    /// Remove and return the state value for `node`, if present.
    pub fn remove(&mut self, node: NodeId) -> Option<Box<dyn Any + Send>> {
        self.slots.remove(&node)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use graph_core::graph::GraphBuilder;
    use graph_core::ops::Op;
    use graph_core::types::{dim::Dim, DType, Layout, TensorType};

    use super::*;

    /// Build a minimal graph with `n` chained Relu nodes and return their NodeIds in order.
    fn make_node_ids(n: usize) -> Vec<NodeId> {
        let t = TensorType::new(DType::F32, vec![Dim::Fixed(1)], Layout::RowMajor).unwrap();
        let mut b = GraphBuilder::new().source("in", t.clone());
        let names: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        for (i, name) in names.iter().enumerate() {
            let nb = b.add_node(name.as_str()).device("cpu").op(Op::Relu);
            let nb = if i == 0 {
                nb.input_from_source("in")
            } else {
                nb.input_from(names[i - 1].as_str(), 0)
            };
            b = nb.output(t.clone()).done();
        }
        if n > 0 {
            b = b
                .sink("out", t.clone())
                .from(names.last().unwrap().as_str(), 0)
                .done();
        }
        let graph = b.build().unwrap();
        graph.nodes().iter().map(|nd| nd.id()).collect()
    }

    #[test]
    fn new_state_is_empty() {
        let ids = make_node_ids(1);
        let s = ExecutionState::new();
        assert!(!s.contains(ids[0]));
    }

    #[test]
    fn default_state_is_empty() {
        let ids = make_node_ids(1);
        let s = ExecutionState::default();
        assert!(!s.contains(ids[0]));
    }

    #[test]
    fn insert_and_get_vec_u8() {
        let ids = make_node_ids(1);
        let mut s = ExecutionState::new();
        s.insert(ids[0], vec![1u8, 2, 3]);
        assert!(s.contains(ids[0]));
        assert_eq!(s.get::<Vec<u8>>(ids[0]), Some(&vec![1u8, 2, 3]));
    }

    #[test]
    fn get_wrong_type_returns_none() {
        let ids = make_node_ids(1);
        let mut s = ExecutionState::new();
        s.insert(ids[0], vec![1u8, 2, 3]);
        // Stored Vec<u8>, asking for u32 → None
        assert!(s.get::<u32>(ids[0]).is_none());
    }

    #[test]
    fn get_missing_node_returns_none() {
        let ids = make_node_ids(2);
        let s = ExecutionState::new();
        assert!(s.get::<Vec<u8>>(ids[1]).is_none());
    }

    #[test]
    fn get_mut_allows_modification() {
        let ids = make_node_ids(2);
        let mut s = ExecutionState::new();
        s.insert(ids[1], vec![0u8; 4]);
        {
            let v = s.get_mut::<Vec<u8>>(ids[1]).unwrap();
            v[0] = 42;
        }
        assert_eq!(s.get::<Vec<u8>>(ids[1]).unwrap()[0], 42);
    }

    #[test]
    fn get_mut_missing_returns_none() {
        let ids = make_node_ids(1);
        let mut s = ExecutionState::new();
        assert!(s.get_mut::<Vec<u8>>(ids[0]).is_none());
    }

    #[test]
    fn remove_returns_value() {
        let ids = make_node_ids(3);
        let mut s = ExecutionState::new();
        s.insert(ids[2], vec![7u8]);
        let removed = s.remove(ids[2]);
        assert!(removed.is_some());
        assert!(!s.contains(ids[2]));
    }

    #[test]
    fn remove_missing_returns_none() {
        let ids = make_node_ids(1);
        let mut s = ExecutionState::new();
        assert!(s.remove(ids[0]).is_none());
    }

    #[test]
    fn insert_replaces_existing() {
        let ids = make_node_ids(1);
        let mut s = ExecutionState::new();
        s.insert(ids[0], vec![1u8]);
        s.insert(ids[0], vec![2u8, 3]);
        assert_eq!(s.get::<Vec<u8>>(ids[0]), Some(&vec![2u8, 3]));
    }

    #[test]
    fn multiple_nodes_independent() {
        let ids = make_node_ids(2);
        let mut s = ExecutionState::new();
        s.insert(ids[0], vec![0u8]);
        s.insert(ids[1], vec![1u8]);
        assert_eq!(s.get::<Vec<u8>>(ids[0]), Some(&vec![0u8]));
        assert_eq!(s.get::<Vec<u8>>(ids[1]), Some(&vec![1u8]));
    }
}
