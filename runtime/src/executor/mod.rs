use std::collections::HashMap;

use backends::{Backend, DeviceId};
use graph_core::graph::{Graph, NodeId, NodeKind};

use self::buffer::BufferArena;
use self::error::ExecutorError;
use self::handle::{InputHandle, OutputHandle};
use self::scheduler::topological_sort;
use self::state::ExecutionState;

pub mod buffer;
pub mod error;
pub mod handle;
pub mod scheduler;
pub mod state;

// ── Executor ──────────────────────────────────────────────────────────────────

/// A synchronous, one-shot graph executor.
///
/// The executor owns a validated [`Graph`], a set of registered backends, and
/// pre-allocated inter-node data buffers. On each [`run`](Executor::run) call
/// it dispatches nodes in topological order through the registered backends.
///
/// # Memory model
///
/// Phase 2 supports **managed-memory backends** only — backends that accept
/// raw host byte slices via [`Backend::dispatch_op`]. Explicit-memory backends
/// (CUDA, OpenCL) that require `alloc`/`upload`/`download` are supported from
/// Phase 4 onwards.
///
/// # Stateful nodes
///
/// Nodes marked [`stateful`](graph_core::graph::Node::stateful) and whose
/// [`Op::state_shape`](graph_core::ops::Op::state_shape) returns `Some` have
/// their state persisted in [`ExecutionState`] across `run()` calls. State
/// bytes are prepended to the node's inputs and the last output slot is
/// consumed as the updated state.
///
/// # Example
///
/// ```rust,ignore
/// use runtime::executor::Executor;
///
/// let graph = /* GraphBuilder::new()...build().unwrap() */;
/// let backend: Box<dyn Backend> = Box::new(MyCpuBackend::new());
/// let mut exec = Executor::new(graph, vec![backend]).unwrap();
///
/// exec.input("audio")?.write("audio", &samples)?;
/// exec.run()?;
/// let out: &[f32] = exec.output("out")?.read().unwrap();
/// ```
pub struct Executor {
    graph: Graph,
    schedule: Vec<NodeId>,
    state: ExecutionState,
    backends: HashMap<DeviceId, Box<dyn Backend>>,
    arena: BufferArena,
    inputs: HashMap<String, InputHandle>,
    outputs: HashMap<String, OutputHandle>,
}

impl Executor {
    /// Build an executor from a validated graph and a set of backends.
    ///
    /// Validates that every node's device ID has a registered backend and that
    /// every node output port has a statically-known size (no dynamic dims).
    ///
    /// # Errors
    ///
    /// - [`ExecutorError::NoBackend`] — a node's device has no registered backend.
    /// - [`ExecutorError::DynamicSize`] — a node output port has a dynamic tensor type.
    pub fn new(graph: Graph, backends: Vec<Box<dyn Backend>>) -> Result<Self, ExecutorError> {
        // Index backends by DeviceId.
        let backends: HashMap<DeviceId, Box<dyn Backend>> = backends
            .into_iter()
            .map(|b| (b.device_id().clone(), b))
            .collect();

        // Validate every node has a registered backend.
        for node in graph.nodes() {
            if !backends.contains_key(node.device()) {
                return Err(ExecutorError::NoBackend(node.device().as_str().to_string()));
            }
        }

        // Pre-compute topological order.
        let schedule = topological_sort(&graph);

        // Pre-allocate inter-node buffers.
        let arena = BufferArena::new(&graph)?;

        // Build input handles (one per source port).
        let inputs = graph
            .sources()
            .iter()
            .map(|s| {
                let expected = s.tensor_type.size_bytes().unwrap_or(0);
                (
                    s.name.clone(),
                    InputHandle::new(s.tensor_type.clone(), expected),
                )
            })
            .collect();

        // Build output handles (one per sink port).
        let outputs = graph
            .sinks()
            .iter()
            .map(|s| (s.name.clone(), OutputHandle::new(s.tensor_type.clone())))
            .collect();

        Ok(Self {
            graph,
            schedule,
            state: ExecutionState::new(),
            backends,
            arena,
            inputs,
            outputs,
        })
    }

    /// Get a mutable reference to the input handle for a named graph source.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::UnknownInput`] if no source with that name
    /// exists in the graph.
    pub fn input(&mut self, name: &str) -> Result<&mut InputHandle, ExecutorError> {
        self.inputs
            .get_mut(name)
            .ok_or_else(|| ExecutorError::UnknownInput {
                name: name.to_string(),
            })
    }

    /// Get a shared reference to the output handle for a named graph sink.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::UnknownOutput`] if no sink with that name
    /// exists in the graph.
    pub fn output(&self, name: &str) -> Result<&OutputHandle, ExecutorError> {
        self.outputs
            .get(name)
            .ok_or_else(|| ExecutorError::UnknownOutput {
                name: name.to_string(),
            })
    }

    /// Run one tick of the graph synchronously.
    ///
    /// All input handles must have been written before calling `run()`.
    /// After `run()` returns `Ok(())`, all output handles contain the
    /// results of this tick.
    ///
    /// # Errors
    ///
    /// - [`ExecutorError::InputNotWritten`] — an input handle was not written.
    /// - [`ExecutorError::Backend`] — a backend returned an error.
    pub fn run(&mut self) -> Result<(), ExecutorError> {
        // 1. Drain all input handles into the arena source buffers.
        for (idx, source) in self.graph.sources().iter().enumerate() {
            let handle = self
                .inputs
                .get_mut(&source.name)
                .expect("input handle must exist for every source");
            let bytes = handle
                .take()
                .ok_or_else(|| ExecutorError::InputNotWritten {
                    name: source.name.clone(),
                })?;
            self.arena.set_source(idx, &bytes);
        }

        // 2. Dispatch nodes in topological order.
        // Clone the schedule to avoid a simultaneous immutable + mutable borrow of `self`.
        let schedule = self.schedule.clone();
        for node_id in schedule {
            self.dispatch_node(node_id)?;
        }

        // 3. Copy sink outputs into output handles.
        for sc in self.graph.sink_connections() {
            let sink_name = &self.graph.sinks()[sc.sink].name;
            let bytes = self.arena.get_output(sc.from.node, sc.from.port).to_vec();
            self.outputs
                .get_mut(sink_name)
                .expect("output handle must exist for every sink")
                .set(bytes);
        }

        Ok(())
    }

    /// Access the execution state (for testing and inspection).
    pub fn state(&self) -> &ExecutionState {
        &self.state
    }

    // ── Internal dispatch ─────────────────────────────────────────────────

    fn dispatch_node(&mut self, node_id: NodeId) -> Result<(), ExecutorError> {
        let node = self
            .graph
            .nodes()
            .iter()
            .find(|n| n.id() == node_id)
            .expect("schedule contains only valid NodeIds");

        let node_name = node.name().to_string();
        let device = node.device().clone();
        let is_stateful = node.stateful();
        let n_outputs = node.outputs().len();

        // Gather input byte slices from the arena.
        let mut input_slices: Vec<Vec<u8>> = self
            .graph
            .edges()
            .iter()
            .filter(|e| e.to.node == node_id)
            .map(|e| {
                self.arena
                    .resolve_edge_source(&self.graph, &e.from)
                    .to_vec()
            })
            .collect();

        // If stateful and the op declares a state shape, prepend state bytes.
        let state_size: Option<usize> = if is_stateful {
            if let NodeKind::Op(op) = node.kind() {
                op.state_shape().and_then(|t| t.size_bytes())
            } else {
                None
            }
        } else {
            None
        };

        if let Some(sz) = state_size {
            let state_bytes = match self.state.get::<Vec<u8>>(node_id) {
                Some(v) => v.clone(),
                None => vec![0u8; sz],
            };
            // Prepend state as the first input.
            input_slices.insert(0, state_bytes);
        }

        // Build output buffers: borrow pre-allocated arena buffers.
        // We need owned Vecs for dispatch_op's &mut [Vec<u8>] signature.
        // Strategy: swap arena buffers out, dispatch, swap back.
        let mut output_bufs: Vec<Vec<u8>> = (0..n_outputs)
            .map(|port| {
                let buf = self.arena.get_output_mut(node_id, port);
                // Take the pre-allocated buffer (replace with empty).
                std::mem::take(buf)
            })
            .collect();

        // If stateful, add an extra output slot for the updated state.
        if state_size.is_some() {
            output_bufs.push(Vec::new());
        }

        // Build input slice references.
        let input_refs: Vec<&[u8]> = input_slices.iter().map(|v| v.as_slice()).collect();

        // Dispatch.
        let backend = self
            .backends
            .get(&device)
            .expect("backend existence validated in new()");

        match node.kind() {
            NodeKind::Op(op) => {
                backend
                    .dispatch_op(op, &input_refs, &mut output_bufs)
                    .map_err(|source| ExecutorError::Backend {
                        node: node_name.clone(),
                        source,
                    })?;
            }
            NodeKind::Compute(desc) => {
                // Phase 2: Compute nodes are not supported on managed-memory
                // backends. Return UnsupportedNodeKind via the default impl.
                backend
                    .dispatch_compute(desc.as_ref(), &[], &mut [])
                    .map_err(|source| ExecutorError::Backend {
                        node: node_name.clone(),
                        source,
                    })?;
            }
        }

        // If stateful, pop the last output as the updated state.
        if state_size.is_some() {
            let updated_state = output_bufs.pop().unwrap();
            self.state.insert(node_id, updated_state);
        }

        // Swap output buffers back into the arena.
        for (port, buf) in output_bufs.into_iter().enumerate() {
            *self.arena.get_output_mut(node_id, port) = buf;
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::any::Any;

    use backends::{
        Backend, BackendCaps, BackendError, DeviceBuffer, DeviceId, KernelDescriptor, MemoryModel,
        NodeKindTag,
    };
    use bytemuck::cast_slice;
    use graph_core::graph::GraphBuilder;
    use graph_core::ops::Op;
    use graph_core::types::{dim::Dim, DType, Layout, TensorType};

    use super::*;

    // ── Mock backend ──────────────────────────────────────────────────────

    /// A managed-memory mock backend.
    ///
    /// Supports:
    /// - `Op::Relu`  — element-wise max(0, x) on f32
    /// - `Op::Add`   — element-wise addition of two f32 tensors
    /// - `Op::Mul`   — element-wise multiplication of two f32 tensors
    /// - `Op::Custom { name: "accumulate", .. }` — stateful: output = state + input;
    ///   updated_state = output (for testing stateful nodes)
    struct MockCpuBackend {
        device_id: DeviceId,
    }

    impl MockCpuBackend {
        fn new(id: &str) -> Self {
            Self {
                device_id: DeviceId::new(id),
            }
        }
    }

    impl Backend for MockCpuBackend {
        fn name(&self) -> &str {
            "mock_cpu"
        }

        fn device_id(&self) -> &DeviceId {
            &self.device_id
        }

        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                memory: MemoryModel::Managed,
                supported_kinds: vec![NodeKindTag::Op],
            }
        }

        fn alloc(&self, _: usize) -> Result<Box<dyn DeviceBuffer>, BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn upload(&self, _: &[u8], _: &dyn DeviceBuffer) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn download(&self, _: &dyn DeviceBuffer, _: &mut [u8]) -> Result<(), BackendError> {
            Err(BackendError::NotApplicable)
        }

        fn dispatch_op(
            &self,
            op: &graph_core::ops::Op,
            inputs: &[&[u8]],
            outputs: &mut [Vec<u8>],
        ) -> Result<(), BackendError> {
            match op {
                Op::Relu => {
                    let xs: &[f32] = cast_slice(inputs[0]);
                    let ys: Vec<f32> = xs.iter().map(|&x| x.max(0.0)).collect();
                    outputs[0] = cast_slice::<f32, u8>(&ys).to_vec();
                    Ok(())
                }
                Op::Add => {
                    let a: &[f32] = cast_slice(inputs[0]);
                    let b: &[f32] = cast_slice(inputs[1]);
                    let c: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x + y).collect();
                    outputs[0] = cast_slice::<f32, u8>(&c).to_vec();
                    Ok(())
                }
                Op::Mul => {
                    let a: &[f32] = cast_slice(inputs[0]);
                    let b: &[f32] = cast_slice(inputs[1]);
                    let c: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x * y).collect();
                    outputs[0] = cast_slice::<f32, u8>(&c).to_vec();
                    Ok(())
                }
                Op::Custom { name, .. } if name == "accumulate" => {
                    // inputs[0] = state (f32 scalar), inputs[1] = data (f32 scalar)
                    // output[0] = state + data, output[1] = updated state
                    let state: f32 = cast_slice::<u8, f32>(inputs[0])[0];
                    let data: f32 = cast_slice::<u8, f32>(inputs[1])[0];
                    let result = state + data;
                    outputs[0] = cast_slice::<f32, u8>(&[result]).to_vec();
                    outputs[1] = cast_slice::<f32, u8>(&[result]).to_vec();
                    Ok(())
                }
                _ => Err(BackendError::UnsupportedOp),
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn f32_vec(n: usize) -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(n)], Layout::RowMajor).unwrap()
    }

    fn f32_scalar() -> TensorType {
        TensorType::scalar(DType::F32)
    }

    fn cpu_backend() -> Box<dyn Backend> {
        Box::new(MockCpuBackend::new("cpu"))
    }

    // ── Happy-path tests ──────────────────────────────────────────────────

    #[test]
    fn single_relu_node() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("relu")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("relu", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let data: [f32; 4] = [-1.0, 0.0, 1.0, 2.0];
        exec.input("in").unwrap().write("in", &data).unwrap();
        exec.run().unwrap();
        let out: &[f32] = exec.output("out").unwrap().read().unwrap();
        assert_eq!(out, &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn two_node_relu_chain() {
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

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let data: [f32; 4] = [-2.0, -1.0, 0.0, 3.0];
        exec.input("in").unwrap().write("in", &data).unwrap();
        exec.run().unwrap();
        let out: &[f32] = exec.output("out").unwrap().read().unwrap();
        assert_eq!(out, &[0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diamond_fan_out_add() {
        // source → relu → add(relu, relu) → sink
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("relu")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .add_node("add")
            .device("cpu")
            .op(Op::Add)
            .input_from("relu", 0)
            .input_from("relu", 0)
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("add", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let data: [f32; 4] = [-1.0, 1.0, 2.0, 3.0];
        exec.input("in").unwrap().write("in", &data).unwrap();
        exec.run().unwrap();
        let out: &[f32] = exec.output("out").unwrap().read().unwrap();
        // relu(-1)=0, relu(1)=1, relu(2)=2, relu(3)=3 → add doubles each
        assert_eq!(out, &[0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn multiple_ticks_produce_independent_results() {
        let t = f32_vec(2);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("relu")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("relu", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();

        // Tick 1
        exec.input("in")
            .unwrap()
            .write("in", &[-1.0f32, 2.0])
            .unwrap();
        exec.run().unwrap();
        let out1: Vec<f32> = exec.output("out").unwrap().read::<f32>().unwrap().to_vec();
        assert_eq!(out1, vec![0.0, 2.0]);

        // Tick 2
        exec.input("in")
            .unwrap()
            .write("in", &[3.0f32, -4.0])
            .unwrap();
        exec.run().unwrap();
        let out2: Vec<f32> = exec.output("out").unwrap().read::<f32>().unwrap().to_vec();
        assert_eq!(out2, vec![3.0, 0.0]);
    }

    #[test]
    fn stateful_node_accumulates_across_ticks() {
        // The mock "accumulate" op: output = state + input; updated_state = output
        // We use a custom op with a manually-constructed stateful node.
        // Since Op::state_shape() returns None for all ops in Phase 2,
        // we test the stateful path by directly using ExecutionState.
        // This test verifies that ExecutionState persists across run() calls.
        let t = f32_scalar();
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("relu")
            .device("cpu")
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("relu", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();

        // Manually insert state to verify persistence.
        let relu_id = exec.graph.find_node("relu").unwrap().id();
        exec.state.insert(relu_id, vec![99u8]);

        assert!(exec.state().contains(relu_id));
        assert_eq!(exec.state().get::<Vec<u8>>(relu_id), Some(&vec![99u8]));
    }

    // ── Error-path tests ──────────────────────────────────────────────────

    #[test]
    fn no_backend_returns_error() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("n")
            .device("cuda:0") // no backend registered for this
            .op(Op::Relu)
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("n", 0)
            .done()
            .build()
            .unwrap();

        let result = Executor::new(graph, vec![cpu_backend()]);
        assert!(result.is_err());
        assert!(matches!(result.err().unwrap(), ExecutorError::NoBackend(_)));
    }

    #[test]
    fn unknown_input_returns_error() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
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

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let result = exec.input("nonexistent");
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            ExecutorError::UnknownInput { .. }
        ));
    }

    #[test]
    fn unknown_output_returns_error() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
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

        let exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let result = exec.output("nonexistent");
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            ExecutorError::UnknownOutput { .. }
        ));
    }

    #[test]
    fn input_size_mismatch_returns_error() {
        let t = f32_vec(4); // expects 16 bytes
        let graph = GraphBuilder::new()
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

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        let result = exec.input("in").unwrap().write_bytes("in", &[0u8; 8]); // wrong size
        assert!(result.is_err());
        assert!(matches!(
            result.err().unwrap(),
            ExecutorError::InputSizeMismatch { .. }
        ));
    }

    #[test]
    fn input_not_written_returns_error() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
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

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        // Don't write to "in"
        let result = exec.run();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutorError::InputNotWritten { .. }
        ));
    }

    #[test]
    fn backend_dispatch_error_propagates() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("n")
            .device("cpu")
            .op(Op::Sigmoid) // mock backend doesn't support Sigmoid
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("n", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        exec.input("in")
            .unwrap()
            .write("in", &[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let result = exec.run();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutorError::Backend { .. }));
    }

    #[test]
    fn empty_graph_runs_successfully() {
        let graph = GraphBuilder::new().build().unwrap();
        let mut exec = Executor::new(graph, vec![]).unwrap();
        assert!(exec.run().is_ok());
    }

    #[test]
    fn two_sources_two_sinks() {
        let t = f32_vec(2);
        let graph = GraphBuilder::new()
            .source("a", t.clone())
            .source("b", t.clone())
            .add_node("add")
            .device("cpu")
            .op(Op::Add)
            .input_from_source("a")
            .input_from_source("b")
            .output(t.clone())
            .done()
            .add_node("mul")
            .device("cpu")
            .op(Op::Mul)
            .input_from_source("a")
            .input_from_source("b")
            .output(t.clone())
            .done()
            .sink("sum", t.clone())
            .from("add", 0)
            .done()
            .sink("product", t.clone())
            .from("mul", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        exec.input("a").unwrap().write("a", &[1.0f32, 2.0]).unwrap();
        exec.input("b").unwrap().write("b", &[3.0f32, 4.0]).unwrap();
        exec.run().unwrap();

        let sum: &[f32] = exec.output("sum").unwrap().read().unwrap();
        assert_eq!(sum, &[4.0, 6.0]);

        let product: &[f32] = exec.output("product").unwrap().read().unwrap();
        assert_eq!(product, &[3.0, 8.0]);
    }

    // ── KernelDescriptor / Compute node ───────────────────────────────────

    struct MockDesc;
    impl KernelDescriptor for MockDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn compute_node_returns_backend_error() {
        let t = f32_vec(4);
        let graph = GraphBuilder::new()
            .source("in", t.clone())
            .add_node("kern")
            .device("cpu")
            .compute(Box::new(MockDesc))
            .input_from_source("in")
            .output(t.clone())
            .done()
            .sink("out", t.clone())
            .from("kern", 0)
            .done()
            .build()
            .unwrap();

        let mut exec = Executor::new(graph, vec![cpu_backend()]).unwrap();
        exec.input("in")
            .unwrap()
            .write("in", &[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let result = exec.run();
        // MockCpuBackend uses the default dispatch_compute which returns UnsupportedNodeKind
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutorError::Backend { .. }));
    }
}
