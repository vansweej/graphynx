use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use ron::ser::PrettyConfig;
use thiserror::Error;

use crate::graph::edge::EdgeSource;
use crate::graph::node::NodeKind;
use crate::graph::{Graph, GraphBuilder, GraphError};
use crate::ops::Op;
use crate::types::{DType, Dim, Layout, TensorType, TensorTypeError};

/// Current graph persistence format version.
pub const CURRENT_VERSION: u32 = 1;

/// Serializable tensor type description used on source, sink, and node ports.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TensorTypeSpec {
    /// Scalar element type.
    pub dtype: DType,
    /// Tensor dimensions.
    pub shape: Vec<Dim>,
    /// Memory layout constraint.
    pub layout: Layout,
    /// Optional per-dimension names.
    pub dim_names: Option<Vec<String>>,
}

impl TensorTypeSpec {
    fn to_tensor_type(&self, context: impl Into<String>) -> Result<TensorType, GraphFileError> {
        let context = context.into();
        let tensor_type = TensorType::new(self.dtype.clone(), self.shape.clone(), self.layout)
            .map_err(|source| GraphFileError::InvalidTensorType {
                context: context.clone(),
                source,
            })?;

        if let Some(dim_names) = &self.dim_names {
            tensor_type
                .with_dim_names(dim_names.clone())
                .map_err(|source| GraphFileError::InvalidTensorType { context, source })
        } else {
            Ok(tensor_type)
        }
    }
}

impl From<&TensorType> for TensorTypeSpec {
    fn from(tensor_type: &TensorType) -> Self {
        Self {
            dtype: tensor_type.dtype(),
            shape: tensor_type.shape().dims().to_vec(),
            layout: tensor_type.layout(),
            dim_names: tensor_type.dim_names().map(<[String]>::to_vec),
        }
    }
}

/// Serializable representation of a node input connection.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InputSpec {
    /// Connected to a named graph source port.
    Source(String),
    /// Connected to a named node's output port.
    FromNode {
        /// Source node name.
        node: String,
        /// Source node output port.
        port: usize,
    },
}

/// Specifies the computation a node performs in the file format.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OpSpec {
    /// A catalogued operation, fully self-contained in the file.
    Op(Op),
    /// A named reference to an externally registered kernel.
    KernelRef(String),
}

/// Serializable representation of a computation node.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeSpec {
    /// Device identifier used by the scheduler/backend selection path.
    pub device: String,
    /// Operation or external kernel reference.
    pub op: OpSpec,
    /// Input wiring, in input-port order.
    pub inputs: Vec<InputSpec>,
    /// Output tensor types, in output-port order.
    pub outputs: Vec<TensorTypeSpec>,
    /// Whether this node carries executor state across ticks.
    pub stateful: bool,
}

/// Serializable representation of a graph source port.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceSpec {
    /// Source port name.
    pub name: String,
    /// Tensor type expected at this source.
    pub tensor_type: TensorTypeSpec,
}

/// Serializable representation of a graph sink port.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SinkSpec {
    /// Sink port name.
    pub name: String,
    /// Node that feeds this sink.
    pub from_node: String,
    /// Output port on `from_node`.
    pub from_port: usize,
    /// Tensor type expected at this sink.
    pub tensor_type: TensorTypeSpec,
}

/// Top-level graph persistence file.
///
/// The `layout` field is an opaque blob reserved for the visual editor. The
/// graph runtime ignores it, while editor-tier APIs preserve it exactly.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GraphFile {
    /// File format version.
    pub version: u32,
    /// External input ports.
    pub sources: Vec<SourceSpec>,
    /// Computation nodes keyed by name. `BTreeMap` keeps save output stable.
    pub nodes: BTreeMap<String, NodeSpec>,
    /// External output ports.
    pub sinks: Vec<SinkSpec>,
    /// Opaque layout metadata for a visual editor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<ron::Value>,
}

impl GraphFile {
    /// Validate and build an executable [`Graph`].
    ///
    /// Layout metadata is discarded. Use [`load_file`] when editor metadata
    /// must be preserved.
    ///
    /// # Errors
    ///
    /// Returns [`GraphFileError`] for unsupported versions, unresolved kernel
    /// references, invalid tensor specs, or graph validation failures.
    pub fn build(self) -> Result<Graph, GraphFileError> {
        check_version(self.version)?;

        let node_order = topological_node_order(&self.nodes)?;
        let mut builder = GraphBuilder::new();

        for source in self.sources {
            let tensor_type = source
                .tensor_type
                .to_tensor_type(format!("source '{}'", source.name))?;
            builder = builder.source(source.name, tensor_type);
        }

        for node_name in node_order {
            let Some(node_spec) = self.nodes.get(&node_name) else {
                return Err(GraphFileError::InvalidGraph(vec![
                    GraphError::UnknownNode { name: node_name },
                ]));
            };

            let mut node_builder = builder
                .add_node(node_name.clone())
                .device(&node_spec.device);

            node_builder = match &node_spec.op {
                OpSpec::Op(op) => node_builder.op(op.clone()),
                OpSpec::KernelRef(kernel_ref) => {
                    return Err(GraphFileError::UnresolvedKernelRef(kernel_ref.clone()));
                }
            };

            for input in &node_spec.inputs {
                node_builder = match input {
                    InputSpec::Source(source_name) => node_builder.input_from_source(source_name),
                    InputSpec::FromNode { node, port } => node_builder.input_from(node, *port),
                };
            }

            for (port, output) in node_spec.outputs.iter().enumerate() {
                let tensor_type =
                    output.to_tensor_type(format!("node '{}' output port {}", node_name, port))?;
                node_builder = node_builder.output(tensor_type);
            }

            if node_spec.stateful {
                node_builder = node_builder.stateful();
            }

            builder = node_builder.done();
        }

        for sink in self.sinks {
            let tensor_type = sink
                .tensor_type
                .to_tensor_type(format!("sink '{}'", sink.name))?;
            builder = builder
                .sink(sink.name, tensor_type)
                .from(sink.from_node, sink.from_port)
                .done();
        }

        builder.build().map_err(GraphFileError::InvalidGraph)
    }

    /// Save this file to disk, preserving layout metadata.
    ///
    /// # Errors
    ///
    /// Returns [`GraphFileError`] when serialization or writing fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), GraphFileError> {
        save_file(self, path)
    }
}

/// Errors produced when saving or loading graph persistence files.
#[derive(Debug, Error)]
pub enum GraphFileError {
    /// Filesystem read/write failure.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// RON serialization failure.
    #[error("RON serialization error: {0}")]
    Serialize(String),
    /// RON deserialization failure.
    #[error("RON deserialization error: {0}")]
    Deserialize(String),
    /// File version is newer than this library understands.
    #[error("Unsupported file version {found}; this library supports up to version {supported}")]
    UnsupportedVersion {
        /// Version read from the file.
        found: u32,
        /// Highest version supported by this library.
        supported: u32,
    },
    /// A raw compute node cannot be serialized without a stable kernel name.
    #[error(
        "Node '{name}' uses NodeKind::Compute which cannot be serialized; register a KernelRef name first"
    )]
    UnserializableNode {
        /// Name of the unserializable compute node.
        name: String,
    },
    /// Kernel references are reserved for a future registry-based loader.
    #[error("Unresolved kernel reference '{0}' (no factory registered)")]
    UnresolvedKernelRef(String),
    /// A tensor spec failed validated reconstruction.
    #[error("Invalid tensor type for {context}: {source}")]
    InvalidTensorType {
        /// Human-readable location in the graph file.
        context: String,
        /// Tensor validation error.
        #[source]
        source: TensorTypeError,
    },
    /// GraphBuilder validation failed after replaying the file.
    #[error("Graph validation failed: {0:?}")]
    InvalidGraph(Vec<GraphError>),
}

/// Save a validated [`Graph`] to a `.graphynx.ron` file.
///
/// This executor-tier path writes `layout: None`. Use [`save_file`] to preserve
/// editor layout metadata.
///
/// # Errors
///
/// Returns [`GraphFileError::UnserializableNode`] if the graph contains raw
/// compute nodes, or IO/serialization errors if writing fails.
pub fn save(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphFileError> {
    let file = graph_to_file(graph)?;
    save_file(&file, path)
}

/// Load and validate a [`Graph`] from a `.graphynx.ron` file.
///
/// Layout metadata is discarded. Use [`load_file`] to preserve layout.
///
/// # Errors
///
/// Returns [`GraphFileError`] on IO, parse, version, tensor, kernel-reference,
/// or graph-validation failure.
pub fn load(path: impl AsRef<Path>) -> Result<Graph, GraphFileError> {
    load_file(path)?.build()
}

/// Save a [`GraphFile`] to disk, preserving layout metadata.
///
/// # Errors
///
/// Returns [`GraphFileError::Serialize`] if RON serialization fails, or
/// [`GraphFileError::Io`] if the file cannot be written.
pub fn save_file(file: &GraphFile, path: impl AsRef<Path>) -> Result<(), GraphFileError> {
    let pretty = PrettyConfig::default();
    let ron = ron::ser::to_string_pretty(file, pretty)
        .map_err(|error| GraphFileError::Serialize(error.to_string()))?;
    std::fs::write(path, ron)?;
    Ok(())
}

/// Load a raw [`GraphFile`] from disk, preserving layout metadata.
///
/// # Errors
///
/// Returns [`GraphFileError`] on IO, parse, or unsupported-version failure.
pub fn load_file(path: impl AsRef<Path>) -> Result<GraphFile, GraphFileError> {
    let source = std::fs::read_to_string(path)?;
    let file: GraphFile = ron::de::from_str(&source)
        .map_err(|error| GraphFileError::Deserialize(error.to_string()))?;
    check_version(file.version)?;
    Ok(file)
}

fn check_version(version: u32) -> Result<(), GraphFileError> {
    if version > CURRENT_VERSION {
        Err(GraphFileError::UnsupportedVersion {
            found: version,
            supported: CURRENT_VERSION,
        })
    } else {
        Ok(())
    }
}

fn graph_to_file(graph: &Graph) -> Result<GraphFile, GraphFileError> {
    let sources = graph
        .sources
        .iter()
        .map(|source| SourceSpec {
            name: source.name.clone(),
            tensor_type: TensorTypeSpec::from(&source.tensor_type),
        })
        .collect();

    let mut nodes = BTreeMap::new();
    for node in &graph.nodes {
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for port in 0..node.inputs.len() {
            inputs.push(edge_source_for_input(graph, node.id, &node.name, port)?);
        }

        let op = match &node.kind {
            NodeKind::Op(op) => OpSpec::Op(op.clone()),
            NodeKind::Compute(_) => {
                return Err(GraphFileError::UnserializableNode {
                    name: node.name.clone(),
                });
            }
        };

        let outputs = node.outputs.iter().map(TensorTypeSpec::from).collect();
        nodes.insert(
            node.name.clone(),
            NodeSpec {
                device: node.device.as_str().to_string(),
                op,
                inputs,
                outputs,
                stateful: node.stateful,
            },
        );
    }

    let mut sinks = Vec::with_capacity(graph.sinks.len());
    for (sink_idx, sink) in graph.sinks.iter().enumerate() {
        let Some(connection) = graph
            .sink_connections
            .iter()
            .find(|connection| connection.sink == sink_idx)
        else {
            return Err(GraphFileError::InvalidGraph(vec![
                GraphError::UnconnectedSink {
                    sink: sink.name.clone(),
                },
            ]));
        };

        let Some(source_node) = graph.nodes.get(connection.from.node.0) else {
            return Err(GraphFileError::InvalidGraph(vec![
                GraphError::UnknownNode {
                    name: connection.from.node.to_string(),
                },
            ]));
        };

        sinks.push(SinkSpec {
            name: sink.name.clone(),
            from_node: source_node.name.clone(),
            from_port: connection.from.port,
            tensor_type: TensorTypeSpec::from(&sink.tensor_type),
        });
    }

    Ok(GraphFile {
        version: CURRENT_VERSION,
        sources,
        nodes,
        sinks,
        layout: None,
    })
}

fn edge_source_for_input(
    graph: &Graph,
    node_id: crate::graph::NodeId,
    node_name: &str,
    port: usize,
) -> Result<InputSpec, GraphFileError> {
    let Some(edge) = graph
        .edges
        .iter()
        .find(|edge| edge.to.node == node_id && edge.to.port == port)
    else {
        return Err(GraphFileError::InvalidGraph(vec![
            GraphError::UnconnectedPort {
                node: node_name.to_string(),
                port,
            },
        ]));
    };

    match &edge.from {
        EdgeSource::Source(source_idx) => {
            let Some(source) = graph.sources.get(*source_idx) else {
                return Err(GraphFileError::InvalidGraph(vec![
                    GraphError::UnknownSource {
                        name: format!("source[{source_idx}]"),
                        node: node_name.to_string(),
                    },
                ]));
            };
            Ok(InputSpec::Source(source.name.clone()))
        }
        EdgeSource::Node(port_ref) => {
            let Some(source_node) = graph.nodes.get(port_ref.node.0) else {
                return Err(GraphFileError::InvalidGraph(vec![
                    GraphError::UnknownNode {
                        name: port_ref.node.to_string(),
                    },
                ]));
            };
            Ok(InputSpec::FromNode {
                node: source_node.name.clone(),
                port: port_ref.port,
            })
        }
    }
}

fn topological_node_order(
    nodes: &BTreeMap<String, NodeSpec>,
) -> Result<Vec<String>, GraphFileError> {
    let mut incoming_counts: BTreeMap<String, usize> =
        nodes.keys().map(|name| (name.clone(), 0)).collect();
    let mut outgoing_edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut errors = Vec::new();

    for (node_name, node_spec) in nodes {
        for input in &node_spec.inputs {
            if let InputSpec::FromNode { node, .. } = input {
                if nodes.contains_key(node) {
                    *incoming_counts.entry(node_name.clone()).or_insert(0) += 1;
                    outgoing_edges
                        .entry(node.clone())
                        .or_default()
                        .insert(node_name.clone());
                } else {
                    errors.push(GraphError::UnknownNode { name: node.clone() });
                }
            }
        }
    }

    if !errors.is_empty() {
        return Err(GraphFileError::InvalidGraph(errors));
    }

    let mut ready: BTreeSet<String> = incoming_counts
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut ordered = Vec::with_capacity(nodes.len());

    while let Some(node_name) = ready.pop_first() {
        ordered.push(node_name.clone());

        if let Some(children) = outgoing_edges.get(&node_name) {
            for child in children {
                let Some(count) = incoming_counts.get_mut(child) else {
                    continue;
                };
                *count -= 1;
                if *count == 0 {
                    ready.insert(child.clone());
                }
            }
        }
    }

    if ordered.len() == nodes.len() {
        Ok(ordered)
    } else {
        Err(GraphFileError::InvalidGraph(vec![GraphError::Cycle]))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::graph::edge::{Edge, PortRef};
    use crate::graph::node::{Node, NodeId};
    use crate::graph::{KernelDescriptor, SinkPort, SourcePort};
    use crate::ops::{
        BandDef, BandExtractParams, FftDirection, FftOutput, FftParams, Op, WindowKind,
        WindowParams,
    };

    use super::*;

    struct MockDesc;

    impl KernelDescriptor for MockDesc {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn frame_type() -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(1024)], Layout::RowMajor).unwrap()
    }

    fn spectrum_type() -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(513)], Layout::RowMajor).unwrap()
    }

    fn bands_type() -> TensorType {
        TensorType::new(DType::F32, vec![Dim::Fixed(3)], Layout::RowMajor).unwrap()
    }

    fn build_voice_pipeline() -> Graph {
        let frame = frame_type();
        let spectrum = spectrum_type();
        let bands = bands_type();
        let band_defs = vec![
            BandDef::new(20.0, 250.0, "low").unwrap(),
            BandDef::new(250.0, 4000.0, "mid").unwrap(),
            BandDef::new(4000.0, 20000.0, "high").unwrap(),
        ];

        GraphBuilder::new()
            .source("audio", frame.clone())
            .add_node("window")
            .device("cpu:0")
            .op(Op::Window(
                WindowParams::new(WindowKind::Hann, 1024).unwrap(),
            ))
            .input_from_source("audio")
            .output(frame.clone())
            .done()
            .add_node("fft")
            .device("cpu:0")
            .op(Op::Fft(
                FftParams::new(1024, FftDirection::Forward, FftOutput::Magnitude).unwrap(),
            ))
            .input_from("window", 0)
            .output(spectrum)
            .done()
            .add_node("bands")
            .device("cpu:0")
            .op(Op::BandExtract(
                BandExtractParams::new(band_defs, 44_100.0, 0.6).unwrap(),
            ))
            .input_from("fft", 0)
            .output(bands.clone())
            .stateful()
            .done()
            .sink("energies", bands)
            .from("bands", 0)
            .done()
            .build()
            .unwrap()
    }

    #[test]
    fn voice_pipeline_roundtrips() {
        let original = build_voice_pipeline();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.graphynx.ron");

        save(&original, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(original.sources().len(), loaded.sources().len());
        assert_eq!(original.node_count(), loaded.node_count());
        assert_eq!(original.sinks().len(), loaded.sinks().len());

        for (a, b) in original.sources().iter().zip(loaded.sources()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.tensor_type, b.tensor_type);
        }

        for (a, b) in original.sinks().iter().zip(loaded.sinks()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.tensor_type, b.tensor_type);
        }

        let fft_node = loaded.find_node("fft").unwrap();
        assert!(matches!(
            fft_node.kind(),
            NodeKind::Op(Op::Fft(FftParams { size: 1024, .. }))
        ));
    }

    #[test]
    fn save_load_file_preserves_layout_none() {
        let graph = build_voice_pipeline();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.graphynx.ron");

        save(&graph, &path).unwrap();
        let file = load_file(&path).unwrap();

        assert_eq!(file.version, CURRENT_VERSION);
        assert!(file.layout.is_none());
    }

    #[test]
    fn save_file_preserves_layout() {
        let graph = build_voice_pipeline();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_layout.graphynx.ron");

        save(&graph, &path).unwrap();
        let mut file = load_file(&path).unwrap();
        let fake_layout: ron::Value = ron::de::from_str(r#""layout""#).unwrap();
        file.layout = Some(fake_layout.clone());

        save_file(&file, &path).unwrap();
        let reloaded = load_file(&path).unwrap();
        assert_eq!(reloaded.layout, Some(fake_layout));

        let graph2 = load(&path).unwrap();
        assert_eq!(graph2.node_count(), graph.node_count());
    }

    #[test]
    fn graph_file_save_preserves_layout() {
        let graph = build_voice_pipeline();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("method.graphynx.ron");
        let mut file = graph_to_file(&graph).unwrap();
        file.layout = Some(ron::de::from_str(r#""layout""#).unwrap());

        file.save(&path).unwrap();

        assert_eq!(load_file(&path).unwrap().layout, file.layout);
    }

    #[test]
    fn save_is_idempotent() {
        let graph = build_voice_pipeline();
        let dir = tempfile::tempdir().unwrap();
        let path1 = dir.path().join("a.graphynx.ron");
        let path2 = dir.path().join("b.graphynx.ron");

        save(&graph, &path1).unwrap();
        let loaded = load(&path1).unwrap();
        save(&loaded, &path2).unwrap();

        let s1 = std::fs::read_to_string(&path1).unwrap();
        let s2 = std::fs::read_to_string(&path2).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn unsupported_version_returns_error() {
        let ron_src = r#"GraphFile(version: 9999, sources: [], nodes: {}, sinks: [])"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.graphynx.ron");
        std::fs::write(&path, ron_src).unwrap();

        let err = load(&path).unwrap_err();
        assert!(matches!(err, GraphFileError::UnsupportedVersion { .. }));
    }

    #[test]
    fn compute_node_save_returns_error() {
        let graph = Graph {
            sources: vec![],
            sinks: vec![],
            nodes: vec![Node {
                id: NodeId(0),
                name: "kernel".into(),
                device: crate::types::DeviceId::new("cuda:0"),
                kind: NodeKind::Compute(Box::new(MockDesc)),
                inputs: vec![],
                outputs: vec![frame_type()],
                stateful: false,
            }],
            edges: vec![],
            sink_connections: vec![],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("compute.graphynx.ron");

        let err = save(&graph, &path).unwrap_err();
        assert!(matches!(err, GraphFileError::UnserializableNode { name } if name == "kernel"));
    }

    #[test]
    fn kernel_ref_build_returns_error() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "kernel".to_string(),
            NodeSpec {
                device: "cuda:0".into(),
                op: OpSpec::KernelRef("my_ptx_kernel".into()),
                inputs: vec![],
                outputs: vec![TensorTypeSpec::from(&frame_type())],
                stateful: false,
            },
        );
        let file = GraphFile {
            version: CURRENT_VERSION,
            sources: vec![],
            nodes,
            sinks: vec![],
            layout: None,
        };

        let err = file.build().unwrap_err();
        assert!(
            matches!(err, GraphFileError::UnresolvedKernelRef(name) if name == "my_ptx_kernel")
        );
    }

    #[test]
    fn invalid_tensor_type_returns_error() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "relu".to_string(),
            NodeSpec {
                device: "cpu:0".into(),
                op: OpSpec::Op(Op::Relu),
                inputs: vec![InputSpec::Source("audio".into())],
                outputs: vec![TensorTypeSpec {
                    dtype: DType::F32,
                    shape: vec![Dim::Fixed(0)],
                    layout: Layout::RowMajor,
                    dim_names: None,
                }],
                stateful: false,
            },
        );
        let file = GraphFile {
            version: CURRENT_VERSION,
            sources: vec![SourceSpec {
                name: "audio".into(),
                tensor_type: TensorTypeSpec::from(&frame_type()),
            }],
            nodes,
            sinks: vec![SinkSpec {
                name: "out".into(),
                from_node: "relu".into(),
                from_port: 0,
                tensor_type: TensorTypeSpec::from(&frame_type()),
            }],
            layout: None,
        };

        let err = file.build().unwrap_err();
        assert!(matches!(err, GraphFileError::InvalidTensorType { .. }));
    }

    #[test]
    fn graph_file_build_toposorts_nodes() {
        let graph = build_voice_pipeline();
        let file = graph_to_file(&graph).unwrap();
        let keys = file.nodes.keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys, vec!["bands", "fft", "window"]);

        let rebuilt = file.build().unwrap();

        assert_eq!(rebuilt.node_count(), 3);
        assert!(rebuilt.find_node("window").is_some());
        assert!(rebuilt.find_node("fft").is_some());
        assert!(rebuilt.find_node("bands").is_some());
    }

    #[test]
    fn unknown_node_reference_returns_invalid_graph() {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            "relu".to_string(),
            NodeSpec {
                device: "cpu:0".into(),
                op: OpSpec::Op(Op::Relu),
                inputs: vec![InputSpec::FromNode {
                    node: "missing".into(),
                    port: 0,
                }],
                outputs: vec![TensorTypeSpec::from(&frame_type())],
                stateful: false,
            },
        );
        let file = GraphFile {
            version: CURRENT_VERSION,
            sources: vec![],
            nodes,
            sinks: vec![],
            layout: None,
        };

        let err = file.build().unwrap_err();
        assert!(
            matches!(err, GraphFileError::InvalidGraph(errors) if errors.iter().any(|error| matches!(error, GraphError::UnknownNode { name } if name == "missing")))
        );
    }

    #[test]
    fn graph_to_file_preserves_dim_names() {
        let named = TensorType::new(DType::F32, vec![Dim::Fixed(2)], Layout::RowMajor)
            .unwrap()
            .with_dim_names(vec!["samples".into()])
            .unwrap();
        let graph = GraphBuilder::new()
            .source("in", named.clone())
            .add_node("relu")
            .device("cpu:0")
            .op(Op::Relu)
            .input_from_source("in")
            .output(named.clone())
            .done()
            .sink("out", named)
            .from("relu", 0)
            .done()
            .build()
            .unwrap();

        let file = graph_to_file(&graph).unwrap();

        assert_eq!(
            file.sources[0].tensor_type.dim_names,
            Some(vec!["samples".to_string()])
        );
    }

    #[test]
    fn graph_to_file_reports_unconnected_sink() {
        let graph = Graph {
            sources: vec![],
            sinks: vec![SinkPort {
                name: "out".into(),
                tensor_type: frame_type(),
            }],
            nodes: vec![],
            edges: vec![],
            sink_connections: vec![],
        };

        let err = graph_to_file(&graph).unwrap_err();
        assert!(
            matches!(err, GraphFileError::InvalidGraph(errors) if errors.iter().any(|error| matches!(error, GraphError::UnconnectedSink { sink } if sink == "out")))
        );
    }

    #[test]
    fn graph_to_file_reports_unconnected_input() {
        let graph = Graph {
            sources: vec![SourcePort {
                name: "in".into(),
                tensor_type: frame_type(),
            }],
            sinks: vec![],
            nodes: vec![Node {
                id: NodeId(0),
                name: "relu".into(),
                device: crate::types::DeviceId::new("cpu:0"),
                kind: NodeKind::Op(Op::Relu),
                inputs: vec![frame_type()],
                outputs: vec![frame_type()],
                stateful: false,
            }],
            edges: vec![],
            sink_connections: vec![],
        };

        let err = graph_to_file(&graph).unwrap_err();
        assert!(
            matches!(err, GraphFileError::InvalidGraph(errors) if errors.iter().any(|error| matches!(error, GraphError::UnconnectedPort { node, port } if node == "relu" && *port == 0)))
        );
    }

    #[test]
    fn graph_to_file_reports_unknown_source_edge() {
        let graph = Graph {
            sources: vec![],
            sinks: vec![],
            nodes: vec![Node {
                id: NodeId(0),
                name: "relu".into(),
                device: crate::types::DeviceId::new("cpu:0"),
                kind: NodeKind::Op(Op::Relu),
                inputs: vec![frame_type()],
                outputs: vec![frame_type()],
                stateful: false,
            }],
            edges: vec![Edge {
                from: EdgeSource::Source(99),
                to: PortRef {
                    node: NodeId(0),
                    port: 0,
                },
            }],
            sink_connections: vec![],
        };

        let err = graph_to_file(&graph).unwrap_err();
        assert!(
            matches!(err, GraphFileError::InvalidGraph(errors) if errors.iter().any(|error| matches!(error, GraphError::UnknownSource { name, .. } if name == "source[99]")))
        );
    }

    #[test]
    fn graph_to_file_reports_unknown_node_edge() {
        let graph = Graph {
            sources: vec![],
            sinks: vec![],
            nodes: vec![Node {
                id: NodeId(0),
                name: "relu".into(),
                device: crate::types::DeviceId::new("cpu:0"),
                kind: NodeKind::Op(Op::Relu),
                inputs: vec![frame_type()],
                outputs: vec![frame_type()],
                stateful: false,
            }],
            edges: vec![Edge {
                from: EdgeSource::Node(PortRef {
                    node: NodeId(99),
                    port: 0,
                }),
                to: PortRef {
                    node: NodeId(0),
                    port: 0,
                },
            }],
            sink_connections: vec![],
        };

        let err = graph_to_file(&graph).unwrap_err();
        assert!(
            matches!(err, GraphFileError::InvalidGraph(errors) if errors.iter().any(|error| matches!(error, GraphError::UnknownNode { name } if name == "node_99")))
        );
    }
}
