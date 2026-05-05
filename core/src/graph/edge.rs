use std::fmt;

use super::node::NodeId;

// ── PortRef ───────────────────────────────────────────────────────────────────

/// A reference to a specific output port on a specific node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortRef {
    /// The node that owns this port.
    pub node: NodeId,
    /// Zero-based index into the node's output port list.
    pub port: usize,
}

// ── EdgeSource ────────────────────────────────────────────────────────────────

/// Where the data on an edge originates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EdgeSource {
    /// Data flows from a named graph source port (index into
    /// [`Graph::sources`](super::Graph::sources)).
    Source(usize),
    /// Data flows from a node's output port.
    Node(PortRef),
}

// ── Edge ──────────────────────────────────────────────────────────────────────

/// A directed data edge from an [`EdgeSource`] to a node's input port.
///
/// Edges are the only mutable-state-free connection between nodes and boundary
/// ports. The graph is a DAG of edges; cycles are rejected at build time.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Where data originates.
    pub from: EdgeSource,
    /// The node input port that receives the data.
    pub to: PortRef,
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.from {
            EdgeSource::Source(idx) => {
                write!(f, "source[{idx}] → {}[{}]", self.to.node, self.to.port)
            }
            EdgeSource::Node(pr) => write!(
                f,
                "{}[{}] → {}[{}]",
                pr.node, pr.port, self.to.node, self.to.port
            ),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn port(node: usize, port: usize) -> PortRef {
        PortRef {
            node: NodeId(node),
            port,
        }
    }

    #[test]
    fn port_ref_equality() {
        assert_eq!(port(0, 0), port(0, 0));
        assert_ne!(port(0, 0), port(0, 1));
        assert_ne!(port(0, 0), port(1, 0));
    }

    #[test]
    fn port_ref_clone() {
        let p = port(3, 2);
        assert_eq!(p.clone(), p);
    }

    #[test]
    fn edge_source_source_variant() {
        let s = EdgeSource::Source(2);
        assert_eq!(s, EdgeSource::Source(2));
        assert_ne!(s, EdgeSource::Source(3));
    }

    #[test]
    fn edge_source_node_variant() {
        let a = EdgeSource::Node(port(1, 0));
        let b = EdgeSource::Node(port(1, 0));
        let c = EdgeSource::Node(port(2, 0));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn edge_display_from_source() {
        let e = Edge {
            from: EdgeSource::Source(0),
            to: port(1, 2),
        };
        let s = format!("{e}");
        assert!(s.contains("source[0]"));
        assert!(s.contains("node_1"));
        assert!(s.contains('2'.to_string().as_str()));
    }

    #[test]
    fn edge_display_from_node() {
        let e = Edge {
            from: EdgeSource::Node(port(0, 1)),
            to: port(2, 0),
        };
        let s = format!("{e}");
        assert!(s.contains("node_0"));
        assert!(s.contains("node_2"));
    }

    #[test]
    fn edge_debug() {
        let e = Edge {
            from: EdgeSource::Source(0),
            to: port(0, 0),
        };
        let s = format!("{e:?}");
        assert!(s.contains("Edge"));
    }

    #[test]
    fn edge_clone() {
        let e = Edge {
            from: EdgeSource::Source(1),
            to: port(2, 3),
        };
        let c = e.clone();
        assert_eq!(c.from, e.from);
        assert_eq!(c.to, e.to);
    }
}
