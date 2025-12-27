/*!
 * Causal Graph - Build and analyze event causality relationships
 *
 * Transforms correlated event traces into analyzable causal graphs,
 * enabling queries like "Did X cause Y?" and "What's the root cause?"
 *
 * **Revolutionary Feature**: From event logs to causal understanding.
 */

use super::types::Trace;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use chrono::{DateTime, Utc};

/// Causal graph representing event relationships
///
/// Built from trace events with correlation IDs and parent relationships.
/// Enables causal analysis, root cause detection, and critical path finding.
///
/// # Example
///
/// ```ignore
/// use symthaea::observability::causal_graph::CausalGraph;
/// use symthaea::observability::types::Trace;
///
/// let trace = Trace::load_from_file("trace.json")?;
/// let graph = CausalGraph::from_trace(&trace);
///
/// // Find what caused an event
/// let causes = graph.find_causes("evt_003");
/// // ["evt_001", "evt_002"]
///
/// // Get complete causal chain
/// let chain = graph.get_causal_chain("evt_004");
/// // [evt_001 -> evt_002 -> evt_003 -> evt_004]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// All nodes (events) in the graph
    pub nodes: HashMap<String, CausalNode>,

    /// All edges (causal relationships)
    pub edges: Vec<CausalEdge>,

    /// Root events (no parents)
    pub root_events: Vec<String>,

    /// Leaf events (no children)
    pub leaf_events: Vec<String>,
}

/// Node in the causal graph (represents an event)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// Event ID
    pub id: String,

    /// Event type (e.g., "phi_measurement", "router_selection")
    pub event_type: String,

    /// When the event occurred
    pub timestamp: DateTime<Utc>,

    /// Correlation ID (groups related events)
    pub correlation_id: Option<String>,

    /// Parent event ID (direct cause)
    pub parent_id: Option<String>,

    /// Duration in milliseconds (if available)
    pub duration_ms: Option<u64>,

    /// Event-specific metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Edge in the causal graph (represents causation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Source event ID (cause)
    pub from: String,

    /// Target event ID (effect)
    pub to: String,

    /// Strength of causal relationship (0.0-1.0)
    pub strength: f64,

    /// Type of causal relationship
    pub edge_type: EdgeType,
}

/// Type of causal relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Explicitly marked parent-child relationship
    Direct,

    /// Inferred from correlation and timing
    Inferred,

    /// Just temporal proximity (weak causation)
    Temporal,
}

impl CausalGraph {
    /// Create empty causal graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            root_events: Vec::new(),
            leaf_events: Vec::new(),
        }
    }

    /// Add a node to the graph (for streaming analysis)
    pub fn add_node(&mut self, node: CausalNode) {
        let has_parent = node.parent_id.is_some();
        let node_id = node.id.clone();

        self.nodes.insert(node.id.clone(), node);

        // Update root/leaf tracking
        if !has_parent && !self.root_events.contains(&node_id) {
            self.root_events.push(node_id.clone());
        }
        if !self.leaf_events.contains(&node_id) {
            self.leaf_events.push(node_id);
        }
    }

    /// Add an edge to the graph (for streaming analysis)
    pub fn add_edge(&mut self, edge: CausalEdge) {
        // Remove "from" node from leaf events (it now has children)
        self.leaf_events.retain(|id| id != &edge.from);

        // Remove "to" node from root events (it now has a parent)
        self.root_events.retain(|id| id != &edge.to);

        self.edges.push(edge);
    }

    /// Build causal graph from trace
    ///
    /// Analyzes event relationships based on:
    /// - Explicit parent_id fields (Direct edges)
    /// - Correlation IDs (Inferred edges)
    /// - Temporal proximity (Temporal edges)
    pub fn from_trace(trace: &Trace) -> Self {
        let mut graph = Self::new();

        // Phase 1: Create nodes from events
        for event in &trace.events {
            let node = Self::event_to_node(event);
            graph.nodes.insert(node.id.clone(), node);
        }

        // Phase 2: Create edges from parent relationships
        for event in &trace.events {
            if let Some(event_id) = Self::extract_event_id(event) {
                if let Some(parent_id) = Self::extract_parent_id(event) {
                    // Direct causal edge from parent to this event
                    graph.edges.push(CausalEdge {
                        from: parent_id,
                        to: event_id,
                        strength: 1.0, // Direct relationship = 100% strength
                        edge_type: EdgeType::Direct,
                    });
                }
            }
        }

        // Phase 3: Infer additional edges from correlation and timing
        graph.infer_temporal_edges(trace);

        // Phase 4: Identify root and leaf events
        graph.identify_roots_and_leaves();

        graph
    }

    /// Convert trace event to causal node
    fn event_to_node(event: &super::types::Event) -> CausalNode {
        CausalNode {
            id: Self::extract_event_id(event).unwrap_or_else(|| format!("evt_{}", uuid::Uuid::new_v4())),
            event_type: event.event_type.clone(),
            timestamp: event.timestamp,
            correlation_id: Self::extract_correlation_id(event),
            parent_id: Self::extract_parent_id(event),
            duration_ms: Self::extract_duration(event),
            metadata: Self::extract_metadata(event),
        }
    }

    /// Extract event ID from event data
    fn extract_event_id(event: &super::types::Event) -> Option<String> {
        event.data.get("metadata")
            .and_then(|m| m.get("id"))
            .and_then(|id| id.as_str())
            .map(String::from)
            .or_else(|| {
                event.data.get("id")
                    .and_then(|id| id.as_str())
                    .map(String::from)
            })
    }

    /// Extract parent ID from event data
    fn extract_parent_id(event: &super::types::Event) -> Option<String> {
        event.data.get("metadata")
            .and_then(|m| m.get("parent_id"))
            .and_then(|p| p.as_str())
            .map(String::from)
            .or_else(|| {
                event.data.get("parent_id")
                    .and_then(|p| p.as_str())
                    .map(String::from)
            })
    }

    /// Extract correlation ID from event data
    fn extract_correlation_id(event: &super::types::Event) -> Option<String> {
        event.data.get("metadata")
            .and_then(|m| m.get("correlation_id"))
            .and_then(|c| c.as_str())
            .map(String::from)
            .or_else(|| {
                event.data.get("correlation_id")
                    .and_then(|c| c.as_str())
                    .map(String::from)
            })
    }

    /// Extract duration from event data
    fn extract_duration(event: &super::types::Event) -> Option<u64> {
        event.data.get("metadata")
            .and_then(|m| m.get("duration_ms"))
            .and_then(|d| d.as_u64())
            .or_else(|| {
                event.data.get("duration_ms")
                    .and_then(|d| d.as_u64())
            })
    }

    /// Extract metadata from event
    fn extract_metadata(event: &super::types::Event) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        // Include all event data fields except metadata itself
        for (key, value) in event.data.as_object().unwrap_or(&serde_json::Map::new()) {
            if key != "metadata" {
                metadata.insert(key.clone(), value.clone());
            }
        }

        metadata
    }

    /// Infer temporal edges from event timing
    fn infer_temporal_edges(&mut self, trace: &Trace) {
        // Group events by correlation ID
        let mut correlation_groups: HashMap<String, Vec<String>> = HashMap::new();

        for (id, node) in &self.nodes {
            if let Some(corr_id) = &node.correlation_id {
                correlation_groups.entry(corr_id.clone())
                    .or_insert_with(Vec::new)
                    .push(id.clone());
            }
        }

        // For each correlation group, infer temporal relationships
        for (_corr_id, event_ids) in correlation_groups {
            // Sort by timestamp
            let mut sorted_events: Vec<_> = event_ids.iter()
                .filter_map(|id| self.nodes.get(id).map(|n| (id.clone(), n.timestamp)))
                .collect();
            sorted_events.sort_by_key(|(_, ts)| *ts);

            // Create weak temporal edges between consecutive events
            for window in sorted_events.windows(2) {
                let (from_id, from_ts) = &window[0];
                let (to_id, to_ts) = &window[1];

                // Only create temporal edge if there's no direct edge already
                let has_direct_edge = self.edges.iter().any(|e| {
                    &e.from == from_id && &e.to == to_id && e.edge_type == EdgeType::Direct
                });

                if !has_direct_edge {
                    // Strength decreases with time gap
                    let time_gap_ms = (*to_ts - *from_ts).num_milliseconds() as f64;
                    let strength = (1000.0 / (time_gap_ms + 1000.0)).min(0.5); // Max 0.5 for temporal

                    self.edges.push(CausalEdge {
                        from: from_id.clone(),
                        to: to_id.clone(),
                        strength,
                        edge_type: EdgeType::Temporal,
                    });
                }
            }
        }
    }

    /// Identify root and leaf events
    fn identify_roots_and_leaves(&mut self) {
        let has_parent: HashSet<_> = self.edges.iter().map(|e| e.to.clone()).collect();
        let has_child: HashSet<_> = self.edges.iter().map(|e| e.from.clone()).collect();

        for id in self.nodes.keys() {
            if !has_parent.contains(id) {
                self.root_events.push(id.clone());
            }
            if !has_child.contains(id) {
                self.leaf_events.push(id.clone());
            }
        }
    }

    /// Find all direct causes of an event
    ///
    /// Returns event IDs that directly caused the given event.
    pub fn find_causes(&self, event_id: &str) -> Vec<&CausalNode> {
        self.edges.iter()
            .filter(|e| e.to == event_id)
            .filter_map(|e| self.nodes.get(&e.from))
            .collect()
    }

    /// Find all direct effects of an event
    ///
    /// Returns event IDs that were directly caused by the given event.
    pub fn find_effects(&self, event_id: &str) -> Vec<&CausalNode> {
        self.edges.iter()
            .filter(|e| e.from == event_id)
            .filter_map(|e| self.nodes.get(&e.to))
            .collect()
    }

    /// Get complete causal chain from root to event
    ///
    /// Returns all events in the causal path from root to the given event.
    pub fn get_causal_chain(&self, event_id: &str) -> Vec<&CausalNode> {
        let mut chain = Vec::new();
        let mut current_id = event_id;

        // Walk backwards from event to root
        while let Some(node) = self.nodes.get(current_id) {
            chain.push(node);

            // Find parent (direct edge coming into this node)
            match self.edges.iter()
                .filter(|e| e.to == current_id && e.edge_type == EdgeType::Direct)
                .next()
            {
                Some(edge) => current_id = &edge.from,
                None => break,
            }
        }

        // Reverse to get root -> event order
        chain.reverse();
        chain
    }

    /// Get all root causes of an event (transitive closure)
    pub fn find_root_causes(&self, event_id: &str) -> Vec<&CausalNode> {
        let mut roots = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(event_id);

        while let Some(current) = queue.pop_front() {
            if visited.contains(current) {
                continue;
            }
            visited.insert(current);

            let causes: Vec<_> = self.edges.iter()
                .filter(|e| e.to == current)
                .map(|e| e.from.as_str())
                .collect();

            if causes.is_empty() {
                // This is a root
                if let Some(node) = self.nodes.get(current) {
                    roots.push(node);
                }
            } else {
                // Continue searching
                for cause in causes {
                    queue.push_back(cause);
                }
            }
        }

        roots
    }

    /// Find critical path (longest dependency chain)
    ///
    /// Returns the chain of events with the longest total duration.
    pub fn find_critical_path(&self) -> Vec<&CausalNode> {
        // Use dynamic programming to find longest path
        let mut longest_paths: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize roots
        for root_id in &self.root_events {
            longest_paths.insert(root_id.clone(), vec![root_id.clone()]);
        }

        // Topological sort and process
        let sorted = self.topological_sort();

        for id in sorted {
            let causes = self.find_causes(&id);

            if let Some(best_cause) = causes.iter()
                .max_by_key(|c| {
                    longest_paths.get(&c.id)
                        .map(|path| path.len())
                        .unwrap_or(0)
                })
            {
                let mut new_path = longest_paths.get(&best_cause.id)
                    .cloned()
                    .unwrap_or_else(|| vec![best_cause.id.clone()]);
                new_path.push(id.clone());
                longest_paths.insert(id.clone(), new_path);
            }
        }

        // Find overall longest path
        let longest = longest_paths.values()
            .max_by_key(|path| path.len())
            .cloned()
            .unwrap_or_default();

        longest.iter()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    /// Topological sort of events
    fn topological_sort(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        for id in self.nodes.keys() {
            self.topological_visit(id, &mut visited, &mut visiting, &mut result);
        }

        result
    }

    fn topological_visit(
        &self,
        id: &str,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
        result: &mut Vec<String>,
    ) {
        if visited.contains(id) {
            return;
        }

        if visiting.contains(id) {
            // Cycle detected - skip
            return;
        }

        visiting.insert(id.to_string());

        // Visit all parents first
        for edge in &self.edges {
            if edge.to == id {
                self.topological_visit(&edge.from, visited, visiting, result);
            }
        }

        visiting.remove(id);
        visited.insert(id.to_string());
        result.push(id.to_string());
    }

    /// Check if X caused Y (direct or indirect)
    pub fn did_cause(&self, cause_id: &str, effect_id: &str) -> CausalAnswer {
        // Check for direct edge
        if let Some(edge) = self.edges.iter()
            .find(|e| e.from == cause_id && e.to == effect_id)
        {
            return CausalAnswer::DirectCause { strength: edge.strength };
        }

        // Check for indirect path
        if let Some(path) = self.find_path(cause_id, effect_id) {
            let strength = self.calculate_path_strength(&path);
            return CausalAnswer::IndirectCause { path, strength };
        }

        CausalAnswer::NotCaused
    }

    /// Find path from cause to effect
    fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<String, String> = HashMap::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![current.clone()];
                let mut node = current;

                while let Some(parent) = parent_map.get(&node) {
                    path.push(parent.clone());
                    node = parent.clone();
                }

                path.reverse();
                return Some(path);
            }

            // Explore children
            for edge in &self.edges {
                if edge.from == current && !visited.contains(&edge.to) {
                    visited.insert(edge.to.clone());
                    parent_map.insert(edge.to.clone(), current.clone());
                    queue.push_back(edge.to.clone());
                }
            }
        }

        None
    }

    /// Calculate strength of a causal path
    fn calculate_path_strength(&self, path: &[String]) -> f64 {
        let mut total_strength = 1.0;

        for window in path.windows(2) {
            if let Some(edge) = self.edges.iter()
                .find(|e| e.from == window[0] && e.to == window[1])
            {
                total_strength *= edge.strength;
            }
        }

        total_strength
    }

    /// Export to Mermaid diagram format
    pub fn to_mermaid(&self) -> String {
        let mut output = String::from("graph TD\n");

        // Add nodes
        for (id, node) in &self.nodes {
            let label = format!("{}\\n{}", node.event_type, id);
            output.push_str(&format!("    {}[\"{}\"]\n", Self::sanitize_id(id), label));
        }

        // Add edges
        for edge in &self.edges {
            let style = match edge.edge_type {
                EdgeType::Direct => "-->",
                EdgeType::Inferred => "-.->",
                EdgeType::Temporal => "--.",
            };
            output.push_str(&format!(
                "    {} {} {}\n",
                Self::sanitize_id(&edge.from),
                style,
                Self::sanitize_id(&edge.to)
            ));
        }

        output
    }

    /// Export to GraphViz DOT format
    pub fn to_dot(&self) -> String {
        let mut output = String::from("digraph CausalGraph {\n");
        output.push_str("    rankdir=LR;\n");
        output.push_str("    node [shape=box];\n");

        // Add nodes
        for (id, node) in &self.nodes {
            let label = format!("{}\\n{}", node.event_type, id);
            output.push_str(&format!(
                "    \"{}\" [label=\"{}\"];\n",
                id, label
            ));
        }

        // Add edges
        for edge in &self.edges {
            let style = match edge.edge_type {
                EdgeType::Direct => "solid",
                EdgeType::Inferred => "dashed",
                EdgeType::Temporal => "dotted",
            };
            output.push_str(&format!(
                "    \"{}\" -> \"{}\" [style={}, label=\"{:.2}\"];\n",
                edge.from, edge.to, style, edge.strength
            ));
        }

        output.push_str("}\n");
        output
    }

    /// Sanitize ID for Mermaid (remove hyphens)
    fn sanitize_id(id: &str) -> String {
        id.replace('-', "_")
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Answer to "Did X cause Y?" query
#[derive(Debug, Clone, PartialEq)]
pub enum CausalAnswer {
    /// X directly caused Y with given strength
    DirectCause { strength: f64 },

    /// X indirectly caused Y through the given path
    IndirectCause { path: Vec<String>, strength: f64 },

    /// X did not cause Y
    NotCaused,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::types::{Event, Trace};

    fn create_test_trace() -> Trace {
        let mut trace = Trace::new("test_session".to_string());

        // Event 1: Root (phi measurement)
        trace.events.push(Event {
            timestamp: Utc::now(),
            event_type: "phi_measurement".to_string(),
            data: serde_json::json!({
                "id": "evt_001",
                "correlation_id": "req_123",
                "parent_id": null,
                "phi": 0.65
            }),
        });

        // Event 2: Child of evt_001 (routing)
        trace.events.push(Event {
            timestamp: Utc::now(),
            event_type: "router_selection".to_string(),
            data: serde_json::json!({
                "id": "evt_002",
                "correlation_id": "req_123",
                "parent_id": "evt_001",
                "selected_router": "Standard"
            }),
        });

        // Event 3: Child of evt_002 (response)
        trace.events.push(Event {
            timestamp: Utc::now(),
            event_type: "language_step".to_string(),
            data: serde_json::json!({
                "id": "evt_003",
                "correlation_id": "req_123",
                "parent_id": "evt_002",
                "step_type": "response_generation"
            }),
        });

        trace
    }

    #[test]
    fn test_causal_graph_construction() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        assert_eq!(graph.nodes.len(), 3);
        assert!(!graph.edges.is_empty());
        assert_eq!(graph.root_events.len(), 1);
        assert_eq!(graph.root_events[0], "evt_001");
    }

    #[test]
    fn test_find_causes() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let causes = graph.find_causes("evt_003");
        assert_eq!(causes.len(), 1);
        assert_eq!(causes[0].id, "evt_002");
    }

    #[test]
    fn test_find_effects() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let effects = graph.find_effects("evt_001");
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].id, "evt_002");
    }

    #[test]
    fn test_causal_chain() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let chain = graph.get_causal_chain("evt_003");
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].id, "evt_001");
        assert_eq!(chain[1].id, "evt_002");
        assert_eq!(chain[2].id, "evt_003");
    }

    #[test]
    fn test_did_cause_direct() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let answer = graph.did_cause("evt_001", "evt_002");
        assert!(matches!(answer, CausalAnswer::DirectCause { .. }));
    }

    #[test]
    fn test_did_cause_indirect() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let answer = graph.did_cause("evt_001", "evt_003");
        match answer {
            CausalAnswer::IndirectCause { path, .. } => {
                assert_eq!(path.len(), 3);
                assert_eq!(path[0], "evt_001");
                assert_eq!(path[2], "evt_003");
            }
            _ => panic!("Expected indirect cause"),
        }
    }

    #[test]
    fn test_did_not_cause() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let answer = graph.did_cause("evt_003", "evt_001");
        assert_eq!(answer, CausalAnswer::NotCaused);
    }

    #[test]
    fn test_mermaid_export() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("evt_001"));
        assert!(mermaid.contains("evt_002"));
        assert!(mermaid.contains("evt_003"));
    }

    #[test]
    fn test_dot_export() {
        let trace = create_test_trace();
        let graph = CausalGraph::from_trace(&trace);

        let dot = graph.to_dot();
        assert!(dot.contains("digraph CausalGraph"));
        assert!(dot.contains("evt_001"));
        assert!(dot.contains("->"));
    }
}
