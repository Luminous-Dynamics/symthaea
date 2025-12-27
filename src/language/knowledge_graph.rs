//! Knowledge Graph - World Knowledge for Reasoning
//!
//! This module provides an in-memory knowledge graph for storing and querying
//! facts about the world. Unlike LLMs which have implicit knowledge buried in
//! weights, our knowledge is explicit, queryable, and editable.
//!
//! Features:
//! - Hierarchical concepts (IS-A relationships)
//! - Property inheritance
//! - Relationship traversal
//! - Common-sense facts

use std::collections::{HashMap, HashSet};
use crate::hdc::binary_hv::HV16;

// ============================================================================
// Core Types
// ============================================================================

/// A node in the knowledge graph
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: NodeId,
    pub name: String,
    pub encoding: HV16,
    pub node_type: NodeType,
    pub properties: HashMap<String, PropertyValue>,
    pub aliases: Vec<String>,
}

/// Node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Types of nodes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    /// Category or class (e.g., Animal, Vehicle)
    Category,
    /// Instance (e.g., Fido the dog)
    Instance,
    /// Property (e.g., color, size)
    Property,
    /// Action/Verb (e.g., eat, sleep)
    Action,
    /// Abstract concept (e.g., love, justice)
    Abstract,
    /// Location (e.g., Paris, kitchen)
    Location,
    /// Time period (e.g., morning, 2025)
    TimePeriod,
}

/// Property value types
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    NodeRef(NodeId),
    List(Vec<PropertyValue>),
}

/// An edge in the knowledge graph
#[derive(Debug, Clone)]
pub struct KnowledgeEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub weight: f32,  // Confidence/strength
    pub source: KnowledgeSource,
}

/// Types of relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    // Taxonomic
    IsA,              // Dog IS_A Mammal
    InstanceOf,       // Fido INSTANCE_OF Dog
    SubclassOf,       // Mammal SUBCLASS_OF Animal

    // Part-whole
    HasPart,          // Car HAS_PART Wheel
    PartOf,           // Wheel PART_OF Car
    MemberOf,         // Paris MEMBER_OF France

    // Properties
    HasProperty,      // Sky HAS_PROPERTY Blue
    PropertyOf,       // Blue PROPERTY_OF Sky

    // Spatial
    LocatedIn,        // Paris LOCATED_IN France
    Contains,         // France CONTAINS Paris
    NearBy,           // Paris NEARBY Eiffel Tower

    // Temporal
    Before,           // Spring BEFORE Summer
    After,            // Summer AFTER Spring
    During,           // Lunch DURING Day

    // Causal
    Causes,           // Rain CAUSES Wet
    CausedBy,         // Wet CAUSED_BY Rain
    Enables,          // Fuel ENABLES Driving

    // Actions
    CapableOf,        // Bird CAPABLE_OF Fly
    UsedFor,          // Hammer USED_FOR Nail
    Requires,         // Cooking REQUIRES Heat

    // Abstract
    Similar,          // Happy SIMILAR Joyful
    Opposite,         // Hot OPPOSITE Cold
    RelatedTo,        // Love RELATED_TO Heart
}

impl EdgeType {
    /// Get the inverse edge type
    pub fn inverse(&self) -> Option<EdgeType> {
        match self {
            EdgeType::IsA => None,
            EdgeType::HasPart => Some(EdgeType::PartOf),
            EdgeType::PartOf => Some(EdgeType::HasPart),
            EdgeType::LocatedIn => Some(EdgeType::Contains),
            EdgeType::Contains => Some(EdgeType::LocatedIn),
            EdgeType::Before => Some(EdgeType::After),
            EdgeType::After => Some(EdgeType::Before),
            EdgeType::Causes => Some(EdgeType::CausedBy),
            EdgeType::CausedBy => Some(EdgeType::Causes),
            EdgeType::Similar => Some(EdgeType::Similar),
            EdgeType::Opposite => Some(EdgeType::Opposite),
            _ => None,
        }
    }
}

/// Source of knowledge
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnowledgeSource {
    /// Built-in common sense
    BuiltIn,
    /// Learned from user
    Learned,
    /// Inferred from reasoning
    Inferred,
    /// External source
    External(String),
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub nodes: Vec<KnowledgeNode>,
    pub edges: Vec<KnowledgeEdge>,
    pub path: Vec<NodeId>,
}

// ============================================================================
// Knowledge Graph
// ============================================================================

/// The main knowledge graph
pub struct KnowledgeGraph {
    nodes: HashMap<NodeId, KnowledgeNode>,
    edges: Vec<KnowledgeEdge>,
    name_index: HashMap<String, NodeId>,
    next_id: u64,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            name_index: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create with common-sense knowledge
    pub fn with_common_sense() -> Self {
        let mut kg = Self::new();
        kg.initialize_common_sense();
        kg
    }

    /// Initialize with basic common-sense facts
    fn initialize_common_sense(&mut self) {
        // Categories
        let thing = self.add_node("thing", NodeType::Category);
        let living_thing = self.add_node("living thing", NodeType::Category);
        let animal = self.add_node("animal", NodeType::Category);
        let mammal = self.add_node("mammal", NodeType::Category);
        let bird = self.add_node("bird", NodeType::Category);
        let plant = self.add_node("plant", NodeType::Category);
        let object = self.add_node("object", NodeType::Category);
        let vehicle = self.add_node("vehicle", NodeType::Category);
        let place = self.add_node("place", NodeType::Category);
        let person = self.add_node("person", NodeType::Category);
        let emotion = self.add_node("emotion", NodeType::Category);
        let time = self.add_node("time", NodeType::Category);

        // Taxonomic hierarchy
        self.add_edge(living_thing, EdgeType::IsA, thing);
        self.add_edge(animal, EdgeType::IsA, living_thing);
        self.add_edge(mammal, EdgeType::IsA, animal);
        self.add_edge(bird, EdgeType::IsA, animal);
        self.add_edge(plant, EdgeType::IsA, living_thing);
        self.add_edge(object, EdgeType::IsA, thing);
        self.add_edge(vehicle, EdgeType::IsA, object);
        self.add_edge(person, EdgeType::IsA, mammal);

        // Common animals
        let dog = self.add_node("dog", NodeType::Category);
        let cat = self.add_node("cat", NodeType::Category);
        let fish = self.add_node("fish", NodeType::Category);
        self.add_edge(dog, EdgeType::IsA, mammal);
        self.add_edge(cat, EdgeType::IsA, mammal);
        self.add_edge(fish, EdgeType::IsA, animal);

        // Properties
        self.set_property(dog, "can_bark", PropertyValue::Boolean(true));
        self.set_property(cat, "can_purr", PropertyValue::Boolean(true));
        self.set_property(bird, "can_fly", PropertyValue::Boolean(true));
        self.set_property(fish, "lives_in", PropertyValue::String("water".to_string()));

        // Emotions
        let happy = self.add_node("happy", NodeType::Abstract);
        let sad = self.add_node("sad", NodeType::Abstract);
        let angry = self.add_node("angry", NodeType::Abstract);
        let love = self.add_node("love", NodeType::Abstract);
        let fear = self.add_node("fear", NodeType::Abstract);

        self.add_edge(happy, EdgeType::IsA, emotion);
        self.add_edge(sad, EdgeType::IsA, emotion);
        self.add_edge(angry, EdgeType::IsA, emotion);
        self.add_edge(love, EdgeType::IsA, emotion);
        self.add_edge(fear, EdgeType::IsA, emotion);

        self.add_edge(happy, EdgeType::Opposite, sad);
        self.add_edge(love, EdgeType::Opposite, fear);

        // Time
        let morning = self.add_node("morning", NodeType::TimePeriod);
        let afternoon = self.add_node("afternoon", NodeType::TimePeriod);
        let evening = self.add_node("evening", NodeType::TimePeriod);
        let night = self.add_node("night", NodeType::TimePeriod);

        self.add_edge(morning, EdgeType::IsA, time);
        self.add_edge(afternoon, EdgeType::IsA, time);
        self.add_edge(evening, EdgeType::IsA, time);
        self.add_edge(night, EdgeType::IsA, time);

        self.add_edge(morning, EdgeType::Before, afternoon);
        self.add_edge(afternoon, EdgeType::Before, evening);
        self.add_edge(evening, EdgeType::Before, night);

        // Actions and capabilities
        let eat = self.add_node("eat", NodeType::Action);
        let sleep = self.add_node("sleep", NodeType::Action);
        let move_action = self.add_node("move", NodeType::Action);
        let think = self.add_node("think", NodeType::Action);

        self.add_edge(animal, EdgeType::CapableOf, eat);
        self.add_edge(animal, EdgeType::CapableOf, sleep);
        self.add_edge(animal, EdgeType::CapableOf, move_action);
        self.add_edge(person, EdgeType::CapableOf, think);

        // Causal relationships
        let rain = self.add_node("rain", NodeType::Abstract);
        let wet = self.add_node("wet", NodeType::Abstract);
        let fire = self.add_node("fire", NodeType::Abstract);
        let smoke = self.add_node("smoke", NodeType::Abstract);
        let heat = self.add_node("heat", NodeType::Abstract);

        self.add_edge(rain, EdgeType::Causes, wet);
        self.add_edge(fire, EdgeType::Causes, smoke);
        self.add_edge(fire, EdgeType::Causes, heat);

        // Locations
        let earth = self.add_node("earth", NodeType::Location);
        let continent = self.add_node("continent", NodeType::Location);
        let country = self.add_node("country", NodeType::Location);
        let city = self.add_node("city", NodeType::Location);

        self.add_edge(continent, EdgeType::IsA, place);
        self.add_edge(country, EdgeType::IsA, place);
        self.add_edge(city, EdgeType::IsA, place);
        self.add_edge(continent, EdgeType::PartOf, earth);
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, name: &str, node_type: NodeType) -> NodeId {
        // Check if exists
        if let Some(&id) = self.name_index.get(&name.to_lowercase()) {
            return id;
        }

        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let seed = name.bytes().fold(self.next_id, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

        let node = KnowledgeNode {
            id,
            name: name.to_string(),
            encoding: HV16::random(seed),
            node_type,
            properties: HashMap::new(),
            aliases: Vec::new(),
        };

        self.nodes.insert(id, node);
        self.name_index.insert(name.to_lowercase(), id);

        id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&KnowledgeNode> {
        self.nodes.get(&id)
    }

    /// Get a node by name
    pub fn get_node_by_name(&self, name: &str) -> Option<&KnowledgeNode> {
        self.name_index.get(&name.to_lowercase())
            .and_then(|id| self.nodes.get(id))
    }

    /// Get node ID by name
    pub fn get_id(&self, name: &str) -> Option<NodeId> {
        self.name_index.get(&name.to_lowercase()).copied()
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: NodeId, edge_type: EdgeType, to: NodeId) {
        self.edges.push(KnowledgeEdge {
            from,
            to,
            edge_type,
            weight: 1.0,
            source: KnowledgeSource::BuiltIn,
        });
    }

    /// Add an edge with weight and source
    pub fn add_edge_with_meta(&mut self, from: NodeId, edge_type: EdgeType, to: NodeId, weight: f32, source: KnowledgeSource) {
        self.edges.push(KnowledgeEdge {
            from,
            to,
            edge_type,
            weight,
            source,
        });
    }

    /// Set a property on a node
    pub fn set_property(&mut self, node: NodeId, key: &str, value: PropertyValue) {
        if let Some(n) = self.nodes.get_mut(&node) {
            n.properties.insert(key.to_string(), value);
        }
    }

    /// Get a property from a node
    pub fn get_property(&self, node: NodeId, key: &str) -> Option<&PropertyValue> {
        self.nodes.get(&node)
            .and_then(|n| n.properties.get(key))
    }

    /// Check if a relationship exists
    pub fn has_edge(&self, from: NodeId, edge_type: &EdgeType, to: NodeId) -> bool {
        self.edges.iter().any(|e| e.from == from && &e.edge_type == edge_type && e.to == to)
    }

    /// Get all edges from a node
    pub fn edges_from(&self, from: NodeId) -> Vec<&KnowledgeEdge> {
        self.edges.iter().filter(|e| e.from == from).collect()
    }

    /// Get all edges to a node
    pub fn edges_to(&self, to: NodeId) -> Vec<&KnowledgeEdge> {
        self.edges.iter().filter(|e| e.to == to).collect()
    }

    /// Get edges of a specific type from a node
    pub fn edges_of_type(&self, from: NodeId, edge_type: &EdgeType) -> Vec<&KnowledgeEdge> {
        self.edges.iter()
            .filter(|e| e.from == from && &e.edge_type == edge_type)
            .collect()
    }

    /// Check if X is-a Y (including through inheritance)
    pub fn is_a(&self, x: NodeId, y: NodeId) -> bool {
        if x == y {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = vec![x];

        while let Some(current) = queue.pop() {
            if current == y {
                return true;
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            // Follow IS_A edges
            for edge in self.edges_of_type(current, &EdgeType::IsA) {
                queue.push(edge.to);
            }
        }

        false
    }

    /// Get all ancestors (parents, grandparents, etc.)
    pub fn ancestors(&self, node: NodeId) -> Vec<NodeId> {
        let mut ancestors = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![node];

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for edge in self.edges_of_type(current, &EdgeType::IsA) {
                ancestors.push(edge.to);
                queue.push(edge.to);
            }
        }

        ancestors
    }

    /// Get all descendants (children, grandchildren, etc.)
    pub fn descendants(&self, node: NodeId) -> Vec<NodeId> {
        let mut descendants = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![node];

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for edge in &self.edges {
                if edge.edge_type == EdgeType::IsA && edge.to == current {
                    descendants.push(edge.from);
                    queue.push(edge.from);
                }
            }
        }

        descendants
    }

    /// Get inherited properties (from node and all ancestors)
    pub fn inherited_properties(&self, node: NodeId) -> HashMap<String, PropertyValue> {
        let mut props = HashMap::new();

        // Collect from ancestors first (so node's own properties override)
        let mut ancestors = self.ancestors(node);
        ancestors.reverse();  // Process from root down

        for ancestor in ancestors {
            if let Some(n) = self.nodes.get(&ancestor) {
                for (k, v) in &n.properties {
                    props.insert(k.clone(), v.clone());
                }
            }
        }

        // Node's own properties
        if let Some(n) = self.nodes.get(&node) {
            for (k, v) in &n.properties {
                props.insert(k.clone(), v.clone());
            }
        }

        props
    }

    /// Find path between two nodes
    pub fn find_path(&self, from: NodeId, to: NodeId, max_depth: usize) -> Option<Vec<NodeId>> {
        if from == to {
            return Some(vec![from]);
        }

        let mut visited = HashSet::new();
        let mut queue: Vec<(NodeId, Vec<NodeId>)> = vec![(from, vec![from])];

        while let Some((current, path)) = queue.pop() {
            if path.len() > max_depth {
                continue;
            }

            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for edge in self.edges_from(current) {
                if edge.to == to {
                    let mut full_path = path.clone();
                    full_path.push(to);
                    return Some(full_path);
                }

                if !visited.contains(&edge.to) {
                    let mut new_path = path.clone();
                    new_path.push(edge.to);
                    queue.push((edge.to, new_path));
                }
            }
        }

        None
    }

    /// Query: What IS X?
    pub fn what_is(&self, name: &str) -> Option<String> {
        let node = self.get_node_by_name(name)?;

        let mut result = format!("{} is a {:?}", node.name, node.node_type);

        // Get IS-A parents
        let is_a_edges: Vec<_> = self.edges_of_type(node.id, &EdgeType::IsA);
        if !is_a_edges.is_empty() {
            result.push_str(". It is a ");
            let parents: Vec<_> = is_a_edges.iter()
                .filter_map(|e| self.nodes.get(&e.to).map(|n| n.name.as_str()))
                .collect();
            result.push_str(&parents.join(", "));
        }

        // Get properties
        if !node.properties.is_empty() {
            result.push_str(". Properties: ");
            let props: Vec<_> = node.properties.iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect();
            result.push_str(&props.join(", "));
        }

        Some(result)
    }

    /// Query: What can X do?
    pub fn what_can_do(&self, name: &str) -> Vec<String> {
        let node = match self.get_node_by_name(name) {
            Some(n) => n,
            None => return vec![],
        };

        let mut abilities = Vec::new();

        // Direct capabilities
        for edge in self.edges_of_type(node.id, &EdgeType::CapableOf) {
            if let Some(action) = self.nodes.get(&edge.to) {
                abilities.push(action.name.clone());
            }
        }

        // Inherited capabilities from ancestors
        for ancestor_id in self.ancestors(node.id) {
            for edge in self.edges_of_type(ancestor_id, &EdgeType::CapableOf) {
                if let Some(action) = self.nodes.get(&edge.to) {
                    if !abilities.contains(&action.name) {
                        abilities.push(action.name.clone());
                    }
                }
            }
        }

        abilities
    }

    /// Query: What causes X?
    pub fn what_causes(&self, name: &str) -> Vec<String> {
        let node = match self.get_node_by_name(name) {
            Some(n) => n,
            None => return vec![],
        };

        self.edges.iter()
            .filter(|e| e.to == node.id && e.edge_type == EdgeType::Causes)
            .filter_map(|e| self.nodes.get(&e.from).map(|n| n.name.clone()))
            .collect()
    }

    /// Query: What does X cause?
    pub fn what_results(&self, name: &str) -> Vec<String> {
        let node = match self.get_node_by_name(name) {
            Some(n) => n,
            None => return vec![],
        };

        self.edges_of_type(node.id, &EdgeType::Causes)
            .into_iter()
            .filter_map(|e| self.nodes.get(&e.to).map(|n| n.name.clone()))
            .collect()
    }

    /// Get stats
    pub fn stats(&self) -> KnowledgeStats {
        KnowledgeStats {
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            categories: self.nodes.values().filter(|n| n.node_type == NodeType::Category).count(),
            instances: self.nodes.values().filter(|n| n.node_type == NodeType::Instance).count(),
        }
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::with_common_sense()
    }
}

/// Statistics about knowledge graph
#[derive(Debug, Clone)]
pub struct KnowledgeStats {
    pub nodes: usize,
    pub edges: usize,
    pub categories: usize,
    pub instances: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_creation() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.nodes.len(), 0);
    }

    #[test]
    fn test_common_sense_initialization() {
        let kg = KnowledgeGraph::with_common_sense();
        assert!(kg.nodes.len() > 20);  // Should have many concepts
        assert!(kg.edges.len() > 20);  // Should have many relations
    }

    #[test]
    fn test_add_node() {
        let mut kg = KnowledgeGraph::new();
        let id = kg.add_node("test", NodeType::Category);
        assert!(kg.get_node(id).is_some());
        assert_eq!(kg.get_node(id).unwrap().name, "test");
    }

    #[test]
    fn test_get_by_name() {
        let kg = KnowledgeGraph::with_common_sense();
        assert!(kg.get_node_by_name("dog").is_some());
        assert!(kg.get_node_by_name("DOG").is_some());  // Case insensitive
    }

    #[test]
    fn test_is_a_relationship() {
        let kg = KnowledgeGraph::with_common_sense();
        let dog = kg.get_id("dog").unwrap();
        let mammal = kg.get_id("mammal").unwrap();
        let animal = kg.get_id("animal").unwrap();

        assert!(kg.is_a(dog, mammal));  // Direct
        assert!(kg.is_a(dog, animal));  // Inherited
    }

    #[test]
    fn test_ancestors() {
        let kg = KnowledgeGraph::with_common_sense();
        let dog = kg.get_id("dog").unwrap();

        let ancestors = kg.ancestors(dog);
        assert!(!ancestors.is_empty());

        // Should include mammal and animal
        let ancestor_names: Vec<_> = ancestors.iter()
            .filter_map(|&id| kg.get_node(id).map(|n| n.name.as_str()))
            .collect();
        assert!(ancestor_names.contains(&"mammal"));
    }

    #[test]
    fn test_properties() {
        let kg = KnowledgeGraph::with_common_sense();
        let dog = kg.get_id("dog").unwrap();

        let prop = kg.get_property(dog, "can_bark");
        assert!(prop.is_some());
        assert_eq!(*prop.unwrap(), PropertyValue::Boolean(true));
    }

    #[test]
    fn test_inherited_properties() {
        let mut kg = KnowledgeGraph::new();
        let animal = kg.add_node("animal", NodeType::Category);
        let dog = kg.add_node("dog", NodeType::Category);

        kg.add_edge(dog, EdgeType::IsA, animal);
        kg.set_property(animal, "alive", PropertyValue::Boolean(true));
        kg.set_property(dog, "barks", PropertyValue::Boolean(true));

        let props = kg.inherited_properties(dog);
        assert_eq!(props.len(), 2);
        assert!(props.contains_key("alive"));
        assert!(props.contains_key("barks"));
    }

    #[test]
    fn test_what_is() {
        let kg = KnowledgeGraph::with_common_sense();
        let result = kg.what_is("dog");
        assert!(result.is_some());
        assert!(result.unwrap().contains("mammal"));
    }

    #[test]
    fn test_what_can_do() {
        let kg = KnowledgeGraph::with_common_sense();
        let abilities = kg.what_can_do("dog");
        // Dogs inherit capabilities from animal (eat, sleep, move)
        assert!(!abilities.is_empty());
    }

    #[test]
    fn test_causal_queries() {
        let kg = KnowledgeGraph::with_common_sense();

        let causes = kg.what_causes("wet");
        assert!(causes.contains(&"rain".to_string()));

        let results = kg.what_results("fire");
        assert!(results.contains(&"smoke".to_string()));
    }

    #[test]
    fn test_find_path() {
        let kg = KnowledgeGraph::with_common_sense();
        let dog = kg.get_id("dog").unwrap();
        let animal = kg.get_id("animal").unwrap();

        let path = kg.find_path(dog, animal, 5);
        assert!(path.is_some());
        assert!(path.unwrap().len() >= 2);
    }

    #[test]
    fn test_edge_types() {
        assert_eq!(EdgeType::HasPart.inverse(), Some(EdgeType::PartOf));
        assert_eq!(EdgeType::Similar.inverse(), Some(EdgeType::Similar));
        assert!(EdgeType::IsA.inverse().is_none());
    }

    #[test]
    fn test_stats() {
        let kg = KnowledgeGraph::with_common_sense();
        let stats = kg.stats();
        assert!(stats.nodes > 0);
        assert!(stats.edges > 0);
        assert!(stats.categories > 0);
    }
}
