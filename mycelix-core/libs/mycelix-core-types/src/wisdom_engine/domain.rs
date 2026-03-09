//! Domain System: Hierarchical Knowledge Organization
//!
//! Component 10 adds a comprehensive domain hierarchy for organizing patterns
//! and enabling cross-domain discovery.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::DomainId;

// ==============================================================================
// COMPONENT 10: DOMAIN SYSTEM - Hierarchical Knowledge Organization
// ==============================================================================
//
// Domains provide structured organization for patterns:
// - Hierarchical relationships (web/frontend/react)
// - Cross-domain pattern discovery
// - Domain metadata (criticality, expertise required)
// - Semantic relationships between knowledge areas

/// Unique identifier for a domain

/// Criticality level for a domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DomainCriticality {
    /// Low criticality - mistakes are easily recoverable
    Low,
    /// Normal criticality - standard care required
    Normal,
    /// High criticality - requires careful validation
    High,
    /// Critical - mistakes can cause significant harm
    Critical,
}

impl Default for DomainCriticality {
    fn default() -> Self {
        Self::Normal
    }
}

/// A domain represents a knowledge area or problem space
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Domain {
    /// Unique domain identifier
    pub id: DomainId,

    /// Human-readable name (e.g., "frontend", "security")
    pub name: String,

    /// Full path name for display (e.g., "engineering/web/frontend")
    pub full_path: String,

    /// Parent domain (None for root domains)
    pub parent: Option<DomainId>,

    /// Description of what this domain covers
    pub description: String,

    /// Criticality level for patterns in this domain
    pub criticality: DomainCriticality,

    /// Tags for cross-cutting concerns
    pub tags: Vec<String>,

    /// Depth in the hierarchy (0 for root)
    pub depth: u32,

    /// When this domain was created
    pub created_at: u64,
}

impl Domain {
    /// Check if this domain has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// Registry for managing domains with hierarchical relationships
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DomainRegistry {
    /// All registered domains
    domains: HashMap<DomainId, Domain>,

    /// Lookup by name (lowercase)
    #[cfg_attr(feature = "serde", serde(skip))]
    by_name: HashMap<String, DomainId>,

    /// Lookup by full path (lowercase)
    #[cfg_attr(feature = "serde", serde(skip))]
    by_path: HashMap<String, DomainId>,

    /// Children of each domain
    #[cfg_attr(feature = "serde", serde(skip))]
    children: HashMap<DomainId, Vec<DomainId>>,

    /// Next available ID
    next_id: DomainId,

    /// Root domain ID (auto-created)
    pub root_id: DomainId,
}

impl Default for DomainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainRegistry {
    /// Create a new domain registry with a root domain
    pub fn new() -> Self {
        let mut registry = Self {
            domains: HashMap::new(),
            by_name: HashMap::new(),
            by_path: HashMap::new(),
            children: HashMap::new(),
            next_id: 1,
            root_id: 0,
        };

        // Create root domain
        let root = Domain {
            id: 0,
            name: "root".to_string(),
            full_path: "".to_string(),
            parent: None,
            description: "Root domain containing all knowledge".to_string(),
            criticality: DomainCriticality::Normal,
            tags: vec![],
            depth: 0,
            created_at: 0,
        };
        registry.domains.insert(0, root);
        registry.by_name.insert("root".to_string(), 0);
        registry.children.insert(0, Vec::new());

        registry
    }

    /// Register a new domain under a parent
    pub fn register(
        &mut self,
        name: &str,
        parent: Option<DomainId>,
        description: &str,
        timestamp: u64,
    ) -> DomainId {
        let parent_id = parent.unwrap_or(self.root_id);

        // Calculate full path
        let full_path = if parent_id == self.root_id {
            name.to_string()
        } else if let Some(parent_domain) = self.domains.get(&parent_id) {
            format!("{}/{}", parent_domain.full_path, name)
        } else {
            name.to_string()
        };

        // Check if already exists
        if let Some(&existing_id) = self.by_path.get(&full_path.to_lowercase()) {
            return existing_id;
        }

        // Calculate depth
        let depth = if parent_id == self.root_id {
            1
        } else {
            self.domains
                .get(&parent_id)
                .map(|p| p.depth + 1)
                .unwrap_or(1)
        };

        let id = self.next_id;
        self.next_id += 1;

        let domain = Domain {
            id,
            name: name.to_string(),
            full_path: full_path.clone(),
            parent: Some(parent_id),
            description: description.to_string(),
            criticality: DomainCriticality::Normal,
            tags: vec![],
            depth,
            created_at: timestamp,
        };

        self.domains.insert(id, domain);
        self.by_name.insert(name.to_lowercase(), id);
        self.by_path.insert(full_path.to_lowercase(), id);
        self.children.entry(parent_id).or_default().push(id);
        self.children.insert(id, Vec::new());

        id
    }

    /// Register a domain with full configuration
    pub fn register_with_config(
        &mut self,
        name: &str,
        parent: Option<DomainId>,
        description: &str,
        criticality: DomainCriticality,
        tags: Vec<String>,
        timestamp: u64,
    ) -> DomainId {
        let id = self.register(name, parent, description, timestamp);

        // Update with config
        if let Some(domain) = self.domains.get_mut(&id) {
            domain.criticality = criticality;
            domain.tags = tags;
        }

        id
    }

    /// Register a hierarchical path like "engineering/web/frontend"
    pub fn register_path(&mut self, path: &str, timestamp: u64) -> DomainId {
        let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

        let mut parent_id = self.root_id;
        let mut last_id = self.root_id;

        for segment in segments {
            last_id = self.register(segment, Some(parent_id), "", timestamp);
            parent_id = last_id;
        }

        last_id
    }

    /// Get a domain by ID
    pub fn get(&self, id: DomainId) -> Option<&Domain> {
        self.domains.get(&id)
    }

    /// Get a mutable domain by ID
    pub fn get_mut(&mut self, id: DomainId) -> Option<&mut Domain> {
        self.domains.get_mut(&id)
    }

    /// Find a domain by name (case-insensitive)
    pub fn find_by_name(&self, name: &str) -> Option<DomainId> {
        self.by_name.get(&name.to_lowercase()).copied()
    }

    /// Find a domain by full path (case-insensitive)
    pub fn find_by_path(&self, path: &str) -> Option<DomainId> {
        self.by_path.get(&path.to_lowercase()).copied()
    }

    /// Get all ancestors of a domain (from immediate parent to root)
    pub fn ancestors(&self, id: DomainId) -> Vec<DomainId> {
        let mut result = Vec::new();
        let mut current = id;

        while let Some(domain) = self.domains.get(&current) {
            if let Some(parent_id) = domain.parent {
                result.push(parent_id);
                current = parent_id;
            } else {
                break;
            }
        }

        result
    }

    /// Get all descendants of a domain (breadth-first)
    pub fn descendants(&self, id: DomainId) -> Vec<DomainId> {
        let mut result = Vec::new();
        let mut queue = vec![id];

        while let Some(current) = queue.pop() {
            if let Some(children) = self.children.get(&current) {
                for &child in children {
                    result.push(child);
                    queue.push(child);
                }
            }
        }

        result
    }

    /// Get direct children of a domain
    pub fn children_of(&self, id: DomainId) -> Vec<DomainId> {
        self.children.get(&id).cloned().unwrap_or_default()
    }

    /// Check if one domain is an ancestor of another
    pub fn is_ancestor_of(&self, potential_ancestor: DomainId, descendant: DomainId) -> bool {
        self.ancestors(descendant).contains(&potential_ancestor)
    }

    /// Check if two domains are related (one is ancestor/descendant of other)
    pub fn are_related(&self, a: DomainId, b: DomainId) -> bool {
        a == b || self.is_ancestor_of(a, b) || self.is_ancestor_of(b, a)
    }

    /// Find the common ancestor of two domains
    pub fn common_ancestor(&self, a: DomainId, b: DomainId) -> Option<DomainId> {
        if a == b {
            return Some(a);
        }

        let ancestors_a: std::collections::HashSet<_> =
            std::iter::once(a).chain(self.ancestors(a)).collect();

        // Check b and its ancestors against a's ancestors
        if ancestors_a.contains(&b) {
            return Some(b);
        }

        for ancestor in self.ancestors(b) {
            if ancestors_a.contains(&ancestor) {
                return Some(ancestor);
            }
        }

        // Root is always common ancestor
        Some(self.root_id)
    }

    /// Calculate the distance (hops) between two domains
    ///
    /// Returns None if domains are unrelated (shouldn't happen in a proper tree)
    pub fn distance(&self, a: DomainId, b: DomainId) -> Option<u32> {
        if a == b {
            return Some(0);
        }

        let common = self.common_ancestor(a, b)?;

        let depth_a = self.domains.get(&a)?.depth;
        let depth_b = self.domains.get(&b)?.depth;
        let depth_common = self.domains.get(&common)?.depth;

        Some((depth_a - depth_common) + (depth_b - depth_common))
    }

    /// Calculate similarity between domains (0.0-1.0)
    ///
    /// Based on distance and shared ancestry
    pub fn similarity(&self, a: DomainId, b: DomainId) -> f32 {
        if a == b {
            return 1.0;
        }

        let distance = match self.distance(a, b) {
            Some(d) => d,
            None => return 0.0,
        };

        // Exponential decay based on distance
        // Distance 1 = 0.8, Distance 2 = 0.64, etc.
        0.8_f32.powi(distance as i32)
    }

    /// Find all domains matching a tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<DomainId> {
        self.domains
            .values()
            .filter(|d| d.has_tag(tag))
            .map(|d| d.id)
            .collect()
    }

    /// Get all domains at a specific depth
    pub fn at_depth(&self, depth: u32) -> Vec<DomainId> {
        self.domains
            .values()
            .filter(|d| d.depth == depth)
            .map(|d| d.id)
            .collect()
    }

    /// Get the number of registered domains (excluding root)
    pub fn len(&self) -> usize {
        self.domains.len().saturating_sub(1) // Exclude root
    }

    /// Check if registry is empty (only has root)
    pub fn is_empty(&self) -> bool {
        self.domains.len() <= 1
    }

    /// Rebuild internal indices (call after deserialization)
    pub fn rebuild_indices(&mut self) {
        self.by_name.clear();
        self.by_path.clear();
        self.children.clear();

        for domain in self.domains.values() {
            self.by_name.insert(domain.name.to_lowercase(), domain.id);
            self.by_path
                .insert(domain.full_path.to_lowercase(), domain.id);

            if let Some(parent_id) = domain.parent {
                self.children.entry(parent_id).or_default().push(domain.id);
            }
            self.children.entry(domain.id).or_default();
        }
    }
}

/// Helper for building domain paths fluently
#[derive(Debug, Clone)]
pub struct DomainPath {
    segments: Vec<String>,
}

impl DomainPath {
    /// Create a new domain path from a string like "engineering/web/frontend"
    pub fn new(path: &str) -> Self {
        let segments = path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        Self { segments }
    }

    /// Create from individual segments
    pub fn from_segments(segments: Vec<String>) -> Self {
        Self { segments }
    }

    /// Add a segment to the path
    pub fn push(mut self, segment: &str) -> Self {
        self.segments.push(segment.to_string());
        self
    }

    /// Get the path as a string
    pub fn as_str(&self) -> String {
        self.segments.join("/")
    }

    /// Get the segments
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Register all segments in a registry and return the final domain ID
    pub fn register_all(&self, registry: &mut DomainRegistry, timestamp: u64) -> DomainId {
        registry.register_path(&self.as_str(), timestamp)
    }

    /// Check if this path is a prefix of another
    pub fn is_prefix_of(&self, other: &DomainPath) -> bool {
        if self.segments.len() > other.segments.len() {
            return false;
        }
        self.segments
            .iter()
            .zip(other.segments.iter())
            .all(|(a, b)| a == b)
    }
}

impl From<&str> for DomainPath {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl std::fmt::Display for DomainPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
