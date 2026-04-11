// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cluster manifest types for the Mycelix modular architecture.
//!
//! Each cluster declares its identity, capabilities, dependencies,
//! bridge connections, and entry types. The portal uses these manifests
//! to build the cluster catalog, data sovereignty dashboard, and
//! dependency graph.
//!
//! Manifests are Rust structs (not YAML/TOML files) to prevent drift
//! between the manifest and the actual code.

use serde::{Deserialize, Serialize};

/// Direction of a bridge connection between clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeDirection {
    /// This cluster calls into the target cluster.
    Outbound,
    /// The target cluster calls into this cluster.
    Inbound,
    /// Both clusters call each other.
    Bidirectional,
}

/// Sensitivity classification for data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataSensitivity {
    /// Publicly visible on the DHT.
    Public,
    /// Internal to the cluster.
    Internal,
    /// Requires consent to share.
    Sensitive,
    /// Highest protection level.
    Restricted,
}

impl DataSensitivity {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Public => "Public",
            Self::Internal => "Internal",
            Self::Sensitive => "Sensitive",
            Self::Restricted => "Restricted",
        }
    }
}

/// A dependency on another cluster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterDependency {
    /// Domain ID of the dependency (e.g., "identity").
    pub cluster_id: String,
    /// Whether this is a hard requirement or optional enhancement.
    pub required: bool,
    /// Human-readable reason (e.g., "DID resolution for patient identity").
    pub reason: String,
}

/// A bridge connection declaration between clusters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeDeclaration {
    /// Target cluster ID.
    pub target_cluster: String,
    /// Direction of the bridge.
    pub direction: BridgeDirection,
    /// Zome names allowed across this bridge.
    pub allowed_zomes: Vec<String>,
    /// Human-readable purpose.
    pub purpose: String,
}

/// An entry type declaration for the data sovereignty dashboard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryTypeDeclaration {
    /// Zome this entry type belongs to.
    pub zome: String,
    /// Entry type identifier.
    pub entry_type: String,
    /// Human-readable label.
    pub label: String,
    /// Data sensitivity classification.
    pub sensitivity: DataSensitivity,
    /// Short description.
    pub description: String,
}

/// Consciousness tier required for a cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CivicTier {
    Observer = 0,
    Participant = 1,
    Citizen = 2,
    Steward = 3,
    Guardian = 4,
}

impl CivicTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Observer => "Observer",
            Self::Participant => "Participant",
            Self::Citizen => "Citizen",
            Self::Steward => "Steward",
            Self::Guardian => "Guardian",
        }
    }
}

/// Complete manifest for a Mycelix cluster.
///
/// Describes everything the portal needs to display the cluster
/// in the catalog, build the dependency graph, and populate the
/// data sovereignty dashboard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterManifest {
    /// Unique cluster identifier (e.g., "health", "governance").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Biological metaphor name (e.g., "Homeostasis").
    pub bio_name: String,
    /// Version string.
    pub version: String,
    /// Short description.
    pub description: String,
    /// Minimum consciousness tier to access.
    pub min_tier: CivicTier,
    /// Holochain hApp role name.
    pub happ_role: String,
    /// Zome names in this cluster.
    pub zomes: Vec<String>,
    /// Dependencies on other clusters.
    pub dependencies: Vec<ClusterDependency>,
    /// Bridge connections to other clusters.
    pub bridges: Vec<BridgeDeclaration>,
    /// Entry types managed by this cluster.
    pub entry_types: Vec<EntryTypeDeclaration>,
    /// Primary CSS color.
    pub color_primary: String,
    /// Glow/accent CSS color.
    pub color_glow: String,
    /// Optional external frontend URL (for community extensions).
    pub frontend_url: Option<String>,
}

/// External frontend manifest for community extensions.
///
/// Loaded from `~/.mycelix/extensions.json` or a registry DNA.
/// Describes a frontend that runs as a separate web app and
/// connects to the same Holochain conductor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExternalFrontendManifest {
    /// Unique extension identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string.
    pub version: String,
    /// Author or organization.
    pub author: String,
    /// Biological metaphor name.
    pub bio_name: String,
    /// Primary CSS color.
    pub color_primary: String,
    /// Glow/accent CSS color.
    pub color_glow: String,
    /// Minimum consciousness tier.
    pub min_tier: CivicTier,
    /// URL where the frontend is hosted.
    pub frontend_url: String,
    /// Clusters this extension requires.
    pub required_clusters: Vec<String>,
    /// Clusters this extension can optionally use.
    pub optional_clusters: Vec<String>,
    /// Short description.
    pub description: String,
}

/// A catalog of all known clusters and external extensions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ClusterCatalog {
    /// Built-in cluster manifests (compiled from feature flags).
    pub clusters: Vec<ClusterManifest>,
    /// External community extensions (loaded at runtime).
    pub extensions: Vec<ExternalFrontendManifest>,
}

impl ClusterCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Find a cluster by ID.
    pub fn get_cluster(&self, id: &str) -> Option<&ClusterManifest> {
        self.clusters.iter().find(|c| c.id == id)
    }

    /// Find an extension by ID.
    pub fn get_extension(&self, id: &str) -> Option<&ExternalFrontendManifest> {
        self.extensions.iter().find(|e| e.id == id)
    }

    /// Get all cluster IDs.
    pub fn cluster_ids(&self) -> Vec<&str> {
        self.clusters.iter().map(|c| c.id.as_str()).collect()
    }

    /// Compute data flows between installed clusters.
    /// Returns (source_id, target_id, zome_count, purpose).
    pub fn data_flows(&self, installed_ids: &[&str]) -> Vec<DataFlow> {
        let mut flows = Vec::new();
        for cluster in &self.clusters {
            if !installed_ids.contains(&cluster.id.as_str()) {
                continue;
            }
            for bridge in &cluster.bridges {
                if !installed_ids.contains(&bridge.target_cluster.as_str()) {
                    continue;
                }
                flows.push(DataFlow {
                    source_id: cluster.id.clone(),
                    target_id: bridge.target_cluster.clone(),
                    direction: bridge.direction,
                    zome_count: bridge.allowed_zomes.len(),
                    purpose: bridge.purpose.clone(),
                });
            }
        }
        flows
    }
}

/// A data flow between two clusters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataFlow {
    pub source_id: String,
    pub target_id: String,
    pub direction: BridgeDirection,
    pub zome_count: usize,
    pub purpose: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cluster(id: &str, deps: Vec<ClusterDependency>, bridges: Vec<BridgeDeclaration>) -> ClusterManifest {
        ClusterManifest {
            id: id.to_string(),
            name: id.to_string(),
            bio_name: "Test".to_string(),
            version: "0.1.0".to_string(),
            description: "test cluster".to_string(),
            min_tier: CivicTier::Participant,
            happ_role: id.to_string(),
            zomes: vec!["test_zome".to_string()],
            dependencies: deps,
            bridges,
            entry_types: vec![],
            color_primary: "#000".to_string(),
            color_glow: "#111".to_string(),
            frontend_url: None,
        }
    }

    #[test]
    fn consciousness_tier_ordering() {
        assert!(CivicTier::Observer < CivicTier::Participant);
        assert!(CivicTier::Participant < CivicTier::Citizen);
        assert!(CivicTier::Citizen < CivicTier::Steward);
        assert!(CivicTier::Steward < CivicTier::Guardian);
    }

    #[test]
    fn catalog_find_cluster() {
        let mut catalog = ClusterCatalog::new();
        catalog.clusters.push(sample_cluster("health", vec![], vec![]));
        catalog.clusters.push(sample_cluster("governance", vec![], vec![]));

        assert!(catalog.get_cluster("health").is_some());
        assert!(catalog.get_cluster("governance").is_some());
        assert!(catalog.get_cluster("unknown").is_none());
    }

    #[test]
    fn catalog_cluster_ids() {
        let mut catalog = ClusterCatalog::new();
        catalog.clusters.push(sample_cluster("a", vec![], vec![]));
        catalog.clusters.push(sample_cluster("b", vec![], vec![]));

        let ids = catalog.cluster_ids();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn data_flows_between_installed() {
        let mut catalog = ClusterCatalog::new();
        catalog.clusters.push(sample_cluster("health", vec![], vec![
            BridgeDeclaration {
                target_cluster: "identity".to_string(),
                direction: BridgeDirection::Outbound,
                allowed_zomes: vec!["did_registry".to_string()],
                purpose: "DID resolution".to_string(),
            },
        ]));
        catalog.clusters.push(sample_cluster("identity", vec![], vec![]));

        let flows = catalog.data_flows(&["health", "identity"]);
        assert_eq!(flows.len(), 1);
        assert_eq!(flows[0].source_id, "health");
        assert_eq!(flows[0].target_id, "identity");
        assert_eq!(flows[0].zome_count, 1);
    }

    #[test]
    fn data_flows_skip_uninstalled() {
        let mut catalog = ClusterCatalog::new();
        catalog.clusters.push(sample_cluster("health", vec![], vec![
            BridgeDeclaration {
                target_cluster: "identity".to_string(),
                direction: BridgeDirection::Outbound,
                allowed_zomes: vec!["did_registry".to_string()],
                purpose: "DID resolution".to_string(),
            },
        ]));

        // identity not installed
        let flows = catalog.data_flows(&["health"]);
        assert_eq!(flows.len(), 0);
    }

    #[test]
    fn external_frontend_serde_roundtrip() {
        let ext = ExternalFrontendManifest {
            id: "garden".to_string(),
            name: "Garden Tracker".to_string(),
            version: "1.0.0".to_string(),
            author: "community".to_string(),
            bio_name: "Cultivation".to_string(),
            color_primary: "#22C55E".to_string(),
            color_glow: "#86EFAC".to_string(),
            min_tier: CivicTier::Participant,
            frontend_url: "https://garden.example.com".to_string(),
            required_clusters: vec!["commons".to_string()],
            optional_clusters: vec![],
            description: "Garden tracking".to_string(),
        };

        let json = serde_json::to_string(&ext).unwrap();
        let back: ExternalFrontendManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "garden");
        assert_eq!(back.min_tier, CivicTier::Participant);
    }

    #[test]
    fn sensitivity_labels() {
        assert_eq!(DataSensitivity::Public.label(), "Public");
        assert_eq!(DataSensitivity::Restricted.label(), "Restricted");
    }
}
