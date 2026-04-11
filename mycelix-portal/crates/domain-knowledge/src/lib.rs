// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge domain module — epistemic commons, claims, fact-checking, inference.

pub mod types;

use portal_domain_trait::{
    ClusterDependency, ColorFamily, CivicTier, DataSensitivity, DomainModule, EntryTypeInfo, NavItem,
};

pub struct KnowledgeDomain;

impl DomainModule for KnowledgeDomain {
    fn id(&self) -> &'static str { "knowledge" }
    fn name(&self) -> &'static str { "Knowledge" }
    fn bio_name(&self) -> &'static str { "Noosphere" }
    fn description(&self) -> &'static str {
        "Epistemic commons: distributed knowledge claims, inference graphs, decentralized fact-checking, and prediction markets for collective intelligence."
    }

    fn color_family(&self) -> ColorFamily { ColorFamily { primary: "#0891B2", glow: "#22D3EE" } }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem { label: "Claims", bio_label: "Spores", path: "/knowledge/claims" },
            NavItem { label: "Graph", bio_label: "Mycelial Web", path: "/knowledge/graph" },
            NavItem { label: "Fact Check", bio_label: "Immune Response", path: "/knowledge/factcheck" },
            NavItem { label: "Markets", bio_label: "Prediction Soil", path: "/knowledge/markets" },
        ]
    }

    fn min_tier(&self) -> CivicTier { CivicTier::Steward }
    fn key_context(&self) -> &'static [u8] { b"mycelix-knowledge-v1" }
    fn happ_role(&self) -> &'static str { "knowledge" }

    fn zomes(&self) -> &'static [&'static str] {
        &["claims", "graph", "query", "inference", "factcheck", "markets", "dkg", "bridge"]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency { cluster_id: "identity", reason: "Claim author DID attestation", required: true },
            ClusterDependency { cluster_id: "governance", reason: "Epistemic standards proposals", required: false },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo { label: "Knowledge Claim", zome: "claims", sensitivity: DataSensitivity::Public },
            EntryTypeInfo { label: "Inference Rule", zome: "inference", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Fact Check", zome: "factcheck", sensitivity: DataSensitivity::Community },
            EntryTypeInfo { label: "Prediction Market", zome: "markets", sensitivity: DataSensitivity::Community },
        ]
    }
}
