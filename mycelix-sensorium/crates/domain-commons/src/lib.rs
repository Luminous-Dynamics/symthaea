// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commons domain module — shared resources for community wellbeing.
//!
//! Covers 39 zomes across property, housing, care, mutual aid, water,
//! food, transport, mesh-time, and resource-mesh.

pub mod types;

use sensorium_domain_trait::{
    AttentionLevel, CivicTier, ClusterDependency, ColorFamily, DataSensitivity,
    DomainAttentionItem, DomainAvailability, DomainLaunchTarget, DomainMetric, DomainModule,
    DomainSummaryCard, EntryTypeInfo, LaunchKind, NavItem,
};

pub struct CommonsDomain;

impl DomainModule for CommonsDomain {
    fn id(&self) -> &'static str {
        "commons"
    }
    fn name(&self) -> &'static str {
        "Commons"
    }
    fn bio_name(&self) -> &'static str {
        "Mutualism"
    }
    fn description(&self) -> &'static str {
        "Shared infrastructure for community wellbeing: property registries, housing, mutual aid, water, food, transport, and resource meshes."
    }

    fn color_family(&self) -> ColorFamily {
        ColorFamily {
            primary: "#059669",
            glow: "#34D399",
        }
    }

    fn nav_items(&self) -> Vec<NavItem> {
        vec![
            NavItem {
                label: "Property",
                bio_label: "Territory",
                path: "/property",
            },
            NavItem {
                label: "Housing",
                bio_label: "Shelter",
                path: "/housing",
            },
            NavItem {
                label: "Care",
                bio_label: "Tending",
                path: "/care",
            },
            NavItem {
                label: "Mutual Aid",
                bio_label: "Symbiosis",
                path: "/resources",
            },
            NavItem {
                label: "Water",
                bio_label: "Flow",
                path: "/resources",
            },
            NavItem {
                label: "Food",
                bio_label: "Nourishment",
                path: "/food",
            },
            NavItem {
                label: "Transport",
                bio_label: "Mycelial Paths",
                path: "/transport",
            },
        ]
    }

    fn min_tier(&self) -> CivicTier {
        CivicTier::Citizen
    }
    fn key_context(&self) -> &'static [u8] {
        b"mycelix-commons-v1"
    }
    fn happ_role(&self) -> &'static str {
        "commons"
    }

    fn zomes(&self) -> &'static [&'static str] {
        &[
            "property_registry",
            "housing_units",
            "care_plans",
            "mutualaid_requests",
            "water_flow",
            "food_distribution",
            "transport_routes",
            "mesh_time",
            "resource_mesh",
        ]
    }

    fn dependencies(&self) -> &'static [ClusterDependency] {
        &[
            ClusterDependency {
                cluster_id: "identity",
                reason: "Stewardship rights linked to DID",
                required: true,
            },
            ClusterDependency {
                cluster_id: "governance",
                reason: "Resource allocation proposals",
                required: false,
            },
        ]
    }

    fn entry_types(&self) -> &'static [EntryTypeInfo] {
        &[
            EntryTypeInfo {
                label: "Property Claim",
                zome: "property_registry",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Housing Unit",
                zome: "housing_units",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Mutual Aid Request",
                zome: "mutualaid_requests",
                sensitivity: DataSensitivity::Protected,
            },
            EntryTypeInfo {
                label: "Resource Entry",
                zome: "resource_mesh",
                sensitivity: DataSensitivity::Community,
            },
            EntryTypeInfo {
                label: "Care Plan",
                zome: "care_plans",
                sensitivity: DataSensitivity::Protected,
            },
        ]
    }

    fn primary_launch(&self) -> Option<DomainLaunchTarget> {
        Some(DomainLaunchTarget {
            id: "resources",
            label: "Open Resources",
            path: "/resources",
            kind: LaunchKind::InternalRoute,
            requires_unlock: false,
            recommended: true,
        })
    }

    fn secondary_launches(&self) -> Vec<DomainLaunchTarget> {
        vec![
            DomainLaunchTarget {
                id: "care",
                label: "Care",
                path: "/care",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "housing",
                label: "Housing",
                path: "/housing",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
            DomainLaunchTarget {
                id: "transport",
                label: "Transport",
                path: "/transport",
                kind: LaunchKind::InternalRoute,
                requires_unlock: false,
                recommended: false,
            },
        ]
    }

    fn summary_card(&self) -> Option<DomainSummaryCard> {
        Some(DomainSummaryCard {
            domain_id: "commons",
            title: "Commons Stewardship".into(),
            availability: DomainAvailability::Mock,
            status_line:
                "Commons can summarize stewardship pressure and resource coordination before opening deeper operational views."
                    .into(),
            metrics: vec![
                DomainMetric {
                    id: "resources",
                    label: "Resources".into(),
                    value: "18".into(),
                    hint: Some("tracked entries".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "care",
                    label: "Care Plans".into(),
                    value: "6".into(),
                    hint: Some("active".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "housing",
                    label: "Housing Units".into(),
                    value: "12".into(),
                    hint: Some("community inventory".into()),
                    tone: None,
                },
                DomainMetric {
                    id: "aid",
                    label: "Aid Requests".into(),
                    value: "3".into(),
                    hint: Some("1 urgent".into()),
                    tone: Some("notice"),
                },
            ],
            attention: vec![DomainAttentionItem {
                id: "commons-aid".into(),
                label: "Mutual aid needs review".into(),
                detail: "Current mutual-aid demand suggests one or more requests need stewardship attention."
                    .into(),
                level: AttentionLevel::ActionNeeded,
                path: Some("/resources".into()),
            }],
            primary_launch: self.primary_launch(),
            secondary_launches: self.secondary_launches(),
            updated_at: Some(1_776_700_800_000_000),
        })
    }
}
