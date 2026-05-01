// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Conductor data bridge for Sol Atlas.
//!
//! Provides a trait `ClusterDataSource` that can be implemented
//! for mock (stub) data or live Holochain conductor data.
//!
//! Phase A (current): `MockSource` returns the existing stub layers.
//! Phase B (future): `ConductorSource` connects via WebSocket to
//! ws://localhost:8888 and fetches real project/object data.

use crate::cluster_layers::{GeoFeature, VisualizationLayer};
use std::collections::HashMap;

/// Trait for fetching geo-features from a cluster.
pub trait ClusterDataSource: Send + Sync {
    /// Fetch features for a given cluster name.
    fn fetch_features(&self, cluster: &str) -> Vec<GeoFeature>;

    /// Fetch a complete visualization layer for a cluster.
    fn fetch_layer(&self, cluster: &str) -> Option<VisualizationLayer> {
        let features = self.fetch_features(cluster);
        if features.is_empty() {
            return None;
        }
        Some(VisualizationLayer {
            cluster: cluster.to_string(),
            display_name: cluster.to_string(),
            toggle_key: cluster.chars().next().unwrap_or('x').to_string(),
            features,
            enabled: true,
        })
    }
}

/// Mock data source — returns the existing stub layers from cluster_layers.rs.
pub struct MockSource;

impl ClusterDataSource for MockSource {
    fn fetch_features(&self, cluster: &str) -> Vec<GeoFeature> {
        match cluster {
            "energy" => crate::cluster_layers::energy_layer().features,
            "climate" => crate::cluster_layers::climate_layer().features,
            "commons" => crate::cluster_layers::commons_layer().features,
            "supplychain" => crate::cluster_layers::supply_chain_layer().features,
            "orbital" => crate::cluster_layers::orbital_layer().features,
            _ => vec![],
        }
    }
}

/// Conductor data source — fetches live data from Holochain.
///
/// Phase B implementation. Currently a stub that delegates to MockSource
/// until the conductor WebSocket client is wired.
pub struct ConductorSource {
    /// Conductor app WebSocket URL (e.g., "ws://localhost:8888").
    pub url: String,
    /// Fallback to mock data when conductor is unreachable.
    pub fallback: MockSource,
}

impl ConductorSource {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            fallback: MockSource,
        }
    }
}

impl ClusterDataSource for ConductorSource {
    fn fetch_features(&self, cluster: &str) -> Vec<GeoFeature> {
        // Phase B TODO:
        // 1. Connect to self.url via tokio-tungstenite
        // 2. Send AppRequest::CallZome for the cluster's project listing
        // 3. Map Record entries to GeoFeature (lat/lon from project location)
        // 4. On error, fall back to mock data

        // For now, delegate to mock
        self.fallback.fetch_features(cluster)
    }
}

/// Convert mycelix-energy project data to GeoFeatures.
///
/// Utility for Phase B: maps EnergyProjectView lat/lon to globe pins.
pub fn energy_project_to_feature(
    name: &str,
    project_type: &str,
    lat: f64,
    lon: f64,
    capacity_mw: f64,
) -> GeoFeature {
    let color = match project_type {
        "Solar" => [1.0, 0.9, 0.2, 1.0],          // Yellow
        "Wind" => [0.4, 0.7, 1.0, 1.0],           // Blue
        "Hydro" => [0.2, 0.8, 0.9, 1.0],          // Cyan
        "Nuclear" => [0.8, 0.4, 1.0, 1.0],        // Purple
        "BatteryStorage" => [0.3, 0.9, 0.5, 1.0], // Green
        _ => [0.7, 0.7, 0.7, 1.0],                // Gray
    };

    let size = (capacity_mw / 100.0).clamp(0.5, 5.0) as f32;

    let mut properties = HashMap::new();
    properties.insert("type".into(), project_type.into());
    properties.insert("capacity_mw".into(), format!("{capacity_mw:.0}"));

    GeoFeature {
        lat,
        lon,
        feature_type: "energy_project".into(),
        name: name.into(),
        properties,
        color,
        size,
    }
}

/// Convert mycelix-climate project data to GeoFeatures.
pub fn climate_project_to_feature(
    name: &str,
    project_type: &str,
    lat: f64,
    lon: f64,
    expected_credits: f64,
) -> GeoFeature {
    let color = match project_type {
        "Reforestation" => [0.2, 0.8, 0.3, 1.0],
        "RenewableEnergy" => [1.0, 0.9, 0.2, 1.0],
        "OceanRestoration" => [0.2, 0.5, 0.9, 1.0],
        "MethaneCapture" => [0.9, 0.5, 0.2, 1.0],
        "DirectAirCapture" => [0.6, 0.6, 0.9, 1.0],
        _ => [0.7, 0.7, 0.7, 1.0],
    };

    let size = (expected_credits / 5000.0).clamp(0.5, 4.0) as f32;

    let mut properties = HashMap::new();
    properties.insert("type".into(), project_type.into());
    properties.insert("expected_credits".into(), format!("{expected_credits:.0}"));

    GeoFeature {
        lat,
        lon,
        feature_type: "climate_project".into(),
        name: name.into(),
        properties,
        color,
        size,
    }
}
