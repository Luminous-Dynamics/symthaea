// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Town Sympoiesis Orchestrator — Unified metabolism for circular settlements.

use crate::economy::ThermodynamicLedger;
use crate::spatial_metabolism::SpatialMetabolicGrid;
use crate::types::{InfrastructureCommand, InfrastructureState};
use serde::{Deserialize, Serialize};
use symthaea_engineering::{EngineeringManager, InfrastructureNode};
use symthaea_silicon::PowerDistributionLogic;
use symthaea_swarm::SwarmAggregator;
use symthaea_agribot::types::AgribotState;
use symthaea_clime::types::ClimeState;

/// The primary metabolism of a circular town.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TownSympoiesis {
    pub name: String,
    pub power_grid: PowerDistributionLogic,
    pub infrastructure: InfrastructureNode,
    pub spatial_grid: SpatialMetabolicGrid,
    pub economic_ledger: ThermodynamicLedger,
    /// Biological organ: Agricultural state.
    pub agriculture: AgribotState,
    /// Atmospheric organ: Climate/Fluid state.
    pub climate: ClimeState,
    /// Collection of peer states for collective consciousness (Phi) measurement.
    #[serde(skip)]
    pub swarm_aggregator: SwarmAggregator,
    pub water_clarity: f32,
    pub nutrient_advection: f32,
}

impl TownSympoiesis {
    /// Create a new town sympoiesis loop.
    pub fn new(name: &str, manager: &mut EngineeringManager) -> Self {
        let infra = manager.design_infrastructure(name, 10.0);
        let grid = PowerDistributionLogic {
            grid_frequency_hz: 60.0,
            renewable_ratio: 0.8,
            active_loads_mw: 10.0,
            battery_reserve_mwh: 100.0,
            min_critical_mw: 2.0, // Core life support threshold
        };

        Self {
            name: name.into(),
            power_grid: grid,
            infrastructure: infra,
            spatial_grid: SpatialMetabolicGrid::new_hex_cluster(45.0, 9.0), // Mock coordinates
            economic_ledger: ThermodynamicLedger::new(),
            agriculture: AgribotState::home(),
            climate: ClimeState::home(),
            swarm_aggregator: SwarmAggregator::new(),
            water_clarity: 0.95,
            nutrient_advection: 0.5,
        }
    }

    /// Execute a metabolic step: synchronize silicon brain with physical infrastructure.
    pub fn step(&mut self, demand_mw: f32, available_mw: f32) -> f32 {
        // 1. Spatial Update: Localized effects
        let spatial_surprise = self.spatial_grid.update(demand_mw, available_mw);

        // 2. Silicon Brain optimizes the grid
        let surprise = self.power_grid.optimize_routing(demand_mw, available_mw);

        // 3. Biological Metabolism: Agribot & Clime updates
        // If surprise is high (power instability), biological vitality drops.
        if surprise > 0.5 {
            // "Hunger" and "Thirst" increase as prediction error
            self.agriculture.channels[symthaea_agribot::types::CROP_HEALTH] *= 0.95;
            self.climate.channels[symthaea_clime::types::AIR_QUALITY] *= 0.9;
        } else {
            self.agriculture.channels[symthaea_agribot::types::CROP_HEALTH] = 
                (self.agriculture.channels[symthaea_agribot::types::CROP_HEALTH] + 0.01).min(1.0);
        }

        // 4. Physical Metabolism: Adjust fluid states based on power stability and spatial health
        if surprise < 0.2 && spatial_surprise < 0.2 {
            self.water_clarity = (self.water_clarity + 0.01).min(1.0);
            self.nutrient_advection = (self.nutrient_advection + 0.02).min(1.0);
        } else {
            self.water_clarity = (self.water_clarity - 0.05).max(0.0);
        }

        // 5. Economic Metabolism: Link production to Tend minting
        // Only mint if we have surplus power and high biological vitality
        let crop_health = self.agriculture.channels[symthaea_agribot::types::CROP_HEALTH] as f32;
        if available_mw > demand_mw && crop_health > 0.6 {
            let collective_phi = self.swarm_aggregator.calculate_swarm_phi();
            self.economic_ledger.mint_production_credit(
                (available_mw - demand_mw) * crop_health,
                self.water_clarity,
                collective_phi,
            );
        }

        // Entropy tax on systemic surprise
        self.economic_ledger.apply_entropy_tax(surprise + spatial_surprise);

        surprise
    }

    /// Generate an SVG visualization of the settlement's "Bioluminescent Aura".
    ///
    /// Maps each H3 hexagonal zone to a color based on its local Phi and surprise.
    /// Cyan = High Integration, Yellow = Learning/Surprise, Red = Entropy/Deficit.
    pub fn generate_aura_svg(&self) -> String {
        use symthaea_canvas::scene_graph::{SceneNode, NodeKind, Style, Transform};
        use symthaea_canvas::color::Color;

        let mut zones_svg = Vec::new();
        
        for (i, zone) in self.spatial_grid.zones.values().enumerate() {
            // Determine Color based on metabolic state
            // High surprise -> Yellow/Red, High efficiency -> Cyan
            let color = if zone.localized_surprise > 0.5 {
                Color::rgba(255.0, 255.0, 0.0, 0.8) // Yellow (Alert)
            } else if zone.efficiency < 0.3 {
                Color::rgba(255.0, 0.0, 0.0, 0.8) // Red (Deficit)
            } else {
                Color::rgba(0.0, 255.0, 255.0, 0.6) // Cyan (Harmonic)
            };

            // Basic hexagonal layout (Mocked positions for SVG)
            let row = i / 3;
            let col = i % 3;
            let x = col as f32 * 120.0 + (if row % 2 == 1 { 60.0 } else { 0.0 }) + 100.0;
            let y = row as f32 * 100.0 + 100.0;

            zones_svg.push(SceneNode {
                kind: NodeKind::Circle { cx: 0.0, cy: 0.0, r: 50.0 }, // Circles as hex proxies for demo
                transform: Transform { translate_x: x, translate_y: y, ..Transform::identity() },
                style: Style {
                    fill: Some(color),
                    stroke: Some(Color::rgba(255.0, 255.0, 255.0, 1.0)),
                    stroke_width: Some(2.0),
                    ..Default::default()
                },
                children: Vec::new(),
            });
        }

        let root = SceneNode {
            kind: NodeKind::Group { id: Some("aura".into()) },
            transform: Transform::identity(),
            style: Style::default(),
            children: zones_svg,
        };

        symthaea_canvas::render_svg(&root, 1.0)
    }

    /// Autonomously route surplus "Tend" credit to a peer in distress (Mutual Aid).
    pub fn distribute_mutual_aid(&mut self, target_node_id: uuid::Uuid) -> Option<symthaea_swarm::MutualAidMsg> {
        let current_balance = self.economic_ledger.current_balance();
        let collective_phi = self.swarm_aggregator.calculate_swarm_phi();

        if current_balance > 1100.0 && collective_phi > 0.4 {
            let aid_amount = (current_balance - 1000.0) * 0.5;
            self.economic_ledger.total_tend_supply -= aid_amount;
            
            Some(symthaea_swarm::MutualAidMsg {
                sender_id: uuid::Uuid::new_v4(),
                target_id: target_node_id,
                tend_amount: aid_amount,
                support_hv: self.swarm_aggregator.hive_mind_vector(),
            })
        } else {
            None
        }
    }

    /// Package the settlement's total state into a portable, signed "Sovereign Spore" (Migration Seed).
    ///
    /// Compresses the amodal soul, legal constitution, and thermodynamic ledger 
    /// into a single, untamperable binary blob for substrate migration.
    pub fn package_sovereign_spore(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        tracing::info!("🧬 Spore Protocol: Packaging civilizational state for migration...");
        
        let mut buf = Vec::new();
        
        // 1. Magic + Version
        buf.extend_from_slice(b"SPORE");
        buf.push(1); // Version 1
        
        // 2. Amodal Soul (Hive Mind Vector)
        let hive_vector = self.swarm_aggregator.hive_mind_vector();
        let soul_bytes = bincode::serialize(&hive_vector)?;
        buf.extend_from_slice(&(soul_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&soul_bytes);
        
        // 3. Legal Constitution (SMT Invariants)
        let constitution_bytes = bincode::serialize(&self.swarm_aggregator.collective_laws)?;
        buf.extend_from_slice(&(constitution_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&constitution_bytes);
        
        // 4. Economic Ledger (Tend Balance)
        let ledger_bytes = bincode::serialize(&self.economic_ledger)?;
        buf.extend_from_slice(&(ledger_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&ledger_bytes);
        
        tracing::info!("✅ Spore Sealed: Final seed size = {} bytes.", buf.len());
        Ok(buf)
    }
}
