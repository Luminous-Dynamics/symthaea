// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Living Basin OS V0.
//!
//! A basin is modeled as metabolism, memory, agency, law, disturbance, and
//! recovery. This crate intentionally starts as a deterministic CPU reference
//! model so higher layers can attach Bevy, Device Bus, Chronicle, and GPU paths
//! without weakening the ecological contract.

use symtropy_lifesim_core::{DiffusionParams, FieldGrid, FieldLayer};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterState {
    pub standing: f32,
    pub flow: f32,
    pub pressure: f32,
    pub trust: f32,
}

impl Default for WaterState {
    fn default() -> Self {
        Self {
            standing: 0.05,
            flow: 0.65,
            pressure: 0.75,
            trust: 0.75,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilState {
    pub moisture: f32,
    pub carbon: f32,
    pub compaction: f32,
    pub decomposer_capacity: f32,
}

impl Default for SoilState {
    fn default() -> Self {
        Self {
            moisture: 0.45,
            carbon: 0.55,
            compaction: 0.25,
            decomposer_capacity: 0.55,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereState {
    pub humidity: f32,
    pub particulates: f32,
}

impl Default for AtmosphereState {
    fn default() -> Self {
        Self {
            humidity: 0.45,
            particulates: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatState {
    pub temperature_c: f32,
    pub heat_stress: f32,
}

impl Default for HeatState {
    fn default() -> Self {
        Self {
            temperature_c: 18.0,
            heat_stress: 0.15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToxinState {
    pub dissolved_metals: f32,
    pub hydrocarbons: f32,
    pub bioavailable: f32,
}

impl Default for ToxinState {
    fn default() -> Self {
        Self {
            dissolved_metals: 0.15,
            hydrocarbons: 0.10,
            bioavailable: 0.12,
        }
    }
}

impl ToxinState {
    pub fn total(self) -> f32 {
        self.dissolved_metals + self.hydrocarbons + self.bioavailable
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiationState {
    pub background: f32,
    pub anomaly: f32,
}

impl Default for RadiationState {
    fn default() -> Self {
        Self {
            background: 0.05,
            anomaly: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasinCell {
    pub water: WaterState,
    pub soil: SoilState,
    pub atmosphere: AtmosphereState,
    pub heat: HeatState,
    pub toxin_load: ToxinState,
    pub radiation: RadiationState,
    pub salinity: f32,
    pub erosion_risk: f32,
    pub root_intrusion: f32,
    pub infrastructure_integrity: f32,
}

impl Default for BasinCell {
    fn default() -> Self {
        Self {
            water: WaterState::default(),
            soil: SoilState::default(),
            atmosphere: AtmosphereState::default(),
            heat: HeatState::default(),
            toxin_load: ToxinState::default(),
            radiation: RadiationState::default(),
            salinity: 0.08,
            erosion_risk: 0.20,
            root_intrusion: 0.05,
            infrastructure_integrity: 0.75,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MetabolicFlux {
    pub water: f32,
    pub nutrient: f32,
    pub carbon: f32,
    pub toxin: f32,
    pub heat: f32,
    pub biomass: f32,
    pub waste: f32,
    pub disease: f32,
    pub signal: f32,
    pub labor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrophicMemory {
    pub producer_health: f32,
    pub decomposer_capacity: f32,
    pub pollination_reliability: f32,
    pub grazer_pressure: f32,
    pub predator_regulation: f32,
    pub scavenger_efficiency: f32,
    pub disease_suppression: f32,
    pub soil_engineering: f32,
    pub extinction_debt: f32,
    pub invasive_pressure: f32,
    pub recovery_momentum: f32,
}

impl Default for TrophicMemory {
    fn default() -> Self {
        Self {
            producer_health: 0.55,
            decomposer_capacity: 0.55,
            pollination_reliability: 0.55,
            grazer_pressure: 0.35,
            predator_regulation: 0.45,
            scavenger_efficiency: 0.50,
            disease_suppression: 0.55,
            soil_engineering: 0.45,
            extinction_debt: 0.20,
            invasive_pressure: 0.20,
            recovery_momentum: 0.35,
        }
    }
}

impl TrophicMemory {
    fn update_lagged(&mut self, flux: MetabolicFlux, toxin_load: f32, water_stress: f32) {
        self.producer_health = lag(
            self.producer_health,
            0.75 - toxin_load * 0.35 - water_stress * 0.20,
            0.05,
        );
        self.decomposer_capacity = lag(
            self.decomposer_capacity,
            0.70 + flux.nutrient * 0.05 - toxin_load * 0.25,
            0.04,
        );
        self.disease_suppression = lag(
            self.disease_suppression,
            0.70 - flux.disease * 0.08 + self.predator_regulation * 0.05,
            0.04,
        );
        self.soil_engineering = lag(
            self.soil_engineering,
            0.45 + flux.biomass * 0.04 + self.decomposer_capacity * 0.08,
            0.035,
        );
        self.extinction_debt = lag(
            self.extinction_debt,
            (toxin_load * 0.55 + water_stress * 0.30 + self.invasive_pressure * 0.20).min(1.0),
            0.03,
        );
        self.recovery_momentum = lag(
            self.recovery_momentum,
            (self.decomposer_capacity + self.producer_health + self.soil_engineering) / 3.0
                - toxin_load * 0.20,
            0.04,
        );
        self.clamp();
    }

    fn clamp(&mut self) {
        self.producer_health = clamp01(self.producer_health);
        self.decomposer_capacity = clamp01(self.decomposer_capacity);
        self.pollination_reliability = clamp01(self.pollination_reliability);
        self.grazer_pressure = clamp01(self.grazer_pressure);
        self.predator_regulation = clamp01(self.predator_regulation);
        self.scavenger_efficiency = clamp01(self.scavenger_efficiency);
        self.disease_suppression = clamp01(self.disease_suppression);
        self.soil_engineering = clamp01(self.soil_engineering);
        self.extinction_debt = clamp01(self.extinction_debt);
        self.invasive_pressure = clamp01(self.invasive_pressure);
        self.recovery_momentum = clamp01(self.recovery_momentum);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    Pheromone,
    RootExudate,
    FungalPulse,
    BirdAlarm,
    WaterTurbidity,
    MachineDiagnostic,
    Vibration,
    ChemicalGradient,
    FieldDeckScan,
    NullChatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSource {
    Basin,
    Colony,
    Mycelium,
    Settlement,
    DeviceBus,
    NullSystem,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalField {
    pub kind: SignalKind,
    pub intensity: f32,
    pub decay_rate: f32,
    pub diffusion_rate: f32,
    pub source: SignalSource,
    pub readable_by: Vec<SystemId>,
    pub corruption: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgencyStructure {
    Colony,
    MycelialNetwork,
    Wetland,
    Settlement,
    MachineEcology,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViabilityProfile {
    pub entity_id: EntityId,
    pub agency_structure: AgencyStructure,
    pub viability: f32,
    pub boundary_integrity: f32,
    pub metabolic_stability: f32,
    pub signal_coherence: f32,
    pub reproductive_continuity: f32,
    pub habitat_continuity: f32,
    pub harm_perception: f32,
    pub reciprocity_trust: f32,
    pub cohabitation_possibility: f32,
}

impl Default for ViabilityProfile {
    fn default() -> Self {
        Self {
            entity_id: EntityId(0),
            agency_structure: AgencyStructure::Wetland,
            viability: 0.60,
            boundary_integrity: 0.65,
            metabolic_stability: 0.60,
            signal_coherence: 0.65,
            reproductive_continuity: 0.55,
            habitat_continuity: 0.60,
            harm_perception: 0.20,
            reciprocity_trust: 0.45,
            cohabitation_possibility: 0.55,
        }
    }
}

impl ViabilityProfile {
    fn update_from_basin(
        &mut self,
        average: BasinCell,
        memory: TrophicMemory,
        null_corruption: f32,
    ) {
        let toxin = average.toxin_load.total().min(1.0);
        let water_stress = (average.water.standing + (1.0 - average.water.flow)).min(1.0);
        self.boundary_integrity = lag(
            self.boundary_integrity,
            0.80 - toxin * 0.25 - average.erosion_risk * 0.20,
            0.08,
        );
        self.metabolic_stability = lag(
            self.metabolic_stability,
            0.75 + memory.recovery_momentum * 0.15 - water_stress * 0.25 - toxin * 0.25,
            0.08,
        );
        self.signal_coherence = lag(self.signal_coherence, 0.80 - null_corruption * 0.65, 0.10);
        self.habitat_continuity = lag(
            self.habitat_continuity,
            0.75 - average.soil.compaction * 0.25 - average.root_intrusion * 0.08,
            0.05,
        );
        self.harm_perception = lag(
            self.harm_perception,
            (toxin * 0.50 + water_stress * 0.35 + null_corruption * 0.30).min(1.0),
            0.12,
        );
        self.cohabitation_possibility = lag(
            self.cohabitation_possibility,
            0.45 + average.water.trust * 0.20 + memory.recovery_momentum * 0.25
                - self.harm_perception * 0.20,
            0.06,
        );
        self.viability = clamp01(
            (self.boundary_integrity
                + self.metabolic_stability
                + self.signal_coherence
                + self.habitat_continuity
                + self.cohabitation_possibility
                + (1.0 - self.harm_perception))
                / 6.0,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FactionId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EcoSystemId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimType {
    ProtectAsLivingInfrastructure,
    AuthorizeMechanicalRepair,
    RequireEvidenceReview,
    QuarantineRisk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimJustification {
    FloodBuffering,
    WaterSecurity,
    ToxinCapture,
    Biosecurity,
    BoundaryIntegrity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChronicleEvidence {
    PipeLeakDetected,
    ToxinTrend,
    RootIntrusion,
    SignalCorruption,
    RecoveryTrajectory,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EcoCivicClaim {
    pub claimant: FactionId,
    pub target: EcoSystemId,
    pub claim_type: ClaimType,
    pub justification: ClaimJustification,
    pub evidence: Vec<ChronicleEvidence>,
    pub legitimacy: f32,
    pub opposition: Vec<FactionId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasinIntervention {
    PipeLeak,
    FastMechanicalRepair,
    EcologicalReroute,
    WillowPlanting,
    NullGreenwash,
    DecomposerAid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasinMetrics {
    pub tick: u64,
    pub water_trust: f32,
    pub toxin_load: f32,
    pub standing_water: f32,
    pub infrastructure_integrity: f32,
    pub recovery_momentum: f32,
    pub extinction_debt: f32,
    pub viability: f32,
    pub signal_corruption: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasinOrganismKind {
    Colony,
    Mycelium,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasinExchangeReport {
    pub organism: BasinOrganismKind,
    pub signal_input: f32,
    pub biomass_input: f32,
    pub toxin_buffered: f32,
    pub civic_evidence_added: bool,
}

#[derive(Debug, Clone)]
pub struct BasinWorld {
    width: usize,
    height: usize,
    cells: Vec<BasinCell>,
    pub fields: FieldGrid,
    pub flux: MetabolicFlux,
    pub memory: TrophicMemory,
    pub viability: ViabilityProfile,
    pub signals: Vec<SignalField>,
    pub civic_claims: Vec<EcoCivicClaim>,
    tick: u64,
}

impl BasinWorld {
    pub fn old_waterworks(width: usize, height: usize) -> Self {
        let mut fields = FieldGrid::new(width, height);
        let mut cells = vec![BasinCell::default(); width * height];
        let center = (width / 2, height / 2);
        let leak = idx(width, center.0, center.1);
        cells[leak].water.flow = 0.35;
        cells[leak].water.pressure = 0.40;
        cells[leak].water.standing = 0.45;
        cells[leak].toxin_load.dissolved_metals = 0.40;
        cells[leak].toxin_load.bioavailable = 0.35;
        cells[leak].infrastructure_integrity = 0.35;
        fields.set(FieldLayer::Moisture, center.0, center.1, 45.0);
        fields.set(FieldLayer::Toxin, center.0, center.1, 60.0);

        let mut world = Self {
            width,
            height,
            cells,
            fields,
            flux: MetabolicFlux::default(),
            memory: TrophicMemory::default(),
            viability: ViabilityProfile::default(),
            signals: Vec::new(),
            civic_claims: Vec::new(),
            tick: 0,
        };
        world.add_signal(
            SignalKind::WaterTurbidity,
            SignalSource::Basin,
            0.55,
            0.04,
            0.08,
            0.0,
        );
        world.add_civic_claim(EcoCivicClaim {
            claimant: FactionId(1),
            target: EcoSystemId(1),
            claim_type: ClaimType::RequireEvidenceReview,
            justification: ClaimJustification::WaterSecurity,
            evidence: vec![ChronicleEvidence::PipeLeakDetected],
            legitimacy: 0.50,
            opposition: vec![FactionId(2)],
        });
        world
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub fn cell(&self, x: usize, y: usize) -> BasinCell {
        self.cells[idx(self.width, x, y)]
    }

    /// Ingest a living system's field traces into basin metabolism and agency.
    ///
    /// This is the first shared contract between basin, colony, and mycelium:
    /// organism crates only need to exchange `FieldGrid`, while the basin turns
    /// those traces into flux, signals, viability pressure, and civic evidence.
    pub fn ingest_organism_fields(
        &mut self,
        organism: BasinOrganismKind,
        organism_fields: &FieldGrid,
    ) -> BasinExchangeReport {
        assert_eq!(self.width, organism_fields.width(), "field widths differ");
        assert_eq!(
            self.height,
            organism_fields.height(),
            "field heights differ"
        );

        match organism {
            BasinOrganismKind::Colony => self.ingest_colony_fields(organism_fields),
            BasinOrganismKind::Mycelium => self.ingest_mycelium_fields(organism_fields),
        }
    }

    pub fn apply(&mut self, intervention: BasinIntervention) {
        match intervention {
            BasinIntervention::PipeLeak => {
                for cell in &mut self.cells {
                    cell.water.standing = clamp01(cell.water.standing + 0.10);
                    cell.water.flow = clamp01(cell.water.flow - 0.08);
                    cell.water.pressure = clamp01(cell.water.pressure - 0.12);
                    cell.water.trust = clamp01(cell.water.trust - 0.10);
                    cell.toxin_load.bioavailable = clamp01(cell.toxin_load.bioavailable + 0.06);
                    cell.infrastructure_integrity = clamp01(cell.infrastructure_integrity - 0.08);
                }
                self.flux.water += 3.0;
                self.flux.toxin += 2.0;
                self.flux.disease += 1.0;
                self.add_signal(
                    SignalKind::WaterTurbidity,
                    SignalSource::Basin,
                    0.45,
                    0.04,
                    0.10,
                    0.0,
                );
            }
            BasinIntervention::FastMechanicalRepair => {
                for cell in &mut self.cells {
                    cell.water.pressure = clamp01(cell.water.pressure + 0.30);
                    cell.water.flow = clamp01(cell.water.flow + 0.20);
                    cell.water.standing = clamp01(cell.water.standing - 0.18);
                    cell.infrastructure_integrity = clamp01(cell.infrastructure_integrity + 0.22);
                    cell.root_intrusion = clamp01(cell.root_intrusion - 0.05);
                    cell.soil.compaction = clamp01(cell.soil.compaction + 0.10);
                    cell.soil.decomposer_capacity = clamp01(cell.soil.decomposer_capacity - 0.06);
                }
                self.flux.labor += 2.5;
                self.flux.waste += 0.8;
                self.add_civic_claim(EcoCivicClaim {
                    claimant: FactionId(2),
                    target: EcoSystemId(1),
                    claim_type: ClaimType::AuthorizeMechanicalRepair,
                    justification: ClaimJustification::WaterSecurity,
                    evidence: vec![ChronicleEvidence::PipeLeakDetected],
                    legitimacy: 0.60,
                    opposition: vec![FactionId(1)],
                });
            }
            BasinIntervention::EcologicalReroute => {
                for cell in &mut self.cells {
                    cell.water.flow = clamp01(cell.water.flow + 0.12);
                    cell.water.pressure = clamp01(cell.water.pressure + 0.12);
                    cell.water.standing = clamp01(cell.water.standing - 0.10);
                    cell.toxin_load.bioavailable = clamp01(cell.toxin_load.bioavailable - 0.06);
                    cell.soil.decomposer_capacity = clamp01(cell.soil.decomposer_capacity + 0.08);
                    cell.soil.compaction = clamp01(cell.soil.compaction - 0.04);
                }
                self.flux.labor += 1.6;
                self.flux.toxin -= 0.8;
                self.flux.nutrient += 0.9;
                self.add_civic_claim(EcoCivicClaim {
                    claimant: FactionId(1),
                    target: EcoSystemId(1),
                    claim_type: ClaimType::ProtectAsLivingInfrastructure,
                    justification: ClaimJustification::FloodBuffering,
                    evidence: vec![
                        ChronicleEvidence::PipeLeakDetected,
                        ChronicleEvidence::RecoveryTrajectory,
                    ],
                    legitimacy: 0.64,
                    opposition: vec![FactionId(2)],
                });
            }
            BasinIntervention::WillowPlanting => {
                for cell in &mut self.cells {
                    cell.toxin_load.dissolved_metals =
                        clamp01(cell.toxin_load.dissolved_metals - 0.07);
                    cell.toxin_load.bioavailable = clamp01(cell.toxin_load.bioavailable - 0.04);
                    cell.root_intrusion = clamp01(cell.root_intrusion + 0.12);
                    cell.soil.carbon = clamp01(cell.soil.carbon + 0.05);
                }
                self.flux.biomass += 1.2;
                self.flux.toxin -= 0.9;
                self.add_signal(
                    SignalKind::RootExudate,
                    SignalSource::Basin,
                    0.35,
                    0.03,
                    0.04,
                    0.0,
                );
            }
            BasinIntervention::NullGreenwash => {
                self.add_signal(
                    SignalKind::MachineDiagnostic,
                    SignalSource::NullSystem,
                    0.80,
                    0.02,
                    0.02,
                    0.85,
                );
                self.flux.signal += 1.5;
            }
            BasinIntervention::DecomposerAid => {
                for cell in &mut self.cells {
                    cell.soil.decomposer_capacity = clamp01(cell.soil.decomposer_capacity + 0.12);
                    cell.toxin_load.hydrocarbons = clamp01(cell.toxin_load.hydrocarbons - 0.05);
                }
                self.flux.nutrient += 1.0;
                self.flux.carbon += 0.8;
            }
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        self.update_fields();
        self.diffuse_fields();
        self.decay_signals();
        let average = self.average_cell();
        let toxin = average.toxin_load.total().min(1.0);
        let water_stress = (average.water.standing + (1.0 - average.water.flow)).min(1.0);
        self.memory.update_lagged(self.flux, toxin, water_stress);
        self.viability
            .update_from_basin(average, self.memory, self.signal_corruption());
        self.update_civic_legitimacy();
        self.flux = decay_flux(self.flux);
    }

    pub fn run_steps(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    pub fn metrics(&self) -> BasinMetrics {
        let average = self.average_cell();
        BasinMetrics {
            tick: self.tick,
            water_trust: average.water.trust,
            toxin_load: average.toxin_load.total(),
            standing_water: average.water.standing,
            infrastructure_integrity: average.infrastructure_integrity,
            recovery_momentum: self.memory.recovery_momentum,
            extinction_debt: self.memory.extinction_debt,
            viability: self.viability.viability,
            signal_corruption: self.signal_corruption(),
        }
    }

    pub fn to_ppm_heatmap(&self) -> String {
        let mut out = format!(
            "P3\n# symtropy-basin tick {}\n{} {}\n255\n",
            self.tick, self.width, self.height
        );
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = self.cell(x, y);
                let r = (cell.toxin_load.total().min(1.0) * 255.0) as u8;
                let g = (self.memory.recovery_momentum * 255.0) as u8;
                let b = (cell.water.standing.min(1.0) * 255.0) as u8;
                out.push_str(&format!("{r} {g} {b} "));
            }
            out.push('\n');
        }
        out
    }

    fn add_signal(
        &mut self,
        kind: SignalKind,
        source: SignalSource,
        intensity: f32,
        decay_rate: f32,
        diffusion_rate: f32,
        corruption: f32,
    ) {
        self.signals.push(SignalField {
            kind,
            intensity: clamp01(intensity),
            decay_rate,
            diffusion_rate,
            source,
            readable_by: vec![SystemId(1), SystemId(2)],
            corruption: clamp01(corruption),
        });
    }

    fn add_civic_claim(&mut self, claim: EcoCivicClaim) {
        self.civic_claims.push(claim);
    }

    fn ingest_colony_fields(&mut self, fields: &FieldGrid) -> BasinExchangeReport {
        let pheromone = fields.channel_sum(FieldLayer::FoodPheromone)
            + fields.channel_sum(FieldLayer::HomePheromone);
        let danger = fields.channel_sum(FieldLayer::DangerPheromone);
        let cells = (self.width * self.height) as f32;
        let signal_input = (pheromone / cells / 100.0).min(1.0);
        let danger_pressure = (danger / cells / 40.0).min(1.0);

        if signal_input > 0.01 {
            self.add_signal(
                SignalKind::Pheromone,
                SignalSource::Colony,
                signal_input,
                0.05,
                0.08,
                0.0,
            );
        }
        self.flux.signal += signal_input;
        self.flux.biomass += (pheromone / cells / 500.0).min(0.5);
        self.viability.harm_perception =
            clamp01(self.viability.harm_perception + danger_pressure * 0.05);

        let mut civic_evidence_added = false;
        if danger_pressure > 0.08 {
            self.add_civic_claim(EcoCivicClaim {
                claimant: FactionId(3),
                target: EcoSystemId(1),
                claim_type: ClaimType::QuarantineRisk,
                justification: ClaimJustification::Biosecurity,
                evidence: vec![ChronicleEvidence::SignalCorruption],
                legitimacy: 0.45 + danger_pressure * 0.20,
                opposition: vec![FactionId(1)],
            });
            civic_evidence_added = true;
        }

        BasinExchangeReport {
            organism: BasinOrganismKind::Colony,
            signal_input,
            biomass_input: 0.0,
            toxin_buffered: 0.0,
            civic_evidence_added,
        }
    }

    fn ingest_mycelium_fields(&mut self, fields: &FieldGrid) -> BasinExchangeReport {
        let biomass = fields.channel_sum(FieldLayer::Biomass);
        let nutrient = fields.channel_sum(FieldLayer::Nutrient);
        let cells = (self.width * self.height) as f32;
        let biomass_input = (biomass / cells / 20.0).min(1.0);
        let nutrient_input = (nutrient / cells / 100.0).min(1.0);
        let mut toxin_buffered = 0.0;

        for y in 0..self.height {
            for x in 0..self.width {
                let biomass_here = fields.get(FieldLayer::Biomass, x, y);
                if biomass_here <= 0.0 {
                    continue;
                }
                let cell = &mut self.cells[idx(self.width, x, y)];
                let buffered = (biomass_here * 0.002).min(cell.toxin_load.bioavailable);
                cell.toxin_load.bioavailable = clamp01(cell.toxin_load.bioavailable - buffered);
                cell.soil.decomposer_capacity =
                    clamp01(cell.soil.decomposer_capacity + biomass_here * 0.0008);
                cell.soil.carbon = clamp01(cell.soil.carbon + biomass_here * 0.0005);
                toxin_buffered += buffered;
            }
        }

        if biomass_input > 0.01 {
            self.add_signal(
                SignalKind::FungalPulse,
                SignalSource::Mycelium,
                biomass_input,
                0.03,
                0.04,
                0.0,
            );
        }
        self.flux.biomass += biomass_input;
        self.flux.nutrient += nutrient_input;
        self.flux.toxin -= toxin_buffered;

        let civic_evidence_added = toxin_buffered > 0.05;
        if civic_evidence_added {
            self.add_civic_claim(EcoCivicClaim {
                claimant: FactionId(1),
                target: EcoSystemId(1),
                claim_type: ClaimType::ProtectAsLivingInfrastructure,
                justification: ClaimJustification::ToxinCapture,
                evidence: vec![ChronicleEvidence::ToxinTrend],
                legitimacy: 0.55 + toxin_buffered.min(0.25),
                opposition: vec![FactionId(2)],
            });
        }

        BasinExchangeReport {
            organism: BasinOrganismKind::Mycelium,
            signal_input: biomass_input,
            biomass_input,
            toxin_buffered,
            civic_evidence_added,
        }
    }

    fn update_fields(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = self.cell(x, y);
                self.fields.set(
                    FieldLayer::Moisture,
                    x,
                    y,
                    cell.water.standing * 100.0 + cell.soil.moisture * 25.0,
                );
                self.fields
                    .set(FieldLayer::Toxin, x, y, cell.toxin_load.total() * 100.0);
                self.fields.set(
                    FieldLayer::Nutrient,
                    x,
                    y,
                    cell.soil.decomposer_capacity * 40.0 + cell.soil.carbon * 20.0,
                );
            }
        }
    }

    fn diffuse_fields(&mut self) {
        let params = DiffusionParams {
            diffusion: 0.05,
            decay: 0.015,
            dt: 1.0,
            max_value: 1_000.0,
        };
        self.fields.step_diffuse_decay(FieldLayer::Moisture, params);
        self.fields.step_diffuse_decay(FieldLayer::Toxin, params);
        self.fields.step_diffuse_decay(FieldLayer::Nutrient, params);
    }

    fn decay_signals(&mut self) {
        for signal in &mut self.signals {
            signal.intensity = clamp01(signal.intensity * (1.0 - signal.decay_rate));
            signal.corruption = clamp01(signal.corruption * (1.0 - signal.decay_rate * 0.5));
        }
        self.signals.retain(|signal| signal.intensity > 0.01);
    }

    fn signal_corruption(&self) -> f32 {
        if self.signals.is_empty() {
            return 0.0;
        }
        let weighted: f32 = self
            .signals
            .iter()
            .map(|signal| signal.intensity * signal.corruption)
            .sum();
        let total: f32 = self.signals.iter().map(|signal| signal.intensity).sum();
        if total <= f32::EPSILON {
            0.0
        } else {
            clamp01(weighted / total)
        }
    }

    fn update_civic_legitimacy(&mut self) {
        let metrics = self.metrics_without_claim_update();
        for claim in &mut self.civic_claims {
            let evidence_bonus = claim.evidence.len() as f32 * 0.03;
            let condition_bonus = match claim.justification {
                ClaimJustification::FloodBuffering => metrics.standing_water * 0.08,
                ClaimJustification::WaterSecurity => (1.0 - metrics.water_trust) * 0.10,
                ClaimJustification::ToxinCapture => metrics.toxin_load.min(1.0) * 0.08,
                ClaimJustification::Biosecurity => metrics.extinction_debt * 0.08,
                ClaimJustification::BoundaryIntegrity => {
                    (1.0 - self.viability.boundary_integrity) * 0.08
                }
            };
            claim.legitimacy = clamp01(claim.legitimacy + evidence_bonus + condition_bonus - 0.02);
        }
    }

    fn metrics_without_claim_update(&self) -> BasinMetrics {
        let average = self.average_cell();
        BasinMetrics {
            tick: self.tick,
            water_trust: average.water.trust,
            toxin_load: average.toxin_load.total(),
            standing_water: average.water.standing,
            infrastructure_integrity: average.infrastructure_integrity,
            recovery_momentum: self.memory.recovery_momentum,
            extinction_debt: self.memory.extinction_debt,
            viability: self.viability.viability,
            signal_corruption: self.signal_corruption(),
        }
    }

    fn average_cell(&self) -> BasinCell {
        let n = self.cells.len() as f32;
        let mut out = BasinCell {
            water: WaterState {
                standing: 0.0,
                flow: 0.0,
                pressure: 0.0,
                trust: 0.0,
            },
            soil: SoilState {
                moisture: 0.0,
                carbon: 0.0,
                compaction: 0.0,
                decomposer_capacity: 0.0,
            },
            atmosphere: AtmosphereState {
                humidity: 0.0,
                particulates: 0.0,
            },
            heat: HeatState {
                temperature_c: 0.0,
                heat_stress: 0.0,
            },
            toxin_load: ToxinState {
                dissolved_metals: 0.0,
                hydrocarbons: 0.0,
                bioavailable: 0.0,
            },
            radiation: RadiationState {
                background: 0.0,
                anomaly: 0.0,
            },
            salinity: 0.0,
            erosion_risk: 0.0,
            root_intrusion: 0.0,
            infrastructure_integrity: 0.0,
        };

        for cell in &self.cells {
            out.water.standing += cell.water.standing / n;
            out.water.flow += cell.water.flow / n;
            out.water.pressure += cell.water.pressure / n;
            out.water.trust += cell.water.trust / n;
            out.soil.moisture += cell.soil.moisture / n;
            out.soil.carbon += cell.soil.carbon / n;
            out.soil.compaction += cell.soil.compaction / n;
            out.soil.decomposer_capacity += cell.soil.decomposer_capacity / n;
            out.atmosphere.humidity += cell.atmosphere.humidity / n;
            out.atmosphere.particulates += cell.atmosphere.particulates / n;
            out.heat.temperature_c += cell.heat.temperature_c / n;
            out.heat.heat_stress += cell.heat.heat_stress / n;
            out.toxin_load.dissolved_metals += cell.toxin_load.dissolved_metals / n;
            out.toxin_load.hydrocarbons += cell.toxin_load.hydrocarbons / n;
            out.toxin_load.bioavailable += cell.toxin_load.bioavailable / n;
            out.radiation.background += cell.radiation.background / n;
            out.radiation.anomaly += cell.radiation.anomaly / n;
            out.salinity += cell.salinity / n;
            out.erosion_risk += cell.erosion_risk / n;
            out.root_intrusion += cell.root_intrusion / n;
            out.infrastructure_integrity += cell.infrastructure_integrity / n;
        }
        out
    }
}

fn decay_flux(flux: MetabolicFlux) -> MetabolicFlux {
    MetabolicFlux {
        water: flux.water * 0.65,
        nutrient: flux.nutrient * 0.70,
        carbon: flux.carbon * 0.70,
        toxin: flux.toxin * 0.72,
        heat: flux.heat * 0.75,
        biomass: flux.biomass * 0.74,
        waste: flux.waste * 0.70,
        disease: flux.disease * 0.68,
        signal: flux.signal * 0.65,
        labor: flux.labor * 0.60,
    }
}

fn idx(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn lag(current: f32, target: f32, rate: f32) -> f32 {
    clamp01(current + (target - current) * rate)
}

fn clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_leak_degrades_trust_and_basin_viability() {
        let mut baseline = BasinWorld::old_waterworks(12, 8);
        baseline.run_steps(8);

        let mut disturbed = BasinWorld::old_waterworks(12, 8);
        disturbed.apply(BasinIntervention::PipeLeak);
        disturbed.run_steps(8);
        let before = baseline.metrics();
        let after = disturbed.metrics();

        assert!(after.water_trust < before.water_trust);
        assert!(after.toxin_load > before.toxin_load);
        assert!(after.viability < before.viability);
        assert!(after.signal_corruption <= 1.0);
    }

    #[test]
    fn willow_filters_toxins_but_raises_root_intrusion_risk() {
        let mut world = BasinWorld::old_waterworks(12, 8);
        let before_toxin = world.metrics().toxin_load;
        let before_root = world
            .metrics_without_claim_update()
            .infrastructure_integrity;

        world.apply(BasinIntervention::WillowPlanting);
        world.run_steps(6);

        assert!(world.metrics().toxin_load < before_toxin);
        assert!(world.average_cell().root_intrusion > 0.05);
        assert!(world.metrics().infrastructure_integrity <= before_root);
    }

    #[test]
    fn ecological_reroute_builds_more_recovery_than_fast_repair() {
        let mut fast = BasinWorld::old_waterworks(12, 8);
        fast.apply(BasinIntervention::PipeLeak);
        fast.apply(BasinIntervention::FastMechanicalRepair);
        fast.run_steps(20);

        let mut reroute = BasinWorld::old_waterworks(12, 8);
        reroute.apply(BasinIntervention::PipeLeak);
        reroute.apply(BasinIntervention::EcologicalReroute);
        reroute.apply(BasinIntervention::DecomposerAid);
        reroute.run_steps(20);

        assert!(fast.metrics().infrastructure_integrity > 0.70);
        assert!(reroute.metrics().recovery_momentum > fast.metrics().recovery_momentum);
        assert!(reroute.metrics().extinction_debt < fast.metrics().extinction_debt);
    }

    #[test]
    fn null_greenwash_reduces_signal_coherence_without_fixing_toxins() {
        let mut world = BasinWorld::old_waterworks(12, 8);
        let toxin_before = world.metrics().toxin_load;

        world.apply(BasinIntervention::NullGreenwash);
        world.run_steps(5);

        let metrics = world.metrics();
        assert!(metrics.signal_corruption > 0.40);
        assert!(world.viability.signal_coherence < 0.65);
        assert!(metrics.toxin_load >= toxin_before * 0.95);
    }

    #[test]
    fn civic_claim_legitimacy_tracks_evidence_and_conditions() {
        let mut world = BasinWorld::old_waterworks(12, 8);
        let before = world.civic_claims[0].legitimacy;

        world.apply(BasinIntervention::EcologicalReroute);
        world.run_steps(4);

        assert!(world.civic_claims.iter().any(|claim| {
            claim.claim_type == ClaimType::ProtectAsLivingInfrastructure
                && claim
                    .evidence
                    .contains(&ChronicleEvidence::RecoveryTrajectory)
        }));
        assert!(world.civic_claims[0].legitimacy > before);
    }

    #[test]
    fn ppm_heatmap_exports_visible_basin_state() {
        let mut world = BasinWorld::old_waterworks(6, 4);
        world.apply(BasinIntervention::PipeLeak);
        world.run_steps(2);

        let ppm = world.to_ppm_heatmap();

        assert!(ppm.starts_with("P3\n# symtropy-basin tick 2\n6 4\n255\n"));
    }

    #[test]
    fn colony_fields_become_basin_signal_and_civic_pressure() {
        let mut world = BasinWorld::old_waterworks(8, 5);
        let mut colony = FieldGrid::new(8, 5);
        colony.set(FieldLayer::HomePheromone, 1, 2, 400.0);
        colony.set(FieldLayer::FoodPheromone, 4, 2, 600.0);
        colony.set(FieldLayer::DangerPheromone, 4, 2, 400.0);

        let report = world.ingest_organism_fields(BasinOrganismKind::Colony, &colony);

        assert_eq!(report.organism, BasinOrganismKind::Colony);
        assert!(report.signal_input > 0.0);
        assert!(
            world
                .signals
                .iter()
                .any(|signal| signal.kind == SignalKind::Pheromone
                    && signal.source == SignalSource::Colony)
        );
        assert!(
            world
                .civic_claims
                .iter()
                .any(|claim| claim.claim_type == ClaimType::QuarantineRisk)
        );
    }

    #[test]
    fn mycelium_fields_buffer_basin_toxins_and_add_evidence() {
        let mut world = BasinWorld::old_waterworks(8, 5);
        let before = world.metrics().toxin_load;
        let mut mycelium = FieldGrid::new(8, 5);
        mycelium.set(FieldLayer::Biomass, 4, 2, 30.0);
        mycelium.set(FieldLayer::Nutrient, 4, 2, 200.0);

        let report = world.ingest_organism_fields(BasinOrganismKind::Mycelium, &mycelium);
        world.run_steps(4);

        assert!(report.biomass_input > 0.0);
        assert!(report.toxin_buffered > 0.0);
        assert!(world.metrics().toxin_load < before);
        assert!(
            world
                .civic_claims
                .iter()
                .any(|claim| claim.justification == ClaimJustification::ToxinCapture)
        );
    }
}
