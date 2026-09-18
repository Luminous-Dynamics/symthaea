// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight state-aware procedural boot ecology.
//!
//! `BootStateReceipt` records operational machine facts. `BootEcologyComposer`
//! deterministically turns those facts plus a bounded `MorphologyLineage` into a
//! `BootGenome` that low-level renderers can consume. This crate deliberately has
//! only `serde` and `blake3` dependencies so both Spore and an early-boot DRM/KMS
//! renderer can share one protocol without pulling the cognitive runtime into the
//! boot path.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

pub const BOOT_ECOLOGY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PreviousTermination {
    FirstBoot,
    CleanPoweroff,
    CleanReboot,
    Suspend,
    Hibernate,
    PowerLoss,
    KernelPanic,
    WatchdogReset,
    ThermalEmergency,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GenerationTransition {
    Same,
    Updated { from: String, to: String },
    RolledBack { attempted: String, restored: String },
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageState {
    Clean,
    JournalReplayed,
    Repaired { repairs: u32 },
    Degraded,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GenerationHealth {
    KnownGood,
    PreviousBootIncomplete,
    Recovery,
    Unknown,
}

/// Facts captured before visual composition.
///
/// User mood, biometrics, private content, filenames, process names, journal
/// text, and peer identities are intentionally excluded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BootStateReceipt {
    pub schema_version: u16,
    /// Stable random per-machine visual identity. Never use this as a security key.
    pub machine_visual_seed: [u8; 32],
    pub boot_counter: u64,
    pub previous_termination: PreviousTermination,
    pub previous_uptime_secs: u64,
    pub generation_transition: GenerationTransition,
    pub generation_health: GenerationHealth,
    pub storage_state: StorageState,
    pub oom_events: u32,
    pub thermal_events: u32,
    /// Privacy-preserving digest of coarse hardware topology.
    pub hardware_fingerprint: Option<String>,
    pub previous_hardware_fingerprint: Option<String>,
    pub mesh_enabled: bool,
    /// Coarse count only; identities do not enter the visual seed.
    pub mesh_peers_last_seen: u32,
}

impl BootStateReceipt {
    pub fn first_boot(machine_visual_seed: [u8; 32]) -> Self {
        Self {
            schema_version: BOOT_ECOLOGY_SCHEMA_VERSION,
            machine_visual_seed,
            boot_counter: 0,
            previous_termination: PreviousTermination::FirstBoot,
            previous_uptime_secs: 0,
            generation_transition: GenerationTransition::Unknown,
            generation_health: GenerationHealth::Unknown,
            storage_state: StorageState::Clean,
            oom_events: 0,
            thermal_events: 0,
            hardware_fingerprint: None,
            previous_hardware_fingerprint: None,
            mesh_enabled: false,
            mesh_peers_last_seen: 0,
        }
    }

    pub fn hardware_changed(&self) -> bool {
        match (&self.previous_hardware_fingerprint, &self.hardware_fingerprint) {
            (Some(previous), Some(current)) => previous != current,
            _ => false,
        }
    }

    pub fn needs_repair_visual(&self) -> bool {
        matches!(
            self.previous_termination,
            PreviousTermination::PowerLoss
                | PreviousTermination::KernelPanic
                | PreviousTermination::WatchdogReset
                | PreviousTermination::ThermalEmergency
        ) || !matches!(self.storage_state, StorageState::Clean | StorageState::Unknown)
            || matches!(self.generation_health, GenerationHealth::Recovery)
    }
}

/// A family is a visual grammar, not a prerecorded animation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MorphologyFamily {
    CentralSpore,
    MycelialFan,
    LichenCells,
    ConstellationHyphae,
    RiverDelta,
    AnastomoticWeb,
    FairyRing,
    HdcOrganic,
    CrystalThaw,
    KintsugiRepair,
    MemoryGarden,
    MinimalRelight,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BootStageKind {
    Blackout,
    DormantCore,
    Relight,
    Germinate,
    Grow,
    Anastomose,
    Repair,
    GrowthRing,
    HardwareBud,
    RetractFailedGrowth,
    MeshLink,
    Settle,
    Handoff,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BootStage {
    pub kind: BootStageKind,
    /// Rendering budget only. Never a boot delay.
    pub duration_ms: u32,
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MorphologyParameters {
    pub symmetry: f32,
    pub curvature: f32,
    pub branching_probability: f32,
    pub anastomosis_probability: f32,
    pub spore_count: u8,
    pub growth_anisotropy: f32,
    pub node_density: f32,
    pub pulse_velocity: f32,
    pub turbulence: f32,
    pub color_temperature_k: f32,
    pub solar_gold_fraction: f32,
    pub leaf_green_fraction: f32,
    pub mycelial_white_fraction: f32,
    pub glow_radius: f32,
    pub growth_velocity: f32,
    pub camera_scale: f32,
    pub repair_intensity: f32,
    pub mesh_opacity: f32,
    pub maturity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BootRenderPolicy {
    pub fail_open: bool,
    pub acquire_timeout_ms: u32,
    pub hard_deadline_ms: u32,
    pub release_before_display_manager: bool,
    pub progress_source_optional: bool,
    pub target_fps: u16,
}

impl Default for BootRenderPolicy {
    fn default() -> Self {
        Self {
            fail_open: true,
            acquire_timeout_ms: 500,
            hard_deadline_ms: 9_000,
            release_before_display_manager: true,
            progress_source_optional: true,
            target_fps: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BootCue {
    FirstBoot,
    Starting,
    Resuming,
    ApplyingGeneration,
    RestoringKnownGood,
    RecoveringState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BootGenome {
    pub schema_version: u16,
    pub seed: [u8; 32],
    pub family: MorphologyFamily,
    pub accent_family: Option<MorphologyFamily>,
    pub parameters: MorphologyParameters,
    pub stages: Vec<BootStage>,
    pub cue: BootCue,
    pub render_policy: BootRenderPolicy,
}

impl BootGenome {
    pub fn seed_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.seed {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }

    pub fn visual_budget_ms(&self) -> u32 {
        self.stages.iter().map(|stage| stage.duration_ms).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BootOutcome {
    ReachedGraphicalTarget { generation: String },
    RolledBack { restored_generation: String },
    FailedBeforeHealthGate,
}

/// Bounded abstract visual history. No framebuffers or personal data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MorphologyLineage {
    pub schema_version: u16,
    pub successful_boots: u64,
    pub recovery_marks: u32,
    pub maturity: f32,
    pub last_genome_seed: Option<[u8; 32]>,
    pub last_known_good_generation: Option<String>,
}

impl Default for MorphologyLineage {
    fn default() -> Self {
        Self {
            schema_version: BOOT_ECOLOGY_SCHEMA_VERSION,
            successful_boots: 0,
            recovery_marks: 0,
            maturity: 0.0,
            last_genome_seed: None,
            last_known_good_generation: None,
        }
    }
}

impl MorphologyLineage {
    pub fn record_outcome(&mut self, genome: &BootGenome, outcome: &BootOutcome) {
        self.last_genome_seed = Some(genome.seed);
        match outcome {
            BootOutcome::ReachedGraphicalTarget { generation } => {
                self.successful_boots = self.successful_boots.saturating_add(1);
                self.maturity = (self.maturity + 0.004).min(1.0);
                self.last_known_good_generation = Some(generation.clone());
            }
            BootOutcome::RolledBack { restored_generation } => {
                self.recovery_marks = self.recovery_marks.saturating_add(1);
                self.last_known_good_generation = Some(restored_generation.clone());
            }
            BootOutcome::FailedBeforeHealthGate => {}
        }
    }
}

pub struct BootEcologyComposer;

impl BootEcologyComposer {
    pub fn compose(receipt: &BootStateReceipt, lineage: &MorphologyLineage) -> BootGenome {
        let seed = derive_seed(receipt, lineage);
        let family = select_family(receipt, seed[0]);
        let accent_family = select_accent_family(family, seed[1]);
        let cue = select_cue(receipt);
        let parameters = compose_parameters(receipt, lineage, &seed);
        let stages = compose_stages(receipt, &parameters);

        BootGenome {
            schema_version: BOOT_ECOLOGY_SCHEMA_VERSION,
            seed,
            family,
            accent_family,
            parameters,
            stages,
            cue,
            render_policy: BootRenderPolicy::default(),
        }
    }
}

fn derive_seed(receipt: &BootStateReceipt, lineage: &MorphologyLineage) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"spore-boot-ecology-v0.3\0");
    hasher.update(&receipt.machine_visual_seed);
    hasher.update(&receipt.schema_version.to_le_bytes());
    hasher.update(&receipt.boot_counter.to_le_bytes());
    hasher.update(&receipt.previous_uptime_secs.to_le_bytes());
    hash_debug(&mut hasher, &receipt.previous_termination);
    hash_debug(&mut hasher, &receipt.generation_transition);
    hash_debug(&mut hasher, &receipt.generation_health);
    hash_debug(&mut hasher, &receipt.storage_state);
    hasher.update(&receipt.oom_events.to_le_bytes());
    hasher.update(&receipt.thermal_events.to_le_bytes());
    hash_optional_string(&mut hasher, receipt.hardware_fingerprint.as_deref());
    // Only the changed/not-changed fact from the prior digest contributes.
    hasher.update(&[u8::from(receipt.hardware_changed())]);
    hasher.update(&[u8::from(receipt.mesh_enabled)]);
    hasher.update(&receipt.mesh_peers_last_seen.to_le_bytes());
    hasher.update(&lineage.successful_boots.to_le_bytes());
    hasher.update(&lineage.recovery_marks.to_le_bytes());
    if let Some(last_seed) = lineage.last_genome_seed {
        hasher.update(&last_seed);
    }
    *hasher.finalize().as_bytes()
}

fn hash_debug<T: std::fmt::Debug>(hasher: &mut Hasher, value: &T) {
    let rendered = format!("{value:?}");
    hasher.update(&(rendered.len() as u64).to_le_bytes());
    hasher.update(rendered.as_bytes());
}

fn hash_optional_string(hasher: &mut Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn select_family(receipt: &BootStateReceipt, selector: u8) -> MorphologyFamily {
    if matches!(receipt.previous_termination, PreviousTermination::FirstBoot) {
        return MorphologyFamily::CentralSpore;
    }
    if matches!(receipt.previous_termination, PreviousTermination::Suspend) {
        return MorphologyFamily::MinimalRelight;
    }
    if matches!(receipt.previous_termination, PreviousTermination::Hibernate) {
        return MorphologyFamily::CrystalThaw;
    }
    if matches!(receipt.generation_transition, GenerationTransition::RolledBack { .. })
        || receipt.needs_repair_visual()
    {
        return MorphologyFamily::KintsugiRepair;
    }
    if matches!(receipt.generation_transition, GenerationTransition::Updated { .. }) {
        return MorphologyFamily::FairyRing;
    }

    const HEALTHY: [MorphologyFamily; 8] = [
        MorphologyFamily::MycelialFan,
        MorphologyFamily::LichenCells,
        MorphologyFamily::ConstellationHyphae,
        MorphologyFamily::RiverDelta,
        MorphologyFamily::AnastomoticWeb,
        MorphologyFamily::FairyRing,
        MorphologyFamily::HdcOrganic,
        MorphologyFamily::MemoryGarden,
    ];
    HEALTHY[selector as usize % HEALTHY.len()]
}

fn select_accent_family(primary: MorphologyFamily, selector: u8) -> Option<MorphologyFamily> {
    const ACCENTS: [MorphologyFamily; 4] = [
        MorphologyFamily::ConstellationHyphae,
        MorphologyFamily::AnastomoticWeb,
        MorphologyFamily::HdcOrganic,
        MorphologyFamily::LichenCells,
    ];
    let candidate = ACCENTS[selector as usize % ACCENTS.len()];
    (candidate != primary).then_some(candidate)
}

fn select_cue(receipt: &BootStateReceipt) -> BootCue {
    if matches!(receipt.previous_termination, PreviousTermination::FirstBoot) {
        BootCue::FirstBoot
    } else if matches!(receipt.generation_transition, GenerationTransition::RolledBack { .. }) {
        BootCue::RestoringKnownGood
    } else if receipt.needs_repair_visual() {
        BootCue::RecoveringState
    } else if matches!(
        receipt.previous_termination,
        PreviousTermination::Suspend | PreviousTermination::Hibernate
    ) {
        BootCue::Resuming
    } else if matches!(receipt.generation_transition, GenerationTransition::Updated { .. }) {
        BootCue::ApplyingGeneration
    } else {
        BootCue::Starting
    }
}

fn compose_parameters(
    receipt: &BootStateReceipt,
    lineage: &MorphologyLineage,
    seed: &[u8; 32],
) -> MorphologyParameters {
    let jitter = |index: usize| seed[index] as f32 / 255.0;
    let interrupted = receipt.needs_repair_visual();
    let resumed = matches!(
        receipt.previous_termination,
        PreviousTermination::Suspend | PreviousTermination::Hibernate
    );
    let rollback = matches!(receipt.generation_transition, GenerationTransition::RolledBack { .. });
    let updated = matches!(receipt.generation_transition, GenerationTransition::Updated { .. });

    let uptime_maturity = (receipt.previous_uptime_secs as f32 / (14.0 * 86_400.0)).min(1.0);
    let maturity = (lineage.maturity * 0.65 + uptime_maturity * 0.35).clamp(0.0, 1.0);

    let mut symmetry = if interrupted { 0.42 } else { 0.90 };
    if resumed {
        symmetry = 0.96;
    }
    symmetry = (symmetry + (jitter(2) - 0.5) * 0.08).clamp(0.2, 1.0);

    let repair_intensity = if interrupted {
        0.65 + jitter(3) * 0.30
    } else if rollback {
        0.35
    } else {
        0.0
    };

    let mut solar_gold_fraction = 0.18 + jitter(4) * 0.10;
    if updated || rollback || matches!(receipt.storage_state, StorageState::Repaired { .. }) {
        solar_gold_fraction += 0.18;
    }
    solar_gold_fraction = solar_gold_fraction.min(0.52);

    let mut leaf_green_fraction = 0.52 + jitter(5) * 0.12;
    let mut mycelial_white_fraction = 1.0 - solar_gold_fraction - leaf_green_fraction;
    if mycelial_white_fraction < 0.12 {
        let deficit = 0.12 - mycelial_white_fraction;
        leaf_green_fraction = (leaf_green_fraction - deficit).max(0.25);
        mycelial_white_fraction = 1.0 - solar_gold_fraction - leaf_green_fraction;
    }

    let temperature = match receipt.previous_termination {
        PreviousTermination::ThermalEmergency => 4_000.0,
        PreviousTermination::PowerLoss
        | PreviousTermination::KernelPanic
        | PreviousTermination::WatchdogReset => 4_500.0,
        PreviousTermination::Hibernate => 5_800.0,
        _ => 5_300.0,
    } + (jitter(6) - 0.5) * 300.0;

    MorphologyParameters {
        symmetry,
        curvature: 0.35 + jitter(7) * 0.50,
        branching_probability: 0.48 + jitter(8) * 0.30,
        anastomosis_probability: 0.18 + maturity * 0.30 + jitter(9) * 0.14,
        spore_count: if matches!(receipt.previous_termination, PreviousTermination::FirstBoot) {
            1
        } else {
            1 + (seed[10] % 5)
        },
        growth_anisotropy: 0.15 + jitter(11) * 0.55,
        node_density: 0.35 + maturity * 0.35 + jitter(12) * 0.15,
        pulse_velocity: if resumed { 1.65 } else { 0.75 + jitter(13) * 0.55 },
        turbulence: if interrupted {
            0.55 + jitter(14) * 0.25
        } else {
            0.12 + jitter(14) * 0.30
        },
        color_temperature_k: temperature,
        solar_gold_fraction,
        leaf_green_fraction,
        mycelial_white_fraction,
        glow_radius: 0.55 + jitter(15) * 0.35,
        growth_velocity: if resumed { 1.75 } else { 0.80 + jitter(16) * 0.45 },
        camera_scale: 0.88 + jitter(17) * 0.20,
        repair_intensity,
        mesh_opacity: if receipt.mesh_enabled {
            (0.30 + (receipt.mesh_peers_last_seen.min(12) as f32 / 12.0) * 0.70)
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
        maturity,
    }
}

fn compose_stages(receipt: &BootStateReceipt, parameters: &MorphologyParameters) -> Vec<BootStage> {
    let mut stages = vec![stage(BootStageKind::Blackout, 180, 0.15)];

    match receipt.previous_termination {
        PreviousTermination::FirstBoot => {
            stages.push(stage(BootStageKind::DormantCore, 650, 0.45));
            stages.push(stage(BootStageKind::Germinate, 1_200, 0.85));
        }
        PreviousTermination::Suspend => {
            stages.push(stage(BootStageKind::Relight, 450, 0.70));
        }
        PreviousTermination::Hibernate => {
            stages.push(stage(BootStageKind::Relight, 700, 0.75));
            stages.push(stage(BootStageKind::Anastomose, 550, 0.45));
        }
        _ => {
            stages.push(stage(BootStageKind::DormantCore, 350, 0.35));
            stages.push(stage(BootStageKind::Grow, 900, 0.70));
        }
    }

    if matches!(receipt.generation_transition, GenerationTransition::RolledBack { .. }) {
        stages.push(stage(BootStageKind::RetractFailedGrowth, 650, 0.75));
    } else if matches!(receipt.generation_transition, GenerationTransition::Updated { .. }) {
        stages.push(stage(BootStageKind::GrowthRing, 750, 0.80));
    }

    if parameters.repair_intensity > 0.0 {
        stages.push(stage(BootStageKind::Repair, 900, parameters.repair_intensity));
    }
    if receipt.hardware_changed() {
        stages.push(stage(BootStageKind::HardwareBud, 600, 0.65));
    }
    if !matches!(receipt.previous_termination, PreviousTermination::Suspend) {
        stages.push(stage(BootStageKind::Anastomose, 650, 0.55));
    }
    if receipt.mesh_enabled {
        stages.push(stage(BootStageKind::MeshLink, 550, parameters.mesh_opacity.max(0.15)));
    }

    stages.push(stage(BootStageKind::Settle, 450, 0.45));
    stages.push(stage(BootStageKind::Handoff, 500, 0.35));
    stages
}

fn stage(kind: BootStageKind, duration_ms: u32, intensity: f32) -> BootStage {
    BootStage {
        kind,
        duration_ms,
        intensity: intensity.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_receipt() -> BootStateReceipt {
        BootStateReceipt {
            schema_version: BOOT_ECOLOGY_SCHEMA_VERSION,
            machine_visual_seed: [0x42; 32],
            boot_counter: 12,
            previous_termination: PreviousTermination::CleanPoweroff,
            previous_uptime_secs: 86_400,
            generation_transition: GenerationTransition::Same,
            generation_health: GenerationHealth::KnownGood,
            storage_state: StorageState::Clean,
            oom_events: 0,
            thermal_events: 0,
            hardware_fingerprint: Some("hw-b".into()),
            previous_hardware_fingerprint: Some("hw-b".into()),
            mesh_enabled: true,
            mesh_peers_last_seen: 4,
        }
    }

    #[test]
    fn composition_is_deterministic() {
        let receipt = clean_receipt();
        let lineage = MorphologyLineage::default();
        assert_eq!(
            BootEcologyComposer::compose(&receipt, &lineage),
            BootEcologyComposer::compose(&receipt, &lineage)
        );
    }

    #[test]
    fn boot_counter_changes_genome_without_changing_machine_identity() {
        let a_receipt = clean_receipt();
        let mut b_receipt = a_receipt.clone();
        b_receipt.boot_counter += 1;
        let lineage = MorphologyLineage::default();
        let a = BootEcologyComposer::compose(&a_receipt, &lineage);
        let b = BootEcologyComposer::compose(&b_receipt, &lineage);
        assert_ne!(a.seed, b.seed);
        assert_eq!(a_receipt.machine_visual_seed, b_receipt.machine_visual_seed);
    }

    #[test]
    fn first_boot_is_germination() {
        let receipt = BootStateReceipt::first_boot([7; 32]);
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        assert_eq!(genome.family, MorphologyFamily::CentralSpore);
        assert_eq!(genome.cue, BootCue::FirstBoot);
        assert!(genome.stages.iter().any(|s| s.kind == BootStageKind::Germinate));
        assert_eq!(genome.parameters.spore_count, 1);
    }

    #[test]
    fn suspend_relights_instead_of_regrowing() {
        let mut receipt = clean_receipt();
        receipt.previous_termination = PreviousTermination::Suspend;
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        assert_eq!(genome.family, MorphologyFamily::MinimalRelight);
        assert!(genome.stages.iter().any(|s| s.kind == BootStageKind::Relight));
        assert!(!genome.stages.iter().any(|s| s.kind == BootStageKind::Germinate));
    }

    #[test]
    fn interrupted_boot_gets_repair_behavior() {
        let mut receipt = clean_receipt();
        receipt.previous_termination = PreviousTermination::PowerLoss;
        receipt.storage_state = StorageState::JournalReplayed;
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        assert_eq!(genome.family, MorphologyFamily::KintsugiRepair);
        assert_eq!(genome.cue, BootCue::RecoveringState);
        assert!(genome.parameters.repair_intensity >= 0.65);
        assert!(genome.stages.iter().any(|s| s.kind == BootStageKind::Repair));
    }

    #[test]
    fn rollback_retracts_candidate_growth() {
        let mut receipt = clean_receipt();
        receipt.generation_transition = GenerationTransition::RolledBack {
            attempted: "generation-330".into(),
            restored: "generation-329".into(),
        };
        let genome = BootEcologyComposer::compose(&receipt, &MorphologyLineage::default());
        assert_eq!(genome.cue, BootCue::RestoringKnownGood);
        assert!(genome
            .stages
            .iter()
            .any(|s| s.kind == BootStageKind::RetractFailedGrowth));
        assert!(genome.parameters.solar_gold_fraction > 0.30);
    }

    #[test]
    fn long_uptime_matures_visuals() {
        let mut short = clean_receipt();
        short.previous_uptime_secs = 60;
        let mut long = short.clone();
        long.previous_uptime_secs = 14 * 86_400;
        let lineage = MorphologyLineage::default();
        assert!(
            BootEcologyComposer::compose(&long, &lineage).parameters.maturity
                > BootEcologyComposer::compose(&short, &lineage).parameters.maturity
        );
    }

    #[test]
    fn palette_fractions_are_normalized() {
        let genome = BootEcologyComposer::compose(&clean_receipt(), &MorphologyLineage::default());
        let p = &genome.parameters;
        let sum = p.solar_gold_fraction + p.leaf_green_fraction + p.mycelial_white_fraction;
        assert!((sum - 1.0).abs() < 0.0001, "palette sum was {sum}");
        assert!(p.mycelial_white_fraction >= 0.12);
    }

    #[test]
    fn renderer_policy_is_fail_open_and_bounded() {
        let genome = BootEcologyComposer::compose(&clean_receipt(), &MorphologyLineage::default());
        assert!(genome.render_policy.fail_open);
        assert!(genome.render_policy.progress_source_optional);
        assert!(genome.render_policy.release_before_display_manager);
        assert!(genome.render_policy.hard_deadline_ms <= 9_000);
        assert!(genome.visual_budget_ms() <= genome.render_policy.hard_deadline_ms);
    }

    #[test]
    fn failed_candidate_does_not_advance_last_known_good() {
        let genome = BootEcologyComposer::compose(&clean_receipt(), &MorphologyLineage::default());
        let mut lineage = MorphologyLineage {
            last_known_good_generation: Some("generation-10".into()),
            ..MorphologyLineage::default()
        };
        lineage.record_outcome(&genome, &BootOutcome::FailedBeforeHealthGate);
        assert_eq!(lineage.last_known_good_generation.as_deref(), Some("generation-10"));
        assert_eq!(lineage.successful_boots, 0);
    }

    #[test]
    fn health_gate_advances_lineage_only_on_success() {
        let genome = BootEcologyComposer::compose(&clean_receipt(), &MorphologyLineage::default());
        let mut lineage = MorphologyLineage::default();
        lineage.record_outcome(
            &genome,
            &BootOutcome::ReachedGraphicalTarget {
                generation: "generation-11".into(),
            },
        );
        assert_eq!(lineage.successful_boots, 1);
        assert_eq!(lineage.last_known_good_generation.as_deref(), Some("generation-11"));
        assert_eq!(lineage.last_genome_seed, Some(genome.seed));
    }
}
