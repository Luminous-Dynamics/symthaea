// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research bridge: integrates PNI coupling (Direction F) and
//! pharmacogenomics (Direction B) into the neuromodulator bath.
//!
//! This module provides a unified interface for research directions
//! that modulate the neuromodulator bath from external signals.
//!
//! **Direction F — Psychoneuroimmunology**: threat/stress/sleep signals
//! drive cytokine cascades that modulate serotonin, dopamine, noradrenaline,
//! adenosine, and oxytocin (Dantzer 2009; Slavich & Irwin 2014).
//!
//! **Direction B — Pharmacogenomics**: genetic metabolizer phenotypes
//! (CYP2D6, CYP2C19, etc.) and receptor variants (5-HT2A, COMT, OPRM1)
//! shape drug response curves and consciousness sensitivity profiles
//! (Kirchheiner 2004; Hicks 2015; Carhart-Harris 2018).

use serde::{Deserialize, Serialize};

use crate::pharmacogenomics::{ConsciousnessSensitivity, DrugClass, PgxProfile, PharmaResponse};
use crate::pni_coupling::PniCoupling;

// ── Named constants ──────────────────────────────────────────────────────

/// PGx consciousness cost coefficient: scales total neuromod shift into Phi cost.
/// Conservative: even large pharmacological perturbations cap below PNI's 0.3 max.
/// Science: Carhart-Harris (2018) — drug-induced entropy ≠ consciousness abolition.
const PGX_CONSCIOUSNESS_COST_COEFF: f64 = 0.05;

/// Maximum combined consciousness cost from all research modulations.
/// Ensures PNI + PGx together cannot fully abolish consciousness.
const MAX_COMBINED_CONSCIOUSNESS_COST: f64 = 0.45;

// ── Types ────────────────────────────────────────────────────────────────

/// Combined neuromodulator deltas from all active research modulations.
///
/// These are additive adjustments to be applied to the `NeuromodulatorBath`
/// during the cognitive loop's neuromodulator update phase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinedDeltas {
    /// Serotonin change (PNI: inflammation depletes; PGx: drug-dependent).
    pub delta_serotonin: f32,
    /// Dopamine change (PNI: sickness reduces; PGx: drug-dependent).
    pub delta_dopamine: f32,
    /// Noradrenaline change (PNI: IL-1beta boosts; PGx: drug-dependent).
    pub delta_noradrenaline: f32,
    /// Acetylcholine change (currently PGx-only pathway).
    pub delta_acetylcholine: f32,
    /// Adenosine change (PNI: TNF-alpha fatigue signal).
    pub delta_adenosine: f32,
    /// Oxytocin change (PNI: IL-10 social repair signal).
    pub delta_oxytocin: f32,
    /// GABA change (PGx: benzodiazepines, antipsychotics).
    pub delta_gaba: f32,
    /// Whether PNI contributed to these deltas.
    pub source_pni: bool,
    /// Whether PGx (drug response) contributed to these deltas.
    pub source_pgx: bool,
}

/// Unified research modulation state.
///
/// Owns both the PNI coupling engine (Direction F) and an optional
/// pharmacogenomic profile (Direction B), providing a single interface
/// for the cognitive loop to query combined neuromodulator effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchModulation {
    /// Psychoneuroimmunology state (Direction F).
    pub pni: PniCoupling,
    /// Pharmacogenomic profile (Direction B) — `None` if no genomic data.
    pub pgx_profile: Option<PgxProfile>,
    /// Whether PNI modulation is active.
    pub pni_active: bool,
    /// Whether PGx modulation is active.
    pub pgx_active: bool,
    /// Accumulated consciousness cost from all research modulations.
    pub total_consciousness_cost: f64,
    /// Most recent PGx drug response (if any drug has been applied).
    last_pharma_response: Option<PharmaResponse>,
}

impl Default for ResearchModulation {
    fn default() -> Self {
        Self::new()
    }
}

impl ResearchModulation {
    /// Create a new research modulation state with PNI active and no PGx profile.
    pub fn new() -> Self {
        Self {
            pni: PniCoupling::new(),
            pgx_profile: None,
            pni_active: true,
            pgx_active: false,
            total_consciousness_cost: 0.0,
            last_pharma_response: None,
        }
    }

    /// Create with a pharmacogenomic profile (enables both PNI and PGx).
    pub fn with_pgx(pgx: PgxProfile) -> Self {
        Self {
            pni: PniCoupling::new(),
            pgx_profile: Some(pgx),
            pni_active: true,
            pgx_active: true,
            total_consciousness_cost: 0.0,
            last_pharma_response: None,
        }
    }

    /// Update PNI state from threat/stress/sleep signals.
    ///
    /// Delegates to the three PNI update pathways in sequence:
    /// 1. Threat detection → pro-inflammatory cytokine release
    /// 2. Stress hormones → immune suppression
    /// 3. Sleep state → anti-inflammatory recovery
    ///
    /// Science: Slavich & Irwin (2014) — social threat → inflammatory cascade.
    pub fn update_pni(
        &mut self,
        threat_level: f32,
        cortisol: f32,
        allostatic_load: f32,
        is_sleeping: bool,
        sleep_quality: f32,
    ) {
        if !self.pni_active {
            return;
        }
        // Threat duration is approximated as 1 cycle per call; the PNI engine
        // accumulates IL-6 proportionally via sqrt(duration).
        self.pni.update_from_threat(threat_level, 1);
        self.pni.update_from_stress(cortisol, allostatic_load);
        self.pni.update_from_sleep(is_sleeping, sleep_quality);
    }

    /// Apply a pharmacological intervention, modulated by PGx if available.
    ///
    /// If a `PgxProfile` is present, the response is personalized via
    /// metabolizer phenotype and genetic variant effects. Otherwise,
    /// a default European genotype is used as the reference baseline.
    ///
    /// Science: Hicks et al. (2015) — CPIC clinical guidelines for dosing.
    pub fn apply_drug(&mut self, drug: DrugClass, dose: f32) -> PharmaResponse {
        let response = match &self.pgx_profile {
            Some(profile) => profile.predict_response(drug, dose),
            None => PgxProfile::default_european().predict_response(drug, dose),
        };
        self.last_pharma_response = Some(response.clone());
        response
    }

    /// Get combined neuromod deltas from all active research modulations.
    ///
    /// Sums PNI cytokine-driven deltas with the most recent PGx drug response.
    /// Each source is independently gated by its `_active` flag.
    pub fn combined_deltas(&self) -> CombinedDeltas {
        let mut deltas = CombinedDeltas::default();

        // Direction F: PNI cytokine → neuromodulator mapping
        if self.pni_active {
            let pni_deltas = self.pni.compute_neuromod_deltas();
            deltas.delta_serotonin += pni_deltas.delta_serotonin;
            deltas.delta_dopamine += pni_deltas.delta_dopamine;
            deltas.delta_noradrenaline += pni_deltas.delta_noradrenaline;
            deltas.delta_adenosine += pni_deltas.delta_adenosine;
            deltas.delta_oxytocin += pni_deltas.delta_oxytocin;
            deltas.source_pni = true;
        }

        // Direction B: PGx drug response → neuromodulator mapping
        if self.pgx_active
            && let Some(ref resp) = self.last_pharma_response
        {
            deltas.delta_serotonin += resp.neuromod_effects.delta_serotonin;
            deltas.delta_dopamine += resp.neuromod_effects.delta_dopamine;
            deltas.delta_noradrenaline += resp.neuromod_effects.delta_noradrenaline;
            deltas.delta_gaba += resp.neuromod_effects.delta_gaba;
            // Glutamate mapped to acetylcholine pathway (both excitatory,
            // and ACh is the only non-GABA channel not covered by PNI).
            deltas.delta_acetylcholine += resp.neuromod_effects.delta_glutamate * 0.3;
            deltas.source_pgx = true;
        }

        deltas
    }

    /// Get total consciousness cost from research modulations.
    ///
    /// Combines PNI inflammation cost with PGx drug perturbation cost,
    /// capped at `MAX_COMBINED_CONSCIOUSNESS_COST` (0.45).
    pub fn consciousness_cost(&self) -> f64 {
        let mut cost = 0.0;

        if self.pni_active {
            cost += self.pni.compute_consciousness_cost();
        }

        if self.pgx_active
            && let Some(ref resp) = self.last_pharma_response
        {
            cost += resp.consciousness_impact.abs() * PGX_CONSCIOUSNESS_COST_COEFF;
        }

        cost.min(MAX_COMBINED_CONSCIOUSNESS_COST)
    }

    /// Get consciousness sensitivity profile (if PGx active and profile present).
    ///
    /// Returns genetic-variant-derived modifiers for Phi, receptor density,
    /// clearance rates, and neuroplasticity.
    pub fn consciousness_sensitivity(&self) -> Option<ConsciousnessSensitivity> {
        if self.pgx_active {
            self.pgx_profile
                .as_ref()
                .map(|p| p.consciousness_sensitivity())
        } else {
            None
        }
    }

    /// Tick all active research modulations forward by one cognitive cycle.
    ///
    /// Advances PNI cytokine dynamics (exponential decay, sickness behavior
    /// tracking) and recomputes the aggregated consciousness cost.
    pub fn tick(&mut self, dt: f32) {
        if self.pni_active {
            self.pni.tick(dt);
        }
        self.total_consciousness_cost = self.consciousness_cost();
    }

    /// Access the most recent pharmacological response, if any.
    pub fn last_drug_response(&self) -> Option<&PharmaResponse> {
        self.last_pharma_response.as_ref()
    }

    /// Whether the system is currently exhibiting sickness behavior (PNI).
    pub fn is_sick(&self) -> bool {
        self.pni_active && self.pni.is_sick()
    }

    /// Current PNI immune competence level (1.0 = healthy, 0.0 = suppressed).
    pub fn immune_competence(&self) -> f32 {
        if self.pni_active {
            self.pni.immune_competence()
        } else {
            1.0
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pharmacogenomics::{
        AncestryGroup, CypEnzyme, GeneticVariant, MetabolizerPhenotype, Zygosity,
    };
    use std::collections::HashMap;

    #[test]
    fn default_research_modulation_has_zero_deltas() {
        let rm = ResearchModulation::new();
        let deltas = rm.combined_deltas();

        assert!(
            deltas.delta_serotonin.abs() < 1e-6,
            "default serotonin delta should be ~0, got {}",
            deltas.delta_serotonin
        );
        assert!(
            deltas.delta_dopamine.abs() < 1e-6,
            "default dopamine delta should be ~0, got {}",
            deltas.delta_dopamine
        );
        assert!(
            deltas.delta_noradrenaline.abs() < 1e-6,
            "default NE delta should be ~0, got {}",
            deltas.delta_noradrenaline
        );
        assert!(deltas.source_pni, "PNI should be active by default");
        assert!(!deltas.source_pgx, "PGx should be inactive without drug");
        assert!(
            rm.consciousness_cost() < 1e-6,
            "default consciousness cost should be ~0"
        );
    }

    #[test]
    fn pni_threat_produces_nonzero_serotonin_delta() {
        let mut rm = ResearchModulation::new();
        rm.update_pni(0.8, 0.0, 0.0, false, 0.0);

        let deltas = rm.combined_deltas();
        assert!(
            deltas.delta_serotonin < -1e-4,
            "threat should deplete serotonin via IL-6→IDO pathway, got {}",
            deltas.delta_serotonin
        );
        assert!(deltas.source_pni);
    }

    #[test]
    fn pgx_drug_response_differs_from_no_pgx() {
        // Poor CYP2D6 metabolizer should have different SSRI response
        let mut metabolizer_map = HashMap::new();
        metabolizer_map.insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::Poor);
        metabolizer_map.insert(CypEnzyme::CYP2C19, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP3A4, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP2C9, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP1A2, MetabolizerPhenotype::Normal);

        let pgx_poor = PgxProfile {
            metabolizer_map,
            genetic_variants: vec![
                (GeneticVariant::HTR2A_rs6311, Zygosity::Homozygous),
                (GeneticVariant::COMT_Val158Met, Zygosity::WildType),
                (GeneticVariant::OPRM1_A118G, Zygosity::WildType),
                (GeneticVariant::BDNF_Val66Met, Zygosity::WildType),
                (GeneticVariant::SLC6A4_5HTTLPR, Zygosity::WildType),
            ],
            ancestry: AncestryGroup::European,
        };

        let mut rm_pgx = ResearchModulation::with_pgx(pgx_poor);
        let mut rm_default = ResearchModulation::new();
        rm_default.pgx_active = true; // Enable PGx with default (no profile)

        let resp_pgx = rm_pgx.apply_drug(DrugClass::SSRI, 1.0);
        let resp_default = rm_default.apply_drug(DrugClass::SSRI, 1.0);

        // Poor metabolizer should have higher peak concentration (1.8× vs 1.0×)
        assert!(
            resp_pgx.peak_concentration > resp_default.peak_concentration,
            "poor metabolizer should accumulate more drug: pgx={} vs default={}",
            resp_pgx.peak_concentration,
            resp_default.peak_concentration
        );
        assert!(
            resp_pgx.accumulation_warning,
            "poor CYP2D6 + SSRI should warn"
        );
    }

    #[test]
    fn combined_deltas_sum_pni_and_pgx() {
        let pgx = PgxProfile::default_european();
        let mut rm = ResearchModulation::with_pgx(pgx);

        // Induce PNI inflammation
        rm.update_pni(0.7, 0.0, 0.0, false, 0.0);
        let pni_only = rm.combined_deltas();

        // Now also apply a drug
        rm.apply_drug(DrugClass::SSRI, 1.0);
        let combined = rm.combined_deltas();

        // Combined serotonin should differ from PNI-only (SSRI adds positive 5-HT)
        assert!(
            (combined.delta_serotonin - pni_only.delta_serotonin).abs() > 0.01,
            "SSRI should add serotonin delta on top of PNI: combined={}, pni_only={}",
            combined.delta_serotonin,
            pni_only.delta_serotonin
        );
        assert!(combined.source_pni, "PNI should be flagged");
        assert!(combined.source_pgx, "PGx should be flagged");
    }

    #[test]
    fn consciousness_cost_aggregates_both_sources() {
        let pgx = PgxProfile::default_european();
        let mut rm = ResearchModulation::with_pgx(pgx);

        // Induce inflammation (PNI cost)
        rm.update_pni(1.0, 0.5, 0.3, false, 0.0);
        let pni_cost = rm.consciousness_cost();
        assert!(
            pni_cost > 0.0,
            "PNI inflammation should produce consciousness cost"
        );

        // Apply drug (PGx cost)
        rm.apply_drug(DrugClass::Psychedelic, 1.0);
        let combined_cost = rm.consciousness_cost();
        assert!(
            combined_cost >= pni_cost,
            "adding drug should not decrease cost: combined={}, pni={}",
            combined_cost,
            pni_cost
        );
        assert!(
            combined_cost <= MAX_COMBINED_CONSCIOUSNESS_COST,
            "cost should be capped at {}, got {}",
            MAX_COMBINED_CONSCIOUSNESS_COST,
            combined_cost
        );
    }

    #[test]
    fn tick_advances_pni_state() {
        let mut rm = ResearchModulation::new();

        // Inflame, then tick several times — inflammation should decay
        rm.update_pni(0.9, 0.0, 0.0, false, 0.0);
        let inflamed_il6 = rm.pni.cytokines().il6;

        for _ in 0..10 {
            rm.tick(0.05);
        }

        let post_tick_il6 = rm.pni.cytokines().il6;
        assert!(
            post_tick_il6 < inflamed_il6,
            "IL-6 should decay after ticks: before={}, after={}",
            inflamed_il6,
            post_tick_il6
        );
        // total_consciousness_cost should have been updated by tick
        assert!(
            (rm.total_consciousness_cost - rm.consciousness_cost()).abs() < 1e-9,
            "tick should keep total_consciousness_cost in sync"
        );
    }

    #[test]
    fn consciousness_sensitivity_requires_pgx_active() {
        let rm_no_pgx = ResearchModulation::new();
        assert!(
            rm_no_pgx.consciousness_sensitivity().is_none(),
            "no PGx profile → no sensitivity"
        );

        let rm_with_pgx = ResearchModulation::with_pgx(PgxProfile::default_european());
        let cs = rm_with_pgx
            .consciousness_sensitivity()
            .expect("PGx profile present → sensitivity should exist");
        assert!(
            (cs.phi_baseline_modifier - 1.0).abs() < 0.2,
            "default European profile should have near-neutral Phi modifier"
        );
    }

    #[test]
    fn disabled_pni_produces_no_deltas() {
        let mut rm = ResearchModulation::new();
        rm.pni_active = false;

        rm.update_pni(1.0, 1.0, 1.0, false, 0.0);
        let deltas = rm.combined_deltas();

        assert!(
            deltas.delta_serotonin.abs() < 1e-6,
            "disabled PNI should produce zero serotonin delta"
        );
        assert!(
            !deltas.source_pni,
            "source_pni should be false when disabled"
        );
    }
}
