// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pharmacogenomic consciousness: mapping genetic metabolizer phenotypes
//! to neuromodulator response curves and consciousness sensitivity profiles.
//!
//! Models how genetic variation (CYP2D6, CYP2C19, CYP3A4, 5-HTR2A, COMT,
//! OPRM1) modulates drug pharmacokinetics and consciousness parameters.
//!
//! References:
//! - Kirchheiner, J. et al. (2004). CYP2D6 pharmacogenetics.
//! - Hicks, J. K. et al. (2015). CPIC guideline for CYP2D6/CYP2C19.
//! - Carhart-Harris, R. L. (2018). Entropic brain hypothesis.
//! - Vollenweider, F. X. (2001). 5-HT2A in consciousness.
//! - Lotta, T. et al. (1995). COMT and dopamine metabolism.
//! - Lotsch, J. et al. (2009). OPRM1 A118G and opioid sensitivity.
//! - Egan, M. F. et al. (2003). BDNF Val66Met and neuroplasticity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Cytochrome P450 enzymes governing drug metabolism.
///
/// Science: Zanger & Schwab (2013) — CYP enzymes metabolize ~75% of all drugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CypEnzyme {
    /// Primary metabolizer for codeine, SSRIs (fluoxetine, paroxetine), TCAs.
    CYP2D6,
    /// Clopidogrel, PPIs, some SSRIs (citalopram, escitalopram).
    CYP2C19,
    /// Benzodiazepines, ketamine, many psychedelics (psilocybin).
    CYP3A4,
    /// Warfarin, NSAIDs, some cannabinoids.
    CYP2C9,
    /// Caffeine, clozapine, melatonin.
    CYP1A2,
}

/// Metabolizer phenotype derived from diplotype star-allele calling.
///
/// Science: Hicks et al. (2015) — CPIC clinical guidelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetabolizerPhenotype {
    /// Gene duplications / increased function — rapid drug clearance.
    UltraRapid,
    /// Two normal-function alleles — reference pharmacokinetics.
    Normal,
    /// One reduced-function allele — slower metabolism.
    Intermediate,
    /// Two loss-of-function alleles — markedly impaired clearance.
    Poor,
}

impl MetabolizerPhenotype {
    /// Half-life multiplier relative to Normal metabolizer.
    ///
    /// Poor metabolizers clear drugs ~2.5× slower (Kirchheiner 2004).
    pub fn half_life_multiplier(&self) -> f32 {
        match self {
            Self::UltraRapid => 0.5,
            Self::Normal => 1.0,
            Self::Intermediate => 1.5,
            Self::Poor => 2.5,
        }
    }

    /// Peak plasma concentration multiplier relative to Normal.
    ///
    /// Poor metabolizers accumulate higher peak levels (Hicks 2015).
    pub fn peak_concentration_multiplier(&self) -> f32 {
        match self {
            Self::UltraRapid => 0.6,
            Self::Normal => 1.0,
            Self::Intermediate => 1.3,
            Self::Poor => 1.8,
        }
    }
}

/// Genetic variants affecting neuromodulator pathways directly.
// Variant names follow domain-standard genetic nomenclature (rs6311, Val158Met, etc.)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneticVariant {
    /// 5-HT2A promoter polymorphism (rs6311). T allele → higher receptor density.
    /// Science: Turecki et al. (1999); Vollenweider (2001).
    HTR2A_rs6311,
    /// COMT Val158Met (rs4680). Val = high activity → rapid DA clearance.
    /// Science: Lotta et al. (1995); Egan et al. (2001).
    COMT_Val158Met,
    /// OPRM1 A118G (rs1799971). G allele → reduced mu-opioid binding.
    /// Science: Lotsch et al. (2009).
    OPRM1_A118G,
    /// BDNF Val66Met (rs6265). Met allele → reduced activity-dependent secretion.
    /// Science: Egan et al. (2003).
    BDNF_Val66Met,
    /// SLC6A4 5-HTTLPR. Short allele → lower serotonin transporter expression.
    /// Science: Caspi et al. (2003).
    SLC6A4_5HTTLPR,
}

/// Zygosity of a genetic variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Zygosity {
    /// Two copies of the variant allele.
    Homozygous,
    /// One variant, one reference allele.
    Heterozygous,
    /// Two reference (wild-type) alleles.
    WildType,
}

impl Zygosity {
    /// Effect size multiplier: Homozygous = 1.0, Heterozygous = 0.5, WildType = 0.0.
    fn effect_size(&self) -> f32 {
        match self {
            Self::Homozygous => 1.0,
            Self::Heterozygous => 0.5,
            Self::WildType => 0.0,
        }
    }
}

/// Ancestry group for population-specific allele frequency context.
///
/// Science: PharmGKB population-stratified allele frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AncestryGroup {
    European,
    African,
    EastAsian,
    SouthAsian,
    NativeAmerican,
    Mixed,
}

impl AncestryGroup {
    /// CYP2D6 poor-metabolizer prevalence in this ancestry group.
    ///
    /// Science: Bradford (2002) — CYP2D6 PM prevalence by ethnicity.
    pub fn cyp2d6_poor_prevalence(&self) -> f32 {
        match self {
            Self::European => 0.07,   // ~7%
            Self::African => 0.03,    // ~3%
            Self::EastAsian => 0.01,  // ~1%
            Self::SouthAsian => 0.04, // ~4%
            Self::NativeAmerican => 0.02,
            Self::Mixed => 0.05,
        }
    }

    /// CYP2D6 ultra-rapid prevalence.
    pub fn cyp2d6_ultra_rapid_prevalence(&self) -> f32 {
        match self {
            Self::European => 0.05,
            Self::African => 0.29, // ~29% — notably high
            Self::EastAsian => 0.01,
            Self::SouthAsian => 0.08,
            Self::NativeAmerican => 0.03,
            Self::Mixed => 0.07,
        }
    }
}

/// Drug class for pharmacokinetic modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrugClass {
    /// Selective serotonin reuptake inhibitors (fluoxetine, sertraline, etc.)
    SSRI,
    /// Serotonin-norepinephrine reuptake inhibitors (venlafaxine, duloxetine).
    SNRI,
    /// GABAergic anxiolytics/sedatives (diazepam, alprazolam).
    Benzodiazepine,
    /// 5-HT2A agonists (psilocybin, LSD, DMT).
    Psychedelic,
    /// Amphetamines, methylphenidate.
    Stimulant,
    /// Mu-opioid agonists (morphine, codeine, oxycodone).
    Opioid,
    /// D2 antagonists (haloperidol, olanzapine, risperidone).
    Antipsychotic,
    /// NMDA antagonist / dissociative (ketamine, esketamine).
    Ketamine,
}

impl DrugClass {
    /// Primary CYP enzyme responsible for metabolism of this drug class.
    ///
    /// Simplified — real pharmacology often involves multiple CYPs.
    pub fn primary_cyp(&self) -> CypEnzyme {
        match self {
            Self::SSRI => CypEnzyme::CYP2D6,
            Self::SNRI => CypEnzyme::CYP2D6,
            Self::Benzodiazepine => CypEnzyme::CYP3A4,
            Self::Psychedelic => CypEnzyme::CYP3A4,
            Self::Stimulant => CypEnzyme::CYP2D6,
            Self::Opioid => CypEnzyme::CYP2D6,
            Self::Antipsychotic => CypEnzyme::CYP2D6,
            Self::Ketamine => CypEnzyme::CYP3A4,
        }
    }

    /// Typical half-life in hours for the drug class (reference/Normal metabolizer).
    fn base_half_life_hours(&self) -> f32 {
        match self {
            Self::SSRI => 24.0,
            Self::SNRI => 12.0,
            Self::Benzodiazepine => 40.0,
            Self::Psychedelic => 4.0,
            Self::Stimulant => 10.0,
            Self::Opioid => 4.0,
            Self::Antipsychotic => 20.0,
            Self::Ketamine => 2.5,
        }
    }

    /// Base neuromodulator effects at peak concentration (normalized dose = 1.0).
    fn base_neuromod_effects(&self) -> NeuromodEffects {
        match self {
            Self::SSRI => NeuromodEffects {
                delta_serotonin: 0.30,
                delta_dopamine: 0.05,
                delta_noradrenaline: 0.02,
                delta_gaba: 0.0,
                delta_glutamate: 0.0,
            },
            Self::SNRI => NeuromodEffects {
                delta_serotonin: 0.25,
                delta_dopamine: 0.05,
                delta_noradrenaline: 0.20,
                delta_gaba: 0.0,
                delta_glutamate: 0.0,
            },
            Self::Benzodiazepine => NeuromodEffects {
                delta_serotonin: 0.0,
                delta_dopamine: 0.0,
                delta_noradrenaline: -0.10,
                delta_gaba: 0.40,
                delta_glutamate: -0.15,
            },
            Self::Psychedelic => NeuromodEffects {
                delta_serotonin: 0.50,
                delta_dopamine: 0.15,
                delta_noradrenaline: 0.10,
                delta_gaba: -0.10,
                delta_glutamate: 0.20,
            },
            Self::Stimulant => NeuromodEffects {
                delta_serotonin: 0.05,
                delta_dopamine: 0.45,
                delta_noradrenaline: 0.30,
                delta_gaba: 0.0,
                delta_glutamate: 0.10,
            },
            Self::Opioid => NeuromodEffects {
                delta_serotonin: 0.05,
                delta_dopamine: 0.20,
                delta_noradrenaline: -0.15,
                delta_gaba: 0.15,
                delta_glutamate: -0.10,
            },
            Self::Antipsychotic => NeuromodEffects {
                delta_serotonin: 0.10,
                delta_dopamine: -0.35,
                delta_noradrenaline: -0.05,
                delta_gaba: 0.10,
                delta_glutamate: -0.05,
            },
            Self::Ketamine => NeuromodEffects {
                delta_serotonin: 0.10,
                delta_dopamine: 0.20,
                delta_noradrenaline: 0.15,
                delta_gaba: 0.05,
                delta_glutamate: -0.30,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Core structs
// ---------------------------------------------------------------------------

/// Per-transmitter deltas at peak drug concentration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodEffects {
    pub delta_serotonin: f32,
    pub delta_dopamine: f32,
    pub delta_noradrenaline: f32,
    pub delta_gaba: f32,
    pub delta_glutamate: f32,
}

/// Consciousness sensitivity derived from genetic variants.
///
/// Maps pharmacogenomic profile → consciousness parameter modifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSensitivity {
    /// Genetic predisposition to information integration (Phi modifier).
    /// 1.0 = neutral, >1.0 = enhanced integration capacity.
    pub phi_baseline_modifier: f64,
    /// 5-HT2A receptor density (0.0–1.0). Higher → stronger psychedelic/consciousness response.
    /// Science: Vollenweider (2001) — 5-HT2A mediates psychedelic consciousness expansion.
    pub serotonin_receptor_density: f32,
    /// Dopamine clearance rate (0.0–1.0). Higher → faster DA degradation.
    /// Science: Lotta et al. (1995) — COMT Val allele = 3–4× faster catechol-O-methylation.
    pub dopamine_clearance_rate: f32,
    /// Mu-opioid sensitivity (0.0–1.0). Lower → reduced pain/pleasure response.
    /// Science: Lotsch et al. (2009) — OPRM1 118G reduces binding affinity.
    pub opioid_sensitivity: f32,
    /// Neuroplasticity factor (0.0–1.0). Modulates learning rate.
    /// Science: Egan et al. (2003) — BDNF Met reduces activity-dependent secretion.
    pub neuroplasticity_factor: f32,
    /// Serotonin transporter efficiency (0.0–1.0).
    /// Science: Caspi et al. (2003) — 5-HTTLPR short → lower reuptake, higher synaptic 5-HT.
    pub serotonin_transporter_efficiency: f32,
}

impl Default for ConsciousnessSensitivity {
    fn default() -> Self {
        Self {
            phi_baseline_modifier: 1.0,
            serotonin_receptor_density: 0.5,
            dopamine_clearance_rate: 0.5,
            opioid_sensitivity: 0.5,
            neuroplasticity_factor: 0.7,
            serotonin_transporter_efficiency: 0.6,
        }
    }
}

/// Predicted drug response for a specific pharmacogenomic profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaResponse {
    /// Drug class being modeled.
    pub drug_class: DrugClass,
    /// Effective half-life in hours (modified by metabolizer phenotype).
    pub effective_half_life: f32,
    /// Peak plasma concentration (relative, modified by metabolizer phenotype).
    pub peak_concentration: f32,
    /// Per-transmitter deltas at peak.
    pub neuromod_effects: NeuromodEffects,
    /// Expected Phi change at peak concentration.
    pub consciousness_impact: f64,
    /// Clinical warning if dangerous accumulation predicted.
    pub accumulation_warning: bool,
}

/// Full pharmacogenomic profile for an individual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PgxProfile {
    /// CYP enzyme → metabolizer phenotype mapping.
    pub metabolizer_map: HashMap<CypEnzyme, MetabolizerPhenotype>,
    /// Non-CYP genetic variants with zygosity.
    pub genetic_variants: Vec<(GeneticVariant, Zygosity)>,
    /// Self-reported or genetically-inferred ancestry group.
    pub ancestry: AncestryGroup,
}

impl PgxProfile {
    /// Common European genotype: all Normal metabolizers, wild-type variants.
    pub fn default_european() -> Self {
        let mut metabolizer_map = HashMap::new();
        metabolizer_map.insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP2C19, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP3A4, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP2C9, MetabolizerPhenotype::Normal);
        metabolizer_map.insert(CypEnzyme::CYP1A2, MetabolizerPhenotype::Normal);

        Self {
            metabolizer_map,
            genetic_variants: vec![
                (GeneticVariant::HTR2A_rs6311, Zygosity::WildType),
                (GeneticVariant::COMT_Val158Met, Zygosity::Heterozygous),
                (GeneticVariant::OPRM1_A118G, Zygosity::WildType),
                (GeneticVariant::BDNF_Val66Met, Zygosity::WildType),
                (GeneticVariant::SLC6A4_5HTTLPR, Zygosity::Heterozygous),
            ],
            ancestry: AncestryGroup::European,
        }
    }

    /// Look up metabolizer phenotype for a CYP enzyme; defaults to Normal.
    pub fn metabolizer(&self, enzyme: CypEnzyme) -> MetabolizerPhenotype {
        self.metabolizer_map
            .get(&enzyme)
            .copied()
            .unwrap_or(MetabolizerPhenotype::Normal)
    }

    /// Look up zygosity for a genetic variant; defaults to WildType.
    fn variant_zygosity(&self, variant: GeneticVariant) -> Zygosity {
        self.genetic_variants
            .iter()
            .find(|(v, _)| *v == variant)
            .map(|(_, z)| *z)
            .unwrap_or(Zygosity::WildType)
    }

    /// Derive consciousness sensitivity profile from genetic variants.
    ///
    /// Each variant modulates a specific neuromodulator pathway, which in turn
    /// affects consciousness parameters (Phi, learning, mood stability).
    pub fn consciousness_sensitivity(&self) -> ConsciousnessSensitivity {
        let mut cs = ConsciousnessSensitivity::default();

        // HTR2A rs6311: T allele → higher 5-HT2A density → stronger psychedelic response.
        // Homozygous T/T ≈ 1.4× receptor density (Turecki 1999).
        let htr2a_effect = self
            .variant_zygosity(GeneticVariant::HTR2A_rs6311)
            .effect_size();
        cs.serotonin_receptor_density = 0.5 + 0.4 * htr2a_effect;

        // COMT Val158Met: Val = high activity → fast DA clearance.
        // Val/Val (WildType in our encoding) = 0.8 clearance rate.
        // Met/Met (Homozygous) = 0.3 clearance rate.
        let comt_effect = self
            .variant_zygosity(GeneticVariant::COMT_Val158Met)
            .effect_size();
        cs.dopamine_clearance_rate = 0.8 - 0.5 * comt_effect;

        // OPRM1 A118G: G allele → reduced mu-opioid binding.
        let oprm1_effect = self
            .variant_zygosity(GeneticVariant::OPRM1_A118G)
            .effect_size();
        cs.opioid_sensitivity = 0.7 - 0.3 * oprm1_effect;

        // BDNF Val66Met: Met → reduced neuroplasticity.
        let bdnf_effect = self
            .variant_zygosity(GeneticVariant::BDNF_Val66Met)
            .effect_size();
        cs.neuroplasticity_factor = 0.8 - 0.3 * bdnf_effect;

        // 5-HTTLPR: short allele → lower transporter efficiency → higher synaptic 5-HT.
        let sht_effect = self
            .variant_zygosity(GeneticVariant::SLC6A4_5HTTLPR)
            .effect_size();
        cs.serotonin_transporter_efficiency = 0.7 - 0.3 * sht_effect;

        // Phi baseline modifier: composite of integration-relevant variants.
        // Higher 5-HT2A density and lower DA clearance both correlate with richer
        // phenomenal integration (Carhart-Harris 2018 entropic brain).
        cs.phi_baseline_modifier = 1.0
            + 0.1 * (cs.serotonin_receptor_density as f64 - 0.5)
            + 0.05 * (0.5 - cs.dopamine_clearance_rate as f64)
            + 0.05 * (cs.neuroplasticity_factor as f64 - 0.5);

        cs
    }

    /// Predict drug response for a given drug class and dose.
    ///
    /// Applies metabolizer-phenotype PK modifiers, then scales neuromod deltas
    /// by genetic sensitivity, and estimates consciousness impact.
    pub fn predict_response(&self, drug: DrugClass, base_dose: f32) -> PharmaResponse {
        let cyp = drug.primary_cyp();
        let phenotype = self.metabolizer(cyp);

        let effective_half_life = drug.base_half_life_hours() * phenotype.half_life_multiplier();
        let peak_concentration = base_dose * phenotype.peak_concentration_multiplier();

        let cs = self.consciousness_sensitivity();
        let base_effects = drug.base_neuromod_effects();

        // Scale effects by dose and genetic sensitivity.
        let neuromod_effects = NeuromodEffects {
            delta_serotonin: base_effects.delta_serotonin
                * base_dose
                * (1.0 + 0.3 * (cs.serotonin_receptor_density - 0.5)),
            delta_dopamine: base_effects.delta_dopamine
                * base_dose
                * (1.0 - 0.2 * (cs.dopamine_clearance_rate - 0.5)),
            delta_noradrenaline: base_effects.delta_noradrenaline * base_dose,
            delta_gaba: base_effects.delta_gaba * base_dose,
            delta_glutamate: base_effects.delta_glutamate * base_dose,
        };

        // Consciousness impact: sum of absolute neuromod deltas × phi modifier.
        let total_neuromod_shift = neuromod_effects.delta_serotonin.abs()
            + neuromod_effects.delta_dopamine.abs()
            + neuromod_effects.delta_noradrenaline.abs()
            + neuromod_effects.delta_gaba.abs()
            + neuromod_effects.delta_glutamate.abs();

        let consciousness_impact = total_neuromod_shift as f64 * 0.1 * cs.phi_baseline_modifier;

        // Accumulation warning for poor metabolizers on drugs with narrow therapeutic index.
        let accumulation_warning = phenotype == MetabolizerPhenotype::Poor
            && matches!(
                drug,
                DrugClass::SSRI | DrugClass::Opioid | DrugClass::Antipsychotic
            );

        PharmaResponse {
            drug_class: drug,
            effective_half_life,
            peak_concentration,
            neuromod_effects,
            consciousness_impact,
            accumulation_warning,
        }
    }

    /// Detailed SSRI response model.
    ///
    /// CYP2D6 metabolizer → half-life; 5-HTTLPR → transporter efficiency →
    /// magnitude of serotonin boost. Poor CYP2D6 → dangerous accumulation risk.
    pub fn ssri_response(&self, dose: f32) -> PharmaResponse {
        let phenotype = self.metabolizer(CypEnzyme::CYP2D6);
        let cs = self.consciousness_sensitivity();

        let effective_half_life = 24.0 * phenotype.half_life_multiplier();
        let peak_concentration = dose * phenotype.peak_concentration_multiplier();

        // 5-HTTLPR modulates SSRI efficacy: low transporter efficiency means
        // the transporter is already partially blocked → smaller marginal effect.
        let sert_modifier = 0.5 + 0.5 * cs.serotonin_transporter_efficiency;

        let delta_5ht =
            0.30 * dose * sert_modifier * (1.0 + 0.3 * (cs.serotonin_receptor_density - 0.5));

        let neuromod_effects = NeuromodEffects {
            delta_serotonin: delta_5ht,
            delta_dopamine: 0.05 * dose,
            delta_noradrenaline: 0.02 * dose,
            delta_gaba: 0.0,
            delta_glutamate: 0.0,
        };

        // Moderate serotonin boost → mild Phi increase (mood stabilization).
        // Excessive → serotonin syndrome risk (very high 5-HT → consciousness disruption).
        let consciousness_impact = if delta_5ht > 0.6 {
            -0.05 * cs.phi_baseline_modifier // serotonin syndrome territory
        } else {
            delta_5ht as f64 * 0.08 * cs.phi_baseline_modifier
        };

        PharmaResponse {
            drug_class: DrugClass::SSRI,
            effective_half_life,
            peak_concentration,
            neuromod_effects,
            consciousness_impact,
            accumulation_warning: phenotype == MetabolizerPhenotype::Poor,
        }
    }

    /// Psychedelic response model (5-HT2A agonism).
    ///
    /// 5-HT2A density → consciousness expansion (Carhart-Harris 2018).
    /// COMT → duration (dopamine clearance affects trip length).
    /// High 5-HT2A + slow COMT = intense, long-duration experience.
    pub fn psychedelic_response(&self, dose: f32) -> PharmaResponse {
        let phenotype = self.metabolizer(CypEnzyme::CYP3A4);
        let cs = self.consciousness_sensitivity();

        let effective_half_life = 4.0
            * phenotype.half_life_multiplier()
            * (1.0 + 0.3 * (1.0 - cs.dopamine_clearance_rate)); // slower COMT → longer

        let peak_concentration = dose * phenotype.peak_concentration_multiplier();

        // 5-HT2A density is the primary driver of psychedelic intensity.
        let intensity = cs.serotonin_receptor_density * dose;

        let neuromod_effects = NeuromodEffects {
            delta_serotonin: 0.50 * intensity,
            delta_dopamine: 0.15 * intensity,
            delta_noradrenaline: 0.10 * intensity,
            delta_gaba: -0.10 * intensity,
            delta_glutamate: 0.20 * intensity,
        };

        // Psychedelics increase entropy → Phi increase (Carhart-Harris entropic brain).
        let consciousness_impact = intensity as f64 * 0.15 * cs.phi_baseline_modifier;

        PharmaResponse {
            drug_class: DrugClass::Psychedelic,
            effective_half_life,
            peak_concentration,
            neuromod_effects,
            consciousness_impact,
            accumulation_warning: false,
        }
    }

    /// Ketamine response model (NMDA antagonist / dissociative).
    ///
    /// CYP3A4/CYP2B6 metabolism. OPRM1 variant → analgesic component.
    /// Consciousness impact: dissociative → temporary Phi reduction then rebound.
    pub fn ketamine_response(&self, dose: f32) -> PharmaResponse {
        let phenotype = self.metabolizer(CypEnzyme::CYP3A4);
        let cs = self.consciousness_sensitivity();

        let effective_half_life = 2.5 * phenotype.half_life_multiplier();
        let peak_concentration = dose * phenotype.peak_concentration_multiplier();

        let analgesic_component = cs.opioid_sensitivity * 0.3 * dose;

        let neuromod_effects = NeuromodEffects {
            delta_serotonin: 0.10 * dose,
            delta_dopamine: 0.20 * dose,
            delta_noradrenaline: 0.15 * dose,
            delta_gaba: 0.05 * dose + analgesic_component,
            delta_glutamate: -0.30 * dose, // NMDA blockade
        };

        // Dissociative Phi dip: glutamate blockade disrupts integration temporarily.
        // Rebound not modeled here (would need time-series).
        let consciousness_impact = -0.10 * dose as f64 * cs.phi_baseline_modifier;

        PharmaResponse {
            drug_class: DrugClass::Ketamine,
            effective_half_life,
            peak_concentration,
            neuromod_effects,
            consciousness_impact,
            accumulation_warning: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poor_metabolizer_longer_half_life() {
        let poor = MetabolizerPhenotype::Poor;
        let normal = MetabolizerPhenotype::Normal;
        assert!(poor.half_life_multiplier() > normal.half_life_multiplier());
        assert!((poor.half_life_multiplier() - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ultra_rapid_lower_peak_concentration() {
        let ur = MetabolizerPhenotype::UltraRapid;
        let normal = MetabolizerPhenotype::Normal;
        assert!(ur.peak_concentration_multiplier() < normal.peak_concentration_multiplier());
        assert!((ur.peak_concentration_multiplier() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ssri_response_varies_by_5httlpr() {
        // Short/short (Homozygous) → lower transporter efficiency → smaller SSRI boost
        let mut profile_short = PgxProfile::default_european();
        profile_short
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::SLC6A4_5HTTLPR);
        profile_short
            .genetic_variants
            .push((GeneticVariant::SLC6A4_5HTTLPR, Zygosity::Homozygous));

        let mut profile_wt = PgxProfile::default_european();
        profile_wt
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::SLC6A4_5HTTLPR);
        profile_wt
            .genetic_variants
            .push((GeneticVariant::SLC6A4_5HTTLPR, Zygosity::WildType));

        let resp_short = profile_short.ssri_response(1.0);
        let resp_wt = profile_wt.ssri_response(1.0);

        // Short allele carriers have lower transporter efficiency → smaller delta
        assert!(
            resp_short.neuromod_effects.delta_serotonin < resp_wt.neuromod_effects.delta_serotonin
        );
    }

    #[test]
    fn test_psychedelic_response_scales_with_5ht2a() {
        // Homozygous T/T → higher 5-HT2A density → stronger response
        let mut profile_high = PgxProfile::default_european();
        profile_high
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::HTR2A_rs6311);
        profile_high
            .genetic_variants
            .push((GeneticVariant::HTR2A_rs6311, Zygosity::Homozygous));

        let mut profile_low = PgxProfile::default_european();
        profile_low
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::HTR2A_rs6311);
        profile_low
            .genetic_variants
            .push((GeneticVariant::HTR2A_rs6311, Zygosity::WildType));

        let resp_high = profile_high.psychedelic_response(1.0);
        let resp_low = profile_low.psychedelic_response(1.0);

        assert!(resp_high.consciousness_impact > resp_low.consciousness_impact);
        assert!(
            resp_high.neuromod_effects.delta_serotonin > resp_low.neuromod_effects.delta_serotonin
        );
    }

    #[test]
    fn test_default_european_reasonable_baselines() {
        let profile = PgxProfile::default_european();
        let cs = profile.consciousness_sensitivity();

        assert!((cs.phi_baseline_modifier - 1.0).abs() < 0.15);
        assert!(cs.serotonin_receptor_density > 0.0 && cs.serotonin_receptor_density <= 1.0);
        assert!(cs.dopamine_clearance_rate > 0.0 && cs.dopamine_clearance_rate <= 1.0);
        assert!(cs.neuroplasticity_factor > 0.0 && cs.neuroplasticity_factor <= 1.0);
        assert!(cs.opioid_sensitivity > 0.0 && cs.opioid_sensitivity <= 1.0);
    }

    #[test]
    fn test_consciousness_sensitivity_comt_variant() {
        // COMT Val/Val (WildType) → fast clearance; Met/Met (Homozygous) → slow clearance
        let mut profile_val = PgxProfile::default_european();
        profile_val
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::COMT_Val158Met);
        profile_val
            .genetic_variants
            .push((GeneticVariant::COMT_Val158Met, Zygosity::WildType));

        let mut profile_met = PgxProfile::default_european();
        profile_met
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::COMT_Val158Met);
        profile_met
            .genetic_variants
            .push((GeneticVariant::COMT_Val158Met, Zygosity::Homozygous));

        let cs_val = profile_val.consciousness_sensitivity();
        let cs_met = profile_met.consciousness_sensitivity();

        assert!(
            cs_val.dopamine_clearance_rate > cs_met.dopamine_clearance_rate,
            "Val/Val should have faster DA clearance than Met/Met"
        );
    }

    #[test]
    fn test_drug_class_maps_to_primary_cyp() {
        assert_eq!(DrugClass::SSRI.primary_cyp(), CypEnzyme::CYP2D6);
        assert_eq!(DrugClass::Benzodiazepine.primary_cyp(), CypEnzyme::CYP3A4);
        assert_eq!(DrugClass::Ketamine.primary_cyp(), CypEnzyme::CYP3A4);
        assert_eq!(DrugClass::Opioid.primary_cyp(), CypEnzyme::CYP2D6);
        assert_eq!(DrugClass::Psychedelic.primary_cyp(), CypEnzyme::CYP3A4);
    }

    #[test]
    fn test_ketamine_dissociative_phi_dip() {
        let profile = PgxProfile::default_european();
        let resp = profile.ketamine_response(1.0);

        // Ketamine should produce a negative consciousness impact (dissociation).
        assert!(resp.consciousness_impact < 0.0, "Ketamine should dip Phi");
        // Glutamate should be blocked.
        assert!(resp.neuromod_effects.delta_glutamate < 0.0);
    }

    #[test]
    fn test_ancestry_affects_prevalence() {
        // African ancestry has notably higher CYP2D6 ultra-rapid prevalence.
        assert!(
            AncestryGroup::African.cyp2d6_ultra_rapid_prevalence()
                > AncestryGroup::European.cyp2d6_ultra_rapid_prevalence()
        );
        // European ancestry has higher poor-metabolizer prevalence than East Asian.
        assert!(
            AncestryGroup::European.cyp2d6_poor_prevalence()
                > AncestryGroup::EastAsian.cyp2d6_poor_prevalence()
        );
    }

    #[test]
    fn test_bdnf_variant_modulates_neuroplasticity() {
        let mut profile_met = PgxProfile::default_european();
        profile_met
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::BDNF_Val66Met);
        profile_met
            .genetic_variants
            .push((GeneticVariant::BDNF_Val66Met, Zygosity::Homozygous));

        let mut profile_val = PgxProfile::default_european();
        profile_val
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::BDNF_Val66Met);
        profile_val
            .genetic_variants
            .push((GeneticVariant::BDNF_Val66Met, Zygosity::WildType));

        let cs_met = profile_met.consciousness_sensitivity();
        let cs_val = profile_val.consciousness_sensitivity();

        assert!(
            cs_met.neuroplasticity_factor < cs_val.neuroplasticity_factor,
            "BDNF Met/Met should have lower neuroplasticity than Val/Val"
        );
    }

    #[test]
    fn test_predict_response_all_drug_classes() {
        let profile = PgxProfile::default_european();
        let drug_classes = [
            DrugClass::SSRI,
            DrugClass::SNRI,
            DrugClass::Benzodiazepine,
            DrugClass::Psychedelic,
            DrugClass::Stimulant,
            DrugClass::Opioid,
            DrugClass::Antipsychotic,
            DrugClass::Ketamine,
        ];
        for drug in &drug_classes {
            let resp = profile.predict_response(*drug, 1.0);
            assert!(
                resp.effective_half_life > 0.0,
                "Half-life must be positive for {:?}",
                drug
            );
            assert!(
                resp.peak_concentration > 0.0,
                "Peak must be positive for {:?}",
                drug
            );
        }
    }

    #[test]
    fn test_poor_cyp2d6_ssri_accumulation_warning() {
        let mut profile = PgxProfile::default_european();
        profile
            .metabolizer_map
            .insert(CypEnzyme::CYP2D6, MetabolizerPhenotype::Poor);

        let resp = profile.ssri_response(1.0);
        assert!(
            resp.accumulation_warning,
            "Poor CYP2D6 metabolizer on SSRI should trigger accumulation warning"
        );
        assert!(
            resp.effective_half_life > 24.0,
            "Poor metabolizer half-life should exceed normal 24h"
        );
    }

    #[test]
    fn test_metabolizer_phenotype_ordering() {
        // Verify the expected ordering of multipliers.
        let phenotypes = [
            MetabolizerPhenotype::UltraRapid,
            MetabolizerPhenotype::Normal,
            MetabolizerPhenotype::Intermediate,
            MetabolizerPhenotype::Poor,
        ];
        for w in phenotypes.windows(2) {
            assert!(w[0].half_life_multiplier() < w[1].half_life_multiplier());
            assert!(w[0].peak_concentration_multiplier() < w[1].peak_concentration_multiplier());
        }
    }

    #[test]
    fn test_psychedelic_longer_with_slow_comt() {
        // Met/Met COMT → slower DA clearance → longer psychedelic duration.
        let mut profile_slow = PgxProfile::default_european();
        profile_slow
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::COMT_Val158Met);
        profile_slow
            .genetic_variants
            .push((GeneticVariant::COMT_Val158Met, Zygosity::Homozygous));

        let mut profile_fast = PgxProfile::default_european();
        profile_fast
            .genetic_variants
            .retain(|(v, _)| *v != GeneticVariant::COMT_Val158Met);
        profile_fast
            .genetic_variants
            .push((GeneticVariant::COMT_Val158Met, Zygosity::WildType));

        let resp_slow = profile_slow.psychedelic_response(1.0);
        let resp_fast = profile_fast.psychedelic_response(1.0);

        assert!(
            resp_slow.effective_half_life > resp_fast.effective_half_life,
            "Slow COMT should prolong psychedelic half-life"
        );
    }
}
