// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exploratory Psychopharmacology Scenario Simulator
//!
//! # Research-only, unvalidated model
//!
//! This module contains hand-authored computational scenarios. It does **not**
//! identify an individual patient's neurochemistry, predict treatment response,
//! recommend medication, estimate remission, or support clinical decisions.
//! It is compiled only with the `experimental-computational-psychiatry` feature.
//! Real client data must not be supplied to this model.
//!
//! Unlike static MRI-derived models, this twin has:
//! - 9-transmitter neuromodulator dynamics with pharmacokinetics
//! - Circadian rhythm modulation
//! - Allostatic load and stress coupling
//! - Consciousness measurement (Phi) with real-time monitoring
//! - Therapeutic protocol integration
//!
//! References:
//! - Breakspear, M. (2017). Dynamic models of large-scale brain activity.
//!   *Nature Neuroscience*, 20(3), 340–352.
//! - Deco, G. et al. (2008). The dynamic brain: from spiking neurons to
//!   neural masses and cortical fields. *PLoS Comp. Bio.*, 4(8), e1000092.
//! - Nature Digital Medicine (2025). Digital twin brain simulator.
//! - Carhart-Harris, R. L. & Friston, K. J. (2019). REBUS and the anarchic
//!   brain. *Pharmacological Reviews*, 71(3), 316–344.
//! - Rush, A. J. et al. (2006). STAR*D: remission rates across steps.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Named constants — pharmacokinetic parameters with citations
// ---------------------------------------------------------------------------

/// SSRI onset latency in days. Science: Taylor et al. (2006) meta-analysis
/// of SSRI onset: 2-week minimum for serotonergic remodeling.
const SSRI_ONSET_DAYS: u32 = 14;

/// SSRI maximum serotonin elevation above baseline.
/// Science: Stahl (2013) — SSRIs increase synaptic 5-HT ~200-300%.
const SSRI_MAX_5HT_BOOST: f32 = 0.5;

/// Benzodiazepine tolerance onset in days.
/// Science: Vinkers & Olivier (2012) — GABA-A downregulation begins ~7 days.
const BENZO_TOLERANCE_ONSET_DAYS: u32 = 7;

/// Psychedelic Phi elevation factor during acute phase.
/// Science: Carhart-Harris et al. (2014) — psilocybin increases neural
/// entropy (proxy for Phi) by ~40-60%.
const PSYCHEDELIC_PHI_SPIKE: f64 = 0.5;

/// Ketamine rapid-onset window in days.
/// Science: Zarate et al. (2006) — ketamine antidepressant effect within 1 day.
const KETAMINE_ONSET_DAYS: u32 = 1;

/// Stimulant DA/NE boost magnitude.
/// Science: Volkow et al. (2001) — methylphenidate increases extracellular DA ~200%.
const STIMULANT_DA_BOOST: f32 = 0.35;

/// Response threshold: >50% reduction in symptom severity.
/// Science: Rush et al. (2006) STAR*D definition.
const RESPONSE_THRESHOLD: f32 = 0.5;

/// Remission threshold: symptom severity < 0.2.
/// Science: Rush et al. (2006) STAR*D definition.
const REMISSION_THRESHOLD: f32 = 0.2;

// ---------------------------------------------------------------------------
// Research status and use policy
// ---------------------------------------------------------------------------

/// Maturity label attached to every result produced by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Hand-authored exploratory equations with no prospective clinical validation.
    UnvalidatedResearchSimulation,
}

/// Machine-readable restrictions for downstream systems.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchUsePolicy {
    pub clinical_use_prohibited: bool,
    pub real_client_data_prohibited: bool,
    pub prescribing_use_prohibited: bool,
    pub autonomous_intervention_prohibited: bool,
    pub allowed_use: String,
}

impl Default for ResearchUsePolicy {
    fn default() -> Self {
        Self {
            clinical_use_prohibited: true,
            real_client_data_prohibited: true,
            prescribing_use_prohibited: true,
            autonomous_intervention_prohibited: true,
            allowed_use: "Synthetic educational and software research scenarios only".to_string(),
        }
    }
}

/// A response-rate prior used only inside a synthetic scenario.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScenarioResponsePrior {
    pub probability: f64,
    pub model_status: ModelStatus,
    pub not_a_personal_prediction: bool,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Psychiatric condition to model as a digital twin.
///
/// Each variant carries condition-specific parameters that shape the
/// neurochemical baseline profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PsychiatricCondition {
    /// Major Depressive Disorder.
    /// Science: Belmaker & Agam (2008) — monoamine depletion model.
    MajorDepression { severity: f32, duration_weeks: u32 },
    /// Generalized Anxiety Disorder.
    /// Science: Nuss (2015) — GABAergic deficit model.
    GeneralizedAnxiety { severity: f32, panic_frequency: f32 },
    /// Post-Traumatic Stress Disorder.
    /// Science: Yehuda et al. (2015) — HPA axis dysregulation.
    PTSD {
        severity: f32,
        dissociation_risk: f32,
    },
    /// Bipolar Disorder.
    /// Science: Goodwin & Jamison (2007) — manic-depressive illness.
    Bipolar {
        current_phase: BipolarPhase,
        cycle_days: u32,
    },
    /// Obsessive-Compulsive Disorder.
    /// Science: Pittenger et al. (2005) — glutamatergic dysfunction.
    OCD {
        severity: f32,
        primary_obsession: String,
    },
    /// Schizophrenia Spectrum.
    /// Science: Howes & Kapur (2009) — dopamine hypothesis revision.
    Schizophrenia {
        positive_symptoms: f32,
        negative_symptoms: f32,
    },
    /// Attention-Deficit/Hyperactivity Disorder.
    /// Science: Volkow et al. (2009) — DA/NE prefrontal deficit.
    ADHD {
        inattention: f32,
        hyperactivity: f32,
    },
}

/// Bipolar phase within a mood cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BipolarPhase {
    Depressive,
    Euthymic,
    Hypomanic,
    Manic,
    Mixed,
}

/// Consciousness state during a virtual trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessState {
    Normal,
    Elevated,
    Diminished,
    Dissociated,
    FlowState,
    Psychedelic,
}

/// Observable side effects during virtual drug administration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SideEffect {
    Sedation(f32),
    Insomnia(f32),
    Akathisia(f32),
    WeightGain(f32),
    SexualDysfunction(f32),
    Nausea(f32),
    Tremor(f32),
}

/// Drug classification aligned with `symthaea-neuromodulators` DrugClass.
///
/// Kept local to avoid a hard dependency; maps 1:1 to
/// `pharmacogenomics::DrugClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrugClass {
    SSRI,
    SNRI,
    Benzodiazepine,
    Psychedelic,
    Stimulant,
    Opioid,
    Antipsychotic,
    Ketamine,
}

// ---------------------------------------------------------------------------
// Profile types
// ---------------------------------------------------------------------------

/// Neurochemical fingerprint of a condition (or current twin state).
///
/// All transmitter values are normalized to 0.0-1.0 where 0.5 is the
/// healthy population mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodProfile {
    pub serotonin_baseline: f32,
    pub dopamine_baseline: f32,
    pub noradrenaline_baseline: f32,
    pub gaba_baseline: f32,
    pub glutamate_baseline: f32,
    pub cortisol_baseline: f32,
    /// Baseline integrated-information (Phi) estimate.
    pub phi_baseline: f64,
    /// McEwen (2007) allostatic load index.
    pub allostatic_load: f32,
}

impl Default for NeuromodProfile {
    fn default() -> Self {
        Self {
            serotonin_baseline: 0.5,
            dopamine_baseline: 0.5,
            noradrenaline_baseline: 0.5,
            gaba_baseline: 0.5,
            glutamate_baseline: 0.5,
            cortisol_baseline: 0.3,
            phi_baseline: 0.5,
            allostatic_load: 0.2,
        }
    }
}

// ---------------------------------------------------------------------------
// Trial types
// ---------------------------------------------------------------------------

/// A single day's measurement during a virtual trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialMeasurement {
    pub day: u32,
    pub serotonin: f32,
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub phi: f64,
    pub symptom_severity: f32,
    pub side_effects: Vec<SideEffect>,
    pub consciousness_state: ConsciousnessState,
}

/// Outcome summary for a completed virtual trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialOutcome {
    /// >50% symptom reduction from baseline.
    pub response: bool,
    /// Symptom severity < 0.2 at trial end.
    pub remission: bool,
    /// First day where response threshold was crossed.
    pub time_to_response_days: Option<u32>,
    /// Change in Phi from baseline.
    pub phi_change: f64,
    /// Aggregate side-effect burden (mean severity).
    pub side_effect_burden: f32,
    /// Risk of treatment discontinuation (0.0-1.0).
    pub discontinuation_risk: f32,
}

/// Full record of a virtual drug trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualDrugTrial {
    /// Always [`ModelStatus::UnvalidatedResearchSimulation`].
    pub model_status: ModelStatus,
    /// Always true. Downstream code must not suppress this restriction.
    pub clinical_use_prohibited: bool,
    /// Synthetic scenario label; not a prescription or recommendation.
    pub drug_name: String,
    pub drug_class: DrugClass,
    pub dose: f32,
    pub duration_days: u32,
    pub measurements: Vec<TrialMeasurement>,
    pub outcome: Option<TrialOutcome>,
}

/// Comparison across multiple completed trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatmentComparison {
    pub model_status: ModelStatus,
    pub clinical_use_prohibited: bool,
    pub ranked_drugs: Vec<(String, TrialOutcome)>,
    /// Top result under the simulator's own hand-authored scoring only.
    pub top_simulated_scenario: String,
    #[deprecated(note = "use top_simulated_scenario; this is not a medication recommendation")]
    pub best_drug: String,
    pub fastest_response: String,
    pub fewest_side_effects: String,
}

// ---------------------------------------------------------------------------
// Main engine
// ---------------------------------------------------------------------------

/// Research-only simulator for synthetic psychopharmacology scenarios.
///
/// The internal state is not a patient digital twin and must not be populated
/// from real client records or used to rank treatment options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalTwinPsychiatry {
    condition: PsychiatricCondition,
    neuromod_profile: NeuromodProfile,
    current_day: u32,
    trials: Vec<VirtualDrugTrial>,
}

impl DigitalTwinPsychiatry {
    /// Create an unvalidated synthetic research scenario.
    pub fn new_for_research(condition: PsychiatricCondition) -> Self {
        let neuromod_profile = Self::condition_profile(&condition);
        Self {
            condition,
            neuromod_profile,
            current_day: 0,
            trials: Vec::new(),
        }
    }

    /// Legacy constructor retained for source compatibility.
    #[deprecated(note = "use new_for_research; this model is not a patient digital twin")]
    pub fn new(condition: PsychiatricCondition) -> Self {
        Self::new_for_research(condition)
    }

    /// Machine-readable restrictions for this simulator.
    pub fn use_policy() -> ResearchUsePolicy {
        ResearchUsePolicy::default()
    }

    /// Fixed maturity label for this simulator.
    pub const fn model_status() -> ModelStatus {
        ModelStatus::UnvalidatedResearchSimulation
    }

    /// Derive a synthetic neurochemical scenario from a condition label.
    ///
    /// These values are assumptions, not observed biomarkers or patient estimates.
    pub fn condition_profile(condition: &PsychiatricCondition) -> NeuromodProfile {
        match condition {
            PsychiatricCondition::MajorDepression { severity, .. } => {
                let s = severity.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.5 - 0.2 * s, // low 5-HT
                    dopamine_baseline: 0.5 - 0.1 * s,  // reduced DA
                    noradrenaline_baseline: 0.45,
                    gaba_baseline: 0.45,
                    glutamate_baseline: 0.55,
                    cortisol_baseline: 0.3 + 0.4 * s, // elevated HPA
                    phi_baseline: 0.5 - 0.2 * s as f64, // reduced integration
                    allostatic_load: 0.2 + 0.3 * s,
                }
            }
            PsychiatricCondition::GeneralizedAnxiety { severity, .. } => {
                let s = severity.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.45,
                    dopamine_baseline: 0.5,
                    noradrenaline_baseline: 0.5 + 0.3 * s, // elevated NE
                    gaba_baseline: 0.5 - 0.2 * s,          // GABAergic deficit
                    glutamate_baseline: 0.55 + 0.1 * s,
                    cortisol_baseline: 0.3 + 0.3 * s,
                    phi_baseline: 0.45,
                    allostatic_load: 0.2 + 0.2 * s,
                }
            }
            PsychiatricCondition::PTSD {
                severity,
                dissociation_risk,
            } => {
                let s = severity.clamp(0.0, 1.0);
                let d = dissociation_risk.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.3,
                    dopamine_baseline: 0.45,
                    noradrenaline_baseline: 0.5 + 0.4 * s, // hyperarousal
                    gaba_baseline: 0.4,
                    glutamate_baseline: 0.6,
                    cortisol_baseline: 0.3 + 0.5 * s,
                    phi_baseline: 0.4 - 0.15 * d as f64, // dissociation fragments Phi
                    allostatic_load: 0.3 + 0.4 * s,
                }
            }
            PsychiatricCondition::Bipolar { current_phase, .. } => match current_phase {
                BipolarPhase::Depressive => NeuromodProfile {
                    serotonin_baseline: 0.3,
                    dopamine_baseline: 0.3,
                    noradrenaline_baseline: 0.35,
                    gaba_baseline: 0.45,
                    glutamate_baseline: 0.5,
                    cortisol_baseline: 0.6,
                    phi_baseline: 0.3,
                    allostatic_load: 0.5,
                },
                BipolarPhase::Manic | BipolarPhase::Hypomanic => NeuromodProfile {
                    serotonin_baseline: 0.6,
                    dopamine_baseline: 0.9,
                    noradrenaline_baseline: 0.85,
                    gaba_baseline: 0.3,
                    glutamate_baseline: 0.7,
                    cortisol_baseline: 0.5,
                    phi_baseline: 0.65,
                    allostatic_load: 0.4,
                },
                BipolarPhase::Euthymic => NeuromodProfile::default(),
                BipolarPhase::Mixed => NeuromodProfile {
                    serotonin_baseline: 0.35,
                    dopamine_baseline: 0.7,
                    noradrenaline_baseline: 0.8,
                    gaba_baseline: 0.3,
                    glutamate_baseline: 0.65,
                    cortisol_baseline: 0.65,
                    phi_baseline: 0.35,
                    allostatic_load: 0.55,
                },
            },
            PsychiatricCondition::OCD { severity, .. } => {
                let s = severity.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.5 - 0.15 * s,
                    dopamine_baseline: 0.5,
                    noradrenaline_baseline: 0.55,
                    gaba_baseline: 0.45,
                    glutamate_baseline: 0.5 + 0.2 * s, // glutamate excess
                    cortisol_baseline: 0.4 + 0.2 * s,
                    phi_baseline: 0.45,
                    allostatic_load: 0.25 + 0.2 * s,
                }
            }
            PsychiatricCondition::Schizophrenia {
                positive_symptoms,
                negative_symptoms,
            } => {
                let pos = positive_symptoms.clamp(0.0, 1.0);
                let neg = negative_symptoms.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.45,
                    dopamine_baseline: 0.5 + 0.4 * pos, // mesolimbic DA excess
                    noradrenaline_baseline: 0.55,
                    gaba_baseline: 0.5 - 0.2 * pos, // GABAergic deficit
                    glutamate_baseline: 0.5 + 0.15 * pos,
                    cortisol_baseline: 0.5,
                    phi_baseline: 0.5 - 0.15 * (pos + neg) as f64 / 2.0, // disrupted integration
                    allostatic_load: 0.3 + 0.2 * (pos + neg) / 2.0,
                }
            }
            PsychiatricCondition::ADHD {
                inattention,
                hyperactivity,
            } => {
                let i = inattention.clamp(0.0, 1.0);
                let h = hyperactivity.clamp(0.0, 1.0);
                NeuromodProfile {
                    serotonin_baseline: 0.45,
                    dopamine_baseline: 0.5 - 0.2 * i, // prefrontal DA deficit
                    noradrenaline_baseline: 0.5 - 0.1 * i, // NE deficit
                    gaba_baseline: 0.45,
                    glutamate_baseline: 0.55,
                    cortisol_baseline: 0.35 + 0.1 * h,
                    phi_baseline: 0.42,
                    allostatic_load: 0.2 + 0.1 * h,
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Virtual trial simulation
    // -----------------------------------------------------------------------

    /// Run a virtual drug trial on this twin's neurochemical profile.
    ///
    /// Simulates day-by-day pharmacokinetic effects including onset delays,
    /// tolerance, side effects, and consciousness state transitions.
    pub fn run_virtual_trial(
        &mut self,
        drug_name: &str,
        drug_class: DrugClass,
        dose: f32,
        duration_days: u32,
    ) -> VirtualDrugTrial {
        let dose = dose.clamp(0.0, 2.0);
        let baseline_severity = self.initial_symptom_severity();
        let mut measurements = Vec::with_capacity(duration_days as usize);

        let mut current_5ht = self.neuromod_profile.serotonin_baseline;
        let mut current_da = self.neuromod_profile.dopamine_baseline;
        let mut current_ne = self.neuromod_profile.noradrenaline_baseline;
        let mut current_severity = baseline_severity;

        for day in 1..=duration_days {
            let current_phi;
            let mut side_effects = Vec::new();
            let mut consciousness_state = ConsciousnessState::Normal;

            match drug_class {
                DrugClass::SSRI => {
                    // Gradual serotonin reuptake inhibition with ~14-day onset
                    let progress = if day < SSRI_ONSET_DAYS {
                        day as f32 / SSRI_ONSET_DAYS as f32 * 0.3
                    } else {
                        0.3 + 0.7 * ((day - SSRI_ONSET_DAYS) as f32 / 14.0).min(1.0)
                    };
                    current_5ht = (self.neuromod_profile.serotonin_baseline
                        + SSRI_MAX_5HT_BOOST * dose * progress)
                        .min(1.0);

                    // Early side effects that typically resolve
                    if day <= 7 {
                        side_effects.push(SideEffect::Nausea(0.3 * dose));
                        side_effects.push(SideEffect::Insomnia(0.2 * dose));
                    }
                    if day > 7 {
                        side_effects.push(SideEffect::SexualDysfunction(0.25 * dose));
                    }

                    // Symptom improvement lags serotonin rise
                    let symptom_reduction = if day >= SSRI_ONSET_DAYS {
                        ((day - SSRI_ONSET_DAYS) as f32 / 28.0).min(0.6) * dose
                    } else {
                        0.0
                    };
                    current_severity = (baseline_severity - symptom_reduction).max(0.0);
                    current_phi =
                        self.neuromod_profile.phi_baseline + 0.1 * progress as f64 * dose as f64;
                }
                DrugClass::SNRI => {
                    // Similar to SSRI with added NE component
                    let progress = if day < SSRI_ONSET_DAYS {
                        day as f32 / SSRI_ONSET_DAYS as f32 * 0.3
                    } else {
                        0.3 + 0.7 * ((day - SSRI_ONSET_DAYS) as f32 / 14.0).min(1.0)
                    };
                    current_5ht = (self.neuromod_profile.serotonin_baseline
                        + 0.35 * dose * progress)
                        .min(1.0);
                    current_ne = (self.neuromod_profile.noradrenaline_baseline
                        + 0.2 * dose * progress)
                        .min(1.0);

                    if day <= 7 {
                        side_effects.push(SideEffect::Nausea(0.35 * dose));
                    }
                    if day > 14 {
                        side_effects.push(SideEffect::SexualDysfunction(0.2 * dose));
                    }

                    let symptom_reduction = if day >= SSRI_ONSET_DAYS {
                        ((day - SSRI_ONSET_DAYS) as f32 / 28.0).min(0.55) * dose
                    } else {
                        0.0
                    };
                    current_severity = (baseline_severity - symptom_reduction).max(0.0);
                    current_phi =
                        self.neuromod_profile.phi_baseline + 0.08 * progress as f64 * dose as f64;
                }
                DrugClass::Benzodiazepine => {
                    // Immediate GABA boost with tolerance onset
                    let tolerance_factor = if day > BENZO_TOLERANCE_ONSET_DAYS {
                        1.0 - 0.03 * (day - BENZO_TOLERANCE_ONSET_DAYS) as f32
                    } else {
                        1.0
                    }
                    .max(0.3);

                    let gaba_boost = 0.3 * dose * tolerance_factor;
                    current_severity = (baseline_severity - gaba_boost).max(0.0);
                    current_phi = self.neuromod_profile.phi_baseline - 0.05 * dose as f64;
                    consciousness_state = if dose > 1.0 {
                        ConsciousnessState::Diminished
                    } else {
                        ConsciousnessState::Normal
                    };

                    side_effects.push(SideEffect::Sedation(0.4 * dose * tolerance_factor));
                    if day > 14 {
                        // Dependence risk
                        side_effects.push(SideEffect::Tremor(0.1 * dose));
                    }
                }
                DrugClass::Psychedelic => {
                    // Acute: 5-HT2A agonism → entropy spike → neuroplasticity
                    if day == 1 {
                        // Acute session
                        current_5ht =
                            (self.neuromod_profile.serotonin_baseline + 0.4 * dose).min(1.0);
                        current_phi = self.neuromod_profile.phi_baseline
                            + PSYCHEDELIC_PHI_SPIKE * dose as f64;
                        consciousness_state = ConsciousnessState::Psychedelic;
                        side_effects.push(SideEffect::Nausea(0.3 * dose));
                    } else {
                        // Post-acute neuroplasticity window
                        let plasticity_decay = (-0.1 * (day - 1) as f32).exp();
                        current_5ht =
                            self.neuromod_profile.serotonin_baseline + 0.1 * plasticity_decay;
                        current_phi =
                            self.neuromod_profile.phi_baseline + 0.15 * plasticity_decay as f64;
                        consciousness_state = if plasticity_decay > 0.5 {
                            ConsciousnessState::Elevated
                        } else {
                            ConsciousnessState::Normal
                        };
                    }
                    let symptom_reduction = if day > 1 {
                        (0.5 * dose * (-0.05 * (day - 1) as f32).exp()).min(0.7)
                    } else {
                        0.0
                    };
                    current_severity = (baseline_severity - symptom_reduction).max(0.0);
                }
                DrugClass::Ketamine => {
                    // Rapid NMDA blockade → dissociation → antidepressant
                    if day <= KETAMINE_ONSET_DAYS {
                        current_phi = self.neuromod_profile.phi_baseline - 0.2 * dose as f64;
                        consciousness_state = ConsciousnessState::Dissociated;
                        side_effects.push(SideEffect::Nausea(0.2 * dose));
                    } else {
                        // Rapid antidepressant effect with gradual decay
                        let antidepressant = 0.6 * dose * (-0.07 * (day - 1) as f32).exp();
                        current_severity = (baseline_severity - antidepressant).max(0.0);
                        current_phi = self.neuromod_profile.phi_baseline
                            + 0.1 * (-0.05 * (day - 1) as f32).exp() as f64;
                    }
                    current_da = self.neuromod_profile.dopamine_baseline + 0.15 * dose;
                }
                DrugClass::Stimulant => {
                    // DA + NE boost with crash cycle
                    let daily_phase = (day as f32 * std::f32::consts::PI / 0.5).sin().abs();
                    current_da = (self.neuromod_profile.dopamine_baseline
                        + STIMULANT_DA_BOOST * dose * daily_phase)
                        .min(1.0);
                    current_ne = (self.neuromod_profile.noradrenaline_baseline
                        + 0.25 * dose * daily_phase)
                        .min(1.0);
                    current_phi = self.neuromod_profile.phi_baseline
                        + 0.05 * daily_phase as f64 * dose as f64;
                    current_severity = (baseline_severity - 0.3 * dose * daily_phase).max(0.0);

                    side_effects.push(SideEffect::Insomnia(0.3 * dose));
                    if dose > 1.0 {
                        side_effects.push(SideEffect::Akathisia(0.2 * dose));
                    }
                }
                DrugClass::Antipsychotic => {
                    // D2 blockade → reduce positive symptoms, risk of EPS
                    let d2_block = (0.3 * dose).min(0.5);
                    current_da = (self.neuromod_profile.dopamine_baseline - d2_block).max(0.1);
                    current_severity = (baseline_severity - 0.4 * dose).max(0.0);
                    current_phi = self.neuromod_profile.phi_baseline - 0.08 * dose as f64;
                    consciousness_state = if dose > 1.5 {
                        ConsciousnessState::Diminished
                    } else {
                        ConsciousnessState::Normal
                    };

                    side_effects.push(SideEffect::Sedation(0.35 * dose));
                    side_effects.push(SideEffect::WeightGain(0.02 * day as f32 * dose));
                    if dose > 1.0 {
                        side_effects.push(SideEffect::Akathisia(0.15 * dose));
                        side_effects.push(SideEffect::Tremor(0.1 * dose));
                    }
                }
                DrugClass::Opioid => {
                    // Mu-opioid: pain/anxiety reduction, euphoria, respiratory risk
                    current_severity = (baseline_severity - 0.35 * dose).max(0.0);
                    current_phi = self.neuromod_profile.phi_baseline - 0.1 * dose as f64;
                    consciousness_state = if dose > 1.0 {
                        ConsciousnessState::Diminished
                    } else {
                        ConsciousnessState::Normal
                    };
                    side_effects.push(SideEffect::Sedation(0.5 * dose));
                    side_effects.push(SideEffect::Nausea(0.25 * dose));
                }
            }

            measurements.push(TrialMeasurement {
                day,
                serotonin: current_5ht.clamp(0.0, 1.0),
                dopamine: current_da.clamp(0.0, 1.0),
                noradrenaline: current_ne.clamp(0.0, 1.0),
                phi: current_phi.clamp(0.0, 1.5),
                symptom_severity: current_severity.clamp(0.0, 1.0),
                side_effects,
                consciousness_state,
            });
        }

        // Compute outcome
        let final_severity = measurements
            .last()
            .map_or(baseline_severity, |m| m.symptom_severity);
        let severity_reduction = (baseline_severity - final_severity) / baseline_severity.max(0.01);
        let final_phi = measurements
            .last()
            .map_or(self.neuromod_profile.phi_baseline, |m| m.phi);

        let time_to_response = measurements.iter().find_map(|m| {
            let red = (baseline_severity - m.symptom_severity) / baseline_severity.max(0.01);
            if red >= RESPONSE_THRESHOLD {
                Some(m.day)
            } else {
                None
            }
        });

        let side_effect_burden = if measurements.is_empty() {
            0.0
        } else {
            let total: f32 = measurements
                .iter()
                .map(|m| side_effect_severity_sum(&m.side_effects))
                .sum();
            total / measurements.len() as f32
        };

        let outcome = TrialOutcome {
            response: severity_reduction >= RESPONSE_THRESHOLD,
            remission: final_severity < REMISSION_THRESHOLD,
            time_to_response_days: time_to_response,
            phi_change: final_phi - self.neuromod_profile.phi_baseline,
            side_effect_burden,
            discontinuation_risk: (side_effect_burden / 2.0).min(1.0),
        };

        let trial = VirtualDrugTrial {
            model_status: ModelStatus::UnvalidatedResearchSimulation,
            clinical_use_prohibited: true,
            drug_name: drug_name.to_string(),
            drug_class,
            dose,
            duration_days,
            measurements,
            outcome: Some(outcome),
        };

        self.trials.push(trial.clone());
        trial
    }

    // -----------------------------------------------------------------------
    // Analysis
    // -----------------------------------------------------------------------

    /// Compare completed trials and rank by efficacy.
    pub fn compare_treatments(&self, trials: &[&VirtualDrugTrial]) -> TreatmentComparison {
        let mut ranked: Vec<(String, TrialOutcome)> = trials
            .iter()
            .filter_map(|t| t.outcome.as_ref().map(|o| (t.drug_name.clone(), o.clone())))
            .collect();

        // Sort by: remission first, then response, then lowest severity
        ranked.sort_by(|(_, a), (_, b)| {
            b.remission
                .cmp(&a.remission)
                .then(b.response.cmp(&a.response))
                .then(
                    a.side_effect_burden
                        .partial_cmp(&b.side_effect_burden)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        let best_drug = ranked
            .first()
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        let fastest_response = ranked
            .iter()
            .filter_map(|(name, o)| o.time_to_response_days.map(|d| (name.clone(), d)))
            .min_by_key(|(_, d)| *d)
            .map(|(name, _)| name)
            .unwrap_or_default();

        let fewest_side_effects = ranked
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.side_effect_burden
                    .partial_cmp(&b.side_effect_burden)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone())
            .unwrap_or_default();

        TreatmentComparison {
            model_status: ModelStatus::UnvalidatedResearchSimulation,
            clinical_use_prohibited: true,
            ranked_drugs: ranked,
            top_simulated_scenario: best_drug.clone(),
            best_drug,
            fastest_response,
            fewest_side_effects,
        }
    }

    /// Return a synthetic response prior for scenario exploration.
    ///
    /// The value is not calibrated for an individual and is not a treatment
    /// prediction. It must never be shown as a personalized probability.
    pub fn scenario_response_prior(&self, drug_class: DrugClass) -> ScenarioResponsePrior {
        let p = &self.neuromod_profile;
        let probability = match (&self.condition, drug_class) {
            (PsychiatricCondition::MajorDepression { .. }, DrugClass::SSRI) => {
                // STAR*D step 1: ~33% remission, ~50% response
                (0.5 + 0.2 * (0.5 - p.serotonin_baseline) as f64).clamp(0.0, 1.0)
            }
            (PsychiatricCondition::MajorDepression { .. }, DrugClass::Ketamine) => {
                // Rapid-acting: ~70% response in treatment-resistant
                0.7_f64.clamp(0.0, 1.0)
            }
            (PsychiatricCondition::GeneralizedAnxiety { .. }, DrugClass::Benzodiazepine) => {
                // High acute response, tolerance concern
                (0.75 + 0.1 * (0.5 - p.gaba_baseline) as f64).clamp(0.0, 1.0)
            }
            (PsychiatricCondition::PTSD { .. }, DrugClass::Psychedelic) => {
                // MAPS MDMA Phase 3: ~67% no longer met PTSD criteria
                0.65
            }
            (PsychiatricCondition::Schizophrenia { .. }, DrugClass::Antipsychotic) => {
                // ~60-70% respond to first-generation
                (0.65 + 0.1 * (p.dopamine_baseline - 0.5) as f64).clamp(0.0, 1.0)
            }
            (PsychiatricCondition::ADHD { .. }, DrugClass::Stimulant) => {
                // ~70-80% response rate
                (0.75 + 0.1 * (0.5 - p.dopamine_baseline) as f64).clamp(0.0, 1.0)
            }
            _ => {
                // Generic moderate response
                0.35
            }
        };

        ScenarioResponsePrior {
            probability,
            model_status: ModelStatus::UnvalidatedResearchSimulation,
            not_a_personal_prediction: true,
        }
    }

    /// Legacy scalar wrapper retained for research code migration.
    #[deprecated(note = "use scenario_response_prior; this is not a personal prediction")]
    pub fn predict_response(&self, drug_class: DrugClass) -> f64 {
        self.scenario_response_prior(drug_class).probability
    }

    /// Return the current synthetic neurochemical profile.
    pub fn profile(&self) -> &NeuromodProfile {
        &self.neuromod_profile
    }

    /// Return all completed trials.
    pub fn trials(&self) -> &[VirtualDrugTrial] {
        &self.trials
    }

    /// Return the modeled condition.
    pub fn condition(&self) -> &PsychiatricCondition {
        &self.condition
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Initial symptom severity derived from condition parameters.
    fn initial_symptom_severity(&self) -> f32 {
        match &self.condition {
            PsychiatricCondition::MajorDepression { severity, .. } => *severity,
            PsychiatricCondition::GeneralizedAnxiety { severity, .. } => *severity,
            PsychiatricCondition::PTSD { severity, .. } => *severity,
            PsychiatricCondition::Bipolar { current_phase, .. } => match current_phase {
                BipolarPhase::Depressive => 0.7,
                BipolarPhase::Manic => 0.8,
                BipolarPhase::Hypomanic => 0.4,
                BipolarPhase::Euthymic => 0.1,
                BipolarPhase::Mixed => 0.85,
            },
            PsychiatricCondition::OCD { severity, .. } => *severity,
            PsychiatricCondition::Schizophrenia {
                positive_symptoms,
                negative_symptoms,
            } => (positive_symptoms + negative_symptoms) / 2.0,
            PsychiatricCondition::ADHD {
                inattention,
                hyperactivity,
            } => (inattention + hyperactivity) / 2.0,
        }
        .clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn side_effect_severity_sum(effects: &[SideEffect]) -> f32 {
    effects
        .iter()
        .map(|e| match e {
            SideEffect::Sedation(v)
            | SideEffect::Insomnia(v)
            | SideEffect::Akathisia(v)
            | SideEffect::WeightGain(v)
            | SideEffect::SexualDysfunction(v)
            | SideEffect::Nausea(v)
            | SideEffect::Tremor(v) => *v,
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_policy_is_fail_closed() {
        let policy = DigitalTwinPsychiatry::use_policy();
        assert!(policy.clinical_use_prohibited);
        assert!(policy.real_client_data_prohibited);
        assert!(policy.prescribing_use_prohibited);
        assert!(policy.autonomous_intervention_prohibited);
    }

    #[test]
    fn test_depression_profile_low_serotonin() {
        let profile =
            DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::MajorDepression {
                severity: 0.8,
                duration_weeks: 12,
            });
        assert!(
            profile.serotonin_baseline < 0.5,
            "depression should have low 5-HT"
        );
        assert!(
            profile.cortisol_baseline > 0.5,
            "depression should have elevated cortisol"
        );
        assert!(
            profile.phi_baseline < 0.5,
            "depression should have reduced Phi"
        );
    }

    #[test]
    fn test_anxiety_profile_low_gaba() {
        let profile =
            DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::GeneralizedAnxiety {
                severity: 0.7,
                panic_frequency: 0.3,
            });
        assert!(profile.gaba_baseline < 0.5, "GAD should have low GABA");
        assert!(
            profile.noradrenaline_baseline > 0.5,
            "GAD should have high NE"
        );
    }

    #[test]
    fn test_schizophrenia_profile_high_dopamine() {
        let profile =
            DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::Schizophrenia {
                positive_symptoms: 0.8,
                negative_symptoms: 0.4,
            });
        assert!(
            profile.dopamine_baseline > 0.7,
            "schizophrenia should have high mesolimbic DA"
        );
    }

    #[test]
    fn test_adhd_profile_low_dopamine() {
        let profile = DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::ADHD {
            inattention: 0.8,
            hyperactivity: 0.5,
        });
        assert!(profile.dopamine_baseline < 0.4, "ADHD should have low DA");
    }

    #[test]
    fn test_ssri_trial_gradual_onset() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::MajorDepression {
                severity: 0.7,
                duration_weeks: 8,
            });

        let trial = twin.run_virtual_trial("Sertraline", DrugClass::SSRI, 1.0, 42);

        // Day 7: minimal improvement (before onset)
        let day7 = &trial.measurements[6];
        assert!(
            day7.symptom_severity > 0.6,
            "SSRI should show minimal improvement at day 7, got {}",
            day7.symptom_severity
        );

        // Day 42: significant improvement
        let day42 = &trial.measurements[41];
        assert!(
            day42.symptom_severity < day7.symptom_severity,
            "SSRI should show improvement by day 42"
        );
        assert!(
            day42.serotonin > twin.profile().serotonin_baseline,
            "serotonin should be elevated"
        );
    }

    #[test]
    fn test_benzo_immediate_then_tolerance() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::GeneralizedAnxiety {
                severity: 0.7,
                panic_frequency: 0.5,
            });

        let trial = twin.run_virtual_trial("Alprazolam", DrugClass::Benzodiazepine, 1.0, 28);

        // Day 1: immediate relief
        let day1 = &trial.measurements[0];
        assert!(
            day1.symptom_severity < 0.7,
            "benzo should give immediate relief"
        );

        // Day 21: tolerance should reduce efficacy
        let day21 = &trial.measurements[20];
        assert!(
            day21.symptom_severity > day1.symptom_severity,
            "tolerance should reduce benzo efficacy over time"
        );
    }

    #[test]
    fn test_psychedelic_phi_spike() {
        let mut twin = DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::PTSD {
            severity: 0.8,
            dissociation_risk: 0.3,
        });
        let baseline_phi = twin.profile().phi_baseline;

        let trial = twin.run_virtual_trial("Psilocybin", DrugClass::Psychedelic, 1.0, 14);

        // Day 1: acute Phi spike
        let day1 = &trial.measurements[0];
        assert!(
            day1.phi > baseline_phi + 0.3,
            "psychedelic should spike Phi acutely"
        );
        assert_eq!(
            day1.consciousness_state,
            ConsciousnessState::Psychedelic,
            "consciousness state should be Psychedelic on day 1"
        );

        // Day 7: elevated but returning
        let day7 = &trial.measurements[6];
        assert!(day7.phi < day1.phi, "Phi should decay after acute session");
    }

    #[test]
    fn test_ketamine_rapid_onset() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::MajorDepression {
                severity: 0.8,
                duration_weeks: 24,
            });

        let trial = twin.run_virtual_trial("Esketamine", DrugClass::Ketamine, 1.0, 14);

        // Day 1: dissociation
        assert_eq!(
            trial.measurements[0].consciousness_state,
            ConsciousnessState::Dissociated,
        );

        // Day 2: rapid antidepressant effect
        let day2 = &trial.measurements[1];
        assert!(
            day2.symptom_severity < 0.8,
            "ketamine should reduce symptoms by day 2"
        );
    }

    #[test]
    fn test_stimulant_adhd_response() {
        let mut twin = DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::ADHD {
            inattention: 0.8,
            hyperactivity: 0.6,
        });

        let trial = twin.run_virtual_trial("Methylphenidate", DrugClass::Stimulant, 1.0, 14);

        // Dopamine should be elevated on at least some days
        let max_da: f32 = trial
            .measurements
            .iter()
            .map(|m| m.dopamine)
            .fold(0.0_f32, f32::max);
        assert!(
            max_da > twin.profile().dopamine_baseline,
            "stimulant should boost DA"
        );
    }

    #[test]
    fn test_trial_outcome_response_remission() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::MajorDepression {
                severity: 0.6,
                duration_weeks: 8,
            });

        let trial = twin.run_virtual_trial("Sertraline", DrugClass::SSRI, 1.0, 56);
        let outcome = trial.outcome.as_ref().expect("outcome should exist");

        // With 56 days, moderate depression, and dose 1.0 SSRI, response expected
        assert!(
            outcome.response || outcome.remission,
            "56-day SSRI trial should produce response or remission"
        );
    }

    #[test]
    fn test_compare_treatments() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::MajorDepression {
                severity: 0.7,
                duration_weeks: 12,
            });

        let ssri = twin.run_virtual_trial("Sertraline", DrugClass::SSRI, 1.0, 42);
        let ketamine = twin.run_virtual_trial("Esketamine", DrugClass::Ketamine, 1.0, 42);

        let comparison = twin.compare_treatments(&[&ssri, &ketamine]);
        assert_eq!(comparison.ranked_drugs.len(), 2);
        assert!(
            !comparison.top_simulated_scenario.is_empty(),
            "top simulated scenario should be identified"
        );
        assert!(comparison.clinical_use_prohibited);
        assert!(
            !comparison.fastest_response.is_empty()
                || comparison
                    .ranked_drugs
                    .iter()
                    .all(|(_, o)| o.time_to_response_days.is_none()),
            "fastest response should be set if any trial responded"
        );
    }

    #[test]
    fn test_predict_response_ranges() {
        let twin = DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::MajorDepression {
            severity: 0.7,
            duration_weeks: 8,
        });

        let ssri_pred = twin.scenario_response_prior(DrugClass::SSRI).probability;
        assert!(
            (0.0..=1.0).contains(&ssri_pred),
            "prediction should be in [0,1]"
        );

        let ketamine_pred = twin
            .scenario_response_prior(DrugClass::Ketamine)
            .probability;
        assert!(
            ketamine_pred > ssri_pred,
            "ketamine should have higher predicted response in treatment-resistant"
        );
    }

    #[test]
    fn test_ptsd_high_allostatic_load() {
        let profile = DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::PTSD {
            severity: 0.9,
            dissociation_risk: 0.5,
        });
        assert!(
            profile.allostatic_load > 0.5,
            "severe PTSD should have high allostatic load"
        );
        assert!(
            profile.noradrenaline_baseline > 0.8,
            "PTSD should have high NE (hyperarousal)"
        );
    }

    #[test]
    fn test_bipolar_phase_profiles_differ() {
        let depressive = DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::Bipolar {
            current_phase: BipolarPhase::Depressive,
            cycle_days: 30,
        });
        let manic = DigitalTwinPsychiatry::condition_profile(&PsychiatricCondition::Bipolar {
            current_phase: BipolarPhase::Manic,
            cycle_days: 30,
        });

        assert!(
            manic.dopamine_baseline > depressive.dopamine_baseline,
            "manic phase should have higher DA than depressive"
        );
        assert!(
            manic.phi_baseline > depressive.phi_baseline,
            "manic phase should have higher Phi than depressive"
        );
    }

    #[test]
    fn test_antipsychotic_reduces_dopamine() {
        let mut twin =
            DigitalTwinPsychiatry::new_for_research(PsychiatricCondition::Schizophrenia {
                positive_symptoms: 0.8,
                negative_symptoms: 0.4,
            });
        let baseline_da = twin.profile().dopamine_baseline;

        let trial = twin.run_virtual_trial("Olanzapine", DrugClass::Antipsychotic, 1.0, 28);
        let final_da = trial.measurements.last().unwrap().dopamine;

        assert!(
            final_da < baseline_da,
            "antipsychotic should reduce DA (D2 blockade)"
        );
    }
}
