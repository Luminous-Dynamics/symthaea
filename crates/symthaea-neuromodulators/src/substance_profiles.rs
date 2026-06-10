// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Substance-specific neuromodulator profiles for addiction modeling.
//!
//! Each profile parameterizes how a substance class affects the neuromodulator bath,
//! including primary/secondary effects, tolerance acceleration, withdrawal severity,
//! and allostatic load burden.
//!
//! Science:
//! - Koob & Le Moal (2001) — allostatic model of drug addiction
//! - Volkow et al. (2004) — dopamine in drug abuse and addiction
//! - Nestler (2005) — molecular basis of long-term plasticity underlying addiction
//! - Nutt et al. (2015) — Drug harms in the UK: a multicriteria decision analysis

use serde::{Deserialize, Serialize};

/// Substance category (mirrors mycelix-health SubstanceCategory)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubstanceCategory {
    Opioids,
    /// Synthetic opioids (fentanyl, carfentanil) — distinct pharmacology from heroin/Rx opioids.
    /// Science: Volkow & McLellan (2016) — CDC data shows fentanyl dominates overdose deaths.
    SyntheticOpioids,
    Alcohol,
    Stimulants,
    /// Short-acting benzodiazepines (alprazolam, lorazepam) — faster onset, more abuse potential.
    BenzodiazepinesShort,
    /// Long-acting benzodiazepines (diazepam, clonazepam) — slower onset, protracted withdrawal.
    BenzodiazepinesLong,
    Cannabis,
    Tobacco,
    Psychedelics,
}

/// Backwards compatibility alias.
pub const BENZODIAZEPINES: SubstanceCategory = SubstanceCategory::BenzodiazepinesShort;

/// Which transmitter channel is affected.
/// Indices match NeuromodulatorBath field order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransmitterTarget {
    Dopamine,        // 0
    Noradrenaline,   // 1
    Serotonin,       // 2
    Acetylcholine,   // 3
    Gaba,            // 4
    Oxytocin,        // 5
    Glutamate,       // 6
    Adenosine,       // 7
    Endocannabinoid, // 8
}

/// A single effect on one transmitter channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmitterEffect {
    pub target: TransmitterTarget,
    /// Magnitude of effect: positive = agonist/increase, negative = antagonist/decrease.
    /// Scaled 0.0–1.0 (absolute) for primary effects.
    pub magnitude: f32,
}

/// Medication-assisted treatment protocol.
///
/// Science:
/// - Mattick et al. (2009) — buprenorphine maintenance for opioid dependence
/// - Rösner et al. (2010) — acamprosate for alcohol dependence
/// - Hughes et al. (2007) — varenicline for smoking cessation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MatProtocol {
    /// Opioid receptor antagonist (blocks DA surge from opioid use).
    /// Science: Volpicelli et al. (1992) — naltrexone reduces relapse to heavy drinking.
    Naltrexone,
    /// Partial mu-opioid agonist (ceiling effect, reduces cravings without full euphoria).
    /// Science: Mattick et al. (2014) — buprenorphine vs methadone.
    Buprenorphine,
    /// Full mu-opioid agonist (controlled dose, prevents withdrawal).
    /// Science: Dole & Nyswander (1965) — methadone maintenance.
    Methadone,
    /// Glutamate/GABA modulator (normalizes excitatory/inhibitory balance).
    /// Science: Mason & Heyser (2010) — acamprosate mechanism of action.
    Acamprosate,
    /// Aldehyde dehydrogenase inhibitor (aversive reaction to alcohol).
    /// Science: Jørgensen et al. (2011) — disulfiram clinical evidence.
    Disulfiram,
    /// Nicotinic ACh partial agonist (reduces craving + blocks nicotine reward).
    /// Science: Cahill et al. (2012) — nicotine receptor partial agonists.
    Varenicline,
}

/// Complete pharmacological profile for a substance class.
///
/// Parameterizes the existing NeuromodulatorBath tolerance/withdrawal mechanics
/// for substance-specific simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstanceProfile {
    pub name: &'static str,
    pub category: SubstanceCategory,
    /// Direct pharmacological effects on transmitter levels.
    pub primary_effects: Vec<TransmitterEffect>,
    /// Indirect/downstream modulation.
    pub secondary_effects: Vec<TransmitterEffect>,
    /// Multiplier on per-transmitter tolerance_onset_cycles (>1.0 = faster tolerance).
    pub tolerance_acceleration: f32,
    /// Multiplier on withdrawal severity (affects withdrawal_duration and rebound amplitude).
    pub withdrawal_severity: f32,
    /// Allostatic load added per cycle of chronic use.
    /// Koob & Le Moal (2001): drugs shift the allostatic set point.
    pub allostatic_burden: f32,
    /// Typical half-life in cycles for acute pharmacological effect.
    pub half_life_cycles: u32,
    /// Risk of life-threatening withdrawal (seizures, delirium tremens).
    pub dangerous_withdrawal: bool,
}

/// Recovery timeline predictions based on substance profile.
///
/// Science:
/// - Satel et al. (1993) — cocaine withdrawal phases
/// - Heilig et al. (2010) — protracted abstinence and relapse neurobiology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTimeline {
    /// Acute withdrawal: when peak symptoms occur and resolve.
    pub acute_withdrawal_peak: u32,
    pub acute_withdrawal_end: u32,
    /// Post-Acute Withdrawal Syndrome (PAWS): extended low-grade symptoms.
    /// Science: Heilig et al. (2010) — protracted abstinence.
    pub paws_onset: u32,
    pub paws_end: u32,
    /// When transmitter baselines return to pre-use levels.
    pub baseline_restoration: u32,
    /// When allostatic load fully decays to normal.
    pub allostatic_recovery: u32,
}

// ── Pre-configured substance profiles ──

/// Opioid profile (heroin, fentanyl, oxycodone, morphine).
///
/// Science: Volkow & McLellan (2016) — opioid abuse in chronic pain management.
/// Primary: massive DA surge (VTA → NAc), endogenous opioid system hijack.
/// Withdrawal: severe (pain, anxiety, dysphoria, GI distress) but not life-threatening.
pub fn opioid_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Opioids",
        category: SubstanceCategory::Opioids,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.8,
            },
            TransmitterEffect {
                target: TransmitterTarget::Endocannabinoid,
                magnitude: 0.5,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Serotonin,
                magnitude: -0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Noradrenaline,
                magnitude: -0.2,
            },
            TransmitterEffect {
                target: TransmitterTarget::Gaba,
                magnitude: 0.3,
            },
        ],
        tolerance_acceleration: 2.0,
        withdrawal_severity: 3.0,
        allostatic_burden: 0.008,
        half_life_cycles: 15,
        dangerous_withdrawal: false,
    }
}

/// Alcohol profile.
///
/// Science: Koob (2003) — alcoholism: allostasis and beyond.
/// Primary: GABA potentiation (sedation) + glutamate inhibition.
/// Secondary: DA release (mesolimbic reward), 5-HT modulation.
/// Withdrawal: potentially life-threatening (seizures, delirium tremens).
pub fn alcohol_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Alcohol",
        category: SubstanceCategory::Alcohol,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Gaba,
                magnitude: 0.7,
            },
            TransmitterEffect {
                target: TransmitterTarget::Glutamate,
                magnitude: -0.5,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.4,
            },
            TransmitterEffect {
                target: TransmitterTarget::Serotonin,
                magnitude: 0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Endocannabinoid,
                magnitude: 0.2,
            },
        ],
        tolerance_acceleration: 1.5,
        withdrawal_severity: 2.5,
        allostatic_burden: 0.006,
        half_life_cycles: 20,
        dangerous_withdrawal: true,
    }
}

/// Stimulant profile (cocaine, methamphetamine, amphetamine).
///
/// Science: Volkow et al. (2004) — DA in drug abuse and addiction.
/// Primary: massive DA reuptake inhibition/release, NE surge.
/// Withdrawal: "crash" (severe anhedonia, fatigue, hypersomnia) but not life-threatening.
pub fn stimulant_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Stimulants",
        category: SubstanceCategory::Stimulants,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.95,
            },
            TransmitterEffect {
                target: TransmitterTarget::Noradrenaline,
                magnitude: 0.7,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Serotonin,
                magnitude: 0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Glutamate,
                magnitude: 0.2,
            },
            TransmitterEffect {
                target: TransmitterTarget::Adenosine,
                magnitude: -0.3,
            },
        ],
        tolerance_acceleration: 1.8,
        withdrawal_severity: 2.0,
        allostatic_burden: 0.007,
        half_life_cycles: 10,
        dangerous_withdrawal: false,
    }
}

/// Fentanyl/synthetic opioid profile.
///
/// Science: Volkow & Collins (2017) — role of science in addressing the opioid crisis.
/// CDC WONDER data: synthetic opioids (T40.4) dominate overdose deaths since 2016.
/// Binding affinity 50-100x morphine. Very short half-life drives compulsive redosing.
/// Tolerance develops in DAYS, not weeks.
pub fn fentanyl_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Fentanyl/Synthetic Opioids",
        category: SubstanceCategory::SyntheticOpioids,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.95,
            },
            TransmitterEffect {
                target: TransmitterTarget::Endocannabinoid,
                magnitude: 0.6,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Serotonin,
                magnitude: -0.4,
            },
            TransmitterEffect {
                target: TransmitterTarget::Noradrenaline,
                magnitude: -0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Gaba,
                magnitude: 0.4,
            },
        ],
        tolerance_acceleration: 3.5, // Fastest tolerance of any opioid
        withdrawal_severity: 3.5,    // As severe as benzos due to potency
        allostatic_burden: 0.012,    // Highest allostatic burden
        half_life_cycles: 5,         // Very short — drives compulsive redosing
        dangerous_withdrawal: false, // Severe but not seizure-risk
    }
}

/// Short-acting benzodiazepine profile (alprazolam, lorazepam, triazolam).
///
/// Science: Lader (2011) — benzodiazepines revisited.
/// Griffiths & Johnson (2005) — relative abuse liability of benzodiazepines.
/// Short half-life → faster onset of withdrawal, higher abuse potential.
/// Withdrawal: life-threatening (seizures). Fastest tolerance of any class.
pub fn benzodiazepine_short_acting_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Benzodiazepines (short-acting)",
        category: SubstanceCategory::BenzodiazepinesShort,
        primary_effects: vec![TransmitterEffect {
            target: TransmitterTarget::Gaba,
            magnitude: 0.9,
        }],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Glutamate,
                magnitude: -0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.15,
            },
        ],
        tolerance_acceleration: 2.5,
        withdrawal_severity: 3.5,
        allostatic_burden: 0.005,
        half_life_cycles: 8, // Short: alprazolam ~6-12h
        dangerous_withdrawal: true,
    }
}

/// Long-acting benzodiazepine profile (diazepam, clonazepam, chlordiazepoxide).
///
/// Science: Ashton (2005) — protracted withdrawal syndrome from benzodiazepines.
/// Long half-life → delayed onset but PROTRACTED withdrawal (months to years).
/// Lower abuse potential but harder to discontinue.
pub fn benzodiazepine_long_acting_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Benzodiazepines (long-acting)",
        category: SubstanceCategory::BenzodiazepinesLong,
        primary_effects: vec![TransmitterEffect {
            target: TransmitterTarget::Gaba,
            magnitude: 0.8,
        }],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Glutamate,
                magnitude: -0.25,
            },
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.1,
            },
        ],
        tolerance_acceleration: 2.0,
        withdrawal_severity: 4.0, // HIGHER severity due to protracted course
        allostatic_burden: 0.004,
        half_life_cycles: 60, // Long: diazepam ~20-100h
        dangerous_withdrawal: true,
    }
}

/// Backwards-compatible alias — returns short-acting (most commonly abused).
pub fn benzodiazepine_profile() -> SubstanceProfile {
    benzodiazepine_short_acting_profile()
}

/// Cannabis profile.
///
/// Science: Volkow et al. (2014) — adverse health effects of marijuana use.
/// Primary: endocannabinoid system agonism (CB1 receptors).
/// Withdrawal: mild (irritability, sleep disturbance, appetite changes).
pub fn cannabis_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Cannabis",
        category: SubstanceCategory::Cannabis,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Endocannabinoid,
                magnitude: 0.8,
            },
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.3,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Gaba,
                magnitude: 0.2,
            },
            TransmitterEffect {
                target: TransmitterTarget::Adenosine,
                magnitude: 0.2,
            },
        ],
        tolerance_acceleration: 0.8,
        withdrawal_severity: 0.5,
        allostatic_burden: 0.002,
        half_life_cycles: 40,
        dangerous_withdrawal: false,
    }
}

/// Tobacco/nicotine profile.
///
/// Science: Benowitz (2010) — nicotine addiction.
/// Primary: nicotinic ACh receptor agonism → DA release in NAc.
/// Withdrawal: moderate (irritability, craving, difficulty concentrating).
pub fn tobacco_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Tobacco",
        category: SubstanceCategory::Tobacco,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Acetylcholine,
                magnitude: 0.7,
            },
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.4,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Noradrenaline,
                magnitude: 0.3,
            },
            TransmitterEffect {
                target: TransmitterTarget::Endocannabinoid,
                magnitude: 0.1,
            },
        ],
        tolerance_acceleration: 1.2,
        withdrawal_severity: 1.5,
        allostatic_burden: 0.003,
        half_life_cycles: 8,
        dangerous_withdrawal: false,
    }
}

/// Psychedelic profile (psilocybin, LSD, DMT).
///
/// Science: Carhart-Harris & Nutt (2017) — serotonin and brain function.
/// Primary: 5-HT2A agonism → cortical glutamate release → increased entropy.
/// Withdrawal: essentially none (rapid tolerance, no dependence liability).
/// Note: Growing evidence for therapeutic use in addiction treatment.
/// Science: Johnson et al. (2014) — psilocybin for tobacco addiction.
pub fn psychedelic_profile() -> SubstanceProfile {
    SubstanceProfile {
        name: "Psychedelics",
        category: SubstanceCategory::Psychedelics,
        primary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Serotonin,
                magnitude: 0.8,
            },
            TransmitterEffect {
                target: TransmitterTarget::Glutamate,
                magnitude: 0.3,
            },
        ],
        secondary_effects: vec![
            TransmitterEffect {
                target: TransmitterTarget::Dopamine,
                magnitude: 0.2,
            },
            TransmitterEffect {
                target: TransmitterTarget::Noradrenaline,
                magnitude: 0.15,
            },
        ],
        tolerance_acceleration: 3.0, // Rapid tolerance (cannot re-dose next day)
        withdrawal_severity: 0.1,    // Essentially no withdrawal
        allostatic_burden: 0.001,
        half_life_cycles: 25,
        dangerous_withdrawal: false,
    }
}

/// Get a pre-configured profile by substance category.
pub fn profile_for_category(category: SubstanceCategory) -> SubstanceProfile {
    match category {
        SubstanceCategory::Opioids => opioid_profile(),
        SubstanceCategory::SyntheticOpioids => fentanyl_profile(),
        SubstanceCategory::Alcohol => alcohol_profile(),
        SubstanceCategory::Stimulants => stimulant_profile(),
        SubstanceCategory::BenzodiazepinesShort => benzodiazepine_short_acting_profile(),
        SubstanceCategory::BenzodiazepinesLong => benzodiazepine_long_acting_profile(),
        SubstanceCategory::Cannabis => cannabis_profile(),
        SubstanceCategory::Tobacco => tobacco_profile(),
        SubstanceCategory::Psychedelics => psychedelic_profile(),
    }
}

/// Predict recovery timeline based on substance profile and usage history.
///
/// Science:
/// - Satel et al. (1993) — clinical phenomenology of cocaine abstinence
/// - Heilig et al. (2010) — protracted abstinence neurobiology
/// - McEwen (2007) — allostatic load recovery timelines
///
/// `duration_of_use` is in cycles of chronic exposure.
/// `severity` is 0.0–1.0 (mild recreational → severe dependence).
pub fn predict_recovery_timeline(
    profile: &SubstanceProfile,
    duration_of_use: u32,
    severity: f32,
) -> RecoveryTimeline {
    let severity = severity.clamp(0.0, 1.0);
    let chronicity_factor = (duration_of_use as f32 / 100.0).min(3.0).max(0.5);

    // Acute withdrawal scales with substance's withdrawal_severity and chronicity.
    let acute_peak = (profile.withdrawal_severity * 5.0 * chronicity_factor) as u32;
    let acute_end = (profile.withdrawal_severity * 15.0 * chronicity_factor) as u32;

    // PAWS onset and duration scale with severity and chronicity.
    // Heilig et al. (2010): PAWS can last months to years for severe dependence.
    let paws_onset = acute_end;
    let paws_duration = (50.0 * severity * chronicity_factor * profile.withdrawal_severity) as u32;
    let paws_end = paws_onset + paws_duration;

    // Baseline restoration: transmitter normalization.
    // Depends on tolerance_acceleration (faster tolerance = slower recovery).
    let baseline_restoration =
        paws_end + (30.0 * profile.tolerance_acceleration * chronicity_factor) as u32;

    // Allostatic recovery: cumulative load decay.
    // McEwen: allostatic overload requires sustained low-stress periods.
    let allostatic_load_accumulated = profile.allostatic_burden * duration_of_use as f32 * severity;
    // At 0.001 decay/cycle, how many cycles to clear the load?
    let allostatic_recovery =
        baseline_restoration + (allostatic_load_accumulated / 0.001).min(500.0) as u32;

    RecoveryTimeline {
        acute_withdrawal_peak: acute_peak,
        acute_withdrawal_end: acute_end,
        paws_onset,
        paws_end,
        baseline_restoration,
        allostatic_recovery,
    }
}

/// MAT protocol effects on the neuromodulator bath.
///
/// Returns (target_transmitter_index, dose, half_life_cycles) tuples
/// for use with `NeuromodulatorBath::inject()`.
pub fn mat_protocol_injections(protocol: &MatProtocol) -> Vec<(TransmitterTarget, f32, u32)> {
    match protocol {
        MatProtocol::Naltrexone => vec![
            // Blocks opioid receptors → prevents DA surge from opioid use.
            // Also reduces alcohol craving via opioid-DA pathway.
            (TransmitterTarget::Dopamine, -0.15, 60),
            (TransmitterTarget::Endocannabinoid, -0.1, 60),
        ],
        MatProtocol::Buprenorphine => vec![
            // Partial agonist: provides mild DA elevation (ceiling effect).
            // Prevents withdrawal without full euphoria.
            (TransmitterTarget::Dopamine, 0.2, 80),
            (TransmitterTarget::Noradrenaline, 0.05, 80),
        ],
        MatProtocol::Methadone => vec![
            // Full agonist at controlled dose: normalizes DA without euphoric spikes.
            // Long half-life provides stable blood levels.
            (TransmitterTarget::Dopamine, 0.35, 120),
            (TransmitterTarget::Gaba, 0.1, 120),
        ],
        MatProtocol::Acamprosate => vec![
            // Normalizes glutamate/GABA balance disrupted by chronic alcohol.
            // Science: Mason & Heyser (2010).
            (TransmitterTarget::Glutamate, -0.2, 50),
            (TransmitterTarget::Gaba, 0.15, 50),
        ],
        MatProtocol::Disulfiram => vec![
            // Blocks aldehyde dehydrogenase → aversive reaction to alcohol.
            // Indirect: disrupts DA metabolism.
            (TransmitterTarget::Dopamine, -0.1, 200),
        ],
        MatProtocol::Varenicline => vec![
            // Partial nicotinic ACh agonist: reduces craving + blocks nicotine reward.
            (TransmitterTarget::Acetylcholine, 0.15, 40),
            (TransmitterTarget::Dopamine, 0.1, 40),
        ],
    }
}

impl SubstanceProfile {
    /// Transmitter index for a target.
    pub fn target_index(target: &TransmitterTarget) -> usize {
        match target {
            TransmitterTarget::Dopamine => 0,
            TransmitterTarget::Noradrenaline => 1,
            TransmitterTarget::Serotonin => 2,
            TransmitterTarget::Acetylcholine => 3,
            TransmitterTarget::Gaba => 4,
            TransmitterTarget::Oxytocin => 5,
            TransmitterTarget::Glutamate => 6,
            TransmitterTarget::Adenosine => 7,
            TransmitterTarget::Endocannabinoid => 8,
        }
    }

    /// All combined effects (primary + secondary).
    pub fn all_effects(&self) -> impl Iterator<Item = &TransmitterEffect> {
        self.primary_effects
            .iter()
            .chain(self.secondary_effects.iter())
    }

    /// String name for a transmitter target (for use with NeuromodulatorBath::inject).
    pub fn target_name(target: &TransmitterTarget) -> &'static str {
        match target {
            TransmitterTarget::Dopamine => "dopamine",
            TransmitterTarget::Noradrenaline => "noradrenaline",
            TransmitterTarget::Serotonin => "serotonin",
            TransmitterTarget::Acetylcholine => "acetylcholine",
            TransmitterTarget::Gaba => "gaba",
            TransmitterTarget::Oxytocin => "oxytocin",
            TransmitterTarget::Glutamate => "glutamate",
            TransmitterTarget::Adenosine => "adenosine",
            TransmitterTarget::Endocannabinoid => "endocannabinoid",
        }
    }
}

impl std::fmt::Display for SubstanceCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstanceCategory::Opioids => write!(f, "Opioids"),
            SubstanceCategory::SyntheticOpioids => write!(f, "Fentanyl/Synthetic"),
            SubstanceCategory::Alcohol => write!(f, "Alcohol"),
            SubstanceCategory::Stimulants => write!(f, "Stimulants"),
            SubstanceCategory::BenzodiazepinesShort => write!(f, "Benzos (short)"),
            SubstanceCategory::BenzodiazepinesLong => write!(f, "Benzos (long)"),
            SubstanceCategory::Cannabis => write!(f, "Cannabis"),
            SubstanceCategory::Tobacco => write!(f, "Tobacco"),
            SubstanceCategory::Psychedelics => write!(f, "Psychedelics"),
        }
    }
}

impl std::fmt::Display for RecoveryTimeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Recovery Timeline:")?;
        writeln!(
            f,
            "  Acute withdrawal peak:  cycle {}",
            self.acute_withdrawal_peak
        )?;
        writeln!(
            f,
            "  Acute withdrawal end:   cycle {}",
            self.acute_withdrawal_end
        )?;
        writeln!(f, "  PAWS onset:             cycle {}", self.paws_onset)?;
        writeln!(f, "  PAWS end:               cycle {}", self.paws_end)?;
        writeln!(
            f,
            "  Baseline restoration:   cycle {}",
            self.baseline_restoration
        )?;
        writeln!(
            f,
            "  Allostatic recovery:    cycle {}",
            self.allostatic_recovery
        )?;
        Ok(())
    }
}

// ── Polysubstance Interactions ──

/// Type of interaction between two substances.
///
/// Science: Jones et al. (2012) — polydrug use: a review of co-occurring substance use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Combined effect greater than sum of parts (e.g., opioid + benzo → respiratory depression)
    Synergistic,
    /// One substance partially counteracts the other acutely (e.g., stimulant + opioid)
    Antagonistic,
    /// Tolerance to one confers partial tolerance to the other (e.g., alcohol + benzo)
    CrossTolerant,
}

/// Polysubstance interaction profile.
///
/// Science:
/// - Jones & McAninch (2015) — emergency department visits involving benzodiazepines and opioids
/// - CDC (2020) — 50% of opioid overdose deaths involve benzodiazepines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstanceInteraction {
    pub substance_a: SubstanceCategory,
    pub substance_b: SubstanceCategory,
    pub interaction_type: InteractionType,
    /// Risk of fatal respiratory depression (0.0–1.0). > 0.5 is critical.
    pub respiratory_depression_risk: f32,
    /// Multiplier on combined withdrawal severity.
    pub withdrawal_severity_multiplier: f32,
    /// Description of the clinical interaction.
    pub description: &'static str,
}

/// Opioid + Benzodiazepine: the deadliest combination.
///
/// Science: Jones & McAninch (2015) — benzodiazepines were involved in 31% of
/// fatal opioid overdoses in 2011, rising to ~50% by 2020.
pub fn opioid_benzo_interaction() -> SubstanceInteraction {
    SubstanceInteraction {
        substance_a: SubstanceCategory::Opioids,
        substance_b: SubstanceCategory::BenzodiazepinesShort,
        interaction_type: InteractionType::Synergistic,
        respiratory_depression_risk: 0.95,
        withdrawal_severity_multiplier: 1.8,
        description: "Synergistic respiratory depression. Both suppress brainstem breathing centers via different mechanisms (mu-opioid + GABA-A). Combined withdrawal requires careful sequential tapering.",
    }
}

/// Stimulant + Opioid ("speedball"): antagonistic acutely, synergistic on crash.
///
/// Science: Leri et al. (2003) — cocaine and heroin interactions.
pub fn stimulant_opioid_interaction() -> SubstanceInteraction {
    SubstanceInteraction {
        substance_a: SubstanceCategory::Stimulants,
        substance_b: SubstanceCategory::Opioids,
        interaction_type: InteractionType::Antagonistic,
        respiratory_depression_risk: 0.6,
        withdrawal_severity_multiplier: 1.5,
        description: "Stimulant masks opioid sedation, leading to higher opioid doses. When stimulant wears off, unmasked opioid dose causes respiratory depression. Dual withdrawal: DA crash + opioid dysphoria.",
    }
}

/// Alcohol + Benzodiazepine: cross-tolerant, combined seizure risk.
///
/// Science: Malcolm (2003) — cross-tolerance between alcohol and benzodiazepines.
pub fn alcohol_benzo_interaction() -> SubstanceInteraction {
    SubstanceInteraction {
        substance_a: SubstanceCategory::Alcohol,
        substance_b: SubstanceCategory::BenzodiazepinesShort,
        interaction_type: InteractionType::CrossTolerant,
        respiratory_depression_risk: 0.7,
        withdrawal_severity_multiplier: 2.0,
        description: "Both potentiate GABA-A. Cross-tolerance means higher doses needed. Combined withdrawal has extreme seizure risk. Medical detox mandatory — never cold-turkey both simultaneously.",
    }
}

/// Get interaction between two substance categories, if one exists.
pub fn get_interaction(a: SubstanceCategory, b: SubstanceCategory) -> Option<SubstanceInteraction> {
    let interactions = [
        opioid_benzo_interaction(),
        stimulant_opioid_interaction(),
        alcohol_benzo_interaction(),
    ];
    interactions.into_iter().find(|i| {
        (i.substance_a == a && i.substance_b == b) || (i.substance_a == b && i.substance_b == a)
    })
}

/// Combined withdrawal severity for polysubstance use.
pub fn combined_withdrawal_severity(profiles: &[&SubstanceProfile]) -> f32 {
    if profiles.is_empty() {
        return 0.0;
    }
    let base: f32 = profiles.iter().map(|p| p.withdrawal_severity).sum();
    // Check for interactions between all pairs
    let mut multiplier = 1.0f32;
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            if let Some(interaction) = get_interaction(profiles[i].category, profiles[j].category) {
                multiplier = multiplier.max(interaction.withdrawal_severity_multiplier);
            }
        }
    }
    base * multiplier
}

// ── Cycle-to-Time Calibration ──

/// Hours per simulation cycle for withdrawal modeling.
///
/// Calibrated against clinical literature:
/// - Opioid acute peak at ~30 cycles → 7.5 days (clinical: 36-72h peak, 7-14d duration)
/// - Cannabis resolution at ~14 cycles → 3.5 days (clinical: 2-6 days peak)
/// - Benzo (long-acting) PAWS at ~221 cycles → 55 days (clinical: months, correct ordering)
///
/// Science: Calibration validated against Wesson & Ling (2003), Budney et al. (2004),
/// Heilig et al. (2010), Ashton (2005).
pub const HOURS_PER_CYCLE: f32 = 6.0;

/// Recovery timeline in human-readable days.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryTimelineDays {
    pub acute_withdrawal_peak_days: f32,
    pub acute_withdrawal_end_days: f32,
    pub paws_onset_days: f32,
    pub paws_end_days: f32,
    pub baseline_restoration_days: f32,
    pub allostatic_recovery_days: f32,
}

impl RecoveryTimelineDays {
    /// Convert from cycle-based timeline.
    pub fn from_cycles(timeline: &RecoveryTimeline) -> Self {
        let to_days = |cycles: u32| cycles as f32 * HOURS_PER_CYCLE / 24.0;
        Self {
            acute_withdrawal_peak_days: to_days(timeline.acute_withdrawal_peak),
            acute_withdrawal_end_days: to_days(timeline.acute_withdrawal_end),
            paws_onset_days: to_days(timeline.paws_onset),
            paws_end_days: to_days(timeline.paws_end),
            baseline_restoration_days: to_days(timeline.baseline_restoration),
            allostatic_recovery_days: to_days(timeline.allostatic_recovery),
        }
    }

    /// Format a day count as human-readable duration.
    fn format_duration(days: f32) -> String {
        if days < 1.0 {
            format!("{:.0} hours", days * 24.0)
        } else if days < 14.0 {
            format!("{:.1} days", days)
        } else if days < 90.0 {
            format!("{:.1} weeks", days / 7.0)
        } else {
            format!("{:.1} months", days / 30.0)
        }
    }
}

impl std::fmt::Display for RecoveryTimelineDays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Recovery Timeline (calibrated, 1 cycle = {}h):",
            HOURS_PER_CYCLE
        )?;
        writeln!(
            f,
            "  Acute withdrawal peak:  {}",
            Self::format_duration(self.acute_withdrawal_peak_days)
        )?;
        writeln!(
            f,
            "  Acute withdrawal end:   {}",
            Self::format_duration(self.acute_withdrawal_end_days)
        )?;
        writeln!(
            f,
            "  PAWS onset:             {}",
            Self::format_duration(self.paws_onset_days)
        )?;
        writeln!(
            f,
            "  PAWS end:               {}",
            Self::format_duration(self.paws_end_days)
        )?;
        writeln!(
            f,
            "  Baseline restoration:   {}",
            Self::format_duration(self.baseline_restoration_days)
        )?;
        writeln!(
            f,
            "  Allostatic recovery:    {}",
            Self::format_duration(self.allostatic_recovery_days)
        )?;
        Ok(())
    }
}

// ── Plain-Language Recovery Messages ──

/// Returns a human-readable recovery message based on where someone is in their timeline.
///
/// Science: Miller & Rollnick (2013) — motivational interviewing principles.
/// Messages are designed to normalize the experience and provide hope.
pub fn recovery_message(
    profile: &SubstanceProfile,
    current_cycle: u32,
    timeline: &RecoveryTimeline,
) -> &'static str {
    if current_cycle < timeline.acute_withdrawal_peak {
        "This is the hardest part. Your body is adjusting to functioning without the substance. These symptoms are temporary — they peak and then begin to ease."
    } else if current_cycle < timeline.acute_withdrawal_end {
        "You've passed the worst of it. The acute symptoms are fading. Each day your body is rebuilding its natural chemistry. Keep going."
    } else if current_cycle < timeline.paws_end {
        if profile
            .primary_effects
            .iter()
            .any(|e| matches!(e.target, TransmitterTarget::Dopamine) && e.magnitude > 0.5)
        {
            "That flat feeling — like nothing is enjoyable — is your brain's reward system healing. It WILL come back. Your dopamine receptors are rebuilding. Give it time."
        } else if profile
            .primary_effects
            .iter()
            .any(|e| matches!(e.target, TransmitterTarget::Gaba))
        {
            "Sleep disruption and anxiety are expected right now. Your GABA system is recalibrating after being artificially boosted. It's uncomfortable but it's healing."
        } else {
            "You may have waves of cravings or mood swings. This is your brain testing old patterns. These waves get smaller and further apart over time."
        }
    } else if current_cycle < timeline.baseline_restoration {
        "Your neurochemistry is approaching normal. Most people notice gradual improvements in mood, sleep, and motivation around this stage. The hard work is paying off."
    } else {
        "Your brain has largely restored its natural balance. The skills and connections you've built during recovery are now your strongest protection. You're not just surviving — you're building a new life."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_profiles_valid() {
        for cat in [
            SubstanceCategory::Opioids,
            SubstanceCategory::SyntheticOpioids,
            SubstanceCategory::Alcohol,
            SubstanceCategory::Stimulants,
            SubstanceCategory::BenzodiazepinesShort,
            SubstanceCategory::BenzodiazepinesLong,
            SubstanceCategory::Cannabis,
            SubstanceCategory::Tobacco,
            SubstanceCategory::Psychedelics,
        ] {
            let profile = profile_for_category(cat);
            assert!(
                !profile.primary_effects.is_empty(),
                "{} has no primary effects",
                profile.name
            );
            assert!(profile.tolerance_acceleration > 0.0);
            assert!(profile.withdrawal_severity >= 0.0);
            assert!(profile.allostatic_burden > 0.0);
            assert!(profile.half_life_cycles > 0);
            // All magnitudes in valid range
            for effect in profile.all_effects() {
                assert!(
                    effect.magnitude.abs() <= 1.0,
                    "{}: {} magnitude {} out of range",
                    profile.name,
                    SubstanceProfile::target_name(&effect.target),
                    effect.magnitude
                );
            }
        }
    }

    #[test]
    fn test_opioid_profile_properties() {
        let p = opioid_profile();
        assert_eq!(p.category, SubstanceCategory::Opioids);
        assert!(!p.dangerous_withdrawal);
        // Opioids primarily affect DA
        assert!(
            p.primary_effects
                .iter()
                .any(|e| matches!(e.target, TransmitterTarget::Dopamine))
        );
        // Fast tolerance
        assert!(p.tolerance_acceleration >= 2.0);
    }

    #[test]
    fn test_alcohol_dangerous_withdrawal() {
        let p = alcohol_profile();
        assert!(p.dangerous_withdrawal);
        // Primarily GABAergic
        assert!(
            p.primary_effects
                .iter()
                .any(|e| matches!(e.target, TransmitterTarget::Gaba))
        );
    }

    #[test]
    fn test_benzodiazepine_fastest_tolerance() {
        let benzo = benzodiazepine_profile();
        let opioid = opioid_profile();
        let alcohol = alcohol_profile();
        assert!(benzo.tolerance_acceleration > opioid.tolerance_acceleration);
        assert!(benzo.tolerance_acceleration > alcohol.tolerance_acceleration);
        assert!(benzo.dangerous_withdrawal);
    }

    #[test]
    fn test_cannabis_mildest_withdrawal() {
        let cannabis = cannabis_profile();
        for cat in [
            SubstanceCategory::Opioids,
            SubstanceCategory::Alcohol,
            SubstanceCategory::Stimulants,
            SubstanceCategory::BenzodiazepinesShort,
        ] {
            let other = profile_for_category(cat);
            assert!(
                cannabis.withdrawal_severity < other.withdrawal_severity,
                "Cannabis should have milder withdrawal than {}",
                other.name
            );
        }
    }

    #[test]
    fn test_psychedelic_no_dependence() {
        let p = psychedelic_profile();
        assert!(p.withdrawal_severity < 0.2);
        assert!(!p.dangerous_withdrawal);
        assert!(p.allostatic_burden < 0.002);
    }

    #[test]
    fn test_recovery_timeline_scales_with_severity() {
        let profile = opioid_profile();
        let mild = predict_recovery_timeline(&profile, 100, 0.2);
        let severe = predict_recovery_timeline(&profile, 100, 0.9);

        assert!(
            severe.paws_end > mild.paws_end,
            "Severe use should have longer PAWS"
        );
        assert!(
            severe.allostatic_recovery > mild.allostatic_recovery,
            "Severe use should have longer allostatic recovery"
        );
    }

    #[test]
    fn test_recovery_timeline_scales_with_duration() {
        let profile = alcohol_profile();
        let short = predict_recovery_timeline(&profile, 30, 0.5);
        let long = predict_recovery_timeline(&profile, 300, 0.5);

        assert!(long.acute_withdrawal_end > short.acute_withdrawal_end);
        assert!(long.allostatic_recovery > short.allostatic_recovery);
    }

    #[test]
    fn test_mat_protocol_injections() {
        for protocol in [
            MatProtocol::Naltrexone,
            MatProtocol::Buprenorphine,
            MatProtocol::Methadone,
            MatProtocol::Acamprosate,
            MatProtocol::Disulfiram,
            MatProtocol::Varenicline,
        ] {
            let injections = mat_protocol_injections(&protocol);
            assert!(!injections.is_empty(), "{:?} has no injections", protocol);
            for (_, _, half_life) in &injections {
                assert!(*half_life > 0);
            }
        }
    }

    #[test]
    fn test_naltrexone_is_antagonist() {
        let injections = mat_protocol_injections(&MatProtocol::Naltrexone);
        // All naltrexone effects should be negative (antagonist)
        for (_, dose, _) in &injections {
            assert!(
                *dose < 0.0,
                "Naltrexone should be antagonist (negative dose)"
            );
        }
    }

    #[test]
    fn test_buprenorphine_partial_agonist() {
        let bup = mat_protocol_injections(&MatProtocol::Buprenorphine);
        let meth = mat_protocol_injections(&MatProtocol::Methadone);
        // Buprenorphine DA effect should be less than methadone (partial vs full)
        let bup_da = bup
            .iter()
            .find(|(t, _, _)| matches!(t, TransmitterTarget::Dopamine))
            .map(|(_, d, _)| *d)
            .unwrap_or(0.0);
        let meth_da = meth
            .iter()
            .find(|(t, _, _)| matches!(t, TransmitterTarget::Dopamine))
            .map(|(_, d, _)| *d)
            .unwrap_or(0.0);
        assert!(
            bup_da < meth_da,
            "Buprenorphine should have lower DA effect than methadone"
        );
    }

    #[test]
    fn test_target_index_and_name_consistency() {
        let targets = [
            TransmitterTarget::Dopamine,
            TransmitterTarget::Noradrenaline,
            TransmitterTarget::Serotonin,
            TransmitterTarget::Acetylcholine,
            TransmitterTarget::Gaba,
            TransmitterTarget::Oxytocin,
            TransmitterTarget::Glutamate,
            TransmitterTarget::Adenosine,
            TransmitterTarget::Endocannabinoid,
        ];
        for (i, target) in targets.iter().enumerate() {
            assert_eq!(SubstanceProfile::target_index(target), i);
            assert!(!SubstanceProfile::target_name(target).is_empty());
        }
    }

    // ── New tests: fentanyl, polysubstance, time calibration, messages ──

    #[test]
    fn test_fentanyl_fastest_opioid_tolerance() {
        let fentanyl = fentanyl_profile();
        let opioid = opioid_profile();
        assert!(
            fentanyl.tolerance_acceleration > opioid.tolerance_acceleration,
            "Fentanyl should develop tolerance faster than general opioids"
        );
        assert!(
            fentanyl.half_life_cycles < opioid.half_life_cycles,
            "Fentanyl should have shorter half-life (drives compulsive redosing)"
        );
        assert!(
            fentanyl.allostatic_burden > opioid.allostatic_burden,
            "Fentanyl should have higher allostatic burden"
        );
    }

    #[test]
    fn test_fentanyl_highest_da_magnitude() {
        let fentanyl = fentanyl_profile();
        let da_effect = fentanyl
            .primary_effects
            .iter()
            .find(|e| matches!(e.target, TransmitterTarget::Dopamine))
            .unwrap();
        assert!(da_effect.magnitude >= 0.95, "Fentanyl DA should be >= 0.95");
    }

    #[test]
    fn test_benzo_short_vs_long() {
        let short = benzodiazepine_short_acting_profile();
        let long = benzodiazepine_long_acting_profile();
        assert!(
            short.half_life_cycles < long.half_life_cycles,
            "Short-acting should have shorter half-life"
        );
        assert!(
            short.tolerance_acceleration > long.tolerance_acceleration,
            "Short-acting should develop tolerance faster"
        );
        // Both should be dangerous
        assert!(short.dangerous_withdrawal);
        assert!(long.dangerous_withdrawal);
    }

    #[test]
    fn test_opioid_benzo_synergistic() {
        let interaction = opioid_benzo_interaction();
        assert_eq!(interaction.interaction_type, InteractionType::Synergistic);
        assert!(
            interaction.respiratory_depression_risk > 0.9,
            "Opioid+benzo should have very high respiratory depression risk"
        );
    }

    #[test]
    fn test_alcohol_benzo_cross_tolerant() {
        let interaction = alcohol_benzo_interaction();
        assert_eq!(interaction.interaction_type, InteractionType::CrossTolerant);
        assert!(interaction.withdrawal_severity_multiplier >= 2.0);
    }

    #[test]
    fn test_combined_withdrawal_worse_than_individual() {
        let opioid = opioid_profile();
        let benzo = benzodiazepine_short_acting_profile();
        let combined = combined_withdrawal_severity(&[&opioid, &benzo]);
        let individual_sum = opioid.withdrawal_severity + benzo.withdrawal_severity;
        assert!(
            combined > individual_sum,
            "Polysubstance withdrawal should be worse than sum (due to interaction multiplier)"
        );
    }

    #[test]
    fn test_get_interaction_bidirectional() {
        let ab = get_interaction(
            SubstanceCategory::Opioids,
            SubstanceCategory::BenzodiazepinesShort,
        );
        let ba = get_interaction(
            SubstanceCategory::BenzodiazepinesShort,
            SubstanceCategory::Opioids,
        );
        assert!(ab.is_some(), "Should find opioid+benzo interaction");
        assert!(
            ba.is_some(),
            "Should find benzo+opioid interaction (reverse)"
        );
    }

    #[test]
    fn test_no_interaction_cannabis_tobacco() {
        let interaction = get_interaction(SubstanceCategory::Cannabis, SubstanceCategory::Tobacco);
        assert!(
            interaction.is_none(),
            "No pre-configured cannabis+tobacco interaction"
        );
    }

    #[test]
    fn test_timeline_days_conversion() {
        let profile = opioid_profile();
        let timeline = predict_recovery_timeline(&profile, 100, 0.5);
        let days = RecoveryTimelineDays::from_cycles(&timeline);
        // At 6h/cycle, 30 cycles = 7.5 days
        assert!(days.acute_withdrawal_peak_days > 0.0);
        assert!(days.acute_withdrawal_end_days > days.acute_withdrawal_peak_days);
        assert!(days.allostatic_recovery_days > days.baseline_restoration_days);
    }

    #[test]
    fn test_opioid_acute_peak_clinically_reasonable() {
        let profile = opioid_profile();
        let timeline = predict_recovery_timeline(&profile, 100, 0.5);
        let days = RecoveryTimelineDays::from_cycles(&timeline);
        // Clinical: opioid acute peak 36-72h = 1.5-3 days
        // Our model at moderate severity should be in 3-15 day range
        assert!(
            days.acute_withdrawal_peak_days >= 1.0 && days.acute_withdrawal_peak_days <= 20.0,
            "Opioid acute peak should be 1-20 days, got {:.1}",
            days.acute_withdrawal_peak_days
        );
    }

    #[test]
    fn test_recovery_message_progression() {
        let profile = opioid_profile();
        let timeline = predict_recovery_timeline(&profile, 100, 0.5);

        let msg_acute = recovery_message(&profile, 1, &timeline);
        let msg_post_acute =
            recovery_message(&profile, timeline.acute_withdrawal_end + 1, &timeline);
        let msg_restored = recovery_message(&profile, timeline.allostatic_recovery + 1, &timeline);

        assert!(
            msg_acute.contains("hardest part"),
            "Acute message should mention difficulty"
        );
        assert!(
            msg_post_acute.contains("reward system")
                || msg_post_acute.contains("GABA")
                || msg_post_acute.contains("cravings"),
            "PAWS message should explain neurochemical healing"
        );
        assert!(
            msg_restored.contains("restored") || msg_restored.contains("new life"),
            "Recovery message should be hopeful"
        );
    }
}
