// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Medical Isotope Physics
//!
//! Comprehensive module for medical isotope science:
//!
//! - **Database**: Clinically-used radioisotopes with decay, clinical, and production data
//! - **Theranostic pairs**: Imaging + therapy isotopes of the same element
//! - **Production routes**: Q-value computation for (n,gamma), (p,n), and generator systems
//! - **Novel candidates**: ML-driven search for unmeasured isotopes with therapeutic potential
//!
//! All binding energies computed via [`MlMassPredictor`] (DZ + Random Forest on AME2020 residuals).
//!
//! ## References
//!
//! - Qaim, S. M. (2017). Nuclear data for production and medical application of radionuclides.
//!   *J. Radioanalytical and Nuclear Chemistry*, 305, 535-546.
//! - Kondev, F. G. et al. (2021). NUBASE2020 evaluation. *Chinese Physics C*, 45(3), 030001.
//! - Sgouros, G. et al. (2020). Radiopharmaceutical therapy in cancer. *Nature Reviews Drug Discovery*.

use crate::ml_mass::MlMassPredictor;
use serde::{Deserialize, Serialize};

// ── Physical constants ───────────────────────────────────────────────────────

/// Neutron mass excess (MeV) — AME2020
const M_N: f64 = 8.07132;
/// Hydrogen atom mass excess (MeV) — AME2020
const M_H: f64 = 7.28897;
/// He-4 binding energy (MeV)
const BE_HE4: f64 = 28.296;
/// Avogadro constant (mol^-1)
const N_A: f64 = 6.022_140_76e23;
/// ln(2)
const LN2: f64 = 0.693_147_180_559_945_3;

// ── Enums ────────────────────────────────────────────────────────────────────

/// Radioactive decay mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecayMode {
    /// Beta-minus (neutron -> proton + e- + antineutrino)
    BetaMinus,
    /// Beta-plus / electron capture (proton -> neutron + e+ + neutrino)
    BetaPlusEC,
    /// Alpha decay (emits He-4)
    Alpha,
    /// Isomeric transition (gamma emission from metastable state)
    IsomericTransition,
    /// Double beta-minus (extremely rare, included for completeness)
    DoubleBetaMinus,
}

impl std::fmt::Display for DecayMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecayMode::BetaMinus => write!(f, "beta-"),
            DecayMode::BetaPlusEC => write!(f, "beta+/EC"),
            DecayMode::Alpha => write!(f, "alpha"),
            DecayMode::IsomericTransition => write!(f, "IT"),
            DecayMode::DoubleBetaMinus => write!(f, "2beta-"),
        }
    }
}

/// Clinical imaging/therapy modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClinicalModality {
    /// Positron Emission Tomography
    PET,
    /// Single-Photon Emission Computed Tomography
    SPECT,
    /// Beta-minus therapy (external beam or targeted)
    BetaTherapy,
    /// Alpha-particle therapy (targeted alpha therapy, TAT)
    AlphaTherapy,
    /// Auger electron therapy (intranuclear / DNA-targeted)
    AugerTherapy,
    /// Brachytherapy (sealed-source implants)
    Brachytherapy,
}

impl ClinicalModality {
    /// Is this an imaging modality?
    pub fn is_imaging(&self) -> bool {
        matches!(self, ClinicalModality::PET | ClinicalModality::SPECT)
    }

    /// Is this a therapeutic modality?
    pub fn is_therapy(&self) -> bool {
        !self.is_imaging()
    }
}

impl std::fmt::Display for ClinicalModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClinicalModality::PET => write!(f, "PET"),
            ClinicalModality::SPECT => write!(f, "SPECT"),
            ClinicalModality::BetaTherapy => write!(f, "beta-therapy"),
            ClinicalModality::AlphaTherapy => write!(f, "alpha-therapy"),
            ClinicalModality::AugerTherapy => write!(f, "Auger-therapy"),
            ClinicalModality::Brachytherapy => write!(f, "brachytherapy"),
        }
    }
}

/// Production route for a medical isotope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProductionRoute {
    /// Reactor: neutron capture (n,gamma)
    NeutronCapture {
        /// Target isotope description (e.g. "Mo-98")
        target: String,
        target_z: u16,
        target_n: u16,
    },
    /// Cyclotron: proton bombardment (p,n)
    ProtonBombardment {
        target: String,
        target_z: u16,
        target_n: u16,
    },
    /// Generator: parent -> daughter via decay
    Generator {
        parent: String,
        parent_z: u16,
        parent_n: u16,
        parent_half_life_s: f64,
    },
    /// Cyclotron: deuteron bombardment (d,n)
    DeuteronBombardment {
        target: String,
        target_z: u16,
        target_n: u16,
    },
    /// Cyclotron: (p,alpha)
    ProtonAlpha {
        target: String,
        target_z: u16,
        target_n: u16,
    },
    /// Spallation or other exotic production
    Spallation { description: String },
}

// ── Core data structures ─────────────────────────────────────────────────────

/// Complete record for a clinically-used medical isotope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalIsotope {
    /// Element symbol (e.g. "Tc")
    pub symbol: String,
    /// Isotope name (e.g. "Tc-99m")
    pub name: String,
    /// Proton number
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Mass number A = Z + N
    pub a: u16,
    /// Primary decay mode
    pub decay_mode: DecayMode,
    /// Half-life in seconds
    pub half_life_s: f64,
    /// Experimental Q-value (MeV) — from NUBASE/NNDC
    pub q_value_mev: f64,
    /// Daughter isotope description
    pub daughter: String,
    /// Clinical modality
    pub modality: ClinicalModality,
    /// Target organ/disease description
    pub clinical_use: String,
    /// Primary production route
    pub production: ProductionRoute,
}

impl MedicalIsotope {
    /// Decay constant lambda = ln(2) / t_{1/2} (s^-1)
    pub fn decay_constant(&self) -> f64 {
        LN2 / self.half_life_s
    }

    /// Specific activity: SA = lambda * N_A / A (Bq/mol)
    pub fn specific_activity_bq_per_mol(&self) -> f64 {
        self.decay_constant() * N_A / self.a as f64
    }

    /// Specific activity in GBq/g
    pub fn specific_activity_gbq_per_g(&self) -> f64 {
        self.specific_activity_bq_per_mol() / (self.a as f64) / 1e9
    }
}

/// A theranostic pair: one imaging isotope + one therapy isotope of the same element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheranosticPair {
    pub element: String,
    pub z: u16,
    pub imaging_isotope: String,
    pub imaging_a: u16,
    pub imaging_modality: ClinicalModality,
    pub therapy_isotope: String,
    pub therapy_a: u16,
    pub therapy_modality: ClinicalModality,
    /// Whether this is a known clinical pair or ML-predicted candidate
    pub is_known: bool,
    /// Notes on the pairing
    pub notes: String,
}

/// Production route analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionAnalysis {
    pub isotope_name: String,
    pub route_description: String,
    /// Q-value computed from ML binding energies (MeV)
    pub q_value_ml: f64,
    /// Uncertainty in Q-value from ML (MeV)
    pub q_uncertainty: f64,
    /// Whether reaction is exothermic (Q > 0)
    pub is_exothermic: bool,
    /// Specific activity of product (GBq/g)
    pub specific_activity_gbq_per_g: f64,
    /// For generators: equilibrium type
    pub equilibrium: Option<GeneratorEquilibrium>,
}

/// Generator equilibrium classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratorEquilibrium {
    /// Parent t1/2 >> daughter t1/2
    Secular,
    /// Parent t1/2 > daughter t1/2 (but not >>)
    Transient,
    /// Parent t1/2 < daughter t1/2 (no equilibrium)
    NoEquilibrium,
}

impl std::fmt::Display for GeneratorEquilibrium {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneratorEquilibrium::Secular => write!(f, "secular"),
            GeneratorEquilibrium::Transient => write!(f, "transient"),
            GeneratorEquilibrium::NoEquilibrium => write!(f, "no equilibrium"),
        }
    }
}

/// A novel isotope candidate identified by ML scanning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovelCandidate {
    pub symbol: String,
    pub z: u16,
    pub n: u16,
    pub a: u16,
    /// ML-predicted binding energy (MeV)
    pub binding_energy: f64,
    /// ML uncertainty (MeV)
    pub uncertainty: f64,
    /// Predicted Q-value for the relevant decay (MeV)
    pub q_value: f64,
    /// Estimated half-life category
    pub half_life_category: String,
    /// Suggested modality based on decay properties
    pub suggested_modality: ClinicalModality,
    /// Why this candidate is interesting
    pub rationale: String,
    /// Daughter-toxicity note (empty if no concern)
    pub daughter_toxicity_note: String,
}

// ── Medical isotope database ─────────────────────────────────────────────────

/// Helper: seconds from human-readable time units.
const fn hours(h: f64) -> f64 {
    h * 3600.0
}
const fn days(d: f64) -> f64 {
    d * 86400.0
}
const fn minutes(m: f64) -> f64 {
    m * 60.0
}
const fn years(y: f64) -> f64 {
    y * 365.25 * 86400.0
}

/// Build the comprehensive medical isotope database.
///
/// All values from NUBASE2020, IAEA Nuclear Data Services, and NNDC BNL.
pub fn medical_isotope_database() -> Vec<MedicalIsotope> {
    vec![
        // ── PET isotopes ─────────────────────────────────────────────────
        MedicalIsotope {
            symbol: "F".into(),
            name: "F-18".into(),
            z: 9,
            n: 9,
            a: 18,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: minutes(109.77),
            q_value_mev: 0.634,
            daughter: "O-18".into(),
            modality: ClinicalModality::PET,
            clinical_use: "FDG-PET: glucose metabolism imaging (oncology, neurology, cardiology)"
                .into(),
            production: ProductionRoute::ProtonBombardment {
                target: "O-18 (water)".into(),
                target_z: 8,
                target_n: 10,
            },
        },
        MedicalIsotope {
            symbol: "C".into(),
            name: "C-11".into(),
            z: 6,
            n: 5,
            a: 11,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: minutes(20.334),
            q_value_mev: 0.960,
            daughter: "B-11".into(),
            modality: ClinicalModality::PET,
            clinical_use: "C-11 methionine/choline: amino acid metabolism, brain tumors".into(),
            production: ProductionRoute::ProtonBombardment {
                target: "N-14 (gas)".into(),
                target_z: 7,
                target_n: 7,
            },
        },
        MedicalIsotope {
            symbol: "N".into(),
            name: "N-13".into(),
            z: 7,
            n: 6,
            a: 13,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: minutes(9.965),
            q_value_mev: 1.199,
            daughter: "C-13".into(),
            modality: ClinicalModality::PET,
            clinical_use: "N-13 ammonia: myocardial perfusion imaging".into(),
            production: ProductionRoute::ProtonBombardment {
                target: "O-16 (water)".into(),
                target_z: 8,
                target_n: 8,
            },
        },
        MedicalIsotope {
            symbol: "O".into(),
            name: "O-15".into(),
            z: 8,
            n: 7,
            a: 15,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: minutes(2.037),
            q_value_mev: 1.732,
            daughter: "N-15".into(),
            modality: ClinicalModality::PET,
            clinical_use: "O-15 water: cerebral blood flow measurement".into(),
            production: ProductionRoute::DeuteronBombardment {
                target: "N-14 (gas)".into(),
                target_z: 7,
                target_n: 7,
            },
        },
        MedicalIsotope {
            symbol: "Ga".into(),
            name: "Ga-68".into(),
            z: 31,
            n: 37,
            a: 68,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: minutes(67.71),
            q_value_mev: 1.899,
            daughter: "Zn-68".into(),
            modality: ClinicalModality::PET,
            clinical_use: "DOTATATE-PET: neuroendocrine tumors, PSMA-PET: prostate cancer".into(),
            production: ProductionRoute::Generator {
                parent: "Ge-68".into(),
                parent_z: 32,
                parent_n: 36,
                parent_half_life_s: days(270.95),
            },
        },
        MedicalIsotope {
            symbol: "Cu".into(),
            name: "Cu-64".into(),
            z: 29,
            n: 35,
            a: 64,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: hours(12.701),
            q_value_mev: 0.653,
            daughter: "Ni-64/Zn-64".into(),
            modality: ClinicalModality::PET,
            clinical_use: "Cu-64 ATSM: hypoxia imaging, Cu-64 DOTATATE: NET imaging".into(),
            production: ProductionRoute::ProtonBombardment {
                target: "Ni-64".into(),
                target_z: 28,
                target_n: 36,
            },
        },
        // ── SPECT isotopes ───────────────────────────────────────────────
        MedicalIsotope {
            symbol: "Tc".into(),
            name: "Tc-99m".into(),
            z: 43,
            n: 56,
            a: 99,
            decay_mode: DecayMode::IsomericTransition,
            half_life_s: hours(6.006),
            q_value_mev: 0.1426,
            daughter: "Tc-99".into(),
            modality: ClinicalModality::SPECT,
            clinical_use: "Most-used medical isotope: bone, cardiac, renal, brain, thyroid scans"
                .into(),
            production: ProductionRoute::Generator {
                parent: "Mo-99".into(),
                parent_z: 42,
                parent_n: 57,
                parent_half_life_s: hours(65.924),
            },
        },
        MedicalIsotope {
            symbol: "Gd".into(),
            name: "Gd-153".into(),
            z: 64,
            n: 89,
            a: 153,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: days(240.4),
            q_value_mev: 0.485,
            daughter: "Eu-153".into(),
            modality: ClinicalModality::SPECT,
            clinical_use: "Bone densitometry calibration, quality control source".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Gd-152".into(),
                target_z: 64,
                target_n: 88,
            },
        },
        // ── Beta therapy isotopes ────────────────────────────────────────
        MedicalIsotope {
            symbol: "I".into(),
            name: "I-131".into(),
            z: 53,
            n: 78,
            a: 131,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(8.0207),
            q_value_mev: 0.606,
            daughter: "Xe-131".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use:
                "Thyroid cancer ablation, hyperthyroidism treatment, MIBG therapy (neuroblastoma)"
                    .into(),
            production: ProductionRoute::NeutronCapture {
                target: "Te-130".into(),
                target_z: 52,
                target_n: 78,
            },
        },
        MedicalIsotope {
            symbol: "Lu".into(),
            name: "Lu-177".into(),
            z: 71,
            n: 106,
            a: 177,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(6.647),
            q_value_mev: 0.498,
            daughter: "Hf-177".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "DOTATATE (Lutathera): NET therapy; PSMA-617 (Pluvicto): prostate cancer"
                .into(),
            production: ProductionRoute::NeutronCapture {
                target: "Lu-176".into(),
                target_z: 71,
                target_n: 105,
            },
        },
        MedicalIsotope {
            symbol: "Y".into(),
            name: "Y-90".into(),
            z: 39,
            n: 51,
            a: 90,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: hours(64.053),
            q_value_mev: 2.280,
            daughter: "Zr-90".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use:
                "Zevalin (NHL lymphoma), SIR-Spheres (liver metastases radioembolization)".into(),
            production: ProductionRoute::Generator {
                parent: "Sr-90".into(),
                parent_z: 38,
                parent_n: 52,
                parent_half_life_s: years(28.79),
            },
        },
        MedicalIsotope {
            symbol: "Sr".into(),
            name: "Sr-89".into(),
            z: 38,
            n: 51,
            a: 89,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(50.563),
            q_value_mev: 1.495,
            daughter: "Y-89".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Metastron: painful bone metastases palliation".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Sr-88".into(),
                target_z: 38,
                target_n: 50,
            },
        },
        MedicalIsotope {
            symbol: "Sm".into(),
            name: "Sm-153".into(),
            z: 62,
            n: 91,
            a: 153,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: hours(46.284),
            q_value_mev: 0.808,
            daughter: "Eu-153".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Quadramet: painful bone metastases (with EDTMP chelate)".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Sm-152".into(),
                target_z: 62,
                target_n: 90,
            },
        },
        MedicalIsotope {
            symbol: "Re".into(),
            name: "Re-186".into(),
            z: 75,
            n: 111,
            a: 186,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(3.7186),
            q_value_mev: 1.070,
            daughter: "Os-186".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Bone pain palliation (Re-186 HEDP), rheumatoid arthritis synovectomy"
                .into(),
            production: ProductionRoute::NeutronCapture {
                target: "Re-185".into(),
                target_z: 75,
                target_n: 110,
            },
        },
        MedicalIsotope {
            symbol: "Re".into(),
            name: "Re-188".into(),
            z: 75,
            n: 113,
            a: 188,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: hours(17.003),
            q_value_mev: 2.120,
            daughter: "Os-188".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Coronary artery restenosis, hepatocellular carcinoma, skin cancer"
                .into(),
            production: ProductionRoute::Generator {
                parent: "W-188".into(),
                parent_z: 74,
                parent_n: 114,
                parent_half_life_s: days(69.78),
            },
        },
        MedicalIsotope {
            symbol: "Er".into(),
            name: "Er-169".into(),
            z: 68,
            n: 101,
            a: 169,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(9.392),
            q_value_mev: 0.351,
            daughter: "Tm-169".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Radiation synovectomy (small joints), arthritis treatment".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Er-168".into(),
                target_z: 68,
                target_n: 100,
            },
        },
        MedicalIsotope {
            symbol: "Tb".into(),
            name: "Tb-161".into(),
            z: 65,
            n: 96,
            a: 161,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: days(6.89),
            q_value_mev: 0.593,
            daughter: "Dy-161".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use:
                "Targeted radionuclide therapy (similar to Lu-177 but with Auger electrons)".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Gd-160".into(),
                target_z: 64,
                target_n: 96,
            },
        },
        MedicalIsotope {
            symbol: "Cu".into(),
            name: "Cu-67".into(),
            z: 29,
            n: 38,
            a: 67,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: hours(61.83),
            q_value_mev: 0.577,
            daughter: "Zn-67".into(),
            modality: ClinicalModality::BetaTherapy,
            clinical_use: "Radioimmunotherapy (NHL lymphoma), theranostic partner to Cu-64".into(),
            production: ProductionRoute::ProtonBombardment {
                target: "Zn-68".into(),
                target_z: 30,
                target_n: 38,
            },
        },
        // ── Alpha therapy isotopes ───────────────────────────────────────
        MedicalIsotope {
            symbol: "Ac".into(),
            name: "Ac-225".into(),
            z: 89,
            n: 136,
            a: 225,
            decay_mode: DecayMode::Alpha,
            half_life_s: days(9.920),
            q_value_mev: 5.935,
            daughter: "Fr-221".into(),
            modality: ClinicalModality::AlphaTherapy,
            clinical_use: "Targeted alpha therapy: prostate (PSMA-225Ac), AML, glioblastoma".into(),
            production: ProductionRoute::Spallation {
                description: "Th-229 generator, Ra-226 proton spallation at TRIUMF/BNL".into(),
            },
        },
        MedicalIsotope {
            symbol: "Ra".into(),
            name: "Ra-223".into(),
            z: 88,
            n: 135,
            a: 223,
            decay_mode: DecayMode::Alpha,
            half_life_s: days(11.43),
            q_value_mev: 5.979,
            daughter: "Rn-219".into(),
            modality: ClinicalModality::AlphaTherapy,
            clinical_use: "Xofigo: bone metastases from castration-resistant prostate cancer"
                .into(),
            production: ProductionRoute::Generator {
                parent: "Ac-227".into(),
                parent_z: 89,
                parent_n: 138,
                parent_half_life_s: years(21.772),
            },
        },
        MedicalIsotope {
            symbol: "Pb".into(),
            name: "Pb-212".into(),
            z: 82,
            n: 130,
            a: 212,
            decay_mode: DecayMode::BetaMinus,
            half_life_s: hours(10.64),
            q_value_mev: 0.574,
            daughter: "Bi-212 (then alpha to Tl-208 or beta to Po-212)".into(),
            modality: ClinicalModality::AlphaTherapy,
            clinical_use: "In-vivo alpha generator: Pb-212-TCMC-trastuzumab (HER2+ cancers)".into(),
            production: ProductionRoute::Generator {
                parent: "Ra-224".into(),
                parent_z: 88,
                parent_n: 136,
                parent_half_life_s: days(3.6319),
            },
        },
        MedicalIsotope {
            symbol: "Bi".into(),
            name: "Bi-213".into(),
            z: 83,
            n: 130,
            a: 213,
            decay_mode: DecayMode::Alpha,
            half_life_s: minutes(45.59),
            q_value_mev: 5.870,
            daughter: "Tl-209".into(),
            modality: ClinicalModality::AlphaTherapy,
            clinical_use: "Targeted alpha therapy: AML (anti-CD33), bladder cancer, glioblastoma"
                .into(),
            production: ProductionRoute::Generator {
                parent: "Ac-225".into(),
                parent_z: 89,
                parent_n: 136,
                parent_half_life_s: days(9.920),
            },
        },
        MedicalIsotope {
            symbol: "At".into(),
            name: "At-211".into(),
            z: 85,
            n: 126,
            a: 211,
            decay_mode: DecayMode::Alpha,
            half_life_s: hours(7.214),
            q_value_mev: 5.982,
            daughter: "Bi-207".into(),
            modality: ClinicalModality::AlphaTherapy,
            clinical_use: "Brain tumors (glioblastoma), ovarian cancer, thyroid cancer".into(),
            production: ProductionRoute::ProtonAlpha {
                target: "Bi-209".into(),
                target_z: 83,
                target_n: 126,
            },
        },
        // ── Brachytherapy / Auger ────────────────────────────────────────
        MedicalIsotope {
            symbol: "I".into(),
            name: "I-125".into(),
            z: 53,
            n: 72,
            a: 125,
            decay_mode: DecayMode::BetaPlusEC,
            half_life_s: days(59.407),
            q_value_mev: 0.186,
            daughter: "Te-125".into(),
            modality: ClinicalModality::Brachytherapy,
            clinical_use: "Prostate brachytherapy seeds, brain tumor implants, lab tracer".into(),
            production: ProductionRoute::NeutronCapture {
                target: "Xe-124".into(),
                target_z: 54,
                target_n: 70,
            },
        },
    ]
}

// ── Theranostic pair identification ──────────────────────────────────────────

/// Element data for theranostic pair searching.
struct ElementIsotopeSet {
    symbol: String,
    z: u16,
    imaging: Vec<(u16, ClinicalModality, String)>, // (A, modality, name)
    therapy: Vec<(u16, ClinicalModality, String)>,
}

/// Identify theranostic pairs from the medical isotope database.
///
/// A theranostic pair consists of two isotopes of the same element where one
/// is used for imaging and the other for therapy, enabling "see it and treat it"
/// with the same targeting vector.
pub fn find_theranostic_pairs(db: &[MedicalIsotope]) -> Vec<TheranosticPair> {
    // Group by element
    let mut elements: std::collections::HashMap<u16, ElementIsotopeSet> =
        std::collections::HashMap::new();

    for iso in db {
        let entry = elements.entry(iso.z).or_insert_with(|| ElementIsotopeSet {
            symbol: iso.symbol.clone(),
            z: iso.z,
            imaging: Vec::new(),
            therapy: Vec::new(),
        });

        if iso.modality.is_imaging() {
            entry.imaging.push((iso.a, iso.modality, iso.name.clone()));
        } else {
            entry.therapy.push((iso.a, iso.modality, iso.name.clone()));
        }
    }

    let known_pairs: &[(&str, u16, u16)] = &[
        ("Cu", 64, 67),
        ("I", 125, 131), // I-125 (Auger/brachy) + I-131 (beta therapy)
    ];

    let mut pairs = Vec::new();

    for eset in elements.values() {
        for &(img_a, img_mod, ref img_name) in &eset.imaging {
            for &(ther_a, ther_mod, ref ther_name) in &eset.therapy {
                let is_known = known_pairs.iter().any(|(sym, ia, ta)| {
                    *sym == eset.symbol
                        && ((*ia == img_a && *ta == ther_a) || (*ta == img_a && *ia == ther_a))
                });

                pairs.push(TheranosticPair {
                    element: eset.symbol.clone(),
                    z: eset.z,
                    imaging_isotope: img_name.clone(),
                    imaging_a: img_a,
                    imaging_modality: img_mod,
                    therapy_isotope: ther_name.clone(),
                    therapy_a: ther_a,
                    therapy_modality: ther_mod,
                    is_known,
                    notes: if is_known {
                        "Established clinical theranostic pair".into()
                    } else {
                        "Database-derived pair from known medical isotopes".into()
                    },
                });
            }
        }
    }

    pairs.sort_by_key(|p| (p.z, p.imaging_a, p.therapy_a));
    pairs
}

/// Use ML predictor to search for novel theranostic pair candidates.
///
/// Scans elements with known medical isotopes for neighboring isotopes that
/// could serve as imaging/therapy partners based on Q-value and separation
/// energy analysis.
pub fn find_novel_theranostic_candidates(
    predictor: &MlMassPredictor,
    db: &[MedicalIsotope],
) -> Vec<TheranosticPair> {
    let mut candidates = Vec::new();

    // Elements with medical use — scan neighboring isotopes
    let medical_elements: Vec<(u16, String)> = {
        let mut seen = std::collections::HashSet::new();
        db.iter()
            .filter(|iso| seen.insert(iso.z))
            .map(|iso| (iso.z, iso.symbol.clone()))
            .collect()
    };

    for &(z, ref symbol) in &medical_elements {
        // Scan A range around known isotopes
        let known_a: Vec<u16> = db
            .iter()
            .filter(|iso| iso.z == z)
            .map(|iso| iso.a)
            .collect();
        let min_a = known_a
            .iter()
            .copied()
            .min()
            .unwrap_or(z * 2)
            .saturating_sub(5);
        let max_a = known_a.iter().copied().max().unwrap_or(z * 2 + 20) + 5;

        let mut imaging_candidates: Vec<(u16, ClinicalModality, f64)> = Vec::new();
        let mut therapy_candidates: Vec<(u16, ClinicalModality, f64)> = Vec::new();

        for a in min_a..=max_a {
            if a <= z {
                continue;
            }
            let n = a - z;
            if known_a.contains(&a) {
                continue;
            }

            let pred = predictor.predict(z, n);
            if pred.uncertainty > 2.0 {
                continue;
            } // Skip high-uncertainty predictions

            // Check for beta+ (PET candidate): Q_beta+ = BE(Z-1, N+1) - BE(Z, N) - 2*m_e
            // Simplified: Q_EC = M(Z,A) - M(Z-1,A) which requires mass excess
            // Use: Q_beta+ ~ BE(Z, N) - BE(Z-1, N+1) + (M_H - M_N) - 1.022
            if z > 1 {
                let daughter_pred = predictor.predict(z - 1, n + 1);
                let q_ec = pred.binding_energy - daughter_pred.binding_energy + (M_N - M_H);
                let q_beta_plus = q_ec - 1.022; // 2 * m_e

                if q_beta_plus > 0.1 {
                    imaging_candidates.push((a, ClinicalModality::PET, q_beta_plus));
                }
            }

            // Check for beta- (therapy candidate): Q_beta- = BE(Z+1, N-1) - BE(Z, N) + (M_N - M_H)
            if n > 1 {
                let daughter_pred = predictor.predict(z + 1, n - 1);
                let q_beta_minus = daughter_pred.binding_energy - pred.binding_energy + (M_N - M_H);

                if q_beta_minus > 0.3 && q_beta_minus < 3.0 {
                    therapy_candidates.push((a, ClinicalModality::BetaTherapy, q_beta_minus));
                }
            }

            // Check for alpha (TAT candidate): Q_alpha = BE(Z-2, N-2) + BE_He4 - BE(Z, N)
            if z > 2 && n > 2 {
                let daughter_pred = predictor.predict(z - 2, n - 2);
                let q_alpha = daughter_pred.binding_energy + BE_HE4 - pred.binding_energy;

                if q_alpha > 4.0 && q_alpha < 9.0 {
                    therapy_candidates.push((a, ClinicalModality::AlphaTherapy, q_alpha));
                }
            }
        }

        // Form pairs from candidates
        for &(img_a, img_mod, img_q) in &imaging_candidates {
            for &(ther_a, ther_mod, ther_q) in &therapy_candidates {
                if img_a == ther_a {
                    continue;
                }
                candidates.push(TheranosticPair {
                    element: symbol.clone(),
                    z,
                    imaging_isotope: format!("{}-{} (ML)", symbol, img_a),
                    imaging_a: img_a,
                    imaging_modality: img_mod,
                    therapy_isotope: format!("{}-{} (ML)", symbol, ther_a),
                    therapy_a: ther_a,
                    therapy_modality: ther_mod,
                    is_known: false,
                    notes: format!(
                        "ML-predicted: imaging Q={:.3} MeV, therapy Q={:.3} MeV",
                        img_q, ther_q,
                    ),
                });
            }
        }

        // Also pair ML candidates with known isotopes
        let known_imaging: Vec<_> = db
            .iter()
            .filter(|iso| iso.z == z && iso.modality.is_imaging())
            .collect();
        let known_therapy: Vec<_> = db
            .iter()
            .filter(|iso| iso.z == z && iso.modality.is_therapy())
            .collect();

        for &(ther_a, ther_mod, ther_q) in &therapy_candidates {
            for known_img in &known_imaging {
                candidates.push(TheranosticPair {
                    element: symbol.clone(),
                    z,
                    imaging_isotope: known_img.name.clone(),
                    imaging_a: known_img.a,
                    imaging_modality: known_img.modality,
                    therapy_isotope: format!("{}-{} (ML)", symbol, ther_a),
                    therapy_a: ther_a,
                    therapy_modality: ther_mod,
                    is_known: false,
                    notes: format!("Known imaging + ML therapy candidate: Q={:.3} MeV", ther_q,),
                });
            }
        }

        for &(img_a, img_mod, img_q) in &imaging_candidates {
            for known_ther in &known_therapy {
                candidates.push(TheranosticPair {
                    element: symbol.clone(),
                    z,
                    imaging_isotope: format!("{}-{} (ML)", symbol, img_a),
                    imaging_a: img_a,
                    imaging_modality: img_mod,
                    therapy_isotope: known_ther.name.clone(),
                    therapy_a: known_ther.a,
                    therapy_modality: known_ther.modality,
                    is_known: false,
                    notes: format!("ML imaging candidate + known therapy: Q={:.3} MeV", img_q,),
                });
            }
        }
    }

    candidates.sort_by_key(|p| (p.z, p.imaging_a, p.therapy_a));
    candidates
        .dedup_by(|a, b| a.z == b.z && a.imaging_a == b.imaging_a && a.therapy_a == b.therapy_a);
    candidates
}

// ── Production route optimization ────────────────────────────────────────────

/// Analyze production routes for a medical isotope using ML binding energies.
pub fn analyze_production_routes(
    isotope: &MedicalIsotope,
    predictor: &MlMassPredictor,
) -> Vec<ProductionAnalysis> {
    let mut routes = Vec::new();

    let product_pred = predictor.predict(isotope.z, isotope.n);
    let sa = isotope.specific_activity_gbq_per_g();

    // Analyze the primary production route
    match &isotope.production {
        ProductionRoute::NeutronCapture {
            target,
            target_z,
            target_n,
            ..
        } => {
            let target_pred = predictor.predict(*target_z, *target_n);
            // (n,gamma): Q = BE(Z, N+1) - BE(Z, N)
            // product has N = target_n + 1, Z = target_z
            let q = product_pred.binding_energy - target_pred.binding_energy;
            let q_unc = (product_pred.uncertainty.powi(2) + target_pred.uncertainty.powi(2)).sqrt();

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!("{}(n,gamma){}", target, isotope.name),
                q_value_ml: q,
                q_uncertainty: q_unc,
                is_exothermic: q > 0.0,
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
        ProductionRoute::ProtonBombardment {
            target,
            target_z,
            target_n,
            ..
        } => {
            let target_pred = predictor.predict(*target_z, *target_n);
            // (p,n): Q = BE(Z+1, N-1) - BE(Z, N) + (M_n - M_p)
            // where M_n - M_p = 1.293 MeV
            let q = product_pred.binding_energy - target_pred.binding_energy + (M_N - M_H);
            let q_unc = (product_pred.uncertainty.powi(2) + target_pred.uncertainty.powi(2)).sqrt();

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!("{}(p,n){}", target, isotope.name),
                q_value_ml: q,
                q_uncertainty: q_unc,
                is_exothermic: q > 0.0,
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
        ProductionRoute::Generator {
            parent,
            parent_z,
            parent_n,
            parent_half_life_s,
            ..
        } => {
            let parent_pred = predictor.predict(*parent_z, *parent_n);
            // Parent decays to daughter — compute Q from mass difference
            let q = parent_pred.binding_energy - product_pred.binding_energy;
            // Note: for generators, the "Q" is the parent decay Q-value — sign depends on decay
            // We report the absolute separation
            let q_unc = (parent_pred.uncertainty.powi(2) + product_pred.uncertainty.powi(2)).sqrt();

            // Classify equilibrium
            let ratio = parent_half_life_s / isotope.half_life_s;
            let eq = if ratio > 100.0 {
                GeneratorEquilibrium::Secular
            } else if ratio > 1.0 {
                GeneratorEquilibrium::Transient
            } else {
                GeneratorEquilibrium::NoEquilibrium
            };

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!("{} -> {} (generator)", parent, isotope.name),
                q_value_ml: q.abs(),
                q_uncertainty: q_unc,
                is_exothermic: true, // Generators are always spontaneous decay
                specific_activity_gbq_per_g: sa,
                equilibrium: Some(eq),
            });
        }
        ProductionRoute::DeuteronBombardment {
            target,
            target_z,
            target_n,
            ..
        } => {
            let target_pred = predictor.predict(*target_z, *target_n);
            // (d,n): Q = BE(products) - BE(reactants)
            // Deuteron BE = 2.224 MeV
            let deuteron_be = 2.224;
            let q = product_pred.binding_energy - target_pred.binding_energy - deuteron_be;
            let q_unc = (product_pred.uncertainty.powi(2) + target_pred.uncertainty.powi(2)).sqrt();

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!("{}(d,n){}", target, isotope.name),
                q_value_ml: q,
                q_uncertainty: q_unc,
                is_exothermic: q > 0.0,
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
        ProductionRoute::ProtonAlpha {
            target,
            target_z,
            target_n,
            ..
        } => {
            let target_pred = predictor.predict(*target_z, *target_n);
            // (alpha,2n) or similar — approximate as Q = BE(product) + BE(alpha) - BE(target) - BE(projectile)
            // For At-211: Bi-209(alpha,2n)At-211
            let q = product_pred.binding_energy + BE_HE4 - target_pred.binding_energy;
            let q_unc = (product_pred.uncertainty.powi(2) + target_pred.uncertainty.powi(2)).sqrt();

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!("{}(p,alpha) or equivalent -> {}", target, isotope.name),
                q_value_ml: q,
                q_uncertainty: q_unc,
                is_exothermic: q > 0.0,
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
        ProductionRoute::Spallation { description } => {
            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: description.clone(),
                q_value_ml: 0.0, // Spallation Q is complex multi-body
                q_uncertainty: 0.0,
                is_exothermic: true, // Driven by accelerator
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
    }

    // Also compute alternative (n,gamma) route if not already the primary
    if !matches!(isotope.production, ProductionRoute::NeutronCapture { .. }) {
        // (n,gamma) on (Z, N-1) target
        if isotope.n > 1 {
            let target_z = isotope.z;
            let target_n = isotope.n - 1;
            let target_pred = predictor.predict(target_z, target_n);
            let q = product_pred.binding_energy - target_pred.binding_energy;
            let q_unc = (product_pred.uncertainty.powi(2) + target_pred.uncertainty.powi(2)).sqrt();

            routes.push(ProductionAnalysis {
                isotope_name: isotope.name.clone(),
                route_description: format!(
                    "Z={},N={}(n,gamma){} [alternative]",
                    target_z, target_n, isotope.name
                ),
                q_value_ml: q,
                q_uncertainty: q_unc,
                is_exothermic: q > 0.0,
                specific_activity_gbq_per_g: sa,
                equilibrium: None,
            });
        }
    }

    routes
}

/// Comprehensive production analysis for Ac-225 — the hardest medical isotope to produce.
///
/// Explores all known and proposed routes:
/// 1. Th-229 alpha decay (natural, extremely limited)
/// 2. Ra-226(p,2n) spallation
/// 3. Th-232(p,spallation) at high energy
/// 4. Ra-226(n,gamma) chain: Ra-226 -> Ra-227 -> Ac-227 -> Th-227 -> Ra-223 (not Ac-225!)
/// 5. Th-229 generator
pub fn analyze_ac225_production(predictor: &MlMassPredictor) -> Vec<ProductionAnalysis> {
    let mut routes = Vec::new();

    let ac225 = predictor.predict(89, 136); // Ac-225: Z=89, N=136
    let sa_ac225 = {
        let lambda = LN2 / days(9.920);
        lambda * N_A / 225.0 / 225.0 / 1e9
    };

    // Route 1: Th-229 alpha decay generator
    let th229 = predictor.predict(90, 139);
    {
        let q = th229.binding_energy + BE_HE4 - ac225.binding_energy;
        let q_unc = (th229.uncertainty.powi(2) + ac225.uncertainty.powi(2)).sqrt();
        let th229_half_life = years(7932.0);
        let ac225_half_life = days(9.920);
        let eq = if th229_half_life / ac225_half_life > 100.0 {
            GeneratorEquilibrium::Secular
        } else {
            GeneratorEquilibrium::Transient
        };

        routes.push(ProductionAnalysis {
            isotope_name: "Ac-225".into(),
            route_description: "Th-229(alpha)Ac-225 generator (ORNL stockpile, ~1.7 Ci worldwide)"
                .into(),
            q_value_ml: q,
            q_uncertainty: q_unc,
            is_exothermic: q > 0.0,
            specific_activity_gbq_per_g: sa_ac225,
            equilibrium: Some(eq),
        });
    }

    // Route 2: Ra-226(p,2n)Ac-225 — most promising new route
    let ra226 = predictor.predict(88, 138);
    {
        // (p,2n): Q = BE(Ac-225) + 2*M_n_excess - BE(Ra-226) - M_H_excess
        // Simplified: Q = BE(Ac-225) - BE(Ra-226) + (M_N - M_H) - separation energy of 1 neutron
        // More accurately: Q = [M(Ra-226) + M(p)] - [M(Ac-225) + 2*M(n)]
        // In BE terms: Q = BE(Ac-225) - BE(Ra-226) + 2*M_N - M_H (approximate)
        let two_n_sep = {
            let ac226 = predictor.predict(89, 137);
            ac226.binding_energy - ac225.binding_energy
        };
        let q = ac225.binding_energy - ra226.binding_energy + (M_N - M_H) - two_n_sep;
        let q_unc = (ac225.uncertainty.powi(2) + ra226.uncertainty.powi(2)).sqrt();

        routes.push(ProductionAnalysis {
            isotope_name: "Ac-225".into(),
            route_description:
                "Ra-226(p,2n)Ac-225 (TRIUMF/BNL, ~16 MeV protons, most promising scale-up)".into(),
            q_value_ml: q,
            q_uncertainty: q_unc,
            is_exothermic: q > 0.0,
            specific_activity_gbq_per_g: sa_ac225,
            equilibrium: None,
        });
    }

    // Route 3: Th-232(p,spallation) high-energy proton
    let th232 = predictor.predict(90, 142);
    {
        // Spallation is multi-body, approximate Q from mass difference
        let q = ac225.binding_energy - th232.binding_energy;
        let q_unc = (ac225.uncertainty.powi(2) + th232.uncertainty.powi(2)).sqrt();

        routes.push(ProductionAnalysis {
            isotope_name: "Ac-225".into(),
            route_description:
                "Th-232(p,X)Ac-225 high-energy spallation (~1 GeV, LANL/CERN-ISOLDE)".into(),
            q_value_ml: q,
            q_uncertainty: q_unc,
            is_exothermic: false, // Requires high-energy protons
            specific_activity_gbq_per_g: sa_ac225,
            equilibrium: None,
        });
    }

    // Route 4: Ra-226 neutron irradiation chain
    // Ra-226(n,g)Ra-227 -> Ac-227(beta-) -> Th-227(alpha) -> Ra-223 (NOT Ac-225!)
    // But: Ra-225(beta-)Ac-225 is an alternative if Ra-225 can be made
    let ra225 = predictor.predict(88, 137);
    {
        // Ra-225(beta-)Ac-225: Q = BE(Ac-225) - BE(Ra-225) + (M_N - M_H)
        let q = ac225.binding_energy - ra225.binding_energy + (M_N - M_H);
        let q_unc = (ac225.uncertainty.powi(2) + ra225.uncertainty.powi(2)).sqrt();

        routes.push(ProductionAnalysis {
            isotope_name: "Ac-225".into(),
            route_description: "Ra-225(beta-)Ac-225 (Ra-225 from Ra-226(gamma,n) photonuclear)"
                .into(),
            q_value_ml: q,
            q_uncertainty: q_unc,
            is_exothermic: q > 0.0,
            specific_activity_gbq_per_g: sa_ac225,
            equilibrium: None,
        });
    }

    // Route 5: U-233 natural decay chain
    // U-233 -> Th-229 -> Ra-225 -> Ac-225
    let u233 = predictor.predict(92, 141);
    {
        let q = u233.binding_energy + BE_HE4 - th229.binding_energy;
        let q_unc = (u233.uncertainty.powi(2) + th229.uncertainty.powi(2)).sqrt();

        routes.push(ProductionAnalysis {
            isotope_name: "Ac-225".into(),
            route_description:
                "U-233(alpha)Th-229(alpha)..Ra-225(beta-)Ac-225 (natural chain, DOE stockpile)"
                    .into(),
            q_value_ml: q,
            q_uncertainty: q_unc,
            is_exothermic: q > 0.0,
            specific_activity_gbq_per_g: sa_ac225,
            equilibrium: Some(GeneratorEquilibrium::Secular),
        });
    }

    routes
}

// ── Novel isotope candidate scanning ─────────────────────────────────────────

/// Scan for novel isotope candidates using ML predictions.
///
/// Searches elements with established medical use for unmeasured/unstudied
/// isotopes that could have useful clinical properties.
pub fn scan_novel_candidates(predictor: &MlMassPredictor) -> Vec<NovelCandidate> {
    let mut candidates = Vec::new();

    // Elements with established medical isotope programs
    // (Z, symbol, medical context)
    let medical_elements: &[(u16, &str, &str)] = &[
        (29, "Cu", "theranostic pairs"),
        (31, "Ga", "PET imaging"),
        (38, "Sr", "bone therapy"),
        (39, "Y", "radioimmunotherapy"),
        (43, "Tc", "SPECT imaging"),
        (53, "I", "thyroid therapy"),
        (62, "Sm", "bone palliation"),
        (64, "Gd", "neutron capture therapy"),
        (65, "Tb", "Swiss army knife of nuclear medicine"),
        (68, "Er", "synovectomy"),
        (71, "Lu", "peptide receptor therapy"),
        (75, "Re", "bone palliation"),
        (82, "Pb", "alpha generators"),
        (83, "Bi", "targeted alpha therapy"),
        (85, "At", "targeted alpha therapy"),
        (88, "Ra", "bone-seeking alpha"),
        (89, "Ac", "targeted alpha therapy"),
    ];

    for &(z, symbol, context) in medical_elements {
        // Scan a reasonable A range for each element
        let a_min = (2 * z).saturating_sub(10).max(z + 1);
        let a_max = 2 * z + 50;

        for a in a_min..=a_max {
            let n = a - z;
            if n == 0 {
                continue;
            }

            let pred = predictor.predict(z, n);
            if pred.uncertainty > 3.0 {
                continue;
            } // Too uncertain

            // ── Alpha emitter scan (TAT candidates) ──
            // Q_alpha = BE(Z-2, N-2) + BE_He4 - BE(Z, N)
            if z > 2 && n > 2 {
                let daughter = predictor.predict(z - 2, n - 2);
                let q_alpha = daughter.binding_energy + BE_HE4 - pred.binding_energy;

                // Want: Q_alpha > 3.5 MeV (emitter), < 10 MeV (not too fast).
                // Model uncertainty ~0.7 MeV, so widen from ideal 5-8 MeV window.
                if q_alpha > 3.5 && q_alpha < 10.0 {
                    // Estimate half-life category from Q using Geiger-Nuttall (very rough)
                    let hl_category = estimate_alpha_half_life_category(z, q_alpha);
                    if is_therapeutic_half_life_alpha(&hl_category) {
                        // ── Daughter-toxicity filter ──
                        // For astatine (Z=85), even-A isotopes (At-210, At-208, etc.)
                        // decay primarily by electron capture (~99.8%) to Polonium (Z=84),
                        // NOT by alpha emission. Po-210 in particular is extremely toxic
                        // (t½=138d, the Litvinenko poison). Only odd-A astatine (At-211)
                        // decays predominantly by alpha and is considered safe for TAT.
                        //
                        // More generally: reject alpha therapy candidates whose alpha-decay
                        // daughters are Polonium isotopes (Z=84), since all Po isotopes
                        // are highly radiotoxic with no stable forms.
                        let daughter_z = z - 2;
                        let is_daughter_toxic = if z == 85 && a % 2 == 0 {
                            // Even-A astatine: EC/beta+ dominates → produces Po daughter
                            // (e.g., At-210 →(EC 99.8%)→ Po-210, a potent alpha poison)
                            true
                        } else if daughter_z == 84 {
                            // Any candidate whose alpha daughter is Polonium — all Po
                            // isotopes are dangerously radiotoxic (no stable Po exists)
                            true
                        } else {
                            false
                        };

                        if is_daughter_toxic {
                            continue;
                        }

                        candidates.push(NovelCandidate {
                            symbol: symbol.to_string(),
                            z,
                            n,
                            a,
                            binding_energy: pred.binding_energy,
                            uncertainty: pred.uncertainty,
                            q_value: q_alpha,
                            half_life_category: hl_category,
                            suggested_modality: ClinicalModality::AlphaTherapy,
                            rationale: format!(
                                "Alpha emitter with Q={:.2} MeV in {} ({}). \
                                 Daughter: Z={}, N={}, BE={:.1} MeV.",
                                q_alpha,
                                symbol,
                                context,
                                z - 2,
                                n - 2,
                                daughter.binding_energy,
                            ),
                            daughter_toxicity_note: String::new(),
                        });
                    }
                }
            }

            // ── Beta+ emitter scan (PET candidates) ──
            if z > 1 {
                let daughter = predictor.predict(z - 1, n + 1);
                let q_ec = pred.binding_energy - daughter.binding_energy + (M_N - M_H);
                let q_beta_plus = q_ec - 1.022;

                // Want: Q_beta+ > 0.5 MeV, half-life 10 min to 6 hours
                if q_beta_plus > 0.5 && q_beta_plus < 5.0 {
                    let hl_category = estimate_beta_half_life_category(q_beta_plus, z, a);
                    if is_pet_useful_half_life(&hl_category) {
                        candidates.push(NovelCandidate {
                            symbol: symbol.to_string(),
                            z,
                            n,
                            a,
                            binding_energy: pred.binding_energy,
                            uncertainty: pred.uncertainty,
                            q_value: q_beta_plus,
                            half_life_category: hl_category,
                            suggested_modality: ClinicalModality::PET,
                            rationale: format!(
                                "Beta+ emitter with Q={:.2} MeV in {} ({}). \
                                 Potential PET isotope for {}-labeled tracers.",
                                q_beta_plus, symbol, context, symbol,
                            ),
                            daughter_toxicity_note: String::new(),
                        });
                    }
                }
            }

            // ── Beta- emitter scan (therapy candidates) ──
            if n > 1 {
                let daughter = predictor.predict(z + 1, n - 1);
                let q_beta_minus = daughter.binding_energy - pred.binding_energy + (M_N - M_H);

                // Want: Q in 0.5-2 MeV, half-life 1-14 days
                if q_beta_minus > 0.5 && q_beta_minus < 2.0 {
                    let hl_category = estimate_beta_half_life_category(q_beta_minus, z, a);
                    if is_therapy_useful_half_life(&hl_category) {
                        candidates.push(NovelCandidate {
                            symbol: symbol.to_string(),
                            z,
                            n,
                            a,
                            binding_energy: pred.binding_energy,
                            uncertainty: pred.uncertainty,
                            q_value: q_beta_minus,
                            half_life_category: hl_category,
                            suggested_modality: ClinicalModality::BetaTherapy,
                            rationale: format!(
                                "Beta- emitter with Q={:.2} MeV in {} ({}). \
                                 Suitable energy for targeted radionuclide therapy.",
                                q_beta_minus, symbol, context,
                            ),
                            daughter_toxicity_note: String::new(),
                        });
                    }
                }
            }
        }
    }

    // Deduplicate by (Z, A, modality)
    candidates.sort_by(|a, b| {
        a.z.cmp(&b.z)
            .then(a.a.cmp(&b.a))
            .then(format!("{:?}", a.suggested_modality).cmp(&format!("{:?}", b.suggested_modality)))
    });
    candidates
        .dedup_by(|a, b| a.z == b.z && a.a == b.a && a.suggested_modality == b.suggested_modality);

    candidates
}

// ── Half-life estimation heuristics ──────────────────────────────────────────

/// Rough alpha half-life category from Geiger-Nuttall systematics.
///
/// The Geiger-Nuttall law: log10(t1/2) ~ a + b / sqrt(Q)
/// where a, b depend on Z. We use simplified categories.
fn estimate_alpha_half_life_category(_z: u16, q_alpha: f64) -> String {
    // Simplified Q-based classification for medical isotope screening.
    // For heavy elements (Z>80), Q_alpha is the dominant factor:
    //   Q > 8 MeV → microseconds to seconds (Po-212: 8.95 MeV, 0.3μs)
    //   Q ~ 6-8 MeV → minutes to days (At-211: 5.98, 7.2h; Bi-213: 5.87, 46m)
    //   Q ~ 5-6 MeV → days to years (Ac-225: 5.94, 10d; Ra-223: 5.98, 11.4d)
    //   Q ~ 4-5 MeV → years to geological (Ra-226: 4.87, 1600yr)
    //   Q < 4 MeV → essentially stable
    if q_alpha > 8.0 {
        "sub-second".to_string()
    } else if q_alpha > 6.5 {
        "seconds to minutes".to_string()
    } else if q_alpha > 5.0 {
        "hours to days".to_string()
    } else if q_alpha > 4.0 {
        "days to months".to_string()
    } else {
        "years to geological".to_string()
    }
}

/// Check if alpha half-life category is therapeutic (minutes to ~30 days ideal).
fn is_therapeutic_half_life_alpha(category: &str) -> bool {
    category.contains("hours to days")
        || category.contains("seconds to minutes")
        || category.contains("days to months")
}

/// Rough beta half-life category (very approximate — beta lifetimes depend on
/// log ft values which require nuclear matrix elements, not just Q).
fn estimate_beta_half_life_category(q_mev: f64, z: u16, a: u16) -> String {
    // ft values typically 3-20 for allowed transitions
    // t ~ ft / f(Z, Q) where f is the Fermi function
    // Very rough: higher Q -> shorter half-life
    let _ = (z, a); // Z and A affect Fermi function but we keep it simple
    if q_mev > 4.0 {
        "seconds to minutes".to_string()
    } else if q_mev > 2.0 {
        "minutes to hours".to_string()
    } else if q_mev > 1.0 {
        "hours to days".to_string()
    } else if q_mev > 0.3 {
        "days to weeks".to_string()
    } else {
        "weeks to months".to_string()
    }
}

/// PET useful: 10 min to 6 hours.
fn is_pet_useful_half_life(category: &str) -> bool {
    category.contains("minutes to hours") || category.contains("hours to days")
}

/// Therapy useful: 1-14 days.
fn is_therapy_useful_half_life(category: &str) -> bool {
    category.contains("hours to days") || category.contains("days to weeks")
}

// ── Utility functions ────────────────────────────────────────────────────────

/// Compute neutron separation energy: S_n(Z, N) = BE(Z, N) - BE(Z, N-1).
pub fn neutron_separation_energy(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let be_full = predictor.predict(z, n).binding_energy;
    let be_minus = predictor.predict(z, n - 1).binding_energy;
    be_full - be_minus
}

/// Compute proton separation energy: S_p(Z, N) = BE(Z, N) - BE(Z-1, N).
pub fn proton_separation_energy(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    if z == 0 {
        return 0.0;
    }
    let be_full = predictor.predict(z, n).binding_energy;
    let be_minus = predictor.predict(z - 1, n).binding_energy;
    be_full - be_minus
}

/// Compute alpha separation energy: S_alpha(Z, N) = BE(Z, N) - BE(Z-2, N-2) - BE_He4.
pub fn alpha_separation_energy(predictor: &MlMassPredictor, z: u16, n: u16) -> f64 {
    if z < 2 || n < 2 {
        return 0.0;
    }
    let be_full = predictor.predict(z, n).binding_energy;
    let be_daughter = predictor.predict(z - 2, n - 2).binding_energy;
    be_full - be_daughter - BE_HE4
}

/// Compute Q-value for (n,gamma) reaction: target(n,gamma)product.
pub fn q_neutron_capture(predictor: &MlMassPredictor, target_z: u16, target_n: u16) -> f64 {
    // Product: (target_z, target_n + 1)
    let be_product = predictor.predict(target_z, target_n + 1).binding_energy;
    let be_target = predictor.predict(target_z, target_n).binding_energy;
    be_product - be_target
}

/// Compute Q-value for (p,n) reaction: target(p,n)product.
pub fn q_proton_neutron(predictor: &MlMassPredictor, target_z: u16, target_n: u16) -> f64 {
    // Product: (target_z + 1, target_n - 1)
    if target_n == 0 {
        return 0.0;
    }
    let be_product = predictor.predict(target_z + 1, target_n - 1).binding_energy;
    let be_target = predictor.predict(target_z, target_n).binding_energy;
    be_product - be_target + (M_N - M_H)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medical_isotope_database_completeness() {
        let db = medical_isotope_database();

        // Verify minimum required isotopes are present
        let required = [
            "Tc-99m", "I-131", "I-125", "Lu-177", "Ac-225", "Ra-223", "Y-90", "Ga-68", "Cu-64",
            "Cu-67", "F-18", "C-11", "N-13", "O-15", "Pb-212", "Bi-213", "At-211", "Re-186",
            "Re-188", "Sm-153", "Er-169", "Tb-161", "Gd-153", "Sr-89",
        ];

        for name in &required {
            assert!(
                db.iter().any(|iso| iso.name == *name),
                "Missing required isotope: {}",
                name,
            );
        }

        println!("\n=== MEDICAL ISOTOPE DATABASE ({} isotopes) ===", db.len());
        println!(
            "{:<10} {:>3} {:>3} {:>4} {:<10} {:>14} {:>8} {:<12} {}",
            "Name", "Z", "N", "A", "Decay", "Half-life", "Q (MeV)", "Modality", "Clinical Use"
        );
        println!("{}", "-".repeat(120));

        for iso in &db {
            let hl_str = format_half_life(iso.half_life_s);
            println!(
                "{:<10} {:>3} {:>3} {:>4} {:<10} {:>14} {:>8.3} {:<12} {}",
                iso.name,
                iso.z,
                iso.n,
                iso.a,
                format!("{}", iso.decay_mode),
                hl_str,
                iso.q_value_mev,
                format!("{}", iso.modality),
                &iso.clinical_use[..iso.clinical_use.len().min(50)],
            );
        }
    }

    #[test]
    fn test_specific_activity() {
        let db = medical_isotope_database();

        println!("\n=== SPECIFIC ACTIVITY ===");
        println!(
            "{:<10} {:>14} {:>16} {:>16}",
            "Isotope", "Half-life", "SA (GBq/g)", "SA (Ci/g)"
        );
        println!("{}", "-".repeat(60));

        for iso in &db {
            let sa_gbq = iso.specific_activity_gbq_per_g();
            let sa_ci = sa_gbq / 37.0; // 1 Ci = 37 GBq
            let hl_str = format_half_life(iso.half_life_s);

            println!(
                "{:<10} {:>14} {:>16.2e} {:>16.2e}",
                iso.name, hl_str, sa_gbq, sa_ci,
            );
        }

        // Tc-99m should have very high specific activity (short half-life)
        let tc99m = db.iter().find(|i| i.name == "Tc-99m").unwrap();
        let f18 = db.iter().find(|i| i.name == "F-18").unwrap();
        assert!(
            tc99m.specific_activity_gbq_per_g() > 1e6,
            "Tc-99m SA should be very high"
        );
        assert!(
            f18.specific_activity_gbq_per_g() > tc99m.specific_activity_gbq_per_g(),
            "F-18 (shorter half-life) should have higher SA than Tc-99m"
        );
    }

    #[test]
    fn test_decay_constants() {
        let db = medical_isotope_database();

        let tc99m = db.iter().find(|i| i.name == "Tc-99m").unwrap();
        let lambda = tc99m.decay_constant();
        // lambda = ln(2) / (6.006 * 3600) = 3.206e-5 s^-1
        assert!(
            (lambda - 3.206e-5).abs() < 1e-7,
            "Tc-99m decay constant: got {}, expected ~3.206e-5",
            lambda
        );

        let i131 = db.iter().find(|i| i.name == "I-131").unwrap();
        let lambda_i = i131.decay_constant();
        // lambda = ln(2) / (8.0207 * 86400) = 1.0e-6 s^-1
        assert!(
            lambda_i > 9e-7 && lambda_i < 1.1e-6,
            "I-131 decay constant: got {}",
            lambda_i
        );
    }

    #[test]
    fn test_theranostic_pairs_known() {
        let db = medical_isotope_database();
        let pairs = find_theranostic_pairs(&db);

        println!("\n=== KNOWN THERANOSTIC PAIRS ===");
        println!(
            "{:<8} {:<14} {:<12} {:<14} {:<14} {}",
            "Element", "Imaging", "Modality", "Therapy", "Modality", "Notes"
        );
        println!("{}", "-".repeat(90));

        for pair in &pairs {
            println!(
                "{:<8} {:<14} {:<12} {:<14} {:<14} {}",
                pair.element,
                pair.imaging_isotope,
                format!("{}", pair.imaging_modality),
                pair.therapy_isotope,
                format!("{}", pair.therapy_modality),
                if pair.is_known {
                    "KNOWN"
                } else {
                    "database-derived"
                },
            );
        }

        // Cu-64 (PET) / Cu-67 (therapy) must be found
        assert!(
            pairs
                .iter()
                .any(|p| { p.element == "Cu" && p.imaging_a == 64 && p.therapy_a == 67 }),
            "Cu-64/Cu-67 theranostic pair not found"
        );

        assert!(
            !pairs.is_empty(),
            "Should find at least one theranostic pair"
        );
    }

    #[test]
    fn test_novel_theranostic_candidates() {
        let predictor = MlMassPredictor::new();
        let db = medical_isotope_database();
        let novel = find_novel_theranostic_candidates(&predictor, &db);

        println!(
            "\n=== ML-PREDICTED NOVEL THERANOSTIC CANDIDATES ({} found) ===",
            novel.len()
        );
        println!(
            "{:<8} {:<16} {:<12} {:<16} {:<14} {}",
            "Element", "Imaging", "Modality", "Therapy", "Modality", "Notes"
        );
        println!("{}", "-".repeat(100));

        for pair in novel.iter().take(30) {
            println!(
                "{:<8} {:<16} {:<12} {:<16} {:<14} {}",
                pair.element,
                pair.imaging_isotope,
                format!("{}", pair.imaging_modality),
                pair.therapy_isotope,
                format!("{}", pair.therapy_modality),
                &pair.notes[..pair.notes.len().min(40)],
            );
        }

        if novel.len() > 30 {
            println!("... and {} more candidates", novel.len() - 30);
        }
    }

    #[test]
    fn test_production_routes_ml() {
        let predictor = MlMassPredictor::new();
        let db = medical_isotope_database();

        println!("\n=== PRODUCTION ROUTE ANALYSIS (ML binding energies) ===");
        println!(
            "{:<10} {:<50} {:>10} {:>8} {:>6} {:>14}",
            "Isotope", "Route", "Q (MeV)", "+/- MeV", "Exo?", "SA (GBq/g)"
        );
        println!("{}", "-".repeat(110));

        for iso in &db {
            let routes = analyze_production_routes(iso, &predictor);
            for route in &routes {
                println!(
                    "{:<10} {:<50} {:>10.3} {:>8.3} {:>6} {:>14.2e}",
                    route.isotope_name,
                    &route.route_description[..route.route_description.len().min(50)],
                    route.q_value_ml,
                    route.q_uncertainty,
                    if route.is_exothermic { "yes" } else { "no" },
                    route.specific_activity_gbq_per_g,
                );
            }
        }

        // Neutron capture should generally be exothermic
        let tc99m = db.iter().find(|i| i.name == "Tc-99m").unwrap();
        let tc_routes = analyze_production_routes(tc99m, &predictor);
        // Generator route should exist
        assert!(
            tc_routes.iter().any(|r| r.equilibrium.is_some()),
            "Tc-99m should have a generator route"
        );
    }

    #[test]
    fn test_ac225_production_analysis() {
        let predictor = MlMassPredictor::new();
        let routes = analyze_ac225_production(&predictor);

        println!("\n=== Ac-225 PRODUCTION ROUTE COMPARISON ===");
        println!("(The hardest and most valuable medical isotope to produce)\n");
        println!(
            "{:<65} {:>10} {:>8} {:>6} {:>12}",
            "Route", "Q (MeV)", "+/- MeV", "Exo?", "Equilibrium"
        );
        println!("{}", "-".repeat(110));

        for route in &routes {
            println!(
                "{:<65} {:>10.3} {:>8.3} {:>6} {:>12}",
                &route.route_description[..route.route_description.len().min(65)],
                route.q_value_ml,
                route.q_uncertainty,
                if route.is_exothermic { "yes" } else { "no" },
                route
                    .equilibrium
                    .map(|e| format!("{}", e))
                    .unwrap_or_default(),
            );
        }

        assert!(routes.len() >= 4, "Should analyze at least 4 Ac-225 routes");

        // Th-229 generator should be secular equilibrium
        let th229_route = routes
            .iter()
            .find(|r| r.route_description.contains("Th-229"))
            .unwrap();
        assert_eq!(th229_route.equilibrium, Some(GeneratorEquilibrium::Secular));
    }

    #[test]
    fn test_novel_candidate_scan() {
        let predictor = MlMassPredictor::new();
        let candidates = scan_novel_candidates(&predictor);

        println!(
            "\n=== NOVEL ISOTOPE CANDIDATE SCAN ({} candidates) ===",
            candidates.len()
        );

        // Group by modality
        let alpha: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::AlphaTherapy)
            .collect();
        let pet: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::PET)
            .collect();
        let beta_therapy: Vec<_> = candidates
            .iter()
            .filter(|c| c.suggested_modality == ClinicalModality::BetaTherapy)
            .collect();

        println!("\n--- Alpha therapy candidates ({}) ---", alpha.len());
        println!(
            "{:<8} {:>4} {:>4} {:>8} {:>8} {:>20} {}",
            "Symbol", "Z", "A", "Q (MeV)", "+/- MeV", "Half-life est.", "Rationale"
        );
        println!("{}", "-".repeat(100));
        for c in alpha.iter().take(15) {
            println!(
                "{:<8} {:>4} {:>4} {:>8.3} {:>8.3} {:>20} {}",
                c.symbol,
                c.z,
                c.a,
                c.q_value,
                c.uncertainty,
                c.half_life_category,
                &c.rationale[..c.rationale.len().min(45)],
            );
        }

        println!("\n--- PET candidates ({}) ---", pet.len());
        println!(
            "{:<8} {:>4} {:>4} {:>8} {:>8} {:>20} {}",
            "Symbol", "Z", "A", "Q (MeV)", "+/- MeV", "Half-life est.", "Rationale"
        );
        println!("{}", "-".repeat(100));
        for c in pet.iter().take(15) {
            println!(
                "{:<8} {:>4} {:>4} {:>8.3} {:>8.3} {:>20} {}",
                c.symbol,
                c.z,
                c.a,
                c.q_value,
                c.uncertainty,
                c.half_life_category,
                &c.rationale[..c.rationale.len().min(45)],
            );
        }

        println!("\n--- Beta therapy candidates ({}) ---", beta_therapy.len());
        println!(
            "{:<8} {:>4} {:>4} {:>8} {:>8} {:>20} {}",
            "Symbol", "Z", "A", "Q (MeV)", "+/- MeV", "Half-life est.", "Rationale"
        );
        println!("{}", "-".repeat(100));
        for c in beta_therapy.iter().take(15) {
            println!(
                "{:<8} {:>4} {:>4} {:>8.3} {:>8.3} {:>20} {}",
                c.symbol,
                c.z,
                c.a,
                c.q_value,
                c.uncertainty,
                c.half_life_category,
                &c.rationale[..c.rationale.len().min(45)],
            );
        }

        // Should find candidates in each category
        assert!(!alpha.is_empty(), "Should find alpha therapy candidates");
        assert!(!pet.is_empty(), "Should find PET candidates");
        assert!(
            !beta_therapy.is_empty(),
            "Should find beta therapy candidates"
        );
    }

    #[test]
    fn test_separation_energies() {
        let predictor = MlMassPredictor::new();

        // Neutron separation energy for some known nuclei
        // O-16 (Z=8, N=8): S_n ~ 15.7 MeV (doubly magic)
        let sn_o16 = neutron_separation_energy(&predictor, 8, 8);
        println!("S_n(O-16) = {:.2} MeV (expected ~15.7)", sn_o16);
        assert!(
            sn_o16 > 10.0 && sn_o16 < 25.0,
            "O-16 S_n out of range: {}",
            sn_o16
        );

        // Proton separation energy for Ca-48 (Z=20, N=28): S_p ~ 15.8 MeV
        let sp_ca48 = proton_separation_energy(&predictor, 20, 28);
        println!("S_p(Ca-48) = {:.2} MeV (expected ~15.8)", sp_ca48);
        assert!(
            sp_ca48 > 5.0 && sp_ca48 < 25.0,
            "Ca-48 S_p out of range: {}",
            sp_ca48
        );

        // Alpha separation energy for Pb-208 (Z=82, N=126): S_alpha should be large (doubly magic)
        let sa_pb208 = alpha_separation_energy(&predictor, 82, 126);
        println!("S_alpha(Pb-208) = {:.2} MeV", sa_pb208);
    }

    #[test]
    fn test_q_value_functions() {
        let predictor = MlMassPredictor::new();

        // Q(n,gamma) for Fe-56 target: should be positive (~7.6 MeV)
        let q_fe56 = q_neutron_capture(&predictor, 26, 30);
        println!("Q(n,gamma) Fe-56 = {:.2} MeV (expected ~7.6)", q_fe56);
        assert!(
            q_fe56 > 4.0 && q_fe56 < 12.0,
            "Fe-56 (n,gamma) Q out of range: {}",
            q_fe56
        );

        // Q(p,n) for O-18 -> F-18: should be endothermic (~-2.4 MeV)
        let q_o18 = q_proton_neutron(&predictor, 8, 10);
        println!("Q(p,n) O-18 = {:.2} MeV (expected ~-2.4)", q_o18);
    }

    #[test]
    fn test_generator_equilibrium_classification() {
        let db = medical_isotope_database();

        println!("\n=== GENERATOR SYSTEMS ===");
        println!(
            "{:<12} {:<20} {:>14} {:>14} {:>10} {}",
            "Daughter", "Parent", "Parent t1/2", "Daughter t1/2", "Ratio", "Equilibrium"
        );
        println!("{}", "-".repeat(90));

        for iso in &db {
            if let ProductionRoute::Generator {
                parent,
                parent_half_life_s,
                ..
            } = &iso.production
            {
                let ratio = parent_half_life_s / iso.half_life_s;
                let eq = if ratio > 100.0 {
                    "secular"
                } else if ratio > 1.0 {
                    "transient"
                } else {
                    "none"
                };

                println!(
                    "{:<12} {:<20} {:>14} {:>14} {:>10.1} {}",
                    iso.name,
                    parent,
                    format_half_life(*parent_half_life_s),
                    format_half_life(iso.half_life_s),
                    ratio,
                    eq,
                );
            }
        }

        // Mo-99/Tc-99m should be transient equilibrium (65.9h / 6.0h ~ 11)
        let tc99m = db.iter().find(|i| i.name == "Tc-99m").unwrap();
        if let ProductionRoute::Generator {
            parent_half_life_s, ..
        } = &tc99m.production
        {
            let ratio = parent_half_life_s / tc99m.half_life_s;
            assert!(
                ratio > 5.0 && ratio < 20.0,
                "Mo-99/Tc-99m ratio should be ~11, got {}",
                ratio
            );
        }
    }

    // ── Helper for test output ───────────────────────────────────────────

    fn format_half_life(seconds: f64) -> String {
        if seconds < 60.0 {
            format!("{:.1} s", seconds)
        } else if seconds < 3600.0 {
            format!("{:.1} min", seconds / 60.0)
        } else if seconds < 86400.0 {
            format!("{:.2} h", seconds / 3600.0)
        } else if seconds < 365.25 * 86400.0 {
            format!("{:.2} d", seconds / 86400.0)
        } else {
            format!("{:.1} y", seconds / (365.25 * 86400.0))
        }
    }
}
