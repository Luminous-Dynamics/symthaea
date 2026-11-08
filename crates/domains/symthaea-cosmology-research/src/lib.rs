// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-first contracts for cosmology research.
//!
//! Numerical cosmology belongs in established external backends. This crate
//! controls what Symthaea is allowed to conclude from those backends.

pub mod identity;
pub mod reproduction;

use identity::{GitObjectId, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Epistemically distinct claims. Evidence never inherits authority merely
/// because a higher layer can explain a lower-layer result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimLayer {
    /// A reproducible feature is difficult to account for under the baseline.
    ObservationalAnomaly,
    /// Multiple independent reconstructions support non-constant effective
    /// dark-energy behaviour.
    PhenomenologicalDynamics,
    /// A concrete physical theory survives consistency and predictive tests.
    PhysicalMechanism,
}

impl ClaimLayer {
    pub const fn rank(self) -> u8 {
        match self {
            Self::ObservationalAnomaly => 0,
            Self::PhenomenologicalDynamics => 1,
            Self::PhysicalMechanism => 2,
        }
    }

    pub const fn is_adjacent_successor_of(self, prior: Self) -> bool {
        self.rank() == prior.rank() + 1
    }
}

/// Scientific datasets/probes used as an axis of the evidence cube.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetFamily {
    DesiDr2Bao,
    DesiDr2LymanAlphaFullShape,
    Cmb,
    Supernova,
    WeakLensing,
    GrowthRate,
    SyntheticLcdm,
    SyntheticDynamicDarkEnergy,
    Other(String),
}

/// Mathematical representation used to describe late-time acceleration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepresentationFamily {
    CplW0Wa,
    DirectRhoDe,
    BinnedW,
    SplineW,
    GaussianProcessW,
    PcaEigenmodes,
    PhysicalModel,
}

/// Statistical/inference lane. Agreement across lanes is evidence; no lane is
/// privileged by type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceFamily {
    FrequentistProfile,
    BayesianPosterior,
    BayesianEvidence,
    PosteriorPredictive,
    SimulationCalibration,
}

/// Systematics treatment used for a result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystematicsFamily {
    Baseline,
    CalibrationShift,
    CovariancePerturbation,
    SelectionFunction,
    RedshiftPerturbation,
    NuisanceModel,
    PriorSensitivity,
    Other(String),
}

/// One coordinate of the dataset × representation × inference × systematics
/// evidence cube.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCubeCoordinate {
    pub dataset: DatasetFamily,
    pub representation: RepresentationFamily,
    pub inference: InferenceFamily,
    pub systematics: SystematicsFamily,
}

/// DE-001 is intentionally a staged falsification protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum De001Gate {
    ExactReproduction,
    IndependentInference,
    RepresentationInvariance,
    DatasetLocalization,
    SystematicsAdversary,
    LcdmCounterfeitChallenge,
    SignalInjection,
    HoldoutUnblinding,
    IndependentReplication,
    Qualification,
}

impl De001Gate {
    pub const ORDER: [Self; 10] = [
        Self::ExactReproduction,
        Self::IndependentInference,
        Self::RepresentationInvariance,
        Self::DatasetLocalization,
        Self::SystematicsAdversary,
        Self::LcdmCounterfeitChallenge,
        Self::SignalInjection,
        Self::HoldoutUnblinding,
        Self::IndependentReplication,
        Self::Qualification,
    ];

    pub const fn ordinal(self) -> usize {
        match self {
            Self::ExactReproduction => 0,
            Self::IndependentInference => 1,
            Self::RepresentationInvariance => 2,
            Self::DatasetLocalization => 3,
            Self::SystematicsAdversary => 4,
            Self::LcdmCounterfeitChallenge => 5,
            Self::SignalInjection => 6,
            Self::HoldoutUnblinding => 7,
            Self::IndependentReplication => 8,
            Self::Qualification => 9,
        }
    }

    pub const fn next(self) -> Option<Self> {
        let idx = self.ordinal() + 1;
        if idx < Self::ORDER.len() {
            Some(Self::ORDER[idx])
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationVerdict {
    Pass,
    Mixed,
    Negative,
    Null,
    Indeterminate,
}

/// Minimal provenance needed before a DE-001 result can be considered for
/// confirmatory interpretation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    pub experiment_id: String,
    pub protocol_version: String,
    pub subject_commit: GitObjectId,
    pub config_digest: Sha256Digest,
    pub dataset_digest: Sha256Digest,
    pub result_digest: Sha256Digest,
    pub gate: De001Gate,
    pub verdict: QualificationVerdict,
    pub coordinate: EvidenceCubeCoordinate,
    pub claim_layer: ClaimLayer,
    /// True when analysis/model development continued after confirmatory
    /// holdout information was exposed. Such a lineage cannot license a
    /// confirmatory claim without a new holdout.
    pub development_after_unblinding: bool,
}

impl EvidenceReceipt {
    pub fn has_complete_identity(&self) -> bool {
        !self.experiment_id.trim().is_empty() && !self.protocol_version.trim().is_empty()
    }

    /// DE-001 can license *only* the observational-anomaly layer.
    pub fn licenses_de001_observational_anomaly(&self) -> bool {
        self.has_complete_identity()
            && self.gate == De001Gate::Qualification
            && self.verdict == QualificationVerdict::Pass
            && self.claim_layer == ClaimLayer::ObservationalAnomaly
            && !self.development_after_unblinding
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimBoundaryError {
    PriorLayerNotQualified,
    NonAdjacentPromotion,
}

impl fmt::Display for ClaimBoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PriorLayerNotQualified => {
                write!(f, "prior claim layer is not independently qualified")
            }
            Self::NonAdjacentPromotion => {
                write!(f, "claim promotion must advance exactly one epistemic layer")
            }
        }
    }
}

impl std::error::Error for ClaimBoundaryError {}

/// A tiny fail-closed state machine for claim promotion.
///
/// Qualification of one layer permits *testing* the next layer. It does not
/// automatically qualify the next layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimState {
    pub layer: ClaimLayer,
    pub qualified: bool,
}

impl ClaimState {
    pub const fn new(layer: ClaimLayer) -> Self {
        Self {
            layer,
            qualified: false,
        }
    }

    pub const fn qualified(layer: ClaimLayer) -> Self {
        Self {
            layer,
            qualified: true,
        }
    }

    pub fn begin_next_layer(self, next: ClaimLayer) -> Result<Self, ClaimBoundaryError> {
        if !self.qualified {
            return Err(ClaimBoundaryError::PriorLayerNotQualified);
        }
        if !next.is_adjacent_successor_of(self.layer) {
            return Err(ClaimBoundaryError::NonAdjacentPromotion);
        }
        Ok(Self::new(next))
    }
}

/// Direct dark-energy-density reconstruction point:
/// X(z) = rho_DE(z) / rho_DE(0). LCDM predicts X(z) = 1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DarkEnergyDensityPoint {
    pub redshift: f64,
    pub x: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityPointError {
    NonFinite,
    NegativeRedshift,
    NonPositiveUncertainty,
}

impl DarkEnergyDensityPoint {
    pub fn new(redshift: f64, x: f64, sigma: f64) -> Result<Self, DensityPointError> {
        if !redshift.is_finite() || !x.is_finite() || !sigma.is_finite() {
            return Err(DensityPointError::NonFinite);
        }
        if redshift < 0.0 {
            return Err(DensityPointError::NegativeRedshift);
        }
        if sigma <= 0.0 {
            return Err(DensityPointError::NonPositiveUncertainty);
        }
        Ok(Self {
            redshift,
            x,
            sigma,
        })
    }

    /// Local consistency check only; this is not a global model-comparison
    /// statistic and must not be reported as one.
    pub fn is_locally_lcdm_consistent(self, n_sigma: f64) -> bool {
        n_sigma.is_finite()
            && n_sigma >= 0.0
            && (self.x - 1.0).abs() <= n_sigma * self.sigma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Digest {
        Sha256Digest::parse(&byte.to_string().repeat(64)).unwrap()
    }

    fn coordinate() -> EvidenceCubeCoordinate {
        EvidenceCubeCoordinate {
            dataset: DatasetFamily::SyntheticLcdm,
            representation: RepresentationFamily::DirectRhoDe,
            inference: InferenceFamily::SimulationCalibration,
            systematics: SystematicsFamily::Baseline,
        }
    }

    #[test]
    fn de001_gate_order_is_total_and_monotonic() {
        for window in De001Gate::ORDER.windows(2) {
            assert_eq!(window[0].next(), Some(window[1]));
            assert_eq!(window[1].ordinal(), window[0].ordinal() + 1);
        }
        assert_eq!(De001Gate::Qualification.next(), None);
    }

    #[test]
    fn cannot_skip_claim_layers() {
        let anomaly = ClaimState::qualified(ClaimLayer::ObservationalAnomaly);
        assert_eq!(
            anomaly.begin_next_layer(ClaimLayer::PhysicalMechanism),
            Err(ClaimBoundaryError::NonAdjacentPromotion)
        );
    }

    #[test]
    fn cannot_promote_unqualified_layer() {
        let anomaly = ClaimState::new(ClaimLayer::ObservationalAnomaly);
        assert_eq!(
            anomaly.begin_next_layer(ClaimLayer::PhenomenologicalDynamics),
            Err(ClaimBoundaryError::PriorLayerNotQualified)
        );
    }

    #[test]
    fn promotion_opens_unqualified_next_layer() {
        let anomaly = ClaimState::qualified(ClaimLayer::ObservationalAnomaly);
        let dynamics = anomaly
            .begin_next_layer(ClaimLayer::PhenomenologicalDynamics)
            .unwrap();
        assert_eq!(dynamics.layer, ClaimLayer::PhenomenologicalDynamics);
        assert!(!dynamics.qualified);
    }

    #[test]
    fn only_clean_de001_q_receipt_licenses_anomaly_claim() {
        let receipt = EvidenceReceipt {
            experiment_id: "DE-001".into(),
            protocol_version: "v1".into(),
            subject_commit: GitObjectId::parse(&"a".repeat(40)).unwrap(),
            config_digest: digest('b'),
            dataset_digest: digest('c'),
            result_digest: digest('d'),
            gate: De001Gate::Qualification,
            verdict: QualificationVerdict::Pass,
            coordinate: coordinate(),
            claim_layer: ClaimLayer::ObservationalAnomaly,
            development_after_unblinding: false,
        };
        assert!(receipt.licenses_de001_observational_anomaly());

        let mut contaminated = receipt.clone();
        contaminated.development_after_unblinding = true;
        assert!(!contaminated.licenses_de001_observational_anomaly());

        let mut overclaim = receipt;
        overclaim.claim_layer = ClaimLayer::PhenomenologicalDynamics;
        assert!(!overclaim.licenses_de001_observational_anomaly());
    }

    #[test]
    fn receipt_deserialization_rejects_fake_digests() {
        let json = r#"{
            "experiment_id":"DE-001",
            "protocol_version":"v1",
            "subject_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "config_digest":"not-a-digest",
            "dataset_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "result_digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "gate":"Qualification",
            "verdict":"Pass",
            "coordinate":{
                "dataset":"SyntheticLcdm",
                "representation":"DirectRhoDe",
                "inference":"SimulationCalibration",
                "systematics":"Baseline"
            },
            "claim_layer":"ObservationalAnomaly",
            "development_after_unblinding":false
        }"#;
        assert!(serde_json::from_str::<EvidenceReceipt>(json).is_err());
    }

    #[test]
    fn density_points_fail_closed_on_invalid_inputs() {
        assert_eq!(
            DarkEnergyDensityPoint::new(-0.1, 1.0, 0.1),
            Err(DensityPointError::NegativeRedshift)
        );
        assert_eq!(
            DarkEnergyDensityPoint::new(0.1, 1.0, 0.0),
            Err(DensityPointError::NonPositiveUncertainty)
        );
        assert_eq!(
            DarkEnergyDensityPoint::new(0.1, f64::NAN, 0.1),
            Err(DensityPointError::NonFinite)
        );
    }

    #[test]
    fn lcdm_density_consistency_is_explicitly_local() {
        let point = DarkEnergyDensityPoint::new(0.3, 1.05, 0.03).unwrap();
        assert!(point.is_locally_lcdm_consistent(2.0));
        assert!(!point.is_locally_lcdm_consistent(1.0));
    }
}
