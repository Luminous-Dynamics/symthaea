//! Typed, evidence-preserving intake for Subterra v0.1 site and network records.
//!
//! This adapter follows the public JSON record shapes in Symtropy PR #5486.
//! It preserves the separate states of morphology, origin, connectivity,
//! accessibility, interior survey, engineering assessment and preservation.
//!
//! It deliberately does not:
//! - authenticate or dereference evidence references;
//! - infer posterior probabilities from qualitative claim states;
//! - promote a candidate/geophysical edge into observed topology;
//! - infer traversal from physical connectivity;
//! - grant sample, biological, engineering or actuator authority;
//! - validate planetary coordinate transforms, SLAM, geomechanics or safety.

pub mod profile;

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

const SITE_SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PlanetaryBody {
    Earth,
    Moon,
    Mars,
    Venus,
    Mercury,
    Europa,
    Enceladus,
    Ganymede,
    Callisto,
    Titan,
    Asteroid,
    Comet,
    Other,
}

impl PlanetaryBody {
    pub const fn is_earth(self) -> bool {
        matches!(self, Self::Earth)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimState {
    Candidate,
    Supported,
    DirectlyObserved,
    Contradicted,
    Unknown,
}

impl ClaimState {
    const fn requires_evidence(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MorphologyClass {
    Pit,
    Skylight,
    CollapseChain,
    LinearDepression,
    CavityOpening,
    SurfaceFracture,
    MappedVoid,
    EngineeredGallery,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeologicalOrigin {
    VolcanicLavaTube,
    Collapse,
    KarstDissolution,
    TectonicFracture,
    IceRelated,
    ImpactRelated,
    EngineeredExcavation,
    Other,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityState {
    Unknown,
    Candidate,
    GeophysicallySupported,
    DirectlyObservedConnected,
    Contradicted,
}

impl ConnectivityState {
    const fn requires_evidence(self) -> bool {
        matches!(
            self,
            Self::GeophysicallySupported | Self::DirectlyObservedConnected
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessibilityState {
    Unknown,
    Candidate,
    RobotAccessDemonstrated,
    HumanAccessAssessed,
    Inaccessible,
    OutOfDomain,
}

impl AccessibilityState {
    const fn requires_access_details(self) -> bool {
        matches!(
            self,
            Self::RobotAccessDemonstrated | Self::HumanAccessAssessed
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteriorSurveyState {
    NotSurveyed,
    Partial,
    SurveyedWithinDeclaredExtent,
    Invalid,
}

impl InteriorSurveyState {
    const fn has_survey_extent(self) -> bool {
        matches!(self, Self::Partial | Self::SurveyedWithinDeclaredExtent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineeringUseCase {
    Science,
    RobotStaging,
    Storage,
    ResourceProcessing,
    Industrial,
    PressurizedModule,
    Habitation,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineeringClaimState {
    NotAssessed,
    Candidate,
    AssessedWithLimitations,
    QualifiedForDeclaredEnvelope,
    Rejected,
    OutOfDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanetaryProtectionState {
    NotApplicable,
    NotAssessed,
    ReviewRequired,
    Restricted,
    AuthorizedForDeclaredActivity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityPermission {
    NotAssessed,
    Prohibited,
    ReviewRequired,
    AuthorizedForDeclaredActivity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationModality {
    Optical,
    Stereo,
    ThermalIr,
    Sar,
    RadarSounder,
    Gpr,
    Lidar,
    Photogrammetry,
    Seismic,
    Gravity,
    Magnetic,
    Spectroscopy,
    InSituRobot,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coordinates {
    pub coordinate_frame: String,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    #[serde(default)]
    pub elevation_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphologyClaim {
    pub class: MorphologyClass,
    pub state: ClaimState,
    pub description: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginHypothesis {
    pub mechanism: GeologicalOrigin,
    pub state: ClaimState,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectivityClaim {
    pub state: ConnectivityState,
    #[serde(default)]
    pub known_extent_m: Option<f64>,
    pub basis: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessibilityClaim {
    pub state: AccessibilityState,
    #[serde(default)]
    pub access_method: Option<String>,
    #[serde(default)]
    pub egress_method: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteriorSurveyClaim {
    pub state: InteriorSurveyState,
    #[serde(default)]
    pub surveyed_extent_m: Option<f64>,
    #[serde(default)]
    pub coverage_fraction: Option<f64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineeringAssessment {
    pub use_case: EngineeringUseCase,
    pub state: EngineeringClaimState,
    #[serde(default)]
    pub envelope_id: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreservationAssessment {
    pub planetary_protection_state: PlanetaryProtectionState,
    pub sampling_permission: ActivityPermission,
    pub biological_activity_permission: ActivityPermission,
    pub basis_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationRecord {
    pub observation_id: String,
    pub modality: ObservationModality,
    pub captured_at: String,
    #[serde(default)]
    pub native_resolution: Option<String>,
    pub coordinate_frame: String,
    pub provenance_ref: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteEvidenceRecord {
    pub schema_version: String,
    pub site_id: String,
    pub body: PlanetaryBody,
    #[serde(default)]
    pub coordinates: Option<Coordinates>,
    pub morphology: MorphologyClaim,
    pub origin_hypotheses: Vec<OriginHypothesis>,
    pub connectivity: ConnectivityClaim,
    pub accessibility: AccessibilityClaim,
    pub interior_survey: InteriorSurveyClaim,
    pub engineering_assessments: Vec<EngineeringAssessment>,
    pub preservation: PreservationAssessment,
    pub observations: Vec<ObservationRecord>,
    pub unknowns: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiteIssue {
    UnsupportedSchemaVersion(String),
    EmptySiteId,
    EmptyCoordinateFrame,
    InvalidLatitude,
    InvalidLongitude,
    InvalidElevation,
    InvalidConnectivityExtent,
    InvalidSurveyExtent,
    InvalidSurveyCoverage,
    MissingEvidenceReference(&'static str),
    EmptyEvidenceReference(&'static str),
    DuplicateEvidenceReference(&'static str),
    DuplicateEngineeringUseCase,
    DemonstratedAccessMissingMethod,
    DemonstratedAccessMissingEgress,
    DemonstratedAccessMissingEvidence,
    QualifiedEngineeringMissingEnvelope,
    QualifiedEngineeringMissingEvidence,
    DuplicateObservationId(String),
    EmptyObservationField(&'static str),
    NoObservations,
    AuthorizedActivityMissingBasis,
    AuthorizedActivityWithoutProtectionAuthorization,
    RestrictedSiteHasDeclaredAuthorizedActivity,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn inspect_refs(
    refs: &[String],
    field: &'static str,
    required: bool,
    issues: &mut Vec<SiteIssue>,
) {
    if required && refs.is_empty() {
        issues.push(SiteIssue::MissingEvidenceReference(field));
    }

    let mut seen = BTreeSet::new();
    for reference in refs {
        if !nonempty(reference) {
            issues.push(SiteIssue::EmptyEvidenceReference(field));
        }
        if !seen.insert(reference.as_str()) {
            issues.push(SiteIssue::DuplicateEvidenceReference(field));
        }
    }
}

impl SiteEvidenceRecord {
    /// Check local record semantics without collapsing independent claims.
    ///
    /// In particular, directly observed connectivity is NOT required to imply
    /// demonstrated robot access or a completed interior survey. An interior
    /// survey is also not proof that every network edge is traversable.
    pub fn validate(&self) -> Result<(), Vec<SiteIssue>> {
        let mut issues = Vec::new();

        if self.schema_version != SITE_SCHEMA_VERSION {
            issues.push(SiteIssue::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if !nonempty(&self.site_id) {
            issues.push(SiteIssue::EmptySiteId);
        }

        if let Some(coordinates) = &self.coordinates {
            if !nonempty(&coordinates.coordinate_frame) {
                issues.push(SiteIssue::EmptyCoordinateFrame);
            }
            if !coordinates.latitude_deg.is_finite()
                || !(-90.0..=90.0).contains(&coordinates.latitude_deg)
            {
                issues.push(SiteIssue::InvalidLatitude);
            }
            if !coordinates.longitude_deg.is_finite()
                || !(-360.0..=360.0).contains(&coordinates.longitude_deg)
            {
                issues.push(SiteIssue::InvalidLongitude);
            }
            if coordinates
                .elevation_m
                .is_some_and(|value| !value.is_finite())
            {
                issues.push(SiteIssue::InvalidElevation);
            }
        }

        inspect_refs(
            &self.morphology.evidence_refs,
            "morphology",
            self.morphology.state.requires_evidence(),
            &mut issues,
        );

        for hypothesis in &self.origin_hypotheses {
            inspect_refs(
                &hypothesis.evidence_refs,
                "origin_hypotheses",
                hypothesis.state.requires_evidence(),
                &mut issues,
            );
        }

        if self
            .connectivity
            .known_extent_m
            .is_some_and(|extent| !extent.is_finite() || extent < 0.0)
        {
            issues.push(SiteIssue::InvalidConnectivityExtent);
        }
        inspect_refs(
            &self.connectivity.evidence_refs,
            "connectivity",
            self.connectivity.state.requires_evidence(),
            &mut issues,
        );

        if self.accessibility.state.requires_access_details() {
            if !self
                .accessibility
                .access_method
                .as_deref()
                .is_some_and(nonempty)
            {
                issues.push(SiteIssue::DemonstratedAccessMissingMethod);
            }
            if !self
                .accessibility
                .egress_method
                .as_deref()
                .is_some_and(nonempty)
            {
                issues.push(SiteIssue::DemonstratedAccessMissingEgress);
            }
            if self.accessibility.evidence_refs.is_empty() {
                issues.push(SiteIssue::DemonstratedAccessMissingEvidence);
            }
            inspect_refs(
                &self.accessibility.evidence_refs,
                "accessibility",
                false,
                &mut issues,
            );
        } else {
            inspect_refs(
                &self.accessibility.evidence_refs,
                "accessibility",
                false,
                &mut issues,
            );
        }

        if self.interior_survey.state.has_survey_extent() {
            if !self
                .interior_survey
                .surveyed_extent_m
                .is_some_and(|extent| extent.is_finite() && extent > 0.0)
            {
                issues.push(SiteIssue::InvalidSurveyExtent);
            }
            if !self
                .interior_survey
                .coverage_fraction
                .is_some_and(|coverage| {
                    coverage.is_finite() && coverage > 0.0 && coverage <= 1.0
                })
            {
                issues.push(SiteIssue::InvalidSurveyCoverage);
            }
            inspect_refs(
                &self.interior_survey.evidence_refs,
                "interior_survey",
                true,
                &mut issues,
            );
        } else {
            if self.interior_survey.surveyed_extent_m.is_some()
                && self.interior_survey.state == InteriorSurveyState::NotSurveyed
            {
                issues.push(SiteIssue::InvalidSurveyExtent);
            }
            inspect_refs(
                &self.interior_survey.evidence_refs,
                "interior_survey",
                false,
                &mut issues,
            );
        }

        let mut use_cases = BTreeSet::new();
        for assessment in &self.engineering_assessments {
            if !use_cases.insert(assessment.use_case as u8) {
                // The schema permits duplicates; the typed intake rejects them
                // because contradictory states for one use case are ambiguous.
                issues.push(SiteIssue::DuplicateEngineeringUseCase);
            }
            if assessment.state == EngineeringClaimState::QualifiedForDeclaredEnvelope {
                if !assessment.envelope_id.as_deref().is_some_and(nonempty) {
                    issues.push(SiteIssue::QualifiedEngineeringMissingEnvelope);
                }
                if assessment.evidence_refs.is_empty() {
                    issues.push(SiteIssue::QualifiedEngineeringMissingEvidence);
                }
                inspect_refs(
                    &assessment.evidence_refs,
                    "engineering_assessments",
                    true,
                    &mut issues,
                );
            } else {
                inspect_refs(
                    &assessment.evidence_refs,
                    "engineering_assessments",
                    false,
                    &mut issues,
                );
            }
        }

        if self.observations.is_empty() {
            issues.push(SiteIssue::NoObservations);
        }
        let mut observation_ids = BTreeSet::new();
        for observation in &self.observations {
            if !nonempty(&observation.observation_id) {
                issues.push(SiteIssue::EmptyObservationField("observation_id"));
            }
            if !observation_ids.insert(observation.observation_id.as_str()) {
                issues.push(SiteIssue::DuplicateObservationId(
                    observation.observation_id.clone(),
                ));
            }
            if !nonempty(&observation.captured_at) {
                issues.push(SiteIssue::EmptyObservationField("captured_at"));
            }
            if !nonempty(&observation.coordinate_frame) {
                issues.push(SiteIssue::EmptyObservationField("coordinate_frame"));
            }
            if !nonempty(&observation.provenance_ref) {
                issues.push(SiteIssue::EmptyObservationField("provenance_ref"));
            }
            if observation
                .native_resolution
                .as_deref()
                .is_some_and(|value| !nonempty(value))
            {
                issues.push(SiteIssue::EmptyObservationField("native_resolution"));
            }
        }

        let sampling_authorized = self.preservation.sampling_permission
            == ActivityPermission::AuthorizedForDeclaredActivity;
        let biology_authorized = self.preservation.biological_activity_permission
            == ActivityPermission::AuthorizedForDeclaredActivity;
        let any_activity_authorized = sampling_authorized || biology_authorized;

        if any_activity_authorized && self.preservation.basis_refs.is_empty() {
            issues.push(SiteIssue::AuthorizedActivityMissingBasis);
        }
        inspect_refs(
            &self.preservation.basis_refs,
            "preservation.basis_refs",
            false,
            &mut issues,
        );

        if any_activity_authorized
            && !self.body.is_earth()
            && self.preservation.planetary_protection_state
                != PlanetaryProtectionState::AuthorizedForDeclaredActivity
        {
            issues.push(SiteIssue::AuthorizedActivityWithoutProtectionAuthorization);
        }

        // A declared non-Earth permission is preserved as source data. The
        // converted belief input always marks planetary-protection review as
        // required, so this adapter cannot turn the declaration into action
        // authority. The v0.1 record has no named authorization-case field.
        if self.preservation.planetary_protection_state == PlanetaryProtectionState::Restricted
            && any_activity_authorized
        {
            issues.push(SiteIssue::RestrictedSiteHasDeclaredAuthorizedActivity);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Convert the source record into a typed, non-probabilistic belief input.
    /// This copies claim states and evidence references; it does not infer
    /// priors, posterior probabilities, independence or operational authority.
    pub fn into_belief_input(self) -> SiteBeliefInput {
        let planetary_protection_review_required = !self.body.is_earth();
        let has_declared_authorized_activity = self.preservation.sampling_permission
            == ActivityPermission::AuthorizedForDeclaredActivity
            || self.preservation.biological_activity_permission
                == ActivityPermission::AuthorizedForDeclaredActivity;

        SiteBeliefInput {
            site_id: self.site_id,
            body: self.body,
            coordinates: self.coordinates,
            morphology: EvidenceClaim {
                value: self.morphology.class,
                state: self.morphology.state,
                evidence_refs: self.morphology.evidence_refs,
                description: self.morphology.description,
            },
            origin_hypotheses: self
                .origin_hypotheses
                .into_iter()
                .map(|hypothesis| OriginBeliefHypothesis {
                    mechanism: hypothesis.mechanism,
                    state: hypothesis.state,
                    rationale: hypothesis.rationale,
                    evidence_refs: hypothesis.evidence_refs,
                })
                .collect(),
            connectivity: ConnectivityBelief {
                state: self.connectivity.state,
                known_extent_m: self.connectivity.known_extent_m,
                basis: self.connectivity.basis,
                evidence_refs: self.connectivity.evidence_refs,
            },
            accessibility: AccessibilityBelief {
                state: self.accessibility.state,
                access_method: self.accessibility.access_method,
                egress_method: self.accessibility.egress_method,
                evidence_refs: self.accessibility.evidence_refs,
            },
            interior_survey: InteriorSurveyBelief {
                state: self.interior_survey.state,
                surveyed_extent_m: self.interior_survey.surveyed_extent_m,
                coverage_fraction: self.interior_survey.coverage_fraction,
                evidence_refs: self.interior_survey.evidence_refs,
            },
            engineering_assessments: self.engineering_assessments,
            preservation: self.preservation,
            observations: self.observations,
            unknowns: self.unknowns,
            planetary_protection_review_required,
            has_declared_authorized_activity,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceClaim<T> {
    pub value: T,
    pub state: ClaimState,
    pub evidence_refs: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OriginBeliefHypothesis {
    pub mechanism: GeologicalOrigin,
    pub state: ClaimState,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectivityBelief {
    pub state: ConnectivityState,
    pub known_extent_m: Option<f64>,
    pub basis: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessibilityBelief {
    pub state: AccessibilityState,
    pub access_method: Option<String>,
    pub egress_method: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InteriorSurveyBelief {
    pub state: InteriorSurveyState,
    pub surveyed_extent_m: Option<f64>,
    pub coverage_fraction: Option<f64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SiteBeliefInput {
    pub site_id: String,
    pub body: PlanetaryBody,
    pub coordinates: Option<Coordinates>,
    pub morphology: EvidenceClaim<MorphologyClass>,
    /// Qualitative hypotheses only. No probability is fabricated by intake.
    pub origin_hypotheses: Vec<OriginBeliefHypothesis>,
    pub connectivity: ConnectivityBelief,
    pub accessibility: AccessibilityBelief,
    pub interior_survey: InteriorSurveyBelief,
    pub engineering_assessments: Vec<EngineeringAssessment>,
    pub preservation: PreservationAssessment,
    pub observations: Vec<ObservationRecord>,
    pub unknowns: Vec<String>,
    /// Non-Earth activity proposals always require the responsible external
    /// planetary-protection review; this is not itself an authorization.
    pub planetary_protection_review_required: bool,
    /// Mirrors the source declaration; it is not verified permission.
    pub has_declared_authorized_activity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiteIntakeError {
    Json(String),
    Semantic(Vec<SiteIssue>),
}

impl std::fmt::Display for SiteIntakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SiteIntakeError {}

pub fn parse_site_record(input: &str) -> Result<SiteEvidenceRecord, SiteIntakeError> {
    let record: SiteEvidenceRecord =
        serde_json::from_str(input).map_err(|error| SiteIntakeError::Json(error.to_string()))?;
    record.validate().map_err(SiteIntakeError::Semantic)?;
    Ok(record)
}

pub fn intake_site_json(input: &str) -> Result<SiteBeliefInput, SiteIntakeError> {
    Ok(parse_site_record(input)?.into_belief_input())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionEvidence {
    Unknown,
    Candidate,
    GeophysicallySupported,
    DirectlyObserved,
    Contradicted,
}

impl ConnectionEvidence {
    const fn enters_observed_topology(self) -> bool {
        matches!(self, Self::DirectlyObserved)
    }

    const fn requires_evidence(self) -> bool {
        matches!(
            self,
            Self::Candidate | Self::GeophysicallySupported | Self::DirectlyObserved
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraversabilityEvidence {
    Unknown,
    Candidate,
    Demonstrated,
    NonTraversable,
    OutOfDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeDirection {
    Undirected,
    AToB,
    BToA,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaveNetworkEdge {
    pub edge_id: String,
    pub node_a: String,
    pub node_b: String,
    pub length_m: f64,
    pub connection_evidence: ConnectionEvidence,
    pub traversability: TraversabilityEvidence,
    pub direction: EdgeDirection,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaveNetworkRecord {
    pub network_id: String,
    pub nodes: Vec<String>,
    pub edges: Vec<CaveNetworkEdge>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkIssue {
    EmptyNetworkId,
    NoNodes,
    EmptyNodeId,
    DuplicateNodeId(String),
    EmptyEdgeId,
    DuplicateEdgeId(String),
    UnknownEndpoint { edge_id: String, node_id: String },
    SelfLoop(String),
    InvalidEdgeLength(String),
    EdgeMissingEvidence(String),
    DemonstratedTraversalWithoutObservedConnection(String),
    ContradictedConnectionMarkedTraversable(String),
    EmptyEvidenceReference(String),
    DuplicateEvidenceReference(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteKnowledge {
    DemonstratedRoute {
        node_ids: Vec<String>,
        edge_ids: Vec<String>,
    },
    /// This is an evidence statement only, not proof of physical separation.
    NoDemonstratedRoute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkSummary {
    /// Components use directly observed connections irrespective of robot
    /// traversability. This is physical-topology evidence, not a route graph.
    pub directly_observed_components: Vec<Vec<String>>,
    pub candidate_edge_ids: Vec<String>,
    pub geophysically_supported_edge_ids: Vec<String>,
    pub contradicted_edge_ids: Vec<String>,
    pub unknown_edge_ids: Vec<String>,
    pub directly_observed_not_demonstrated_ids: Vec<String>,
    pub demonstrated_traversable_edge_ids: Vec<String>,
}

impl CaveNetworkRecord {
    pub fn validate(&self) -> Result<(), Vec<NetworkIssue>> {
        let mut issues = Vec::new();

        if !nonempty(&self.network_id) {
            issues.push(NetworkIssue::EmptyNetworkId);
        }
        if self.nodes.is_empty() {
            issues.push(NetworkIssue::NoNodes);
        }

        let mut node_ids = BTreeSet::new();
        for node_id in &self.nodes {
            if !nonempty(node_id) {
                issues.push(NetworkIssue::EmptyNodeId);
            }
            if !node_ids.insert(node_id.as_str()) {
                issues.push(NetworkIssue::DuplicateNodeId(node_id.clone()));
            }
        }

        let mut edge_ids = BTreeSet::new();
        for edge in &self.edges {
            if !nonempty(&edge.edge_id) {
                issues.push(NetworkIssue::EmptyEdgeId);
            }
            if !edge_ids.insert(edge.edge_id.as_str()) {
                issues.push(NetworkIssue::DuplicateEdgeId(edge.edge_id.clone()));
            }
            if !node_ids.contains(edge.node_a.as_str()) {
                issues.push(NetworkIssue::UnknownEndpoint {
                    edge_id: edge.edge_id.clone(),
                    node_id: edge.node_a.clone(),
                });
            }
            if !node_ids.contains(edge.node_b.as_str()) {
                issues.push(NetworkIssue::UnknownEndpoint {
                    edge_id: edge.edge_id.clone(),
                    node_id: edge.node_b.clone(),
                });
            }
            if edge.node_a == edge.node_b {
                issues.push(NetworkIssue::SelfLoop(edge.edge_id.clone()));
            }
            if !edge.length_m.is_finite() || edge.length_m <= 0.0 {
                issues.push(NetworkIssue::InvalidEdgeLength(edge.edge_id.clone()));
            }

            let mut refs = BTreeSet::new();
            for reference in &edge.evidence_refs {
                if !nonempty(reference) {
                    issues.push(NetworkIssue::EmptyEvidenceReference(
                        edge.edge_id.clone(),
                    ));
                }
                if !refs.insert(reference.as_str()) {
                    issues.push(NetworkIssue::DuplicateEvidenceReference(
                        edge.edge_id.clone(),
                    ));
                }
            }
            if edge.connection_evidence.requires_evidence() && refs.is_empty() {
                issues.push(NetworkIssue::EdgeMissingEvidence(edge.edge_id.clone()));
            }

            if edge.traversability == TraversabilityEvidence::Demonstrated {
                match edge.connection_evidence {
                    ConnectionEvidence::DirectlyObserved => {}
                    ConnectionEvidence::Contradicted => {
                        issues.push(
                            NetworkIssue::ContradictedConnectionMarkedTraversable(
                                edge.edge_id.clone(),
                            ),
                        );
                    }
                    _ => {
                        issues.push(
                            NetworkIssue::DemonstratedTraversalWithoutObservedConnection(
                                edge.edge_id.clone(),
                            ),
                        );
                    }
                }
                if refs.is_empty() {
                    issues.push(NetworkIssue::EdgeMissingEvidence(edge.edge_id.clone()));
                }
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Connected components of directly observed physical topology. This
    /// deliberately does not require a traversal demonstration.
    pub fn directly_observed_components(
        &self,
    ) -> Result<Vec<Vec<String>>, Vec<NetworkIssue>> {
        self.validate()?;

        let mut adjacency = BTreeMap::<String, Vec<String>>::new();
        for node in &self.nodes {
            adjacency.entry(node.clone()).or_default();
        }
        for edge in &self.edges {
            if edge.connection_evidence.enters_observed_topology() {
                adjacency
                    .entry(edge.node_a.clone())
                    .or_default()
                    .push(edge.node_b.clone());
                adjacency
                    .entry(edge.node_b.clone())
                    .or_default()
                    .push(edge.node_a.clone());
            }
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort();
            neighbours.dedup();
        }

        let mut visited = BTreeSet::new();
        let mut components = Vec::new();
        for node_id in adjacency.keys() {
            if visited.contains(node_id) {
                continue;
            }
            let mut queue = VecDeque::from([node_id.clone()]);
            let mut component = Vec::new();
            visited.insert(node_id.clone());

            while let Some(current) = queue.pop_front() {
                component.push(current.clone());
                if let Some(neighbours) = adjacency.get(&current) {
                    for neighbour in neighbours {
                        if visited.insert(neighbour.clone()) {
                            queue.push_back(neighbour.clone());
                        }
                    }
                }
            }

            component.sort();
            components.push(component);
        }
        components.sort_by(|left, right| left.first().cmp(&right.first()));
        Ok(components)
    }

    pub fn summarize(&self) -> Result<NetworkSummary, Vec<NetworkIssue>> {
        self.validate()?;

        let mut summary = NetworkSummary {
            directly_observed_components: self.directly_observed_components()?,
            candidate_edge_ids: Vec::new(),
            geophysically_supported_edge_ids: Vec::new(),
            contradicted_edge_ids: Vec::new(),
            unknown_edge_ids: Vec::new(),
            directly_observed_not_demonstrated_ids: Vec::new(),
            demonstrated_traversable_edge_ids: Vec::new(),
        };

        for edge in &self.edges {
            match edge.connection_evidence {
                ConnectionEvidence::Unknown => summary.unknown_edge_ids.push(edge.edge_id.clone()),
                ConnectionEvidence::Candidate => summary.candidate_edge_ids.push(edge.edge_id.clone()),
                ConnectionEvidence::GeophysicallySupported => {
                    summary.geophysically_supported_edge_ids.push(edge.edge_id.clone())
                }
                ConnectionEvidence::DirectlyObserved => {
                    if edge.traversability == TraversabilityEvidence::Demonstrated {
                        summary.demonstrated_traversable_edge_ids.push(edge.edge_id.clone());
                    } else {
                        summary.directly_observed_not_demonstrated_ids.push(edge.edge_id.clone());
                    }
                }
                ConnectionEvidence::Contradicted => {
                    summary.contradicted_edge_ids.push(edge.edge_id.clone())
                }
            }
        }

        summary.candidate_edge_ids.sort();
        summary.geophysically_supported_edge_ids.sort();
        summary.contradicted_edge_ids.sort();
        summary.unknown_edge_ids.sort();
        summary.directly_observed_not_demonstrated_ids.sort();
        summary.demonstrated_traversable_edge_ids.sort();

        Ok(summary)
    }

    /// Find a route only through directly observed edges whose traversal has
    /// separately been demonstrated. Directed edges preserve asymmetric
    /// access/retreat knowledge.
    pub fn demonstrated_route(
        &self,
        start_node: &str,
        goal_node: &str,
    ) -> Result<RouteKnowledge, Vec<NetworkIssue>> {
        self.validate()?;

        let node_ids = self.nodes.iter().map(String::as_str).collect::<BTreeSet<_>>();
        if !node_ids.contains(start_node) {
            return Err(vec![NetworkIssue::UnknownEndpoint {
                edge_id: "route-query".to_string(),
                node_id: start_node.to_string(),
            }]);
        }
        if !node_ids.contains(goal_node) {
            return Err(vec![NetworkIssue::UnknownEndpoint {
                edge_id: "route-query".to_string(),
                node_id: goal_node.to_string(),
            }]);
        }
        if start_node == goal_node {
            return Ok(RouteKnowledge::DemonstratedRoute {
                node_ids: vec![start_node.to_string()],
                edge_ids: Vec::new(),
            });
        }

        let mut adjacency = BTreeMap::<String, Vec<(String, String)>>::new();
        for edge in &self.edges {
            if edge.connection_evidence != ConnectionEvidence::DirectlyObserved
                || edge.traversability != TraversabilityEvidence::Demonstrated
            {
                continue;
            }
            match edge.direction {
                EdgeDirection::Undirected => {
                    adjacency.entry(edge.node_a.clone()).or_default()
                        .push((edge.node_b.clone(), edge.edge_id.clone()));
                    adjacency.entry(edge.node_b.clone()).or_default()
                        .push((edge.node_a.clone(), edge.edge_id.clone()));
                }
                EdgeDirection::AToB => {
                    adjacency.entry(edge.node_a.clone()).or_default()
                        .push((edge.node_b.clone(), edge.edge_id.clone()));
                }
                EdgeDirection::BToA => {
                    adjacency.entry(edge.node_b.clone()).or_default()
                        .push((edge.node_a.clone(), edge.edge_id.clone()));
                }
            }
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_by(|left, right| {
                left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
            });
        }

        let mut queue = VecDeque::from([start_node.to_string()]);
        let mut visited = BTreeSet::from([start_node.to_string()]);
        let mut predecessor = BTreeMap::<String, (String, String)>::new();

        while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = adjacency.get(&current) {
                for (neighbour, edge_id) in neighbours {
                    if visited.insert(neighbour.clone()) {
                        predecessor.insert(
                            neighbour.clone(),
                            (current.clone(), edge_id.clone()),
                        );
                        if neighbour.as_str() == goal_node {
                            let mut nodes = vec![goal_node.to_string()];
                            let mut edges = Vec::new();
                            let mut cursor = goal_node.to_string();
                            while cursor.as_str() != start_node {
                                let Some((previous, used_edge)) = predecessor.get(&cursor) else {
                                    return Ok(RouteKnowledge::NoDemonstratedRoute);
                                };
                                edges.push(used_edge.clone());
                                nodes.push(previous.clone());
                                cursor = previous.clone();
                            }
                            nodes.reverse();
                            edges.reverse();
                            return Ok(RouteKnowledge::DemonstratedRoute {
                                node_ids: nodes,
                                edge_ids: edges,
                            });
                        }
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }

        Ok(RouteKnowledge::NoDemonstratedRoute)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkIntakeError {
    Json(String),
    Semantic(Vec<NetworkIssue>),
}

impl std::fmt::Display for NetworkIntakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for NetworkIntakeError {}

pub fn parse_network_record(input: &str) -> Result<CaveNetworkRecord, NetworkIntakeError> {
    let record: CaveNetworkRecord =
        serde_json::from_str(input).map_err(|error| NetworkIntakeError::Json(error.to_string()))?;
    record.validate().map_err(NetworkIntakeError::Semantic)?;
    Ok(record)
}

pub fn intake_network_json(input: &str) -> Result<CaveNetworkRecord, NetworkIntakeError> {
    parse_network_record(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE_FIXTURE: &str = include_str!("../fixtures/site-evidence-synthetic.json");
    const NETWORK_FIXTURE: &str = include_str!("../fixtures/cave-network-synthetic.json");

    #[test]
    fn synthetic_site_fixture_deserializes_without_strengthening_claims() {
        let input = intake_site_json(SITE_FIXTURE).expect("valid synthetic site fixture");
        assert_eq!(input.body, PlanetaryBody::Mars);
        assert_eq!(input.morphology.state, ClaimState::Supported);
        assert_eq!(input.connectivity.state, ConnectivityState::Candidate);
        assert_eq!(input.accessibility.state, AccessibilityState::Unknown);
        assert_eq!(input.interior_survey.state, InteriorSurveyState::NotSurveyed);
        assert_eq!(input.origin_hypotheses.len(), 2);
        assert_eq!(input.origin_hypotheses[0].state, ClaimState::Candidate);
        assert_eq!(input.origin_hypotheses[1].state, ClaimState::Unknown);
        assert!(input.planetary_protection_review_required);
        assert!(!input.has_declared_authorized_activity);
    }

    #[test]
    fn directly_observed_connectivity_does_not_imply_access_or_survey() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.connectivity.state = ConnectivityState::DirectlyObservedConnected;
        record.connectivity.evidence_refs = vec!["direct-interior-observation".into()];

        assert!(record.validate().is_ok());
        assert_eq!(record.accessibility.state, AccessibilityState::Unknown);
        assert_eq!(record.interior_survey.state, InteriorSurveyState::NotSurveyed);
    }

    #[test]
    fn remote_or_tethered_interior_survey_does_not_imply_network_edge_traversal() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.interior_survey = InteriorSurveyClaim {
            state: InteriorSurveyState::Partial,
            surveyed_extent_m: Some(15.0),
            coverage_fraction: Some(0.25),
            evidence_refs: vec!["remote-survey-receipt".into()],
        };
        assert!(record.validate().is_ok());
        assert_eq!(record.accessibility.state, AccessibilityState::Unknown);
    }

    #[test]
    fn demonstrated_access_requires_access_and_egress_descriptions() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.accessibility.state = AccessibilityState::RobotAccessDemonstrated;
        record.accessibility.evidence_refs = vec!["robot-route-receipt".into()];
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&SiteIssue::DemonstratedAccessMissingMethod));
        record.accessibility.access_method = Some("tethered rover".into());
        record.accessibility.egress_method = Some("reverse tether".into());
        assert!(record.validate().is_ok());
    }

    #[test]
    fn qualified_engineering_claim_requires_envelope_and_evidence() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.engineering_assessments[0].state =
            EngineeringClaimState::QualifiedForDeclaredEnvelope;
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&SiteIssue::QualifiedEngineeringMissingEnvelope));
        record.engineering_assessments[0].envelope_id = Some("habitat-envelope-01".into());
        record.engineering_assessments[0].evidence_refs =
            vec!["qualification-report-01".into()];
        assert!(record.validate().is_ok());
    }

    #[test]
    fn non_earth_authorized_activity_is_preserved_but_requires_external_review() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.preservation.planetary_protection_state =
            PlanetaryProtectionState::AuthorizedForDeclaredActivity;
        record.preservation.sampling_permission =
            ActivityPermission::AuthorizedForDeclaredActivity;
        record.preservation.basis_refs = vec!["declared-review-reference".into()];

        // Preserve the source declaration while retaining the external
        // planetary-protection gate in the typed belief input.
        assert!(record.validate().is_ok());
        let input = record.into_belief_input();
        assert!(input.has_declared_authorized_activity);
        assert!(input.planetary_protection_review_required);
    }

    #[test]
    fn demonstrated_access_requires_its_own_evidence_reference() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.accessibility.state = AccessibilityState::RobotAccessDemonstrated;
        record.accessibility.access_method = Some("tethered rover".into());
        record.accessibility.egress_method = Some("reverse tether".into());
        record.accessibility.evidence_refs.clear();
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&SiteIssue::DemonstratedAccessMissingEvidence));
    }

    #[test]
    fn unsupported_site_schema_version_is_rejected() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.schema_version = "0.2".into();
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&SiteIssue::UnsupportedSchemaVersion("0.2".into())));
    }

    #[test]
    fn site_deserializer_rejects_unknown_top_level_fields() {
        let invalid = format!("{{\n  \"unexpected\": true,{}", &SITE_FIXTURE[1..]);
        assert!(matches!(
            intake_site_json(&invalid),
            Err(SiteIntakeError::Json(_))
        ));
    }

    #[test]
    fn non_earth_activity_declaration_requires_protection_state() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.preservation.sampling_permission =
            ActivityPermission::AuthorizedForDeclaredActivity;
        record.preservation.basis_refs = vec!["declared-review-reference".into()];
        assert!(record
            .validate()
            .unwrap_err()
            .contains(&SiteIssue::AuthorizedActivityWithoutProtectionAuthorization));
    }

    #[test]
    fn duplicate_observation_ids_are_rejected() {
        let mut record: SiteEvidenceRecord =
            serde_json::from_str(SITE_FIXTURE).expect("fixture JSON");
        record.observations[1].observation_id =
            record.observations[0].observation_id.clone();
        assert!(record
            .validate()
            .unwrap_err()
            .iter()
            .any(|issue| matches!(issue, SiteIssue::DuplicateObservationId(_))));
    }

    #[test]
    fn network_fixture_preserves_topology_and_traversal_separately() {
        let network =
            intake_network_json(NETWORK_FIXTURE).expect("valid synthetic network fixture");
        let summary = network.summarize().expect("valid topology");
        assert_eq!(
            summary.directly_observed_components,
            vec![
                vec![
                    "chamber-a".to_string(),
                    "chamber-b".to_string(),
                    "entrance".to_string()
                ],
                vec!["suspected-chamber".to_string()]
            ]
        );
        assert_eq!(
            summary.demonstrated_traversable_edge_ids,
            vec!["entrance-to-a".to_string()]
        );
        assert_eq!(
            summary.directly_observed_not_demonstrated_ids,
            vec!["a-to-b".to_string()]
        );
        assert_eq!(
            summary.geophysically_supported_edge_ids,
            vec!["b-to-suspected".to_string()]
        );
        assert_eq!(
            summary.candidate_edge_ids,
            vec!["alternative-shortcut".to_string()]
        );
    }

    #[test]
    fn route_requires_both_observed_connection_and_demonstrated_traversal() {
        let network =
            intake_network_json(NETWORK_FIXTURE).expect("valid synthetic network fixture");
        assert!(matches!(
            network.demonstrated_route("entrance", "chamber-a"),
            Ok(RouteKnowledge::DemonstratedRoute { .. })
        ));
        assert_eq!(
            network.demonstrated_route("entrance", "chamber-b"),
            Ok(RouteKnowledge::NoDemonstratedRoute)
        );
        assert_eq!(
            network.demonstrated_route("entrance", "suspected-chamber"),
            Ok(RouteKnowledge::NoDemonstratedRoute)
        );
    }

    #[test]
    fn directed_traversability_can_make_outbound_and_retreat_differ() {
        let network = CaveNetworkRecord {
            network_id: "directed-test".into(),
            nodes: vec!["a".into(), "b".into()],
            edges: vec![CaveNetworkEdge {
                edge_id: "one-way".into(),
                node_a: "a".into(),
                node_b: "b".into(),
                length_m: 3.0,
                connection_evidence: ConnectionEvidence::DirectlyObserved,
                traversability: TraversabilityEvidence::Demonstrated,
                direction: EdgeDirection::AToB,
                evidence_refs: vec!["one-way-traversal".into()],
            }],
            notes: "synthetic directed test".into(),
        };
        assert!(matches!(
            network.demonstrated_route("a", "b"),
            Ok(RouteKnowledge::DemonstratedRoute { .. })
        ));
        assert_eq!(
            network.demonstrated_route("b", "a"),
            Ok(RouteKnowledge::NoDemonstratedRoute)
        );
    }

    #[test]
    fn candidate_edge_cannot_be_marked_traversal_demonstrated() {
        let mut network: CaveNetworkRecord =
            serde_json::from_str(NETWORK_FIXTURE).expect("fixture JSON");
        network.edges[2].traversability = TraversabilityEvidence::Demonstrated;
        assert!(network.validate().unwrap_err().contains(
            &NetworkIssue::DemonstratedTraversalWithoutObservedConnection(
                "b-to-suspected".into()
            )
        ));
    }

    #[test]
    fn contradicted_connection_cannot_be_traversable() {
        let mut network: CaveNetworkRecord =
            serde_json::from_str(NETWORK_FIXTURE).expect("fixture JSON");
        network.edges[0].connection_evidence = ConnectionEvidence::Contradicted;
        network.edges[0].traversability = TraversabilityEvidence::Demonstrated;
        assert!(network.validate().unwrap_err().contains(
            &NetworkIssue::DemonstratedTraversalWithoutObservedConnection(
                "entrance-to-a".into()
            )
        ));
    }

    #[test]
    fn candidate_edges_require_evidence_references() {
        let mut network: CaveNetworkRecord =
            serde_json::from_str(NETWORK_FIXTURE).expect("fixture JSON");
        network.edges[3].evidence_refs.clear();
        assert!(network
            .validate()
            .unwrap_err()
            .contains(&NetworkIssue::EdgeMissingEvidence(
                "alternative-shortcut".into()
            )));
    }

    #[test]
    fn missing_demonstrated_route_does_not_assert_disconnection() {
        let network =
            intake_network_json(NETWORK_FIXTURE).expect("valid synthetic network fixture");
        assert_eq!(
            network.demonstrated_route("entrance", "suspected-chamber"),
            Ok(RouteKnowledge::NoDemonstratedRoute)
        );
        // The suspected chamber is not isolated in the hypothesis set; it is
        // simply outside the directly observed component and route graph.
        assert_eq!(network.nodes.len(), 4);
    }

    #[test]
    fn unknown_nodes_and_duplicate_edge_ids_are_rejected() {
        let mut network: CaveNetworkRecord =
            serde_json::from_str(NETWORK_FIXTURE).expect("fixture JSON");
        network.edges[0].node_b = "missing-node".into();
        network.edges[1].edge_id = network.edges[0].edge_id.clone();
        let issues = network.validate().unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            NetworkIssue::UnknownEndpoint { .. }
        )));
        assert!(issues.contains(&NetworkIssue::DuplicateEdgeId("entrance-to-a".into())));
    }
}
