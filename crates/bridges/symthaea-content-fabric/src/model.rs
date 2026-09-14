use serde::{Deserialize, Serialize};

pub const EXTERNAL_PLANNER_PROTOCOL_V1: u16 = 1;
pub const PENALTY_PPM_MAX: u32 = 1_000_000;
pub const ACTION_REF_LEN: usize = 39;
pub const AGENT_REF_LEN: usize = 39;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExternalPlannerRequestIdV1(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlannerInputIdV1(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StorageIntentIdV1(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlannerProfileIdV1(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlacementProposalIdV1(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ObjectIdV1(pub [u8; 32]);

/// Holochain action references are 39-byte opaque identifiers. `Vec<u8>` is
/// intentional here: it mirrors the JSON array shape without introducing an
/// HDK/Holochain dependency into Symthaea. Structural validation enforces the
/// exact length before planning.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActionRefV1(pub Vec<u8>);

/// Holochain agent references are likewise treated as opaque 39-byte values.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentRefV1(pub Vec<u8>);

impl ActionRefV1 {
    pub fn is_well_formed(&self) -> bool {
        self.0.len() == ACTION_REF_LEN
    }
}

impl AgentRefV1 {
    pub fn is_well_formed(&self) -> bool {
        self.0.len() == AGENT_REF_LEN
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DigestAlgorithmV1 {
    Blake3_256,
    Sha256,
}

impl DigestAlgorithmV1 {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Blake3_256 => "blake3-256",
            Self::Sha256 => "sha256",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentDigestV1 {
    pub algorithm: DigestAlgorithmV1,
    pub bytes: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementTargetV1 {
    pub object_id: ObjectIdV1,
    pub digest: ContentDigestV1,
    pub size_bytes: u64,
    pub client_side_encrypted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FailureDomainKindV1 {
    Operator,
    Machine,
    Site,
    NetworkAsn,
    Region,
    Jurisdiction,
    PowerDomain,
}

impl FailureDomainKindV1 {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Operator => 0,
            Self::Machine => 1,
            Self::Site => 2,
            Self::NetworkAsn => 3,
            Self::Region => 4,
            Self::Jurisdiction => 5,
            Self::PowerDomain => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SoftEvidenceStateV1 {
    Fresh,
    Missing,
    Stale,
    Future,
    InvalidWindow,
    IdentityMismatch,
    Conflicting,
}

impl SoftEvidenceStateV1 {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Fresh => 0,
            Self::Missing => 1,
            Self::Stale => 2,
            Self::Future => 3,
            Self::InvalidWindow => 4,
            Self::IdentityMismatch => 5,
            Self::Conflicting => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SoftMetricKindV1 {
    Cost,
    Latency,
    Energy,
    Locality,
}

impl SoftMetricKindV1 {
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::Cost => 0,
            Self::Latency => 1,
            Self::Energy => 2,
            Self::Locality => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPlannerProfileV1 {
    pub id: PlannerProfileIdV1,
    pub cost_ceiling_microunits_per_gib: u64,
    pub latency_ceiling_ms: u32,
    pub energy_ceiling_millijoules_per_gib: u64,
    pub locality_ceiling_km: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPlannerPreferencesV1 {
    pub target_latency_ms: Option<u32>,
    pub cost_weight: u16,
    pub latency_weight: u16,
    pub energy_weight: u16,
    pub locality_weight: u16,
}

impl ExternalPlannerPreferencesV1 {
    pub fn total_weight(self) -> u64 {
        u64::from(self.cost_weight)
            + u64::from(self.latency_weight)
            + u64::from(self.energy_weight)
            + u64::from(self.locality_weight)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExternalFailureDomainRequirementV1 {
    pub kind: FailureDomainKindV1,
    pub minimum_distinct: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExternalAcceptedFailureDomainV1 {
    pub kind: FailureDomainKindV1,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPlannerCandidateV1 {
    pub availability_action: ActionRefV1,
    pub advertisement_action: ActionRefV1,
    pub provider: AgentRefV1,
    pub baseline_rank: u64,
    pub baseline_weighted_penalty: u64,
    pub cost_penalty_ppm: u32,
    pub latency_penalty_ppm: u32,
    pub energy_penalty_ppm: u32,
    pub locality_penalty_ppm: u32,
    pub evidence_state: SoftEvidenceStateV1,
    pub missing_weighted_metrics: Vec<SoftMetricKindV1>,
    pub accepted_failure_domains: Vec<ExternalAcceptedFailureDomainV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPlannerRequestV1 {
    pub schema_version: u16,
    pub id: ExternalPlannerRequestIdV1,
    pub planner_input_id: PlannerInputIdV1,
    pub storage_intent_id: StorageIntentIdV1,
    pub target: PlacementTargetV1,
    pub evaluated_at_unix_ms: u64,
    pub profile: ExternalPlannerProfileV1,
    pub preferences: ExternalPlannerPreferencesV1,
    pub minimum_replicas: u16,
    pub failure_domain_requirements: Vec<ExternalFailureDomainRequirementV1>,
    pub baseline_proposal_id: PlacementProposalIdV1,
    pub baseline_selected_availability_actions: Vec<ActionRefV1>,
    pub candidates: Vec<ExternalPlannerCandidateV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalPlannerRecommendationV1 {
    pub schema_version: u16,
    pub request_id: ExternalPlannerRequestIdV1,
    pub planner_input_id: PlannerInputIdV1,
    pub engine_id: String,
    pub engine_version: String,
    pub ranking: Vec<ActionRefV1>,
    pub selected_availability_actions: Vec<ActionRefV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdcCandidateTraceV1 {
    pub availability_action: ActionRefV1,
    pub baseline_rank: u64,
    pub hdc_rank: u64,
    pub similarity_to_ideal: f32,
    pub baseline_weighted_penalty: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdcShadowTraceV1 {
    pub baseline_selected_availability_actions: Vec<ActionRefV1>,
    pub hdc_selected_availability_actions: Vec<ActionRefV1>,
    pub ranking_changed: bool,
    pub selection_changed: bool,
    pub candidates: Vec<HdcCandidateTraceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdcShadowPlanV1 {
    pub recommendation: ExternalPlannerRecommendationV1,
    pub trace: HdcShadowTraceV1,
}
