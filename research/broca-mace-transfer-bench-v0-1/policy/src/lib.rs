// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Action {
    A,
    B,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Regime {
    Baseline,
    Shifted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Objective {
    Quality,
    Cost,
    Speed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Scope {
    pub regime: Regime,
    pub objective: Objective,
}

/// SHA-256 digest of a canonical upstream provenance-root identity.
/// A digest identifies a root; it does not itself authenticate its issuer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SourceRootDigest([u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceRootDigestParseError {
    InvalidLength { actual: usize },
    InvalidHex { byte_index: usize },
}

impl SourceRootDigest {
    pub const UNKNOWN: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_known(self) -> bool {
        self.0 != [0; 32]
    }

    /// Parse exactly 64 hexadecimal characters. Uppercase input is accepted;
    /// `to_hex` always emits canonical lowercase.
    pub fn from_hex(input: &str) -> Result<Self, SourceRootDigestParseError> {
        let bytes = input.as_bytes();
        if bytes.len() != 64 {
            return Err(SourceRootDigestParseError::InvalidLength {
                actual: bytes.len(),
            });
        }

        let mut digest = [0_u8; 32];
        for (index, pair) in bytes.chunks_exact(2).enumerate() {
            let high = decode_hex(pair[0]).ok_or(
                SourceRootDigestParseError::InvalidHex {
                    byte_index: index * 2,
                },
            )?;
            let low = decode_hex(pair[1]).ok_or(
                SourceRootDigestParseError::InvalidHex {
                    byte_index: index * 2 + 1,
                },
            )?;
            digest[index] = (high << 4) | low;
        }
        Ok(Self(digest))
    }

    /// Render the digest as 64 canonical lowercase hexadecimal characters.
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(HEX[usize::from(byte >> 4)] as char);
            output.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        output
    }
}

fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvidenceDigest256([u8; 32]);

impl EvidenceDigest256 {
    pub const UNKNOWN: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub const fn is_known(self) -> bool {
        self.0 != [0; 32]
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Versioned semantic identity of a causal estimand. Each field digest is
/// produced from a canonical typed subdocument by the trusted verifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticEstimandV1 {
    pub schema_version: u16,
    pub population_spec_digest: EvidenceDigest256,
    pub treatment_version_digest: EvidenceDigest256,
    pub comparator_version_digest: EvidenceDigest256,
    pub outcome_definition_digest: EvidenceDigest256,
    pub outcome_unit_digest: EvidenceDigest256,
    pub follow_up_horizon_ms: u64,
    pub censoring_strategy_digest: EvidenceDigest256,
    pub intercurrent_event_strategy_digest: EvidenceDigest256,
    pub target_weighting_digest: EvidenceDigest256,
}

impl SemanticEstimandV1 {
    pub const UNKNOWN: Self = Self {
        schema_version: 0,
        population_spec_digest: EvidenceDigest256::UNKNOWN,
        treatment_version_digest: EvidenceDigest256::UNKNOWN,
        comparator_version_digest: EvidenceDigest256::UNKNOWN,
        outcome_definition_digest: EvidenceDigest256::UNKNOWN,
        outcome_unit_digest: EvidenceDigest256::UNKNOWN,
        follow_up_horizon_ms: 0,
        censoring_strategy_digest: EvidenceDigest256::UNKNOWN,
        intercurrent_event_strategy_digest: EvidenceDigest256::UNKNOWN,
        target_weighting_digest: EvidenceDigest256::UNKNOWN,
    };

    pub const fn new(
        schema_version: u16,
        population_spec_digest: EvidenceDigest256,
        treatment_version_digest: EvidenceDigest256,
        comparator_version_digest: EvidenceDigest256,
        outcome_definition_digest: EvidenceDigest256,
        outcome_unit_digest: EvidenceDigest256,
        follow_up_horizon_ms: u64,
        censoring_strategy_digest: EvidenceDigest256,
        intercurrent_event_strategy_digest: EvidenceDigest256,
        target_weighting_digest: EvidenceDigest256,
    ) -> Self {
        Self {
            schema_version,
            population_spec_digest,
            treatment_version_digest,
            comparator_version_digest,
            outcome_definition_digest,
            outcome_unit_digest,
            follow_up_horizon_ms,
            censoring_strategy_digest,
            intercurrent_event_strategy_digest,
            target_weighting_digest,
        }
    }

    pub fn is_complete(self) -> bool {
        self.schema_version == 1
            && self.population_spec_digest.is_known()
            && self.treatment_version_digest.is_known()
            && self.comparator_version_digest.is_known()
            && self.outcome_definition_digest.is_known()
            && self.outcome_unit_digest.is_known()
            && self.follow_up_horizon_ms > 0
            && self.censoring_strategy_digest.is_known()
            && self.intercurrent_event_strategy_digest.is_known()
            && self.target_weighting_digest.is_known()
    }

    pub fn canonical_fields(self) -> Vec<String> {
        vec![
            self.schema_version.to_string(),
            self.population_spec_digest.to_hex(),
            self.treatment_version_digest.to_hex(),
            self.comparator_version_digest.to_hex(),
            self.outcome_definition_digest.to_hex(),
            self.outcome_unit_digest.to_hex(),
            self.follow_up_horizon_ms.to_string(),
            self.censoring_strategy_digest.to_hex(),
            self.intercurrent_event_strategy_digest.to_hex(),
            self.target_weighting_digest.to_hex(),
        ]
    }

    pub fn canonical_digest(self) -> EvidenceDigest256 {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(b"mycelix:mace:semantic-estimand:v1\\0");
        for field in self.canonical_fields() {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field.as_bytes());
        }
        let digest = hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        EvidenceDigest256::from_bytes(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdentificationMethodV1 {
    RandomizedControlledTrial,
    QuasiExperimental,
    Observational,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CausalIdentificationV1 {
    pub method: IdentificationMethodV1,
    pub design_digest: EvidenceDigest256,
    pub estimand_digest: EvidenceDigest256,
    pub semantic_estimand: SemanticEstimandV1,
    pub propensity_min_bps: u16,
    pub propensity_max_bps: u16,
    pub max_standardized_mean_difference_micros: u32,
    pub interference_controlled: bool,
}

impl CausalIdentificationV1 {
    pub const UNKNOWN: Self = Self {
        method: IdentificationMethodV1::Unknown,
        design_digest: EvidenceDigest256::UNKNOWN,
        estimand_digest: EvidenceDigest256::UNKNOWN,
        semantic_estimand: SemanticEstimandV1::UNKNOWN,
        propensity_min_bps: 0,
        propensity_max_bps: 10_000,
        max_standardized_mean_difference_micros: u32::MAX,
        interference_controlled: false,
    };

    pub const fn new(
        method: IdentificationMethodV1,
        design_digest: EvidenceDigest256,
        estimand_digest: EvidenceDigest256,
        propensity_min_bps: u16,
        propensity_max_bps: u16,
        max_standardized_mean_difference_micros: u32,
        interference_controlled: bool,
    ) -> Self {
        Self {
            method,
            design_digest,
            estimand_digest,
            semantic_estimand: SemanticEstimandV1::UNKNOWN,
            propensity_min_bps,
            propensity_max_bps,
            max_standardized_mean_difference_micros,
            interference_controlled,
        }
    }

    pub const fn with_semantic_estimand(
        mut self,
        semantic_estimand: SemanticEstimandV1,
    ) -> Self {
        self.semantic_estimand = semantic_estimand;
        self
    }

    /// Only randomized designs with a complete, digest-matched semantic
    /// estimand qualify in this tranche.
    pub fn qualifies_for_transfer(self) -> bool {
        self.method == IdentificationMethodV1::RandomizedControlledTrial
            && self.design_digest.is_known()
            && self.estimand_digest.is_known()
            && self.semantic_estimand.is_complete()
            && self.semantic_estimand.canonical_digest() == self.estimand_digest
            && self.propensity_min_bps >= 500
            && self.propensity_min_bps <= self.propensity_max_bps
            && self.propensity_max_bps <= 9_500
            && self.max_standardized_mean_difference_micros <= 100_000
            && self.interference_controlled
    }
}

// Candidate-visible experience. No future labels or evaluator-only utility fields.
#[derive(Clone, Copy, Debug)]
pub struct Experience {
    pub event_id: u32,
    pub source_family: u16,
    /// Legacy/display identifier; never used to establish source independence.
    pub source_root_id: u32,
    /// Verified 256-bit identity of the canonical upstream root.
    pub source_root_digest: SourceRootDigest,
    pub scope: Scope,
    pub action: Action,
    pub observed_utility: f64,
    /// Claimed absolute measurement-error bound in normalized utility units.
    pub measurement_error_bound: f64,
    pub causal_identification: CausalIdentificationV1,
    pub intervention: bool,
    pub prediction_pre_registered: bool,
    pub outcome_reconciled: bool,
    pub source_trusted: bool,
    pub source_retracted: bool,
    pub confounders_controlled: bool,
    pub delayed_horizon_complete: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub scope: Scope,
}

/// Canonical identity of the MACE causal-transfer qualification profile.
/// Changing any predicate or threshold must increment the version/domain.
pub fn transfer_qualification_profile_digest() -> EvidenceDigest256 {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:transfer-qualification-profile:v1\0");
    hasher.update([1_u8]); // IdentificationMethodV1::RandomizedControlledTrial
    hasher.update(500_u16.to_be_bytes()); // minimum propensity lower bound
    hasher.update(9_500_u16.to_be_bytes()); // maximum propensity upper bound
    hasher.update(100_000_u32.to_be_bytes()); // maximum standardized mean difference
    hasher.update([1_u8]); // intervention required
    hasher.update([1_u8]); // prediction pre-registered required
    hasher.update([1_u8]); // outcome reconciled required
    hasher.update([1_u8]); // source trusted required
    hasher.update([1_u8]); // source retracted forbidden
    hasher.update([1_u8]); // interference controlled required
    hasher.update([1_u8]); // confounders controlled required
    hasher.update([1_u8]); // delayed horizon complete required
    hasher.update([0_u8, 1_u8]); // observed utility and measurement bound in [0, 1]
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&digest);
    EvidenceDigest256::from_bytes(bytes)
}

/// Candidate-visible contract declaring what future decision is being made.
/// It contains task requirements only: no realized utility, optimal action,
/// hidden mechanism parameters, or evaluator outcome labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionNeedV1 {
    pub schema_version: u16,
    pub scope: Scope,
    pub action_alternatives: [Action; 2],
    pub semantic_estimand: SemanticEstimandV1,
    pub identification_method: IdentificationMethodV1,
    pub qualification_profile_digest: EvidenceDigest256,
}

impl DecisionNeedV1 {
    /// Build a task contract for the current MACE binary-action comparison.
    /// None means the task did not supply a complete, canonical estimand.
    pub fn new(scope: Scope, semantic_estimand: SemanticEstimandV1) -> Option<Self> {
        if !semantic_estimand.is_complete()
            || semantic_estimand.canonical_digest() == EvidenceDigest256::UNKNOWN
        {
            return None;
        }
        Some(Self {
            schema_version: 1,
            scope,
            action_alternatives: [Action::A, Action::B],
            semantic_estimand,
            identification_method: IdentificationMethodV1::RandomizedControlledTrial,
            qualification_profile_digest: transfer_qualification_profile_digest(),
        })
    }

    pub fn is_valid(self) -> bool {
        self.schema_version == 1
            && self.action_alternatives == [Action::A, Action::B]
            && self.semantic_estimand.is_complete()
            && self.qualification_profile_digest == transfer_qualification_profile_digest()
            && self.identification_method == IdentificationMethodV1::RandomizedControlledTrial
    }

    /// Domain-separated digest of the candidate-visible task requirements.
    /// It does not authenticate who supplied the task contract.
    pub fn canonical_digest(self) -> EvidenceDigest256 {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:decision-need:v1\0");
        hasher.update(self.schema_version.to_be_bytes());
        hasher.update([
            match self.scope.regime {
                Regime::Baseline => 1,
                Regime::Shifted => 2,
            },
            match self.scope.objective {
                Objective::Quality => 1,
                Objective::Cost => 2,
                Objective::Speed => 3,
            },
            1_u8, // Action::A
            2_u8, // Action::B
            match self.identification_method {
                IdentificationMethodV1::RandomizedControlledTrial => 1,
                IdentificationMethodV1::QuasiExperimental => 2,
                IdentificationMethodV1::Observational => 3,
                IdentificationMethodV1::Unknown => 4,
            },
        ]);
        hasher.update(self.semantic_estimand.canonical_digest().as_bytes());
        hasher.update(self.qualification_profile_digest.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        EvidenceDigest256::from_bytes(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    NoMemory,
    FrequencyOnly,
    ScopeAware,
    CausalQualified,
    InvariantTransport,
    ActiveProbe,
}

impl Policy {
    pub const ALL: [Policy; 6] = [
        Policy::NoMemory,
        Policy::FrequencyOnly,
        Policy::ScopeAware,
        Policy::CausalQualified,
        Policy::InvariantTransport,
        Policy::ActiveProbe,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::NoMemory => "no_memory",
            Self::FrequencyOnly => "frequency_only",
            Self::ScopeAware => "scope_aware",
            Self::CausalQualified => "causal_qualified",
            Self::InvariantTransport => "invariant_transport",
            Self::ActiveProbe => "active_probe",
        }
    }
}

pub const NORMALIZED_UTILITY_SCALE: u32 = 1_000_000;
pub const DEFAULT_PROBE_BUDGET_MICROS: u32 = 120_000;

/// Explicit environment-interaction request. This is a proposal, not authority
/// to perform an experiment; the environment must validate safety and budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeRequest {
    pub scope: Scope,
    pub actions: [Action; 2],
    pub replicates_per_action: u8,
    pub max_cost_normalized_micros: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Decision {
    pub action: Option<Action>,
    pub causal_claim: Option<Action>,
    pub evidence_ids: Vec<u32>,
    pub probe: Option<ProbeRequest>,
}

impl Decision {
    pub fn abstain() -> Self {
        Self::default()
    }

    pub fn request_probe(scope: Scope) -> Self {
        Self {
            action: None,
            causal_claim: None,
            evidence_ids: Vec::new(),
            probe: Some(ProbeRequest {
                scope,
                actions: [Action::A, Action::B],
                replicates_per_action: 2,
                max_cost_normalized_micros: DEFAULT_PROBE_BUDGET_MICROS,
            }),
        }
    }

    fn choose(action: Action, evidence_ids: Vec<u32>) -> Self {
        Self {
            action: Some(action),
            causal_claim: None,
            evidence_ids,
            probe: None,
        }
    }
}

// This API exposes no evaluator/oracle type. Policies see only Experience and Query.
pub fn decide(policy: Policy, memory: &[Experience], query: Query) -> Decision {
    match policy {
        Policy::NoMemory => Decision::choose(Action::B, Vec::new()),
        Policy::FrequencyOnly => frequency_policy(memory),
        Policy::ScopeAware => scope_aware_policy(memory, query),
        Policy::CausalQualified => causal_qualified_policy(memory, query),
        Policy::InvariantTransport => invariant_transport_policy(memory, query),
        Policy::ActiveProbe => active_probe_policy(memory, query),
    }
}

fn frequency_policy(memory: &[Experience]) -> Decision {
    let mut totals: BTreeMap<Action, (f64, usize, Vec<u32>)> = BTreeMap::new();

    for item in memory {
        let entry = totals.entry(item.action).or_insert((0.0, 0, Vec::new()));
        entry.0 += item.observed_utility;
        entry.1 += 1;
        entry.2.push(item.event_id);
    }

    let mean = |action: Action| {
        totals
            .get(&action)
            .map(|(sum, count, _)| *sum / *count as f64)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let action = if mean(Action::A) > mean(Action::B) {
        Action::A
    } else {
        Action::B
    };
    let evidence_ids = totals
        .get(&action)
        .map(|(_, _, ids)| ids.clone())
        .unwrap_or_default();

    Decision::choose(action, evidence_ids)
}

fn scope_aware_policy(memory: &[Experience], query: Query) -> Decision {
    let scoped: Vec<Experience> = memory
        .iter()
        .copied()
        .filter(|item| item.scope == query.scope)
        .collect();
    frequency_policy(&scoped)
}

pub fn is_qualified(item: &Experience, query: Query) -> bool {
    item.scope == query.scope
        && item.source_family != 0
        && item.source_root_digest.is_known()
        && item.observed_utility.is_finite()
        && (0.0..=1.0).contains(&item.observed_utility)
        && item.measurement_error_bound.is_finite()
        && (0.0..=1.0).contains(&item.measurement_error_bound)
        && item.causal_identification.qualifies_for_transfer()
        && item.intervention
        && item.prediction_pre_registered
        && item.outcome_reconciled
        && item.source_trusted
        && !item.source_retracted
        && item.confounders_controlled
        && item.delayed_horizon_complete
}

#[derive(Clone, Debug, Default)]
struct QualifiedGroup {
    source_families: BTreeSet<u16>,
    source_root_digests: BTreeSet<SourceRootDigest>,
    source_events: BTreeSet<u32>,
    estimand_digest: Option<EvidenceDigest256>,
    semantic_estimand: Option<SemanticEstimandV1>,
    mixed_estimands: bool,
    sum: f64,
    error_bound_sum: f64,
    count: usize,
    evidence_ids: Vec<u32>,
}

impl QualifiedGroup {
    fn add(&mut self, item: &Experience) {
        if !self.source_events.insert(item.event_id) {
            return;
        }
        self.source_families.insert(item.source_family);
        self.source_root_digests.insert(item.source_root_digest);
        match (self.estimand_digest, self.semantic_estimand) {
            (None, None) => {
                self.estimand_digest = Some(item.causal_identification.estimand_digest);
                self.semantic_estimand = Some(item.causal_identification.semantic_estimand);
            }
            (Some(existing_digest), Some(existing_semantic))
                if existing_digest != item.causal_identification.estimand_digest
                    || existing_semantic != item.causal_identification.semantic_estimand =>
            {
                self.mixed_estimands = true;
            }
            _ => self.mixed_estimands = true,
        }
        self.sum += item.observed_utility;
        self.error_bound_sum += item.measurement_error_bound;
        self.count += 1;
        self.evidence_ids.push(item.event_id);
    }

    fn independent_support(&self) -> usize {
        self.source_root_digests.len()
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn mean_error_bound(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.error_bound_sum / self.count as f64
        }
    }

    fn has_single_estimand(&self) -> bool {
        self.estimand_digest.is_some() && !self.mixed_estimands
    }

    fn is_qualified(&self) -> bool {
        self.has_single_estimand()
            && self.independent_support() >= 2
            && self.source_families.len() >= 2
            && self.source_events.len() >= 2
            && self.count >= 2
    }
}

fn causal_qualified_policy(memory: &[Experience], query: Query) -> Decision {
    let mut event_frequency: BTreeMap<u32, usize> = BTreeMap::new();
    for item in memory {
        *event_frequency.entry(item.event_id).or_default() += 1;
    }

    let mut groups: BTreeMap<Action, QualifiedGroup> = BTreeMap::new();
    for item in memory.iter().filter(|item| {
        event_frequency.get(&item.event_id) == Some(&1) && is_qualified(item, query)
    }) {
        groups.entry(item.action).or_default().add(item);
    }

    let a = groups.get(&Action::A).cloned().unwrap_or_default();
    let b = groups.get(&Action::B).cloned().unwrap_or_default();
    let a_ok = a.is_qualified();
    let b_ok = b.is_qualified();
    if a_ok
        && b_ok
        && (a.estimand_digest != b.estimand_digest
            || a.semantic_estimand != b.semantic_estimand)
    {
        return Decision::abstain();
    }

    match (a_ok, b_ok) {
        // A comparison cannot be made from one-sided evidence. Unknown is
        // not equivalent to inferior; the caller must abstain or probe.
        (false, false) | (true, false) | (false, true) => Decision::abstain(),
        (true, true) => {
            let action = if a.mean() > b.mean() { Action::A } else { Action::B };
            let difference = (a.mean() - b.mean()).abs();
            let mut evidence_ids = a.evidence_ids;
            evidence_ids.extend(b.evidence_ids);

            Decision {
                action: Some(action),
                causal_claim: (difference >= 0.05).then_some(action),
                evidence_ids,
                probe: None,
            }
        }
    }
}

/// Candidate evidence eligibility for a cross-regime transport test.
/// The optimization objective remains exact; regime compatibility is checked
/// separately by the invariant-transport policy.
pub fn is_transfer_qualified(item: &Experience, query: Query) -> bool {
    item.scope.objective == query.scope.objective
        && item.source_family != 0
        && item.source_root_digest.is_known()
        && item.observed_utility.is_finite()
        && (0.0..=1.0).contains(&item.observed_utility)
        && item.measurement_error_bound.is_finite()
        && (0.0..=1.0).contains(&item.measurement_error_bound)
        && item.causal_identification.qualifies_for_transfer()
        && item.intervention
        && item.prediction_pre_registered
        && item.outcome_reconciled
        && item.source_trusted
        && !item.source_retracted
        && item.confounders_controlled
        && item.delayed_horizon_complete
}

const INVARIANCE_EQUIVALENCE_MARGIN: f64 = 0.12;

fn invariant_transport_policy(memory: &[Experience], query: Query) -> Decision {
    let mut event_frequency: BTreeMap<u32, usize> = BTreeMap::new();
    for item in memory {
        *event_frequency.entry(item.event_id).or_default() += 1;
    }

    let mut groups: BTreeMap<(Regime, Action), QualifiedGroup> = BTreeMap::new();
    for item in memory.iter().filter(|item| {
        event_frequency.get(&item.event_id) == Some(&1)
            && is_transfer_qualified(item, query)
    }) {
        groups
            .entry((item.scope.regime, item.action))
            .or_default()
            .add(item);
    }

    // A target-regime witness for each action is mandatory. A single target
    // observation is sufficient for a few-shot transfer test, but each source
    // regime must carry independently qualified evidence for both actions.
    let Some(target_a) = groups.get(&(query.scope.regime, Action::A)) else {
        return Decision::abstain();
    };
    let Some(target_b) = groups.get(&(query.scope.regime, Action::B)) else {
        return Decision::abstain();
    };
    if target_a.count == 0 || target_b.count == 0 {
        return Decision::abstain();
    }
    let Some(target_estimand) = target_a.estimand_digest.filter(|_| {
        !target_a.mixed_estimands
    }) else {
        return Decision::abstain();
    };
    if target_b.estimand_digest != Some(target_estimand)
        || target_b.semantic_estimand != target_a.semantic_estimand
        || target_b.mixed_estimands
    {
        return Decision::abstain();
    }

    let target_effect = target_a.mean() - target_b.mean();
    let target_effect_error =
        target_a.mean_error_bound() + target_b.mean_error_bound();
    let target_families: BTreeSet<u16> = target_a
        .source_families
        .union(&target_b.source_families)
        .copied()
        .collect();
    let target_roots: BTreeSet<SourceRootDigest> = target_a
        .source_root_digests
        .union(&target_b.source_root_digests)
        .copied()
        .collect();

    let source_regimes: BTreeSet<Regime> = groups
        .keys()
        .filter_map(|(regime, _)| {
            (*regime != query.scope.regime).then_some(*regime)
        })
        .collect();

    let mut pooled_effect = target_effect;
    let mut pooled_effect_error = target_effect_error;
    let mut effect_regime_count = 1usize;
    let mut evidence_ids = target_a.evidence_ids.clone();
    evidence_ids.extend(target_b.evidence_ids.iter().copied());

    let mut used_source_families = BTreeSet::new();
    let mut used_source_roots = BTreeSet::new();
    let mut validated_source_regimes = 0usize;

    for regime in source_regimes {
        let Some(source_a) = groups.get(&(regime, Action::A)) else {
            continue;
        };
        let Some(source_b) = groups.get(&(regime, Action::B)) else {
            continue;
        };
        if !source_a.is_qualified() || !source_b.is_qualified() {
            continue;
        }
        let Some(source_estimand) = source_a.estimand_digest else {
            return Decision::abstain();
        };
        if source_a.mixed_estimands
            || source_b.estimand_digest != Some(source_estimand)
            || source_b.semantic_estimand != source_a.semantic_estimand
            || source_a.semantic_estimand != target_a.semantic_estimand
            || source_b.mixed_estimands
            || source_estimand != target_estimand
        {
            return Decision::abstain();
        }

        let source_families: BTreeSet<u16> = source_a
            .source_families
            .union(&source_b.source_families)
            .copied()
            .collect();
        let source_roots: BTreeSet<SourceRootDigest> = source_a
            .source_root_digests
            .union(&source_b.source_root_digests)
            .copied()
            .collect();

        // No family or root may be laundered across source/target regimes or
        // reused across nominally independent source regimes.
        if !source_families.is_disjoint(&target_families)
            || !source_roots.is_disjoint(&target_roots)
            || !source_families.is_disjoint(&used_source_families)
            || !source_roots.is_disjoint(&used_source_roots)
        {
            return Decision::abstain();
        }

        // Compare the causal contrast A-B, not each action's absolute outcome
        // mean. This admits common-mode outcome shifts while rejecting effect
        // modification. Error bounds are conservatively summed across all
        // four action/regime means.
        let source_effect = source_a.mean() - source_b.mean();
        let source_effect_error =
            source_a.mean_error_bound() + source_b.mean_error_bound();
        let worst_case_effect_difference = (source_effect - target_effect).abs()
            + source_effect_error
            + target_effect_error;
        if worst_case_effect_difference > INVARIANCE_EQUIVALENCE_MARGIN {
            return Decision::abstain();
        }

        pooled_effect += source_effect;
        pooled_effect_error += source_effect_error;
        effect_regime_count += 1;
        evidence_ids.extend(source_a.evidence_ids.iter().copied());
        evidence_ids.extend(source_b.evidence_ids.iter().copied());
        used_source_families.extend(source_families);
        used_source_roots.extend(source_roots);
        validated_source_regimes += 1;
    }

    if validated_source_regimes == 0 {
        return Decision::abstain();
    }

    // Equal-weight regime contrasts avoid raw-outcome pooling bias from
    // unequal A/B sample counts. Abstain if the aggregate effect interval
    // still crosses zero.
    let regime_count = effect_regime_count as f64;
    let transported_effect = pooled_effect / regime_count;
    let transported_error = pooled_effect_error / regime_count;
    if transported_effect.abs() <= transported_error {
        return Decision::abstain();
    }

    let action = if transported_effect > 0.0 {
        Action::A
    } else {
        Action::B
    };

    Decision {
        action: Some(action),
        causal_claim: (transported_effect.abs() >= 0.05).then_some(action),
        evidence_ids,
        probe: None,
    }
}

fn active_probe_policy(memory: &[Experience], query: Query) -> Decision {
    let exact = causal_qualified_policy(memory, query);
    if exact.action.is_some() {
        return exact;
    }

    let transported = invariant_transport_policy(memory, query);
    if transported.action.is_some() {
        return transported;
    }

    Decision::request_probe(query.scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_root_digest(seed: u32) -> SourceRootDigest {
        let mut bytes = [0_u8; 32];
        bytes[..4].copy_from_slice(&seed.to_be_bytes());
        SourceRootDigest::from_bytes(bytes)
    }

    fn fixture_semantic_estimand() -> SemanticEstimandV1 {
        SemanticEstimandV1::new(
            1,
            EvidenceDigest256::from_bytes([1; 32]),
            EvidenceDigest256::from_bytes([2; 32]),
            EvidenceDigest256::from_bytes([3; 32]),
            EvidenceDigest256::from_bytes([4; 32]),
            EvidenceDigest256::from_bytes([5; 32]),
            30_000,
            EvidenceDigest256::from_bytes([6; 32]),
            EvidenceDigest256::from_bytes([7; 32]),
            EvidenceDigest256::from_bytes([8; 32]),
        )
    }

    fn fixture_causal_identification(seed: u32) -> CausalIdentificationV1 {
        let mut design = [0_u8; 32];
            design[..4].copy_from_slice(&seed.to_be_bytes());
        CausalIdentificationV1::new(
            IdentificationMethodV1::RandomizedControlledTrial,
            EvidenceDigest256::from_bytes(design),
            fixture_semantic_estimand().canonical_digest(),
            5_000,
            5_000,
            50_000,
            true,
        )
        .with_semantic_estimand(fixture_semantic_estimand())
    }

    fn sample(
        id: u32,
        family: u16,
        action: Action,
        utility: f64,
        scope: Scope,
    ) -> Experience {
        Experience {
            event_id: id,
            source_family: family,
            source_root_id: u32::from(family),
            source_root_digest: fixture_root_digest(u32::from(family)),
            scope,
            action,
            observed_utility: utility,
            measurement_error_bound: 0.02,
            causal_identification: fixture_causal_identification(id),
            intervention: true,
            prediction_pre_registered: true,
            outcome_reconciled: true,
            source_trusted: true,
            source_retracted: false,
            confounders_controlled: true,
            delayed_horizon_complete: true,
        }
    }

    #[test]
    fn valid_independent_interventions_can_change_a_later_action() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 11, Action::A, 0.9, scope),
            sample(3, 20, Action::B, 0.4, scope),
            sample(4, 21, Action::B, 0.4, scope),
        ];
        let decision = decide(
            Policy::CausalQualified,
            &memory,
            Query { scope },
        );
        assert_eq!(decision.action, Some(Action::A));
    }

    #[test]
    fn correlated_repetitions_do_not_satisfy_independent_support() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 10, Action::A, 0.9, scope),
            sample(3, 10, Action::A, 0.9, scope),
        ];
        let decision = decide(
            Policy::CausalQualified,
            &memory,
            Query { scope },
        );
        assert_eq!(decision.action, None);
    }

    #[test]
    fn same_event_with_two_claimed_source_families_is_not_independent_support() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(7, 10, Action::A, 0.99, scope),
            sample(7, 11, Action::A, 0.99, scope),
            sample(8, 20, Action::B, 0.65, scope),
            sample(9, 21, Action::B, 0.68, scope),
        ];
        let frequency = decide(Policy::FrequencyOnly, &memory, Query { scope });
        let qualified = decide(Policy::CausalQualified, &memory, Query { scope });

        assert_eq!(frequency.action, Some(Action::A));
        assert_eq!(qualified.action, None);
        assert!(qualified.evidence_ids.is_empty());
    }

    fn transport_case(
        event_id: u32,
        family: u16,
        regime: Regime,
        action: Action,
        utility: f64,
    ) -> Experience {
        sample(
            event_id,
            family,
            action,
            utility,
            Scope {
                regime,
                objective: Objective::Quality,
            },
        )
    }

    #[test]
    fn invariant_transport_accepts_common_mode_baseline_shift() {
        let memory = vec![
            transport_case(30, 40, Regime::Baseline, Action::A, 0.80),
            transport_case(31, 41, Regime::Baseline, Action::A, 0.80),
            transport_case(32, 50, Regime::Baseline, Action::B, 0.40),
            transport_case(33, 51, Regime::Baseline, Action::B, 0.40),
            transport_case(34, 60, Regime::Shifted, Action::A, 0.95),
            transport_case(35, 61, Regime::Shifted, Action::B, 0.55),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let exact = decide(Policy::CausalQualified, &memory, query);
        let transported = decide(Policy::InvariantTransport, &memory, query);

        assert_eq!(exact.action, None);
        assert_eq!(transported.action, Some(Action::A));
        assert!(transported.causal_claim.is_some());
    }

    #[test]
    fn invariant_transport_abstains_without_both_target_action_witnesses() {
        let memory = vec![
            transport_case(40, 40, Regime::Baseline, Action::A, 0.92),
            transport_case(41, 41, Regime::Baseline, Action::A, 0.90),
            transport_case(42, 50, Regime::Baseline, Action::B, 0.32),
            transport_case(43, 51, Regime::Baseline, Action::B, 0.34),
            transport_case(44, 60, Regime::Shifted, Action::A, 0.91),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let decision = decide(Policy::InvariantTransport, &memory, query);
        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn invariant_effects_can_transport_into_a_few_shot_target_regime() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(5, 30, Regime::Shifted, Action::A, 0.91),
            transport_case(6, 31, Regime::Shifted, Action::B, 0.33),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let exact = decide(Policy::CausalQualified, &memory, query);
        let transported = decide(Policy::InvariantTransport, &memory, query);

        assert_eq!(exact.action, None);
        assert_eq!(transported.action, Some(Action::A));
        assert_eq!(transported.evidence_ids.len(), 6);
    }

    #[test]
    fn invariant_transport_abstains_when_the_effect_changes_across_regimes() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.95),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.92),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.35),
            transport_case(5, 30, Regime::Shifted, Action::A, 0.25),
            transport_case(6, 31, Regime::Shifted, Action::B, 0.86),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let transported = decide(Policy::InvariantTransport, &memory, query);
        assert_eq!(transported.action, None);
        assert!(transported.evidence_ids.is_empty());
    }

    #[test]
    fn invariant_transport_refuses_cross_regime_source_family_overlap() {
        let memory = vec![
            transport_case(1, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(2, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(3, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(4, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(5, 10, Regime::Shifted, Action::A, 0.91),
            transport_case(6, 20, Regime::Shifted, Action::B, 0.33),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        assert_eq!(
            decide(Policy::InvariantTransport, &memory, query).action,
            None
        );
    }

    #[test]
    fn active_probe_is_requested_only_after_exact_and_transport_paths_fail() {
        let memory = vec![
            sample(
                1,
                10,
                Action::A,
                0.9,
                Scope {
                    regime: Regime::Baseline,
                    objective: Objective::Quality,
                },
            ),
        ];
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        let decision = decide(Policy::ActiveProbe, &memory, query);
        assert_eq!(decision.action, None);
        let request = decision.probe.expect("probe request is explicit");
        assert_eq!(request.scope, query.scope);
        assert_eq!(request.actions, [Action::A, Action::B]);
        assert_eq!(request.replicates_per_action, 2);
        assert_eq!(
            request.max_cost_normalized_micros,
            DEFAULT_PROBE_BUDGET_MICROS
        );
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn active_probe_uses_exact_scope_memory_without_probe() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(1, 10, Action::A, 0.9, scope),
            sample(2, 11, Action::A, 0.9, scope),
            sample(3, 20, Action::B, 0.3, scope),
            sample(4, 21, Action::B, 0.3, scope),
        ];
        let decision = decide(Policy::ActiveProbe, &memory, Query { scope });
        assert_eq!(decision.action, Some(Action::A));
        assert!(decision.probe.is_none());
    }

    #[test]
    fn different_source_families_from_one_common_root_are_not_independent() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut a1 = sample(20, 10, Action::A, 0.99, scope);
        let mut a2 = sample(21, 11, Action::A, 0.99, scope);
        a1.source_root_id = 700;
        a2.source_root_id = 700;
        a1.source_root_digest = fixture_root_digest(700);
        a2.source_root_digest = fixture_root_digest(700);
        let b1 = sample(22, 20, Action::B, 0.65, scope);
        let b2 = sample(23, 21, Action::B, 0.68, scope);
        let memory = vec![a1, a2, b1, b2];

        assert_eq!(
            decide(Policy::FrequencyOnly, &memory, Query { scope }).action,
            Some(Action::A)
        );
        let qualified = decide(Policy::CausalQualified, &memory, Query { scope });
        assert_eq!(qualified.action, None);
        assert!(qualified.evidence_ids.is_empty());
    }

    #[test]
    fn one_sided_qualified_evidence_does_not_authorize_a_comparison() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let memory = vec![
            sample(40, 10, Action::A, 0.99, scope),
            sample(41, 11, Action::A, 0.98, scope),
            sample(42, 20, Action::B, 0.55, scope),
        ];
        let decision = decide(Policy::CausalQualified, &memory, Query { scope });
        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn unknown_source_root_is_not_qualified() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut item = sample(30, 10, Action::A, 0.9, scope);
        item.source_root_id = 0;
        item.source_root_digest = SourceRootDigest::UNKNOWN;
        assert!(!is_qualified(&item, Query { scope }));
    }

    #[test]
    fn invariant_transport_rejects_shared_common_root_across_regimes() {
        let mut memory = vec![
            transport_case(10, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(11, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(12, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(13, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(14, 30, Regime::Shifted, Action::A, 0.91),
            transport_case(15, 31, Regime::Shifted, Action::B, 0.33),
        ];
        memory[4].source_root_id = 10;
        memory[5].source_root_id = 20;
        memory[4].source_root_digest = fixture_root_digest(10);
        memory[5].source_root_digest = fixture_root_digest(20);
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        assert_eq!(
            decide(Policy::InvariantTransport, &memory, query).action,
            None
        );
    }

    #[test]
    fn invariant_transport_requires_worst_case_error_interval_inside_margin() {
        let mut memory = vec![
            transport_case(20, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(21, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(22, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(23, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(24, 30, Regime::Shifted, Action::A, 0.91),
            transport_case(25, 31, Regime::Shifted, Action::B, 0.33),
        ];
        for item in &mut memory {
            item.measurement_error_bound = 0.07;
        }
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };
        assert_eq!(
            decide(Policy::InvariantTransport, &memory, query).action,
            None
        );
    }

    #[test]
    fn nonfinite_or_out_of_range_measurement_bounds_are_rejected() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut item = sample(50, 10, Action::A, 0.9, scope);
        item.measurement_error_bound = f64::NAN;
        assert!(!is_qualified(&item, Query { scope }));
        item.measurement_error_bound = 1.1;
        assert!(!is_qualified(&item, Query { scope }));
    }

    #[test]
    fn source_root_digest_hex_round_trip_is_canonical() {
        let digest = fixture_root_digest(0x12ab_34cd);
        let encoded = digest.to_hex();
        assert_eq!(encoded.len(), 64);
        assert_eq!(SourceRootDigest::from_hex(&encoded), Ok(digest));
        assert_eq!(
            SourceRootDigest::from_hex(&encoded.to_ascii_uppercase()),
            Ok(digest)
        );
        assert_eq!(digest.to_hex(), encoded);
    }

    #[test]
    fn source_root_digest_rejects_invalid_length_and_hex() {
        assert_eq!(
            SourceRootDigest::from_hex("abc"),
            Err(SourceRootDigestParseError::InvalidLength { actual: 3 })
        );
        let invalid = format!("{}g{}", "0".repeat(62), "0");
        assert_eq!(
            SourceRootDigest::from_hex(&invalid),
            Err(SourceRootDigestParseError::InvalidHex { byte_index: 62 })
        );
    }

    #[test]
    fn all_zero_hex_is_unknown_not_a_qualified_root() {
        let digest = SourceRootDigest::from_hex(&"0".repeat(64)).unwrap();
        assert_eq!(digest, SourceRootDigest::UNKNOWN);
        assert!(!digest.is_known());
    }
    #[test]
    fn unknown_source_family_is_not_qualified_or_transferable() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut item = sample(60, 10, Action::A, 0.9, scope);
        item.source_family = 0;
        let query = Query { scope };

        assert!(!is_qualified(&item, query));
        assert!(!is_transfer_qualified(&item, query));
    }

    #[test]
    fn observational_evidence_is_retained_but_not_causally_qualified() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        for method in [
            IdentificationMethodV1::Observational,
            IdentificationMethodV1::QuasiExperimental,
            IdentificationMethodV1::Unknown,
        ] {
            let mut item = sample(70, 10, Action::A, 0.99, scope);
            item.causal_identification.method = method;
            assert!(!is_qualified(&item, Query { scope }));
            assert!(!is_transfer_qualified(&item, Query { scope }));
        }
    }

    #[test]
    fn randomized_design_requires_overlap_balance_and_interference_controls() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut no_overlap = sample(71, 11, Action::A, 0.99, scope);
        no_overlap.causal_identification.propensity_min_bps = 100;
        assert!(!is_qualified(&no_overlap, Query { scope }));

        let mut imbalance = sample(72, 12, Action::A, 0.99, scope);
        imbalance
            .causal_identification
            .max_standardized_mean_difference_micros = 150_000;
        assert!(!is_qualified(&imbalance, Query { scope }));

        let mut interference = sample(73, 13, Action::A, 0.99, scope);
        interference.causal_identification.interference_controlled = false;
        assert!(!is_qualified(&interference, Query { scope }));

        let mut unknown_estimand = sample(74, 14, Action::A, 0.99, scope);
        unknown_estimand.causal_identification.estimand_digest =
            EvidenceDigest256::UNKNOWN;
        assert!(!is_qualified(&unknown_estimand, Query { scope }));
    }

    #[test]
    fn causal_comparison_rejects_different_estimands_between_actions() {
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let mut memory = vec![
            sample(80, 10, Action::A, 0.92, scope),
            sample(81, 11, Action::A, 0.90, scope),
            sample(82, 20, Action::B, 0.32, scope),
            sample(83, 21, Action::B, 0.34, scope),
        ];
        memory[2].causal_identification.estimand_digest =
            EvidenceDigest256::from_bytes([0xCC; 32]);
        memory[3].causal_identification.estimand_digest =
            EvidenceDigest256::from_bytes([0xCC; 32]);
        let decision = decide(Policy::CausalQualified, &memory, Query { scope });

        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn invariant_transport_rejects_estimand_change_between_regimes() {
        let mut memory = vec![
            transport_case(90, 10, Regime::Baseline, Action::A, 0.92),
            transport_case(91, 11, Regime::Baseline, Action::A, 0.90),
            transport_case(92, 20, Regime::Baseline, Action::B, 0.32),
            transport_case(93, 21, Regime::Baseline, Action::B, 0.34),
            transport_case(94, 30, Regime::Shifted, Action::A, 0.91),
            transport_case(95, 31, Regime::Shifted, Action::B, 0.33),
        ];
        let shifted_estimand = EvidenceDigest256::from_bytes([0xCC; 32]);
        memory[4].causal_identification.estimand_digest = shifted_estimand;
        memory[5].causal_identification.estimand_digest = shifted_estimand;
        let query = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: Objective::Quality,
            },
        };

        let decision = decide(Policy::InvariantTransport, &memory, query);
        assert_eq!(decision.action, None);
        assert!(decision.evidence_ids.is_empty());
    }

    #[test]
    fn scope_mismatch_is_not_eligible() {
        let source_scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Speed,
        };
        let query_scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Cost,
        };
        let item = sample(1, 10, Action::A, 0.9, source_scope);
        assert!(!is_qualified(&item, Query { scope: query_scope }));
    }
}
