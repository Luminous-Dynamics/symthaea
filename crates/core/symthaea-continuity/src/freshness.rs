// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Freshness/invalidation context for authenticated continuity evidence.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::contract::{ContinuityContractId, ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::verifier::{AuthenticatedVerificationEvidenceV1, VerifierProfileId};
use crate::witness::{ObligationDispositionV1, QualifiedContinuityWitnessV1, TargetRealizationId, WitnessId};

const CLOCK_DOMAIN: &[u8] = b"symthaea.continuity.trusted-clock.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.continuity.freshness-policy.v1\0";
const EVIDENCE_DOMAIN: &[u8] = b"symthaea.continuity.fresh-evidence.v1\0";
const CONTEXT_DOMAIN: &[u8] = b"symthaea.continuity.qualified-witness-context.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedClockSnapshotId([u8; 32]);
impl TrustedClockSnapshotId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FreshnessPolicyId([u8; 32]);
impl FreshnessPolicyId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FreshVerificationEvidenceId([u8; 32]);
impl FreshVerificationEvidenceId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QualifiedWitnessContextId([u8; 32]);
impl QualifiedWitnessContextId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TrustedClockLineageV1 { source_id: String, epoch: u64 }
impl TrustedClockLineageV1 {
    pub fn source_id(&self) -> &str { &self.source_id }
    pub fn epoch(&self) -> u64 { self.epoch }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedClockSnapshotV1 {
    source_id: String,
    epoch: u64,
    earliest_unix_ms: u64,
    latest_unix_ms: u64,
    snapshot_id: TrustedClockSnapshotId,
}
impl TrustedClockSnapshotV1 {
    pub fn new(source_id: impl Into<String>, epoch: u64, earliest_unix_ms: u64, latest_unix_ms: u64) -> Result<Self, FreshnessError> {
        let source_id = checked_text("clock source_id", source_id.into())?;
        if epoch == 0 { return Err(FreshnessError::ZeroClockEpoch); }
        if earliest_unix_ms == 0 || latest_unix_ms == 0 { return Err(FreshnessError::ZeroClockTime); }
        if earliest_unix_ms > latest_unix_ms { return Err(FreshnessError::InvalidClockInterval); }
        let snapshot_id = TrustedClockSnapshotId(hash_clock(&source_id, epoch, earliest_unix_ms, latest_unix_ms));
        Ok(Self { source_id, epoch, earliest_unix_ms, latest_unix_ms, snapshot_id })
    }
    pub fn id(&self) -> TrustedClockSnapshotId { self.snapshot_id }
    pub fn lineage(&self) -> TrustedClockLineageV1 { TrustedClockLineageV1 { source_id: self.source_id.clone(), epoch: self.epoch } }
    pub fn earliest_unix_ms(&self) -> u64 { self.earliest_unix_ms }
    pub fn latest_unix_ms(&self) -> u64 { self.latest_unix_ms }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStabilityClassV1 { Stable, Refreshable, Volatile, Transactional }
impl EvidenceStabilityClassV1 {
    fn tag(self) -> u8 { match self { Self::Stable => 1, Self::Refreshable => 2, Self::Volatile => 3, Self::Transactional => 4 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EvidenceFreshnessRequirementV1 {
    Immutable,
    MaxAge { class: EvidenceStabilityClassV1, max_age_ms: u64 },
}
impl EvidenceFreshnessRequirementV1 {
    fn validate(self) -> Result<(), FreshnessError> {
        if matches!(self, Self::MaxAge { max_age_ms: 0, .. }) { return Err(FreshnessError::ZeroMaxAge); }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessPolicyEntryV1 { requirement_id: ContinuityRequirementId, freshness: EvidenceFreshnessRequirementV1 }
impl FreshnessPolicyEntryV1 {
    pub fn new(requirement_id: ContinuityRequirementId, freshness: EvidenceFreshnessRequirementV1) -> Result<Self, FreshnessError> {
        freshness.validate()?; Ok(Self { requirement_id, freshness })
    }
    pub fn requirement_id(&self) -> ContinuityRequirementId { self.requirement_id }
    pub fn freshness(&self) -> EvidenceFreshnessRequirementV1 { self.freshness }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessPolicyV1 { contract_id: ContinuityContractId, entries: Vec<FreshnessPolicyEntryV1>, policy_id: FreshnessPolicyId }
impl FreshnessPolicyV1 {
    pub fn new(contract: &ValidatedContinuityContractV1, mut entries: Vec<FreshnessPolicyEntryV1>) -> Result<Self, FreshnessError> {
        if entries.is_empty() { return Err(FreshnessError::EmptyFreshnessPolicy); }
        entries.sort_by_key(FreshnessPolicyEntryV1::requirement_id);
        if entries.windows(2).any(|p| p[0].requirement_id == p[1].requirement_id) { return Err(FreshnessError::DuplicateFreshnessRequirement); }
        let contract_ids: Vec<_> = contract.requirements().iter().map(|r| r.id()).collect();
        let policy_ids: Vec<_> = entries.iter().map(FreshnessPolicyEntryV1::requirement_id).collect();
        if contract_ids != policy_ids { return Err(FreshnessError::FreshnessContractMismatch); }
        let policy_id = FreshnessPolicyId(hash_policy(contract.id(), &entries));
        Ok(Self { contract_id: contract.id(), entries, policy_id })
    }
    pub fn id(&self) -> FreshnessPolicyId { self.policy_id }
    pub fn contract_id(&self) -> ContinuityContractId { self.contract_id }
    pub fn entries(&self) -> &[FreshnessPolicyEntryV1] { &self.entries }
    fn requirement(&self, id: ContinuityRequirementId) -> Option<EvidenceFreshnessRequirementV1> {
        self.entries.binary_search_by_key(&id, FreshnessPolicyEntryV1::requirement_id).ok().map(|i| self.entries[i].freshness())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierRootDependencyV1 { profile_id: VerifierProfileId, root_epoch: u64 }
impl VerifierRootDependencyV1 {
    pub fn profile_id(&self) -> VerifierProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
}

#[derive(Debug, Clone)]
pub(crate) struct FreshAuthenticatedVerificationEvidenceV1 {
    authenticated: AuthenticatedVerificationEvidenceV1,
    freshness_policy_id: FreshnessPolicyId,
    clock_lineage: Option<TrustedClockLineageV1>,
    valid_until_unix_ms: Option<u64>,
    evidence_id: FreshVerificationEvidenceId,
}
impl FreshAuthenticatedVerificationEvidenceV1 {
    pub(crate) fn contract_id(&self) -> ContinuityContractId { self.authenticated.contract_id() }
    pub(crate) fn target_realization_id(&self) -> TargetRealizationId { self.authenticated.target_realization_id() }
    pub(crate) fn requirement_id(&self) -> ContinuityRequirementId { self.authenticated.requirement_id() }
    pub(crate) fn freshness_policy_id(&self) -> FreshnessPolicyId { self.freshness_policy_id }
    pub(crate) fn verifier_dependency(&self) -> VerifierRootDependencyV1 { VerifierRootDependencyV1 { profile_id: self.authenticated.profile_id(), root_epoch: self.authenticated.root_epoch() } }
    pub(crate) fn clock_lineage(&self) -> Option<&TrustedClockLineageV1> { self.clock_lineage.as_ref() }
    pub(crate) fn valid_until_unix_ms(&self) -> Option<u64> { self.valid_until_unix_ms }
    pub(crate) fn disposition(&self) -> ObligationDispositionV1 {
        let digest = *self.evidence_id.as_bytes();
        match self.authenticated.disposition() {
            ObligationDispositionV1::Satisfied { evidence_class, .. } => ObligationDispositionV1::Satisfied { evidence_digest: digest, evidence_class },
            ObligationDispositionV1::Failed { .. } => ObligationDispositionV1::Failed { evidence_digest: digest },
            ObligationDispositionV1::Inconclusive { .. } => ObligationDispositionV1::Inconclusive { evidence_digest: digest },
            ObligationDispositionV1::InfrastructureFailure { .. } => ObligationDispositionV1::InfrastructureFailure { evidence_digest: digest },
            ObligationDispositionV1::NotExecuted { .. } => ObligationDispositionV1::NotExecuted { evidence_digest: digest },
        }
    }
}

pub(crate) fn admit_freshness(policy: &FreshnessPolicyV1, clock: Option<&TrustedClockSnapshotV1>, authenticated: AuthenticatedVerificationEvidenceV1) -> Result<FreshAuthenticatedVerificationEvidenceV1, FreshnessError> {
    if authenticated.contract_id() != policy.contract_id() { return Err(FreshnessError::EvidenceContractMismatch); }
    let requirement = policy.requirement(authenticated.requirement_id()).ok_or(FreshnessError::UnknownFreshnessRequirement)?;
    let (clock_lineage, valid_until, clock_id) = match requirement {
        EvidenceFreshnessRequirementV1::Immutable => (None, None, None),
        EvidenceFreshnessRequirementV1::MaxAge { max_age_ms, .. } => {
            let clock = clock.ok_or(FreshnessError::MissingTrustedClock)?;
            let observed = authenticated.observed_at_unix_ms();
            if observed > clock.earliest_unix_ms() { return Err(FreshnessError::PossiblyFutureEvidence); }
            let age_at_latest = clock.latest_unix_ms().checked_sub(observed).ok_or(FreshnessError::PossiblyFutureEvidence)?;
            if age_at_latest > max_age_ms { return Err(FreshnessError::StaleEvidence); }
            let valid_until = observed.checked_add(max_age_ms).ok_or(FreshnessError::ValidityOverflow)?;
            (Some(clock.lineage()), Some(valid_until), Some(clock.id()))
        }
    };
    let evidence_id = FreshVerificationEvidenceId(hash_evidence(*authenticated.id().as_bytes(), policy.id(), requirement, clock_id, valid_until));
    Ok(FreshAuthenticatedVerificationEvidenceV1 { authenticated, freshness_policy_id: policy.id(), clock_lineage, valid_until_unix_ms: valid_until, evidence_id })
}

/// Qualified witness plus the mutable verifier/clock dependencies required for
/// later commit-time currentness checks.
#[derive(Debug, Clone)]
pub struct QualifiedContinuityWitnessContextV1 {
    witness: QualifiedContinuityWitnessV1,
    freshness_policy_id: FreshnessPolicyId,
    verifier_dependencies: Vec<VerifierRootDependencyV1>,
    clock_lineage: Option<TrustedClockLineageV1>,
    valid_until_unix_ms: Option<u64>,
    context_id: QualifiedWitnessContextId,
}
impl QualifiedContinuityWitnessContextV1 {
    pub(crate) fn new(witness: QualifiedContinuityWitnessV1, policy_id: FreshnessPolicyId, evidence: &[FreshAuthenticatedVerificationEvidenceV1]) -> Result<Self, FreshnessError> {
        if evidence.iter().any(|item| item.freshness_policy_id != policy_id) { return Err(FreshnessError::MixedFreshnessPolicy); }
        let mut verifier_dependencies: Vec<_> = evidence.iter().map(FreshAuthenticatedVerificationEvidenceV1::verifier_dependency).collect();
        verifier_dependencies.sort_unstable(); verifier_dependencies.dedup();
        let mut clock_lineage = None;
        let mut valid_until_unix_ms = None;
        for item in evidence {
            if let Some(lineage) = item.clock_lineage() {
                match &clock_lineage { None => clock_lineage = Some(lineage.clone()), Some(existing) if existing == lineage => {}, Some(_) => return Err(FreshnessError::MixedTrustedClockLineage) }
            }
            if let Some(v) = item.valid_until_unix_ms() { valid_until_unix_ms = Some(valid_until_unix_ms.map_or(v, |cur: u64| cur.min(v))); }
        }
        let context_id = QualifiedWitnessContextId(hash_context(witness.id(), policy_id, &verifier_dependencies, clock_lineage.as_ref(), valid_until_unix_ms));
        Ok(Self { witness, freshness_policy_id: policy_id, verifier_dependencies, clock_lineage, valid_until_unix_ms, context_id })
    }
    pub fn id(&self) -> QualifiedWitnessContextId { self.context_id }
    pub fn witness_id(&self) -> WitnessId { self.witness.id() }
    pub fn contract_id(&self) -> ContinuityContractId { self.witness.contract_id() }
    pub fn target_realization_id(&self) -> TargetRealizationId { self.witness.target_realization_id() }
    pub fn freshness_policy_id(&self) -> FreshnessPolicyId { self.freshness_policy_id }
    pub fn verifier_dependencies(&self) -> &[VerifierRootDependencyV1] { &self.verifier_dependencies }
    pub fn clock_lineage(&self) -> Option<&TrustedClockLineageV1> { self.clock_lineage.as_ref() }
    pub fn valid_until_unix_ms(&self) -> Option<u64> { self.valid_until_unix_ms }
    pub(crate) fn witness(&self) -> &QualifiedContinuityWitnessV1 { &self.witness }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FreshnessError {
    #[error("{field} must not be blank")] BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")] TextTooLong { field: &'static str },
    #[error("{field} contains control characters")] ControlCharacters { field: &'static str },
    #[error("trusted clock epoch must be non-zero")] ZeroClockEpoch,
    #[error("trusted clock bounds must be non-zero")] ZeroClockTime,
    #[error("trusted clock earliest time exceeds latest time")] InvalidClockInterval,
    #[error("max evidence age must be non-zero")] ZeroMaxAge,
    #[error("freshness policy must contain at least one requirement")] EmptyFreshnessPolicy,
    #[error("freshness policy contains duplicate requirement identity")] DuplicateFreshnessRequirement,
    #[error("freshness policy must exactly cover the validated contract")] FreshnessContractMismatch,
    #[error("authenticated evidence belongs to a different contract")] EvidenceContractMismatch,
    #[error("authenticated evidence requirement is absent from freshness policy")] UnknownFreshnessRequirement,
    #[error("time-bounded evidence requires a trusted clock snapshot")] MissingTrustedClock,
    #[error("evidence may be in the future under current clock uncertainty")] PossiblyFutureEvidence,
    #[error("evidence is stale at the latest possible current time")] StaleEvidence,
    #[error("evidence validity bound overflowed u64")] ValidityOverflow,
    #[error("witness evidence was admitted under different freshness policies")] MixedFreshnessPolicy,
    #[error("one witness may not mix time-bounded evidence from different trusted-clock lineages in v1")] MixedTrustedClockLineage,
}

fn checked_text(field: &'static str, value: String) -> Result<String, FreshnessError> {
    let t = value.trim();
    if t.is_empty() { return Err(FreshnessError::BlankText { field }); }
    if t.len() > 1024 { return Err(FreshnessError::TextTooLong { field }); }
    if t.chars().any(char::is_control) { return Err(FreshnessError::ControlCharacters { field }); }
    Ok(t.to_string())
}
fn hash_clock(source: &str, epoch: u64, earliest: u64, latest: u64) -> [u8; 32] { let mut b=Vec::new(); put_str(&mut b,source); b.extend_from_slice(&epoch.to_le_bytes()); b.extend_from_slice(&earliest.to_le_bytes()); b.extend_from_slice(&latest.to_le_bytes()); domain_hash(CLOCK_DOMAIN,&b) }
fn hash_policy(contract: ContinuityContractId, entries: &[FreshnessPolicyEntryV1]) -> [u8; 32] { let mut b=Vec::new(); b.extend_from_slice(contract.as_bytes()); b.extend_from_slice(&(entries.len() as u64).to_le_bytes()); for e in entries { b.extend_from_slice(e.requirement_id().as_bytes()); encode_freshness(&mut b,e.freshness()); } domain_hash(POLICY_DOMAIN,&b) }
fn hash_evidence(auth:[u8;32], policy:FreshnessPolicyId, req:EvidenceFreshnessRequirementV1, clock:Option<TrustedClockSnapshotId>, valid:Option<u64>) -> [u8;32] { let mut b=Vec::new(); b.extend_from_slice(&auth); b.extend_from_slice(policy.as_bytes()); encode_freshness(&mut b,req); match clock {Some(id)=>{b.push(1);b.extend_from_slice(id.as_bytes())},None=>b.push(0)}; match valid {Some(v)=>{b.push(1);b.extend_from_slice(&v.to_le_bytes())},None=>b.push(0)}; domain_hash(EVIDENCE_DOMAIN,&b) }
fn hash_context(witness:WitnessId, policy:FreshnessPolicyId, deps:&[VerifierRootDependencyV1], clock:Option<&TrustedClockLineageV1>, valid:Option<u64>) -> [u8;32] { let mut b=Vec::new(); b.extend_from_slice(witness.as_bytes()); b.extend_from_slice(policy.as_bytes()); b.extend_from_slice(&(deps.len() as u64).to_le_bytes()); for d in deps { b.extend_from_slice(d.profile_id.as_bytes()); b.extend_from_slice(&d.root_epoch.to_le_bytes()); } match clock {Some(c)=>{b.push(1);put_str(&mut b,&c.source_id);b.extend_from_slice(&c.epoch.to_le_bytes())},None=>b.push(0)}; match valid {Some(v)=>{b.push(1);b.extend_from_slice(&v.to_le_bytes())},None=>b.push(0)}; domain_hash(CONTEXT_DOMAIN,&b) }
fn encode_freshness(b:&mut Vec<u8>, req:EvidenceFreshnessRequirementV1) { match req { EvidenceFreshnessRequirementV1::Immutable=>b.push(1), EvidenceFreshnessRequirementV1::MaxAge{class,max_age_ms}=>{b.push(2);b.push(class.tag());b.extend_from_slice(&max_age_ms.to_le_bytes())} } }
fn domain_hash(domain:&[u8], bytes:&[u8])->[u8;32]{let mut h=blake3::Hasher::new();h.update(domain);h.update(bytes);*h.finalize().as_bytes()}
fn put_str(b:&mut Vec<u8>,s:&str){b.extend_from_slice(&(s.len() as u64).to_le_bytes());b.extend_from_slice(s.as_bytes())}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate, RequirementCriticality};
    use crate::observation::{DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage, ObservationEnvelopeV1};
    use crate::verifier::{policy_check_verification_evidence, AuthenticatedVerificationEvidenceV1, VerificationEvidenceClaimV1, VerificationOutcomeV1, VerifierProfileV1};
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1, VerificationPolicyV1};

    fn contract()->ValidatedContinuityContractV1{let o=ObservationEnvelopeV1::new("m","dep","fixture","1",1_700_000_000_000,ObservationCoverage::Complete,EvidenceBasis::Tested,[1;32],vec![]).unwrap();let d=DependencyClaimV1::new("role","requires","cap",DependencyBasis::Observed,vec![o.id()],vec![]).unwrap();let r=ContinuityRequirementV1::new(d.id(),"cap",RequirementCriticality::Must,EquivalencePredicate::BehavioralScenario{scenario_id:"s".into()},ApprovalBasis::ExplicitPolicy,[2;32]).unwrap();ContinuityContractV1::new("fleet",[3;32],vec![r]).unwrap().validate().unwrap()}
    fn authenticated(c:&ValidatedContinuityContractV1,observed:u64)->AuthenticatedVerificationEvidenceV1{let target=TargetRealizationId::from_digest([4;32]).unwrap();let p=VerifierProfileV1::new("v",[5;32],2,EvidenceClass::HardwareVerified).unwrap();let vp=VerificationPolicyV1::new("v",vec![VerificationPolicyEntryV1::new(c.requirements()[0].id(),EvidenceClass::Simulated)]).unwrap();let claim=VerificationEvidenceClaimV1::new(c.id(),target,c.requirements()[0].id(),p.id(),[6;32],observed,VerificationOutcomeV1::Satisfied,[7;32]).unwrap();let checked=policy_check_verification_evidence(c,target,&vp,&p,[6;32],claim).unwrap();AuthenticatedVerificationEvidenceV1::authenticate_for_test(checked,[8;32]).unwrap()}

    #[test] fn possible_future_is_rejected(){let c=contract();let fp=FreshnessPolicyV1::new(&c,vec![FreshnessPolicyEntryV1::new(c.requirements()[0].id(),EvidenceFreshnessRequirementV1::MaxAge{class:EvidenceStabilityClassV1::Volatile,max_age_ms:1000}).unwrap()]).unwrap();let clock=TrustedClockSnapshotV1::new("clock",1,1000,1100).unwrap();assert_eq!(admit_freshness(&fp,Some(&clock),authenticated(&c,1050)).unwrap_err(),FreshnessError::PossiblyFutureEvidence)}
    #[test] fn stale_at_latest_bound_is_rejected(){let c=contract();let fp=FreshnessPolicyV1::new(&c,vec![FreshnessPolicyEntryV1::new(c.requirements()[0].id(),EvidenceFreshnessRequirementV1::MaxAge{class:EvidenceStabilityClassV1::Volatile,max_age_ms:100}).unwrap()]).unwrap();let clock=TrustedClockSnapshotV1::new("clock",1,1000,1201).unwrap();assert_eq!(admit_freshness(&fp,Some(&clock),authenticated(&c,1000)).unwrap_err(),FreshnessError::StaleEvidence)}
    #[test] fn immutable_needs_no_clock(){let c=contract();let fp=FreshnessPolicyV1::new(&c,vec![FreshnessPolicyEntryV1::new(c.requirements()[0].id(),EvidenceFreshnessRequirementV1::Immutable).unwrap()]).unwrap();let fresh=admit_freshness(&fp,None,authenticated(&c,1000)).unwrap();assert!(fresh.clock_lineage().is_none());assert!(fresh.valid_until_unix_ms().is_none())}
}
