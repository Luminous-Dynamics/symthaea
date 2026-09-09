// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commit-time continuity preconditions.
//!
//! `QualifiedWitnessContext != CurrentCommitEligibility != ExecutionAuthority`.

use thiserror::Error;

use crate::contract::ContinuityContractId;
use crate::freshness::{
    QualifiedContinuityWitnessContextV1, QualifiedWitnessContextId, TrustedClockSnapshotV1,
};
use crate::verifier::{VerifierProfileId, VerifierProfileV1};
use crate::witness::{TargetRealizationId, WitnessId};

const STATE_DOMAIN: &[u8] = b"symthaea.continuity.transition-state.v2\0";
const PRECONDITIONS_DOMAIN: &[u8] = b"symthaea.continuity.commit-preconditions.v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransitionStateSnapshotId([u8; 32]);
impl TransitionStateSnapshotId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransitionCommitPreconditionsId([u8; 32]);
impl TransitionCommitPreconditionsId { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CurrentVerifierRootV1 { profile_id: VerifierProfileId, root_epoch: u64 }
impl CurrentVerifierRootV1 {
    pub fn new(profile_id: VerifierProfileId, root_epoch: u64) -> Result<Self, CommitPreconditionsError> {
        if root_epoch == 0 { return Err(CommitPreconditionsError::ZeroVerifierRootEpoch); }
        Ok(Self { profile_id, root_epoch })
    }
    pub fn from_profile(profile: &VerifierProfileV1) -> Self { Self { profile_id: profile.id(), root_epoch: profile.root_epoch() } }
    pub fn profile_id(&self) -> VerifierProfileId { self.profile_id }
    pub fn root_epoch(&self) -> u64 { self.root_epoch }
}

/// Current mutable transition state. This remains evidence, not authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionStateSnapshotV1 {
    context_id: QualifiedWitnessContextId,
    witness_id: WitnessId,
    contract_id: ContinuityContractId,
    target_realization_id: TargetRealizationId,
    fleet_id: String,
    fleet_generation: u64,
    machine_binding_digest: [u8; 32],
    machine_identity_epoch: u64,
    authority_epoch: u64,
    current_state_digest: [u8; 32],
    rollback_target_digest: [u8; 32],
    verifier_roots: Vec<CurrentVerifierRootV1>,
    trusted_clock: Option<TrustedClockSnapshotV1>,
    snapshot_id: TransitionStateSnapshotId,
}

impl TransitionStateSnapshotV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &QualifiedContinuityWitnessContextV1,
        fleet_id: impl Into<String>,
        fleet_generation: u64,
        machine_binding_digest: [u8; 32],
        machine_identity_epoch: u64,
        authority_epoch: u64,
        current_state_digest: [u8; 32],
        rollback_target_digest: [u8; 32],
        mut verifier_roots: Vec<CurrentVerifierRootV1>,
        trusted_clock: Option<TrustedClockSnapshotV1>,
    ) -> Result<Self, CommitPreconditionsError> {
        let fleet_id = checked_text("fleet_id", fleet_id.into())?;
        validate_nonzero(fleet_generation, machine_binding_digest, machine_identity_epoch, authority_epoch, current_state_digest, rollback_target_digest)?;
        verifier_roots.sort_unstable();
        if verifier_roots.windows(2).any(|p| p[0] == p[1]) { return Err(CommitPreconditionsError::DuplicateVerifierRoot); }
        let snapshot_id = TransitionStateSnapshotId(hash_state(
            context.id(), context.witness_id(), context.contract_id(), context.target_realization_id(),
            &fleet_id, fleet_generation, machine_binding_digest, machine_identity_epoch,
            authority_epoch, current_state_digest, rollback_target_digest, &verifier_roots,
            trusted_clock.as_ref(),
        ));
        Ok(Self {
            context_id: context.id(), witness_id: context.witness_id(), contract_id: context.contract_id(),
            target_realization_id: context.target_realization_id(), fleet_id, fleet_generation,
            machine_binding_digest, machine_identity_epoch, authority_epoch, current_state_digest,
            rollback_target_digest, verifier_roots, trusted_clock, snapshot_id,
        })
    }

    pub fn id(&self) -> TransitionStateSnapshotId { self.snapshot_id }
    pub fn fleet_id(&self) -> &str { &self.fleet_id }
    pub fn fleet_generation(&self) -> u64 { self.fleet_generation }
    pub fn machine_binding_digest(&self) -> [u8; 32] { self.machine_binding_digest }
    pub fn machine_identity_epoch(&self) -> u64 { self.machine_identity_epoch }
    pub fn authority_epoch(&self) -> u64 { self.authority_epoch }
    pub fn current_state_digest(&self) -> [u8; 32] { self.current_state_digest }
    pub fn rollback_target_digest(&self) -> [u8; 32] { self.rollback_target_digest }
    pub fn verifier_roots(&self) -> &[CurrentVerifierRootV1] { &self.verifier_roots }
    pub fn trusted_clock(&self) -> Option<&TrustedClockSnapshotV1> { self.trusted_clock.as_ref() }

    fn validate_structure(&self, context: &QualifiedContinuityWitnessContextV1) -> Result<(), CommitPreconditionsError> {
        checked_text("fleet_id", self.fleet_id.clone())?;
        validate_nonzero(self.fleet_generation, self.machine_binding_digest, self.machine_identity_epoch, self.authority_epoch, self.current_state_digest, self.rollback_target_digest)?;
        if self.context_id != context.id() || self.witness_id != context.witness_id() || self.contract_id != context.contract_id() || self.target_realization_id != context.target_realization_id() {
            return Err(CommitPreconditionsError::WitnessContextBindingMismatch);
        }
        if self.verifier_roots.windows(2).any(|p| p[0] >= p[1]) { return Err(CommitPreconditionsError::NonCanonicalVerifierRoots); }
        let expected = TransitionStateSnapshotId(hash_state(
            self.context_id, self.witness_id, self.contract_id, self.target_realization_id,
            &self.fleet_id, self.fleet_generation, self.machine_binding_digest,
            self.machine_identity_epoch, self.authority_epoch, self.current_state_digest,
            self.rollback_target_digest, &self.verifier_roots, self.trusted_clock.as_ref(),
        ));
        if expected != self.snapshot_id { return Err(CommitPreconditionsError::StateIdentityMismatch); }
        Ok(())
    }

    fn validate_context_dependencies(&self, context: &QualifiedContinuityWitnessContextV1) -> Result<(), CommitPreconditionsError> {
        let expected_roots: Vec<_> = context.verifier_dependencies().iter().map(|d| CurrentVerifierRootV1 { profile_id: d.profile_id(), root_epoch: d.root_epoch() }).collect();
        if self.verifier_roots != expected_roots { return Err(CommitPreconditionsError::VerifierRootsChanged); }
        match (context.clock_lineage(), context.valid_until_unix_ms(), self.trusted_clock.as_ref()) {
            (None, None, None) => {}
            (None, None, Some(_)) => return Err(CommitPreconditionsError::UnexpectedTrustedClock),
            (Some(lineage), Some(valid_until), Some(clock)) => {
                let current_lineage = clock.lineage();
                if &current_lineage != lineage { return Err(CommitPreconditionsError::TrustedClockLineageChanged); }
                if clock.latest_unix_ms() > valid_until { return Err(CommitPreconditionsError::WitnessExpired); }
            }
            (Some(_), Some(_), None) => return Err(CommitPreconditionsError::MissingTrustedClock),
            _ => return Err(CommitPreconditionsError::InvalidWitnessClockContext),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionCommitPreconditionsV1 {
    context_id: QualifiedWitnessContextId,
    expected_state: TransitionStateSnapshotV1,
    preconditions_id: TransitionCommitPreconditionsId,
}

impl TransitionCommitPreconditionsV1 {
    pub fn capture(context: &QualifiedContinuityWitnessContextV1, expected_state: TransitionStateSnapshotV1) -> Result<Self, CommitPreconditionsError> {
        expected_state.validate_structure(context)?;
        expected_state.validate_context_dependencies(context)?;
        let preconditions_id = TransitionCommitPreconditionsId(hash_preconditions(context.id(), expected_state.id()));
        Ok(Self { context_id: context.id(), expected_state, preconditions_id })
    }
    pub fn id(&self) -> TransitionCommitPreconditionsId { self.preconditions_id }
    pub fn expected_state(&self) -> &TransitionStateSnapshotV1 { &self.expected_state }

    pub fn revalidate(&self, context: &QualifiedContinuityWitnessContextV1, current: &TransitionStateSnapshotV1) -> Result<CommitPreconditionsSatisfiedV1, CommitPreconditionsError> {
        if self.context_id != context.id() { return Err(CommitPreconditionsError::WitnessContextBindingMismatch); }
        self.expected_state.validate_structure(context)?;
        self.expected_state.validate_context_dependencies(context)?;
        current.validate_structure(context)?;
        current.validate_context_dependencies(context)?;
        let expected_id = TransitionCommitPreconditionsId(hash_preconditions(self.context_id, self.expected_state.id()));
        if expected_id != self.preconditions_id { return Err(CommitPreconditionsError::PreconditionsIdentityMismatch); }

        if current.fleet_id != self.expected_state.fleet_id { return Err(CommitPreconditionsError::FleetIdentityChanged); }
        if current.fleet_generation != self.expected_state.fleet_generation { return Err(CommitPreconditionsError::FleetGenerationChanged); }
        if current.machine_binding_digest != self.expected_state.machine_binding_digest { return Err(CommitPreconditionsError::MachineBindingChanged); }
        if current.machine_identity_epoch != self.expected_state.machine_identity_epoch { return Err(CommitPreconditionsError::MachineIdentityEpochChanged); }
        if current.authority_epoch != self.expected_state.authority_epoch { return Err(CommitPreconditionsError::AuthorityEpochChanged); }
        if current.current_state_digest != self.expected_state.current_state_digest { return Err(CommitPreconditionsError::CurrentStateChanged); }
        if current.rollback_target_digest != self.expected_state.rollback_target_digest { return Err(CommitPreconditionsError::RollbackTargetChanged); }

        Ok(CommitPreconditionsSatisfiedV1 { preconditions_id: self.preconditions_id, context_id: context.id(), witness_id: context.witness_id(), current_state_id: current.id() })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitPreconditionsSatisfiedV1 {
    preconditions_id: TransitionCommitPreconditionsId,
    context_id: QualifiedWitnessContextId,
    witness_id: WitnessId,
    current_state_id: TransitionStateSnapshotId,
}
impl CommitPreconditionsSatisfiedV1 {
    pub fn preconditions_id(&self) -> TransitionCommitPreconditionsId { self.preconditions_id }
    pub fn context_id(&self) -> QualifiedWitnessContextId { self.context_id }
    pub fn witness_id(&self) -> WitnessId { self.witness_id }
    pub fn current_state_id(&self) -> TransitionStateSnapshotId { self.current_state_id }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommitPreconditionsError {
    #[error("{field} must not be blank")] BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")] TextTooLong { field: &'static str },
    #[error("{field} contains control characters")] ControlCharacters { field: &'static str },
    #[error("fleet generation must be non-zero")] ZeroFleetGeneration,
    #[error("machine binding digest must be non-zero")] ZeroMachineBinding,
    #[error("machine identity epoch must be non-zero")] ZeroMachineIdentityEpoch,
    #[error("authority epoch must be non-zero")] ZeroAuthorityEpoch,
    #[error("current state digest must be non-zero")] ZeroCurrentState,
    #[error("rollback target digest must be non-zero")] ZeroRollbackTarget,
    #[error("verifier root epoch must be non-zero")] ZeroVerifierRootEpoch,
    #[error("verifier root list contains a duplicate")]
    DuplicateVerifierRoot,
    #[error("verifier roots must be in canonical strict order")]
    NonCanonicalVerifierRoots,
    #[error("state snapshot is bound to a different qualified witness context")]
    WitnessContextBindingMismatch,
    #[error("stored transition-state identity does not match canonical fields")]
    StateIdentityMismatch,
    #[error("stored commit-preconditions identity does not match canonical fields")]
    PreconditionsIdentityMismatch,
    #[error("current verifier root/profile epochs differ from the witness dependencies")]
    VerifierRootsChanged,
    #[error("time-bounded witness requires a trusted clock snapshot")]
    MissingTrustedClock,
    #[error("witness without a clock dependency received an unexpected trusted clock")]
    UnexpectedTrustedClock,
    #[error("trusted clock source or epoch changed since witness qualification")]
    TrustedClockLineageChanged,
    #[error("qualified witness evidence has expired")]
    WitnessExpired,
    #[error("qualified witness has inconsistent clock-lineage/validity metadata")]
    InvalidWitnessClockContext,
    #[error("fleet identity changed since preconditions were captured")]
    FleetIdentityChanged,
    #[error("fleet generation changed since preconditions were captured")]
    FleetGenerationChanged,
    #[error("machine binding changed since preconditions were captured")]
    MachineBindingChanged,
    #[error("machine identity epoch changed since preconditions were captured")]
    MachineIdentityEpochChanged,
    #[error("authority epoch changed since preconditions were captured")]
    AuthorityEpochChanged,
    #[error("current machine state changed since preconditions were captured")]
    CurrentStateChanged,
    #[error("rollback target changed since preconditions were captured")]
    RollbackTargetChanged,
}

#[allow(clippy::too_many_arguments)]
fn validate_nonzero(fleet_generation:u64,machine:[u8;32],machine_epoch:u64,authority_epoch:u64,current:[u8;32],rollback:[u8;32])->Result<(),CommitPreconditionsError>{if fleet_generation==0{return Err(CommitPreconditionsError::ZeroFleetGeneration)}if machine==[0;32]{return Err(CommitPreconditionsError::ZeroMachineBinding)}if machine_epoch==0{return Err(CommitPreconditionsError::ZeroMachineIdentityEpoch)}if authority_epoch==0{return Err(CommitPreconditionsError::ZeroAuthorityEpoch)}if current==[0;32]{return Err(CommitPreconditionsError::ZeroCurrentState)}if rollback==[0;32]{return Err(CommitPreconditionsError::ZeroRollbackTarget)}Ok(())}
fn checked_text(field:&'static str,value:String)->Result<String,CommitPreconditionsError>{let t=value.trim();if t.is_empty(){return Err(CommitPreconditionsError::BlankText{field})}if t.len()>1024{return Err(CommitPreconditionsError::TextTooLong{field})}if t.chars().any(char::is_control){return Err(CommitPreconditionsError::ControlCharacters{field})}Ok(t.to_string())}

#[allow(clippy::too_many_arguments)]
fn hash_state(context:QualifiedWitnessContextId,witness:WitnessId,contract:ContinuityContractId,target:TargetRealizationId,fleet:&str,generation:u64,machine:[u8;32],machine_epoch:u64,authority_epoch:u64,current:[u8;32],rollback:[u8;32],roots:&[CurrentVerifierRootV1],clock:Option<&TrustedClockSnapshotV1>)->[u8;32]{let mut b=Vec::new();b.extend_from_slice(context.as_bytes());b.extend_from_slice(witness.as_bytes());b.extend_from_slice(contract.as_bytes());b.extend_from_slice(target.as_bytes());put_str(&mut b,fleet);b.extend_from_slice(&generation.to_le_bytes());b.extend_from_slice(&machine);b.extend_from_slice(&machine_epoch.to_le_bytes());b.extend_from_slice(&authority_epoch.to_le_bytes());b.extend_from_slice(&current);b.extend_from_slice(&rollback);b.extend_from_slice(&(roots.len() as u64).to_le_bytes());for r in roots{b.extend_from_slice(r.profile_id.as_bytes());b.extend_from_slice(&r.root_epoch.to_le_bytes())}match clock{Some(c)=>{b.push(1);b.extend_from_slice(c.id().as_bytes())},None=>b.push(0)}domain_hash(STATE_DOMAIN,&b)}
fn hash_preconditions(context:QualifiedWitnessContextId,state:TransitionStateSnapshotId)->[u8;32]{let mut b=Vec::new();b.extend_from_slice(context.as_bytes());b.extend_from_slice(state.as_bytes());domain_hash(PRECONDITIONS_DOMAIN,&b)}
fn domain_hash(domain:&[u8],bytes:&[u8])->[u8;32]{let mut h=blake3::Hasher::new();h.update(domain);h.update(bytes);*h.finalize().as_bytes()}
fn put_str(b:&mut Vec<u8>,s:&str){b.extend_from_slice(&(s.len() as u64).to_le_bytes());b.extend_from_slice(s.as_bytes())}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::compose_qualified_witness_context;
    use crate::contract::{ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate, RequirementCriticality};
    use crate::freshness::{admit_freshness, EvidenceFreshnessRequirementV1, EvidenceStabilityClassV1, FreshnessPolicyEntryV1, FreshnessPolicyV1};
    use crate::observation::{DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage, ObservationEnvelopeV1};
    use crate::verifier::{policy_check_verification_evidence, AuthenticatedVerificationEvidenceV1, VerificationEvidenceClaimV1, VerificationOutcomeV1};
    use crate::witness::{EvidenceClass, VerificationPolicyEntryV1, VerificationPolicyV1};

    fn fixture() -> (QualifiedContinuityWitnessContextV1, VerifierProfileV1) {
        let o=ObservationEnvelopeV1::new("m","dep","fixture","1",1000,ObservationCoverage::Complete,EvidenceBasis::Tested,[1;32],vec![]).unwrap();
        let d=DependencyClaimV1::new("role","requires","cap",DependencyBasis::Observed,vec![o.id()],vec![]).unwrap();
        let r=ContinuityRequirementV1::new(d.id(),"cap",RequirementCriticality::Must,EquivalencePredicate::BehavioralScenario{scenario_id:"s".into()},ApprovalBasis::ExplicitPolicy,[2;32]).unwrap();
        let c=ContinuityContractV1::new("fleet",[3;32],vec![r]).unwrap().validate().unwrap();
        let target=TargetRealizationId::from_digest([4;32]).unwrap();
        let profile=VerifierProfileV1::new("v",[5;32],2,EvidenceClass::HardwareVerified).unwrap();
        let vp=VerificationPolicyV1::new("v",vec![VerificationPolicyEntryV1::new(c.requirements()[0].id(),EvidenceClass::Simulated)]).unwrap();
        let claim=VerificationEvidenceClaimV1::new(c.id(),target,c.requirements()[0].id(),profile.id(),[6;32],1000,VerificationOutcomeV1::Satisfied,[7;32]).unwrap();
        let checked=policy_check_verification_evidence(&c,target,&vp,&profile,[6;32],claim).unwrap();
        let auth=AuthenticatedVerificationEvidenceV1::authenticate_for_test(checked,[8;32]).unwrap();
        let fp=FreshnessPolicyV1::new(&c,vec![FreshnessPolicyEntryV1::new(c.requirements()[0].id(),EvidenceFreshnessRequirementV1::MaxAge{class:EvidenceStabilityClassV1::Volatile,max_age_ms:1000}).unwrap()]).unwrap();
        let clock=TrustedClockSnapshotV1::new("clock",1,1100,1150).unwrap();
        let fresh=admit_freshness(&fp,Some(&clock),auth).unwrap();
        (compose_qualified_witness_context(&c,target,&vp,&fp,vec![fresh]).unwrap(),profile)
    }
    fn state(ctx:&QualifiedContinuityWitnessContextV1,profile:&VerifierProfileV1,generation:u64,rollback:[u8;32],clock_latest:u64)->TransitionStateSnapshotV1{
        TransitionStateSnapshotV1::new(ctx,"fleet-a",generation,[10;32],1,1,[11;32],rollback,vec![CurrentVerifierRootV1::from_profile(profile)],Some(TrustedClockSnapshotV1::new("clock",1,clock_latest-10,clock_latest).unwrap())).unwrap()
    }

    #[test] fn unchanged_current_state_satisfies(){let (ctx,p)=fixture();let expected=state(&ctx,&p,1,[12;32],1200);let pre=TransitionCommitPreconditionsV1::capture(&ctx,expected).unwrap();let current=state(&ctx,&p,1,[12;32],1500);assert!(pre.revalidate(&ctx,&current).is_ok())}
    #[test] fn fleet_generation_drift_fails(){let (ctx,p)=fixture();let pre=TransitionCommitPreconditionsV1::capture(&ctx,state(&ctx,&p,1,[12;32],1200)).unwrap();assert_eq!(pre.revalidate(&ctx,&state(&ctx,&p,2,[12;32],1300)).unwrap_err(),CommitPreconditionsError::FleetGenerationChanged)}
    #[test] fn verifier_root_rotation_fails(){let (ctx,p)=fixture();let pre=TransitionCommitPreconditionsV1::capture(&ctx,state(&ctx,&p,1,[12;32],1200)).unwrap();let rotated=VerifierProfileV1::new("v",[5;32],3,EvidenceClass::HardwareVerified).unwrap();let current=state(&ctx,&rotated,1,[12;32],1300);assert_eq!(pre.revalidate(&ctx,&current).unwrap_err(),CommitPreconditionsError::VerifierRootsChanged)}
    #[test] fn witness_expiry_fails(){let (ctx,p)=fixture();let pre=TransitionCommitPreconditionsV1::capture(&ctx,state(&ctx,&p,1,[12;32],1200)).unwrap();assert_eq!(pre.revalidate(&ctx,&state(&ctx,&p,1,[12;32],2001)).unwrap_err(),CommitPreconditionsError::WitnessExpired)}
    #[test] fn rollback_drift_fails(){let (ctx,p)=fixture();let pre=TransitionCommitPreconditionsV1::capture(&ctx,state(&ctx,&p,1,[12;32],1200)).unwrap();assert_eq!(pre.revalidate(&ctx,&state(&ctx,&p,1,[13;32],1300)).unwrap_err(),CommitPreconditionsError::RollbackTargetChanged)}
}
