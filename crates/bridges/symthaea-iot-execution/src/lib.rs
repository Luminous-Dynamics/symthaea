// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact execution binding above `symthaea-iot-authority`.
//!
//! `symthaea-iot-authority` answers whether a command is authorized and inside
//! its physical safety envelope. This crate narrows that successful admission
//! to an exact plan, an exact safety-relevant world snapshot, and a non-zero
//! integer risk charge suitable for later reservation by
//! `symthaea-action-runtime`.
//!
//! This crate is still cognition-free and I/O-free. It does not persist a
//! reservation, dispatch an actuator, authenticate a transport, or prove an
//! external effect occurred.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_authority::{
    AuthorityContext, CapabilityGrant, Digest32, NegativeAuthorityFact, RiskBudget,
};
use symthaea_iot_authority::{
    CyberPhysicalAdmission, CyberPhysicalDecision, CyberPhysicalDenyReason, DeviceCommand,
    DeviceRuntimeState, SafetyEnvelope, evaluate_cyber_physical_command,
};

/// Domain separator for execution-proposal commitments.
pub const EXECUTION_PROPOSAL_DOMAIN: &[u8] = b"symthaea-iot-execution-proposal-v1";
/// Domain separator for safety-relevant physical-world commitments.
pub const SAFETY_WORLD_DOMAIN: &[u8] = b"symthaea-iot-safety-world-v1";

/// One physical effect bound to planning, world state, and accounting cost.
///
/// `world_digest` is mandatory even when the parent grant is not world-bound.
/// This deliberately makes the final execution proposal narrower than a broad
/// capability: a physical command must be re-evaluated if the safety-relevant
/// world changes before dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalExecutionProposal {
    /// The already-typed device command evaluated by the lower admission layer.
    pub command: DeviceCommand,
    /// Exact plan commitment when execution belongs to a larger plan.
    pub plan_digest: Option<Digest32>,
    /// Exact safety-relevant world snapshot against which this effect was proposed.
    pub world_digest: Digest32,
    /// Consequence charge that a crash-conservative runtime must reserve before dispatch.
    pub risk_charge: RiskBudget,
}

impl PhysicalExecutionProposal {
    /// Deterministic commitment to the complete execution proposal.
    pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(EXECUTION_PROPOSAL_DOMAIN);
        t.digest(self.command.digest());
        t.optional_digest(self.plan_digest);
        t.digest(self.world_digest);
        t.risk(self.risk_charge);
        Digest32(*t.finish().as_bytes())
    }
}

/// Successful lower-layer admission plus exact execution context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundExecutionAdmission {
    /// Admission produced by `symthaea-iot-authority`.
    pub cyber_physical: CyberPhysicalAdmission,
    /// Exact proposal commitment to retain across reservation/dispatch/evidence.
    pub proposal_digest: Digest32,
    /// Recomputed safety-relevant world commitment.
    pub runtime_world_digest: Digest32,
    /// Exact risk charge to reserve before dispatch.
    pub risk_charge: RiskBudget,
}

/// Stable reason the stronger execution-binding layer failed closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundExecutionDenyReason {
    /// The lower cyber-physical authority/safety layer denied the command.
    CyberPhysical(CyberPhysicalDenyReason),
    /// A grant-bound plan was omitted or substituted.
    PlanBindingMismatch,
    /// A grant-bound world was omitted/substituted by the proposal.
    GrantWorldBindingMismatch,
    /// The physical world changed relative to the proposal's world commitment.
    RuntimeWorldMismatch,
    /// Physical-effect accounting cannot proceed without an explicit non-zero charge.
    RiskChargeRequired,
    /// The single-effect charge is already broader than the grant's total risk ceiling.
    RiskChargeExceedsGrant,
}

/// Result of exact execution binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundExecutionDecision {
    Allow(BoundExecutionAdmission),
    Deny(BoundExecutionDenyReason),
}

/// Commit the safety-relevant physical state used by `SafetyEnvelope`.
///
/// The digest intentionally excludes `last_accepted_sequence`: anti-replay state is
/// a separate execution-generation concern. It includes the exact protected device,
/// operation, running firmware, safety policy commitment, and every observation the
/// safety envelope requires, in canonical `BTreeMap` order.
///
/// Returns `None` if a required observation is absent. A successful lower-layer
/// admission guarantees those observations exist, but the public helper remains
/// defensive when called independently.
pub fn safety_world_digest(
    runtime: &DeviceRuntimeState,
    safety: &SafetyEnvelope,
) -> Option<Digest32> {
    let mut t = Transcript::new(SAFETY_WORLD_DOMAIN);
    t.string(&safety.device.0);
    t.string(&safety.operation.0);
    t.digest(safety.digest());
    t.digest(runtime.running_firmware);
    t.u32(safety.required_observations.len() as u32);
    for name in safety.required_observations.keys() {
        let value = runtime.observations.get(name)?;
        t.string(name);
        t.i64(*value);
    }
    Some(Digest32(*t.finish().as_bytes()))
}

/// Evaluate a physical execution against authority, safety, exact plan/world
/// commitments, and one-effect risk accounting.
///
/// This is still only *admission*. A consequential caller must reserve
/// `admission.risk_charge` and a use in durable action-runtime state before
/// dispatch. Ambiguous external outcomes must remain charged until reconciled.
pub fn evaluate_bound_execution(
    grant: &CapabilityGrant,
    authority_context: AuthorityContext,
    negative_facts: &[NegativeAuthorityFact],
    proposal: &PhysicalExecutionProposal,
    runtime: &DeviceRuntimeState,
    safety: &SafetyEnvelope,
) -> BoundExecutionDecision {
    let cyber_physical = match evaluate_cyber_physical_command(
        grant,
        authority_context,
        negative_facts,
        &proposal.command,
        runtime,
        safety,
    ) {
        CyberPhysicalDecision::Allow(admission) => admission,
        CyberPhysicalDecision::Deny(reason) => {
            return BoundExecutionDecision::Deny(BoundExecutionDenyReason::CyberPhysical(reason));
        }
    };

    if grant
        .plan_digest
        .is_some_and(|required| proposal.plan_digest != Some(required))
    {
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::PlanBindingMismatch);
    }

    if grant
        .world_digest
        .is_some_and(|required| proposal.world_digest != required)
    {
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::GrantWorldBindingMismatch);
    }

    let Some(runtime_world_digest) = safety_world_digest(runtime, safety) else {
        // Normally unreachable after successful lower-layer safety admission.
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::RuntimeWorldMismatch);
    };
    if proposal.world_digest != runtime_world_digest {
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::RuntimeWorldMismatch);
    }

    if proposal.risk_charge == RiskBudget::default() {
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::RiskChargeRequired);
    }
    if !proposal.risk_charge.attenuates(grant.risk_budget) {
        return BoundExecutionDecision::Deny(BoundExecutionDenyReason::RiskChargeExceedsGrant);
    }

    BoundExecutionDecision::Allow(BoundExecutionAdmission {
        cyber_physical,
        proposal_digest: proposal.digest(),
        runtime_world_digest,
        risk_charge: proposal.risk_charge,
    })
}

struct Transcript(blake3::Hasher);

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(domain.len() as u32).to_be_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    fn u32(&mut self, value: u32) {
        self.0.update(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.0.update(&value.to_be_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.0.update(value.as_bytes());
    }

    fn digest(&mut self, Digest32(value): Digest32) {
        self.0.update(&value);
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(value) => {
                self.0.update(&[1]);
                self.digest(value);
            }
            None => self.0.update(&[0]),
        }
    }

    fn risk(&mut self, risk: RiskBudget) {
        self.0.update(&risk.mutation_units.to_be_bytes());
        self.0.update(&risk.irreversible_units.to_be_bytes());
        self.0
            .update(&risk.external_disclosure_bytes.to_be_bytes());
        self.0.update(&risk.monetary_microunits.to_be_bytes());
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use symthaea_authority::{
        AuthorityEpoch, GrantUseState, Operation, PrincipalId, ResourceRef, TaskId,
    };
    use symthaea_iot_authority::{InclusiveRangeI64, DEVICE_COMMAND_SCHEMA_VERSION, SAFETY_ENVELOPE_SCHEMA_VERSION};

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn risk(units: u64) -> RiskBudget {
        RiskBudget {
            mutation_units: units,
            ..RiskBudget::default()
        }
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "g-valve",
            PrincipalId("human:operator".into()),
            PrincipalId("agent:irrigation".into()),
            AuthorityEpoch(9),
        );
        grant.audience = Some(PrincipalId("gateway:field-a".into()));
        grant.task = Some(TaskId("irrigate:zone-7".into()));
        grant.resources = BTreeSet::from([ResourceRef("iot:valve:72".into())]);
        grant.operations = BTreeSet::from([Operation("valve.open".into())]);
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = 3;
        grant.risk_budget = risk(5);
        grant
    }

    fn context(grant: &CapabilityGrant) -> AuthorityContext {
        AuthorityContext {
            now_unix_s: 5_000,
            current_epoch: grant.authority_epoch,
            use_state: GrantUseState::default(),
        }
    }

    fn command() -> DeviceCommand {
        DeviceCommand {
            schema_version: DEVICE_COMMAND_SCHEMA_VERSION,
            command_id: "cmd-43".into(),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: Some(TaskId("irrigate:zone-7".into())),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: digest(7),
            sequence: 43,
            issued_at_unix_s: 4_995,
            expires_at_unix_s: 5_020,
            parameters: BTreeMap::from([("duration_ms".into(), 60_000)]),
        }
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "safe-valve-open".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: BTreeSet::from([digest(7)]),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 120_000,
                },
            )]),
            required_observations: BTreeMap::from([(
                "tank_pressure_kpa_x100".into(),
                InclusiveRangeI64 {
                    min: 100,
                    max: 350_000,
                },
            )]),
        }
    }

    fn runtime() -> DeviceRuntimeState {
        DeviceRuntimeState {
            running_firmware: digest(7),
            last_accepted_sequence: Some(42),
            observations: BTreeMap::from([("tank_pressure_kpa_x100".into(), 210_000)]),
        }
    }

    fn proposal(runtime: &DeviceRuntimeState, safety: &SafetyEnvelope) -> PhysicalExecutionProposal {
        PhysicalExecutionProposal {
            command: command(),
            plan_digest: Some(digest(3)),
            world_digest: safety_world_digest(runtime, safety).unwrap(),
            risk_charge: risk(1),
        }
    }

    #[test]
    fn exact_plan_world_and_risk_are_admitted() {
        let mut grant = grant();
        let runtime = runtime();
        let safety = safety();
        let proposal = proposal(&runtime, &safety);
        grant.plan_digest = proposal.plan_digest;
        grant.world_digest = Some(proposal.world_digest);

        let decision = evaluate_bound_execution(
            &grant,
            context(&grant),
            &[],
            &proposal,
            &runtime,
            &safety,
        );
        let BoundExecutionDecision::Allow(admission) = decision else {
            panic!("expected bound execution admission");
        };
        assert_eq!(admission.proposal_digest, proposal.digest());
        assert_eq!(admission.runtime_world_digest, proposal.world_digest);
        assert_eq!(admission.risk_charge, risk(1));
        assert_eq!(admission.cyber_physical.command_digest, proposal.command.digest());
    }

    #[test]
    fn grant_bound_plan_cannot_be_omitted_or_substituted() {
        let mut grant = grant();
        grant.plan_digest = Some(digest(3));
        let runtime = runtime();
        let safety = safety();

        let mut omitted = proposal(&runtime, &safety);
        omitted.plan_digest = None;
        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &omitted,
                &runtime,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::PlanBindingMismatch)
        );

        let mut substituted = proposal(&runtime, &safety);
        substituted.plan_digest = Some(digest(4));
        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &substituted,
                &runtime,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::PlanBindingMismatch)
        );
    }

    #[test]
    fn changed_physical_observation_invalidates_proposal() {
        let grant = grant();
        let original = runtime();
        let safety = safety();
        let proposal = proposal(&original, &safety);

        let mut changed = original.clone();
        changed
            .observations
            .insert("tank_pressure_kpa_x100".into(), 220_000);
        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &proposal,
                &changed,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::RuntimeWorldMismatch)
        );
    }

    #[test]
    fn unrelated_observation_does_not_change_safety_world() {
        let mut a = runtime();
        let safety = safety();
        let before = safety_world_digest(&a, &safety).unwrap();
        a.observations.insert("ambient_note".into(), 99);
        let after = safety_world_digest(&a, &safety).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn grant_world_binding_must_match_proposal_world() {
        let mut grant = grant();
        grant.world_digest = Some(digest(9));
        let runtime = runtime();
        let safety = safety();
        let proposal = proposal(&runtime, &safety);

        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &proposal,
                &runtime,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::GrantWorldBindingMismatch)
        );
    }

    #[test]
    fn risk_charge_must_be_explicit_and_within_grant_ceiling() {
        let grant = grant();
        let runtime = runtime();
        let safety = safety();

        let mut zero = proposal(&runtime, &safety);
        zero.risk_charge = RiskBudget::default();
        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &zero,
                &runtime,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::RiskChargeRequired)
        );

        let mut excessive = proposal(&runtime, &safety);
        excessive.risk_charge = risk(6);
        assert_eq!(
            evaluate_bound_execution(
                &grant,
                context(&grant),
                &[],
                &excessive,
                &runtime,
                &safety,
            ),
            BoundExecutionDecision::Deny(BoundExecutionDenyReason::RiskChargeExceedsGrant)
        );
    }

    #[test]
    fn execution_proposal_commitment_binds_risk_and_world() {
        let runtime = runtime();
        let safety = safety();
        let a = proposal(&runtime, &safety);
        let mut b = a.clone();
        b.risk_charge = risk(2);
        assert_ne!(a.digest(), b.digest());

        let mut c = a.clone();
        c.world_digest = digest(11);
        assert_ne!(a.digest(), c.digest());
    }
}
