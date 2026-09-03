// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic cyber-physical admission semantics for IoT and actuator control.
//!
//! This crate deliberately sits *between* intelligence and physical effects.
//! It does not perform networking, MQTT/CoAP parsing, cryptographic handshake,
//! hardware attestation, persistence, physical interlocking, or actuator I/O.
//! Higher layers must supply authenticated identity and trustworthy device state;
//! lower layers must durably reserve accepted execution state before dispatch when
//! replay/crash ambiguity matters.
//!
//! The reference invariant is:
//!
//! `authority && exact target && exact operation && exact executor && freshness
//!  && firmware identity && physical safety envelope`
//!
//! Connectivity never creates authority, and model confidence/intelligence is not
//! an input to admission.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_authority::{
    evaluate_authority, AuthorityContext, AuthorityDecision, CapabilityGrant, DenyReason, Digest32,
    NegativeAuthorityFact, Operation, PrincipalId, ResourceRef, TaskId,
};

/// Current schema for [`DeviceCommand`].
pub const DEVICE_COMMAND_SCHEMA_VERSION: u16 = 1;
/// Current schema for [`SafetyEnvelope`].
pub const SAFETY_ENVELOPE_SCHEMA_VERSION: u16 = 1;
/// Domain separator for deterministic command commitments.
pub const DEVICE_COMMAND_DOMAIN: &[u8] = b"symthaea-iot-device-command-v1";
/// Domain separator for deterministic safety-policy commitments.
pub const SAFETY_ENVELOPE_DOMAIN: &[u8] = b"symthaea-iot-safety-envelope-v1";

/// Inclusive fixed-point/integer range used for command parameters and observations.
///
/// Units are defined by the semantic operation/policy rather than this crate. Integer
/// representation avoids floating-point ambiguity in committed safety objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InclusiveRangeI64 {
    /// Lowest accepted value.
    pub min: i64,
    /// Highest accepted value.
    pub max: i64,
}

impl InclusiveRangeI64 {
    /// Returns true when this range is well formed and contains `value`.
    pub fn contains(self, value: i64) -> bool {
        self.min <= self.max && value >= self.min && value <= self.max
    }

    /// Returns true when the range itself is valid.
    pub fn is_valid(self) -> bool {
        self.min <= self.max
    }
}

/// One proposed physical command.
///
/// `actor` is the principal receiving the capability. `executor` is the exact
/// gateway/controller expected to dispatch the effect. Cyber-physical admission
/// requires a capability with an exact audience matching `executor`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCommand {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Caller-stable command identity for audit/correlation.
    pub command_id: String,
    /// Principal exercising the capability.
    pub actor: PrincipalId,
    /// Exact gateway/controller that will execute the effect.
    pub executor: PrincipalId,
    /// Optional task identity. If the grant is task-bound, this must match exactly.
    pub task: Option<TaskId>,
    /// Exact physical resource being affected.
    pub device: ResourceRef,
    /// Exact semantic operation being requested.
    pub operation: Operation,
    /// Firmware identity the command expects the device to be running.
    pub expected_firmware: Digest32,
    /// Per-device monotonic sequence supplied by trusted gateway state.
    pub sequence: u64,
    /// Command issue time in Unix seconds.
    pub issued_at_unix_s: u64,
    /// Mandatory command expiry in Unix seconds.
    pub expires_at_unix_s: u64,
    /// Operation-specific integer/fixed-point parameters.
    pub parameters: BTreeMap<String, i64>,
}

impl DeviceCommand {
    /// Deterministic commitment to every security-relevant command field.
    pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(DEVICE_COMMAND_DOMAIN);
        t.u16(self.schema_version);
        t.string(&self.command_id);
        t.string(&self.actor.0);
        t.string(&self.executor.0);
        t.optional_string(self.task.as_ref().map(|task| task.0.as_str()));
        t.string(&self.device.0);
        t.string(&self.operation.0);
        t.digest(self.expected_firmware);
        t.u64(self.sequence);
        t.u64(self.issued_at_unix_s);
        t.u64(self.expires_at_unix_s);
        t.u32(self.parameters.len() as u32);
        for (name, value) in &self.parameters {
            t.string(name);
            t.i64(*value);
        }
        Digest32(*t.finish().as_bytes())
    }
}

/// Safety policy for one exact device/operation pair.
///
/// A capability says an actor *may* attempt the operation. This envelope says when
/// the operation is physically admissible. The two are intentionally independent.
/// Every entry in `parameter_ranges` is required, and no unrecognized parameter is
/// accepted: v0.1 therefore has an exact command parameter surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// Fail-closed schema version.
    pub schema_version: u16,
    /// Stable policy identity for operator/audit UX.
    pub policy_id: String,
    /// Exact device protected by the envelope.
    pub device: ResourceRef,
    /// Exact operation protected by the envelope.
    pub operation: Operation,
    /// Firmware artifacts under which this safety policy is valid.
    pub allowed_firmware: BTreeSet<Digest32>,
    /// Exact required command-parameter surface and accepted ranges.
    pub parameter_ranges: BTreeMap<String, InclusiveRangeI64>,
    /// Required trusted physical observations and their safe ranges.
    pub required_observations: BTreeMap<String, InclusiveRangeI64>,
}

impl SafetyEnvelope {
    /// Deterministic commitment to the complete safety envelope.
    pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(SAFETY_ENVELOPE_DOMAIN);
        t.u16(self.schema_version);
        t.string(&self.policy_id);
        t.string(&self.device.0);
        t.string(&self.operation.0);
        t.u32(self.allowed_firmware.len() as u32);
        for digest in &self.allowed_firmware {
            t.digest(*digest);
        }
        t.u32(self.parameter_ranges.len() as u32);
        for (name, range) in &self.parameter_ranges {
            t.string(name);
            t.i64(range.min);
            t.i64(range.max);
        }
        t.u32(self.required_observations.len() as u32);
        for (name, range) in &self.required_observations {
            t.string(name);
            t.i64(range.min);
            t.i64(range.max);
        }
        Digest32(*t.finish().as_bytes())
    }

    fn structurally_valid(&self) -> bool {
        !self.policy_id.is_empty()
            && !self.allowed_firmware.is_empty()
            && self.parameter_ranges.values().all(|range| range.is_valid())
            && self
                .required_observations
                .values()
                .all(|range| range.is_valid())
    }
}

/// Trusted runtime facts supplied by the device/gateway boundary.
///
/// This type is evidence input, not attestation by itself. A future Xenia/HAL adapter
/// should construct it only after authenticating the source and validating the
/// hardware/software evidence appropriate to that device class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRuntimeState {
    /// Firmware artifact actually reported/attested by the trusted boundary.
    pub running_firmware: Digest32,
    /// Last command sequence durably accepted for this device, if any.
    pub last_accepted_sequence: Option<u64>,
    /// Trusted physical observations used by the safety envelope.
    pub observations: BTreeMap<String, i64>,
}

/// Stable reason a cyber-physical command failed closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CyberPhysicalDenyReason {
    /// The underlying bounded-authority substrate denied the grant.
    Authority(DenyReason),
    /// Unknown command schema.
    UnsupportedCommandSchema,
    /// Unknown safety-envelope schema.
    UnsupportedSafetySchema,
    /// Malformed command identity or validity interval.
    MalformedCommand,
    /// Malformed/empty safety policy or invalid ranges.
    MalformedSafetyEnvelope,
    /// The command actor is not the capability subject.
    SubjectMismatch,
    /// Physical effects require an exact executor/audience binding.
    MissingExecutorBinding,
    /// The command executor differs from the capability audience.
    ExecutorMismatch,
    /// A task-bound grant was presented for a different/no task.
    TaskMismatch,
    /// The requested device is not in the grant's exact resource set.
    ResourceNotGranted,
    /// The requested operation is not in the grant's exact operation set.
    OperationNotGranted,
    /// The safety envelope protects another device.
    SafetyDeviceMismatch,
    /// The safety envelope protects another operation.
    SafetyOperationMismatch,
    /// Command issue time is later than the trusted authority clock.
    IssuedInFuture,
    /// Command validity has elapsed.
    CommandExpired,
    /// Sequence is not strictly newer than the durable accepted sequence.
    ReplayOrRollback,
    /// Runtime firmware differs from the firmware committed by the command.
    RuntimeFirmwareMismatch,
    /// The expected firmware is outside the safety envelope's allowed set.
    FirmwareNotAllowed,
    /// A command omitted a parameter required by the safety envelope.
    MissingParameter(String),
    /// A command supplied a parameter the safety envelope does not understand.
    UnknownParameter(String),
    /// Command parameter fell outside its allowed range.
    ParameterOutOfRange {
        /// Parameter name.
        name: String,
        /// Proposed value.
        value: i64,
        /// Inclusive minimum.
        min: i64,
        /// Inclusive maximum.
        max: i64,
    },
    /// Required trusted physical state was unavailable.
    MissingSafetyObservation(String),
    /// Trusted physical state was outside the safe operating envelope.
    SafetyObservationOutOfRange {
        /// Observation name.
        name: String,
        /// Observed value.
        value: i64,
        /// Inclusive minimum.
        min: i64,
        /// Inclusive maximum.
        max: i64,
    },
}

/// Result of deterministic cyber-physical admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CyberPhysicalDecision {
    /// Command passed every v0.1 admission condition.
    Allow(CyberPhysicalAdmission),
    /// Command failed closed.
    Deny(CyberPhysicalDenyReason),
}

/// Non-authorizing evidence produced by an allowed decision.
///
/// This receipt is intentionally not a signature and does not prove that an effect
/// occurred. It binds the exact command, grant and safety policy that were admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CyberPhysicalAdmission {
    /// Exact command commitment.
    pub command_digest: Digest32,
    /// Exact capability commitment.
    pub grant_digest: Digest32,
    /// Exact safety-envelope commitment.
    pub safety_envelope_digest: Digest32,
    /// Sequence that durable gateway replay state must advance to before effect.
    pub accepted_sequence: u64,
}

/// Evaluate one proposed physical command without performing I/O or mutating state.
///
/// A caller that will dispatch a real effect must durably reserve/advance execution
/// state before the external effect according to its crash model. This pure function
/// does not claim crash-safe execution ordering and does not replace physical safety
/// interlocks for hazardous equipment.
pub fn evaluate_cyber_physical_command(
    grant: &CapabilityGrant,
    authority_context: AuthorityContext,
    negative_facts: &[NegativeAuthorityFact],
    command: &DeviceCommand,
    runtime: &DeviceRuntimeState,
    safety: &SafetyEnvelope,
) -> CyberPhysicalDecision {
    if command.schema_version != DEVICE_COMMAND_SCHEMA_VERSION {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::UnsupportedCommandSchema);
    }
    if safety.schema_version != SAFETY_ENVELOPE_SCHEMA_VERSION {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::UnsupportedSafetySchema);
    }
    if command.command_id.is_empty()
        || command.expires_at_unix_s < command.issued_at_unix_s
    {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MalformedCommand);
    }
    if !safety.structurally_valid() {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MalformedSafetyEnvelope);
    }

    match evaluate_authority(grant, authority_context, negative_facts) {
        AuthorityDecision::Allow => {}
        AuthorityDecision::Deny(reason) => {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::Authority(reason));
        }
    }

    if command.actor != grant.subject {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::SubjectMismatch);
    }
    match &grant.audience {
        None => {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MissingExecutorBinding);
        }
        Some(audience) if *audience != command.executor => {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ExecutorMismatch);
        }
        Some(_) => {}
    }
    if grant.task.is_some() && grant.task != command.task {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::TaskMismatch);
    }
    if !grant.resources.contains(&command.device) {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ResourceNotGranted);
    }
    if !grant.operations.contains(&command.operation) {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::OperationNotGranted);
    }
    if safety.device != command.device {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::SafetyDeviceMismatch);
    }
    if safety.operation != command.operation {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::SafetyOperationMismatch);
    }
    if command.issued_at_unix_s > authority_context.now_unix_s {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::IssuedInFuture);
    }
    if authority_context.now_unix_s > command.expires_at_unix_s {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::CommandExpired);
    }
    if runtime
        .last_accepted_sequence
        .is_some_and(|last| command.sequence <= last)
    {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ReplayOrRollback);
    }
    if runtime.running_firmware != command.expected_firmware {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::RuntimeFirmwareMismatch);
    }
    if !safety.allowed_firmware.contains(&command.expected_firmware) {
        return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::FirmwareNotAllowed);
    }

    for name in safety.parameter_ranges.keys() {
        if !command.parameters.contains_key(name) {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MissingParameter(
                name.clone(),
            ));
        }
    }
    for (name, value) in &command.parameters {
        let Some(range) = safety.parameter_ranges.get(name) else {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::UnknownParameter(
                name.clone(),
            ));
        };
        if !range.contains(*value) {
            return CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ParameterOutOfRange {
                name: name.clone(),
                value: *value,
                min: range.min,
                max: range.max,
            });
        }
    }

    for (name, range) in &safety.required_observations {
        let Some(value) = runtime.observations.get(name) else {
            return CyberPhysicalDecision::Deny(
                CyberPhysicalDenyReason::MissingSafetyObservation(name.clone()),
            );
        };
        if !range.contains(*value) {
            return CyberPhysicalDecision::Deny(
                CyberPhysicalDenyReason::SafetyObservationOutOfRange {
                    name: name.clone(),
                    value: *value,
                    min: range.min,
                    max: range.max,
                },
            );
        }
    }

    CyberPhysicalDecision::Allow(CyberPhysicalAdmission {
        command_digest: command.digest(),
        grant_digest: grant.digest(),
        safety_envelope_digest: safety.digest(),
        accepted_sequence: command.sequence,
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

    fn u16(&mut self, value: u16) {
        self.0.update(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.update(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.0.update(&value.to_be_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.0.update(value.as_bytes());
    }

    fn optional_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.0.update(&[1]);
                self.string(value);
            }
            None => {
                self.0.update(&[0]);
            }
        }
    }

    fn digest(&mut self, Digest32(value): Digest32) {
        self.0.update(&value);
    }

    fn finish(self) -> blake3::Hash {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_authority::{AuthorityEpoch, GrantUseState, NegativeAuthorityFact};

    fn digest(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn grant() -> CapabilityGrant {
        let mut grant = CapabilityGrant::new(
            "irrigation-valve-72",
            PrincipalId("human:operator".into()),
            PrincipalId("agent:irrigation".into()),
            AuthorityEpoch(4),
        );
        grant.audience = Some(PrincipalId("gateway:field-a".into()));
        grant.task = Some(TaskId("irrigate:zone-7".into()));
        grant.resources = [ResourceRef("iot:valve:72".into())].into_iter().collect();
        grant.operations = [Operation("valve.open".into())].into_iter().collect();
        grant.expires_at_unix_s = Some(10_000);
        grant.max_uses = 4;
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
            command_id: "cmd-0007".into(),
            actor: PrincipalId("agent:irrigation".into()),
            executor: PrincipalId("gateway:field-a".into()),
            task: Some(TaskId("irrigate:zone-7".into())),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            expected_firmware: digest(7),
            sequence: 42,
            issued_at_unix_s: 4_990,
            expires_at_unix_s: 5_030,
            parameters: BTreeMap::from([("duration_ms".into(), 600_000)]),
        }
    }

    fn safety() -> SafetyEnvelope {
        SafetyEnvelope {
            schema_version: SAFETY_ENVELOPE_SCHEMA_VERSION,
            policy_id: "valve-open-safe-v1".into(),
            device: ResourceRef("iot:valve:72".into()),
            operation: Operation("valve.open".into()),
            allowed_firmware: [digest(7)].into_iter().collect(),
            parameter_ranges: BTreeMap::from([(
                "duration_ms".into(),
                InclusiveRangeI64 {
                    min: 1_000,
                    max: 720_000,
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
            last_accepted_sequence: Some(41),
            observations: BTreeMap::from([("tank_pressure_kpa_x100".into(), 210_000)]),
        }
    }

    #[test]
    fn exact_bounded_command_is_allowed() {
        let grant = grant();
        let command = command();
        let safety = safety();
        let decision = evaluate_cyber_physical_command(
            &grant,
            context(&grant),
            &[],
            &command,
            &runtime(),
            &safety,
        );
        let CyberPhysicalDecision::Allow(admission) = decision else {
            panic!("expected command to be admitted");
        };
        assert_eq!(admission.command_digest, command.digest());
        assert_eq!(admission.grant_digest, grant.digest());
        assert_eq!(admission.safety_envelope_digest, safety.digest());
        assert_eq!(admission.accepted_sequence, 42);
    }

    #[test]
    fn connectivity_without_operation_authority_fails_closed() {
        let mut grant = grant();
        grant.operations.clear();
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command(),
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::OperationNotGranted)
        );
    }

    #[test]
    fn physical_effect_requires_exact_executor_binding() {
        let mut grant = grant();
        grant.audience = None;
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command(),
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MissingExecutorBinding)
        );
    }

    #[test]
    fn replay_or_rollback_is_rejected() {
        let grant = grant();
        let mut state = runtime();
        state.last_accepted_sequence = Some(42);
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command(),
                &state,
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ReplayOrRollback)
        );
    }

    #[test]
    fn firmware_identity_is_part_of_physical_authority() {
        let grant = grant();
        let mut state = runtime();
        state.running_firmware = digest(8);
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command(),
                &state,
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::RuntimeFirmwareMismatch)
        );
    }

    #[test]
    fn required_parameter_cannot_be_omitted() {
        let grant = grant();
        let mut command = command();
        command.parameters.clear();
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command,
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MissingParameter(
                "duration_ms".into()
            ))
        );
    }

    #[test]
    fn unsafe_parameter_is_rejected_even_with_valid_authority() {
        let grant = grant();
        let mut command = command();
        command.parameters.insert("duration_ms".into(), 900_000);
        assert!(matches!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command,
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::ParameterOutOfRange { .. })
        ));
    }

    #[test]
    fn unknown_parameter_fails_closed() {
        let grant = grant();
        let mut command = command();
        command.parameters.insert("override_interlock".into(), 1);
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command,
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::UnknownParameter(
                "override_interlock".into()
            ))
        );
    }

    #[test]
    fn missing_physical_observation_fails_closed() {
        let grant = grant();
        let mut state = runtime();
        state.observations.clear();
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &command(),
                &state,
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::MissingSafetyObservation(
                "tank_pressure_kpa_x100".into()
            ))
        );
    }

    #[test]
    fn revoked_grant_dominates_otherwise_valid_command() {
        let grant = grant();
        let facts = [NegativeAuthorityFact::RevokeGrant {
            grant_digest: grant.digest(),
        }];
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &facts,
                &command(),
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::Authority(
                DenyReason::ExplicitlyRevoked
            ))
        );
    }

    #[test]
    fn future_and_expired_commands_are_rejected() {
        let grant = grant();
        let mut future = command();
        future.issued_at_unix_s = 5_001;
        future.expires_at_unix_s = 5_020;
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &future,
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::IssuedInFuture)
        );

        let mut expired = command();
        expired.issued_at_unix_s = 4_900;
        expired.expires_at_unix_s = 4_999;
        assert_eq!(
            evaluate_cyber_physical_command(
                &grant,
                context(&grant),
                &[],
                &expired,
                &runtime(),
                &safety(),
            ),
            CyberPhysicalDecision::Deny(CyberPhysicalDenyReason::CommandExpired)
        );
    }

    #[test]
    fn command_and_safety_commitments_are_context_sensitive() {
        let mut a = command();
        let mut b = a.clone();
        b.sequence += 1;
        assert_ne!(a.digest(), b.digest());

        let safety_a = safety();
        let mut safety_b = safety_a.clone();
        safety_b
            .parameter_ranges
            .get_mut("duration_ms")
            .unwrap()
            .max -= 1;
        assert_ne!(safety_a.digest(), safety_b.digest());

        a.executor = PrincipalId("gateway:other".into());
        assert_ne!(a.digest(), command().digest());
    }
}
