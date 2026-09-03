// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed authority boundary for maintenance service of isolated actuators.
//!
//! A portable service authorization is not itself execution authority. The
//! authorization must first pass service-specific role/authentication checks and
//! then be accepted by the existing replay-resistant operator authority as a
//! fresh `EnterMaintenance` command.
//!
//! Successful authorization returns an affine permit that holds the exact
//! embodiment under an exclusive mutable borrow. Safe Rust therefore cannot move
//! the permit to another embodiment, mutate operator authority concurrently, or
//! duplicate the service capability. Dropping the permit abandons service while
//! leaving the maintenance restriction and actuator isolation intact.
//!
//! Service never restores productive actuator authority directly. Consuming the
//! permit begins service/requalification while the actuator remains isolated.

use crate::actuator_isolation::PhysicalActuator;
use crate::audit_chain::{AuditDigestProvider, AuditEvent, AuditLedger};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{OperatorAuthorityRejection, OperatorConstraint};
use crate::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use serde::{Deserialize, Serialize};

pub const ACTUATOR_SERVICE_AUTHORIZATION_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuatorServiceAuthorizationV1 {
    schema_version: u16,
    operator: OperatorId,
    role: OperatorRole,
    authentication: AuthenticationLevel,
    epoch: u64,
    sequence: u64,
    service_proposal_id: u64,
    actuator: PhysicalActuator,
    issued_step: u64,
    expires_step: u64,
}

impl ActuatorServiceAuthorizationV1 {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        operator: OperatorId,
        role: OperatorRole,
        authentication: AuthenticationLevel,
        epoch: u64,
        sequence: u64,
        service_proposal_id: u64,
        actuator: PhysicalActuator,
        issued_step: u64,
        expires_step: u64,
    ) -> Self {
        Self {
            schema_version: ACTUATOR_SERVICE_AUTHORIZATION_SCHEMA_VERSION,
            operator,
            role,
            authentication,
            epoch,
            sequence,
            service_proposal_id,
            actuator,
            issued_step,
            expires_step,
        }
    }

    pub const fn operator(self) -> OperatorId {
        self.operator
    }

    pub const fn role(self) -> OperatorRole {
        self.role
    }

    pub const fn authentication(self) -> AuthenticationLevel {
        self.authentication
    }

    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    pub const fn service_proposal_id(self) -> u64 {
        self.service_proposal_id
    }

    pub const fn actuator(self) -> PhysicalActuator {
        self.actuator
    }

    pub const fn issued_step(self) -> u64 {
        self.issued_step
    }

    pub const fn expires_step(self) -> u64 {
        self.expires_step
    }

    fn validate_service_requirements(self) -> Result<Self, ActuatorServiceRejection> {
        if self.schema_version != ACTUATOR_SERVICE_AUTHORIZATION_SCHEMA_VERSION {
            return Err(ActuatorServiceRejection::InvalidSchema);
        }
        if !self.operator.is_valid() {
            return Err(ActuatorServiceRejection::InvalidOperator);
        }
        if !matches!(self.role, OperatorRole::Supervisor | OperatorRole::SafetyOfficer) {
            return Err(ActuatorServiceRejection::InsufficientRole);
        }
        if self.authentication < AuthenticationLevel::HardwareBacked {
            return Err(ActuatorServiceRejection::InsufficientAuthentication);
        }
        if self.epoch == 0 {
            return Err(ActuatorServiceRejection::InvalidEpoch);
        }
        if self.sequence == 0 {
            return Err(ActuatorServiceRejection::InvalidSequence);
        }
        if self.service_proposal_id == 0 {
            return Err(ActuatorServiceRejection::InvalidProposal);
        }
        Ok(self)
    }

    /// Reuse the established operator replay/freshness boundary instead of
    /// introducing a second command-sequence ledger for maintenance service.
    fn as_maintenance_command_envelope(self) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: self.operator,
            role: self.role,
            authentication: self.authentication,
            epoch: self.epoch,
            sequence: self.sequence,
            proposal_id: self.service_proposal_id,
            issued_step: self.issued_step,
            expires_step: self.expires_step,
            command: OperatorCommand::EnterMaintenance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorServiceRejection {
    InvalidSchema,
    InvalidOperator,
    InsufficientRole,
    InsufficientAuthentication,
    InvalidEpoch,
    InvalidSequence,
    InvalidProposal,
    ActuatorNotIsolated,
    Operator(OperatorAuthorityRejection),
    PermitExpired,
    MaintenanceContextChanged,
}

impl From<OperatorAuthorityRejection> for ActuatorServiceRejection {
    fn from(value: OperatorAuthorityRejection) -> Self {
        Self::Operator(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorServiceState {
    RequalificationRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActuatorServiceTransition {
    pub actuator: PhysicalActuator,
    pub service_proposal_id: u64,
    pub state: ActuatorServiceState,
}

/// One-shot authority to begin service on one isolated actuator.
///
/// The permit is intentionally neither cloneable nor serializable. More
/// importantly, it owns an exclusive mutable borrow of the exact embodiment
/// that admitted its maintenance authorization. This makes owner binding a Rust
/// type-system property rather than a runtime equality guess.
pub struct ActuatorServicePermit<'a> {
    owner: &'a mut SubterraneanEmbodiment,
    actuator: PhysicalActuator,
    service_proposal_id: u64,
    authority_generation: u64,
    issued_step: u64,
    expires_step: u64,
}

impl<'a> ActuatorServicePermit<'a> {
    fn issue(
        owner: &'a mut SubterraneanEmbodiment,
        authorization: ActuatorServiceAuthorizationV1,
        authority_generation: u64,
    ) -> Self {
        Self {
            owner,
            actuator: authorization.actuator,
            service_proposal_id: authorization.service_proposal_id,
            authority_generation,
            issued_step: authorization.issued_step,
            expires_step: authorization.expires_step,
        }
    }

    pub const fn actuator(&self) -> PhysicalActuator {
        self.actuator
    }

    pub const fn service_proposal_id(&self) -> u64 {
        self.service_proposal_id
    }

    pub const fn issued_step(&self) -> u64 {
        self.issued_step
    }

    pub const fn expires_step(&self) -> u64 {
        self.expires_step
    }

    /// Consume this exact-owner permit and enter requalification while leaving
    /// productive actuator authority latched closed.
    pub fn begin(self) -> Result<ActuatorServiceTransition, ActuatorServiceRejection> {
        let Self {
            owner,
            actuator,
            service_proposal_id,
            authority_generation,
            expires_step,
            ..
        } = self;

        let now_step = owner.total_steps() as u64;
        if now_step > expires_step {
            return Err(ActuatorServiceRejection::PermitExpired);
        }
        if owner.operator_constraint() != OperatorConstraint::MaintenanceLock
            || owner.operator_authority().last_applied_proposal() != Some(service_proposal_id)
            || owner.operator_authority().accepted_commands() != authority_generation
        {
            return Err(ActuatorServiceRejection::MaintenanceContextChanged);
        }
        if !owner.actuator_isolation_report().is_isolated(actuator) {
            return Err(ActuatorServiceRejection::ActuatorNotIsolated);
        }

        owner.service_isolated_actuator(actuator);
        debug_assert!(owner.actuator_isolation_report().is_isolated(actuator));

        Ok(ActuatorServiceTransition {
            actuator,
            service_proposal_id,
            state: ActuatorServiceState::RequalificationRequired,
        })
    }

    pub fn begin_with_audit(
        self,
        provider: &impl AuditDigestProvider,
        ledger: &mut AuditLedger,
    ) -> Result<ActuatorServiceTransition, ActuatorServiceRejection> {
        let service_proposal_id = self.service_proposal_id;
        let actuator_code = self.actuator.index() as u16;
        let result = self.begin();
        ledger.append(
            provider,
            AuditEvent::ActuatorServiceTransition {
                service_proposal_id,
                actuator_code,
                state_code: if result.is_ok() { 1 } else { 0 },
            },
        );
        result
    }
}

impl SubterraneanEmbodiment {
    /// Validate one exact service authorization and mint one exact-owner affine
    /// permit.
    ///
    /// The service request is itself admitted as a fresh hardware-backed
    /// `EnterMaintenance` command. This makes the existing operator replay
    /// sequence the canonical anti-replay boundary. While the returned permit
    /// exists, its exclusive mutable borrow prevents any other safe-Rust owner
    /// transition until the permit is consumed or dropped.
    pub fn authorize_actuator_service(
        &mut self,
        authorization: ActuatorServiceAuthorizationV1,
    ) -> Result<ActuatorServicePermit<'_>, ActuatorServiceRejection> {
        let authorization = authorization.validate_service_requirements()?;
        if !self
            .actuator_isolation_report()
            .is_isolated(authorization.actuator())
        {
            return Err(ActuatorServiceRejection::ActuatorNotIsolated);
        }

        self.ingest_operator_command(authorization.as_maintenance_command_envelope())?;
        if self.operator_constraint() != OperatorConstraint::MaintenanceLock
            || self.operator_authority().last_applied_proposal()
                != Some(authorization.service_proposal_id())
        {
            return Err(ActuatorServiceRejection::MaintenanceContextChanged);
        }
        let authority_generation = self.operator_authority().accepted_commands();

        Ok(ActuatorServicePermit::issue(
            self,
            authorization,
            authority_generation,
        ))
    }

    pub fn authorize_actuator_service_with_audit(
        &mut self,
        provider: &impl AuditDigestProvider,
        ledger: &mut AuditLedger,
        authorization: ActuatorServiceAuthorizationV1,
    ) -> Result<ActuatorServicePermit<'_>, ActuatorServiceRejection> {
        let operator_id = authorization.operator().0;
        let service_proposal_id = authorization.service_proposal_id();
        let actuator_code = authorization.actuator().index() as u16;
        match self.authorize_actuator_service(authorization) {
            Ok(permit) => {
                ledger.append(
                    provider,
                    AuditEvent::ActuatorServiceAuthorization {
                        operator_id,
                        service_proposal_id,
                        actuator_code,
                        accepted: true,
                    },
                );
                Ok(permit)
            }
            Err(error) => {
                ledger.append(
                    provider,
                    AuditEvent::ActuatorServiceAuthorization {
                        operator_id,
                        service_proposal_id,
                        actuator_code,
                        accepted: false,
                    },
                );
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_isolation::{ActuatorIsolationPolicy, ActuatorIsolationSupervisor};
    use crate::audit_chain::DeterministicAuditDigest;
    use crate::operator_authority::OperatorAuthorityRejection;
    use crate::types::{SubterraneanCommand, SubterraneanState};
    use symthaea_core::genesis::GenesisSeed;

    fn authorization(
        actuator: PhysicalActuator,
        sequence: u64,
        proposal_id: u64,
    ) -> ActuatorServiceAuthorizationV1 {
        ActuatorServiceAuthorizationV1::new(
            OperatorId(7),
            OperatorRole::SafetyOfficer,
            AuthenticationLevel::HardwareBacked,
            1,
            sequence,
            proposal_id,
            actuator,
            0,
            100,
        )
    }

    fn isolated_embodiment(actuators: &[PhysicalActuator]) -> SubterraneanEmbodiment {
        let genesis = GenesisSeed::from_phrase("actuator-service");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        let mut checkpoint = embodiment.operational_checkpoint();
        checkpoint.actuator_isolation = ActuatorIsolationSupervisor::new(ActuatorIsolationPolicy {
            mismatch_penalty: 1.0,
            isolation_threshold: 1.0,
            mismatch_streak_limit: 1,
            ..Default::default()
        });
        let state = SubterraneanState::home();
        for actuator in actuators {
            let mut command = SubterraneanCommand::zero();
            match actuator {
                PhysicalActuator::Cutter => command.set_cutter_head(1.0),
                PhysicalActuator::LeftTrack => command.set_left_track(1.0),
                other => panic!("test isolation fixture not defined for {other:?}"),
            }
            let report = checkpoint
                .actuator_isolation
                .observe(&command, &state, &state);
            assert!(report.is_isolated(*actuator));
        }
        embodiment
            .load_operational_checkpoint(&checkpoint)
            .expect("isolated checkpoint fixture");
        embodiment
    }

    #[test]
    fn service_authorization_requires_hardware_backed_authentication() {
        let mut value = authorization(PhysicalActuator::Cutter, 1, 50);
        value.authentication = AuthenticationLevel::TransportAuthenticated;
        assert_eq!(
            value.validate_service_requirements(),
            Err(ActuatorServiceRejection::InsufficientAuthentication)
        );
    }

    #[test]
    fn service_authorization_requires_supervisory_role() {
        let mut value = authorization(PhysicalActuator::Cutter, 1, 50);
        value.role = OperatorRole::Operator;
        assert_eq!(
            value.validate_service_requirements(),
            Err(ActuatorServiceRejection::InsufficientRole)
        );
    }

    #[test]
    fn nominal_actuator_cannot_receive_service_widening_authority() {
        let mut embodiment = isolated_embodiment(&[]);
        assert_eq!(
            embodiment
                .authorize_actuator_service(authorization(PhysicalActuator::Cutter, 1, 50))
                .err(),
            Some(ActuatorServiceRejection::ActuatorNotIsolated)
        );
        assert_eq!(embodiment.operator_constraint(), OperatorConstraint::None);
    }

    #[test]
    fn service_authorization_is_replay_resistant_through_operator_ledger() {
        let mut embodiment = isolated_embodiment(&[PhysicalActuator::Cutter]);
        let auth = authorization(PhysicalActuator::Cutter, 1, 50);
        let permit = embodiment
            .authorize_actuator_service(auth)
            .expect("first service authorization");
        drop(permit);
        assert_eq!(
            embodiment.authorize_actuator_service(auth).err(),
            Some(ActuatorServiceRejection::Operator(
                OperatorAuthorityRejection::Replay
            ))
        );
    }

    #[test]
    fn service_permit_is_exactly_actuator_bound_and_keeps_latch_closed() {
        let mut embodiment = isolated_embodiment(&[
            PhysicalActuator::Cutter,
            PhysicalActuator::LeftTrack,
        ]);
        let permit = embodiment
            .authorize_actuator_service(authorization(PhysicalActuator::Cutter, 1, 50))
            .expect("service permit");
        assert_eq!(permit.actuator(), PhysicalActuator::Cutter);

        let transition = permit.begin().expect("begin service");
        assert_eq!(transition.actuator, PhysicalActuator::Cutter);
        assert_eq!(transition.state, ActuatorServiceState::RequalificationRequired);
        assert!(
            embodiment
                .actuator_isolation_report()
                .is_isolated(PhysicalActuator::Cutter)
        );
        assert!(
            embodiment
                .actuator_isolation_report()
                .is_isolated(PhysicalActuator::LeftTrack)
        );
    }

    #[test]
    fn dropping_service_permit_never_services_or_reopens_actuator() {
        let mut embodiment = isolated_embodiment(&[PhysicalActuator::Cutter]);
        let before = embodiment.actuator_isolation_report();
        let permit = embodiment
            .authorize_actuator_service(authorization(PhysicalActuator::Cutter, 1, 50))
            .expect("service permit");
        drop(permit);
        let after = embodiment.actuator_isolation_report();
        assert!(after.is_isolated(PhysicalActuator::Cutter));
        assert_eq!(after.health, before.health);
        assert_eq!(after.mismatch_streaks, before.mismatch_streaks);
        assert_eq!(embodiment.operator_constraint(), OperatorConstraint::MaintenanceLock);
    }

    #[test]
    fn service_audit_distinguishes_authorization_from_service_transition() {
        let mut embodiment = isolated_embodiment(&[PhysicalActuator::Cutter]);
        let provider = DeterministicAuditDigest;
        let mut ledger = AuditLedger::new(8, crate::update_control::ArtifactDigest([1; 32]));
        let permit = embodiment
            .authorize_actuator_service_with_audit(
                &provider,
                &mut ledger,
                authorization(PhysicalActuator::Cutter, 1, 50),
            )
            .expect("service authorization");
        permit
            .begin_with_audit(&provider, &mut ledger)
            .expect("service transition");
        let records = ledger.records();
        assert!(matches!(
            records[0].event,
            AuditEvent::ActuatorServiceAuthorization { accepted: true, .. }
        ));
        assert!(matches!(
            records[1].event,
            AuditEvent::ActuatorServiceTransition { state_code: 1, .. }
        ));
    }
}
