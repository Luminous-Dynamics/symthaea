// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit world-presence sessions and declared capability surfaces.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{digest::TypedDigest, types::WorldDescriptor};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PresenceCapability {
    Observe,
    Enter,
    Fork,
    Propose,
    Mutate,
    Persist,
    SpawnAgent,
    ChangePhysics,
    Delete,
    Custom(String),
}

impl PresenceCapability {
    fn requires_external_authority(&self) -> bool {
        matches!(
            self,
            Self::Mutate | Self::Persist | Self::SpawnAgent | Self::ChangePhysics | Self::Delete
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldPresenceSession {
    pub session_id: String,
    pub agent_id: String,
    pub world: WorldDescriptor,
    pub embodiment_id: String,
    pub sensor_suite_digest: TypedDigest,
    pub action_surface_digest: TypedDigest,
    pub capabilities: Vec<PresenceCapability>,
    /// Required when the declared capability surface contains authority-bearing
    /// actions. The ledger never mints this authority itself.
    pub authority_receipt_digest: Option<TypedDigest>,
    pub entry_state_digest: TypedDigest,
    pub exit_state_digest: Option<TypedDigest>,
    pub entered_frame: Option<u64>,
    pub exited_frame: Option<u64>,
}

impl WorldPresenceSession {
    pub fn validate(&self) -> Result<(), PresenceError> {
        for value in [
            self.session_id.as_str(),
            self.agent_id.as_str(),
            self.embodiment_id.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(PresenceError::MissingIdentity);
            }
        }
        self.world
            .validate()
            .map_err(|error| PresenceError::InvalidWorld(error.to_string()))?;
        for digest in [
            &self.sensor_suite_digest,
            &self.action_surface_digest,
            &self.entry_state_digest,
        ] {
            digest
                .validate()
                .map_err(|error| PresenceError::InvalidDigest(error.to_string()))?;
        }
        if let Some(digest) = &self.exit_state_digest {
            digest
                .validate()
                .map_err(|error| PresenceError::InvalidDigest(error.to_string()))?;
        }
        if let Some(digest) = &self.authority_receipt_digest {
            digest
                .validate()
                .map_err(|error| PresenceError::InvalidDigest(error.to_string()))?;
        }

        let mut unique = BTreeSet::new();
        for capability in &self.capabilities {
            if let PresenceCapability::Custom(name) = capability {
                if name.trim().is_empty() {
                    return Err(PresenceError::EmptyCustomCapability);
                }
            }
            if !unique.insert(capability.clone()) {
                return Err(PresenceError::DuplicateCapability);
            }
        }
        if self
            .capabilities
            .iter()
            .any(PresenceCapability::requires_external_authority)
            && self.authority_receipt_digest.is_none()
        {
            return Err(PresenceError::MissingAuthorityReceipt);
        }

        match (self.exit_state_digest.is_some(), self.exited_frame.is_some()) {
            (true, true) | (false, false) => {}
            _ => return Err(PresenceError::PartialExit),
        }
        if let (Some(entered), Some(exited)) = (self.entered_frame, self.exited_frame) {
            if exited < entered {
                return Err(PresenceError::ExitBeforeEntry);
            }
        }
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.exit_state_digest.is_none() && self.exited_frame.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PresenceError {
    #[error("presence identities may not be empty")]
    MissingIdentity,
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("custom capability name may not be empty")]
    EmptyCustomCapability,
    #[error("presence capability list may not contain duplicates")]
    DuplicateCapability,
    #[error("authority-bearing capabilities require an external authority receipt")]
    MissingAuthorityReceipt,
    #[error("exit state and exit frame must be recorded together")]
    PartialExit,
    #[error("presence exit may not precede entry")]
    ExitBeforeEntry,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest::TypedDigest, types::{RealityLayer, WorldId, WorldLineageId, WorldOrigin}};

    fn d(domain: &str) -> TypedDigest {
        TypedDigest::blake3(domain, domain.as_bytes()).unwrap()
    }

    fn world() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("studio".into()),
            lineage_id: WorldLineageId("studio-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "symtropy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "host".into(),
        }
    }

    #[test]
    fn mutation_capability_never_self_authorizes() {
        let session = WorldPresenceSession {
            session_id: "s".into(),
            agent_id: "symthaea".into(),
            world: world(),
            embodiment_id: "camera-body".into(),
            sensor_suite_digest: d("sensors.v1"),
            action_surface_digest: d("actions.v1"),
            capabilities: vec![PresenceCapability::Observe, PresenceCapability::Mutate],
            authority_receipt_digest: None,
            entry_state_digest: d("state.v1"),
            exit_state_digest: None,
            entered_frame: Some(1),
            exited_frame: None,
        };
        assert_eq!(session.validate(), Err(PresenceError::MissingAuthorityReceipt));
    }
}
