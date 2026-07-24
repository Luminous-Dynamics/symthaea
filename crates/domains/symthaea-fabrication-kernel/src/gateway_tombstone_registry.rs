// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only registry of terminal gateway tombstones.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_tombstone::AuthorizedGatewayTombstone;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GATEWAY_TOMBSTONE_REGISTRY_SCHEMA: &str =
    "symthaea.fabrication.gateway-tombstone-registry.v1";
pub const MAX_GATEWAY_TOMBSTONES: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayTombstoneRecord {
    pub gateway_id: String,
    pub tombstone_sequence: u64,
    pub tombstone_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
    pub successor_membership_digest: Sha256Digest,
    pub issued_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayTombstoneRegistry {
    pub schema_version: String,
    records: BTreeMap<String, GatewayTombstoneRecord>,
}

impl Default for GatewayTombstoneRegistry {
    fn default() -> Self {
        Self {
            schema_version: GATEWAY_TOMBSTONE_REGISTRY_SCHEMA.into(),
            records: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayTombstoneRegistryError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidRecord,
    GatewayAlreadyTombstoned,
    SequenceRollback,
    SequenceCollision,
    MissingPriorTombstone(String),
    TombstoneChanged(String),
    Encoding(String),
}

impl GatewayTombstoneRegistry {
    pub fn records(&self) -> &BTreeMap<String, GatewayTombstoneRecord> {
        &self.records
    }

    pub fn validate(&self) -> Result<(), GatewayTombstoneRegistryError> {
        if self.schema_version != GATEWAY_TOMBSTONE_REGISTRY_SCHEMA {
            return Err(GatewayTombstoneRegistryError::UnsupportedSchema);
        }
        if self.records.len() > MAX_GATEWAY_TOMBSTONES {
            return Err(GatewayTombstoneRegistryError::CapacityExceeded);
        }
        let mut sequences = BTreeMap::new();
        for (gateway_id, record) in &self.records {
            if gateway_id != &record.gateway_id
                || gateway_id.trim().is_empty()
                || gateway_id != gateway_id.trim()
                || record.tombstone_sequence == 0
                || record.issued_at_unix_s == 0
            {
                return Err(GatewayTombstoneRegistryError::InvalidRecord);
            }
            if let Some(existing_gateway) = sequences.insert(record.tombstone_sequence, gateway_id)
            {
                if existing_gateway != gateway_id {
                    return Err(GatewayTombstoneRegistryError::SequenceCollision);
                }
            }
        }
        Ok(())
    }

    pub fn insert(
        &mut self,
        tombstone: &AuthorizedGatewayTombstone,
    ) -> Result<Sha256Digest, GatewayTombstoneRegistryError> {
        self.validate()?;
        let body = tombstone.tombstone();
        if let Some(existing) = self.records.get(&body.gateway_id) {
            if existing.tombstone_digest == tombstone.tombstone_digest() {
                return Ok(existing.tombstone_digest);
            }
            return Err(GatewayTombstoneRegistryError::GatewayAlreadyTombstoned);
        }
        if self.records.len() >= MAX_GATEWAY_TOMBSTONES {
            return Err(GatewayTombstoneRegistryError::CapacityExceeded);
        }
        if let Some(latest) = self
            .records
            .values()
            .map(|record| record.tombstone_sequence)
            .max()
        {
            if body.tombstone_sequence < latest {
                return Err(GatewayTombstoneRegistryError::SequenceRollback);
            }
            if body.tombstone_sequence == latest {
                return Err(GatewayTombstoneRegistryError::SequenceCollision);
            }
        }
        self.records.insert(
            body.gateway_id.clone(),
            GatewayTombstoneRecord {
                gateway_id: body.gateway_id.clone(),
                tombstone_sequence: body.tombstone_sequence,
                tombstone_digest: tombstone.tombstone_digest(),
                ceremony_digest: tombstone.ceremony_digest(),
                successor_membership_digest: body.successor_membership_digest,
                issued_at_unix_s: body.issued_at_unix_s,
            },
        );
        Ok(tombstone.tombstone_digest())
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), GatewayTombstoneRegistryError> {
        self.validate()?;
        previous.validate()?;
        for (gateway_id, old) in &previous.records {
            let Some(new) = self.records.get(gateway_id) else {
                return Err(GatewayTombstoneRegistryError::MissingPriorTombstone(
                    gateway_id.clone(),
                ));
            };
            if new != old {
                return Err(GatewayTombstoneRegistryError::TombstoneChanged(
                    gateway_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn permits_authority(&self, gateway_id: &str) -> bool {
        !self.records.contains_key(gateway_id)
    }
}

pub fn digest_gateway_tombstone_registry(
    registry: &GatewayTombstoneRegistry,
) -> Result<Sha256Digest, GatewayTombstoneRegistryError> {
    registry.validate()?;
    let bytes = serde_json::to_vec(registry)
        .map_err(|error| GatewayTombstoneRegistryError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-tombstone-registry-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successor_cannot_remove_tombstone() {
        let previous = GatewayTombstoneRegistry {
            schema_version: GATEWAY_TOMBSTONE_REGISTRY_SCHEMA.into(),
            records: BTreeMap::from([(
                "gateway-a".into(),
                GatewayTombstoneRecord {
                    gateway_id: "gateway-a".into(),
                    tombstone_sequence: 1,
                    tombstone_digest: Sha256Digest([1; 32]),
                    ceremony_digest: Sha256Digest([2; 32]),
                    successor_membership_digest: Sha256Digest([3; 32]),
                    issued_at_unix_s: 10,
                },
            )]),
        };
        let current = GatewayTombstoneRegistry::default();
        assert!(matches!(
            current.verify_successor_of(&previous),
            Err(GatewayTombstoneRegistryError::MissingPriorTombstone(_))
        ));
    }
}
