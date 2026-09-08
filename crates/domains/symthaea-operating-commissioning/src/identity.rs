// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical identity for one complete commissioning record.
//!
//! Commissioning authority must bind more than a configuration digest and an
//! evidence label: the locally commissioned survival envelope is itself
//! authority-bearing. This module defines explicit, cross-tool-friendly bytes for
//! the complete record without relying on Rust enum discriminants, `Debug`, or
//! default Serde encoding.

use crate::{CommissioningError, CommissioningRecord, ConfigurationDigest};
use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
use symthaea_resource_hierarchy::ResourceHierarchy;
use symthaea_resource_model::{ResourceKind, ResourceUnit};
use thiserror::Error;

pub const COMMISSIONING_RECORD_IDENTITY_SCHEMA_V1: &str =
    "symthaea-commissioning-record-identity-v1";
const DOMAIN_SEPARATOR: &[u8] = b"symthaea:commissioning-record:v1\0";

/// Exact digest of one canonical commissioning record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommissioningRecordDigest {
    Blake3_256([u8; 32]),
}

impl CommissioningRecordDigest {
    pub fn blake3_256(bytes: &[u8]) -> Self {
        Self::Blake3_256(ConfigurationDigest::blake3_256(bytes).into_blake3_256())
    }

    pub fn into_blake3_256(self) -> [u8; 32] {
        match self {
            Self::Blake3_256(bytes) => bytes,
        }
    }
}

/// Produce fixed-order bytes for the entire authority-bearing commissioning record.
///
/// V1 deliberately canonicalizes two representations that are operationally
/// equivalent:
/// - resource-constraint vector order is ignored by sorting explicit normalized
///   constraint tuples;
/// - IEEE-754 `-0.0` is normalized to `+0.0` because the safety validators treat
///   them as equal numeric values.
///
/// Everything else remains exact: generation, configuration, evidence ID, local
/// envelope ID/node, allowed modes, every normalized constraint, and both local
/// survival fractions are bound.
pub fn canonical_commissioning_record_bytes(
    record: &CommissioningRecord,
    hierarchy: &ResourceHierarchy,
) -> Result<Vec<u8>, CommissioningIdentityError> {
    record.commissioned().validate(hierarchy)?;

    let commissioned = record.commissioned();
    let local = commissioned.local();
    let binding = commissioned.binding();

    let mut out = Vec::with_capacity(384);
    out.extend_from_slice(DOMAIN_SEPARATOR);
    push_string(
        &mut out,
        "schema_version",
        COMMISSIONING_RECORD_IDENTITY_SCHEMA_V1,
    )?;
    out.extend_from_slice(&record.generation().to_be_bytes());
    push_string(&mut out, "subject_node_id", record.subject_node_id())?;
    push_configuration_digest(&mut out, binding.configuration_digest());
    push_string(&mut out, "commissioning_evidence_id", binding.evidence_id())?;

    push_string(&mut out, "local_envelope_id", &local.id)?;
    push_string(&mut out, "local_subject_node_id", &local.subject_node_id)?;

    let mut mode_tags: Vec<u8> = local.allowed_modes.iter().map(|mode| mode_tag(*mode)).collect();
    mode_tags.sort_unstable();
    push_count(&mut out, "allowed_modes", mode_tags.len())?;
    out.extend_from_slice(&mode_tags);

    let mut constraints: Vec<CanonicalConstraint> = local
        .resource_constraints
        .iter()
        .map(CanonicalConstraint::from_constraint)
        .collect();
    constraints.sort_unstable();
    push_count(&mut out, "resource_constraints", constraints.len())?;
    for constraint in constraints {
        constraint.encode_into(&mut out);
    }

    out.extend_from_slice(&canonical_f64_bits(local.critical_service_floor).to_be_bytes());
    out.extend_from_slice(&canonical_f64_bits(local.max_shed_fraction).to_be_bytes());
    Ok(out)
}

pub fn commissioning_record_digest(
    record: &CommissioningRecord,
    hierarchy: &ResourceHierarchy,
) -> Result<CommissioningRecordDigest, CommissioningIdentityError> {
    Ok(CommissioningRecordDigest::blake3_256(
        &canonical_commissioning_record_bytes(record, hierarchy)?,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalConstraint {
    kind: u8,
    unit: u8,
    metric: u8,
    minimum_bits: Option<u64>,
    maximum_bits: Option<u64>,
}

impl CanonicalConstraint {
    fn from_constraint(constraint: &ResourceConstraint) -> Self {
        Self {
            kind: resource_kind_tag(constraint.key.kind),
            unit: resource_unit_tag(constraint.key.unit),
            metric: boundary_metric_tag(constraint.metric),
            minimum_bits: constraint.minimum.map(canonical_f64_bits),
            maximum_bits: constraint.maximum.map(canonical_f64_bits),
        }
    }

    fn encode_into(self, out: &mut Vec<u8>) {
        out.push(self.kind);
        out.push(self.unit);
        out.push(self.metric);
        push_optional_f64_bits(out, self.minimum_bits);
        push_optional_f64_bits(out, self.maximum_bits);
    }
}

fn push_optional_f64_bits(out: &mut Vec<u8>, bits: Option<u64>) {
    match bits {
        None => out.push(0),
        Some(bits) => {
            out.push(1);
            out.extend_from_slice(&bits.to_be_bytes());
        }
    }
}

fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn push_configuration_digest(out: &mut Vec<u8>, digest: ConfigurationDigest) {
    match digest {
        ConfigurationDigest::Blake3_256(bytes) => {
            out.push(1);
            out.extend_from_slice(&bytes);
        }
    }
}

fn push_count(
    out: &mut Vec<u8>,
    field: &'static str,
    count: usize,
) -> Result<(), CommissioningIdentityError> {
    let count =
        u32::try_from(count).map_err(|_| CommissioningIdentityError::CollectionTooLarge(field))?;
    out.extend_from_slice(&count.to_be_bytes());
    Ok(())
}

fn push_string(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
) -> Result<(), CommissioningIdentityError> {
    let len =
        u32::try_from(value.len()).map_err(|_| CommissioningIdentityError::StringTooLong(field))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

const fn mode_tag(mode: OperatingMode) -> u8 {
    match mode {
        OperatingMode::Normal => 1,
        OperatingMode::Degraded => 2,
        OperatingMode::Islanded => 3,
        OperatingMode::Emergency => 4,
        OperatingMode::Maintenance => 5,
    }
}

const fn boundary_metric_tag(metric: BoundaryMetric) -> u8 {
    match metric {
        BoundaryMetric::Import => 1,
        BoundaryMetric::Export => 2,
        BoundaryMetric::Reserve => 3,
        BoundaryMetric::AvailableCapacity => 4,
        BoundaryMetric::Demand => 5,
    }
}

const fn resource_kind_tag(kind: ResourceKind) -> u8 {
    match kind {
        ResourceKind::Electricity => 1,
        ResourceKind::ThermalEnergy => 2,
        ResourceKind::CoolingCapacity => 3,
        ResourceKind::Compute => 4,
        ResourceKind::Storage => 5,
        ResourceKind::NetworkBandwidth => 6,
        ResourceKind::Water => 7,
        ResourceKind::Material => 8,
    }
}

const fn resource_unit_tag(unit: ResourceUnit) -> u8 {
    match unit {
        ResourceUnit::Joule => 1,
        ResourceUnit::Watt => 2,
        ResourceUnit::CpuSecond => 3,
        ResourceUnit::GpuSecond => 4,
        ResourceUnit::Byte => 5,
        ResourceUnit::BitPerSecond => 6,
        ResourceUnit::CubicMeter => 7,
        ResourceUnit::CubicMeterPerSecond => 8,
        ResourceUnit::Kilogram => 9,
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommissioningIdentityError {
    #[error(transparent)]
    Commissioning(#[from] CommissioningError),
    #[error("commissioning identity string field {0} exceeds canonical u32 length")]
    StringTooLong(&'static str),
    #[error("commissioning identity collection {0} exceeds canonical u32 length")]
    CollectionTooLarge(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommissionedLocalSafetyEnvelope, CommissioningBinding, CommissioningRecord};
    use std::collections::BTreeSet;
    use symthaea_operating_autonomy::LocalSafetyEnvelope;
    use symthaea_operating_envelope::{BoundaryMetric, OperatingMode, ResourceConstraint};
    use symthaea_resource_hierarchy::{NodeScale, ResourceNode};
    use symthaea_resource_model::{
        ResourceAmount, ResourceEnvelope, ResourceKind, ResourceUnit,
    };

    fn digest(byte: u8) -> ConfigurationDigest {
        ConfigurationDigest::Blake3_256([byte; 32])
    }

    fn hierarchy() -> ResourceHierarchy {
        let mut hierarchy = ResourceHierarchy::default();
        hierarchy
            .insert_root(ResourceNode::new(
                "site",
                "Site",
                NodeScale::Site,
                ResourceEnvelope::default(),
            ))
            .unwrap();
        hierarchy
            .insert_child(
                "site",
                ResourceNode::new(
                    "rack",
                    "Rack",
                    NodeScale::Rack,
                    ResourceEnvelope::default(),
                ),
            )
            .unwrap();
        hierarchy
    }

    fn power_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn thermal_key() -> symthaea_resource_model::ResourceKey {
        ResourceAmount::new(ResourceKind::ThermalEnergy, ResourceUnit::Watt, 0.0)
            .unwrap()
            .key
    }

    fn local() -> LocalSafetyEnvelope {
        let mut modes = BTreeSet::new();
        modes.insert(OperatingMode::Normal);
        modes.insert(OperatingMode::Islanded);
        LocalSafetyEnvelope {
            id: "rack-survival-v1".into(),
            subject_node_id: "rack".into(),
            allowed_modes: modes,
            resource_constraints: vec![
                ResourceConstraint {
                    key: power_key(),
                    metric: BoundaryMetric::Import,
                    minimum: None,
                    maximum: Some(100.0),
                },
                ResourceConstraint {
                    key: thermal_key(),
                    metric: BoundaryMetric::Export,
                    minimum: Some(10.0),
                    maximum: Some(80.0),
                },
            ],
            critical_service_floor: 0.70,
            max_shed_fraction: 0.30,
        }
    }

    fn record(generation: u64, config: u8, evidence: &str, local: LocalSafetyEnvelope) -> CommissioningRecord {
        CommissioningRecord::new(
            generation,
            CommissionedLocalSafetyEnvelope::new(
                local,
                CommissioningBinding::new(digest(config), evidence).unwrap(),
            ),
        )
        .unwrap()
    }

    #[test]
    fn exact_record_identity_is_stable() {
        let hierarchy = hierarchy();
        let first = record(1, 0x11, "commissioning/rack/g1", local());
        let second = first.clone();
        assert_eq!(
            canonical_commissioning_record_bytes(&first, &hierarchy).unwrap(),
            canonical_commissioning_record_bytes(&second, &hierarchy).unwrap()
        );
        assert_eq!(
            commissioning_record_digest(&first, &hierarchy).unwrap(),
            commissioning_record_digest(&second, &hierarchy).unwrap()
        );
    }

    #[test]
    fn changing_local_safety_changes_commissioning_identity() {
        let hierarchy = hierarchy();
        let first = record(1, 0x11, "commissioning/rack/g1", local());
        let mut changed_local = local();
        changed_local.critical_service_floor = 0.80;
        let second = record(1, 0x11, "commissioning/rack/g1", changed_local);
        assert_ne!(
            commissioning_record_digest(&first, &hierarchy).unwrap(),
            commissioning_record_digest(&second, &hierarchy).unwrap()
        );
    }

    #[test]
    fn generation_configuration_and_evidence_are_all_bound() {
        let hierarchy = hierarchy();
        let baseline = record(1, 0x11, "commissioning/rack/g1", local());
        assert_ne!(
            commissioning_record_digest(&baseline, &hierarchy).unwrap(),
            commissioning_record_digest(&record(2, 0x11, "commissioning/rack/g1", local()), &hierarchy).unwrap()
        );
        assert_ne!(
            commissioning_record_digest(&baseline, &hierarchy).unwrap(),
            commissioning_record_digest(&record(1, 0x22, "commissioning/rack/g1", local()), &hierarchy).unwrap()
        );
        assert_ne!(
            commissioning_record_digest(&baseline, &hierarchy).unwrap(),
            commissioning_record_digest(&record(1, 0x11, "commissioning/rack/other", local()), &hierarchy).unwrap()
        );
    }

    #[test]
    fn resource_constraint_vector_order_is_not_identity() {
        let hierarchy = hierarchy();
        let first_local = local();
        let mut second_local = first_local.clone();
        second_local.resource_constraints.reverse();
        let first = record(1, 0x11, "commissioning/rack/g1", first_local);
        let second = record(1, 0x11, "commissioning/rack/g1", second_local);
        assert_ne!(first, second, "raw Rust layout still observes Vec order");
        assert_eq!(
            commissioning_record_digest(&first, &hierarchy).unwrap(),
            commissioning_record_digest(&second, &hierarchy).unwrap(),
            "canonical authority identity must ignore semantically irrelevant constraint order"
        );
    }

    #[test]
    fn negative_zero_is_normalized() {
        let hierarchy = hierarchy();
        let mut plus = local();
        plus.resource_constraints[0].minimum = Some(0.0);
        let mut minus = plus.clone();
        minus.resource_constraints[0].minimum = Some(-0.0);
        let first = record(1, 0x11, "commissioning/rack/g1", plus);
        let second = record(1, 0x11, "commissioning/rack/g1", minus);
        assert_eq!(
            commissioning_record_digest(&first, &hierarchy).unwrap(),
            commissioning_record_digest(&second, &hierarchy).unwrap()
        );
    }
}
