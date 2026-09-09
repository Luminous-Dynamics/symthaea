// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prepared exact-equivalence codebook for sensorimotor HDC v1.
//!
//! R3.1's reference encoder reconstructs thermometer-prefix value bundles on
//! every observation. This module prepares the same value geometry once and
//! caches each exact prefix bundle. Runtime encoding becomes digest lookup +
//! bind + binary→continuous conversion while preserving V1 output exactly.
//!
//! This is a performance implementation, not a new physical semantic schema.
//! `SensorimotorAddressV1::semantic_digest()` remains the exact v1 observation-
//! contract identity. `physical_role_digest()` is narrower: it identifies what
//! physical role is measured while deliberately excluding range and bin count.
//! The codebook's encoding profile is separately named from both identities.

use super::{
    feed_component, feed_frame, feed_quantity, feed_subject, feed_unit, sensorimotor_role_hv_v1,
    sensorimotor_value_bin_v1, sensorimotor_value_level_hv_v1, SensorimotorAddressV1,
    SensorimotorHdcEncodingProfileV1, SensorimotorMeasurementV1, SensorimotorObservationV1,
    SensorimotorValueContractV1,
};
use crate::hdc::{BinaryHV, ContinuousHV};
use std::collections::HashMap;
use std::fmt;

const SENSORIMOTOR_PHYSICAL_ROLE_DOMAIN_V1: &[u8] =
    b"symthaea.sensorimotor.physical-role.v1\0";

/// Relationship between two validated numeric ranges for the same physical role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorimotorRangeRelationV1 {
    Exact,
    ContainsOther,
    ContainedByOther,
    Overlaps,
    Disjoint,
}

/// Relationship between two complete sensorimotor address contracts.
///
/// `SamePhysicalRole` does not imply representation equivalence under HDC v1:
/// v1's role/value basis is still derived from the exact contract digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorimotorContractRelationV1 {
    DifferentPhysicalRole,
    ExactContract,
    SamePhysicalRole {
        range_relation: SensorimotorRangeRelationV1,
        same_bins: bool,
    },
}

impl SensorimotorAddressV1 {
    /// Version of the narrower physical-role identity contract.
    pub const PHYSICAL_ROLE_SCHEMA_ID: &'static str = "symthaea.sensorimotor.physical-role.v1";

    /// Exact 256-bit identity of *what physical role is measured*.
    ///
    /// Includes subject, physical quantity, reference frame, component, and
    /// unit. Deliberately excludes numeric range and quantization bins so two
    /// sensors can identify the same physical role even when their envelopes or
    /// resolution differ.
    ///
    /// This does not replace `semantic_digest()`, whose existing v1 semantics
    /// remain unchanged as the exact observation/encoding-contract identity.
    pub fn physical_role_digest(&self) -> Result<[u8; 32], &'static str> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(SENSORIMOTOR_PHYSICAL_ROLE_DOMAIN_V1);
        feed_subject(&mut hasher, &self.subject);
        feed_quantity(&mut hasher, &self.quantity);
        feed_frame(&mut hasher, &self.frame);
        feed_component(&mut hasher, self.component);
        feed_unit(&mut hasher, &self.value_contract.unit);
        Ok(*hasher.finalize().as_bytes())
    }

    /// Explicit name for the existing exact v1 contract digest.
    ///
    /// Kept as an alias rather than renaming `semantic_digest()` so all existing
    /// R3.1-R3.5 identities and serialized/evidence expectations remain stable.
    pub fn contract_digest(&self) -> Result<[u8; 32], &'static str> {
        self.semantic_digest()
    }

    pub fn same_physical_role(&self, other: &Self) -> Result<bool, &'static str> {
        Ok(self.physical_role_digest()? == other.physical_role_digest()?)
    }

    /// Classify two validated contracts without conflating role identity with
    /// representational equivalence.
    pub fn contract_relation(
        &self,
        other: &Self,
    ) -> Result<SensorimotorContractRelationV1, &'static str> {
        self.validate()?;
        other.validate()?;

        if self.physical_role_digest()? != other.physical_role_digest()? {
            return Ok(SensorimotorContractRelationV1::DifferentPhysicalRole);
        }
        if self.contract_digest()? == other.contract_digest()? {
            return Ok(SensorimotorContractRelationV1::ExactContract);
        }

        Ok(SensorimotorContractRelationV1::SamePhysicalRole {
            range_relation: sensorimotor_range_relation_v1(
                &self.value_contract,
                &other.value_contract,
            ),
            same_bins: self.value_contract.bins == other.value_contract.bins,
        })
    }
}

fn sensorimotor_range_relation_v1(
    left: &SensorimotorValueContractV1,
    right: &SensorimotorValueContractV1,
) -> SensorimotorRangeRelationV1 {
    if left.min == right.min && left.max == right.max {
        SensorimotorRangeRelationV1::Exact
    } else if left.min <= right.min && left.max >= right.max {
        SensorimotorRangeRelationV1::ContainsOther
    } else if right.min <= left.min && right.max >= left.max {
        SensorimotorRangeRelationV1::ContainedByOther
    } else if left.max < right.min || right.max < left.min {
        SensorimotorRangeRelationV1::Disjoint
    } else {
        SensorimotorRangeRelationV1::Overlaps
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedSensorimotorCodebookErrorV1 {
    Schema(&'static str),
    AddressNotPrepared([u8; 32]),
}

impl fmt::Display for PreparedSensorimotorCodebookErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(message) => write!(f, "sensorimotor schema: {message}"),
            Self::AddressNotPrepared(digest) => {
                write!(f, "sensorimotor address not prepared: {digest:02x?}")
            }
        }
    }
}

impl std::error::Error for PreparedSensorimotorCodebookErrorV1 {}

impl From<&'static str> for PreparedSensorimotorCodebookErrorV1 {
    fn from(value: &'static str) -> Self {
        Self::Schema(value)
    }
}

/// Prepared representation of one exact sensorimotor address under the existing
/// thermometer-prefix-binary v1 encoding profile.
#[derive(Debug, Clone)]
pub struct PreparedSensorimotorRoleV1 {
    address_digest: [u8; 32],
    role: BinaryHV,
    value_prefixes: Vec<BinaryHV>,
}

impl PreparedSensorimotorRoleV1 {
    pub fn prepare(
        address: &SensorimotorAddressV1,
    ) -> Result<Self, PreparedSensorimotorCodebookErrorV1> {
        address.validate()?;
        let address_digest = address.semantic_digest()?;
        let role = sensorimotor_role_hv_v1(&address_digest);
        let bins = usize::from(address.value_contract.bins);

        // Counts are cumulative bipolar votes. Because V1 bundle semantics use
        // strict majority (`count > 0`), materializing after each new level is
        // exactly equivalent to `BinaryHV::bundle(levels[0..=bin])` without
        // regenerating/rebundling the whole prefix for every future sample.
        let mut counts = vec![[0i16; 8]; BinaryHV::BYTES];
        let mut value_prefixes = Vec::with_capacity(bins);

        for level in 0..address.value_contract.bins {
            let level_hv = sensorimotor_value_level_hv_v1(&address_digest, level);
            for (byte_counts, byte) in counts.iter_mut().zip(level_hv.0.iter().copied()) {
                for (bit, count) in byte_counts.iter_mut().enumerate() {
                    if ((byte >> bit) & 1) == 1 {
                        *count += 1;
                    } else {
                        *count -= 1;
                    }
                }
            }

            let mut prefix = [0u8; BinaryHV::BYTES];
            for (out, byte_counts) in prefix.iter_mut().zip(counts.iter()) {
                let mut byte = 0u8;
                for (bit, count) in byte_counts.iter().enumerate() {
                    if *count > 0 {
                        byte |= 1 << bit;
                    }
                }
                *out = byte;
            }
            value_prefixes.push(BinaryHV(prefix));
        }

        Ok(Self {
            address_digest,
            role,
            value_prefixes,
        })
    }

    pub const fn profile(&self) -> SensorimotorHdcEncodingProfileV1 {
        SensorimotorHdcEncodingProfileV1::ThermometerPrefixBinaryV1
    }

    pub fn address_digest(&self) -> [u8; 32] {
        self.address_digest
    }

    pub fn bins(&self) -> usize {
        self.value_prefixes.len()
    }

    /// Logical payload bytes retained by the exact value-prefix table.
    ///
    /// This intentionally excludes allocator metadata and any spare capacity,
    /// so the result is deterministic across allocators and platforms. It is
    /// suitable for explicit codebook payload budgets, not a claim about total
    /// process RSS or allocator-reserved bytes.
    pub fn value_table_bytes(&self) -> usize {
        self.value_prefixes.len() * std::mem::size_of::<BinaryHV>()
    }

    pub fn encode_measurement(
        &self,
        measurement: &SensorimotorMeasurementV1,
    ) -> Result<ContinuousHV, PreparedSensorimotorCodebookErrorV1> {
        measurement.validate()?;
        let digest = measurement.address.semantic_digest()?;
        if digest != self.address_digest {
            return Err(PreparedSensorimotorCodebookErrorV1::AddressNotPrepared(
                digest,
            ));
        }
        let bin = usize::from(sensorimotor_value_bin_v1(
            &measurement.address.value_contract,
            measurement.value,
        ));
        let value = self.value_prefixes[bin];
        Ok(self.role.bind(&value).to_continuous())
    }
}

/// Multi-role prepared codebook. Duplicate addresses are de-duplicated by their
/// exact semantic digest.
#[derive(Debug, Clone, Default)]
pub struct PreparedSensorimotorCodebookV1 {
    roles: HashMap<[u8; 32], PreparedSensorimotorRoleV1>,
}

impl PreparedSensorimotorCodebookV1 {
    pub fn prepare(
        addresses: &[SensorimotorAddressV1],
    ) -> Result<Self, PreparedSensorimotorCodebookErrorV1> {
        let mut roles = HashMap::with_capacity(addresses.len());
        for address in addresses {
            let digest = address.semantic_digest()?;
            if let std::collections::hash_map::Entry::Vacant(entry) = roles.entry(digest) {
                entry.insert(PreparedSensorimotorRoleV1::prepare(address)?);
            }
        }
        Ok(Self { roles })
    }

    pub fn prepare_observations(
        observations: &[SensorimotorObservationV1],
    ) -> Result<Self, PreparedSensorimotorCodebookErrorV1> {
        let addresses: Vec<SensorimotorAddressV1> =
            observations.iter().map(|o| o.address().clone()).collect();
        Self::prepare(&addresses)
    }

    pub const fn profile(&self) -> SensorimotorHdcEncodingProfileV1 {
        SensorimotorHdcEncodingProfileV1::ThermometerPrefixBinaryV1
    }

    pub fn prepared_role_count(&self) -> usize {
        self.roles.len()
    }

    pub fn value_table_bytes(&self) -> usize {
        self.roles
            .values()
            .map(PreparedSensorimotorRoleV1::value_table_bytes)
            .sum()
    }

    pub fn encode_observation(
        &self,
        observation: &SensorimotorObservationV1,
    ) -> Result<Option<ContinuousHV>, PreparedSensorimotorCodebookErrorV1> {
        observation.validate()?;
        match observation {
            SensorimotorObservationV1::Measured(measurement) => {
                let digest = measurement.address.semantic_digest()?;
                let role = self.roles.get(&digest).ok_or(
                    PreparedSensorimotorCodebookErrorV1::AddressNotPrepared(digest),
                )?;
                Ok(Some(role.encode_measurement(measurement)?))
            }
            SensorimotorObservationV1::Missing { .. } => Ok(None),
        }
    }

    pub fn encode_observations(
        &self,
        observations: &[SensorimotorObservationV1],
    ) -> Result<Option<ContinuousHV>, PreparedSensorimotorCodebookErrorV1> {
        let mut encoded = Vec::new();
        for observation in observations {
            if let Some(hv) = self.encode_observation(observation)? {
                encoded.push(hv);
            }
        }
        if encoded.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&ContinuousHV> = encoded.iter().collect();
        Ok(Some(ContinuousHV::bundle(&refs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::sensorimotor_contingencies::{
        MissingObservationReasonV1, SensorimotorComponentV1, SensorimotorFrameV1,
        SensorimotorHdcEncoderV1, SensorimotorQuantityV1, SensorimotorSubjectV1,
        SensorimotorUnitV1, SensorimotorValueContractV1,
    };

    fn address(component: SensorimotorComponentV1) -> SensorimotorAddressV1 {
        SensorimotorAddressV1::new(
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::AngularVelocity,
            SensorimotorFrameV1::Body,
            component,
            SensorimotorValueContractV1 {
                unit: SensorimotorUnitV1::RadianPerSecond,
                min: -20.0,
                max: 20.0,
                bins: 17,
            },
        )
    }

    fn measured(address: SensorimotorAddressV1, value: f64) -> SensorimotorObservationV1 {
        SensorimotorObservationV1::Measured(SensorimotorMeasurementV1 { address, value })
    }

    #[test]
    fn profile_id_is_explicit_and_stable() {
        let direct = SensorimotorHdcEncoderV1;
        assert_eq!(
            direct.profile().schema_id(),
            "symthaea.sensorimotor.hdc.thermometer-prefix-binary.v1"
        );
        let prepared = PreparedSensorimotorRoleV1::prepare(&address(SensorimotorComponentV1::X))
            .unwrap();
        assert_eq!(direct.profile(), prepared.profile());
    }

    #[test]
    fn prepared_role_is_bit_exact_at_range_edges_interior_and_saturation() {
        let address = address(SensorimotorComponentV1::X);
        let prepared = PreparedSensorimotorRoleV1::prepare(&address).unwrap();
        let direct = SensorimotorHdcEncoderV1;

        for value in [-100.0, -20.0, -17.5, -0.1, 0.0, 7.5, 19.9, 20.0, 100.0] {
            let measurement = SensorimotorMeasurementV1 {
                address: address.clone(),
                value,
            };
            let expected = direct.encode_measurement(&measurement).unwrap();
            let actual = prepared.encode_measurement(&measurement).unwrap();
            assert_eq!(expected.values, actual.values, "value {value}");
        }
    }

    #[test]
    fn prepared_multi_role_bundle_is_exactly_reference_equivalent() {
        let x = address(SensorimotorComponentV1::X);
        let y = address(SensorimotorComponentV1::Y);
        let observations = vec![measured(x, 3.2), measured(y, -4.1)];
        let prepared = PreparedSensorimotorCodebookV1::prepare_observations(&observations).unwrap();
        let direct = SensorimotorHdcEncoderV1
            .encode_observations(&observations)
            .unwrap()
            .unwrap();
        let cached = prepared
            .encode_observations(&observations)
            .unwrap()
            .unwrap();
        assert_eq!(direct.values, cached.values);
        assert_eq!(prepared.prepared_role_count(), 2);
    }

    #[test]
    fn duplicate_addresses_are_prepared_once() {
        let x = address(SensorimotorComponentV1::X);
        let prepared = PreparedSensorimotorCodebookV1::prepare(&[x.clone(), x]).unwrap();
        assert_eq!(prepared.prepared_role_count(), 1);
        assert_eq!(prepared.value_table_bytes(), 17 * BinaryHV::BYTES);
    }

    #[test]
    fn unprepared_measured_role_fails_closed() {
        let x = address(SensorimotorComponentV1::X);
        let y = address(SensorimotorComponentV1::Y);
        let prepared = PreparedSensorimotorCodebookV1::prepare(&[x]).unwrap();
        let error = prepared
            .encode_observation(&measured(y, 0.0))
            .expect_err("unprepared role must not fall back silently");
        assert!(matches!(
            error,
            PreparedSensorimotorCodebookErrorV1::AddressNotPrepared(_)
        ));
    }

    #[test]
    fn missing_observation_stays_missing_without_lookup() {
        let x = address(SensorimotorComponentV1::X);
        let missing = SensorimotorObservationV1::Missing {
            address: x.clone(),
            reason: MissingObservationReasonV1::NotObserved,
        };
        let prepared = PreparedSensorimotorCodebookV1::prepare(&[x]).unwrap();
        assert!(prepared.encode_observation(&missing).unwrap().is_none());
    }

    #[test]
    fn physical_role_identity_ignores_range_and_resolution() {
        let base = address(SensorimotorComponentV1::X);
        let mut wider = base.clone();
        wider.value_contract.min = -40.0;
        wider.value_contract.max = 40.0;
        wider.value_contract.bins = 401;

        assert_eq!(
            base.physical_role_digest().unwrap(),
            wider.physical_role_digest().unwrap()
        );
        assert_ne!(base.contract_digest().unwrap(), wider.contract_digest().unwrap());
        assert!(base.same_physical_role(&wider).unwrap());
        assert_eq!(
            base.contract_relation(&wider).unwrap(),
            SensorimotorContractRelationV1::SamePhysicalRole {
                range_relation: SensorimotorRangeRelationV1::ContainedByOther,
                same_bins: false,
            }
        );
    }

    #[test]
    fn contract_relation_distinguishes_exact_overlap_and_disjoint() {
        let base = address(SensorimotorComponentV1::X);
        assert_eq!(
            base.contract_relation(&base).unwrap(),
            SensorimotorContractRelationV1::ExactContract
        );

        let mut overlap = base.clone();
        overlap.value_contract.min = 10.0;
        overlap.value_contract.max = 30.0;
        assert_eq!(
            base.contract_relation(&overlap).unwrap(),
            SensorimotorContractRelationV1::SamePhysicalRole {
                range_relation: SensorimotorRangeRelationV1::Overlaps,
                same_bins: true,
            }
        );

        let mut disjoint = base.clone();
        disjoint.value_contract.min = 21.0;
        disjoint.value_contract.max = 30.0;
        assert_eq!(
            base.contract_relation(&disjoint).unwrap(),
            SensorimotorContractRelationV1::SamePhysicalRole {
                range_relation: SensorimotorRangeRelationV1::Disjoint,
                same_bins: true,
            }
        );
    }

    #[test]
    fn physical_role_identity_changes_with_component_and_frame() {
        let x = address(SensorimotorComponentV1::X);
        let y = address(SensorimotorComponentV1::Y);
        assert_ne!(x.physical_role_digest().unwrap(), y.physical_role_digest().unwrap());

        let mut world = x.clone();
        world.frame = SensorimotorFrameV1::World;
        assert_ne!(
            x.physical_role_digest().unwrap(),
            world.physical_role_digest().unwrap()
        );
        assert_eq!(
            x.contract_relation(&world).unwrap(),
            SensorimotorContractRelationV1::DifferentPhysicalRole
        );
    }

    #[test]
    fn physical_role_identity_includes_unit() {
        let meter = SensorimotorAddressV1::new(
            SensorimotorSubjectV1::BodyRoot,
            SensorimotorQuantityV1::Custom("test_quantity".into()),
            SensorimotorFrameV1::Body,
            SensorimotorComponentV1::Scalar,
            SensorimotorValueContractV1 {
                unit: SensorimotorUnitV1::Meter,
                min: 0.0,
                max: 1.0,
                bins: 17,
            },
        );
        let mut radian = meter.clone();
        radian.value_contract.unit = SensorimotorUnitV1::Radian;

        assert_ne!(
            meter.physical_role_digest().unwrap(),
            radian.physical_role_digest().unwrap()
        );
    }

    #[test]
    fn contract_digest_alias_preserves_existing_semantic_identity() {
        let role = address(SensorimotorComponentV1::Z);
        assert_eq!(role.contract_digest().unwrap(), role.semantic_digest().unwrap());
    }
}
