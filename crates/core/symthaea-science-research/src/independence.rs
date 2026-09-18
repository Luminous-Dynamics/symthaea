// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::Sha256Digest;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeSet;

/// Describes *how* two evidence lineages differ. Independence is intentionally
/// not a boolean and these dimensions are not treated as one total ordering.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum IndependenceDimension {
    SameExecutionReplay,
    IndependentExecutionSameBinary,
    IndependentImplementationSharedInputs,
    IndependentMethodSharedRawData,
    IndependentDataReduction,
    IndependentDataset,
    IndependentSite,
    IndependentOrganization,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum SharedRootKind {
    RawData,
    Calibration,
    Preprocessing,
    Implementation,
    SolverLibrary,
    ModelCheckpoint,
    Organization,
    Instrument,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SharedRoot {
    pub kind: SharedRootKind,
    pub digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct IndependenceProfile {
    dimensions: BTreeSet<IndependenceDimension>,
    shared_roots: BTreeSet<SharedRoot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependenceIssue {
    EmptyDimensions,
    ReplayContradictsIndependentDimension,
}

impl IndependenceProfile {
    pub fn new(
        dimensions: impl IntoIterator<Item = IndependenceDimension>,
        shared_roots: impl IntoIterator<Item = SharedRoot>,
    ) -> Result<Self, Vec<IndependenceIssue>> {
        let profile = Self {
            dimensions: dimensions.into_iter().collect(),
            shared_roots: shared_roots.into_iter().collect(),
        };
        let issues = profile.validate();
        if issues.is_empty() {
            Ok(profile)
        } else {
            Err(issues)
        }
    }

    pub fn validate(&self) -> Vec<IndependenceIssue> {
        let mut issues = Vec::new();
        if self.dimensions.is_empty() {
            issues.push(IndependenceIssue::EmptyDimensions);
        }
        if self
            .dimensions
            .contains(&IndependenceDimension::SameExecutionReplay)
            && self.dimensions.len() > 1
        {
            issues.push(IndependenceIssue::ReplayContradictsIndependentDimension);
        }
        issues
    }

    pub fn dimensions(&self) -> &BTreeSet<IndependenceDimension> {
        &self.dimensions
    }

    pub fn shared_roots(&self) -> &BTreeSet<SharedRoot> {
        &self.shared_roots
    }

    pub fn shares_root(&self, digest: &Sha256Digest) -> bool {
        self.shared_roots.iter().any(|root| &root.digest == digest)
    }
}

#[derive(Deserialize)]
struct IndependenceProfileWire {
    dimensions: Vec<IndependenceDimension>,
    shared_roots: Vec<SharedRoot>,
}

impl<'de> Deserialize<'de> for IndependenceProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = IndependenceProfileWire::deserialize(deserializer)?;
        Self::new(wire.dimensions, wire.shared_roots)
            .map_err(|issues| serde::de::Error::custom(format!("invalid independence profile: {issues:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_execution_is_not_independent_by_label() {
        let result = IndependenceProfile::new(
            [
                IndependenceDimension::SameExecutionReplay,
                IndependenceDimension::IndependentOrganization,
            ],
            [],
        );
        assert!(result.is_err());
    }

    #[test]
    fn invalid_wire_profile_cannot_bypass_constructor() {
        let forged = r#"{
            "dimensions":["SameExecutionReplay","IndependentOrganization"],
            "shared_roots":[]
        }"#;
        assert!(serde_json::from_str::<IndependenceProfile>(forged).is_err());
    }

    #[test]
    fn shared_data_root_is_preserved_explicitly() {
        let data = Sha256Digest::of_bytes(b"released-dataset");
        let profile = IndependenceProfile::new(
            [IndependenceDimension::IndependentImplementationSharedInputs],
            [SharedRoot {
                kind: SharedRootKind::RawData,
                digest: data.clone(),
            }],
        )
        .unwrap();
        assert!(profile.shares_root(&data));
    }
}
