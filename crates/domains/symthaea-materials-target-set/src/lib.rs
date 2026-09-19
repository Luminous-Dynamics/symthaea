// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-derived construction of historical benchmark target sets.
//!
//! A benchmark target must be generated from disclosure facts whose source granularity
//! actually supports the requested structure/property authority. Hand-authored target
//! JSON is therefore not sufficient evidence of target eligibility.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use symthaea_materials_corpus_audit::{HistoricalBenchmarkTarget, PropertyLabelRef};
use symthaea_materials_target_disclosure::{
    BenchmarkTargetAuthority, DisclosureFact, TargetDisclosureManifest,
};
use thiserror::Error;

/// Requested composition of one scored target from disclosure fact IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetConstructionRequest {
    /// Stable benchmark target ID.
    pub target_id: String,
    /// Exact disclosure fact that establishes the target structure.
    pub structure_fact_id: String,
    /// One or more disclosure facts establishing quantitative scored properties.
    pub property_fact_ids: Vec<String>,
}

/// One quantitative label authorized by source evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedQuantitativeLabel {
    /// Disclosure fact from which this label was derived.
    pub disclosure_fact_id: String,
    /// Stable scientific property ID.
    pub property_id: String,
    /// Exact canonical numeric text from source evidence.
    pub value: String,
    /// Exact source unit string.
    pub unit: String,
    /// Bound method/condition signature.
    pub condition_signature: String,
    /// Exact method/source artifact establishing this label.
    pub method_artifact_sha256: String,
}

/// One structure-specific scored target authorized by the disclosure manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedHistoricalTarget {
    /// Stable benchmark target ID.
    pub target_id: String,
    /// Exact canonical composition identity.
    pub composition_sha256: String,
    /// Exact canonical structure identity.
    pub structure_sha256: String,
    /// Exact structure artifact from the disclosure source.
    pub structure_artifact_sha256: String,
    /// Disclosure fact establishing structure authority.
    pub structure_fact_id: String,
    /// Authorized quantitative properties, ordered by disclosure fact ID.
    pub properties: Vec<AuthorizedQuantitativeLabel>,
}

/// Complete target set derived from one exact disclosure manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedHistoricalTargetSet {
    /// Target-set schema version.
    pub schema_version: u32,
    /// Exact target-disclosure manifest identity.
    pub disclosure_manifest_sha256: String,
    /// Exact captured publication/source artifact identity.
    pub source_artifact_sha256: String,
    /// Source disclosure date.
    pub disclosure_date: String,
    /// Targets ordered lexically by target ID.
    pub targets: Vec<AuthorizedHistoricalTarget>,
}

impl AuthorizedHistoricalTargetSet {
    /// Validate canonical ordering, ID uniqueness, and semantic-target uniqueness.
    pub fn validate(&self) -> Result<(), TargetSetError> {
        if self.schema_version != 1 {
            return Err(TargetSetError::UnsupportedSchema(self.schema_version));
        }
        sha256(&self.disclosure_manifest_sha256)?;
        sha256(&self.source_artifact_sha256)?;
        nonempty("disclosure_date", &self.disclosure_date)?;
        if self.targets.is_empty() {
            return Err(TargetSetError::EmptyTargetSet);
        }

        let mut previous_target: Option<&str> = None;
        let mut target_ids = HashSet::new();
        let mut scientific_targets = HashSet::new();
        for target in &self.targets {
            target.validate()?;
            if !target_ids.insert(target.target_id.as_str()) {
                return Err(TargetSetError::DuplicateTargetId(target.target_id.clone()));
            }
            if previous_target.is_some_and(|prior| target.target_id.as_str() <= prior) {
                return Err(TargetSetError::NonCanonicalTargetOrder);
            }
            previous_target = Some(&target.target_id);

            let semantic_identity = target.semantic_identity_sha256()?;
            if !scientific_targets.insert(semantic_identity) {
                return Err(TargetSetError::DuplicateScientificTarget(
                    target.target_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Deterministic identity of the authority-derived target set.
    pub fn target_set_sha256(&self) -> Result<String, TargetSetError> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }

    /// Narrow contamination-audit view; quantitative values remain in this manifest.
    pub fn to_audit_targets(&self) -> Result<Vec<HistoricalBenchmarkTarget>, TargetSetError> {
        self.validate()?;
        Ok(self
            .targets
            .iter()
            .map(|target| HistoricalBenchmarkTarget {
                target_id: target.target_id.clone(),
                composition_sha256: target.composition_sha256.clone(),
                structure_sha256: Some(target.structure_sha256.clone()),
                scored_property_labels: target
                    .properties
                    .iter()
                    .map(|property| PropertyLabelRef {
                        property_id: property.property_id.clone(),
                        condition_signature: property.condition_signature.clone(),
                    })
                    .collect(),
            })
            .collect())
    }
}

impl AuthorizedHistoricalTarget {
    fn validate(&self) -> Result<(), TargetSetError> {
        nonempty("target_id", &self.target_id)?;
        nonempty("structure_fact_id", &self.structure_fact_id)?;
        sha256(&self.composition_sha256)?;
        sha256(&self.structure_sha256)?;
        sha256(&self.structure_artifact_sha256)?;
        if self.properties.is_empty() {
            return Err(TargetSetError::NoQuantitativeProperties(self.target_id.clone()));
        }
        let mut previous_fact: Option<&str> = None;
        let mut labels = BTreeSet::new();
        for property in &self.properties {
            property.validate()?;
            if previous_fact.is_some_and(|prior| property.disclosure_fact_id.as_str() <= prior) {
                return Err(TargetSetError::NonCanonicalPropertyOrder(self.target_id.clone()));
            }
            previous_fact = Some(&property.disclosure_fact_id);
            let label = (
                property.property_id.as_str(),
                property.condition_signature.as_str(),
            );
            if !labels.insert(label) {
                return Err(TargetSetError::DuplicatePropertyLabel {
                    target_id: self.target_id.clone(),
                    property_id: property.property_id.clone(),
                    condition_signature: property.condition_signature.clone(),
                });
            }
        }
        Ok(())
    }

    /// Identity of the scientific scoring subject, independent of the local target ID
    /// and source fact IDs. The property set is `(property_id, condition_signature)`;
    /// numeric answers are intentionally excluded because they are labels to predict,
    /// not part of target identity.
    pub fn semantic_identity_sha256(&self) -> Result<String, TargetSetError> {
        self.validate()?;
        let mut labels: Vec<(&str, &str)> = self
            .properties
            .iter()
            .map(|property| {
                (
                    property.property_id.as_str(),
                    property.condition_signature.as_str(),
                )
            })
            .collect();
        labels.sort_unstable();
        Ok(sha256_hex(&serde_json::to_vec(&(
            self.composition_sha256.as_str(),
            self.structure_sha256.as_str(),
            labels,
        ))?))
    }
}

impl AuthorizedQuantitativeLabel {
    fn validate(&self) -> Result<(), TargetSetError> {
        nonempty("disclosure_fact_id", &self.disclosure_fact_id)?;
        nonempty("property_id", &self.property_id)?;
        nonempty("value", &self.value)?;
        nonempty("unit", &self.unit)?;
        nonempty("condition_signature", &self.condition_signature)?;
        sha256(&self.method_artifact_sha256)
    }
}

/// Build a scored target set only from disclosure facts with sufficient authority.
pub fn build_authorized_target_set(
    manifest: &TargetDisclosureManifest,
    requests: &[TargetConstructionRequest],
) -> Result<AuthorizedHistoricalTargetSet, TargetSetError> {
    manifest
        .validate()
        .map_err(|error| TargetSetError::Disclosure(error.to_string()))?;
    if requests.is_empty() {
        return Err(TargetSetError::EmptyRequestSet);
    }

    let mut canonical_requests = requests.to_vec();
    canonical_requests.sort_by(|a, b| a.target_id.cmp(&b.target_id));
    let mut request_ids = HashSet::new();
    let mut targets = Vec::with_capacity(canonical_requests.len());

    for request in canonical_requests {
        nonempty("target_id", &request.target_id)?;
        if !request_ids.insert(request.target_id.clone()) {
            return Err(TargetSetError::DuplicateTargetId(request.target_id));
        }
        if request.property_fact_ids.is_empty() {
            return Err(TargetSetError::NoRequestedProperties(request.target_id));
        }

        let structure_fact = find_fact(&manifest.facts, &request.structure_fact_id)?;
        if structure_fact
            .authority(&manifest.source)
            .map_err(|error| TargetSetError::Disclosure(error.to_string()))?
            != BenchmarkTargetAuthority::StructureIdentity
        {
            return Err(TargetSetError::InsufficientStructureAuthority(
                request.structure_fact_id,
            ));
        }
        let (structure_composition, structure_sha, structure_artifact) = match structure_fact {
            DisclosureFact::StructureTarget {
                composition_sha256,
                structure_sha256,
                structure_artifact_sha256,
                ..
            } => (
                composition_sha256.clone(),
                structure_sha256.clone(),
                structure_artifact_sha256.clone(),
            ),
            _ => return Err(TargetSetError::WrongFactKind(request.structure_fact_id)),
        };

        let mut property_fact_ids = request.property_fact_ids;
        property_fact_ids.sort();
        let mut seen_fact_ids = HashSet::new();
        let mut properties = Vec::with_capacity(property_fact_ids.len());
        for property_fact_id in property_fact_ids {
            if !seen_fact_ids.insert(property_fact_id.clone()) {
                return Err(TargetSetError::DuplicatePropertyFact(property_fact_id));
            }
            let property_fact = find_fact(&manifest.facts, &property_fact_id)?;
            if property_fact
                .authority(&manifest.source)
                .map_err(|error| TargetSetError::Disclosure(error.to_string()))?
                != BenchmarkTargetAuthority::QuantitativeProperty
            {
                return Err(TargetSetError::InsufficientPropertyAuthority(property_fact_id));
            }
            let DisclosureFact::QuantitativePropertyMention {
                composition_sha256,
                structure_sha256: Some(property_structure),
                property_id,
                value,
                unit,
                condition_signature: Some(condition_signature),
                method_artifact_sha256: Some(method_artifact_sha256),
                ..
            } = property_fact
            else {
                return Err(TargetSetError::WrongFactKind(property_fact_id));
            };
            if composition_sha256 != &structure_composition {
                return Err(TargetSetError::PropertyCompositionMismatch(property_fact_id));
            }
            if property_structure != &structure_sha {
                return Err(TargetSetError::PropertyStructureMismatch(property_fact_id));
            }
            properties.push(AuthorizedQuantitativeLabel {
                disclosure_fact_id: property_fact_id,
                property_id: property_id.clone(),
                value: value.clone(),
                unit: unit.clone(),
                condition_signature: condition_signature.clone(),
                method_artifact_sha256: method_artifact_sha256.clone(),
            });
        }

        let target = AuthorizedHistoricalTarget {
            target_id: request.target_id,
            composition_sha256: structure_composition,
            structure_sha256: structure_sha,
            structure_artifact_sha256: structure_artifact,
            structure_fact_id: request.structure_fact_id,
            properties,
        };
        target.validate()?;
        targets.push(target);
    }

    let target_set = AuthorizedHistoricalTargetSet {
        schema_version: 1,
        disclosure_manifest_sha256: manifest
            .manifest_sha256()
            .map_err(|error| TargetSetError::Disclosure(error.to_string()))?,
        source_artifact_sha256: manifest.source.source_artifact_sha256.clone(),
        disclosure_date: manifest.source.disclosure_date.clone(),
        targets,
    };
    target_set.validate()?;
    Ok(target_set)
}

fn find_fact<'a>(facts: &'a [DisclosureFact], id: &str) -> Result<&'a DisclosureFact, TargetSetError> {
    nonempty("fact_id", id)?;
    facts
        .iter()
        .find(|fact| fact_id(fact) == id)
        .ok_or_else(|| TargetSetError::UnknownFactId(id.to_string()))
}

fn fact_id(fact: &DisclosureFact) -> &str {
    match fact {
        DisclosureFact::AggregateCount { fact_id, .. }
        | DisclosureFact::CompositionMention { fact_id, .. }
        | DisclosureFact::StructureTarget { fact_id, .. }
        | DisclosureFact::QuantitativePropertyMention { fact_id, .. } => fact_id,
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), TargetSetError> {
    if value.trim().is_empty() {
        Err(TargetSetError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256(value: &str) -> Result<(), TargetSetError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(TargetSetError::InvalidSha256(value.to_string()))
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Target-set construction or validation failure.
#[derive(Debug, Error)]
pub enum TargetSetError {
    /// Disclosure manifest/fact rejected its own evidence.
    #[error("target disclosure authority rejected input: {0}")]
    Disclosure(String),
    /// Required field empty.
    #[error("required target-set field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid SHA-256: {0}")]
    InvalidSha256(String),
    /// Target-set schema unsupported.
    #[error("unsupported target-set schema {0}")]
    UnsupportedSchema(u32),
    /// No construction requests supplied.
    #[error("target construction request set is empty")]
    EmptyRequestSet,
    /// Resulting target set empty.
    #[error("authorized target set is empty")]
    EmptyTargetSet,
    /// Request/target ID repeated.
    #[error("duplicate target ID: {0}")]
    DuplicateTargetId(String),
    /// Same scientific target is repeated under another local target ID.
    #[error("duplicate scientific target under local target ID: {0}")]
    DuplicateScientificTarget(String),
    /// Target order is not canonical.
    #[error("authorized targets are not in strict lexical order")]
    NonCanonicalTargetOrder,
    /// Request contains no quantitative property facts.
    #[error("target request contains no property facts: {0}")]
    NoRequestedProperties(String),
    /// Authorized target contains no quantitative labels.
    #[error("authorized target contains no quantitative properties: {0}")]
    NoQuantitativeProperties(String),
    /// Requested disclosure fact does not exist.
    #[error("unknown disclosure fact ID: {0}")]
    UnknownFactId(String),
    /// Requested fact has an incompatible variant.
    #[error("disclosure fact has wrong kind for requested role: {0}")]
    WrongFactKind(String),
    /// Structure fact lacks structure authority.
    #[error("structure fact lacks structure-target authority: {0}")]
    InsufficientStructureAuthority(String),
    /// Property fact lacks quantitative-property authority.
    #[error("property fact lacks quantitative-property authority: {0}")]
    InsufficientPropertyAuthority(String),
    /// Property fact ID repeated inside one target request.
    #[error("duplicate property disclosure fact: {0}")]
    DuplicatePropertyFact(String),
    /// Property and structure facts refer to different compositions.
    #[error("property fact composition differs from target structure: {0}")]
    PropertyCompositionMismatch(String),
    /// Property and structure facts refer to different structures.
    #[error("property fact structure differs from target structure: {0}")]
    PropertyStructureMismatch(String),
    /// Property ordering inside an authorized target is noncanonical.
    #[error("authorized property order is not canonical for target {0}")]
    NonCanonicalPropertyOrder(String),
    /// Same scientific property/condition appears more than once.
    #[error("duplicate property label for target {target_id}: {property_id} @ {condition_signature}")]
    DuplicatePropertyLabel {
        /// Target ID.
        target_id: String,
        /// Property ID.
        property_id: String,
        /// Condition signature.
        condition_signature: String,
    },
    /// JSON serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_target_disclosure::{
        DisclosureAccess, DisclosureSource, TargetDisclosureManifest,
    };

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn source(access: DisclosureAccess) -> DisclosureSource {
        DisclosureSource {
            source_id: "paper".to_string(),
            publication_id: "10.0000/example".to_string(),
            locator: "source".to_string(),
            source_artifact_sha256: hex('a'),
            access,
            disclosure_date: "2026-02-09".to_string(),
        }
    }

    fn facts() -> Vec<DisclosureFact> {
        vec![
            DisclosureFact::StructureTarget {
                fact_id: "01-structure-a".to_string(),
                composition_sha256: hex('b'),
                structure_sha256: hex('c'),
                structure_artifact_sha256: hex('d'),
            },
            DisclosureFact::QuantitativePropertyMention {
                fact_id: "02-k1-a".to_string(),
                composition_sha256: hex('b'),
                structure_sha256: Some(hex('c')),
                property_id: "k1_mj_m3".to_string(),
                value: "1.1".to_string(),
                unit: "MJ/m^3".to_string(),
                condition_signature: Some("0K|SOC".to_string()),
                method_artifact_sha256: Some(hex('e')),
            },
            DisclosureFact::StructureTarget {
                fact_id: "03-structure-b".to_string(),
                composition_sha256: hex('f'),
                structure_sha256: hex('1'),
                structure_artifact_sha256: hex('2'),
            },
            DisclosureFact::QuantitativePropertyMention {
                fact_id: "04-k1-b".to_string(),
                composition_sha256: hex('f'),
                structure_sha256: Some(hex('1')),
                property_id: "k1_mj_m3".to_string(),
                value: "0.7".to_string(),
                unit: "MJ/m^3".to_string(),
                condition_signature: Some("0K|SOC".to_string()),
                method_artifact_sha256: Some(hex('3')),
            },
        ]
    }

    fn request_a(id: &str) -> TargetConstructionRequest {
        TargetConstructionRequest {
            target_id: id.to_string(),
            structure_fact_id: "01-structure-a".to_string(),
            property_fact_ids: vec!["02-k1-a".to_string()],
        }
    }

    fn request_b(id: &str) -> TargetConstructionRequest {
        TargetConstructionRequest {
            target_id: id.to_string(),
            structure_fact_id: "03-structure-b".to_string(),
            property_fact_ids: vec!["04-k1-b".to_string()],
        }
    }

    #[test]
    fn abstract_cannot_generate_quantitative_scored_target() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Abstract),
            facts: facts(),
        };
        assert!(matches!(
            build_authorized_target_set(&manifest, &[request_a("candidate")]),
            Err(TargetSetError::InsufficientStructureAuthority(_))
        ));
    }

    #[test]
    fn supplement_can_generate_exact_structure_property_target() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Supplement),
            facts: facts(),
        };
        let target_set = build_authorized_target_set(&manifest, &[request_a("candidate")]).unwrap();
        let audit_targets = target_set.to_audit_targets().unwrap();
        let expected_structure = hex('c');
        assert_eq!(audit_targets.len(), 1);
        assert_eq!(
            audit_targets[0].structure_sha256.as_deref(),
            Some(expected_structure.as_str())
        );
        assert_eq!(audit_targets[0].scored_property_labels[0].property_id, "k1_mj_m3");
        assert_eq!(target_set.target_set_sha256().unwrap().len(), 64);
    }

    #[test]
    fn request_order_does_not_change_target_set_identity() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Supplement),
            facts: facts(),
        };
        let first = build_authorized_target_set(
            &manifest,
            &[request_b("b"), request_a("a")],
        )
        .unwrap();
        let second = build_authorized_target_set(
            &manifest,
            &[request_a("a"), request_b("b")],
        )
        .unwrap();
        assert_eq!(first.target_set_sha256().unwrap(), second.target_set_sha256().unwrap());
    }

    #[test]
    fn duplicate_scientific_target_under_different_ids_is_rejected() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Supplement),
            facts: facts(),
        };
        assert!(matches!(
            build_authorized_target_set(
                &manifest,
                &[request_a("alias-a"), request_a("alias-b")],
            ),
            Err(TargetSetError::DuplicateScientificTarget(_))
        ));
    }

    #[test]
    fn deserialized_duplicate_scientific_target_is_rejected_by_validation() {
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Supplement),
            facts: facts(),
        };
        let mut target_set = build_authorized_target_set(&manifest, &[request_a("a")]).unwrap();
        let mut alias = target_set.targets[0].clone();
        alias.target_id = "b".to_string();
        target_set.targets.push(alias);
        assert!(matches!(
            target_set.validate(),
            Err(TargetSetError::DuplicateScientificTarget(_))
        ));
    }

    #[test]
    fn property_for_another_structure_is_rejected() {
        let mut facts = facts();
        if let DisclosureFact::QuantitativePropertyMention { structure_sha256, .. } = &mut facts[1] {
            *structure_sha256 = Some(hex('9'));
        }
        let manifest = TargetDisclosureManifest {
            schema_version: 1,
            source: source(DisclosureAccess::Supplement),
            facts,
        };
        assert!(matches!(
            build_authorized_target_set(&manifest, &[request_a("candidate")]),
            Err(TargetSetError::PropertyStructureMismatch(_))
        ));
    }
}
