// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned cyber-ontology bridge for external ATT&CK/D3FEND knowledge.
//!
//! This crate stores compact object identities and mapping assertions, not a
//! wholesale copy of MITRE knowledge bases. Every framework revision and mapping
//! assertion is bound to an immutable technical-source snapshot owned by
//! `symthaea-support`'s standards registry.
//!
//! ```text
//! ATT&CK technique != observed compromise
//! D3FEND technique != proven effective control
//! official mapping != universal effectiveness
//! framework identifier != framework-version identity
//! deprecated object != nonexistent historical object
//! mapping evidence != execution authority
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use symthaea_support::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
    SourceSnapshotIdV1, TechnicalPublisherV1, TechnicalStandardsRegistryV1,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CyberFrameworkSnapshotIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CyberMappingIdV1(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CyberFrameworkV1 {
    MitreAttack,
    MitreD3fend,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CyberDomainV1 {
    Enterprise,
    Mobile,
    Ics,
    Ot,
    Space,
    CrossDomain,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CyberObjectKindV1 {
    AttackTactic,
    AttackTechnique,
    AttackSubTechnique,
    AttackMitigation,
    AttackDetectionStrategy,
    AttackAnalytic,
    AttackDataComponent,
    D3fendTechnique,
    D3fendArtifact,
    D3fendEvent,
    D3fendWeakness,
    D3fendAgent,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CyberObjectStatusV1 {
    Active,
    Deprecated,
    Revoked,
    Historic,
    Unknown,
}

/// Exact external framework release identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CyberFrameworkSnapshotV1 {
    pub id: CyberFrameworkSnapshotIdV1,
    pub framework: CyberFrameworkV1,
    /// Publisher-native release identity such as `19.2` or `1.6.0`.
    pub framework_version: String,
    /// Exact immutable source snapshot registered in the technical-source registry.
    pub source_snapshot: SourceSnapshotIdV1,
    /// Optional upstream data-model/schema version, distinct from content release.
    pub schema_version: Option<String>,
    #[serde(default)]
    pub domains: BTreeSet<CyberDomainV1>,
}

impl CyberFrameworkSnapshotV1 {
    pub fn validate(&self) -> Result<(), CyberOntologyErrorV1> {
        require_nonempty(&self.id.0, "framework snapshot id")?;
        require_nonempty(&self.framework_version, "framework version")?;
        require_nonempty(&self.source_snapshot.0, "source snapshot id")?;
        if self
            .schema_version
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CyberOntologyErrorV1::EmptyField("framework schema version"));
        }
        for domain in &self.domains {
            if let CyberDomainV1::Other(value) = domain {
                require_nonempty(value, "custom cyber domain")?;
            }
        }
        Ok(())
    }
}

/// Object identity is exact to one framework snapshot. `T1003@ATT&CK-19.2`
/// and `T1003@ATT&CK-19.1` are therefore distinct keys even when the public ID is stable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CyberObjectKeyV1 {
    pub framework_snapshot: CyberFrameworkSnapshotIdV1,
    pub external_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CyberObjectRefV1 {
    pub key: CyberObjectKeyV1,
    pub kind: CyberObjectKindV1,
    pub domain: CyberDomainV1,
    pub status: CyberObjectStatusV1,
    /// Object-native version when the external framework provides one.
    pub object_version: Option<String>,
    pub name: String,
    pub canonical_locator: Option<String>,
}

impl CyberObjectRefV1 {
    pub fn validate(&self) -> Result<(), CyberOntologyErrorV1> {
        require_nonempty(&self.key.framework_snapshot.0, "object framework snapshot")?;
        require_nonempty(&self.key.external_id, "external cyber object id")?;
        require_nonempty(&self.name, "cyber object name")?;
        validate_object_kind(&self.kind)?;
        validate_domain(&self.domain)?;
        if self
            .object_version
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CyberOntologyErrorV1::EmptyField("object version"));
        }
        if self
            .canonical_locator
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CyberOntologyErrorV1::EmptyField("object canonical locator"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CyberMappingRelationV1 {
    Counters,
    Mitigates,
    Detects,
    Observes,
    ProducesArtifact,
    OperatesOnArtifact,
    Prescribes,
    MapsTo,
    RelatedTo,
}

/// One externally sourced relationship between exact-version cyber objects.
/// The mapping source snapshot may be distinct from either framework snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CyberMappingAssertionV1 {
    pub id: CyberMappingIdV1,
    pub from: CyberObjectKeyV1,
    pub relation: CyberMappingRelationV1,
    pub to: CyberObjectKeyV1,
    /// Immutable technical-source snapshot that asserted/published the mapping.
    pub mapping_source_snapshot: SourceSnapshotIdV1,
    /// Normalization/extraction quality only; not defense effectiveness.
    pub extraction_quality: Option<f32>,
    pub note: Option<String>,
}

impl CyberMappingAssertionV1 {
    pub fn validate(&self) -> Result<(), CyberOntologyErrorV1> {
        require_nonempty(&self.id.0, "mapping id")?;
        require_nonempty(&self.from.framework_snapshot.0, "mapping source framework")?;
        require_nonempty(&self.from.external_id, "mapping source object")?;
        require_nonempty(&self.to.framework_snapshot.0, "mapping target framework")?;
        require_nonempty(&self.to.external_id, "mapping target object")?;
        require_nonempty(&self.mapping_source_snapshot.0, "mapping evidence snapshot")?;
        if self.from == self.to {
            return Err(CyberOntologyErrorV1::SelfMapping(self.id.clone()));
        }
        if let Some(quality) = self.extraction_quality {
            if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
                return Err(CyberOntologyErrorV1::InvalidExtractionQuality(quality));
            }
        }
        if self.note.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(CyberOntologyErrorV1::EmptyField("mapping note"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DefensiveCandidateV1 {
    pub defense: CyberObjectRefV1,
    pub mapping_id: CyberMappingIdV1,
    pub relation: CyberMappingRelationV1,
    pub mapping_source_snapshot: SourceSnapshotIdV1,
    pub mapping_source_lifecycle: KnowledgeLifecycleV1,
    pub mapping_source_authority: KnowledgeAuthorityClassV1,
    pub mapping_source_stability: KnowledgeStabilityV1,
    pub extraction_quality: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefensiveCandidatePolicyV1 {
    pub include_deprecated: bool,
    pub include_revoked: bool,
    pub include_historic: bool,
}

impl Default for DefensiveCandidatePolicyV1 {
    fn default() -> Self {
        Self {
            include_deprecated: true,
            include_revoked: false,
            include_historic: true,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CyberOntologyBridgeV1 {
    framework_snapshots: BTreeMap<CyberFrameworkSnapshotIdV1, CyberFrameworkSnapshotV1>,
    objects: BTreeMap<CyberObjectKeyV1, CyberObjectRefV1>,
    mappings: BTreeMap<CyberMappingIdV1, CyberMappingAssertionV1>,
}

impl CyberOntologyBridgeV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn framework_snapshot(
        &self,
        id: &CyberFrameworkSnapshotIdV1,
    ) -> Option<&CyberFrameworkSnapshotV1> {
        self.framework_snapshots.get(id)
    }

    pub fn object(&self, key: &CyberObjectKeyV1) -> Option<&CyberObjectRefV1> {
        self.objects.get(key)
    }

    pub fn mapping(&self, id: &CyberMappingIdV1) -> Option<&CyberMappingAssertionV1> {
        self.mappings.get(id)
    }

    pub fn register_framework_snapshot(
        &mut self,
        snapshot: CyberFrameworkSnapshotV1,
        sources: &TechnicalStandardsRegistryV1,
    ) -> Result<bool, CyberOntologyErrorV1> {
        snapshot.validate()?;
        let source = sources
            .snapshot(&snapshot.source_snapshot)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownTechnicalSourceSnapshot(
                snapshot.source_snapshot.clone(),
            ))?;
        let document = sources
            .document(&source.document_id)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownTechnicalSourceDocument(
                source.document_id.clone(),
            ))?;
        if document.publisher != TechnicalPublisherV1::Mitre {
            return Err(CyberOntologyErrorV1::FrameworkSourceNotMitre(
                snapshot.source_snapshot.clone(),
            ));
        }
        if let Some(source_version) = &source.version {
            if source_version != &snapshot.framework_version {
                return Err(CyberOntologyErrorV1::FrameworkVersionMismatch {
                    framework_version: snapshot.framework_version.clone(),
                    source_version: source_version.clone(),
                });
            }
        }
        if let Some(existing) = self.framework_snapshots.get(&snapshot.id) {
            if existing == &snapshot {
                return Ok(false);
            }
            return Err(CyberOntologyErrorV1::FrameworkSnapshotIdentityConflict(
                snapshot.id,
            ));
        }
        self.framework_snapshots.insert(snapshot.id.clone(), snapshot);
        Ok(true)
    }

    pub fn register_object(
        &mut self,
        object: CyberObjectRefV1,
    ) -> Result<bool, CyberOntologyErrorV1> {
        object.validate()?;
        let snapshot = self
            .framework_snapshots
            .get(&object.key.framework_snapshot)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownFrameworkSnapshot(
                object.key.framework_snapshot.clone(),
            ))?;
        validate_kind_framework(snapshot.framework, &object.kind)?;
        if let Some(existing) = self.objects.get(&object.key) {
            if existing == &object {
                return Ok(false);
            }
            return Err(CyberOntologyErrorV1::ObjectIdentityConflict(object.key));
        }
        self.objects.insert(object.key.clone(), object);
        Ok(true)
    }

    pub fn register_mapping(
        &mut self,
        mapping: CyberMappingAssertionV1,
        sources: &TechnicalStandardsRegistryV1,
    ) -> Result<bool, CyberOntologyErrorV1> {
        mapping.validate()?;
        if !self.objects.contains_key(&mapping.from) {
            return Err(CyberOntologyErrorV1::UnknownObject(mapping.from.clone()));
        }
        if !self.objects.contains_key(&mapping.to) {
            return Err(CyberOntologyErrorV1::UnknownObject(mapping.to.clone()));
        }
        let mapping_source = sources
            .snapshot(&mapping.mapping_source_snapshot)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownTechnicalSourceSnapshot(
                mapping.mapping_source_snapshot.clone(),
            ))?;
        let document = sources
            .document(&mapping_source.document_id)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownTechnicalSourceDocument(
                mapping_source.document_id.clone(),
            ))?;
        if document.publisher != TechnicalPublisherV1::Mitre {
            return Err(CyberOntologyErrorV1::MappingSourceNotMitre(
                mapping.mapping_source_snapshot.clone(),
            ));
        }
        if let Some(existing) = self.mappings.get(&mapping.id) {
            if existing == &mapping {
                return Ok(false);
            }
            return Err(CyberOntologyErrorV1::MappingIdentityConflict(mapping.id));
        }
        self.mappings.insert(mapping.id.clone(), mapping);
        Ok(true)
    }

    /// Return externally mapped D3FEND defensive candidates for one exact ATT&CK
    /// object. This does not claim the candidate is effective in the local system.
    pub fn defensive_candidates_for_attack(
        &self,
        attack: &CyberObjectKeyV1,
        sources: &TechnicalStandardsRegistryV1,
        policy: &DefensiveCandidatePolicyV1,
    ) -> Result<Vec<DefensiveCandidateV1>, CyberOntologyErrorV1> {
        let attack_object = self
            .objects
            .get(attack)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownObject(attack.clone()))?;
        let attack_snapshot = self
            .framework_snapshots
            .get(&attack.framework_snapshot)
            .ok_or_else(|| CyberOntologyErrorV1::UnknownFrameworkSnapshot(
                attack.framework_snapshot.clone(),
            ))?;
        if attack_snapshot.framework != CyberFrameworkV1::MitreAttack
            || !matches!(
                attack_object.kind,
                CyberObjectKindV1::AttackTechnique | CyberObjectKindV1::AttackSubTechnique
            )
        {
            return Err(CyberOntologyErrorV1::NotAttackTechnique(attack.clone()));
        }

        let mut candidates = Vec::new();
        for mapping in self.mappings.values().filter(|mapping| &mapping.from == attack) {
            if !matches!(
                mapping.relation,
                CyberMappingRelationV1::Counters
                    | CyberMappingRelationV1::Mitigates
                    | CyberMappingRelationV1::Prescribes
            ) {
                continue;
            }
            let Some(defense) = self.objects.get(&mapping.to) else {
                continue;
            };
            let Some(defense_snapshot) = self
                .framework_snapshots
                .get(&defense.key.framework_snapshot)
            else {
                continue;
            };
            if defense_snapshot.framework != CyberFrameworkV1::MitreD3fend
                || defense.kind != CyberObjectKindV1::D3fendTechnique
            {
                continue;
            }
            let include = match defense.status {
                CyberObjectStatusV1::Active | CyberObjectStatusV1::Unknown => true,
                CyberObjectStatusV1::Deprecated => policy.include_deprecated,
                CyberObjectStatusV1::Revoked => policy.include_revoked,
                CyberObjectStatusV1::Historic => policy.include_historic,
            };
            if !include {
                continue;
            }
            let source = sources
                .snapshot(&mapping.mapping_source_snapshot)
                .ok_or_else(|| CyberOntologyErrorV1::UnknownTechnicalSourceSnapshot(
                    mapping.mapping_source_snapshot.clone(),
                ))?;
            candidates.push(DefensiveCandidateV1 {
                defense: defense.clone(),
                mapping_id: mapping.id.clone(),
                relation: mapping.relation,
                mapping_source_snapshot: mapping.mapping_source_snapshot.clone(),
                mapping_source_lifecycle: source.lifecycle,
                mapping_source_authority: source.authority,
                mapping_source_stability: source.stability,
                extraction_quality: mapping.extraction_quality,
            });
        }
        candidates.sort_by(|a, b| {
            a.defense
                .key
                .external_id
                .cmp(&b.defense.key.external_id)
                .then_with(|| a.mapping_id.0.cmp(&b.mapping_id.0))
        });
        Ok(candidates)
    }

    pub fn digest(&self) -> Result<String, CyberOntologyErrorV1> {
        let bytes = serde_json::to_vec(&("symthaea-cyber-ontology-bridge-v1", self))
            .map_err(|err| CyberOntologyErrorV1::Serialization(err.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

fn validate_kind_framework(
    framework: CyberFrameworkV1,
    kind: &CyberObjectKindV1,
) -> Result<(), CyberOntologyErrorV1> {
    let valid = match framework {
        CyberFrameworkV1::MitreAttack => matches!(
            kind,
            CyberObjectKindV1::AttackTactic
                | CyberObjectKindV1::AttackTechnique
                | CyberObjectKindV1::AttackSubTechnique
                | CyberObjectKindV1::AttackMitigation
                | CyberObjectKindV1::AttackDetectionStrategy
                | CyberObjectKindV1::AttackAnalytic
                | CyberObjectKindV1::AttackDataComponent
                | CyberObjectKindV1::Other(_)
        ),
        CyberFrameworkV1::MitreD3fend => matches!(
            kind,
            CyberObjectKindV1::D3fendTechnique
                | CyberObjectKindV1::D3fendArtifact
                | CyberObjectKindV1::D3fendEvent
                | CyberObjectKindV1::D3fendWeakness
                | CyberObjectKindV1::D3fendAgent
                | CyberObjectKindV1::Other(_)
        ),
    };
    if valid {
        Ok(())
    } else {
        Err(CyberOntologyErrorV1::ObjectKindFrameworkMismatch {
            framework,
            kind: kind.clone(),
        })
    }
}

fn validate_object_kind(kind: &CyberObjectKindV1) -> Result<(), CyberOntologyErrorV1> {
    if let CyberObjectKindV1::Other(value) = kind {
        require_nonempty(value, "custom cyber object kind")?;
    }
    Ok(())
}

fn validate_domain(domain: &CyberDomainV1) -> Result<(), CyberOntologyErrorV1> {
    if let CyberDomainV1::Other(value) = domain {
        require_nonempty(value, "custom cyber domain")?;
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), CyberOntologyErrorV1> {
    if value.trim().is_empty() {
        Err(CyberOntologyErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CyberOntologyErrorV1 {
    EmptyField(&'static str),
    InvalidExtractionQuality(f32),
    UnknownTechnicalSourceSnapshot(SourceSnapshotIdV1),
    UnknownTechnicalSourceDocument(symthaea_support::SourceDocumentIdV1),
    FrameworkSourceNotMitre(SourceSnapshotIdV1),
    MappingSourceNotMitre(SourceSnapshotIdV1),
    FrameworkVersionMismatch {
        framework_version: String,
        source_version: String,
    },
    FrameworkSnapshotIdentityConflict(CyberFrameworkSnapshotIdV1),
    UnknownFrameworkSnapshot(CyberFrameworkSnapshotIdV1),
    ObjectIdentityConflict(CyberObjectKeyV1),
    UnknownObject(CyberObjectKeyV1),
    ObjectKindFrameworkMismatch {
        framework: CyberFrameworkV1,
        kind: CyberObjectKindV1,
    },
    SelfMapping(CyberMappingIdV1),
    MappingIdentityConflict(CyberMappingIdV1),
    NotAttackTechnique(CyberObjectKeyV1),
    Serialization(String),
}

impl fmt::Display for CyberOntologyErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty cyber ontology field {field}"),
            Self::InvalidExtractionQuality(value) => {
                write!(f, "invalid cyber mapping extraction quality {value}")
            }
            Self::UnknownTechnicalSourceSnapshot(id) => {
                write!(f, "unknown technical source snapshot {}", id.0)
            }
            Self::UnknownTechnicalSourceDocument(id) => {
                write!(f, "unknown technical source document {}", id.0)
            }
            Self::FrameworkSourceNotMitre(id) => write!(
                f,
                "framework snapshot {} is not bound to a MITRE technical source",
                id.0
            ),
            Self::MappingSourceNotMitre(id) => write!(
                f,
                "mapping snapshot {} is not bound to a MITRE technical source",
                id.0
            ),
            Self::FrameworkVersionMismatch {
                framework_version,
                source_version,
            } => write!(
                f,
                "framework version {framework_version:?} does not match technical-source version {source_version:?}"
            ),
            Self::FrameworkSnapshotIdentityConflict(id) => {
                write!(f, "cyber framework snapshot identity {} was rebound", id.0)
            }
            Self::UnknownFrameworkSnapshot(id) => {
                write!(f, "unknown cyber framework snapshot {}", id.0)
            }
            Self::ObjectIdentityConflict(key) => write!(
                f,
                "cyber object identity {}@{} was rebound",
                key.external_id, key.framework_snapshot.0
            ),
            Self::UnknownObject(key) => write!(
                f,
                "unknown cyber object {}@{}",
                key.external_id, key.framework_snapshot.0
            ),
            Self::ObjectKindFrameworkMismatch { framework, kind } => write!(
                f,
                "cyber object kind {kind:?} is invalid for framework {framework:?}"
            ),
            Self::SelfMapping(id) => write!(f, "cyber mapping {} maps an object to itself", id.0),
            Self::MappingIdentityConflict(id) => {
                write!(f, "cyber mapping identity {} was rebound", id.0)
            }
            Self::NotAttackTechnique(key) => write!(
                f,
                "cyber object {}@{} is not an ATT&CK technique/sub-technique",
                key.external_id, key.framework_snapshot.0
            ),
            Self::Serialization(message) => {
                write!(f, "cyber ontology serialization failed: {message}")
            }
        }
    }
}

impl Error for CyberOntologyErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_support::{
        KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
        SourceCaptureV1, SourceDocumentIdV1, SourceDocumentKindV1,
        TechnicalSourceDocumentV1, TechnicalSourceSnapshotV1,
    };

    fn source_registry() -> TechnicalStandardsRegistryV1 {
        let mut sources = TechnicalStandardsRegistryV1::new();
        for (doc_id, snap_id, title, version) in [
            ("mitre-attack", "mitre-attack-19.2", "MITRE ATT&CK", "19.2"),
            ("mitre-d3fend", "mitre-d3fend-1.6.0", "MITRE D3FEND", "1.6.0"),
            (
                "mitre-d3fend-map",
                "mitre-d3fend-map-1.6.0",
                "MITRE D3FEND mappings",
                "1.6.0",
            ),
        ] {
            sources
                .register_document(TechnicalSourceDocumentV1 {
                    id: SourceDocumentIdV1(doc_id.into()),
                    publisher: TechnicalPublisherV1::Mitre,
                    kind: SourceDocumentKindV1::SecurityKnowledgeBase,
                    title: title.into(),
                    canonical_ref: title.into(),
                    canonical_locator: None,
                })
                .unwrap();
            sources
                .register_snapshot(TechnicalSourceSnapshotV1 {
                    id: SourceSnapshotIdV1(snap_id.into()),
                    document_id: SourceDocumentIdV1(doc_id.into()),
                    version: Some(version.into()),
                    lifecycle: KnowledgeLifecycleV1::Active,
                    authority: KnowledgeAuthorityClassV1::SecurityKnowledgeBase,
                    stability: KnowledgeStabilityV1::Stable,
                    published_at_unix_ms: None,
                    source_updated_at_unix_ms: None,
                    fetched_at_unix_ms: 1,
                    capture: SourceCaptureV1::MetadataDigest {
                        algorithm: "sha256".into(),
                        digest: format!("digest-{snap_id}"),
                    },
                    relations: BTreeSet::new(),
                })
                .unwrap();
        }
        sources
    }

    fn bridge(sources: &TechnicalStandardsRegistryV1) -> CyberOntologyBridgeV1 {
        let mut bridge = CyberOntologyBridgeV1::new();
        bridge
            .register_framework_snapshot(
                CyberFrameworkSnapshotV1 {
                    id: CyberFrameworkSnapshotIdV1("attack-19.2".into()),
                    framework: CyberFrameworkV1::MitreAttack,
                    framework_version: "19.2".into(),
                    source_snapshot: SourceSnapshotIdV1("mitre-attack-19.2".into()),
                    schema_version: Some("3.3.0".into()),
                    domains: BTreeSet::from([CyberDomainV1::Enterprise]),
                },
                sources,
            )
            .unwrap();
        bridge
            .register_framework_snapshot(
                CyberFrameworkSnapshotV1 {
                    id: CyberFrameworkSnapshotIdV1("d3fend-1.6.0".into()),
                    framework: CyberFrameworkV1::MitreD3fend,
                    framework_version: "1.6.0".into(),
                    source_snapshot: SourceSnapshotIdV1("mitre-d3fend-1.6.0".into()),
                    schema_version: None,
                    domains: BTreeSet::from([CyberDomainV1::CrossDomain]),
                },
                sources,
            )
            .unwrap();
        bridge
    }

    fn attack_key(version: &str) -> CyberObjectKeyV1 {
        CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1(version.into()),
            external_id: "TTEST".into(),
        }
    }

    #[test]
    fn framework_snapshot_requires_exact_mitre_source_version() {
        let sources = source_registry();
        let mut bridge = CyberOntologyBridgeV1::new();
        let result = bridge.register_framework_snapshot(
            CyberFrameworkSnapshotV1 {
                id: CyberFrameworkSnapshotIdV1("attack-wrong".into()),
                framework: CyberFrameworkV1::MitreAttack,
                framework_version: "19.1".into(),
                source_snapshot: SourceSnapshotIdV1("mitre-attack-19.2".into()),
                schema_version: None,
                domains: BTreeSet::new(),
            },
            &sources,
        );
        assert!(matches!(
            result,
            Err(CyberOntologyErrorV1::FrameworkVersionMismatch { .. })
        ));
    }

    #[test]
    fn same_public_attack_id_in_two_snapshots_is_not_same_identity() {
        let first = CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1("attack-19.1".into()),
            external_id: "T1003".into(),
        };
        let second = CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1("attack-19.2".into()),
            external_id: "T1003".into(),
        };
        assert_ne!(first, second);
    }

    #[test]
    fn exact_mapping_returns_candidate_without_claiming_effectiveness() {
        let sources = source_registry();
        let mut bridge = bridge(&sources);
        let attack = attack_key("attack-19.2");
        let defense = CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1("d3fend-1.6.0".into()),
            external_id: "D3-TEST".into(),
        };
        bridge
            .register_object(CyberObjectRefV1 {
                key: attack.clone(),
                kind: CyberObjectKindV1::AttackTechnique,
                domain: CyberDomainV1::Enterprise,
                status: CyberObjectStatusV1::Active,
                object_version: Some("1.0".into()),
                name: "Synthetic attack technique".into(),
                canonical_locator: None,
            })
            .unwrap();
        bridge
            .register_object(CyberObjectRefV1 {
                key: defense.clone(),
                kind: CyberObjectKindV1::D3fendTechnique,
                domain: CyberDomainV1::CrossDomain,
                status: CyberObjectStatusV1::Active,
                object_version: Some("1.0".into()),
                name: "Synthetic defensive technique".into(),
                canonical_locator: None,
            })
            .unwrap();
        bridge
            .register_mapping(
                CyberMappingAssertionV1 {
                    id: CyberMappingIdV1("map-1".into()),
                    from: attack.clone(),
                    relation: CyberMappingRelationV1::Counters,
                    to: defense,
                    mapping_source_snapshot: SourceSnapshotIdV1(
                        "mitre-d3fend-map-1.6.0".into(),
                    ),
                    extraction_quality: Some(1.0),
                    note: None,
                },
                &sources,
            )
            .unwrap();
        let candidates = bridge
            .defensive_candidates_for_attack(
                &attack,
                &sources,
                &DefensiveCandidatePolicyV1::default(),
            )
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].defense.key.external_id, "D3-TEST");
        assert_eq!(
            candidates[0].mapping_source_authority,
            KnowledgeAuthorityClassV1::SecurityKnowledgeBase
        );
    }

    #[test]
    fn revoked_defense_is_excluded_by_default_but_history_can_be_requested() {
        let sources = source_registry();
        let mut bridge = bridge(&sources);
        let attack = attack_key("attack-19.2");
        let defense = CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1("d3fend-1.6.0".into()),
            external_id: "D3-OLD".into(),
        };
        bridge
            .register_object(CyberObjectRefV1 {
                key: attack.clone(),
                kind: CyberObjectKindV1::AttackTechnique,
                domain: CyberDomainV1::Enterprise,
                status: CyberObjectStatusV1::Active,
                object_version: None,
                name: "Synthetic attack".into(),
                canonical_locator: None,
            })
            .unwrap();
        bridge
            .register_object(CyberObjectRefV1 {
                key: defense.clone(),
                kind: CyberObjectKindV1::D3fendTechnique,
                domain: CyberDomainV1::CrossDomain,
                status: CyberObjectStatusV1::Revoked,
                object_version: None,
                name: "Revoked defense".into(),
                canonical_locator: None,
            })
            .unwrap();
        bridge
            .register_mapping(
                CyberMappingAssertionV1 {
                    id: CyberMappingIdV1("map-old".into()),
                    from: attack.clone(),
                    relation: CyberMappingRelationV1::Mitigates,
                    to: defense,
                    mapping_source_snapshot: SourceSnapshotIdV1(
                        "mitre-d3fend-map-1.6.0".into(),
                    ),
                    extraction_quality: None,
                    note: None,
                },
                &sources,
            )
            .unwrap();
        assert!(bridge
            .defensive_candidates_for_attack(
                &attack,
                &sources,
                &DefensiveCandidatePolicyV1::default(),
            )
            .unwrap()
            .is_empty());
        let history = bridge
            .defensive_candidates_for_attack(
                &attack,
                &sources,
                &DefensiveCandidatePolicyV1 {
                    include_revoked: true,
                    ..DefensiveCandidatePolicyV1::default()
                },
            )
            .unwrap();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn mapping_identity_cannot_be_rebound() {
        let sources = source_registry();
        let mut bridge = bridge(&sources);
        let attack = attack_key("attack-19.2");
        let defense = CyberObjectKeyV1 {
            framework_snapshot: CyberFrameworkSnapshotIdV1("d3fend-1.6.0".into()),
            external_id: "D3-TEST".into(),
        };
        for (key, kind, domain, name) in [
            (
                attack.clone(),
                CyberObjectKindV1::AttackTechnique,
                CyberDomainV1::Enterprise,
                "attack",
            ),
            (
                defense.clone(),
                CyberObjectKindV1::D3fendTechnique,
                CyberDomainV1::CrossDomain,
                "defense",
            ),
        ] {
            bridge
                .register_object(CyberObjectRefV1 {
                    key,
                    kind,
                    domain,
                    status: CyberObjectStatusV1::Active,
                    object_version: None,
                    name: name.into(),
                    canonical_locator: None,
                })
                .unwrap();
        }
        let first = CyberMappingAssertionV1 {
            id: CyberMappingIdV1("stable-map-id".into()),
            from: attack,
            relation: CyberMappingRelationV1::Counters,
            to: defense,
            mapping_source_snapshot: SourceSnapshotIdV1("mitre-d3fend-map-1.6.0".into()),
            extraction_quality: Some(1.0),
            note: None,
        };
        bridge.register_mapping(first.clone(), &sources).unwrap();
        let mut rebound = first;
        rebound.relation = CyberMappingRelationV1::Detects;
        assert!(matches!(
            bridge.register_mapping(rebound, &sources),
            Err(CyberOntologyErrorV1::MappingIdentityConflict(_))
        ));
    }
}
