// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Extensible manufacturing-process identity and extension-profile semantics.
//!
//! This crate deliberately stops before quantities, recipes, CAM, scheduling, or machine
//! execution. It establishes the smallest stable kernel needed to name a manufacturing process
//! without turning every process technology into one closed Rust enum.
//!
//! Core theorem:
//!
//! ```text
//! process family reference
//! != extension profile resolved
//! != process capability
//! != recipe
//! != machine execution authority
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

const PROCESS_ID_DOMAIN: &str = "symthaea-manufacturing-process::process-definition-v1";
const EXTENSION_ID_DOMAIN: &str = "symthaea-manufacturing-process::extension-profile-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManufacturingProcessError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} contains leading/trailing whitespace or control characters")]
    NonCanonicalToken { field: &'static str },
    #[error("process-family reference must be namespaced with ':'")]
    UnnamespacedProcessFamily,
    #[error("digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidSha256,
    #[error("process definition must contain at least one process family")]
    MissingProcessFamily,
    #[error("process definition must contain at least one operation mode")]
    MissingOperationMode,
    #[error("process definition must contain at least one transformation effect")]
    MissingTransformationEffect,
    #[error("duplicate process-family reference: {0}")]
    DuplicateProcessFamily(String),
    #[error("duplicate operation mode: {0}")]
    DuplicateOperationMode(String),
    #[error("duplicate transformation effect: {0}")]
    DuplicateTransformationEffect(String),
    #[error("duplicate extension-profile reference: {0}")]
    DuplicateExtensionProfileRef(String),
    #[error("duplicate extension-profile semantic coordinate: {0}")]
    DuplicateExtensionProfileCoordinate(String),
    #[error("duplicate supersedes reference: {0}")]
    DuplicateSupersedes(String),
}

fn validate_token(field: &'static str, value: &str) -> Result<(), ManufacturingProcessError> {
    if value.is_empty() {
        return Err(ManufacturingProcessError::EmptyField { field });
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(ManufacturingProcessError::NonCanonicalToken { field });
    }
    Ok(())
}

fn validate_sha256_hex(value: &str) -> Result<(), ManufacturingProcessError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ManufacturingProcessError::InvalidSha256);
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProcessDefinitionId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExtensionProfileId(pub String);

/// Namespaced process-family identity.
///
/// Examples:
/// - `iso-astm-52900:material-extrusion`
/// - `luminous:cnc-milling/v1`
/// - `org.example:novel-process/v2`
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProcessFamilyRef(pub String);

impl ProcessFamilyRef {
    pub fn validate(&self) -> Result<(), ManufacturingProcessError> {
        validate_token("process_family_ref", &self.0)?;
        let Some((namespace, local)) = self.0.split_once(':') else {
            return Err(ManufacturingProcessError::UnnamespacedProcessFamily);
        };
        validate_token("process_family_ref.namespace", namespace)?;
        validate_token("process_family_ref.local", local)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ManufacturingOperationModeV1 {
    Discrete,
    Batch,
    Continuous,
    Assembly,
    DisassemblyRepairRework,
    InspectionMeasurement,
    MaterialHandlingStorage,
}

impl ManufacturingOperationModeV1 {
    fn canonical_tag(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Batch => "batch",
            Self::Continuous => "continuous",
            Self::Assembly => "assembly",
            Self::DisassemblyRepairRework => "disassembly-repair-rework",
            Self::InspectionMeasurement => "inspection-measurement",
            Self::MaterialHandlingStorage => "material-handling-storage",
        }
    }
}

/// Broad semantic effects of a process. This intentionally does not enumerate process families.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TransformationEffectV1 {
    MaterialAddition,
    MaterialRemoval,
    PlasticDeformation,
    SolidificationOrMolding,
    JoiningOrBonding,
    SeparationOrDisassembly,
    HeatTreatment,
    DensificationOrSintering,
    ChemicalTransformation,
    ElectrochemicalTransformation,
    SurfaceModification,
    ThinFilmDeposition,
    MicrostructureModification,
    SemiconductorDopingOrActivation,
    PatternTransfer,
    CleaningOrPreparation,
    Assembly,
    InspectionOrMeasurement,
    NonDestructiveTest,
    ElectricalTest,
    LeakOrVacuumTest,
    ConditioningOrCure,
    Transport,
    Storage,
    Packaging,
    Custom(String),
}

impl TransformationEffectV1 {
    fn validate(&self) -> Result<(), ManufacturingProcessError> {
        if let Self::Custom(value) = self {
            validate_token("transformation_effect.custom", value)?;
        }
        Ok(())
    }

    fn canonical_tag(&self) -> String {
        match self {
            Self::MaterialAddition => "material-addition".into(),
            Self::MaterialRemoval => "material-removal".into(),
            Self::PlasticDeformation => "plastic-deformation".into(),
            Self::SolidificationOrMolding => "solidification-or-molding".into(),
            Self::JoiningOrBonding => "joining-or-bonding".into(),
            Self::SeparationOrDisassembly => "separation-or-disassembly".into(),
            Self::HeatTreatment => "heat-treatment".into(),
            Self::DensificationOrSintering => "densification-or-sintering".into(),
            Self::ChemicalTransformation => "chemical-transformation".into(),
            Self::ElectrochemicalTransformation => "electrochemical-transformation".into(),
            Self::SurfaceModification => "surface-modification".into(),
            Self::ThinFilmDeposition => "thin-film-deposition".into(),
            Self::MicrostructureModification => "microstructure-modification".into(),
            Self::SemiconductorDopingOrActivation => "semiconductor-doping-or-activation".into(),
            Self::PatternTransfer => "pattern-transfer".into(),
            Self::CleaningOrPreparation => "cleaning-or-preparation".into(),
            Self::Assembly => "assembly".into(),
            Self::InspectionOrMeasurement => "inspection-or-measurement".into(),
            Self::NonDestructiveTest => "non-destructive-test".into(),
            Self::ElectricalTest => "electrical-test".into(),
            Self::LeakOrVacuumTest => "leak-or-vacuum-test".into(),
            Self::ConditioningOrCure => "conditioning-or-cure".into(),
            Self::Transport => "transport".into(),
            Self::Storage => "storage".into(),
            Self::Packaging => "packaging".into(),
            Self::Custom(value) => format!("custom:{value}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExtensionLifecycleV1 {
    Active,
    DeprecatedReadable,
}

/// Exact extension-profile reference expected by a process definition.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExtensionProfileRefV1 {
    pub namespace: String,
    pub profile_name: String,
    pub semantic_version: String,
    pub publisher_id: String,
    pub schema_sha256: String,
}

impl ExtensionProfileRefV1 {
    pub fn validate(&self) -> Result<(), ManufacturingProcessError> {
        validate_token("extension_ref.namespace", &self.namespace)?;
        validate_token("extension_ref.profile_name", &self.profile_name)?;
        validate_token("extension_ref.semantic_version", &self.semantic_version)?;
        validate_token("extension_ref.publisher_id", &self.publisher_id)?;
        validate_sha256_hex(&self.schema_sha256)
    }

    fn semantic_coordinate(&self) -> String {
        format!(
            "{}::{}::{}::{}",
            self.namespace, self.profile_name, self.semantic_version, self.publisher_id
        )
    }

    pub fn expected_profile_id(&self) -> Result<ExtensionProfileId, ManufacturingProcessError> {
        self.validate()?;
        Ok(extension_profile_id(
            &self.namespace,
            &self.profile_name,
            &self.semantic_version,
            &self.publisher_id,
            &self.schema_sha256,
        ))
    }
}

/// Resolvable schema/profile metadata. `schema_locator` is navigation metadata; exact schema
/// bytes are represented by `schema_sha256`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessExtensionProfileV1 {
    pub namespace: String,
    pub profile_name: String,
    pub semantic_version: String,
    pub publisher_id: String,
    pub schema_sha256: String,
    pub schema_locator: Option<String>,
    pub lifecycle: ExtensionLifecycleV1,
    #[serde(default)]
    pub supersedes: Vec<ExtensionProfileId>,
}

impl ProcessExtensionProfileV1 {
    pub fn validate(&self) -> Result<(), ManufacturingProcessError> {
        validate_token("extension_profile.namespace", &self.namespace)?;
        validate_token("extension_profile.profile_name", &self.profile_name)?;
        validate_token("extension_profile.semantic_version", &self.semantic_version)?;
        validate_token("extension_profile.publisher_id", &self.publisher_id)?;
        validate_sha256_hex(&self.schema_sha256)?;
        let mut seen = BTreeSet::new();
        for id in &self.supersedes {
            validate_token("extension_profile.supersedes", &id.0)?;
            if !seen.insert(id.0.clone()) {
                return Err(ManufacturingProcessError::DuplicateSupersedes(id.0.clone()));
            }
        }
        Ok(())
    }

    fn semantic_coordinate(&self) -> String {
        format!(
            "{}::{}::{}::{}",
            self.namespace, self.profile_name, self.semantic_version, self.publisher_id
        )
    }

    pub fn profile_id(&self) -> Result<ExtensionProfileId, ManufacturingProcessError> {
        self.validate()?;
        Ok(extension_profile_id(
            &self.namespace,
            &self.profile_name,
            &self.semantic_version,
            &self.publisher_id,
            &self.schema_sha256,
        ))
    }

    pub fn as_ref(&self) -> Result<ExtensionProfileRefV1, ManufacturingProcessError> {
        self.validate()?;
        Ok(ExtensionProfileRefV1 {
            namespace: self.namespace.clone(),
            profile_name: self.profile_name.clone(),
            semantic_version: self.semantic_version.clone(),
            publisher_id: self.publisher_id.clone(),
            schema_sha256: self.schema_sha256.clone(),
        })
    }
}

fn extension_profile_id(
    namespace: &str,
    profile_name: &str,
    semantic_version: &str,
    publisher_id: &str,
    schema_sha256: &str,
) -> ExtensionProfileId {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, EXTENSION_ID_DOMAIN);
    hash_field(&mut hasher, namespace);
    hash_field(&mut hasher, profile_name);
    hash_field(&mut hasher, semantic_version);
    hash_field(&mut hasher, publisher_id);
    hash_field(&mut hasher, schema_sha256);
    ExtensionProfileId(hasher.finalize().to_hex().to_string())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionResolutionV1 {
    ResolvedExact { profile_id: ExtensionProfileId },
    DeprecatedReadable { profile_id: ExtensionProfileId },
    Unresolved,
    SchemaDigestMismatch { expected: String, available: String },
}

/// Data-only extension registry. Resolution never loads or executes plugin code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtensionRegistryV1 {
    #[serde(default)]
    pub profiles: Vec<ProcessExtensionProfileV1>,
}

impl ExtensionRegistryV1 {
    pub fn validate(&self) -> Result<(), ManufacturingProcessError> {
        let mut coordinates = BTreeSet::new();
        for profile in &self.profiles {
            profile.validate()?;
            let coordinate = profile.semantic_coordinate();
            if !coordinates.insert(coordinate.clone()) {
                return Err(ManufacturingProcessError::DuplicateExtensionProfileCoordinate(
                    coordinate,
                ));
            }
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        reference: &ExtensionProfileRefV1,
    ) -> Result<ExtensionResolutionV1, ManufacturingProcessError> {
        self.validate()?;
        reference.validate()?;
        let coordinate = reference.semantic_coordinate();
        let Some(profile) = self
            .profiles
            .iter()
            .find(|profile| profile.semantic_coordinate() == coordinate)
        else {
            return Ok(ExtensionResolutionV1::Unresolved);
        };

        if profile.schema_sha256 != reference.schema_sha256 {
            return Ok(ExtensionResolutionV1::SchemaDigestMismatch {
                expected: reference.schema_sha256.clone(),
                available: profile.schema_sha256.clone(),
            });
        }

        let profile_id = profile.profile_id()?;
        Ok(match profile.lifecycle {
            ExtensionLifecycleV1::Active => ExtensionResolutionV1::ResolvedExact { profile_id },
            ExtensionLifecycleV1::DeprecatedReadable => {
                ExtensionResolutionV1::DeprecatedReadable { profile_id }
            }
        })
    }
}

/// Canonical process definition independent of any machine, supplier, recipe, or execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessDefinitionV1 {
    pub namespace: String,
    pub process_name: String,
    pub semantic_version: String,
    #[serde(default)]
    pub family_refs: Vec<ProcessFamilyRef>,
    #[serde(default)]
    pub operation_modes: Vec<ManufacturingOperationModeV1>,
    #[serde(default)]
    pub transformation_effects: Vec<TransformationEffectV1>,
    #[serde(default)]
    pub extension_profile_refs: Vec<ExtensionProfileRefV1>,
    pub display_name: Option<String>,
    pub description: Option<String>,
}

impl ProcessDefinitionV1 {
    pub fn validate(&self) -> Result<(), ManufacturingProcessError> {
        validate_token("process.namespace", &self.namespace)?;
        validate_token("process.process_name", &self.process_name)?;
        validate_token("process.semantic_version", &self.semantic_version)?;

        if self.family_refs.is_empty() {
            return Err(ManufacturingProcessError::MissingProcessFamily);
        }
        if self.operation_modes.is_empty() {
            return Err(ManufacturingProcessError::MissingOperationMode);
        }
        if self.transformation_effects.is_empty() {
            return Err(ManufacturingProcessError::MissingTransformationEffect);
        }

        let mut family_refs = BTreeSet::new();
        for family in &self.family_refs {
            family.validate()?;
            if !family_refs.insert(family.0.clone()) {
                return Err(ManufacturingProcessError::DuplicateProcessFamily(
                    family.0.clone(),
                ));
            }
        }

        let mut modes = BTreeSet::new();
        for mode in &self.operation_modes {
            let key = mode.canonical_tag().to_string();
            if !modes.insert(key.clone()) {
                return Err(ManufacturingProcessError::DuplicateOperationMode(key));
            }
        }

        let mut effects = BTreeSet::new();
        for effect in &self.transformation_effects {
            effect.validate()?;
            let key = effect.canonical_tag();
            if !effects.insert(key.clone()) {
                return Err(ManufacturingProcessError::DuplicateTransformationEffect(
                    key,
                ));
            }
        }

        let mut extension_refs = BTreeSet::new();
        for reference in &self.extension_profile_refs {
            reference.validate()?;
            let key = format!(
                "{}::{}",
                reference.semantic_coordinate(), reference.schema_sha256
            );
            if !extension_refs.insert(key.clone()) {
                return Err(ManufacturingProcessError::DuplicateExtensionProfileRef(key));
            }
        }
        Ok(())
    }

    pub fn process_id(&self) -> Result<ProcessDefinitionId, ManufacturingProcessError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, PROCESS_ID_DOMAIN);
        hash_field(&mut hasher, &self.namespace);
        hash_field(&mut hasher, &self.process_name);
        hash_field(&mut hasher, &self.semantic_version);

        let mut families = self
            .family_refs
            .iter()
            .map(|family| family.0.clone())
            .collect::<Vec<_>>();
        families.sort();
        for family in families {
            hash_field(&mut hasher, &family);
        }

        let mut modes = self
            .operation_modes
            .iter()
            .map(|mode| mode.canonical_tag().to_string())
            .collect::<Vec<_>>();
        modes.sort();
        for mode in modes {
            hash_field(&mut hasher, &mode);
        }

        let mut effects = self
            .transformation_effects
            .iter()
            .map(TransformationEffectV1::canonical_tag)
            .collect::<Vec<_>>();
        effects.sort();
        for effect in effects {
            hash_field(&mut hasher, &effect);
        }

        let mut extension_refs = self.extension_profile_refs.clone();
        extension_refs.sort();
        for reference in extension_refs {
            hash_field(&mut hasher, &reference.namespace);
            hash_field(&mut hasher, &reference.profile_name);
            hash_field(&mut hasher, &reference.semantic_version);
            hash_field(&mut hasher, &reference.publisher_id);
            hash_field(&mut hasher, &reference.schema_sha256);
        }

        Ok(ProcessDefinitionId(hasher.finalize().to_hex().to_string()))
    }

    pub fn resolve_extensions(
        &self,
        registry: &ExtensionRegistryV1,
    ) -> Result<Vec<ExtensionResolutionV1>, ManufacturingProcessError> {
        self.validate()?;
        self.extension_profile_refs
            .iter()
            .map(|reference| registry.resolve(reference))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn profile() -> ProcessExtensionProfileV1 {
        ProcessExtensionProfileV1 {
            namespace: "luminous".into(),
            profile_name: "cnc-milling".into(),
            semantic_version: "1.0.0".into(),
            publisher_id: "luminous-dynamics".into(),
            schema_sha256: digest('a'),
            schema_locator: Some("https://example.invalid/cnc-milling-v1.schema.json".into()),
            lifecycle: ExtensionLifecycleV1::Active,
            supersedes: vec![],
        }
    }

    fn process() -> ProcessDefinitionV1 {
        ProcessDefinitionV1 {
            namespace: "luminous".into(),
            process_name: "precision-milling".into(),
            semantic_version: "1.0.0".into(),
            family_refs: vec![ProcessFamilyRef("luminous:cnc-milling/v1".into())],
            operation_modes: vec![ManufacturingOperationModeV1::Discrete],
            transformation_effects: vec![TransformationEffectV1::MaterialRemoval],
            extension_profile_refs: vec![],
            display_name: Some("Precision milling".into()),
            description: Some("Navigation prose".into()),
        }
    }

    #[test]
    fn display_metadata_does_not_change_process_identity() {
        let a = process();
        let mut b = a.clone();
        b.display_name = Some("Different label".into());
        b.description = Some("Different prose".into());
        assert_eq!(a.process_id().unwrap(), b.process_id().unwrap());
    }

    #[test]
    fn semantic_version_changes_process_identity() {
        let a = process();
        let mut b = a.clone();
        b.semantic_version = "2.0.0".into();
        assert_ne!(a.process_id().unwrap(), b.process_id().unwrap());
    }

    #[test]
    fn family_ref_changes_process_identity() {
        let a = process();
        let mut b = a.clone();
        b.family_refs = vec![ProcessFamilyRef("iso-astm-52900:material-extrusion".into())];
        assert_ne!(a.process_id().unwrap(), b.process_id().unwrap());
    }

    #[test]
    fn duplicate_family_effect_and_mode_reject() {
        let mut p = process();
        p.family_refs.push(p.family_refs[0].clone());
        assert!(matches!(
            p.validate(),
            Err(ManufacturingProcessError::DuplicateProcessFamily(_))
        ));

        let mut p = process();
        p.transformation_effects
            .push(TransformationEffectV1::MaterialRemoval);
        assert!(matches!(
            p.validate(),
            Err(ManufacturingProcessError::DuplicateTransformationEffect(_))
        ));

        let mut p = process();
        p.operation_modes
            .push(ManufacturingOperationModeV1::Discrete);
        assert!(matches!(
            p.validate(),
            Err(ManufacturingProcessError::DuplicateOperationMode(_))
        ));
    }

    #[test]
    fn exact_profile_ref_resolves() {
        let profile = profile();
        let reference = profile.as_ref().unwrap();
        let registry = ExtensionRegistryV1 {
            profiles: vec![profile.clone()],
        };
        assert_eq!(
            registry.resolve(&reference).unwrap(),
            ExtensionResolutionV1::ResolvedExact {
                profile_id: profile.profile_id().unwrap()
            }
        );
    }

    #[test]
    fn unknown_profile_is_unresolved_not_trusted() {
        let mut reference = profile().as_ref().unwrap();
        reference.profile_name = "unknown-process".into();
        let registry = ExtensionRegistryV1 {
            profiles: vec![profile()],
        };
        assert_eq!(
            registry.resolve(&reference).unwrap(),
            ExtensionResolutionV1::Unresolved
        );
    }

    #[test]
    fn schema_digest_mismatch_is_explicit() {
        let profile = profile();
        let mut reference = profile.as_ref().unwrap();
        reference.schema_sha256 = digest('b');
        let registry = ExtensionRegistryV1 {
            profiles: vec![profile],
        };
        assert_eq!(
            registry.resolve(&reference).unwrap(),
            ExtensionResolutionV1::SchemaDigestMismatch {
                expected: digest('b'),
                available: digest('a')
            }
        );
    }

    #[test]
    fn deprecated_profile_remains_readable_but_distinct() {
        let mut profile = profile();
        profile.lifecycle = ExtensionLifecycleV1::DeprecatedReadable;
        let reference = profile.as_ref().unwrap();
        let registry = ExtensionRegistryV1 {
            profiles: vec![profile.clone()],
        };
        assert_eq!(
            registry.resolve(&reference).unwrap(),
            ExtensionResolutionV1::DeprecatedReadable {
                profile_id: profile.profile_id().unwrap()
            }
        );
    }

    #[test]
    fn semantic_collection_order_does_not_change_process_identity() {
        let profile_a = profile();
        let mut profile_b = profile();
        profile_b.profile_name = "fixture-profile".into();
        profile_b.schema_sha256 = digest('b');

        let mut a = process();
        a.family_refs = vec![
            ProcessFamilyRef("luminous:cnc-milling/v1".into()),
            ProcessFamilyRef("org.example:novel-process/v2".into()),
        ];
        a.operation_modes = vec![
            ManufacturingOperationModeV1::Discrete,
            ManufacturingOperationModeV1::InspectionMeasurement,
        ];
        a.transformation_effects = vec![
            TransformationEffectV1::MaterialRemoval,
            TransformationEffectV1::InspectionOrMeasurement,
        ];
        a.extension_profile_refs = vec![profile_a.as_ref().unwrap(), profile_b.as_ref().unwrap()];

        let mut b = a.clone();
        b.family_refs.reverse();
        b.operation_modes.reverse();
        b.transformation_effects.reverse();
        b.extension_profile_refs.reverse();

        assert_eq!(a.process_id().unwrap(), b.process_id().unwrap());
    }

    #[test]
    fn duplicate_profile_coordinate_rejects_even_when_schema_differs() {
        let a = profile();
        let mut b = a.clone();
        b.schema_sha256 = digest('b');
        let registry = ExtensionRegistryV1 {
            profiles: vec![a, b],
        };
        assert!(matches!(
            registry.validate(),
            Err(ManufacturingProcessError::DuplicateExtensionProfileCoordinate(_))
        ));
    }

    #[test]
    fn serde_round_trip_preserves_process_identity_and_unresolved_resolution() {
        let mut p = process();
        p.extension_profile_refs = vec![profile().as_ref().unwrap()];
        let encoded = serde_json::to_string(&p).unwrap();
        let decoded: ProcessDefinitionV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(p.process_id().unwrap(), decoded.process_id().unwrap());

        let empty_registry = ExtensionRegistryV1 { profiles: vec![] };
        assert_eq!(
            decoded.resolve_extensions(&empty_registry).unwrap(),
            vec![ExtensionResolutionV1::Unresolved]
        );
    }

    #[test]
    fn namespaced_family_refs_are_extensible_without_core_enum_changes() {
        let mut p = process();
        p.family_refs = vec![
            ProcessFamilyRef("iso-astm-52900:material-extrusion".into()),
            ProcessFamilyRef("luminous:cnc-milling/v1".into()),
            ProcessFamilyRef("org.example:novel-process/v2".into()),
        ];
        assert!(p.validate().is_ok());
    }
}
