// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data-only manufacturing process packs layered over `symthaea-manufacturing-process`.
//!
//! Core theorem:
//!
//! ```text
//! process profile represented
//! != process capability established
//! != recipe admitted
//! != machine program prepared
//! != execution authorized
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_manufacturing_process::{
    ManufacturingOperationModeV1, ProcessDefinitionId, ProcessDefinitionV1, ProcessFamilyRef,
    TransformationEffectV1,
};
use thiserror::Error;

const PACK_ID_DOMAIN: &str = "symthaea-manufacturing-process-packs::pack-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProcessPackError {
    #[error("{0} must be a canonical non-empty token")]
    NonCanonical(&'static str),
    #[error("process pack requires at least one entry")]
    EmptyPack,
    #[error("duplicate process-pack entry key: {0}")]
    DuplicateEntryKey(String),
    #[error("duplicate process definition id in pack: {0}")]
    DuplicateProcessId(String),
    #[error("duplicate reference in {field}: {value}")]
    DuplicateReference { field: &'static str, value: String },
    #[error("underlying process definition is invalid: {0}")]
    InvalidProcess(String),
}

fn canonical(field: &'static str, value: &str) -> Result<(), ProcessPackError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ProcessPackError::NonCanonical(field));
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn validate_refs(field: &'static str, refs: &[String]) -> Result<(), ProcessPackError> {
    let mut seen = BTreeSet::new();
    for value in refs {
        canonical(field, value)?;
        if !seen.insert(value.clone()) {
            return Err(ProcessPackError::DuplicateReference {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessMaturityV1 {
    EstablishedIndustrial,
    AdvancedIndustrial,
    SpecializedIndustrial,
    ResearchDemonstrated,
    ExperimentalConcept,
}

impl ProcessMaturityV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::EstablishedIndustrial => "established-industrial",
            Self::AdvancedIndustrial => "advanced-industrial",
            Self::SpecializedIndustrial => "specialized-industrial",
            Self::ResearchDemonstrated => "research-demonstrated",
            Self::ExperimentalConcept => "experimental-concept",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessPackKindV1 {
    AdvancedHighLeverage,
    ResearchAndExotic,
}

impl ProcessPackKindV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::AdvancedHighLeverage => "advanced-high-leverage",
            Self::ResearchAndExotic => "research-and-exotic",
        }
    }
}

/// Maximum authority of this crate. Packs describe engineering semantics only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackAuthorityCeilingV1 {
    RepresentationAndPlanningMetadataOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessPackEntryV1 {
    pub key: String,
    pub maturity: ProcessMaturityV1,
    pub process: ProcessDefinitionV1,
    #[serde(default)]
    pub required_domain_refs: Vec<String>,
    #[serde(default)]
    pub safety_profile_refs: Vec<String>,
    #[serde(default)]
    pub metrology_profile_refs: Vec<String>,
}

impl ProcessPackEntryV1 {
    pub fn validate(&self) -> Result<(), ProcessPackError> {
        canonical("entry.key", &self.key)?;
        self.process
            .validate()
            .map_err(|err| ProcessPackError::InvalidProcess(err.to_string()))?;
        validate_refs("entry.required_domain_refs", &self.required_domain_refs)?;
        validate_refs("entry.safety_profile_refs", &self.safety_profile_refs)?;
        validate_refs("entry.metrology_profile_refs", &self.metrology_profile_refs)?;
        Ok(())
    }

    pub fn process_id(&self) -> Result<ProcessDefinitionId, ProcessPackError> {
        self.validate()?;
        self.process
            .process_id()
            .map_err(|err| ProcessPackError::InvalidProcess(err.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessPackV1 {
    pub namespace: String,
    pub pack_name: String,
    pub semantic_version: String,
    pub kind: ProcessPackKindV1,
    pub authority_ceiling: PackAuthorityCeilingV1,
    pub entries: Vec<ProcessPackEntryV1>,
}

impl ProcessPackV1 {
    pub fn validate(&self) -> Result<(), ProcessPackError> {
        canonical("pack.namespace", &self.namespace)?;
        canonical("pack.pack_name", &self.pack_name)?;
        canonical("pack.semantic_version", &self.semantic_version)?;
        if self.entries.is_empty() {
            return Err(ProcessPackError::EmptyPack);
        }

        let mut keys = BTreeSet::new();
        let mut process_ids = BTreeSet::new();
        for entry in &self.entries {
            entry.validate()?;
            if !keys.insert(entry.key.clone()) {
                return Err(ProcessPackError::DuplicateEntryKey(entry.key.clone()));
            }
            let process_id = entry.process_id()?.0;
            if !process_ids.insert(process_id.clone()) {
                return Err(ProcessPackError::DuplicateProcessId(process_id));
            }
        }
        Ok(())
    }

    pub fn pack_id(&self) -> Result<String, ProcessPackError> {
        self.validate()?;
        let mut entries = self.entries.clone();
        entries.sort_by(|a, b| a.key.cmp(&b.key));

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, PACK_ID_DOMAIN);
        hash_field(&mut hasher, &self.namespace);
        hash_field(&mut hasher, &self.pack_name);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, self.kind.tag());
        hash_field(&mut hasher, "representation-and-planning-metadata-only");

        for entry in entries {
            hash_field(&mut hasher, &entry.key);
            hash_field(&mut hasher, entry.maturity.tag());
            hash_field(&mut hasher, &entry.process_id()?.0);
            for mut refs in [
                entry.required_domain_refs,
                entry.safety_profile_refs,
                entry.metrology_profile_refs,
            ] {
                refs.sort();
                for value in refs {
                    hash_field(&mut hasher, &value);
                }
                hash_field(&mut hasher, "--end-ref-set--");
            }
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn process(
    name: &str,
    family: &str,
    effects: Vec<TransformationEffectV1>,
) -> ProcessDefinitionV1 {
    ProcessDefinitionV1 {
        namespace: "luminous.mfg-pack".into(),
        process_name: name.into(),
        semantic_version: "1.0.0".into(),
        family_refs: vec![ProcessFamilyRef(family.into())],
        operation_modes: vec![ManufacturingOperationModeV1::Discrete],
        transformation_effects: effects,
        extension_profile_refs: vec![],
        display_name: Some(name.replace('-', " ")),
        description: Some("Data-only manufacturing process profile; no execution authority".into()),
    }
}

fn entry(
    key: &str,
    maturity: ProcessMaturityV1,
    process: ProcessDefinitionV1,
    domains: &[&str],
    safety: &[&str],
    metrology: &[&str],
) -> ProcessPackEntryV1 {
    ProcessPackEntryV1 {
        key: key.into(),
        maturity,
        process,
        required_domain_refs: domains.iter().map(|v| (*v).to_string()).collect(),
        safety_profile_refs: safety.iter().map(|v| (*v).to_string()).collect(),
        metrology_profile_refs: metrology.iter().map(|v| (*v).to_string()).collect(),
    }
}

/// High-leverage processes already used industrially or in specialized production.
pub fn advanced_high_leverage_pack_v1() -> ProcessPackV1 {
    use ProcessMaturityV1::{AdvancedIndustrial, SpecializedIndustrial};
    use TransformationEffectV1::*;

    ProcessPackV1 {
        namespace: "luminous".into(),
        pack_name: "advanced-high-leverage".into(),
        semantic_version: "1.0.0".into(),
        kind: ProcessPackKindV1::AdvancedHighLeverage,
        authority_ceiling: PackAuthorityCeilingV1::RepresentationAndPlanningMetadataOnly,
        entries: vec![
            entry(
                "wire-edm",
                SpecializedIndustrial,
                process("wire-edm", "luminous:wire-edm/v1", vec![MaterialRemoval]),
                &["symthaea.circuits", "symthaea.materials"],
                &["safety:electrical-machining/v1"],
                &["metrology:dimensional/v1", "metrology:surface-finish/v1"],
            ),
            entry(
                "electrochemical-machining",
                SpecializedIndustrial,
                process(
                    "electrochemical-machining",
                    "luminous:electrochemical-machining/v1",
                    vec![MaterialRemoval, ElectrochemicalTransformation],
                ),
                &["symthaea.circuits", "symthaea.materials"],
                &["safety:electrochemical-process/v1"],
                &["metrology:dimensional/v1", "metrology:surface-chemistry/v1"],
            ),
            entry(
                "abrasive-waterjet",
                AdvancedIndustrial,
                process(
                    "abrasive-waterjet",
                    "luminous:abrasive-waterjet/v1",
                    vec![MaterialRemoval],
                ),
                &["symthaea.materials"],
                &["safety:high-pressure-process/v1"],
                &["metrology:dimensional/v1"],
            ),
            entry(
                "ultrafast-laser-micromachining",
                SpecializedIndustrial,
                process(
                    "ultrafast-laser-micromachining",
                    "luminous:ultrafast-laser-micromachining/v1",
                    vec![MaterialRemoval, SurfaceModification],
                ),
                &["symthaea.photonics", "symthaea.materials", "symthaea.thermal-engineering"],
                &["safety:laser-processing/v1"],
                &["metrology:optical/v1", "metrology:surface-finish/v1"],
            ),
            entry(
                "friction-stir-welding",
                AdvancedIndustrial,
                process(
                    "friction-stir-welding",
                    "luminous:friction-stir-welding/v1",
                    vec![JoiningOrBonding, MicrostructureModification],
                ),
                &["symthaea.materials", "symthaea.thermal-engineering"],
                &["safety:rotating-thermal-process/v1"],
                &["metrology:ndt/v1", "metrology:dimensional/v1"],
            ),
            entry(
                "cold-spray-deposition",
                AdvancedIndustrial,
                process(
                    "cold-spray-deposition",
                    "luminous:cold-spray-deposition/v1",
                    vec![MaterialAddition, SurfaceModification],
                ),
                &["symthaea.materials", "symthaea.continuum-physics"],
                &["safety:pressurized-particle-process/v1"],
                &["metrology:coating-thickness/v1", "metrology:adhesion/v1"],
            ),
            entry(
                "directed-energy-deposition",
                AdvancedIndustrial,
                process(
                    "directed-energy-deposition",
                    "luminous:directed-energy-deposition/v1",
                    vec![MaterialAddition, HeatTreatment],
                ),
                &["symthaea.materials", "symthaea.photonics", "symthaea.thermal-engineering"],
                &["safety:directed-energy-processing/v1"],
                &["metrology:dimensional/v1", "metrology:ndt/v1"],
            ),
            entry(
                "hot-isostatic-pressing",
                SpecializedIndustrial,
                process(
                    "hot-isostatic-pressing",
                    "luminous:hot-isostatic-pressing/v1",
                    vec![DensificationOrSintering, HeatTreatment],
                ),
                &["symthaea.materials", "symthaea.thermal-engineering"],
                &["safety:high-pressure-high-temperature/v1"],
                &["metrology:density-porosity/v1"],
            ),
            entry(
                "fast-sps",
                SpecializedIndustrial,
                process(
                    "field-assisted-sintering",
                    "luminous:fast-sps/v1",
                    vec![DensificationOrSintering, HeatTreatment],
                ),
                &["symthaea.materials", "symthaea.circuits", "symthaea.thermal-engineering"],
                &["safety:electro-thermal-processing/v1"],
                &["metrology:density-porosity/v1", "metrology:microstructure/v1"],
            ),
            entry(
                "atomic-layer-deposition",
                SpecializedIndustrial,
                process(
                    "atomic-layer-deposition",
                    "luminous:atomic-layer-deposition/v1",
                    vec![ThinFilmDeposition, ChemicalTransformation],
                ),
                &["symthaea.vacuum", "symthaea.materials", "symthaea.semiconductor"],
                &["safety:vacuum-chemical-process/v1"],
                &["metrology:thin-film/v1"],
            ),
            entry(
                "ion-beam-figuring",
                SpecializedIndustrial,
                process(
                    "ion-beam-figuring",
                    "luminous:ion-beam-figuring/v1",
                    vec![MaterialRemoval, SurfaceModification],
                ),
                &["symthaea.vacuum", "symthaea.photonics", "symthaea.plasma"],
                &["safety:ion-beam-vacuum-process/v1"],
                &["metrology:optical-surface/v1"],
            ),
            entry(
                "diamond-turning",
                SpecializedIndustrial,
                process(
                    "diamond-turning",
                    "luminous:diamond-turning/v1",
                    vec![MaterialRemoval, SurfaceModification],
                ),
                &["symthaea.photonics", "symthaea.materials"],
                &["safety:precision-machine-tool/v1"],
                &["metrology:optical-surface/v1", "metrology:dimensional/v1"],
            ),
        ],
    }
}

/// Research-oriented profiles. Representation is intentionally weaker than process qualification.
pub fn research_and_exotic_pack_v1() -> ProcessPackV1 {
    use ProcessMaturityV1::{ExperimentalConcept, ResearchDemonstrated};
    use TransformationEffectV1::*;

    ProcessPackV1 {
        namespace: "luminous".into(),
        pack_name: "research-and-exotic".into(),
        semantic_version: "1.0.0".into(),
        kind: ProcessPackKindV1::ResearchAndExotic,
        authority_ceiling: PackAuthorityCeilingV1::RepresentationAndPlanningMetadataOnly,
        entries: vec![
            entry(
                "two-photon-polymerization",
                ResearchDemonstrated,
                process(
                    "two-photon-polymerization",
                    "luminous:two-photon-polymerization/v1",
                    vec![MaterialAddition, PatternTransfer],
                ),
                &["symthaea.photonics", "symthaea.materials"],
                &["safety:laser-processing/v1"],
                &["metrology:micro-nano-geometry/v1"],
            ),
            entry(
                "electrohydrodynamic-printing",
                ResearchDemonstrated,
                process(
                    "electrohydrodynamic-printing",
                    "luminous:electrohydrodynamic-printing/v1",
                    vec![MaterialAddition, PatternTransfer],
                ),
                &["symthaea.circuits", "symthaea.materials"],
                &["safety:high-voltage-process/v1"],
                &["metrology:micro-nano-geometry/v1"],
            ),
            entry(
                "laser-induced-forward-transfer",
                ResearchDemonstrated,
                process(
                    "laser-induced-forward-transfer",
                    "luminous:laser-induced-forward-transfer/v1",
                    vec![MaterialAddition, PatternTransfer],
                ),
                &["symthaea.photonics", "symthaea.materials"],
                &["safety:laser-processing/v1"],
                &["metrology:micro-nano-geometry/v1"],
            ),
            entry(
                "molecular-beam-epitaxy",
                ResearchDemonstrated,
                process(
                    "molecular-beam-epitaxy",
                    "luminous:molecular-beam-epitaxy/v1",
                    vec![ThinFilmDeposition],
                ),
                &["symthaea.vacuum", "symthaea.semiconductor", "symthaea.materials"],
                &["safety:ultra-high-vacuum-deposition/v1"],
                &["metrology:thin-film/v1", "metrology:surface-chemistry/v1"],
            ),
            entry(
                "pulsed-laser-deposition",
                ResearchDemonstrated,
                process(
                    "pulsed-laser-deposition",
                    "luminous:pulsed-laser-deposition/v1",
                    vec![ThinFilmDeposition],
                ),
                &["symthaea.vacuum", "symthaea.photonics", "symthaea.plasma", "symthaea.materials"],
                &["safety:laser-vacuum-deposition/v1"],
                &["metrology:thin-film/v1"],
            ),
            entry(
                "atomic-layer-etching",
                ResearchDemonstrated,
                process(
                    "atomic-layer-etching",
                    "luminous:atomic-layer-etching/v1",
                    vec![MaterialRemoval, SurfaceModification],
                ),
                &["symthaea.vacuum", "symthaea.plasma", "symthaea.semiconductor"],
                &["safety:plasma-chemical-process/v1"],
                &["metrology:thin-film/v1", "metrology:surface-chemistry/v1"],
            ),
            entry(
                "freeze-casting",
                ResearchDemonstrated,
                process(
                    "freeze-casting",
                    "luminous:freeze-casting/v1",
                    vec![SolidificationOrMolding, MicrostructureModification],
                ),
                &["symthaea.materials", "symthaea.thermal-engineering"],
                &["safety:thermal-material-process/v1"],
                &["metrology:microstructure/v1", "metrology:porosity/v1"],
            ),
            entry(
                "containerless-processing",
                ExperimentalConcept,
                process(
                    "containerless-processing",
                    "luminous:containerless-processing/v1",
                    vec![Custom("containerless-material-processing".into())],
                ),
                &["symthaea.materials", "symthaea.continuum-physics"],
                &["safety:research-process-review-required/v1"],
                &["metrology:material-state/v1"],
            ),
            entry(
                "off-world-regolith-sintering",
                ExperimentalConcept,
                process(
                    "off-world-regolith-sintering",
                    "luminous:off-world-regolith-sintering/v1",
                    vec![DensificationOrSintering, Custom("in-situ-resource-processing".into())],
                ),
                &["symthaea.materials", "symthaea.thermal-engineering", "symthaea.orbital"],
                &["safety:research-process-review-required/v1"],
                &["metrology:material-state/v1", "metrology:dimensional/v1"],
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_bootstrap_packs_validate_and_have_distinct_ids() {
        let advanced = advanced_high_leverage_pack_v1();
        let exotic = research_and_exotic_pack_v1();
        assert!(advanced.validate().is_ok());
        assert!(exotic.validate().is_ok());
        assert_ne!(advanced.pack_id().unwrap(), exotic.pack_id().unwrap());
    }

    #[test]
    fn entry_order_does_not_change_pack_identity() {
        let a = advanced_high_leverage_pack_v1();
        let mut b = a.clone();
        b.entries.reverse();
        assert_eq!(a.pack_id().unwrap(), b.pack_id().unwrap());
    }

    #[test]
    fn maturity_change_changes_pack_identity_but_not_process_identity() {
        let a = advanced_high_leverage_pack_v1();
        let mut b = a.clone();
        let original_process_id = b.entries[0].process_id().unwrap();
        b.entries[0].maturity = ProcessMaturityV1::ResearchDemonstrated;
        assert_eq!(original_process_id, b.entries[0].process_id().unwrap());
        assert_ne!(a.pack_id().unwrap(), b.pack_id().unwrap());
    }

    #[test]
    fn duplicate_process_subject_rejects_even_under_different_pack_key() {
        let mut pack = advanced_high_leverage_pack_v1();
        let mut duplicate = pack.entries[0].clone();
        duplicate.key = "different-friendly-key".into();
        pack.entries.push(duplicate);
        assert!(matches!(
            pack.validate(),
            Err(ProcessPackError::DuplicateProcessId(_))
        ));
    }

    #[test]
    fn pack_contains_no_execution_authority_variant() {
        for pack in [advanced_high_leverage_pack_v1(), research_and_exotic_pack_v1()] {
            assert_eq!(
                pack.authority_ceiling,
                PackAuthorityCeilingV1::RepresentationAndPlanningMetadataOnly
            );
        }
    }

    #[test]
    fn serde_round_trip_preserves_pack_identity() {
        let pack = research_and_exotic_pack_v1();
        let encoded = serde_json::to_string(&pack).unwrap();
        let decoded: ProcessPackV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(pack.pack_id().unwrap(), decoded.pack_id().unwrap());
    }
}
