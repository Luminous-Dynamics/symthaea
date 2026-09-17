// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen follow-on protocol for the SYM-RSI-001 grounded-dream comparison.
//!
//! SYM-RSI-001D inherits the original training replay substrate but allocates
//! independent verification, fresh, and OOD seeds for D-vs-C. This module contains
//! protocol identity and seed construction only; it executes no measurement.

use super::sym_rsi_experiment::{
    DomainSeedPlan, ExperimentArm, ExperimentDomainSpec, SymRsiExperimentManifest,
    SYM_RSI_001_MANIFEST_SCHEMA,
};
use super::sym_rsi_fixtures::{
    FixtureDomainKind, SYM_RSI_001_FIXTURE_ADAPTER_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SYM_RSI_001D_PROTOCOL_SCHEMA: &str =
    "symthaea.sym-rsi-001d.dream-extension-protocol.v1";
pub const SYM_RSI_001D_EXPERIMENT_ID: &str = "SYM-RSI-001D";
pub const SYM_RSI_001D_ANALYSIS_RULE: &str =
    "symthaea.sym-rsi-001d.d-vs-c-analysis.v1";
pub const SYM_RSI_001D_QUALITY_TOLERANCE: f64 = 0.02;

pub const INHERITED_TRAINING_SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
pub const DREAM_VERIFICATION_SEEDS: [u64; 4] = [301, 302, 303, 304];
pub const DREAM_FRESH_SEEDS: [u64; 4] = [401, 402, 403, 404];
pub const DREAM_OOD_SEEDS: [u64; 4] = [1201, 1202, 1203, 1204];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamExtensionProtocolBinding {
    pub schema: String,
    pub experiment_id: String,
    pub parent_experiment_id: String,
    pub preregistration_digest: String,
    pub parent_training_corpus_evidence_digest: String,
    pub c_selection_evidence_digest: String,
    pub grounded_dream_model_evidence_digest: String,
    pub subject_digest: String,
    pub environment_digest: String,
}

impl DreamExtensionProtocolBinding {
    pub fn validate(&self) -> Result<(), DreamProtocolError> {
        if self.schema != SYM_RSI_001D_PROTOCOL_SCHEMA
            || self.experiment_id != SYM_RSI_001D_EXPERIMENT_ID
            || self.parent_experiment_id != "SYM-RSI-001"
        {
            return Err(DreamProtocolError::WrongIdentity);
        }
        for value in [
            &self.preregistration_digest,
            &self.parent_training_corpus_evidence_digest,
            &self.c_selection_evidence_digest,
            &self.grounded_dream_model_evidence_digest,
            &self.subject_digest,
            &self.environment_digest,
        ] {
            if value.trim().is_empty() {
                return Err(DreamProtocolError::MissingBinding);
            }
        }
        Ok(())
    }
}

/// Construct the frozen D-vs-C extension manifest.
///
/// Training seeds are inherited to document the provenance of the already-frozen
/// dream model; this manifest does not authorize retraining. Verification/fresh/OOD
/// partitions are new and disjoint from all original SYM-RSI-001 evaluation seeds.
pub fn canonical_sym_rsi_001d_manifest(
    preregistration_digest: impl Into<String>,
    subject_digest: impl Into<String>,
    environment_digest: impl Into<String>,
) -> SymRsiExperimentManifest {
    let seeds = DomainSeedPlan {
        training_replay: INHERITED_TRAINING_SEEDS.to_vec(),
        held_out_replay: DREAM_VERIFICATION_SEEDS.to_vec(),
        fresh_execution: DREAM_FRESH_SEEDS.to_vec(),
        out_of_distribution: DREAM_OOD_SEEDS.to_vec(),
    };

    SymRsiExperimentManifest {
        schema: SYM_RSI_001_MANIFEST_SCHEMA.into(),
        experiment_id: SYM_RSI_001D_EXPERIMENT_ID.into(),
        preregistration_digest: preregistration_digest.into(),
        subject_digest: subject_digest.into(),
        environment_digest: environment_digest.into(),
        arms: ExperimentArm::ALL.to_vec(),
        domains: FixtureDomainKind::ALL
            .into_iter()
            .map(|domain| ExperimentDomainSpec {
                domain_id: domain.id().into(),
                adapter_version: SYM_RSI_001_FIXTURE_ADAPTER_VERSION.into(),
                max_evaluator_calls: 128,
                seeds: seeds.clone(),
            })
            .collect(),
        beta_cost: 0.05,
        beta_parallelism: 0.01,
        held_out_quality_tolerance: SYM_RSI_001D_QUALITY_TOLERANCE,
    }
}

/// Prove that the extension evaluation partitions do not reuse any original
/// SYM-RSI-001 held-out/fresh/OOD seeds.
///
/// Training 1-8 are intentionally inherited and therefore excluded from this check.
pub fn validate_dream_extension_seed_independence() -> Result<(), DreamProtocolError> {
    let original_evaluation = [
        101_u64, 102, 103, 104, 201, 202, 203, 204, 1001, 1002, 1003, 1004,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let extension_evaluation = DREAM_VERIFICATION_SEEDS
        .into_iter()
        .chain(DREAM_FRESH_SEEDS)
        .chain(DREAM_OOD_SEEDS)
        .collect::<BTreeSet<_>>();

    if !original_evaluation.is_disjoint(&extension_evaluation) {
        return Err(DreamProtocolError::EvaluationSeedReuse);
    }

    let expected_len =
        DREAM_VERIFICATION_SEEDS.len() + DREAM_FRESH_SEEDS.len() + DREAM_OOD_SEEDS.len();
    if extension_evaluation.len() != expected_len {
        return Err(DreamProtocolError::DuplicateExtensionSeed);
    }
    Ok(())
}

pub fn validate_canonical_sym_rsi_001d_manifest(
    manifest: &SymRsiExperimentManifest,
) -> Result<(), DreamProtocolError> {
    manifest
        .validate()
        .map_err(DreamProtocolError::ManifestInvalid)?;
    validate_dream_extension_seed_independence()?;

    let expected = canonical_sym_rsi_001d_manifest(
        manifest.preregistration_digest.clone(),
        manifest.subject_digest.clone(),
        manifest.environment_digest.clone(),
    );
    if manifest != &expected {
        return Err(DreamProtocolError::ManifestNotCanonical);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DreamProtocolError {
    WrongIdentity,
    MissingBinding,
    EvaluationSeedReuse,
    DuplicateExtensionSeed,
    ManifestInvalid(super::sym_rsi_experiment::ExperimentHarnessError),
    ManifestNotCanonical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_manifest_is_valid_without_executing_any_measurement() {
        let manifest = canonical_sym_rsi_001d_manifest("dream-pre", "subject", "env");
        assert_eq!(manifest.validate(), Ok(()));
        assert_eq!(validate_canonical_sym_rsi_001d_manifest(&manifest), Ok(()));
        assert_eq!(manifest.experiment_id, SYM_RSI_001D_EXPERIMENT_ID);
        assert_eq!(manifest.domains.len(), 3);
    }

    #[test]
    fn extension_evaluation_seeds_are_independent_from_original_evaluation() {
        assert_eq!(validate_dream_extension_seed_independence(), Ok(()));
        assert_eq!(DREAM_VERIFICATION_SEEDS, [301, 302, 303, 304]);
        assert_eq!(DREAM_FRESH_SEEDS, [401, 402, 403, 404]);
        assert_eq!(DREAM_OOD_SEEDS, [1201, 1202, 1203, 1204]);
    }

    #[test]
    fn inherited_training_seeds_remain_frozen() {
        assert_eq!(INHERITED_TRAINING_SEEDS, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
