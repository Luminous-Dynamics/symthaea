// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advisory-only generation lifecycle facts for boot presentation consumers.
//!
//! This schema is intentionally incapable of selecting a boot generation. It
//! contains observations and qualification facts only: no default, selected,
//! recommended, next-boot, or boot-order field exists. Limine and Spore may use
//! these facts for labels/visuals; recovery authority remains elsewhere.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const GENERATION_LIFECYCLE_SCHEMA_VERSION: u16 = 1;

/// The only authority mode supported by this schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestAuthority {
    AdvisoryOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationHealth {
    Unknown,
    Healthy,
    Degraded,
    Failed,
}

/// Optional measured value with explicit provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredValueV1 {
    pub value: f32,
    pub provenance: String,
}

impl MeasuredValueV1 {
    pub fn new(value: f32, provenance: impl Into<String>) -> Result<Self, String> {
        let measured = Self {
            value,
            provenance: provenance.into(),
        };
        measured.validate_unit_interval("measurement")?;
        Ok(measured)
    }

    fn validate_unit_interval(&self, name: &str) -> Result<(), String> {
        if !self.value.is_finite() || !(0.0..=1.0).contains(&self.value) {
            return Err(format!(
                "{name} must be finite and in [0,1], got {}",
                self.value
            ));
        }
        if self.provenance.trim().is_empty() {
            return Err(format!("{name} provenance must not be empty"));
        }
        Ok(())
    }
}

/// Cognitive information is presentation/advisory metadata only. Its absence is
/// valid and preferable to inventing a score.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CognitiveAdvisoryV1 {
    pub phi: Option<MeasuredValueV1>,
    pub confidence: Option<f32>,
    pub free_energy: Option<f32>,
    pub prediction_error: Option<f32>,
    pub causal_support: Option<f32>,
}

impl CognitiveAdvisoryV1 {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(phi) = &self.phi {
            phi.validate_unit_interval("phi")?;
        }
        validate_unit_interval("confidence", self.confidence)?;
        validate_non_negative("free_energy", self.free_energy)?;
        validate_non_negative("prediction_error", self.prediction_error)?;
        validate_unit_interval("causal_support", self.causal_support)?;
        Ok(())
    }
}

/// Independent facts about one NixOS generation.
///
/// Roles are booleans rather than a single enum because legitimate states
/// overlap: the running generation may simultaneously be the known-good one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationFactsV1 {
    pub generation: u64,
    pub store_path: String,
    pub running: bool,
    pub known_good: bool,
    pub candidate: bool,
    pub previous: bool,
    pub health: GenerationHealth,
    pub rollback_ready: bool,
    /// Digest of qualification evidence, when such evidence exists.
    pub evidence_digest: Option<String>,
    pub cognitive: CognitiveAdvisoryV1,
}

impl GenerationFactsV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_store_path(&self.store_path)?;
        if self.known_good && self.health == GenerationHealth::Failed {
            return Err(format!(
                "generation {} cannot be both known-good and failed",
                self.generation
            ));
        }
        if let Some(digest) = &self.evidence_digest
            && digest.trim().is_empty()
        {
            return Err(format!(
                "generation {} has an empty evidence digest",
                self.generation
            ));
        }
        self.cognitive.validate()?;
        Ok(())
    }

    /// Presentation badges in stable priority order. Multiple badges may apply.
    pub fn semantic_badges(&self) -> Vec<&'static str> {
        let mut badges = Vec::new();
        if self.known_good {
            badges.push("KNOWN GOOD");
        }
        if self.running {
            badges.push("RUNNING");
        }
        if self.candidate {
            badges.push("CURRENT CANDIDATE");
        }
        if self.previous {
            badges.push("PREVIOUS");
        }
        badges
    }
}

/// Versioned, advisory-only snapshot consumed by presentation layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationLifecycleManifestV1 {
    pub schema_version: u16,
    pub authority: ManifestAuthority,
    pub producer: String,
    pub observed_unix_ms: u64,
    pub generations: Vec<GenerationFactsV1>,
}

impl GenerationLifecycleManifestV1 {
    pub fn new(
        producer: impl Into<String>,
        observed_unix_ms: u64,
        generations: Vec<GenerationFactsV1>,
    ) -> Self {
        Self {
            schema_version: GENERATION_LIFECYCLE_SCHEMA_VERSION,
            authority: ManifestAuthority::AdvisoryOnly,
            producer: producer.into(),
            observed_unix_ms,
            generations,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != GENERATION_LIFECYCLE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported generation lifecycle schema {}, expected {}",
                self.schema_version, GENERATION_LIFECYCLE_SCHEMA_VERSION
            ));
        }
        if self.authority != ManifestAuthority::AdvisoryOnly {
            return Err("generation lifecycle manifest must be advisory-only".into());
        }
        if self.producer.trim().is_empty() {
            return Err("generation lifecycle producer must not be empty".into());
        }

        let mut seen_generations = HashSet::new();
        let mut running = 0usize;
        let mut known_good = 0usize;
        let mut candidate = 0usize;
        let mut previous = 0usize;

        for facts in &self.generations {
            facts.validate()?;
            if !seen_generations.insert(facts.generation) {
                return Err(format!(
                    "generation {} appears more than once",
                    facts.generation
                ));
            }
            running += usize::from(facts.running);
            known_good += usize::from(facts.known_good);
            candidate += usize::from(facts.candidate);
            previous += usize::from(facts.previous);
        }

        validate_role_cardinality("running", running)?;
        validate_role_cardinality("known_good", known_good)?;
        validate_role_cardinality("candidate", candidate)?;
        validate_role_cardinality("previous", previous)?;
        Ok(())
    }

    pub fn generation(&self, generation: u64) -> Option<&GenerationFactsV1> {
        self.generations
            .iter()
            .find(|facts| facts.generation == generation)
    }

    /// Deterministic content digest for integrity/evidence binding. This digest
    /// does not grant boot authority.
    pub fn content_digest(&self) -> Result<String, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("serialize generation lifecycle manifest: {error}"))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nixward-generation-lifecycle-v1\0");
        hasher.update(&encoded);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_store_path(path: &str) -> Result<(), String> {
    if !path.starts_with("/nix/store/") || path == "/nix/store/" {
        return Err(format!("generation store path is outside /nix/store: {path}"));
    }
    if path.contains(['\n', '\r', '\0']) {
        return Err("generation store path contains control characters".into());
    }
    Ok(())
}

fn validate_role_cardinality(role: &str, count: usize) -> Result<(), String> {
    if count > 1 {
        return Err(format!("more than one generation is marked {role}"));
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: Option<f32>) -> Result<(), String> {
    if let Some(value) = value
        && (!value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(format!("{name} must be finite and in [0,1], got {value}"));
    }
    Ok(())
}

fn validate_non_negative(name: &str, value: Option<f32>) -> Result<(), String> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(format!("{name} must be finite and non-negative, got {value}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(generation: u64) -> GenerationFactsV1 {
        GenerationFactsV1 {
            generation,
            store_path: format!("/nix/store/test-generation-{generation}"),
            running: false,
            known_good: false,
            candidate: false,
            previous: false,
            health: GenerationHealth::Unknown,
            rollback_ready: false,
            evidence_digest: None,
            cognitive: CognitiveAdvisoryV1::default(),
        }
    }

    #[test]
    fn running_and_known_good_may_overlap_on_same_generation() {
        let mut current = facts(42);
        current.running = true;
        current.known_good = true;
        current.health = GenerationHealth::Healthy;
        current.rollback_ready = true;
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![current]);
        manifest.validate().unwrap();
        assert_eq!(
            manifest.generations[0].semantic_badges(),
            vec!["KNOWN GOOD", "RUNNING"]
        );
    }

    #[test]
    fn roles_are_unique_across_generations() {
        let mut a = facts(1);
        let mut b = facts(2);
        a.running = true;
        b.running = true;
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![a, b]);
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn failed_generation_cannot_be_known_good() {
        let mut bad = facts(7);
        bad.known_good = true;
        bad.health = GenerationHealth::Failed;
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![bad]);
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn store_paths_fail_closed() {
        let mut bad = facts(7);
        bad.store_path = "/tmp/fake-generation".into();
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![bad]);
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn cognition_is_optional_and_validated() {
        let mut item = facts(8);
        item.cognitive.phi = Some(MeasuredValueV1::new(0.5, "iit-small-network").unwrap());
        item.cognitive.free_energy = Some(1.4);
        item.cognitive.confidence = Some(0.8);
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![item]);
        manifest.validate().unwrap();

        let mut invalid = manifest.clone();
        invalid.generations[0].cognitive.confidence = Some(f32::NAN);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn schema_contains_no_boot_selection_instruction() {
        let mut item = facts(9);
        item.running = true;
        let manifest = GenerationLifecycleManifestV1::new("nixward", 1, vec![item]);
        let json = serde_json::to_string(&manifest).unwrap();
        for forbidden in [
            "default_generation",
            "selected_generation",
            "recommended_generation",
            "next_boot_generation",
            "boot_order",
        ] {
            assert!(!json.contains(forbidden), "manifest leaked {forbidden}");
        }
        assert!(json.contains("AdvisoryOnly"));
    }

    #[test]
    fn digest_is_deterministic_and_fact_sensitive() {
        let mut item = facts(10);
        item.candidate = true;
        let manifest = GenerationLifecycleManifestV1::new("nixward", 10, vec![item]);
        let a = manifest.content_digest().unwrap();
        let b = manifest.content_digest().unwrap();
        assert_eq!(a, b);

        let mut changed = manifest.clone();
        changed.generations[0].health = GenerationHealth::Healthy;
        assert_ne!(a, changed.content_digest().unwrap());
    }
}
