//! SWA-011 — deterministic reproducibility witness fixture.
//!
//! A witness binds a claim to a canonical replay manifest and derived artifact.
//! It is evidence, never authority.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct DependencyRevision {
    pub kind: String,
    pub id: String,
    pub revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRecipe {
    pub engine: String,
    pub operation: String,
    pub algorithm_revision: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayInput {
    pub claim_id: String,
    pub recipe: ExecutionRecipe,
    pub dependencies: Vec<DependencyRevision>,
    pub input_values: Vec<(String, i64)>,
}

impl ReplayInput {
    pub fn canonicalize(mut self) -> Self {
        self.dependencies.sort();
        self.input_values.sort();
        self
    }
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ReplayInput is serializable")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayArtifact {
    pub claim_id: String,
    pub result: i64,
    pub dependency_count: usize,
}

impl ReplayArtifact {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ReplayArtifact is serializable")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReproducibilityWitness {
    pub witness_id: String,
    pub input_fingerprint: String,
    pub artifact_fingerprint: String,
    pub input_manifest: ReplayInput,
    pub artifact: ReplayArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WitnessError {
    MissingClaimId,
    MissingEngine,
    MissingOperation,
    MissingAlgorithmRevision,
    MissingDependencies,
    MissingInputs,
}

impl ReproducibilityWitness {
    pub fn build(input: ReplayInput) -> Result<Self, WitnessError> {
        let input = input.canonicalize();
        validate(&input)?;
        let artifact = execute(&input);
        let input_fingerprint = fingerprint(&input.canonical_bytes());
        let artifact_fingerprint = fingerprint(&artifact.canonical_bytes());
        let witness_id = format!("rw:{input_fingerprint}:{artifact_fingerprint}");
        Ok(Self { witness_id, input_fingerprint, artifact_fingerprint, input_manifest: input, artifact })
    }

    pub fn replay(&self) -> Result<Self, WitnessError> {
        Self::build(self.input_manifest.clone())
    }

    pub fn is_reproducible(&self) -> bool {
        self.replay().map(|replayed| replayed == *self).unwrap_or(false)
    }

    pub fn dependency_changed(&self, kind: &str, revision: &str) -> bool {
        self.input_manifest.dependencies.iter().any(|d| d.kind == kind && d.revision != revision)
    }
}

fn validate(input: &ReplayInput) -> Result<(), WitnessError> {
    if input.claim_id.is_empty() { return Err(WitnessError::MissingClaimId); }
    if input.recipe.engine.is_empty() { return Err(WitnessError::MissingEngine); }
    if input.recipe.operation.is_empty() { return Err(WitnessError::MissingOperation); }
    if input.recipe.algorithm_revision.is_empty() { return Err(WitnessError::MissingAlgorithmRevision); }
    if input.dependencies.is_empty() { return Err(WitnessError::MissingDependencies); }
    if input.input_values.is_empty() { return Err(WitnessError::MissingInputs); }
    Ok(())
}

/// Deterministic stand-in for model execution: a pure function of the manifest.
fn execute(input: &ReplayInput) -> ReplayArtifact {
    let result = input.input_values.iter().fold(0_i64, |a, (_, v)| a.saturating_add(*v));
    ReplayArtifact { claim_id: input.claim_id.clone(), result, dependency_count: input.dependencies.len() }
}

/// FNV-1a 64-bit. This is a deterministic fixture fingerprint, not a cryptographic commitment.
fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> ReplayInput {
        ReplayInput {
            claim_id: "claim-001".into(),
            recipe: ExecutionRecipe { engine: "symthaea-engineering".into(), operation: "thermal-counterfactual".into(), algorithm_revision: "swa-011-v1".into() },
            dependencies: vec![
                DependencyRevision { kind: "Model".into(), id: "thermal-model".into(), revision: "m7".into() },
                DependencyRevision { kind: "Parameters".into(), id: "building-a".into(), revision: "p3".into() },
                DependencyRevision { kind: "Scenario".into(), id: "winter-week".into(), revision: "s2".into() },
            ],
            input_values: vec![("zone-a".into(), 18), ("zone-b".into(), 21)],
        }
    }

    #[test]
    fn identical_inputs_replay_identically() {
        let a = ReproducibilityWitness::build(input()).unwrap();
        let b = ReproducibilityWitness::build(input()).unwrap();
        assert_eq!(a, b);
        assert!(a.is_reproducible());
    }

    #[test]
    fn canonical_ordering_does_not_change_identity() {
        let mut reversed = input();
        reversed.dependencies.reverse();
        reversed.input_values.reverse();
        let a = ReproducibilityWitness::build(input()).unwrap();
        let b = ReproducibilityWitness::build(reversed).unwrap();
        assert_eq!(a.input_fingerprint, b.input_fingerprint);
        assert_eq!(a.artifact_fingerprint, b.artifact_fingerprint);
    }

    #[test]
    fn dependency_change_changes_witness() {
        let a = ReproducibilityWitness::build(input()).unwrap();
        let mut changed = input();
        changed.dependencies[0].revision = "m8".into();
        let b = ReproducibilityWitness::build(changed).unwrap();
        assert_ne!(a.input_fingerprint, b.input_fingerprint);
        assert_ne!(a.witness_id, b.witness_id);
        assert!(a.dependency_changed("Model", "m8"));
    }

    #[test]
    fn incomplete_manifest_fails_closed() {
        let mut value = input();
        value.dependencies.clear();
        assert_eq!(ReproducibilityWitness::build(value), Err(WitnessError::MissingDependencies));
    }

    #[test]
    fn artifact_is_derived_from_manifest() {
        let witness = ReproducibilityWitness::build(input()).unwrap();
        assert_eq!(witness.artifact, execute(&witness.input_manifest));
    }
}
