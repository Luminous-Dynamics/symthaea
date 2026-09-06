// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for closed forecast-input derivation lineage.
//!
//! `TypedForecastProvenance` deliberately separates source observations,
//! transformations and the final input snapshot, but today it is an inventory:
//! it does not claim that the final snapshot was actually derived from the listed
//! observations through the listed transformations. This test freezes that
//! distinction before a production graph type exists.
//!
//! The candidate graph below proves declared *topological closure* only. It does
//! not prove that transformation code was executed, that bytes match their
//! declared digests, or that source/timing/custody evidence is trustworthy.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use symthaea_futures_ledger::provenance::{
    ContentAddressedRef, ContentDigest, TypedForecastProvenance,
};

const ZERO_SHA256: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";
const ONE_SHA256: &str =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";
const TWO_SHA256: &str =
    "sha256:2222222222222222222222222222222222222222222222222222222222222222";
const THREE_SHA256: &str =
    "sha256:3333333333333333333333333333333333333333333333333333333333333333";
const FOUR_SHA256: &str =
    "sha256:4444444444444444444444444444444444444444444444444444444444444444";
const FIVE_SHA256: &str =
    "sha256:5555555555555555555555555555555555555555555555555555555555555555";
const SIX_SHA256: &str =
    "sha256:6666666666666666666666666666666666666666666666666666666666666666";
const SEVEN_SHA256: &str =
    "sha256:7777777777777777777777777777777777777777777777777777777777777777";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ArtifactKey {
    namespace: String,
    id: String,
}

impl ArtifactKey {
    fn of(value: &ContentAddressedRef) -> Self {
        Self {
            namespace: value.namespace().to_string(),
            id: value.id().to_string(),
        }
    }

    fn display(&self) -> String {
        format!("{}/{}", self.namespace, self.id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DerivationError {
    EmptyRoots,
    EmptyStepId,
    DuplicateRoot(String),
    DuplicateStepId(String),
    DuplicateProducedArtifact(String),
    UnknownDependency(String),
    DependencyDigestMismatch(String),
    FinalSnapshotUndeclared,
    FinalSnapshotDigestMismatch,
    Cycle(String),
    DecorativeStep(String),
    DecorativeRoot(String),
}

#[derive(Debug, Clone)]
struct DerivationStep {
    step_id: String,
    /// Exact implementation artifact used by this step. This is lineage
    /// metadata, not proof that the implementation ran.
    transform: ContentAddressedRef,
    /// Data artifacts consumed by the transform.
    inputs: Vec<ContentAddressedRef>,
    /// Parameter/configuration artifacts consumed by the transform.
    parameters: Vec<ContentAddressedRef>,
    output: ContentAddressedRef,
}

impl DerivationStep {
    fn new(
        step_id: impl Into<String>,
        transform: ContentAddressedRef,
        inputs: Vec<ContentAddressedRef>,
        parameters: Vec<ContentAddressedRef>,
        output: ContentAddressedRef,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            transform,
            inputs,
            parameters,
            output,
        }
    }

    fn dependencies(&self) -> impl Iterator<Item = &ContentAddressedRef> {
        self.inputs.iter().chain(self.parameters.iter())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClosedInputDerivation {
    final_snapshot: ContentAddressedRef,
    contributing_step_ids: BTreeSet<String>,
    contributing_root_keys: BTreeSet<ArtifactKey>,
    transform_keys: BTreeSet<ArtifactKey>,
}

impl ClosedInputDerivation {
    fn qualify(
        roots: Vec<ContentAddressedRef>,
        steps: Vec<DerivationStep>,
        final_snapshot: ContentAddressedRef,
    ) -> Result<Self, DerivationError> {
        if roots.is_empty() {
            return Err(DerivationError::EmptyRoots);
        }

        let mut declared: BTreeMap<ArtifactKey, ContentAddressedRef> = BTreeMap::new();
        let mut root_keys = BTreeSet::new();
        for root in roots {
            let key = ArtifactKey::of(&root);
            if declared.insert(key.clone(), root).is_some() {
                return Err(DerivationError::DuplicateRoot(key.display()));
            }
            root_keys.insert(key);
        }

        let mut steps_by_output: BTreeMap<ArtifactKey, DerivationStep> = BTreeMap::new();
        let mut step_ids = BTreeSet::new();
        for step in steps {
            if step.step_id.trim().is_empty() {
                return Err(DerivationError::EmptyStepId);
            }
            if !step_ids.insert(step.step_id.clone()) {
                return Err(DerivationError::DuplicateStepId(step.step_id));
            }
            let output_key = ArtifactKey::of(&step.output);
            if declared.contains_key(&output_key) {
                return Err(DerivationError::DuplicateProducedArtifact(
                    output_key.display(),
                ));
            }
            match steps_by_output.entry(output_key) {
                Entry::Vacant(entry) => {
                    entry.insert(step);
                }
                Entry::Occupied(entry) => {
                    return Err(DerivationError::DuplicateProducedArtifact(
                        entry.key().display(),
                    ));
                }
            }
        }

        // Build the complete declared-artifact index before graph traversal so
        // step declaration order carries no semantic meaning.
        for (key, step) in &steps_by_output {
            declared.insert(key.clone(), step.output.clone());
        }

        for step in steps_by_output.values() {
            for dependency in step.dependencies() {
                let key = ArtifactKey::of(dependency);
                let Some(expected) = declared.get(&key) else {
                    return Err(DerivationError::UnknownDependency(key.display()));
                };
                if expected.digest() != dependency.digest() {
                    return Err(DerivationError::DependencyDigestMismatch(key.display()));
                }
            }
        }

        let final_key = ArtifactKey::of(&final_snapshot);
        let Some(declared_final) = declared.get(&final_key) else {
            return Err(DerivationError::FinalSnapshotUndeclared);
        };
        if declared_final.digest() != final_snapshot.digest() {
            return Err(DerivationError::FinalSnapshotDigestMismatch);
        }

        let mut visiting_steps = BTreeSet::new();
        let mut contributing_step_ids = BTreeSet::new();
        let mut contributing_root_keys = BTreeSet::new();
        let mut transform_keys = BTreeSet::new();
        Self::visit_artifact(
            &final_key,
            &root_keys,
            &steps_by_output,
            &mut visiting_steps,
            &mut contributing_step_ids,
            &mut contributing_root_keys,
            &mut transform_keys,
        )?;

        for step_id in &step_ids {
            if !contributing_step_ids.contains(step_id) {
                return Err(DerivationError::DecorativeStep(step_id.clone()));
            }
        }
        for root_key in &root_keys {
            if !contributing_root_keys.contains(root_key) {
                return Err(DerivationError::DecorativeRoot(root_key.display()));
            }
        }

        Ok(Self {
            final_snapshot,
            contributing_step_ids,
            contributing_root_keys,
            transform_keys,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_artifact(
        key: &ArtifactKey,
        root_keys: &BTreeSet<ArtifactKey>,
        steps_by_output: &BTreeMap<ArtifactKey, DerivationStep>,
        visiting_steps: &mut BTreeSet<String>,
        contributing_step_ids: &mut BTreeSet<String>,
        contributing_root_keys: &mut BTreeSet<ArtifactKey>,
        transform_keys: &mut BTreeSet<ArtifactKey>,
    ) -> Result<(), DerivationError> {
        if root_keys.contains(key) {
            contributing_root_keys.insert(key.clone());
            return Ok(());
        }

        let step = steps_by_output
            .get(key)
            .expect("declared non-root artifact must have one producer");
        if contributing_step_ids.contains(&step.step_id) {
            return Ok(());
        }
        if !visiting_steps.insert(step.step_id.clone()) {
            return Err(DerivationError::Cycle(step.step_id.clone()));
        }

        transform_keys.insert(ArtifactKey::of(&step.transform));
        for dependency in step.dependencies() {
            Self::visit_artifact(
                &ArtifactKey::of(dependency),
                root_keys,
                steps_by_output,
                visiting_steps,
                contributing_step_ids,
                contributing_root_keys,
                transform_keys,
            )?;
        }

        visiting_steps.remove(&step.step_id);
        contributing_step_ids.insert(step.step_id.clone());
        Ok(())
    }
}

fn digest(value: &str) -> ContentDigest {
    ContentDigest::parse(value).unwrap()
}

fn artifact(namespace: &str, id: &str, digest_text: &str) -> ContentAddressedRef {
    ContentAddressedRef::new(namespace, id, digest(digest_text)).unwrap()
}

fn observation() -> ContentAddressedRef {
    artifact("observation", "gdp-2026q2-advance", ZERO_SHA256)
}

fn seasonal_parameters() -> ContentAddressedRef {
    artifact("parameter", "seasonal-factors-v7", ONE_SHA256)
}

fn transform_one() -> ContentAddressedRef {
    artifact("transform", "seasonal-adjustment-v2", TWO_SHA256)
}

fn transform_two() -> ContentAddressedRef {
    artifact("transform", "feature-builder-v3", THREE_SHA256)
}

fn adjusted() -> ContentAddressedRef {
    artifact("derived", "gdp-adjusted", FOUR_SHA256)
}

fn final_snapshot() -> ContentAddressedRef {
    artifact("input-snapshot", "forecast-inputs", FIVE_SHA256)
}

fn model() -> ContentAddressedRef {
    artifact("model", "econ-model-v1", SIX_SHA256)
}

fn generator() -> ContentAddressedRef {
    artifact("generator", "econ-generator-v1", SEVEN_SHA256)
}

#[test]
fn current_typed_provenance_is_an_inventory_not_a_derivation_proof() {
    let unrelated_snapshot = artifact("input-snapshot", "unrelated-valid-bytes", FIVE_SHA256);
    let provenance = TypedForecastProvenance::new(
        vec![observation()],
        unrelated_snapshot.clone(),
        vec![transform_one()],
        artifact("observation-policy", "econ-observe-v1", ONE_SHA256),
        vec![model()],
        vec![generator()],
    )
    .unwrap();

    // This is valid by design in #548: all identities are syntactically valid,
    // but no graph currently proves that the final snapshot follows from the
    // listed source and transform.
    assert_eq!(provenance.input_snapshot(), &unrelated_snapshot);
    assert_eq!(provenance.observations().len(), 1);
    assert_eq!(provenance.transformations().len(), 1);
}

#[test]
fn closed_derivation_accepts_a_complete_two_step_chain_independent_of_step_order() {
    let step_adjust = DerivationStep::new(
        "adjust",
        transform_one(),
        vec![observation()],
        vec![seasonal_parameters()],
        adjusted(),
    );
    let step_features = DerivationStep::new(
        "features",
        transform_two(),
        vec![adjusted()],
        vec![],
        final_snapshot(),
    );

    let qualified = ClosedInputDerivation::qualify(
        vec![observation(), seasonal_parameters()],
        vec![step_features, step_adjust],
        final_snapshot(),
    )
    .unwrap();

    assert_eq!(qualified.final_snapshot, final_snapshot());
    assert_eq!(
        qualified.contributing_step_ids,
        BTreeSet::from(["adjust".to_string(), "features".to_string()])
    );
    assert_eq!(qualified.contributing_root_keys.len(), 2);
    assert_eq!(qualified.transform_keys.len(), 2);
}

#[test]
fn unrelated_final_snapshot_fails_even_when_all_digests_are_well_formed() {
    let result = ClosedInputDerivation::qualify(
        vec![observation()],
        vec![DerivationStep::new(
            "adjust",
            transform_one(),
            vec![observation()],
            vec![],
            adjusted(),
        )],
        final_snapshot(),
    );

    assert_eq!(result, Err(DerivationError::FinalSnapshotUndeclared));
}

#[test]
fn undeclared_dependency_and_digest_substitution_fail_closed() {
    let unknown = artifact("parameter", "undeclared-parameter", ONE_SHA256);
    let unknown_result = ClosedInputDerivation::qualify(
        vec![observation()],
        vec![DerivationStep::new(
            "adjust",
            transform_one(),
            vec![observation()],
            vec![unknown],
            final_snapshot(),
        )],
        final_snapshot(),
    );
    assert!(matches!(
        unknown_result,
        Err(DerivationError::UnknownDependency(_))
    ));

    let substituted_observation = artifact("observation", "gdp-2026q2-advance", ONE_SHA256);
    let substitution_result = ClosedInputDerivation::qualify(
        vec![observation()],
        vec![DerivationStep::new(
            "adjust",
            transform_one(),
            vec![substituted_observation],
            vec![],
            final_snapshot(),
        )],
        final_snapshot(),
    );
    assert!(matches!(
        substitution_result,
        Err(DerivationError::DependencyDigestMismatch(_))
    ));
}

#[test]
fn cycles_fail_even_when_every_artifact_is_declared() {
    let a = artifact("derived", "a", FOUR_SHA256);
    let b = artifact("derived", "b", FIVE_SHA256);
    let result = ClosedInputDerivation::qualify(
        vec![observation()],
        vec![
            DerivationStep::new(
                "make-a",
                transform_one(),
                vec![b.clone()],
                vec![],
                a.clone(),
            ),
            DerivationStep::new(
                "make-b",
                transform_two(),
                vec![a],
                vec![],
                b.clone(),
            ),
        ],
        b,
    );

    assert!(matches!(result, Err(DerivationError::Cycle(_))));
}

#[test]
fn decorative_steps_and_roots_are_rejected_not_silently_credited() {
    let orphan_output = artifact("derived", "orphan", SIX_SHA256);
    let decorative_step = ClosedInputDerivation::qualify(
        vec![observation()],
        vec![
            DerivationStep::new(
                "main",
                transform_one(),
                vec![observation()],
                vec![],
                final_snapshot(),
            ),
            DerivationStep::new(
                "decorative",
                transform_two(),
                vec![observation()],
                vec![],
                orphan_output,
            ),
        ],
        final_snapshot(),
    );
    assert_eq!(
        decorative_step,
        Err(DerivationError::DecorativeStep("decorative".into()))
    );

    let decorative_root = ClosedInputDerivation::qualify(
        vec![observation(), seasonal_parameters()],
        vec![DerivationStep::new(
            "main",
            transform_one(),
            vec![observation()],
            vec![],
            final_snapshot(),
        )],
        final_snapshot(),
    );
    assert_eq!(
        decorative_root,
        Err(DerivationError::DecorativeRoot(
            ArtifactKey::of(&seasonal_parameters()).display()
        ))
    );
}

#[test]
fn direct_source_snapshot_is_closed_without_inventing_a_transform() {
    let direct = observation();
    let qualified = ClosedInputDerivation::qualify(vec![direct.clone()], vec![], direct.clone())
        .unwrap();

    assert_eq!(qualified.final_snapshot, direct);
    assert!(qualified.contributing_step_ids.is_empty());
    assert_eq!(qualified.contributing_root_keys.len(), 1);
    assert!(qualified.transform_keys.is_empty());
}
