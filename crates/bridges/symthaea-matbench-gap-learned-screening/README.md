# symthaea-matbench-gap-learned-screening

Truth-blind learned screening over the frozen Matbench composition universe from #2653, using the composition-only grouped model from #2674.

## Semantic learned-model identity

The core semantic model recipe is not sufficient by itself to identify the fitted model because the fitted parameters also depend on the exact training table. This adapter joins:

```text
exact Symthaea exposed-training snapshot SHA-256
+
frozen CompositionOnlyModelRecipe
=
model_identity_sha256
```

`model_identity_sha256` is the **semantic fitted-model identity** used by the benchmark protocol. It is not a cryptographic attestation of implementation bytes, compiler, dependencies or runtime. Exact source/tree/toolchain/environment identity remains a separate qualification subject and must accompany any promoted execution evidence.

The semantic model identity is carried into `ScreeningMethodProvenance.method_id`, while the exact training snapshot is also represented as a singleton `training_slices` entry:

```text
dataset_id = symthaea_bandgap_curated
split_id   = sha256:<exact training snapshot>
```

This means the later Benchmark Zero receipt preserves the machine-readable training snapshot used by the learned model instead of retaining only a friendly method name.

## Prior-knowledge boundary

The machine-readable training slice identifies the curated table used to fit the composition-only Random Forest.

It does **not** exhaust all prior knowledge. In particular, the residual `electronegativity_bandgap` baseline is a legacy empirical/physics-inspired formula whose source comments describe fitted/chosen semiconductor parameters. The screening receipt therefore carries a fixed separate prior-knowledge disclosure.

## Truth-blind screening path

The public adapter consumes only:

- `ExposedTrainingExclusionPlan`;
- preregistered `BandgapTarget`.

It does not accept `BandgapTruthSet`.

Before ranking it requires:

1. structural plan validity;
2. exact current training-snapshot equality;
3. non-empty retained universe.

Every retained `CompositionFingerprint` is converted to positive stoichiometric amounts and passed to `CompositionOnlyBandgapPredictor`, whose own preprocessing canonicalizes the composition before inference.

Candidates are ranked by absolute distance between predicted band gap and the preregistered target midpoint, then canonical candidate id and source row index.

## Separate identities

The receipt binds:

- full source-plan SHA-256 for provenance only;
- source composition-order SHA-256;
- exposure-partition SHA-256;
- exact training-snapshot SHA-256;
- retained-universe SHA-256;
- full serialized composition-only semantic model recipe;
- semantic `model_identity_sha256`;
- target;
- truth-free `screening_subject_sha256`;
- Benchmark Zero ordered ranking digest;
- exact `ScreeningRun`;
- fixed prior-knowledge and uncertainty disclosures.

The full source-plan SHA-256 is deliberately excluded from the screening-subject identity because the full plan transitively binds truth-sensitive official-artifact identities. The screening subject instead binds the truth-free universe plus exact semantic model identity and target.

## Uncertainty boundary

`ScreeningRecord.uncertainty_ev` is populated from Random-Forest inter-tree standard deviation.

That value is not called a calibrated confidence interval, total prediction error, within-composition polymorph ambiguity, or experimental uncertainty. #2674 keeps within-composition observed spread separate in its grouped OOF surface.

## Replay

`verify_learned_screening_receipt(plan, receipt)` independently revalidates:

- current training snapshot;
- all bound plan identities;
- exact semantic recipe;
- semantic model identity;
- singleton training-slice provenance;
- deterministic model fit and ranking;
- exact final receipt equality.

Replay still does not replace exact implementation qualification. A source/toolchain change can require a new qualification lineage even if the semantic recipe remains unchanged.

## Authority boundary

A successful receipt would establish deterministic learned ranking of the exact frozen truth-free universe under the exact recorded training snapshot and semantic recipe.

It would not establish:

- implementation attestation by itself;
- a globally clean holdout;
- absence of historical/manual prior knowledge;
- calibrated uncertainty;
- predictive superiority;
- statistical significance;
- material novelty or feasibility;
- candidate promotion or experiment authority.

Exact-head pinned-toolchain compile/test/rustfmt/strict-Clippy evidence remains required before qualification.
