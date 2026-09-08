# Spark Likelihood Model Profile v1

Status: queue-neutral design and qualification contract only.

Related: #938, #930, #857, #868, #928, #906.

## Problem

A Spark posterior is meaningful only relative to the likelihood semantics that produced it.

Current semantics include code-level choices such as:

```text
MATCH_LIKELIHOOD
ENERGY_RESOLUTION_MEV
outcome-class construction
observation matching
missing-channel behavior
unpredicted-hypothesis behavior
no-class-match residual rule
prediction schema
```

Persisting only a probability vector loses this model lineage.

## Core theorem

```text
posterior values
    !=
posterior inference semantics
```

and:

```text
likelihood-profile drift
    -> new inference lineage or qualified migration
```

not silent continuation.

## Target profile

Conceptually:

```text
SparkLikelihoodModelProfileV1 {
    profile_version,
    hypothesis_set_profile,
    prediction_schema_profile,
    outcome_class_algorithm_profile,
    observation_channel_profile,
    energy_resolution_mev,
    match_likelihood,
    no_match_rule,
    unpredicted_hypothesis_rule,
    missing_channel_rule,
}
```

The final authoritative identity belongs under SCI-002 canonical scientific artifact/profile identity.

## Update receipt binding

`BayesianUpdateReceiptV1` should retain:

```text
prior_belief_ref
observation_evidence_ref
likelihood_model_profile_ref
exact evaluated likelihood vector
posterior_belief_ref
```

This permits replay of historical updates without depending on whatever constants/current code happen to exist later.

## Drift states

Possible compatibility outcome:

```text
CompatibleSameProfile
HistoricalReadOnly
NewInferenceRootRequired
QualifiedMigrationAvailable
```

A posterior from one profile remains valid historical evidence about that inference process even when a better model is introduced.

It simply cannot be silently extended as though the model never changed.

## Relationship to #857 and #868

#857's order-invariant/observation-profile-aware discrimination and #868's missing-prediction separation intentionally change the model semantics.

Those improvements should therefore receive new profile identity rather than retroactively redefining historical posterior values.

## Required qualification cases

- same prior/observation under different match-likelihood profiles -> distinct lineage;
- changed energy-resolution profile -> distinct lineage;
- changed outcome-class algorithm -> distinct lineage;
- update receipt exposes the exact likelihood vector used;
- historical posterior cannot silently continue under incompatible profile;
- planning snapshot binds the exact posterior/profile lineage;
- qualified migration, if ever implemented, is explicit and separately evidenced.

## SCI-003 boundary

Likelihood-model profile and execution capsule are separate:

```text
profile = what inference semantics
capsule = what exact executable environment/run
```

Both may matter to replay, but neither substitutes for the other.

## Non-claims

This contract does not claim the current Spark likelihood profile is calibrated or scientifically optimal.

It preserves the meaning of historical inference while allowing the model to improve honestly.
