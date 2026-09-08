# Spark Canonical Belief State v1 — design / qualification contract

Status: staged architecture only. No product implementation. No scientific or action authority.

Tracks: #904

Base audited: `2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`.

## Problem

`HypothesisBelief::new()` validates and normalizes a caller-provided list of `(HypothesisType, weight)` pairs, but stores them in caller order. `ALL_HYPOTHESES` already defines a canonical semantic order.

That means equal probability mappings can have different ordinary serialized representations. It also leaves MAP tie resolution dependent on storage/iteration order, which can alter the synthetic world used by `greedy_sequence()`.

## Required invariant

For one declared Spark belief profile:

```text
same hypothesis -> probability mapping
    => same canonical hypothesis order
    => same canonical semantic export
    => same deterministic MAP tie-break
```

Caller container order is not scientific state.

## Proposed compatibility-safe behavior

`HypothesisBelief::new()` should:

1. require exactly the declared hypothesis set;
2. reject duplicate hypothesis entries;
3. reject missing hypotheses;
4. reject non-finite or non-positive weights under the current profile;
5. normalize weights;
6. store/export probabilities in `ALL_HYPOTHESES` order.

`map_hypothesis()` should use one explicit tie-break profile. Candidate v1 rule:

```text
maximum probability wins;
exact ties resolve to the first hypothesis in ALL_HYPOTHESES.
```

The final rule must be explicit and tested; it must not be inherited accidentally from iterator behavior.

## Canonical export boundary

A future non-authorizing Spark diagnostic may expose:

```text
CanonicalHypothesisBeliefV1 {
    order_profile: "spark-all-hypotheses-v1",
    probabilities: [(HypothesisType, f64); 5],
    normalization_profile,
    map_tie_break_profile,
}
```

This is a semantic export contract, not yet a cryptographic scientific identity.

Do not define authority as `hash(serde_json(...))`. Canonical number/float representation and scientific content identity belong under SCI-002.

## Negative controls

A future executable qualification must demonstrate current-main failure before repair:

1. Construct the same probability map in at least two different pair orders.
2. Show current iteration/serialization order differs.
3. Construct an exact tied maximum in two pair orders and show the current MAP result can depend on order.
4. Feed those tied beliefs into a synthetic MAP-scenario planning fixture and show the assumed scenario can diverge.

After repair require:

1. identical canonical iteration/export order for all permutations;
2. identical `prob(h)` values;
3. identical MAP result under every permutation;
4. identical one-step/MAP-scenario assumption under every permutation;
5. update operations retain canonical order;
6. deserialization/restore cannot reintroduce a noncanonical semantic state without validation.

## Deserialization boundary

Because `HypothesisBelief` derives `Deserialize`, constructor-only canonicalization may be bypassed by serialized state. Product implementation must choose one explicit policy:

- custom validated `Deserialize`, or
- private raw representation + validated restoration constructor, or
- mandatory `validate_and_canonicalize()` before planner use.

Do not claim representation closure if only `new()` is fixed.

## Non-claims

This contract does not calibrate probabilities, choose correct priors, establish Bayesian optimality, define universal floating-point canonicalization, or authorize experiments.
