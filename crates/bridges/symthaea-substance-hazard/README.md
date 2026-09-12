# Symthaea Exact-Substance Hazard Evidence

This crate supplies one bounded Tier-1 human/environmental hazard input.

It deliberately does **not** infer a compound or material's hazard from the hazards of its constituent elements.

A candidate must be bound to an exact source substance record, and any numerical score must come from an explicit caller-supplied scoring policy.

## Why there is no built-in hazard score

Hazard statements are categorical evidence. Converting them into one scalar objective requires policy choices:

- which domains count;
- how codes are weighted;
- whether the worst code or the sum is used;
- whether a code is intentionally assigned zero weight.

Those choices are not universal physical constants. This crate therefore has **no default scoring policy and no built-in severity table**.

The exact policy JSON bytes and a domain-separated semantic policy digest are both bound into the final receipt.

## Exact substance identity

Two binding modes exist:

### `exact_substance_id`

The Symthaea candidate ID must exactly equal the source substance identifier.

### `explicit_mapping`

A differently named candidate may be associated with a source substance only with a non-empty reviewable note.

That mapping remains an assumption. It is not independent evidence that a generated structure, polymorph, mixture, formulation, or process stream is chemically identical to the source substance.

## Source dataset

The normalized hazard dataset records:

- source title/version/URI;
- SHA-256 of the authoritative source document;
- normalization timestamp and extraction note;
- exact substance identifiers/names;
- hazard code + statement pairs.

The evaluation requires the authoritative source-document bytes and verifies their SHA-256 before producing evidence.

The normalized JSON bytes receive their own exact digest, and the semantic dataset receives a separate domain-separated digest.

## Scoring policy

A policy declares:

- `policy_id`;
- rationale;
- aggregation method;
- included hazard domains;
- an explicit rule for each hazard code that may appear.

Supported domains are:

- `human_health`;
- `environmental`;
- `physical`;
- `other`.

The Tier-1 score must include human-health and/or environmental domains.

Supported aggregation methods are:

- `maximum_rule_weight`;
- `sum_rule_weights`.

Every observed source code must have a rule. A physical hazard can be excluded from the human/environmental objective, but it still requires an explicit rule so it cannot silently disappear because a policy forgot about it.

Weights must be finite and non-negative. Zero is allowed only as an explicit policy choice.

## Policy example

This is a schema example only. The codes and weights are not scientific recommendations:

    {
      "policy_id": "example-policy-v0",
      "rationale": "Demonstration only; not a validated severity scale.",
      "aggregation": "maximum_rule_weight",
      "included_domains": ["human_health", "environmental"],
      "rules": [
        {
          "code": "H-EXAMPLE-1",
          "domain": "human_health",
          "weight": 3.0,
          "rationale": "Example only"
        },
        {
          "code": "H-EXAMPLE-2",
          "domain": "physical",
          "weight": 1.0,
          "rationale": "Mapped explicitly but excluded from this Tier-1 objective"
        }
      ]
    }

## Evidence semantics

A successful calculation emits:

- metric: `human_environmental_hazard_score`;
- unit: `score`;
- fidelity: `Analytical`;
- supporting evidence: `Dataset`;
- source/capture/dataset digests;
- raw policy JSON SHA-256 + semantic policy SHA-256;
- exact substance binding;
- every source hazard statement, its assigned domain/weight, and whether it contributed.

No calibrated uncertainty model is invented; epistemic uncertainty is marked fully unknown.

## Zero-score boundary

A score of zero can occur when:

- the exact source record contains no hazard statements;
- included hazard rules have explicit zero weights;
- all observed hazards belong to explicitly excluded domains.

None of those conditions proves that the substance is hazard-free. Dataset completeness, exposure, dose, route, formulation, impurities, persistence, bioavailability, use conditions and life-cycle impacts remain separate questions.

## Network-free CLI

    substance-hazard <hazard-data.json> <source-document> <policy.json> <candidate-id> <substance-id> [explicit-mapping-note]

Without the optional note, exact-ID mode is used and candidate ID must equal substance ID.

The CLI performs no HTTP and does not include host-local paths in the receipt.

## Deliberate non-claims

A successful receipt is not:

- a toxicological dose-response model;
- an exposure assessment;
- an occupational-safety determination;
- an environmental fate model;
- proof of source-dataset completeness;
- material safety certification;
- experimental validation;
- approval for synthesis/manufacturing/deployment.
