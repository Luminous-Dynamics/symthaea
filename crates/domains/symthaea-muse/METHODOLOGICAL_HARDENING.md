# Symthaea–Muse Cognition Study: V8.2 Methodological Hardening

V8.2 is an additive layer over the green V8 empirical-evaluation system. It
changes neither the Sonata generator nor the temporal cognitive policy. Its
purpose is to make the first confirmatory claim harder to bias, easier to
reproduce, and clearer to interpret.

## Authority order

1. The frozen V8 manifest identifies the study fixtures and policy versions.
2. The V8.2 methodology plan identifies exactly one primary endpoint.
3. The complete theory report determines structural validity.
4. The participant schedule determines presentation order.
5. Raw responses must match that assigned order.
6. Policy-budget evidence proves that all arms saw the same candidate set.
7. Confirmatory inference respects musical-family and participant dependence.

No listener, artist, mechanism, or prediction result may substitute for a
failed primary endpoint.

## Frozen methodology plan

`methodology_plan` binds:

- the V8 manifest digest;
- one primary, zero or more secondary, and zero or more exploratory endpoints;
- practical superiority and heuristic non-inferiority margins;
- an externally registered preregistration record;
- the exact adaptive-model checkpoint, training-data digest, update count,
  hyperparameter digest, pilot cutoff, and RNG seed;
- the verifier source revision, binary digest, rule-set version, and Nix or
  container environment digest;
- one equal resource budget used by all four policy arms.

All endpoints listed by the V8 manifest must be classified. Exactly one may be
primary.

## Participant-specific counterbalancing

The V8 base schedule remains the public artifact identity layer. V8.2 adds a
participant schedule book over it.

For every confirmatory fixture, listeners are assigned one of four Williams
sequences:

- every presentation occurs equally often in positions 1–4;
- every ordered pair of distinct presentations occurs equally often as an
  adjacent carryover;
- participant counts must be a complete multiple of four;
- the first protocol requires at least twelve participants;
- public assignments contain only pseudonymous participant tokens and anonymous
  presentation IDs;
- the private audit contains the policy-arm sequence and remains protected with
  the original codebook.

The evidence envelope verifies that included blocks follow the exact assigned
order. Excluded blocks remain retained and may be incomplete, but any responses
they contain must preserve the assigned relative order.

## Structural evidence

A structural pass can no longer originate from a manually entered summary.
Each presentation is bound to:

- score digest;
- recipe digest;
- complete `TheoryValidationReport`;
- report digest;
- verifier invocation and stdout digests;
- verifier source, binary, rule-set, and environment identities;
- completed-score obligation and transformed-motif evidence.

The V8 `StructuralTrialOutcome` is derived from this bundle. Hard validity,
voice-leading violations, and tonic return are computed from the full theory
report.

## Fair policy comparison

`policy_budget_evidence` requires every confirmatory fixture to provide one
shared candidate-set digest. All arms must:

- evaluate exactly the frozen number of candidates;
- use the same allowed operator set;
- run under the same compute environment;
- remain within identical theory-validation and policy-evaluation limits.

The Symthaea arm must identify the exact checkpoint frozen in the methodology
plan. A larger search or better candidate generator cannot be hidden inside the
cognitive arm.

## Confirmatory inference

### Non-ranking endpoints

`family_clustered_analysis` averages paired fixture effects inside each musical
family before inference. Bootstrap resampling and sign randomization operate on
families, not nominally separate variants of one theme.

The analysis accepts only the single frozen primary endpoint. If preference is
the primary endpoint, this path refuses it because aggregate rank scores are not
an appropriate primary analysis.

### Preference endpoint

`ranked_preference_analysis` uses complete ranks directly. For every listener
and fixture it records whether Symthaea outranked each comparator. It does not
assume that rank gaps are equally spaced.

- the estimand is Symthaea's pairwise win probability;
- the effect is win probability minus 0.5;
- confidence intervals use a two-way participant/family cluster bootstrap;
- one-sided randomization is performed on family-level effects;
- Holm adjustment covers fixed superiority, random-valid superiority, and
  heuristic non-inferiority.

Presentation positions remain in the exported observations for independent
mixed-effects or rank-model analysis.

## Reproducibility

Evidence hashing now uses RustCrypto SHA-256. The canonical JSON profile remains
explicitly versioned as `symthaea-canonical-json-sha256-v1`; V8.2 does not call
it RFC 8785.

The family bootstrap uses a specified SplitMix64 stream. An independent Python
standard-library implementation and golden fixtures must agree with Rust:

```sh
scripts/check-cognition-v82-reference.sh
```

The Python path is a cross-check, not a second opportunity to choose a more
favorable analysis.

## CLI workflow

The `cognitive_study` binary now supports:

- `validate-methodology`;
- `build-participant-schedule`;
- `validate-participant-schedule`;
- `seal-structural-evidence`;
- `compile-structural-evidence`;
- `seal-policy-budget`;
- `validate-policy-budget`;
- `seal-participant-evidence`;
- `compile-participant-evidence`;
- `analyze-family-clustered`;
- `analyze-ranked-preference`.

The V8 commands remain available for migration and descriptive robustness
analysis.

## Claim boundary

V8.2 strengthens the evaluation machinery. It does not contain participant
responses and does not establish that Symthaea improves music.

A future positive result would support only the frozen Sonata fixtures,
candidate operators, policy versions, model checkpoint, renderer, soundfont,
participant population, and endpoints. It would not establish consciousness,
sentience, universal musical taste, or general superiority over human
composition.

## Remaining methodological work

Before confirmatory collection:

- run a pilot and simulation-based power analysis;
- obtain external statistical, music-theory, and human-subjects review;
- publish the manifest, methodology plan, analysis code, environment lock, and
  randomization commitment through an immutable external registry;
- add an independently specified mixed-effects or Plackett–Luce robustness
  analysis;
- keep the adaptive checkpoint frozen until all confirmatory evidence is sealed.
