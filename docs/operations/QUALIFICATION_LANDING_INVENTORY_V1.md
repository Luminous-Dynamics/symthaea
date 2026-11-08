# Qualification Landing Inventory V1

Status: draft inventory only; no qualification authority and no PASS transfer.

Parent program: #3742
Convergence map: #3746
Normative framing guard: #2469

## Purpose

Identify the smallest reviewed generic qualification closure that should eventually be curated onto current `main` without broad-merging historical `tools/qualification-*` ancestry.

The inventory is provenance, not an integration command. Every historical source remains frozen at its own commit. Landing requires an independently reviewed current-main subject and fresh qualification.

## Generic qualification waist

The generic waist ends at exact qualification/evidence facts. Domain-specific interpretation remains above it.

```text
subject/profile/recipe/input closure/environment
  -> attempt history
  -> PASS selection / support closure
  -> receipt-candidate gate
  -> positive receipt core

then, separately:
  -> domain plan conformance / interpretation
```

A generic qualification receipt does not decide scientific, safety, security, constitutional, or deployment truth.

## Candidate source lineages

### Normative framing and semantic IDs

Source candidate:

```text
tools/qualification-semantic-convergence-v2
5d951664ff2608d231da258b2a29db56d07e83f5
```

Role:
- one canonical framing protocol (`qualification_framing_v1.py`);
- converged subject/profile/recipe semantic identities;
- domain-separated deterministic framing.

Landing rule: no second framing magic, field order, set/list convention, or competing semantic-ID preimage may be introduced. #2469 remains the governing migration rule.

### Normative input closure

Source candidate:

```text
tools/qualification-input-closure-convergence-v1
b474d182e1df8995f0a426ae7ab5e8e2bb3c87bf
```

Role:
- import/bind normative qualification input-closure semantics;
- preserve exact subject/profile/recipe relationship to semantic closure.

Important: the later pass-selection projection explicitly distinguishes historical source `input_closure_id` from reconstructed normative `qualification_input_closure_id_v1`; equality is not assumed.

### PASS-selection semantic projection

Source candidate:

```text
tools/qualification-pass-selection-semantic-projection-v2
4471e64ad7fc7a4c7982173547eec721cd2d2e6c
```

Role:
- project historical PassSelectionV2 onto normative subject/profile/recipe/input-closure semantics;
- verify historical selection identity before projection;
- retain source environment ID rather than pretending the projection migrated environment or attempt identity.

Required non-claim to preserve:

```text
semantic projection != rewrite of historical PassSelectionV2 identity
```

### Qualification environment realization convergence

Source candidate:

```text
tools/qualification-environment-realization-convergence-v1
1e1652d879e4609a6b1a4464f832e2503838789a
```

Parent is exact pass-selection projection head `4471e64a...`.

Role:
- converge verified environment realization into receipt lineage;
- preserve environment identity separately from input closure.

QUAL follow-on may strengthen observed executable path/digest sealing, but must not mint a competing `QualificationEnvironmentId`.

### Attempt history / integrity

Relevant historical source:

```text
tools/qualification-attempt-integrity-v1
eba670418a57ac9f8f101e02f98febb41d8e1bd4
```

Related attempt-ledger/attempt-set branches remain provenance inputs and must be reviewed as a family before landing.

Role:
- terminal attempts remain append-only evidence;
- distinguish theorem/source failures from infrastructure/cancelled/unknown outcomes;
- later PASS does not delete previous red or indeterminate attempts.

Landing requirement: identify one authoritative current attempt schema/version and migrate older attempt records explicitly rather than landing parallel attempt formats as peers.

### Exact evidence-byte support closure

Source candidate:

```text
tools/qualification-support-closure-v1
7a1975522821a6f07d7611fb7b937b7b91155228
```

Role:
- require selected evidence content IDs to have exactly one supplied byte payload;
- verify byte/content bindings;
- reject missing or extra selected evidence bytes.

Critical claim ceiling:

```text
byte closure
!= trusted terminal disposition
!= cross-cutting-rule satisfaction
!= provider provenance/authenticity
!= evidence correctness/sufficiency
!= claim interpretation
```

### Receipt-candidate gate

Source candidate:

```text
tools/qualification-receipt-candidate-gate-v1
2654d2590ed9eb6701abd82d147ec9938ea7e7f4
```

Parent is exact support-closure head `7a197552...`.

Role:
- fail closed before portable positive-receipt authority;
- only eligible selected evidence may proceed toward a positive receipt.

### Positive receipt core

Python/core source lineage:

```text
tools/qualification-receipt-core-v1
54e6fc3a688b04cbd4ea4b32e9c1b91454093aee
```

Rust counterpart:

```text
tools/qualification-receipt-core-rust-v1
765fdedefd72e5042f458f682c1e9a176ff5a7ff
```

Role:
- portable provider/retry-neutral positive receipt identity after PASS-selection/candidate eligibility;
- bind qualification subject, profile, input closure, environment, and required recipe theorems.

Claim ceiling:

```text
QualificationReceiptCore
!= provider authenticity
!= trusted PASS by itself
!= scientific validity
!= current admission
!= merge/execution/deployment authority
```

Before landing, Python and Rust encodings/identities must be proven equivalent with frozen cross-language golden vectors or one implementation must be explicitly non-normative.

## Existing generic execution lineage

Current repository owner:

```text
symthaea-evidence-plane::ExecutionLineageV1
```

This binds computational source/tree, locks, toolchains, host/target, optional Nix identity, features, working directory, argv, relevant environment and immutable input artifacts.

Landing must compose qualification environment realization with this lineage where appropriate rather than creating a duplicate generic execution-lineage object.

## Provider plane remains separate

CI-QUAL-001 (#3649) owns runner/provider state ambiguity and focused-subject versus repository-integration evidence.

The landing closure may consume normalized provider facts but must not duplicate CI-QUAL classification logic.

Provider occurrence facts remain separate from provider/retry-neutral positive receipt identity.

## Domain interpretation remains separate

HAK, ASSURE, PHYS and other domains own the semantic policy mapping exact qualification evidence into domain claims.

The generic layer should expose references sufficient for domain conformance/interpretation but must not define one universal `Supported`/`Contradicted` scientific meaning.

Cross-program invariant:

```text
ProviderExecutionOutcome
    != GenericQualificationEvidence
    != DomainClaimInterpretation
```

Examples:
- compiler failure may be exact negative execution evidence without contradicting a scientific hypothesis;
- positive receipt may establish a qualified software/evidence subject without establishing a physical-substrate advantage;
- support closure may establish byte completeness without establishing evidence sufficiency.

## Not in the generic landing closure

Do not import as generic-core semantics merely because related branches exist:

- integration-train admission/current-admission policy;
- HAK claim interpretation policy;
- PHYS experiment manifests or scientific metrics;
- CogSec domain-specific package sets/claims;
- ASSURE runtime/hardware theorem semantics;
- provider authentication/trusted-runner claims;
- deployment/merge/physical authority.

These remain adapters or higher-layer policies.

## Historical branch topology warning

The source candidates above do not form one safe broad-merge ancestry. At least two historical development subchains exist, and some convergence commits project older identities instead of replacing them.

Therefore landing must operate on reviewed files/blobs + theorem dependencies, not on the assumption that one branch tip contains the authoritative union.

## Landing manifest requirements

Before implementation, create a machine-readable manifest binding for every imported primitive:

```text
logical_role
source_branch
source_commit
source_path
source_blob
semantic_schema/domain
required_dependencies
required_golden_vectors
claim_ceiling
migration/equivalence theorem
```

The manifest must reject:
- two normative framing implementations;
- two peer owners for the same semantic ID;
- silent Python/Rust identity divergence;
- missing dependency/source blob;
- old identity renamed as a new identity without migration proof;
- PASS or qualification authority transferred only because the source blob was imported.

## Proposed landing sequence

```text
L0  freeze inventory + source-blob manifest
L1  import normative framing + semantic identities
L2  import normative input closure + environment realization
L3  converge one attempt-history schema
L4  import semantic PASS projection + exact support closure
L5  import receipt-candidate gate + positive receipt core
L6  cross-language identity vectors + historical adapter tests
L7  exact current-main qualification of the combined generic closure
L8  only then build PHYS/HAK/CogSec/ASSURE adapters
```

Each L-step should be a bounded subject. Do not call ancestry itself a qualification proof.

## PHYS consequence

PHYS reconstruction remains blocked on L7. The old PHYS heads and attempts remain historical evidence. Repaired PHYS subjects must bind to the newly qualified generic closure and receive new exact identities.

## Non-claims

This inventory does not assert that any listed branch is merge-ready, mutually compatible, executable-qualified, current, authoritative over similarly named historical branches, or sufficient for QERA-001. It is a source-discovery and ownership map for the next review step.
