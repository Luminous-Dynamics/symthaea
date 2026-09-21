# EPI-SEM-001A-R4 — Three-surface epistemic authority inventory

Parent: #5283 (`EPI-SEM-001`)

Supersedes incomplete/unqualified R1 (#5326/#5327), R2 (#5331/#5333), and R3 (#5358/#5359).

## Purpose

Freeze the legacy E/N/M authority surface before producer repair.

Each preceding audit found a distinct way authority semantics could hide:

```text
R1: one literal E3/E4 spelling was insufficient
R2: aliases + a second Mycelix implementation were missed
R3: E aliases + numeric shapes still did not fail closed on N3/M3 vocabulary
```

R4 therefore treats epistemic authority as three independently inventoried surfaces:

1. empirical proof/reproduction aliases;
2. normative/material high-authority aliases;
3. numeric/local E/N/M encodings.

The audit remains a regression inventory, not an epistemic verifier.

## Core separation law

```text
metric / heuristic / scalar / enum tag / caller claim
!= evidence that establishes the proposition named by an authority state
```

Examples:

```text
z-score                     != cryptographic proof
z-score                     != public reproduction
Phi                         != cryptographic proof
runtime cycles              != cryptographic proof
observation count           != intervention / counterfactual evidence
authenticity score          != cryptographic verification
fact-check scalar           != proof / reproduction / axiom
open source                 != independent reproduction
similarity                  != consensus
constitutional action type  != axiomatic truth
workspace visibility        != corroboration
importance / retention      != foundational truth
same integer                != same semantic proposition
weighted E/N/M score        != component authority
```

## Discovery surface 1 — empirical authority

R4 scans Rust source for known empirical proof/reproduction aliases:

```text
E3CryptographicallyProven
E4PubliclyReproducible
CryptographicallyVerifiable
PubliclyReproducible
E3Cryptographic
E4PublicRepro
```

This freezes the current E3/E4 vocabulary across consciousness, HDC retrieval, Mycelix bridges, physics discovery, compatibility stubs, and tests.

## Discovery surface 2 — high N/M authority

R3 was superseded because it could not fail closed if a new file introduced high N/M authority without any E3/E4 spelling.

R4 independently scans for:

```text
N3Axiomatic
M3Foundational
NormativeLevel::Foundational
MaterialityLevel::Permanent
```

This captures the current vocabulary families for propositions such as:

```text
axiomatic / constitutional truth
foundational / effectively permanent knowledge
```

The important rule is:

```text
N3/M3 vocabulary is authority-bearing even when E remains low.
```

### Newly explicit N3 producer

`src/consciousness/mycelix_bridge.rs` contains:

```text
ActionType::Constitutional
-> NormativeLevel::N3Axiomatic
```

An action being constitutional does not establish that a claim is axiomatic. The action class and epistemic basis are separate propositions.

### Existing N/M producer families

R4 also freezes:

```text
continuous normative score >= threshold -> numeric N3-like state
continuous mythic score >= threshold    -> numeric M3-like state
workspace universal                     -> Foundational -> N3Axiomatic
importance >= threshold                 -> Permanent -> M3Foundational
high importance                         -> Permanent suggestion
```

These are migration debt for #5337, not approved semantics.

## Discovery surface 3 — numeric/local shapes

Authority can also hide behind integers or local enums. R4 retains R3's independent scan for:

```text
pub empirical: u8
pub e_tier: u8
struct LocalEpistemicClassification
enum LocalEmpiricalLevel
```

This specifically keeps visible:

- both physics LEM classifiers;
- `FactcheckEpistemicFeedback`;
- the inlined `symthaea-mycelix-bridge` classification.

Marker coverage is intentionally described as a **ratchet**, not proof that every possible encoding pattern has been discovered.

## Known empirical producer debt

R4 freezes, without endorsing, the known promotion paths:

### HDC retrieval

```text
z-score bands
-> E3CryptographicallyProven / E4PubliclyReproducible
```

### Causal self-explanation

```text
observation count + internally derived confidence
-> E3CryptographicallyProven
```

### Runtime maturity

```text
total cycle count
-> E3CryptographicallyProven
```

### Symthaea↔Mycelix Phi mapping

```text
Phi >= threshold
-> CryptographicallyVerifiable

Phi + caller is_reproducible bool
-> PubliclyReproducible
```

A configuration can also disable the E4 reproducibility requirement, allowing Phi alone to reach the E4-labelled state.

### Evidence-tag suggestion

```text
caller-supplied EvidenceType::CryptographicProof
-> E3 suggestion
```

An enum tag is not verification of a proof artifact.

### Consciousness↔Mycelix fact-check bridge

```text
empirical scalar >= 0.8 -> numeric tier 4
empirical scalar >= 0.6 -> numeric tier 3
"True" verdict -> empirical 0.9 -> tier 4
authenticity > 0.8 -> E3Cryptographic
```

### Physics catalog

```text
open source + simulation confidence -> E4
catalog similarity                  -> N3
```

## High-axis producer debt

R4 gives N/M authority its own witness bucket.

### Consciousness↔Mycelix bridge

```text
normative scalar >= 0.75 -> numeric N3 "axiomatic"
mythic scalar >= 0.75    -> numeric M3 "foundational"
constitutional action    -> N3Axiomatic
```

### Mycelix mapper

```text
consensus-tag count -> empirical upgrade
high importance     -> Permanent M suggestion
```

This demonstrates cross-axis promotion as well as high-state promotion.

### Inlined Symthaea–Mycelix bridge

```text
WorkspaceScope::Universal
-> LocalNormativeLevel::Foundational
-> SDK N3Axiomatic

importance >= 0.75
-> LocalMaterialityLevel::Permanent
-> SDK M3Foundational
```

Its empirical side remains a positive control because observed local validation is capped at E1 unless stronger evidence exists.

## Scalar collapse

R4 continues to freeze the two known E/N/M aggregate-score surfaces.

A combined score may be useful for display, sorting, or bounded heuristics, but:

```text
high aggregate
!= E3
!= E4
!= N3
!= M3
```

No consumer should reconstruct a component theorem from the aggregate.

## Propagation and product seams

R4 separately freezes paths that can preserve or expose a misclassification:

- consciousness tier -> shared tier conversion;
- legacy wire parsing;
- Mycelix SDK ordinal gates;
- integration tests that normalize high E/N/M values;
- browser/WASM physics classification.

The browser surface remains important:

```text
classify_physics_claim(description, confidence, is_open_source)
```

reaches the legacy physics classifier, and unknown claims are currently defaulted to `Domain::ModifiedGravity`.

## Axis-definition collision

The repository currently carries multiple incompatible meanings under similar E/N/M shapes.

### Shared epistemic types

```text
N: who agrees this knowledge claim is valid?
M: how permanent is this knowledge?
```

### `src/mycelix/types.rs`

```text
N: who should have access?
M: how long should this persist?
```

### consciousness fact-check bridge

```text
N: continuous normative/source-consensus score
M: continuous mythic/persistence/foundationality score
```

### inlined Symthaea–Mycelix bridge

```text
N: workspace scope projected into SDK normative levels
M: importance projected into SDK materiality levels
```

Therefore:

```text
(profile A, N3) != (profile B, N3)
(profile A, M3) != (profile B, M3)
```

unless an explicit semantic profile and evidence contract establish equivalence.

This remains the core motivation for #5337.

## Positive controls

R4 preserves two bounded examples.

### Physics bridge

A local simulation supports at most E1 absent replay/integrity/reproduction evidence.

### Inlined Symthaea–Mycelix bridge

Observed local validation supports at most E1; stronger empirical authority requires stronger evidence.

These controls demonstrate the desired pattern:

```text
source-local observation
-> proposition-accurate bounded description

not

source-local observation
-> unrelated external authority theorem
```

## Relationship to #5334

Producer repair remains ordered:

```text
B1 HDC statistical retrieval separation
B2 association vs causal/intervention evidence
B3 runtime maturity separation
B4 Phi / Mycelix / fact-check separation
B5 physics classifier + WASM convergence
B6 cross-producer negative authority theorem
```

R4 adds a repository-wide invariant for B6:

```text
no combination of descriptive proxies may mint E3/E4/N3/M3
without proposition-specific source evidence and admission.
```

## Relationship to #5337

Compatibility/profile migration must distinguish:

```text
parse / preserve / display
```

from:

```text
establish / verify / admit
```

A useful compatibility identity is conceptually:

```text
(profile_id, E, N, M, producer_ref)
```

but even that remains descriptive until the source evidence required by the profile has been verified.

## Relationship to MEL-EPI

Positive machine authority should be constructed from:

```text
source-native evidence
+ source-native verification/admission
+ exact source identity
+ explicit semantic profile
-> namespaced positive claim scope
```

Legacy enums, integers, parsers, conversions, scores, booleans, and caller tags cannot substitute for that chain.

## Qualification contract

Run:

```bash
python3 -m py_compile scripts/audit_epistemic_authority_semantics.py
python3 scripts/audit_epistemic_authority_semantics.py
```

Success is deliberately named:

```text
PASS_INVENTORY
```

not `PASS`.

The exact-head qualifier must require all of these to be empty:

```text
unexpected_empirical_authority_files
unexpected_high_axis_authority_files
unexpected_numeric_shape_files
missing_unsafe_producer_witnesses
missing_high_axis_producer_witnesses
missing_scalar_collapse_witnesses
missing_propagation_witnesses
missing_axis_collision_witnesses
missing_positive_control_witnesses
```

## Claim ceiling

`PASS_INVENTORY` can establish only that the exact frozen repository subject matches this reviewed inventory contract in that qualifying execution.

It does not establish:

- correctness of legacy E/N/M semantics;
- complete discovery of every possible authority encoding;
- cryptographic verification/authentication;
- reproducibility/replication;
- normative consensus or axiomatic truth;
- foundational materiality/durability;
- statistical calibration;
- causal validity;
- scientific truth;
- consciousness validity;
- governance legitimacy;
- MEL-EPI authority;
- physical/action authority.
