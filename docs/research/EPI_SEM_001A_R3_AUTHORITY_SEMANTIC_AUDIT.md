# EPI-SEM-001A-R3 — Cross-system epistemic authority semantic inventory

Parent: #5283 (`EPI-SEM-001`)

Supersedes incomplete, unqualified R1 (#5326/#5327) and R2 (#5331/#5333).

## Purpose

Freeze the actual legacy epistemic-authority surface before any producer repair changes behavior.

R2 improved the audit from literal E3/E4 names to aliases, but an independent preflight found another producer in:

```text
src/consciousness/mycelix_bridge.rs
```

That file demonstrated a more general failure mode:

```text
semantic authority can hide behind numeric E/N/M coordinates
without using an E3/E4 spelling at the production site
```

R3 therefore inventories both:

1. authority-bearing vocabulary aliases; and
2. local/numeric E/N/M classification shapes.

This remains an inventory, not a repair or authority grant.

## Core theorem

```text
internal metric
!= external verification theorem

z-score
!= cryptographic proof
!= reproduction

Phi / cognitive integration
!= cryptographic proof
!= reproduction

runtime duration
!= cryptographic proof

observation count
!= intervention
!= counterfactual proof

caller boolean
!= reproduction evidence

evidence enum tag
!= verification of the evidence artifact

fact-check scalar
!= cryptographic proof
!= public reproduction

authenticity score
!= cryptographic verification

open-source availability
!= independent reproduction

simulation confidence
!= public reproduction

catalog similarity
!= consensus
!= axiom

workspace/access scope
!= corroboration

storage importance / retention
!= epistemic permanence
!= foundational truth

same E/N/M integer
!= same proposition

weighted E/N/M scalar
!= any component authority theorem
```

## Why R2 was superseded

R2's alias scan omitted:

```text
src/consciousness/mycelix_bridge.rs
```

That file contains at least two additional authority-producing families.

### Fact-check scalar promotion

`FactcheckEpistemicFeedback::from_factcheck()` converts continuous values into discrete authority-shaped tiers:

```text
empirical >= 0.8 -> E4-like numeric tier 4
empirical >= 0.6 -> E3-like numeric tier 3
normative >= 0.75 -> N3 "axiomatic"
mythic >= 0.75 -> M3 "foundational"
```

`from_verdict_string()` can manufacture those continuous coordinates from a verdict label; for example `"True"` supplies an empirical value of `0.9`, which then reaches tier 4.

None of those steps independently establishes public reproduction, cryptographic proof, an axiom, or foundational status.

### Authenticity-to-crypto promotion

The outward Mycelix claim path contains:

```text
eval.authenticity > 0.8
-> EmpiricalLevel::E3Cryptographic
```

An authenticity/evaluation score is not cryptographic verification.

R3 freezes both producer families explicitly.

## Additional bridge family discovered during R3 preflight

The repository also has:

```text
crates/bridges/symthaea-mycelix-bridge/src/lib.rs
```

with a local E/N/M representation inlined to avoid a dependency on the full Symthaea crate.

Its empirical side is already usefully bounded:

```text
validation observed -> at most E1 testimonial
stronger levels require replay / crypto / reproduction evidence
```

That is a positive control.

However, the same bridge still maps:

```text
WorkspaceScope::Universal
-> LocalNormativeLevel::Foundational
-> N3Axiomatic

importance >= 0.75
-> LocalMaterialityLevel::Permanent
-> M3Foundational
```

So it is simultaneously:

- a positive control for bounded empirical authority; and
- semantic migration debt for N/M.

This is why R3 separates positive-control witnesses from cross-axis promotion witnesses.

## Inventory dimensions

### 1. Authority vocabulary surface

R3 scans Rust source for known aliases:

```text
E3CryptographicallyProven
E4PubliclyReproducible
CryptographicallyVerifiable
PubliclyReproducible
E3Cryptographic
E4PublicRepro
```

A new file using those terms outside the reviewed set yields:

```text
REVIEW_REQUIRED
```

### 2. Numeric/local classification surface

R3 independently scans for known authority shapes that can avoid those spellings:

```text
pub empirical: u8
pub e_tier: u8
struct LocalEpistemicClassification
enum LocalEmpiricalLevel
```

This second surface is deliberately a regression ratchet, not a proof that every conceivable numeric encoding has been discovered.

### 3. Unsafe producer witnesses

The audit freezes the currently known proxy-driven producers:

```text
HDC z-score
-> E3/E4 names

causal observation-count + derived confidence
-> E3 name

runtime cycle count
-> E3 name

Phi
-> E3

Phi + caller reproducibility bool
-> E4

plain EvidenceType::CryptographicProof tag
-> E3 suggestion

fact-check scalar / verdict label
-> numeric E3/E4/N3/M3

authenticity score
-> E3Cryptographic

open-source + simulation confidence
-> physics E4

catalog similarity
-> physics N3
```

Presence means "known debt", never "approved behavior".

### 4. Cross-axis promotions

R3 freezes cases where evidence for one proposition is used to raise another axis.

Examples include:

```text
consensus-tag count
-> empirical E2 suggestion

importance
-> permanent/foundational M

workspace/access scope
-> normative N

continuous normative scalar
-> N3 axiomatic

continuous mythic scalar
-> M3 foundational
```

These should be repaired under #5334/#5337 rather than hidden by preserving the old integer.

### 5. Scalar collapse

Two known surfaces combine E/N/M into one quality scalar.

A scalar may remain useful for UI, sorting, or bounded heuristics, but:

```text
high combined score
!= cryptographic verification
!= reproduction
!= consensus
!= durability
```

Consumers must not reconstruct component authority from the aggregate.

### 6. Propagation and export seams

R3 separately freezes consumers/exports that can preserve a bad classification:

- local consciousness -> shared epistemic conversion;
- legacy wire parsing;
- Mycelix SDK ordinal gates;
- tests normalizing Phi -> E3/E4;
- browser/WASM physics claim classification.

The WASM surface is important because the unsafe physics catalog classifier is externally reachable through:

```text
classify_physics_claim(description, confidence, is_open_source)
```

and defaults an unknown claim to:

```text
Domain::ModifiedGravity
```

That product boundary belongs in the later physics convergence repair.

### 7. Positive controls

R3 requires two bounded examples to remain visible.

#### Physics bridge

A local simulation is intentionally held to E1 absent replay/integrity/reproduction evidence.

#### Symthaea–Mycelix bridge

Observed local validation is intentionally held to E1; stronger empirical levels require stronger source evidence.

These are useful migration patterns:

```text
descriptive local evidence
-> bounded descriptive claim

not

descriptive local evidence
-> unrelated external authority theorem
```

## Axis-definition collision

At least three semantic families coexist.

### Shared epistemic types

N means approximately:

```text
who agrees this knowledge claim is valid?
```

M means:

```text
how permanent is this knowledge?
```

### `src/mycelix/types.rs`

N means:

```text
who should have access?
```

M means:

```text
how long should this persist?
```

### consciousness fact-check bridge

N is derived from a continuous `normative` score described as source consensus.

M is derived from a continuous `mythic` score described as persistence/foundationality.

Therefore:

```text
N2(profile A) != N2(profile B)
M3(profile A) != M3(profile B)
```

without an explicit semantic profile proving equivalence.

This is the reason for #5337.

## Relationship to #5334

#5334 is the producer-repair train.

Recommended order remains:

```text
B1 HDC retrieval significance
B2 observed association vs causal evidence
B3 runtime maturity
B4 Phi / Mycelix / fact-check boundaries
B5 physics classifier convergence + WASM boundary
B6 cross-producer negative authority regression
```

R3 adds two requirements to that plan:

- B4 must include `src/consciousness/mycelix_bridge.rs`, not only `src/mycelix/{types,mapper}.rs`;
- B5 must include the browser/WASM export, not only the two classifier implementations.

## Relationship to #5337

#5337 owns semantic profile/versioning and compatibility projection.

The newly found `crates/bridges/symthaea-mycelix-bridge/src/lib.rs` strengthens that issue:

```text
WorkspaceScope::Universal != N3 axiom
importance >= 0.75        != M3 foundational truth
```

The empirical E0/E1 cap in that file should be preserved as a positive-control pattern while N/M are separated.

## Relationship to MEL-EPI

Legacy E/N/M is compatibility/descriptive data.

Positive machine authority should follow:

```text
source-native evidence
+ source-native verification/admission
+ exact source identity
+ explicit semantic profile
-> namespaced positive claim scope
```

Therefore:

```text
legacy enum
legacy numeric tier
legacy parser
legacy conversion
legacy scalar score
caller boolean
caller evidence tag
```

cannot independently establish a MEL-EPI positive claim.

## Qualification contract

Run:

```bash
python3 -m py_compile scripts/audit_epistemic_authority_semantics.py
python3 scripts/audit_epistemic_authority_semantics.py
```

The successful result is deliberately:

```text
PASS_INVENTORY
```

not `PASS`.

The qualifier must require all of these to be empty:

```text
unexpected_authority_vocabulary_files
unexpected_numeric_shape_files
missing_unsafe_producer_witnesses
missing_cross_axis_promotion_witnesses
missing_scalar_collapse_witnesses
missing_propagation_witnesses
missing_axis_collision_witnesses
missing_positive_control_witnesses
```

## Claim ceiling

`PASS_INVENTORY` may establish only that the exact frozen source matches this reviewed inventory contract in the qualifying execution.

It does not establish:

- semantic correctness of legacy E/N/M;
- completeness of every possible epistemic encoding;
- cryptographic proof/authentication;
- reproducibility or replication;
- causal validity;
- statistical calibration;
- scientific truth;
- consciousness validity;
- governance legitimacy;
- MEL-EPI authority;
- physical/action authority.

The inventory exists to make repairs safer, not to certify the legacy model.
