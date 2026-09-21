# EPI-SEM-001A-R2 — Legacy epistemic authority semantic audit

Parent: #5283 (`EPI-SEM-001`)

Supersedes unqualified #5326/#5327 before execution.

## Purpose

Freeze the actual cross-repository semantic surface of Symthaea's legacy E/N/M epistemic classifications before changing behavior.

R1 was intentionally closed after static review found that scanning only the literal spellings `E3CryptographicallyProven` / `E4PubliclyReproducible` missed a second authority-producing implementation in the Symthaea↔Mycelix bridge.

R2 therefore audits four distinct layers:

1. authority-bearing vocabulary aliases;
2. proxy-to-authority producers;
3. conversion/gating surfaces that propagate those values;
4. one already-hardened positive-control implementation.

This remains an **inventory**, not a semantic repair.

## Core theorem

```text
statistical separation
!= cryptographic verification

consciousness integration / Phi
!= cryptographic verification

open-source availability
!= independent reproduction

caller says `is_reproducible = true`
!= independent reproduction evidence

observation count + derived confidence
!= cryptographic proof

runtime cycle count
!= cryptographic proof

catalog similarity
!= social consensus
!= axiomatic truth

domain category
!= evidence that a claim is foundational/permanent

E/N/M numeric compatibility
!= semantic equivalence
```

## Current semantic families

### A. Legacy consciousness / shared ledger vocabulary

The consciousness layer and `symthaea-epistemic-types` use E3/E4 names such as:

```text
E3CryptographicallyProven
E4PubliclyReproducible
```

Known problematic producers include:

- HDC z-score thresholds;
- causal relation evidence-count + derived-confidence thresholds;
- cognitive-loop runtime cycle count.

The consciousness coordinate also combines E/N/M into `quality_score()` and `contextual_quality_score()`. Such a scalar may be useful as legacy UI/heuristic metadata, but it cannot establish any individual authority property.

### B. Symthaea↔Mycelix bridge vocabulary

`src/mycelix/types.rs` defines a second E/N/M veneer with:

```text
CryptographicallyVerifiable
PubliclyReproducible
```

and adapters to `mycelix_sdk::epistemic::{E3Cryptographic,E4PublicRepro}`.

Two authority-production paths are especially important:

```text
Phi >= threshold
-> CryptographicallyVerifiable
```

and:

```text
Phi + caller-provided is_reproducible bool
-> PubliclyReproducible
```

Neither input establishes the named verification theorem.

`src/mycelix/mapper.rs` repeats this mapping and also treats a plain `EvidenceType::CryptographicProof` tag as enough to suggest E3. The evidence object is ordinary caller data; an enum tag is not signature/ZKP verification.

The integration test suite currently freezes these Phi→E3/E4 outcomes as expected behavior, so the eventual repair must update tests deliberately rather than leaving contradictory compatibility tests behind.

### C. Physics discovery duplication

Two physics discovery implementations currently disagree semantically.

`crates/domains/symthaea-physics-catalog/src/discovery.rs` can assign:

```text
open source + simulation confidence > 0.8 -> E4
catalog similarity > 0.9               -> N3
physics domain                          -> M tier
```

Those mappings overclaim what the inputs establish.

By contrast, `crates/bridges/symthaea-physics-bridge/src/discovery.rs` has already been hardened so a local simulation supports at most E1 without replay/integrity/reproduction evidence, and it carries a regression named `open_source_and_confidence_do_not_imply_reproduction`.

The bridge implementation is therefore frozen as a **positive control**, not migration debt.

## Cross-axis semantic collision

The audit also freezes one larger problem that must not be hidden behind matching integer values.

The shared ledger describes N as:

```text
WHO agrees this knowledge claim is valid?
```

while the Symthaea↔Mycelix bridge describes N as:

```text
Who should have access to this?
```

and derives it from `WorkspaceScope`.

Those are different propositions.

Likewise the shared ledger's M axis describes permanence of knowledge while the bridge uses M as a persistence/TTL policy derived from `importance`.

The eventual migration must decide the canonical meaning of each axis or version the profiles explicitly. Numeric `N0..N3` / `M0..M3` equality is not enough to justify conversion.

## Compatibility and propagation seams

R2 freezes these separately from producers:

- local consciousness `to_shared()` E/N/M conversion;
- Mycelix `EpistemicClassification::from_code()` wire parsing;
- Mycelix-sdk conversion adapters;
- Mycelix stub `meets_standard()` ordinal gate;
- integration tests that currently normalize Phi→E3/E4 behavior.

A parser may continue to read legacy E3/E4 values for compatibility. Reading a value must not be confused with independently establishing the property represented by its label.

## Mechanical audit

`scripts/audit_epistemic_authority_semantics.py` scans Rust source under:

```text
src/
crates/
apps/
tests/
benches/
examples/
stubs/
```

for known authority aliases:

```text
E3CryptographicallyProven
E4PubliclyReproducible
CryptographicallyVerifiable
PubliclyReproducible
E3Cryptographic
E4PublicRepro
```

Any new file using these markers outside the frozen reviewed vocabulary surface produces `REVIEW_REQUIRED`.

The script additionally requires exact witnesses for:

- known unsafe producers;
- propagation/gating seams;
- the N/M definition collision;
- the hardened physics positive control.

Run:

```bash
python3 scripts/audit_epistemic_authority_semantics.py
```

Success is deliberately named:

```text
PASS_INVENTORY
```

not `PASS`.

## Relationship to MEL-EPI

The MEL-EPI direction remains the correct authority boundary:

```text
source-native evidence
+ source-native verification/admission
-> exact source reference
-> positive closed-world claim scope
```

Therefore:

```text
legacy E3/E4 ordinal
!= MEL-EPI claim

legacy E/N/M quality score
!= MEL-EPI claim

legacy deserialization/conversion
!= source-native verification
```

The repaired MEL-EPI semantic-identity lineage has also moved beyond #5290 to #5312; no EPI-SEM repair should consume that source until its own exact-head qualification is established.

## Relationship to other Symthaea authority repairs

#1276 independently reaches the same architectural principle for embodiment:

```text
high Phi
!= positive physical-safety authority
```

EPI-SEM should generalize that discipline to epistemic authority:

```text
internal cognitive metric
may restrict / inform
but cannot independently mint an external verification theorem
```

## Successor architecture

Do **not** replace E0-E4 with another universal ordinal that again mixes unrelated properties.

Recommended migration sequence:

### EPI-SEM-001B — producer-specific descriptive outputs

Replace proxy-driven E3/E4 production with proposition-accurate outputs at the owning source:

- HDC z-score -> retrieval/statistical-strength result;
- causal evidence heuristic -> causal-support heuristic/result;
- cycle count -> runtime maturity/observation count;
- Phi -> cognitive integration/readiness metadata;
- physics catalog confidence/similarity -> simulation-confidence / analogy metadata.

No producer-specific proxy should mint crypto/reproduction authority.

### EPI-SEM-001C — compatibility projection

Keep legacy E/N/M wire values readable where required, but classify them as legacy descriptive data. Conversion must not create a verified positive claim.

### EPI-SEM-001D — source-verified positive properties

Where cryptographic verification, reproducibility, replication, or public reproduction are actually needed, derive them only from exact evidence/receipts and preferably expose them through MEL-EPI namespaced claim scopes rather than a new overloaded ordinal.

## Qualification / nonclaims

This audit does not establish:

- that every E/N/M use has been semantically repaired;
- the canonical future meaning of N or M;
- cryptographic proof/authentication;
- reproducibility/replication;
- statistical or causal validity;
- scientific truth;
- consciousness validity;
- MEL-EPI claim authority;
- action, economic, governance, or physical authority.

`PASS_INVENTORY` establishes only that the frozen source still matches the reviewed inventory contract.
