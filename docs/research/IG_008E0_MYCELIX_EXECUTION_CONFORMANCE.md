# IG-008E0 — Mycelix execution cross-repository conformance

## Status

Research / `MeasurementOnly` / `CrossImplementationConformance`.

This tranche is the execution-side companion to IG-008A0. It does not claim that the observed Mycelix execution mechanism is safe, current in deployment, or exploitable. It proves only that two independent research implementations agree byte-for-byte on the frozen source-bound execution-authority observations and counterexamples.

## Parent lineage

Symthaea parent:

- IG-008A0 / draft #3233
- exact parent head `358057ec0a7db5e8af377de93c3cad808ced294a`

Mycelix execution evidence:

- IG-007E0 / draft #908 — observed execution profile
- IG-007E1 / draft #910 — execution counterexample corpus
- exact evidence head `197714209c60503f0fba4143409da383bc9cbf83`

## Frozen Mycelix source binding

Observed production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`

Exact source blobs:

- execution coordinator: `3dbb8a8f69b377e494ccf24164c94bd80f54e0ef`
- execution integrity: `657edaee9a314f100a0c4b1609a4596cf243e61d`

## Content-bound execution profile

Profile:

- id: `mycelix-execution-observed-fca2c107-v1`
- revision: `1`
- authority: `ObservedSourceBound`
- content SHA-256: `c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6`

Frozen counterexample corpus SHA-256:

`0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4`

## Independent implementation

Symthaea owns:

`scripts/ig008e0_mycelix_execution_oracle.py`

The oracle is stdlib-only and does not import Mycelix validators, Mycelix counterexample code, Mycelix Rust execution modules, or Symthaea institutional-lab implementation code.

It independently validates the exact profile identity/source binding and reconstructs:

- `CE-TL-01` — missing proposal/action/policy binding in observed timelock construction;
- `CE-TL-02` — creator-only `Pending -> Ready` local predicates without an observed signature predicate;
- `CE-TL-03` — Ready/Pending signature-control-flow differential;
- `CE-TL-04` — executable action dispatch surface;
- `CE-TL-05` — unavailable threshold-signing authority warns and continues source control flow.

## Qualification theorem

A qualified IG-008E0 run establishes only:

```text
same exact source-bound execution profile
+ independent Mycelix corpus implementation
+ independent Symthaea corpus implementation
+ canonical byte equality
+ frozen corpus commitment
= CrossImplementationConformance
```

It does **not** establish:

```text
mechanism safety
live exploitability
downstream authorization failure
deployment currentness
constitutional legitimacy
behavioral validity
```

## Cross-repository qualifier

The exact-head workflow must:

1. check out the exact Symthaea PR subject;
2. check out `Luminous-Dynamics/mycelix` at exact evidence head `197714209c60503f0fba4143409da383bc9cbf83` into an isolated nested path;
3. verify both checkout SHAs;
4. verify the exact execution coordinator/integrity Git blob ids;
5. syntax-compile the independent Symthaea and Mycelix Python implementations;
6. run the Mycelix E0 profile validator;
7. run the Mycelix E1 oracle twice byte-identically and emit its canonical corpus;
8. run the Symthaea oracle twice byte-identically and emit its canonical corpus;
9. require complete byte equality of the two canonical corpus files;
10. assert exact profile/corpus commitments and authority ceilings;
11. verify both working trees remain immutable.

No branch-tip rebinding to `main` or `latest` is allowed. No post-hoc normalization is allowed if the two independent outputs diverge.

## End-to-end Mycelix modeling rule

Once IG-008A0 and IG-008E0 are both qualified, any Symthaea experiment claiming to model the observed end-to-end Mycelix governance mechanism should carry at least two independently content-bound references:

```text
VotingProfileRef
ExecutionProfileRef
```

A simulation that uses the observed voting profile but an idealized or successor execution model is still useful, but must declare that substitution explicitly. It must not be labeled as an exact observed end-to-end Mycelix mechanism.

## Successor semantics

When Mycelix repairs #904, the corrected execution mechanism must receive a new content-bound production profile and new conformance receipt. Historical E0/E1 evidence remains attached to the old subject.

The desirable successor should intentionally cease reproducing the old execution-authority counterexamples where the repair applies.

## Non-claims

This tranche performs no live financial or constitutional mutation and makes no claim that downstream mutation targets would accept unauthorized execution attempts.