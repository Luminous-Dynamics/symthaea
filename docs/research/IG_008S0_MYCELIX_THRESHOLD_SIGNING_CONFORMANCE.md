# IG-008S0 — Mycelix threshold-signing cross-repository conformance

Issue: #3294

Parent: IG-008F0 / draft #3289

## Purpose

IG-008S0 independently reconstructs the frozen Mycelix threshold-signing authority counterexamples before threshold signing is admitted as a covered component of a later composite governance manifest.

It is a conformance measurement, not a signature-security theorem.

## Frozen Mycelix evidence

Exact evidence head:

`a580915d588338077ce6196514c43e24052f86cd`

Observed production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`

Source bindings:

```text
threshold-signing coordinator 3449df8b03a4dd1774a5f22756d06931c72855b2
threshold-signing integrity   3fec8344635600c044a494fa72bbbfe408fbe5ec
proposals consumer            eb8358353ee259ef9c3b46617a61d3439f1c714c
execution consumer            3dbb8a8f69b377e494ccf24164c94bd80f54e0ef
```

## Content-bound profile and corpus

```text
profile id      mycelix-threshold-signing-observed-fca2c107-v1
profile SHA     c15dfd860b759747938af2a13129d729fa0af1e75284418c9ea6b9c172f643ac
profile authority ObservedSourceBound

corpus SHA      0f6532ae8e2c2e421da625592dbb3b38aa2b90c5342f46f3a305bdbec89b0269
corpus authority MeasurementOnly
```

## Independent oracle

`scripts/ig008s0_mycelix_threshold_signing_oracle.py` is stdlib-only.

It does not import Mycelix's S0 validator, S1 oracle, threshold-signing Rust code, or a future Institutional Lab implementation.

It independently validates enough of the source-bound profile to reconstruct exactly:

- CE-SIG-01 structural zero-byte ECDSA fixture / cryptographic truth not established;
- CE-SIG-02 caller-stored `verified` does not affect the observed structural theorem;
- CE-SIG-03 committee lookup/epoch/threshold/membership/scope authority predicates are absent at signature-create validation;
- CE-SIG-04 ProposalToSignature association does not reconstruct exact subject authorization;
- CE-SIG-05 producer/consumer query-contract mismatch.

The independently produced canonical corpus must be byte-identical to the Mycelix S1 corpus.

## Qualification

The exact-head workflow:

1. checks out the exact Symthaea product head;
2. checks out Mycelix at exact S1 evidence head `a580915d...`;
3. binds all four threshold-signing/consumer source blobs;
4. syntax-compiles the Mycelix S0/S1 and independent Symthaea scripts;
5. runs Mycelix S0 validation;
6. runs Mycelix S1 self-test and corpus generation twice deterministically;
7. runs the Symthaea independent oracle twice deterministically;
8. requires byte-identical Mycelix/Symthaea corpus output;
9. asserts exact profile/corpus commitments and authority ceilings;
10. verifies both Git checkouts remain immutable.

## Composite consequence

IG-008F0's v1 composite remains immutable and intentionally lists threshold signing as uncovered.

After IG-008S0 earns the required evidence, a new composite **v2** may add a `ThresholdSigning` component and remove only that corresponding uncovered-stage entry.

Even then:

`Voting + Execution + ThresholdSigning != ObservedEndToEnd`

while proposal lifecycle, downstream action authorization, deployment currentness, or another required stage remains uncovered.

## Non-claims

IG-008S0 does not establish cryptographic signature validity, successful forgery, live threshold bypass, deployment currentness, governance safety, fairness, or constitutional legitimacy.
