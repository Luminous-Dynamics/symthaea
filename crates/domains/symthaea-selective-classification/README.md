# symthaea-selective-classification

Selective classification evidence with explicit abstention.

The crate is designed for high-consequence perception systems where a classifier
must be allowed to say **"I do not know"** instead of being forced to pick one
label.

## Evidence contract

Each classification result binds:

- immutable model identity/digest
- exact calibration release
- qualified deployment domain
- observation time and freshness envelope
- support for each considered label
- a set-valued prediction
- epistemic uncertainty
- out-of-distribution score
- evidence provenance

A deployment-reviewed policy binds the exact expected model, calibration, and
domain and defines uncertainty/OOD/support limits. There are intentionally no
safety-critical defaults.

## Outcomes

```text
EvidenceUsable
Abstain
Incomplete
```

`EvidenceUsable` does **not** mean a singleton or certain answer. A prediction
set such as `{ small-aircraft, bird }` can remain valid evidence while preserving
ambiguity.

OOD, excessive epistemic uncertainty, weak prediction-set support, or an overly
broad prediction set force abstention. Stale evidence, model drift, calibration
drift, deployment-domain mismatch, or structurally invalid evidence make the case
`Incomplete`.

No assessment grants physical authority.

## Verification

```bash
cargo test -p symthaea-selective-classification
```
