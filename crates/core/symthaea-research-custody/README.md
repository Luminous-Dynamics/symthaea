# symthaea-research-custody

Content-addressed policy/provenance for **who may access which research artifact, for what action, and from which experiment phase onward**.

This crate exists because a statistically clean split is not enough if held-out outcomes leak operationally to the model, researcher, tuning loop, or scoring service before predictions are frozen.

It is intentionally **not** an OS sandbox or cryptographic capability system. It records and validates the intended custody contract and access receipts. A future Xenia adapter can enforce the same contract at the actual access-control boundary.

## Predictor input != verification outcome

Final Evaluation data may contain multiple assets with different access schedules.

For example:

```text
Sentinel evaluation scene
  predictor input imagery       -> model may read after EvaluationInputsOpen
  future wetland observation    -> model must not read until OutcomeRevealed
  ground-truth field label      -> model must not read until OutcomeRevealed
```

Treating all Evaluation bytes as either public or secret is too coarse.

## Phases

```text
Development
  ↓
SelectionFrozen
  ↓
EvaluationInputsOpen
  ↓
OutcomeRevealed
  ↓
Published
```

The phase name alone is not trusted as proof. Every access receipt also binds a `phase_evidence_digest` referencing the external artifact that justifies the phase, such as:

- a frozen model-selection manifest;
- a signed forecast/output commitment;
- a verification receipt;
- a final result manifest.

This crate records that reference but does not independently validate the external artifact yet.

## Principals

v1 names four logical roles:

- `ModelProcess`
- `Verifier`
- `ResearchOperator`
- `Public`

These are semantic roles, not authenticated identities. Xenia or another enforcement layer should later bind them to actual principals/capabilities.

## Hard sealed-outcome rules

For `VerificationOutcome` and `GroundTruthLabel` assets, custom access policies cannot weaken several invariants:

- `ModelProcess` cannot read/transform before `OutcomeRevealed`;
- `ResearchOperator` cannot read/transform before `OutcomeRevealed`;
- no principal may score the sealed outcome before `OutcomeRevealed`;
- public reveal cannot occur before `Published`.

A trusted `Verifier` may hold/read the hidden outcome before reveal. That allows an independent evaluator to possess ground truth without making it available to the model or tuning loop.

## Access receipts

Every permitted access can produce an immutable `AccessReceipt` binding:

- custody-manifest digest;
- asset id and asset digest;
- principal role;
- action;
- experiment phase;
- phase-evidence digest;
- timestamp;
- receipt digest.

This gives a later audit something concrete to reconstruct instead of relying on claims such as “the test labels were not used.”

## Sentinel / Wetland Watch example

A future locked evaluation can use:

```text
ResearchSplitManifest
       ↓
ResearchCustodyManifest
       ├─ eval-input-S2-...
       ├─ eval-input-S1-...
       └─ eval-outcome-...

selection receipt frozen
       ↓
EvaluationInputsOpen
       ↓
model receives allowed input assets
       ↓
forecast/output committed
       ↓
OutcomeRevealed
       ↓
verifier scores hidden outcome
```

For a pure forecasting experiment, even the future Sentinel observation itself can be a `VerificationOutcome` and remain sealed until after the forecast.

## Future Xenia enforcement

The intended later mapping is roughly:

```text
CustodyPrincipal + CustodyAction + CustodyPhase
                    ↓
             Xenia capability
                    ↓
          actual file/object access
```

Example model capability:

```text
read evaluation predictor inputs
submit prediction

NOT:
read verification outcomes
read labels
score final evaluation
change custody policy
```

That would turn the present evidence contract into a mechanically enforced scientific firewall.

## Non-claims

This crate does not prove that:

- the filesystem/object store actually enforced the rules;
- a principal has no unrelated ambient data access;
- a digest provides secrecy;
- a verifier is organizationally independent;
- phase evidence is authentic;
- evaluation labels are scientifically correct.

Those require enforcement, identity/authority, independent custody, and domain validation.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-custody --all-targets
cargo test -p symthaea-research-custody
cargo clippy -p symthaea-research-custody --all-targets -- -D warnings
```
