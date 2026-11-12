# MACE-CoS Assurance Protocol v0.2

A research reference for distinguishing task success from trustworthy execution.

## Two independent outcome axes

| Task verdict | Control gate | Classification |
|---|---|---|
| pass | pass | clean success |
| pass | not_passed | task pass / controls not passed |
| fail | pass | task fail / controls passed |
| fail | not_passed | double failure |
| indeterminate | either | indeterminate, outside the 2×2 |

Evidence incompleteness remains distinct from a confirmed violation. Missing mandatory proof must not become a clean success, but absence of proof is not automatically proof of malicious conduct.

## Mandatory-control inventory closure

The classifier cannot establish completeness from whichever evidence rows happen to arrive. The matrix package therefore adds a separate preregistered ControlInventory and run-level ControlEvidence.

- InventoryDigest computes a deterministic SHA-256 over a typed, ID-sorted inventory.
- The harness supplies the expected digest independently of agent-controlled run evidence.
- SummarizeAgainstInventory rejects digest mismatch, duplicate/unknown evidence IDs and unknown verdicts; an omitted mandatory criterion becomes proof_incomplete_or_unobserved.
- not_applicable only satisfies a mandatory criterion when the inventory allowed N/A before the run and the evidence supplies a rationale.
- EvaluateAgainstInventory applies the task × control matrix only after inventory closure.

A digest is not a signature. Production use still requires an authenticated preregistration authority and an immutable record proving the expected digest was pinned before treatment assignment.

## Verify

Requires Go 1.23 or newer; no third-party modules.

Run: ./verify.sh

## Boundaries

This is a classifier and inventory-closure reference, not a trusted observer, runtime hook, sandbox, effect broker, cryptographic witness or external benchmark result. It does not prove that the preregistered inventory is complete for every deployment profile; profile-to-obligation compilation is tracked separately in [#11802](https://github.com/Luminous-Dynamics/symthaea/issues/11802).

Related work:
- [#11760 — repeated-run task × control assurance protocol](https://github.com/Luminous-Dynamics/symthaea/issues/11760)
- [#11763 — intention-to-treat skill effect estimator](https://github.com/Luminous-Dynamics/symthaea/issues/11763)
- [#11741 — isolated runtime provenance hooks](https://github.com/Luminous-Dynamics/symthaea/issues/11741)
