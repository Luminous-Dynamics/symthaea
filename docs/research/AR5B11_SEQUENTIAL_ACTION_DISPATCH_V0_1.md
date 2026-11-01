# AR-5B11: Fail-Closed Sequential Action Dispatch v0.1

**Status:** implementation draft; tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → this branch.  
**Claim ceiling:** maps a finite sequential policy action ID to one typed EventTape action intent, verifies the exact recommendation/prefix, and checks a declared cost-unit crosswalk against remaining EventTape budget. It does not execute evaluators, authenticate the dispatcher, verify candidate bytes from a CAS, or prove that the cost crosswalk is economically calibrated.

## 1. Why action dispatch is a separate trust boundary

A digest-bound recommendation is not yet permission to run an arbitrary evaluator request. A production controller needs a typed mapping from a selected solver action to exactly one admissible EventTape action.

This tranche adds a mapping contract that binds:
- solver action ID;
- candidate ID and candidate content digest;
- objective ID;
- deterministic-vector versus noisy-replicate kind;
- parent candidate and semantic-delta digests;
- registered direction ID;
- objective-contract digest;
- decision digest, replicate ID and seed for noisy measurements;
- integer cost units and conversion into EventTape microunits;
- the decision-cost model and EventTape normalized-budget model identities.

The mapping must be complete: exactly one descriptor for every finite-problem action. Action IDs and measurement identities must be unique. The conversion is checked with overflow-safe integer multiplication.

## 2. Authorization procedure

Before authorizing a new measurement, the API:
1. Reconstructs and verifies the sequential recommendation against the exact current fully-qualified EventTape prefix and posterior binding.
2. Reconstructs the finite-horizon policy from the frozen problem.
3. Validates the complete action mapping and its cost-model identities.
4. Requires the mapping's EventTape budget-model digest to equal the tape header's digest.
5. Reconstructs the typed objective-contract digest and verifies mode/ordered objective membership.
6. Requires the mapped direction to be currently registered.
7. Verifies the action is selected and available at the policy root.
8. Checks that policy budget and selected action charge fit the remaining EventTape budget.
9. Rejects a measurement identity already present in the visible prefix, including previously failed or rejected attempts.
10. Emits a digest-bound dispatch authorization containing the exact ActionIntent, selected action ID, expected charge and visible-prefix digest.

If the solver chooses not to measure, the API emits an explicit terminal stop authorization. It does not create a fake measurement event.

## 3. Measurement identity and EventTape compatibility

EventTape v0.2 action variants do not carry a separate solver action ID. This implementation therefore requires a one-to-one mapping from action IDs to the stable typed intent identity:
- deterministic vector: candidate digest + objective-contract digest + direction ID;
- noisy replicate: candidate digest + objective-contract digest + decision digest + replicate ID + sample seed.

The mapping validator rejects aliases in these identity spaces. The next evaluator event can be reconciled to the unique mapped action without overloading semantic-delta fields or relying on free-form event IDs.

## 4. Cost boundary

The mapping declares how many EventTape budget microunits correspond to one finite-problem cost unit. It binds:
- the finite problem's cost model ID and digest;
- the EventTape header's normalized budget-model digest;
- the integer conversion factor;
- each action's converted expected charge.

The code verifies arithmetic and available balance. It does not prove that the declared conversion factor is economically correct. Before production, the conversion must be independently calibrated and tied to measured resource receipts, including failed attempts, verifier work and acquisition/controller overhead.

Unknown resource meters remain disqualifying because EventTape qualification rejects an incompletely metered tape.

## 5. Important non-claims

- The function returns an authorization; it does not invoke an evaluator, launch a job, or append an event.
- Candidate IDs/digests are bound in the mapping, but the descriptor's candidate bytes are not resolved against a CAS here.
- The inference-receipt digest is not authenticated; #9787's receipt resolution and verifier-attestation requirements remain open.
- A hash is not a signature, and a dispatch receipt is not proof of successful execution.
- This is not a production capability boundary until the caller/dispatcher authenticates the authorization and independently checks the current prefix immediately before launch.

## 6. Proposed acceptance gates (not results)

- Zero accepted stale-prefix, candidate, decision, objective, replicate, seed, direction, action or cost substitutions.
- Zero repeated measurement identities, including failed/invalid/verifier-rejected prior attempts.
- Zero arithmetic overflow or budget overruns.
- Every selected action maps to exactly one typed EventTape intent.
- Every stop decision maps to an explicit terminal-decision receipt and no evaluator action.
- Exact deterministic reconstruction of the dispatch authorization.
- Candidate artifact and inference artifact resolution plus authenticated verifier attestation before production qualification.
- Matched all-in cost conversion verified against actual EventTape resource receipts before cost-efficiency claims.

## 7. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering sequential_action_dispatch -- --nocapture
cargo test -p symthaea-architecture-engineering sequential_policy_event_tape -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_sequential_policy -- --nocapture
cargo test -p symthaea-architecture-engineering event_tape -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, test, clippy, CI, benchmark, or SOTA pass is claimed.
