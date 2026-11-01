# AR-5B9: EventTape-Bound Sequential Recommendations v0.1

**Status:** implementation draft; tests authored, not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → this branch.  
**Claim ceiling:** structurally binds a finite sequential policy recommendation to a fully metered, validated EventTape prefix and a declared posterior/inference receipt digest. It does not authenticate the inference receipt or prove the posterior was computed correctly.

## 1. Why this layer is necessary

The finite-horizon solver in #9754 takes an explicit finite posterior and returns a policy tree. Without a history binding, a recommendation could be replayed against a changed campaign or a different evidence prefix.

This tranche adds an EventTape authorization envelope that binds:
- campaign and source commit;
- policy ID and algorithm ID from the frozen EventTape header;
- exact visible event count and hash of the event prefix;
- exact EventTape header digest and last event-chain digest;
- objective-contract digest, when present;
- exact finite decision problem and sequential policy digests;
- posterior model digest and canonical posterior-distribution digest;
- inference-receipt artifact digest;
- remaining budget, selected action or terminal stop decision, expected risk and cost.

## 2. Fail-closed binding rules

The constructor:
1. Calls EventTape qualification validation; empty, incomplete, unknown-meter or unqualified tapes cannot drive this binding.
2. Canonicalizes the finite decision problem.
3. Independently reconstructs the sequential policy from that problem.
4. Requires EventTape, problem and policy campaign IDs to match.
5. Requires the posterior binding's history digest to equal the exact current EventTape prefix.
6. Requires posterior-model and posterior-distribution digests to equal the finite problem.
7. Requires a selected action to exist in the problem and available root actions, or an explicit terminal decision to stop.
8. Hashes the complete recommendation receipt.

Verification reconstructs the recommendation and rejects stale history, changed header, altered posterior, policy substitution, or receipt mutation.

## 3. Important trust boundary

The inference receipt is represented by a digest. The code checks that the caller binds it to the current history and posterior identity, but it does not resolve the receipt bytes, authenticate the inference service, or prove that the inference service computed the posterior correctly.

A production path must resolve and independently verify the inference receipt, bind the verifier identity and build, and authenticate the verifier attestation. A hash is not a signature.

The current function also does not map the selected finite action ID into a concrete EventTape action intent. That mapper must bind candidate digest, decision digest, replicate slot/seed, objective contract, declared incremental cost and available action namespace before dispatch. The next integration step is the actual online controller and action adapter.

## 4. Proposed acceptance gates (not results)

- Recommendation construction rejects unqualified or incomplete EventTape campaigns.
- Changing any event, header field, source commit, policy, posterior model, posterior distribution, inference receipt digest or selected action causes verification failure.
- A recommendation cannot be used after a new event is appended.
- Reconstructed recommendation bytes and digest are deterministic for the same exact prefix.
- Inference receipt resolution and verifier authentication are required before production qualification.
- The controller never dispatches an action that is not an admissible next action under the frozen policy and available budget.
- Unknown all-in costs force abstention or a clearly labelled unqualified result.

## 5. Exact-head validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering sequential_policy_event_tape -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_sequential_policy -- --nocapture
cargo test -p symthaea-architecture-engineering event_tape -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, format, clippy, CI, benchmark or SOTA pass is claimed.
