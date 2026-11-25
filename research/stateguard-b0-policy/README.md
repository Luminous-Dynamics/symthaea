# StateGuard B0 — Public-State Research Policy Baseline

Issue: [#13666](https://github.com/Luminous-Dynamics/symthaea/issues/13666)

StateGuard B0 is the first deterministic candidate-facing policy baseline for the ResearchState contract. It consumes:
- an exact serialized ResearchState snapshot;
- the current action offers that the upstream AgentView/permission authority has already permitted;
- a componentwise remaining-resource budget.

It returns a selected action-offer ID, ABSTAIN, or STOP. It never executes tools, creates permissions, mutates evidence, performs claim entailment or adjudicates truth.

## Selection rules

Priority order:
1. Repair broken artifact capture/attribution or integrity.
2. Refresh stale evidence or refine out-of-scope claims.
3. Classify unclassified claim-source pairs.
4. Resolve material support/refutation conflicts with a discriminating experiment, independent source or replication.
5. For HIGH/CRITICAL decisions, request independent positive adjudication when required.
6. Seek more independent evidence or abstain when decision usability remains insufficient.
7. STOP only when every claim is SUFFICIENT for the active decision under the validated ResearchState contract.

The policy uses only offers supplied by the caller. No matching/feasible offer means ABSTAIN. It does not assume an offered action succeeded; the caller must provide a new ResearchState after the executor reports its result.

## Cost semantics

Action costs are integer vectors. Every offer must name exactly the same resource dimensions as the remaining budget. Feasibility is componentwise; no weighted sum is introduced. Among same-intent offers, dominated vectors are discarded. If multiple Pareto-incomparable offers remain, StateGuard abstains instead of inventing a tradeoff weight.

## Output binding

Every decision includes:
- policy ID/version;
- exact serialized ResearchState digest;
- upstream evidence-prefix digest;
- action-offer-list digest;
- selected offer ID/kind (only for SELECTED);
- target claim/evidence;
- reason codes and copied resource estimate.

The caller must record these unchanged in the existing #11015 prospective trace.

## Local execution

```bash
cargo fmt --manifest-path research/stateguard-b0-policy/Cargo.toml -- --check
cargo test --manifest-path research/stateguard-b0-policy/Cargo.toml --locked --all-targets
cargo run --manifest-path research/stateguard-b0-policy/Cargo.toml --locked -- policy-input.json
cargo clippy --manifest-path research/stateguard-b0-policy/Cargo.toml --locked --all-targets -- -D warnings
```

## Qualification and claim boundary

This is a transparent rule-based B0 comparator, not a learned scientific researcher or a claim of SOTA. Its tests are synthetic and remain uncompiled/unexecuted until the dedicated Rust 1.96 workflow runs. It must be compared with fixed/random, threshold, EIG, EVSI, minimax, expert and frontier-model policies under matched allowed actions, model, task and inclusive resource limits.

The exact sequential oracle #13567/#13571 is evaluator-only and must never be linked into this crate or candidate process.
