# CMAQ-047 — Controlled Blackwell betting with null-safe diagnostic actions

## Why this is the next paradigm shift

CMAQ-046 adapts the statistical witness direction to a multivariate conditional-mean restriction. It is passive: it observes whatever residual vector arrives.

The controlled Blackwell formulation in Zhao (2026) adds a second decision: after the predictable witness w_t is known, the controller selects the diagnostic experiment or data source whose null conditional-mean payoff remains inside the supporting half-space. This is the bridge from sequential monitoring to active experimental design.

CMAQ-047 implements a finite action-catalog protocol. The statistical monitor never invents an action contract and never authenticates its own authority. It can only select from pre-registered, externally attested contracts.

## Conditional-validity argument

Let S be the global compact convex target for residual means and let S_a be the action-specific conditional-mean set for action a. At time t, CMAQ-046 chooses w_t predictably from prior observations. The controller filters the catalog to actions satisfying

    h_(S_a)(w_t) <= h_S(w_t),

then deterministically selects one among eligible actions. The action is selected using only the pre-observation normal and frozen action metadata.

Under the action-specific null contract

    E[r_t | F_(t-1), a_t] ∈ S_(a_t),

we obtain

    E[<w_t,r_t> - h_S(w_t) | F_(t-1), a_t]
      <= h_(S_(a_t))(w_t) - h_S(w_t)
      <= 0.

Thus the support residual passed to the CMAQ-046 fixed-bet process remains conditionally nonpositive even though the experiment/action is chosen adaptively after the witness. The nonnegative wealth process retains its test-supermartingale / Ville guarantee, conditional on the truth and completeness of the action contracts and the observation norm bound.

This is the central distinction from “choose whichever sensor seems useful”: the action is only eligible if its conditional null mean is known to satisfy the current supporting half-space.

## Protocol and state machine

1. The monitor exposes its next predictable OCO normal.
2. It evaluates each frozen action-specific null box against the target support value for that normal.
3. It selects the admissible action maximizing the pre-registered alignment with the action's diagnostic signature. Ties resolve by lexicographically stable action ID.
4. It emits a plan bound to sequence, action ID, action contract, authority receipt, scope, target and catalog digests, plus the exact normal and support values.
5. The caller executes the selected action.
6. The caller reconciles the action result with the exact outstanding plan. The bounded vector is then passed to CMAQ-046, and only a successful observation clears the pending action.

The state machine permits one outstanding action. There is intentionally no cancel or “try another” operation while an action's outcome is unknown. A mismatched plan, wrong action ID or invalid observation leaves the original plan pending and the evidence state unchanged. This enforces the operating rule: unknown outcome means no new action until reconciliation.

## Action contract

Each catalog entry contains:
- stable action ID and action-scope ID;
- action-specific BoxTarget for the conditional mean under the null;
- pre-registered diagnostic signature used for deterministic action ranking;
- action-contract, authority-receipt and scope SHA-256 references.

The catalog itself binds the global target, the action inventory and the authority inventory. The monitor refuses to run if the target digest supplied at construction differs from the catalog target digest.

All SHA-256 values are shape-checked only. No signature verification is implemented here. The authority system must authenticate the receipts, prove their scope and validity interval, and justify that the action-specific conditional-mean envelope holds for all reachable histories covered by the contract. The software cannot prove a physical sensor or actuator is truthful.

## Implementation

Dependency-free except for the path dependency on CMAQ-046. It reuses that monitor's OCO normal, bounded residual check, fixed-bet wealth and Ville threshold.

The crate provides:
- immutable ActionCatalog with unique action IDs and frozen action contracts;
- support-function eligibility check for each proposed action;
- deterministic maximum-alignment action selection;
- one-use sequence-bound ActionPlan;
- exact action-ID and plan reconciliation;
- transactional observation handoff to CMAQ-046;
- one-pending-action fail-closed state machine;
- structural digest, dimension and diagnostic norm validation.

## Tests

Tests cover action support eligibility, action changes with the predictable witness, no second plan while an outcome is unknown, wrong action/plan rejection, invalid observation reconciliation, exactly-once updates, replay rejection, malformed/duplicate catalog entries, target/dimension mismatch, no-admissible-action failure and catalog immutability.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-047/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-047/Cargo.toml --locked --all-targets -- -D warnings

## Paired protocol probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-047/Cargo.toml --locked --bin controlled_action_probe -- 64 > cmaq-047-controlled-actions.csv

The executable visits four unknown directional alternatives and the action-conditional null. It logs each predictable normal, selected action, support slack, diagnostic alignment and log e-value. Under the null, each action generates a small bounded residual around a point inside its action-specific null box. Under alternatives, the residual mean moves in one of four directions. The first action is selected at the zero normal; later actions adapt to the OCO witness.

This is a protocol smoke test and an action-selection coverage diagnostic. It is not a powered false-alarm experiment or proof that an external device satisfies its action contract.

## Benchmark protocol

Compare adaptive null-safe action selection with:
- passive CMAQ-046;
- fixed diagnostic-action schedule;
- random choice among admissible pre-registered actions;
- oracle action choice (upper bound only; never a deployable comparator).

Use identical action contracts, residual sources, target, norm bound, alpha and scenario streams. Register null paths where action-conditional means satisfy each action contract, unknown-direction alternatives, action-specific sensor degradation, missing/late action results, and common-cause multi-sensor faults.

Report full-horizon null false alarms, power, H+1-penalized detection delay, action coverage/frequency, diagnostic utility, fallback completion, safety violations, unknown-result reconciliation frequency, p50/p95/p99 action-selection latency and evidence-ingestion lag. Use CMAQ-044 for paired effective-block uncertainty. Report simulator, hardware-in-the-loop and physical trials as separate tiers.

## Literature and claim boundary

Jinze Zhao (10 Aug 2026), *From Approachability Residuals to Anytime-Valid Evidence: The Online Convex Geometry of Testing by Betting*, arXiv:2608.09450. The paper formulates controlled stochastic experiments where an action selected after the predictable normal satisfies the supporting-halfspace condition for every null mean payoff. This implementation is a finite action-catalog protocol and uses CMAQ-046's fixed betting rate; it does not reproduce all optimized betting-regret rates or the full controlled Blackwell game.

CMAQ-047 is not an actuation safety certificate. The null guarantee relies on an authenticated action authority, true and complete action-conditional mean sets, predictable action selection, and a valid observation norm bound. Cryptographic verification, durable cross-process replay protection, action execution, actuator interlocks and physical safety validation remain outside this crate.
