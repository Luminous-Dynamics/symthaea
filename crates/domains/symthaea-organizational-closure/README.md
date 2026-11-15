# symthaea-organizational-closure

Evidence-oriented primitives for the AOBench runtime self-maintenance track.

## Scope

This crate verifies a bounded functional self-maintenance contract:

- typed component classes prevent immutable roots and external resources from being counted as internally regenerated;
- process versions and frozen expected executable digests are explicit;
- artifact versions are immutable and keyed by component ID plus version;
- SHA-256 is recomputed over the stored bytes;
- a downstream consumer must reference the exact earlier verified output and producer execution;
- required external dependencies must produce an explicit runtime observation; optional dependencies may be absent;
- observed dependencies must use the process allow-list and available observations need a receipt digest;
- failed, abstained or interrupted events cannot expose committed outputs;
- required regenerative components need both verified producer and consumer evidence.

The report predicate is VerificationReport::qualifies_functional_self_maintenance(). It does not claim runtime organizational recurrence, causal closure, biological autopoiesis, life or consciousness.

## Run

From the package directory (this is an independently locked nested Cargo workspace):

```bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The focused workflow is .github/workflows/aobench-runtime.yml.

## Trust boundary

The verifier recomputes artifact content digests and checks the event's executable digest against the frozen process specification, but it does not authenticate the trace producer or prove that the executable actually ran. It does not validate external receipt signatures or make authorization decisions. The authorization_policy_digest is an identity reference only. Production adapters must bind events to an independently trusted runtime/logger and keep the verifier outside the agent's write authority.

This crate is a first implementation tranche. See docs/research/autopoiesis/RUNTIME_EVIDENCE_LAYER_2026-11-14.md and the [AOBENCH-001 issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).

## Organizational reconfiguration gate

The `reconfiguration` module compares a frozen `OrganizationSpec` with a candidate and reports the exact component/process/edge diff. It enforces explicit addition/removal/contract-change allowlists, a structural-edit budget, referential integrity, required-versus-optional edge consistency, and hard protection for provisioned roots, constraints, viability-critical components and operator-designated processes. Every newly added or changed process must match the frozen approved process version and executable digest; executable identity changes also require their own explicit allowlist. Existing process input edges to provisioned roots and constraints cannot be severed or weakened, and candidate processes cannot produce provisioned roots or externally governed constraints.

This is only a **structural admission** result. A passing candidate is not thereby authorized, viable, causally beneficial, or proven to have executed. Promotion still requires external authority and durable receipts; viability, runtime recurrence, intervention isolation and capability preservation remain separate evidence planes.

Focused API: `evaluate_reconfiguration(baseline, candidate, policy)`.

## Operational continuity witness

The `identity` module evaluates a finite labelled transition-system relation between a baseline and candidate organization. It distinguishes one-way `Simulation` (preserve baseline behavior while allowing candidate extensions) from two-way `Bisimulation` (both transition directions match and state-local capability sets are equal). It permits state-ID renaming, checks exact state observation labels, baseline capability preservation and the frozen required-invariant set, and returns `Inconclusive` if state-pair or transition-match budgets are exhausted.

A pass is relative to the declared finite models and action/observation alphabet. It is not universal behavioral equivalence, a metaphysical identity claim, or proof that the candidate was authorized or actually executed. Governance and evidence continuity still require the external authority and authenticated runtime-receipt planes.

Focused API: `evaluate_operational_continuity(baseline, candidate, policy)`.

## Stochastic sequential assay pilot

The `bernoulli_sequential` module adds a two-arm Bernoulli pilot with a time-uniform Hoeffding confidence sequence. It separates evaluator-only arm probabilities from agent-visible scenarios, requires assignment/delivery/isolation/observation evidence, charges every attempted sample against budgets, rejects premature claims and post-boundary sampling, and records coverage/false-positive scoring against hidden truth after the episode closes.

The confidence sequence allocates alpha/2 to each arm and spends each arm's error budget across qualified sample counts in proportion to 1/n^2. It is valid only for iid Bernoulli observations with predictable allocation and qualified sampling. Any unavailable or rejected evidence blocks a qualified conclusion; the module does not authenticate the receipts themselves.

This is not a general causal estimator, does not establish statistical power from a single episode, and does not cover drift, interference, outcome-dependent missingness, noncompliance estimands or multiple endpoints. Run the package's independent locked-workspace format, check, test and Clippy commands before treating the implementation as validated.

## Suite-level stochastic metrics

The `bernoulli_benchmark` module aggregates frozen per-episode reports without combining distinct scientific outcomes into one score. It reports null false-positive rate, qualified power on materially non-null episodes, final and all-interim-look coverage, claim accuracy, process qualification, and mean attempted samples/cost/risk. Proportions include 95% Wilson score intervals; invalid-oracle episodes remain in process totals but are excluded from truth-conditioned rates.

These are episode-level summaries. Generator-family clustering and independent Monte Carlo runs are still required for confirmatory claims; Wilson intervals do not correct for clustered episodes or benchmark leakage.
