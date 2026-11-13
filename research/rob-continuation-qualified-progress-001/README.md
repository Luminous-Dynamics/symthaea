# ROB-CONTINUATION-QUALIFIED-PROGRESS-001

Status: deterministic descriptive reference for safe accepted subgoal progress that preserves mandatory mission continuation.

Parent issue: Symthaea #11957. Compose with Symtropy #4918 (future-option ledger), Symthaea #11664 (causal mission replay), #11869 (terminal capability retention), #11932 (source-transfer validity) and #11807 (cluster inference).

## Why this benchmark is different

LADDER reports task completion and subgoal progress for 1–10 subgoal tasks. Progress alone can hide a dead end: a robot may complete a prefix but eliminate the only feasible path through the mandatory remainder. This evaluator therefore reports three separate quantities: native task completion, safe accepted progress and continuation-qualified safe progress.

Continuation-qualified progress requires an accepted postcondition, safe execution, exact causal and option-ledger transition identity, and a qualified upstream receipt covering every remaining mandatory successor from the exact terminal physical state.

## Mission graph and receipt contract

The manifest declares unique subgoal IDs, mandatory/optional labels, a unique mission order and a directed acyclic prerequisite graph. A successful subgoal requires independent acceptance and safety receipt identities plus a causal physical-transition receipt. The causal predecessor/result state digests must exactly match the option-ledger predecessor/result digests.

The continuation receipt must bind to the exact physical result state, name the exact remaining mandatory descendants in the mission DAG, name a feasible set that is a subset of those mandatory descendants, and carry an upstream option-ledger/viability witness and disposition.

A Qualified receipt must cover the entire mandatory successor set. An Insufficient receipt may cover only a subset and never earns continuation-qualified progress. Unknown/missing stays inconclusive. Rejected is rejected. Optional successors do not block a mission's mandatory continuation.

## Outputs

Per subgoal: outcome, independent acceptance, safe acceptance, physical/option state join, continuation-qualified result, continuation Unknown, required mandatory successors and feasible successor count.

Per mission: scheduled subgoals, native completions, safe accepted subgoals, successful-but-unsafe outcomes, continuation-qualified subgoals, Unknowns, longest safe prefix, longest continuation-qualified prefix, mandatory safe completion and mandatory continuation completion.

No weighted scalar is emitted. No inferential result is emitted. Mission/subgoal events are not independent statistical units; #11807 remains the inferential owner.

## Limits

The reference does not compute viability, option sets, route connectivity, physical safety or task acceptance. It validates typed upstream receipt identities and their coverage. The soundness of the future-option / continuation oracle remains an independent obligation. It never grants actuation authority.

## Run

    cargo fmt --manifest-path research/rob-continuation-qualified-progress-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-continuation-qualified-progress-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-continuation-qualified-progress-001/reference/Cargo.toml --all-targets -- -D warnings

17 synthetic known-answer/adversarial tests are authored; qualification requires the exact-head Rust 1.96 workflow.
