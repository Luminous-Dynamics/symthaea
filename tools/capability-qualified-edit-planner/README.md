# Capability-qualified audio edit planner (CQEP-001)

A deterministic Rust compiler for normalized audio-edit requests and versioned operator/observer capability profiles.

## Paradigm

Do not let an LLM directly choose arbitrary tools and immediately mutate audio. Compile the normalized request into a typed DAG first. Select only registered operators that satisfy the requested capability, artifact types, preconditions, postconditions, locality and determinism envelope. Add independently declared observers for every required metric.

The plan is a proposal, not a runtime authorization. Each mutation still requires exact-parent, registry, precondition and budget checks.

## Inputs

- A normalized request with content-addressed initial artifacts.
- A DAG of typed stages, each declaring explicit input/output artifact IDs, dependencies, capabilities, preconditions, postconditions, locality ceiling and required observation metrics.
- Versioned operator profiles with artifact digests, typed I/O, capability sets, scope, determinism and additive resource/risk estimates.
- Observer profiles that declare independent qualification, metric coverage and their own artifact digests.
- Hard compute, latency and risk budgets plus transparent objective weights.

## Compile behavior

- Graph stages are topologically sorted deterministically.
- A stage input must be an initial artifact or the output of an explicitly declared dependency.
- Dependencies must actually be consumed; decorative or stale graph edges are rejected.
- Operators are eligible only when every hard requirement is met.
- A global operator cannot satisfy a phrase- or sample-local contract.
- Nondeterministic operators cannot satisfy a deterministic/seeded request.
- Each required observation metric must have an independently qualified observer with a different artifact digest from the selected editor.
- Candidates are sorted by weighted compute/latency/risk score, then stable ID.
- The returned plan carries the exact initial artifact IDs/types/digests, budget, objective weights, determinism ceiling and per-stage locality/determinism ceilings.
- The entire plan is rejected if aggregate resources exceed the request budget.

Errors distinguish:
- InvalidInput — malformed specification/registry or invalid graph.
- Unsupported — no operator or independent observer can meet the contract.
- BudgetExceeded — the only feasible selected plan exceeds a hard aggregate budget.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/capability-qualified-edit-planner/Cargo.toml
cargo +1.96.0 test --manifest-path tools/capability-qualified-edit-planner/Cargo.toml
~~~

## Boundaries

This crate does not parse raw natural language, execute tools, edit audio, calculate acoustic metrics, authenticate capability claims, or issue execution authorization. Operator declarations are inputs to selection; independent empirical qualification remains mandatory.

The observer independence field is also a registry declaration. The compiler additionally requires a different observer artifact digest from the selected operator, but this alone does not prove statistical independence. A separate audit must establish that.

The compiled plan is returned as typed Rust data, including the exact input hashes and policy ceilings used to compile it. Canonical serialization and cryptographic plan digesting remain an explicit versioned external boundary in v1.
