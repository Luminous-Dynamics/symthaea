# SPINE-000B-I1R1 Runtime Manager ID Registry v1

Status: **REGISTRY FROZEN / RUNTIME USE PENDING**

Authority: `measurement-only`

The registry assigns stable `u16` identities to managers actually attributed through the live Phase-B `run_subsystem!` macro. This is the identity consumed by subsystem health and `OutputCollector`; it is therefore the correct observer identity rather than Rust type names or filenames.

## Scope

Only current live macro-attribution labels belong in v1. Test-only `CognitiveSubsystem` implementations and direct paths such as Canvas, Creative, or Knowledge processing are outside this registry unless they later enter the same SPINE subsystem protocol under a separately versioned change.

## ID law

- ID `0` is permanently invalid/reserved.
- Initial IDs `1..N` are frozen by `SPINE_000B_MANAGER_ID_REGISTRY_V1.json`.
- `(id, runtime_name)` mappings are immutable once published.
- Future managers receive `max(id)+1`; source insertion order never causes renumbering.
- Removed managers transition to `RETIRED`; IDs are never reused or reactivated.
- A runtime attribution rename uses a new ID and retires the previous mapping unless a separate migration protocol is qualified.

## Execution-order law

Stable identity and execution order are different concepts. The v1 verifier parses `cycle_phase_dynamics/mod.rs` and records the current `run_subsystem!` literal order. The initial v1 registry intentionally matches that order, but future source reordering must not renumber stable IDs.

A runtime campaign's M1 subject manifest must bind both this registry file and the scheduling source, so order drift creates a new subject identity even when numeric manager IDs remain stable.

## Hot-path law

The future I1 Stage-A observer captures only the numeric manager ID plus bounded POD execution data. String resolution is deferred until after cognition-sensitive work. No runtime string hashing or copying is needed to identify a manager.

## Capacity law

The current manager-observer capacity is 64 events/cycle. Registry population must remain within that bound. Capacity changes belong in a new M1 subject identity and I1 qualification lineage.

## Claim boundary

The registry establishes identity only. A registered manager is not thereby scheduled, executed, admitted, influential, application-relevant, causal, beneficial, or authoritative.