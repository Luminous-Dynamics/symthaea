# SPINE-000B — Observer Non-Interference Contract v1

**Status:** preregistered observer-architecture contract

**Authority:** measurement-only

**Issue:** #3305

This contract freezes where and how SPINE-000B may observe the live cognitive loop before runtime receipt instrumentation is implemented.

The purpose is not merely to prevent direct state mutation. In the current loop, wall-clock elapsed time can change cognitive execution through attention/planning budget gates. An observer that only reads state can therefore still perturb cognition by consuming CPU time at the wrong point.

## 1. Source ordering that creates the hazard

The live manager proposal path executes inside `phase_dynamics` through the `run_subsystem!` macro. Later in the same phase, production evaluates wall-clock budget using `cycle_start.elapsed()` and derives an `available_us` budget from a 20,000 µs target.

Therefore:

```text
observer work at manager seam
        ↓
more elapsed wall time
        ↓
smaller available_us
        ↓
possible budget/planner/gating branch change
        ↓
possible cognitive behavior change
```

`measurement-only` authority is not sufficient to establish non-interference.

## 2. Two-stage observer architecture

### Stage A — bounded hot-path capture

Stage A is the only SPINE code permitted to execute at manager scheduling/execution seams before all cognition-affecting wall-clock budget decisions have completed.

Stage A may capture only information that cannot be reconstructed reliably after the fact:

```text
cycle_number            u64
manager_id               u16
eligible_to_run          bool
health_disabled          bool
execution_outcome        compact enum
proposal_bits            fixed-size optional POD
proposal_reserved        u32
observer_sequence        u16
```

The exact runtime struct may differ in Rust layout; canonical receipt identity remains governed by C1 rather than native ABI layout.

Stage A MUST NOT:

- hash or build canonical receipt digests;
- serialize JSON/text;
- perform leave-one-out integration;
- allocate unbounded strings, vectors, maps, or boxes;
- perform filesystem, network, database, logging-export, or blocking I/O for SPINE evidence;
- read wall-clock time as canonical evidence;
- consume or advance any cognitive RNG;
- call a cognitive manager except the existing production call already present;
- call `OutputCollector::integrate()` for measurement;
- change manager invocation order;
- change `OutputCollector::record()` order or admission semantics;
- change `SubsystemHealthTracker` state transitions;
- catch or suppress a panic differently from production;
- change Phase-C application order.

Tracing already present in production is outside this new observer surface; I1 forbids adding evidence-side tracing in Stage A.

### Stage B — deferred derivation

Expensive evidence work is deferred until after every cognition-affecting wall-clock budget decision covered by the qualified profile.

Stage B may perform:

- proposal-domain classification (N1);
- manager-ID resolution;
- `I_all` and `I_withoutS` recomputation;
- application receipt assembly;
- canonical C1 encoding and SHA-256;
- evidence-chain update;
- JSON/artifact serialization;
- noncanonical performance telemetry.

A source-order verifier must fail if Stage B is moved before a frozen budget-decision seam.

## 3. Stable manager identity at the hot seam

Stage A must not hash/copy arbitrary manager names per execution.

Use an append-only manager-ID registry:

```text
manager_id: u16, 0 reserved invalid
canonical_name: existing CognitiveSubsystem name/literal
source_path: repo-relative path
status: ACTIVE | RETIRED
introduced_registry_version
```

Rules:

- existing IDs never change meaning;
- deleting a manager retires its ID; IDs are never reused;
- new managers receive new unused IDs;
- duplicate names or IDs fail closed;
- Stage B resolves ID → canonical name/path before C1 execution receipts are encoded;
- registry drift creates a new evidence lineage/version.

The first registry may assign IDs deterministically from the frozen current census, but after publication the mapping is append-only rather than re-sorted/re-numbered.

## 4. Fixed-capacity Stage-A buffer

v1 capacity is **64 manager events per cycle**.

Rationale: the current manager census must fit below this limit before instrumentation can qualify, while 64 keeps the hot-path buffer bounded and leaves headroom for additional managers.

Rules:

- one scheduling/execution event at most per registered manager per cycle;
- no dynamic growth when the buffer is full;
- no silent overwrite or ring-buffer loss;
- on attempted event 65, set `observer_overflow=true` and stop recording additional SPINE Stage-A events for that cycle;
- overflow makes the cycle `SPINE_RUNTIME_UNQUALIFIED_OBSERVER_OVERFLOW`;
- overflow must not change manager execution, health, collector admission, or Phase-C application;
- overflow itself is noncanonical runtime/evidence-status metadata until a later commitment version explicitly binds it.

The exact-head static gate must prove the current discovered `CognitiveSubsystem` manager count is <=64.

## 5. Instrumentation placement law

The source verifier must establish at minimum:

1. live `run_subsystem!` exists in `phase_dynamics`;
2. manager execution occurs before the known `available_us` wall-clock budget calculation;
3. Stage-A capture code, when implemented, is confined to the manager scheduling/execution surface;
4. Stage-B derivation entry point occurs after the last frozen cognition-affecting wall-clock budget decision in the qualified profile;
5. no evidence hash/serialization/leave-one-out symbol is reachable from Stage A.

If new cognition-affecting `Instant::elapsed()` decisions appear after the frozen Stage-B insertion point, I1 must fail and be revised.

## 6. Static non-interference theorem

A static PASS must show the SPINE observer has no code path that intentionally mutates:

- manager cognitive state except the existing production `process()` call;
- `OutputCollector` except the existing production `record()` call;
- `SubsystemHealthTracker` except existing success/panic transitions;
- scheduler/urgency/budget state;
- feedback/carryover/neuromodulator state;
- RNG state;
- external application/network state.

Dedicated observer buffers/evidence state are allowed.

Static analysis is necessary but not sufficient because observer CPU overhead can still perturb wall-clock gates.

## 7. Dynamic non-interference campaign

Full-loop independent reruns are not currently bitwise deterministic. Existing genesis tests explicitly allow significant floating-point tolerance while requiring discrete outputs to agree. Therefore I1 must not invent an impossible exact-float theorem.

Preregister a two-phase campaign:

### Baseline phase — blind to observer-ON results

Run paired `OFF-A` and `OFF-B` arms with the same genesis seeds, frozen input sequences, feature profile, async policy, thread/Rayon settings, and environment.

Estimate baseline paired variability separately for each floating endpoint. Freeze equivalence bounds before observer-ON results are examined. Bounds must be scientifically justified and may not simply reuse the historical 0.15 tolerance without remeasurement for the exact I1 profile.

### Observer phase

Run matched `OFF` and `ON` arms.

Required exact discrete endpoints include, where present in the frozen profile:

- manager scheduling/eligibility decisions;
- health-disabled and panic outcomes under the campaign workload;
- admitted contributor identity/order;
- integrated flags;
- learning-occurrence booleans;
- safety gate/veto discrete states;
- `attention_budget_exceeded`;
- predictive-budget-gated decision;
- planner/budget tier;
- application operation sequence;
- observer overflow must remain false for qualified cycles.

Floating cognitive endpoints use preregistered equivalence tests against bounds frozen from baseline variability.

A statistically nonsignificant difference is not evidence of equivalence.

## 8. Timing fields are not cognitive equality endpoints

The following are expected to change and must be analyzed as observer overhead rather than required equal:

```text
cycle_duration_us
module_timings_us
SPINE capture/derivation durations
host scheduler timing
wall-clock timestamps
```

However, if timing changes alter a cognition-affecting budget/control-path endpoint, the campaign FAILS non-interference.

## 9. Initial claim scope

The first runtime non-interference claim should use a deterministic measurement profile that disables/freezes asynchronous external sources where practical and pins relevant execution configuration.

Any PASS is scoped to that profile. It cannot be generalized automatically to all optional features, mesh/network activity, live wall-clock workloads, or hardware.

## 10. Qualification boundary

I1 has two gates:

- `I1-STATIC`: capture/defer architecture, source-order, fixed capacity, manager-ID rules, and forbidden-effect surface are frozen and machine-checked.
- `I1-DYNAMIC`: paired OFF/OFF baseline and OFF/ON observer campaign passes all exact discrete and preregistered floating equivalence gates.

Only `I1-DYNAMIC` establishes observer non-interference for the tested profile. Neither gate establishes subsystem influence, causal load, benefit, truth, or action authority.
