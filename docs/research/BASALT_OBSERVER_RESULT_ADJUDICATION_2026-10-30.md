# BASALT: independent observer-result adjudication

**Status:** implementation proposal plus runner-independent Rust adjudicator; not an observer qualification result.

## Decision

The next boundary after the challenge catalog is a three-artifact protocol:

1. **Challenge catalog** — what is to be tested, its expected disposition, resource IDs, required facts and paired controls.
2. **Oracle ledger** — the harness's claimed ground truth: which injected resource was consumed, deliberately not consumed, unsupported, or associated with an integrity fault; the controlled mutation and a replay claim.
3. **Observer receipt ledger** — raw output from the observer under test. Treat it as untrusted input.

The result adjudicator joins these exact artifacts and checks their byte digests, run/profile identity, scenario coverage, resource attribution, required facts, explicit unresolved channels, integrity signals and evidence-reuse decision. It does not let the observer author the expected result.

**Important trust boundary:** the current adjudicator verifies consistency against the supplied oracle ledger. It does not prove that the oracle was independently produced, that the claimed replay actually ran, or that the harness binary is trustworthy. SHA-256 binds bytes; it does not authenticate an author. Both `oracle_authenticity_established` and `qualification_eligible` are hard-coded false. A passing report means only that one observer receipt set conforms to the supplied oracle assertions. It is not a confirmatory result and cannot qualify the Linux profile.

## Added implementation

- `xtask/src/basalt_observer_result.rs`: strict, bounded parsing and scenario-level adjudication.
- `benchmarks/basalt/schema/observer-oracle.v1.schema.json`: strict Draft 2020-12 oracle-ledger schema.
- `benchmarks/basalt/schema/observer-receipts.v1.schema.json`: strict Draft 2020-12 observer-receipt schema.
- `xtask/src/main.rs`: `basalt-observer-result-audit` command.
- Ten Rust tests cover a paired recall/selectivity success case, stale reuse, event-loss misreporting, false attribution of an unrelated resource, and cross-run receipt mismatch.
- The adjudicator first runs the existing challenge-catalog structural auditor. Catalog validity and execution-result validity remain distinct checks.

The result protocol deliberately rejects malformed, over-sized, duplicate, missing, unknown, cross-run, cross-profile, and cross-catalog records. Oracle mutation digests must be present (including explicit nulls when no mutation applies); applied mutations require distinct before/after SHA-256 values. The current 34-case catalog is the fixture definition, not evidence that any observer has executed it.

## Command

From the repository root:

```sh
cargo run -p xtask -- basalt-observer-result-audit \
  --catalog benchmarks/basalt/coverage/observer-challenges.v1.json \
  --oracle path/to/oracle-ledger.v1.json \
  --receipts path/to/observer-receipts.v1.json
```

The command prints a machine-readable report. It exits non-zero unless every catalog case has exactly one oracle record and one observer receipt and all per-case, pair, digest and envelope checks pass. The report includes the exact input digests and scenario-level error codes. It does not sign output or make the supplied oracle authentic.

## Adjudication contract

| Challenge expectation | Oracle ground truth | Receipt requirement | Evidence-reuse rule |
|---|---|---|---|
| `capture_dependency` | `consumed` | Exact resource ID and post-mutation digest represented; all catalog-required facts present | Old evidence must be invalidated or reverified |
| `preserve_unrelated` | `not_consumed` | Mutated resource absent from dependency attribution | Existing evidence must be retained |
| `report_unresolved` | `unsupported` | Channel explicitly listed as unresolved | Evidence cannot be reused as confirmatory |
| `invalidate_receipt` | `integrity_failure` | Loss/restart/unmonitored-child/race/root-escape signal visible | Receipt and reuse decision both invalidated |

Every preserve control must be paired with a capture control over the **same resource and exact before/after mutation digests**. In the capture arm, the resource must be observed. In the preserve arm, the same resource must be absent from the dependency set. This is stronger than merely checking that the observer emits an event: it measures both recall and selectivity against an independent fixture condition.

## Paradigm shift: benchmark the evidence-retention frontier

A conventional cache benchmark tends to optimize cache hit rate and rebuild time. BASALT should instead optimize a constrained frontier:

- **Dependency recall:** detect changed resources that the task actually consumed.
- **Stale-evidence false acceptance:** never reuse evidence after a relevant dependency changed.
- **Unrelated-evidence retention:** keep valid results when a changed resource was independently shown not to influence the task.
- **Unsupported-channel disclosure:** explicitly downgrade or reject cases outside the observer's envelope.
- **Integrity-loss detection:** overflow, restart, missing process coverage, snapshot races and root escapes must invalidate or become unresolved.
- **Operational cost:** measure rerun work, replay work, CPU, wall time, I/O, storage, trace volume and p50/p95 recovery latency.

Do not average these into one compensatory score. A lower runtime cannot offset one stale-evidence false acceptance. Report safety gates first, then selectivity and cost.

### Matched comparison arms

Use the same tasks, input snapshots, mutation injections, toolchain and execution policy for each arm:

1. No invalidation — intentionally unsafe negative control, never promotable.
2. Global rerun — conservative safety/cost reference.
3. Declared manifests and native build-system dependency metadata.
4. Tracing without hermetic containment.
5. Hermetic candidate observer (read-only snapshot, pinned closure, constrained process tree, network policy).
6. Composed observer plus independent replay and oracle adjudication.
7. Independent reproducible rebuild — a cross-check on artifact output, not a substitute for dependency-observation qualification.

Bazel documents why undeclared inputs can poison incremental and shared remote-cache results, and uses known action inputs as the sandbox execution root; its sandboxing documentation also describes limits on what the sandbox hides from the host. Nix sandboxing restricts builds to declared store dependencies and private namespaces, with documented exceptions for network-enabled fixed-output derivations. These are strong baselines for prevention/hermeticity, not a proof that a particular observation receipt is complete. See:

- https://bazel.build/versions/7.5.0/docs/sandboxing
- https://nix.dev/manual/nix/2.25/command-ref/conf-file
- https://slsa.dev/spec/v1.2-rc2/build-provenance
- https://in-toto.io/docs/getting-started/
- https://reproducible-builds.org/docs/definition/

SLSA provenance and in-toto supply-chain layouts provide valuable build and workflow provenance, authorized-step and integrity checks. BASALT should compose with these formats where useful, not reimplement their signature or supply-chain semantics. A provenance statement still depends on its builder/platform trust boundary; the SLSA specification explicitly distinguishes external inputs from parameters trusted because of the build platform. The BASALT oracle therefore needs its own independently reviewed trust root and replay semantics.

### Proposed research gates (targets, not results)

| Gate | Target |
|---|---|
| Hidden dependency recall | 100% of seeded consumed inputs inside the declared profile, or explicit unresolved/invalidated |
| Stale-evidence false acceptance | Zero in qualification and held-out challenge sets |
| Integrity-loss response | 100% fail-closed or explicit unresolved for injected observer-loss cases |
| Unrelated evidence retention | Preserve every independently proven unaffected case in the paired conformance set |
| False invalidation | At most 5% on independently proven unrelated controls, with scenario-level intervals |
| Re-verification cost | At least 30% less work than global rerun, only after all safety gates pass |
| Shifted engineering outcomes | +3 percentage points paired success over a strong matched baseline, or non-inferior success at materially lower cost |

A 34-case conformance set is not a statistical qualification sample. For a one-sided 95% upper bound near 0.1% after zero misses, the simple binomial rule-of-three requires about 2,995 independent scenario units. Cluster related mutations by task/repository lineage, report channel-stratified intervals, keep an untouched holdout, and do not count syscalls/events as independent trials.

## Critical next work: make the oracle independently executable

The next implementation must replace the oracle's declarative replay claim with evidence generated outside the observer:

1. A fixture runner creates immutable named resources and controls the mutation.
2. The runner records a ground-truth journal from its own control channel, not from the observer's event stream.
3. A paired execution runs the same fixture with and without consuming the mutated resource.
4. An independent replay lane checks the fixture state and deterministic influence witness from a fresh snapshot.
5. An oracle verifier validates the journal against the fixture plan, run nonce, catalog digest, harness binary digest and replay output.
6. The observer adapter emits its receipt into a separate channel. The grader consumes both only after execution closes.
7. The final signed coverage receipt binds kernel/BTF/BPF identities, exact hook attach set, Landlock ABI/rights, process-supervisor identity, event-loss counters, snapshot digest, observer binary, corpus digest, oracle identity, replay binary and expiry.

The ground-truth writer must be outside the observer's trust domain. To resist common-mode failure, the first Linux implementation should compare at least two genuinely different lanes—for example, a fixture supervisor plus deterministic replay, and a separate kernel audit lane. Agreement between two tools that share the same blind spot is not independent confirmation. Any missing/unsupported lane narrows the profile; it must not be silently substituted.

## Linux architecture implications

The existing qualification note already identifies why fanotify alone cannot establish completeness. BPF-LSM is a promising complementary event lane for open, file permission, mmap and descriptor-receive paths, but hook availability, kernel/BTF identity, LSM ordering, attach success, privileges, verifier acceptance and ring-buffer loss must be pinned and challenged. Landlock should reduce accessible state; it should not be represented as a trace. The result adjudicator is intentionally independent of all those Linux choices.

## Execution status and limitations

This branch authors the adjudicator, schemas, CLI integration, tests and corpus correction. The environment used for this work has no `cargo`, `rustc` or `rustfmt`; Rust compilation, formatting, test execution, fixture execution and observer qualification have **not** been run here. The report's strict-false policy remains in force even for a passing unit test or future green CI run. The next step is to compile and execute this narrow module on an exact, reproducible checkout, then implement the fixture runner and independently verifiable oracle journal.
