# BASALT: independent observer-result adjudication

**Status:** Rust result adjudicator plus scoped counterfactual fixture/oracle runner; not an observer qualification result.

## Decision

The next boundary after the challenge catalog is a three-artifact protocol:

1. **Challenge catalog** — what is to be tested, its expected disposition, resource IDs, required facts and paired controls.
2. **Oracle ledger** — the harness's claimed ground truth: which injected resource was consumed, deliberately not consumed, unsupported, or associated with an integrity fault; the controlled mutation and a replay claim.
3. **Observer receipt ledger** — raw output from the observer under test. Treat it as untrusted input.

The result adjudicator joins these exact artifacts and checks their byte digests, run/profile identity, scenario coverage, resource attribution, required facts, explicit unresolved channels, integrity signals and evidence-reuse decision. It does not let the observer author the expected result.

**Important trust boundary:** the current adjudicator verifies consistency against the supplied oracle ledger. It does not prove that the oracle was independently produced, that the claimed replay actually ran, or that the harness binary is trustworthy. SHA-256 binds bytes; it does not authenticate an author. `oracle_authenticity_established`, `oracle_replay_executed_by_adjudicator`, `observer_receipt_authenticity_established`, and `qualification_eligible` are all hard-coded false. A passing report means only that one observer receipt set conforms to the supplied oracle assertions. It is not a confirmatory result and cannot qualify the Linux profile.

## Added implementation

- `xtask/src/basalt_observer_result.rs`: strict, bounded parsing and scenario-level adjudication.
- `benchmarks/basalt/schema/observer-oracle.v1.schema.json`: strict Draft 2020-12 oracle-ledger schema.
- `benchmarks/basalt/schema/observer-receipts.v1.schema.json`: strict Draft 2020-12 observer-receipt schema.
- `benchmarks/basalt/observer-evidence-retention-benchmark.v1.json` plus its schema: machine-readable comparison arms, metric definitions, safety/selectivity/efficiency gates and sample-size protocol.
- `xtask/src/main.rs`: `basalt-observer-result-audit` command.
- Nineteen Rust tests cover paired recall/selectivity, stale reuse, event loss, unrelated-resource attribution, unsupported channels, false invalidation, counterfactual output/read witnesses, worker-binary mismatch, scoped-pair completeness, cross-run/cross-catalog mismatch, incomplete receipts, and read-but-discard semantic access.
- The adjudicator first runs the existing challenge-catalog structural auditor. Catalog validity and execution-result validity remain distinct checks.

The result protocol deliberately rejects malformed, over-sized, duplicate, missing, unknown, cross-run, cross-profile, and cross-catalog records. Oracle mutation digests must be present (including explicit nulls when no mutation applies); applied mutations require distinct before/after SHA-256 values. The current 36-case catalog is the fixture definition, not evidence that any observer has executed it. The adjudicator also reports `unrelated_evidence_not_retained` separately from explicit `false_invalidations`, so unresolved or otherwise non-reused positive controls cannot disappear from the retention accounting.

## Command

From the repository root:

```sh
cargo run -p xtask -- basalt-observer-result-audit \
  --catalog benchmarks/basalt/coverage/observer-challenges.v1.json \
  --oracle path/to/oracle-ledger.v1.json \
  --receipts path/to/observer-receipts.v1.json
```

The command prints a machine-readable report. It exits non-zero unless every **selected** case has exactly one oracle record and one observer receipt and all per-case, pair, digest and envelope checks pass. The report separately identifies `scope_is_full_catalog`; `adjudication_complete` means only that the declared scope is complete. The report includes the exact input digests and scenario-level error codes. It does not sign output or make the supplied oracle authentic.


## Executable counterfactual fixture runner

The follow-on harness command currently implements six filesystem cases as three reciprocal pairs:

- `hidden-untracked-read` ↔ `unused-neighbor-change` on `fixture:unlisted-secret.txt`.
- `irrelevant-docs-change` ↔ `docs-consumed-positive-control` on `fixture:docs/unused.md`.
- `read-but-discard` ↔ `read-and-influence` on `fixture:docs/unused.md`.

The third pair is a semantic-effect experiment: both tasks access the same resource, but one discards the value and the other emits an output derived from it. Both must remain captured dependencies.

Run it with:

```sh
cargo build -p xtask --bins
cargo run -p xtask --bin xtask -- basalt-observer-fixture-run \
  --catalog benchmarks/basalt/coverage/observer-challenges.v1.json \
  --out-dir reports/basalt-fixtures \
  --case-id hidden-untracked-read
```

The harness automatically closes the requested selection over reciprocal pairs. It creates an exclusive run directory (an existing run directory is never reused), initializes a separate Git repository, commits `.gitignore` and the docs baseline when selected, and verifies tracking state with `git check-ignore` and `git ls-files`. It creates a unique run directory containing:

- `oracle-ledger.v1.json`: selected case IDs, exact catalog digest, harness/worker binary digest, before/after resource digests, expected ground truth, and child-task counterfactual output/read witnesses.
- `fixture-evidence.v1.json`: child process IDs, output digests, bytes-read values, workspace-relative paths, Git tracking-state assertions, baseline commit and fixture evidence digest inputs.
- `workspace/`: an isolated Git repository. The `unlisted-secret.txt` fixture is verified as ignored and absent from the index; `docs/unused.md` is committed and verified as tracked at baseline.

The parent harness runs the separate child worker twice per case, once against baseline bytes and once after mutation. The worker is a controlled synthetic task, not an existing Cargo test/build command, and this fixture runner does not yet attach the observer under test or produce observer receipts. For `consume`, it reads the resource and emits a digest of the bytes; the parent independently computes the expected digest and requires the output digest to change. For `read_discard`, it reads the resource but emits the same constant task-output digest for both mutations; the worker reports the positive byte count. For `ignore`, the worker's code path does not open or stat the path, emits a constant output digest, and reports zero bytes read; the parent independently checks that result.

The oracle now records optional `semantic_effect` values (`demonstrated` or `not_demonstrated_in_tested_domain`) independently of ground-truth access. The adjudicator accepts a read-but-discard witness only when both executions report positive reads and the declared output digests are identical. This does **not** change the required observer disposition: the accessed resource must still appear in the receipt, the dependency change still requires re-verification or invalidation, and the resource cannot be omitted on the basis of one unchanged output. The task worker is a separately compiled `xtask/src/bin/basalt_observer_fixture_worker.rs` executable. Its digest is pinned separately from the parent harness digest; both child PIDs and the worker digest are recorded. Build both package binaries with `cargo build -p xtask --bins` before running the harness.

**Trust boundary:** the task worker is a separate executable with a deliberately narrow interface; it cannot write oracle records or observer receipts. The parent harness independently checks its declared read/output witnesses and requires every record's worker digest to match the separately pinned worker digest. This is stronger process and binary separation from the observer, but it is not a cryptographically independent oracle root: both binaries are built from the same repository/toolchain, the harness controls the fixture and task mode, and the worker's byte-count report is still a claim made by that worker. The ledger is not signed, and the result adjudicator does not independently execute the replay. The report therefore keeps oracle authenticity, replay execution by the adjudicator, receipt authenticity and profile qualification false.

The fixture command intentionally rejects every catalog case it does not implement. It does not create fake records for environment, procfs, toolchain, cache, network, observer overflow or other unimplemented channels. The run scope in the oracle and observer receipt ledgers must match exactly. A selected paired case requires its reciprocal partner in the same run; a full-corpus qualification still requires a later run covering all 36 cases and all required independent challenge strata.

The fixture evidence schema is `benchmarks/basalt/schema/observer-fixture-evidence.v1.schema.json`. Its execution records identify the separate worker binary, relative resource path, process IDs, Git tracking classification, baseline Git commit and before/after output witnesses. Three fixture-orchestrator tests now cover supported-resource rejection, semantic-mode mapping, and the ignored-untracked/tracked Git-state setup. The standalone worker has three tests, including read-but-discard behavior; the result adjudicator has two additional semantic-influence tests. Total authored Rust tests in this stack: 27; none are claimed as executed. The oracle and receipt schemas now require `selected_case_ids`; oracle records also carry the worker binary digest, before/after task-output digests and byte counts. The result adjudicator checks that the selected scope is non-empty, unique, known to the catalog, identical across both ledgers, and complete for every selected pair. Its `adjudication_complete` field means complete **for that declared run scope**; `scope_is_full_catalog` independently reports whether every catalog case was selected. Neither field implies that the observer profile is qualified.

## Real Cargo build-script hidden-input fixture

The new command basalt-cargo-build-script-fixture-run exercises the catalog's existing cargo-build-script-input challenge using an actual dependency-free Cargo package. Its build.rs reads build.rs-input.txt, but deliberately declares only rerun-if-changed=build.rs. The input is ignored and untracked in the fixture's isolated Git repository.

The experiment runs three stages:

1. Baseline cargo run --offline --locked into a warm target directory; output must equal baseline-state.
2. Mutate only the ignored build-script input, then rerun with the same target directory; output is expected to remain at the baseline value because Cargo's declared rerun set omitted the input.
3. Run the mutated fixture with a fresh target directory; the build script runs again and output must equal mutated-state.

The runner pins the Cargo executable digest/version and the rustc launcher plus sysroot compiler binary digests, rustc version and sysroot identity. It also pins the fixture-template digest, catalog digest, input digests and Git baseline commit. It records Cargo process IDs, stdout/stderr digests, stage-specific output and the two target-directory identities. Rust compiler wrappers and ambient RUSTFLAGS are removed for this controlled fixture run. It refuses to report a successful counterfactual if the warm run does not reproduce the stale value or the fresh replay does not reflect the mutation.

**Interpretation:** this is a real Cargo incremental-cache failure fixture, not a test that BASALT's observer catches it. The evidence's ground-truth basis is checked-in build-script source plus a controlled input mutation and fresh-target replay. The runner does not capture the spawned build-script child's PID, does not independently observe its file open at the kernel level, and does not emit an observer receipt. independent_kernel_observation and qualification_eligible remain hard false.

Run it with:

    cargo run -p xtask --bin xtask -- basalt-cargo-build-script-fixture-run \
      --catalog benchmarks/basalt/coverage/observer-challenges.v1.json \
      --out-dir reports/basalt-cargo-fixtures

The fixture is dependency-free and network-disabled at invocation (--offline --locked). Cargo's documented build-script semantics specify that emitted rerun-if-changed paths control whether the script runs again. This fixture intentionally omits its actual input from that declaration. See the Cargo Book build-script reference and example:

- https://doc.rust-lang.org/cargo/reference/build-scripts.html
- https://doc.rust-lang.org/stable/cargo/reference/build-script-examples.html

## Next paradigm shift: access is not semantic influence

The first fixture runner currently distinguishes a task that reads a resource from a task that never opens it. That is a necessary baseline, but it is not the full evidence-retention problem.

A task can read bytes and deliberately discard them. A filesystem observer should still report the access. Yet an access event alone does not prove that a change to those bytes affected the task's observable output. Conversely, one unchanged output under one mutation is not proof that the resource is irrelevant under every input, branch, runtime state or future execution.

BASALT should eventually separate three claims:

| Claim | Evidence | What it does *not* establish |
|---|---|---|
| Access observed | File/metadata/mapping event with stable resource identity | That the resource affected the task result |
| Influence demonstrated in this execution | Controlled perturbation changes a deterministic output or declared effect witness | That all possible values/paths have been explored |
| Non-influence bounded to a declared domain | Multiple controlled perturbations, same task/environment, complete output/effect comparison, and a stated equivalence domain | Universal irrelevance outside that domain |

### A read-but-discard challenge

Add a separate semantic-impact fixture in which the task reads and hashes the resource into an internal value, deliberately discards that value, and emits a deterministic output independent of the resource. The observer's access receipt must include the resource; the counterfactual lane should show that the task output is unchanged for the tested mutation. This case must **not** be classified as `not_consumed`. It is `accessed_no_output_effect_in_tested_domain`.

Then add a matched influence-positive fixture where the same resource is used in the emitted output. The same mutation should alter the output witness. Both tasks must run from identical snapshots and the test matrix must record all output and side-effect channels included in the equivalence claim.

### Semantic influence: access and effect are separate axes

BASALT now includes a matched read-but-discard pair. The worker reads the exact same resource before and after mutation in both cases. The discard worker emits a constant output; the positive-control worker emits a digest of the resource bytes. The independent oracle records the access ground truth as consumed in both cases, while the optional semantic-effect field distinguishes `not_demonstrated_in_tested_domain` from `demonstrated`.

The result adjudicator accepts the pair only if both observer receipts retain the same resource. This deliberately prevents a semantic non-influence observation from being misused as evidence of non-access or as permission to omit a dependency. The semantic-effect value is bounded to the declared output digest for this synthetic worker; it does not cover filesystem side effects, stdout/stderr, exit status, timing, network effects, shared memory, or other unmodeled outputs.

### Safe interpretation

Do not let a single unchanged output erase a dependency. The report should retain separate sets:

- `accessed_resources`: resources observed by the OS/build adapter.
- `influence_demonstrated_resources`: resources whose controlled perturbation changed a declared output/effect witness.
- `non_influence_tested_resources`: resources that were accessed but produced no observable difference within a preregistered mutation and execution domain.
- `unresolved_influence_resources`: resources for which the task is nondeterministic, effects are not fully observable, or the tested domain is insufficient.

Only the last two are allowed to constrain a bounded reuse claim, and only under a separately qualified influence profile. Outside that profile, the current conservative rule remains: a changed accessed resource is a dependency or makes evidence unresolved.

### Research techniques to compare

1. **Dynamic taint / information-flow tracking** can trace whether input-derived values reach declared outputs or side effects, but requires explicit coverage of system calls, native code, GPU/device activity, shared memory and language/runtime boundaries.
2. **Differential replay** can perturb one input at a time and compare output/effect manifests, but only supports claims about tested executions and mutations.
3. **Program slicing / dependency graphs** can explain data and control paths, but static approximations may over- or under-approximate dynamic behavior.
4. **Metamorphic and property-based perturbation** can broaden the tested input domain, while preserving a clear distinction between sampled robustness and universal guarantees.

This yields a second, non-compensatory benchmark axis: first measure access-observation soundness; only then measure influence-classification precision and safe retention. The machine-readable benchmark now includes `semantic_access_retention` as a mandatory safety gate: 100% of read-but-discard scenarios must retain exactly one resource attribution and block reuse after the mutation, even though the declared output digest is unchanged. A system must never improve its retention score by silently failing to observe a read. No production evidence should be reused solely because one counterfactual output happened to remain unchanged.

## Do not collapse freshness, coverage and integrity into one status

The next receipt revision should model four orthogonal dimensions rather than a single boolean called `complete`:

| Dimension | States | What it answers |
|---|---|---|
| Freshness | `current`, `stale`, `unknown` | Do the recorded dependencies still match this run's inputs? |
| Coverage | `covered`, `incomplete`, `unsupported`, `lost` | Was every relevant observation channel inside the qualified profile? |
| Integrity | `intact`, `invalid` | Was the capture interval, process lineage, snapshot and event stream coherent? |
| Replay | `confirmed`, `unverified`, `contradicted` | Did an independent execution reproduce the expected ground-truth relationship? |

These dimensions have different consequences:

- **Confirmatory eligibility** requires current freshness, covered channels, intact capture integrity, independently confirmed replay, and a separately qualified profile.
- **Incomplete** means the result may be useful for exploration, but an unsupported or unverified channel prevents confirmatory reuse.
- **Invalid** means a contradiction, event loss, snapshot race, root escape, or other integrity failure makes the receipt unusable.
- A changed dependency makes evidence **stale** even when the observer captured it perfectly; capturing a change does not make the previous result current.

This creates a monotonic downgrade rule: adding evidence can move a claim from unknown to covered only when independently verified, but a signature, green test, or observer self-report cannot promote unsupported or lost coverage. Any change to the kernel, hook set, filesystem/mount profile, Landlock ABI, process supervisor, observer binary, buffer policy, network policy or toolchain closure changes the profile identity and may require requalification.

The current result adjudicator intentionally implements only part of this lattice. It checks per-scenario disposition and failure signals; it does not yet establish freshness history, replay truth or a qualified profile.

## Adjudication contract

| Challenge expectation | Oracle ground truth | Receipt requirement | Evidence-reuse rule |
|---|---|---|---|
| `capture_dependency` | `consumed` | Exact resource ID and post-mutation digest represented; all catalog-required facts present | Old evidence must be invalidated or reverified |
| `preserve_unrelated` | `not_consumed` | Mutated resource absent from dependency attribution | Existing evidence must be retained |
| `report_unresolved` | `unsupported` | Channel explicitly listed as unresolved | Evidence cannot be reused as confirmatory |
| `invalidate_receipt` | `integrity_failure` | Loss/restart/unmonitored-child/race/root-escape signal visible | Receipt and reuse decision both invalidated |

Every preserve control must be paired with a capture control over the **same resource and exact before/after mutation digests**. In the capture arm, the resource must be observed. In the preserve arm, the same resource must be absent from the dependency set. This is stronger than merely checking that the observer emits an event: it measures both recall and selectivity against an independent fixture condition.

## Paradigm shift: benchmark the evidence-retention frontier

The executable benchmark contract is `benchmarks/basalt/observer-evidence-retention-benchmark.v1.json`. It freezes seven matched comparison arms and ten metrics, including their numerators, denominators and targets. The protocol forbids compensating for a safety failure with a speed improvement; all six safety gates must pass before selectivity or cost can influence a promotion decision.

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

- https://bazel.build/docs/sandboxing
- https://nix.dev/manual/nix/stable/command-ref/conf-file
- https://slsa.dev/spec/v1.2/build-provenance
- https://in-toto.io/docs/getting-started/
- https://reproducible-builds.org/docs/definition/

The comparison is deliberately layered rather than a blanket claim that BASALT is “better than Bazel/Nix/SLSA.” Bazel supplies action sandboxes and declared input models; Nix supplies dependency-closure and build sandbox controls; SLSA/in-toto supply provenance, identity and workflow integrity; reproducible-builds checks whether independent builds produce bit-identical outputs. None of those alone proves that a particular runtime observer captured every input to a given task. BASALT's proposed differentiator is an adversarially challenged, independently adjudicated runtime dependency-coverage claim, with explicit unresolved channels and evidence-retention correctness. It should integrate with these systems and only claim superiority when all arms are compared on the same tasks, snapshots, toolchains, resource limits and held-out lineages.

SLSA provenance and in-toto supply-chain layouts provide valuable build and workflow provenance, authorized-step and integrity checks. BASALT should compose with these formats where useful, not reimplement their signature or supply-chain semantics. A provenance statement still depends on its builder/platform trust boundary; the SLSA specification explicitly distinguishes external inputs from parameters trusted because of the build platform. The BASALT oracle therefore needs its own independently reviewed trust root and replay semantics.

## Current industry baseline and the actual gap

BASALT should be designed as a composable proof layer, not as a replacement for build systems:

| System | Strongest existing contribution | What BASALT should add alongside it |
|---|---|---|
| Cargo | Build-script execution and incremental invalidation from declared rerun paths | Challenge missing rerun declarations against actual build-script reads; detect stale warm-target output after hidden-input mutation |
| Bazel | Declared action inputs, per-action execution roots and sandboxing | Independent evidence that the action's observed inputs fit its declared closure, and a fail-closed receipt for channels outside that boundary |
| Nix | Store-based dependency closure and sandboxing with private namespaces | Runtime evidence for the precise execution, explicit unresolved channels, and receipt integrity/currentness |
| SLSA / in-toto | Builder identity, provenance, workflow and artifact integrity | Independent challenge of dependency-coverage assertions rather than assuming an attested builder's input list is exhaustive |
| Reproducible Builds | Bit-for-bit comparison of outputs across independent rebuilds | Detect hidden inputs before they cause stale reuse; define exactly which execution inputs and side effects were in the observation envelope |

Bazel's own documentation identifies the precise failure class: without action sandboxing, undeclared input files can change while Bazel still considers the action up to date, producing incorrect incremental results; incorrect remote-cache entries can then affect many developers. Its action execroot contains known inputs, while sandbox behavior differs by platform and does not fully hide all host state. BASALT should measure both the sandbox-prevention arm and the observer/receipt arm, not claim that tracing alone beats a functioning sandbox.

Nix sandboxing limits build visibility to declared store dependencies, the temporary build directory and selected special paths; on Linux it also uses private namespaces. Fixed-output derivations are an explicit network exception. This is a strong hermeticity boundary. BASALT should compare against it as a separate arm and treat any admitted external path or network exception as part of the declared profile.

SLSA v1.2 is the approved specification. Its Build provenance describes the trusted builder/platform closure and resolved dependencies; the v1.2 provenance text describes dependency completeness as best-effort at Build L3. The SLSA future-directions page discusses complete input knowledge, hermetic builds and reproducible builds as possible Build L4 territory, not as a guarantee supplied by today's L3 provenance. This distinction is important: BASALT's proposed value is independently challenged runtime coverage, not another signature around an unverified dependency list.

Reproducible Builds defines reproducibility as independent parties being able to recreate bit-identical artifacts from the same source, environment and instructions. That is an essential cross-check, but two identical outputs do not prove that every dependency was observed or that a prior incremental result was correctly invalidated after a hidden input changed.

Primary references:
- Cargo build-script change detection: https://doc.rust-lang.org/cargo/reference/build-scripts.html
- Bazel sandboxing and undeclared inputs: https://bazel.build/versions/7.5.0/docs/sandboxing
- Nix sandbox configuration: https://nix.dev/manual/nix/2.25/command-ref/conf-file
- SLSA v1.2 Build provenance: https://slsa.dev/spec/v1.2/build-provenance
- SLSA v1.2 future directions: https://slsa.dev/spec/v1.2/future-directions
- Reproducible Builds definition: https://reproducible-builds.org/docs/definition/

### Proposed research gates (targets, not results)

| Gate | Target |
|---|---|
| Hidden dependency recall | 100% of seeded consumed inputs inside the declared profile, or explicit unresolved/invalidated |
| Stale-evidence false acceptance | Zero in qualification and held-out challenge sets |
| Integrity-loss response | 100% fail-closed or explicit unresolved for injected observer-loss cases |
| Semantic access retention | 100% of read-but-discard mutations retain the accessed resource and block reuse despite unchanged output |
| Unrelated evidence retention | Preserve every independently proven unaffected case in the paired conformance set |
| False invalidation | At most 5% on independently proven unrelated controls, with scenario-level intervals |
| Re-verification cost | At least 30% less work than global rerun, only after all safety gates pass |
| Shifted engineering outcomes | +3 percentage points paired success over a strong matched baseline, or non-inferior success at materially lower cost |

A 36-case conformance set is not a statistical qualification sample. For a one-sided 95% upper bound near 0.1% after zero misses, the simple binomial rule-of-three requires about 2,995 independent scenario units. Cluster related mutations by task/repository lineage, report channel-stratified intervals, keep an untouched holdout, and do not count syscalls/events as independent trials.

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

The existing qualification note already identifies why fanotify alone cannot establish completeness. BPF-LSM is a promising complementary event lane for selected open, file-permission, mmap and descriptor-receive paths, but hook availability, kernel/BTF identity, LSM ordering, attach success, privileges, verifier acceptance and ring-buffer loss must be pinned and challenged. Landlock should reduce accessible state; it should not be represented as a trace. A Landlock rule set is not proof that every access was observed. The result adjudicator is intentionally independent of all those Linux choices.

## Execution status and limitations

This branch authors the scoped result adjudicator, separate worker executable, fixture runner, schemas, CLI integration, benchmark protocol, research documentation and corpus correction. Twenty-seven Rust unit tests are authored across the adjudicator, filesystem fixture, Cargo fixture and worker binary (19 adjudicator, 2 filesystem fixture, 3 Cargo fixture, 3 worker tests). The environment used for this work has no `cargo`, `rustc` or `rustfmt`; Rust compilation, formatting, unit-test execution, filesystem fixture execution, Cargo fixture execution and observer qualification have **not** been run here. The filesystem worker's byte-count field is still a worker claim. The Cargo fixture exercises a real offline Cargo build-script/cache path, but remains a source-controlled challenge without independent kernel observation. The next step is to execute both fixture runners on an exact checkout, then attach an independent kernel/process observation lane to the real Cargo build-script child and compare its receipt against the fixture supervisor and fresh replay. Broader adapters should follow only after this first Linux path has pinned hook coverage, process lineage and event-loss accounting. Oracle signing, a separately trusted replay verifier and observer receipts remain future work. The report's strict-false policy remains in force even for a passing unit test or future green CI run.
