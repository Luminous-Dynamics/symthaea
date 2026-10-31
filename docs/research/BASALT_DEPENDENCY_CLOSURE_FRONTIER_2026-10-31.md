# BASALT dependency-closure frontier: SOTA comparison and qualification design — 2026-10-31

## Executive decision

BASALT should not try to become another generic Linux event monitor, another compiler cache, or another provenance format. Its potentially differentiated product is an **execution-scoped, independently adjudicated dependency-closure contract**:

1. constrain what a build or verification action may consume;
2. observe and identify the resources it actually consumes;
3. bind those observations to the exact task lineage, immutable input snapshot, toolchain, environment and output;
4. challenge the evidence with independent counterfactual mutations and replay;
5. invalidate or retain prior evidence at the smallest safe scope;
6. export the validated dependency graph into interoperable supply-chain provenance.

This is a research target, not a claim that BASALT is already complete, qualified, or better than an industry system. The current Linux profile remains `candidate_not_qualified`; the challenge catalogue is marked `not_executed`. The coverage envelope itself correctly says a self-consistent receipt is not proof that the kernel emitted every event or that a signature represents a trusted observer.

## Starting point in the repository

The existing BASALT programme already has meaningful scaffolding:

- 38 challenge cases across filesystem, process, environment, generated artifact, toolchain, cache, network, observer integrity and snapshot-consistency concerns.
- 9 required observation channels, 33 required event kinds and 7 root classes in the candidate coverage profile.
- A separate oracle ledger and untrusted observer-receipt adjudicator.
- A run-level coverage-envelope auditor bound to a versioned profile.
- Cargo build-script hidden-input, mtime-preserving mutation and build-script environment fixtures.
- Fail-closed gates for stale-evidence reuse, event loss, unsupported channels, semantic reads, root escape, unstable snapshots and unmonitored children.

These are contracts and authored fixtures. They are not evidence that an observer has run or that the candidate profile has passed qualification.

## Paradigm shifts

### 1. From trace-only to enforce + observe + adjudicate

A trace can miss an event. A sandbox can enforce boundaries without explaining every dependency. A manifest can be incomplete. BASALT should combine three distinct mechanisms:

- **Enforcement plane:** action-scoped sandbox/policy denies or explicitly marks undeclared external inputs. If a denied access could affect the output, the action fails closed or is rerun after declaration.
- **Observation plane:** independent sensors record the permitted accesses and relevant process/environment/toolchain context. Every required source has an explicit health and loss state.
- **Oracle plane:** a separate harness injects known resources and mutations. The observer cannot write the oracle or decide whether its own evidence was complete.

No plane is trusted to prove the other two. Sensor disagreement, event loss, unknown scope or a missing process child downgrades the result. Never use a majority vote to erase a credible conflict.

### 2. From global invalidation to dependency-graph invalidation

The useful optimization is not “always rerun less.” It is: preserve evidence for proven-unrelated changes while rejecting reuse after any consumed dependency, relevant environment value, toolchain component or unsupported channel changes.

The unit is a task lineage and its dependency graph, not a repository-wide timestamp or a whole CI run. A dependency edge should bind:

- stable resource identity (not path text alone);
- the relevant content or metadata digest;
- the access/effect class (content read, metadata read, enumeration, mmap, environment value, executable, inherited descriptor, network response, etc.);
- the task and process lineage;
- the immutable snapshot/root manifest;
- the capture and observer-profile identity.

A path can be replaced, a symlink retargeted, a mount changed, or bytes changed without an mtime change. The identity/digest binding must survive those cases. Where a read cannot be bound to stable bytes, the resource remains unresolved and cannot justify a reusable cache result.

### 3. From “the trace saw it” to execution-scoped completeness

An observer can only report what happened on the executed path. It cannot prove that another code path, future input or unexecuted branch would not read another resource. Therefore:

- claim **observed closure for this exact invocation** only when all required channels are healthy and the supervised task tree is closed;
- claim **enforced closure within a profile** only when undeclared external reads are denied or otherwise handled by an enforceable policy;
- keep out-of-profile and unsupported inputs explicit;
- never turn “not observed” into “not used” without a sound enforcement or oracle reason.

This distinction is essential for cache soundness. The target is not omniscient tracing. It is a bounded, enforceable and falsifiable contract.

### 4. From hashes to authentic evidence

SHA-256 binds exact bytes, but does not authenticate the producer or prove that those bytes came from the kernel. The final trust chain needs independently protected identities for the trusted control plane, observer binary/configuration, hook set, sandbox policy, oracle harness, snapshot manifest and receipt signer. Domain separation, expiry/currentness, revocation and replay binding are separate concerns.

The existing code correctly leaves authenticity/qualification false. BASALT must keep that truth boundary until independent credentials and a trusted capture path exist.

### 5. From proprietary receipts to standards-compatible dependency provenance

The output should map into SLSA/in-toto rather than create a parallel supply-chain island. The measured dependency graph can populate `resolvedDependencies`, including transitive fetched artifacts and exact digests. The builder identity and trusted platform still have to satisfy the relevant SLSA build-level requirements; a BASALT receipt alone cannot confer a SLSA level.

## Industry comparison: what BASALT must beat, match or compose with

| System / standard | What it already does well | Fair BASALT comparison | BASALT's distinct target |
|---|---|---|---|
| Bazel sandboxed actions | Action execroots and OS-backed sandboxing restrict filesystem visibility and help prevent undeclared inputs from silently contaminating incremental and remote-cache results. | Run the same action snapshots and mutation schedule under the strongest supported Linux sandbox strategy; pin Bazel version, namespace availability, sandbox strategy and remote-cache settings. | Detect/deny and explain adversarial undeclared inputs, bind resource identity/digest to task lineage, and prove stale-result rejection under held-out oracles. |
| Nix sandboxed derivations | Derivations carry deterministic dependency references and builds run in a sandbox; independent rebuilds can be checked byte-for-byte. | Pin Nix version, derivation/store closure, sandbox and network settings; run the same hidden-input and mutation cases. | Independently adjudicate actual dependency use, explicit unsupported channels and fine-grained evidence retention. |
| SLSA Build L4 | The current Build L4 requirements define technical controls in the trusted build platform that ensure `resolvedDependencies` is complete for every artifact consumed during the build. This is the closest formal peer, not a strawman. | Use an actual build platform that documents and meets Build L4; verify builder identity/trust boundary and compare every oracle-consumed artifact with the provenance dependency set. | Provide an independently challengeable evidence path for completeness, and interoperate with rather than replace the trusted Build L4 control plane. |
| sccache Rust compiler cache | Its documented Rust key includes compiled-file digests, rustc path, host triple, sysroot path, sysroot shared-library digests and parsed invocation arguments. | Pin sccache version/configuration and run the Cargo build-script, mtime and environment challenges through the compiler cache. | Measure whether build-level undeclared inputs can make a higher-level result stale; do not mischaracterize a compiler cache as an action sandbox. |
| Reproducible Builds | Independently rebuilding the same declared source/environment/instructions and comparing specified output bytes gives strong artifact-integrity evidence. | Rebuild on independent builders and compare outputs; include diffoscope-style diagnosis when they differ. | Complement output equality with runtime dependency closure: equal output bytes alone do not prove that every input influencing the build was declared. |
| Falco / Tetragon / Linux BPF and tracepoints | Mature runtime telemetry, kernel hooks, process lifecycle visibility and policy-driven event collection. | Use as observation adapters in a later capture benchmark, with identical workloads, loss counters and task snapshots. | Supply the action-scoped dependency/oracle adjudication layer; do not compete on alert throughput or claim to replace their security telemetry. |

Bazel's own documentation explicitly discusses undeclared inputs and incorrect incremental/cache reuse as reasons for sandboxing, while also noting that its filesystem sandbox does not hide the host environment in every respect. Nix's reproducible-build guidance explicitly notes that deterministic references and sandboxing are not sufficient to eliminate timestamp and other nondeterminism. These are strengths and documented boundaries to benchmark, not grounds for dismissing either system.

SLSA Build L4 is the key strategic target. Its requirements state that the trusted build platform must technically ensure complete `resolvedDependencies`; the provenance specification distinguishes best-effort completeness at Build L3 from the platform-enforced guarantee at Build L4. BASALT should prove that it can independently challenge and validate that guarantee, and then emit interoperable evidence. It should not claim to exceed Build L4 until an actual Build L4 implementation has been tested under matched conditions.

## Benchmark protocol v2

The new machine-readable contract is:

- `benchmarks/basalt/observer-evidence-retention-benchmark.v2.json`
- `benchmarks/basalt/schema/observer-evidence-retention-benchmark.v2.schema.json`

Version 2 leaves the v1 protocol unchanged and adds four named comparison arms:

1. **Bazel strict action sandbox** — pinned version, Linux sandbox strategy, user namespace availability and remote-cache mode.
2. **Nix sandboxed derivation** — pinned derivation/store closure, sandbox policy and network policy.
3. **SLSA Build L4 platform** — only a real platform with documented trusted control-plane controls and verifiable builder identity; a provenance JSON file is not enough.
4. **sccache Rust compiler cache** — pinned cache configuration, explicitly limited to compiler-artifact cache behavior.

The protocol now also measures dependency false attribution: claiming an unused resource as a dependency is not harmless. It creates noisy invalidation, hides observer imprecision and makes a purported dependency graph less useful.

### Non-compensatory targets

Safety is evaluated before selectivity or efficiency. There is no composite score.

| Metric | Proposed target | Why it matters |
|---|---:|---|
| Hidden dependency recall | 100% | Every oracle-confirmed consumed dependency is attributed with the expected identity/digest. |
| Stale-evidence false acceptances | 0 | No cache/evidence reuse after a consumed input or unsupported channel changes. |
| Unsupported-channel disclosure | 100% | Missing/unsupported channels remain visible and block unsupported completeness claims. |
| Integrity-loss fail-closed | 100% | Buffer loss, restart, unmonitored child or snapshot race invalidates the affected claim. |
| Semantic access retention | 100% | Read-but-discard inputs still invalidate when changed. |
| Cargo build-script hidden/mtime/environment challenge detection | 100% | Warm stale state and fresh replay must be distinguished correctly. |
| Dependency false attribution | ≤1% | Resource precision is measured separately from recall. |
| False invalidation | ≤5% | Unrelated mutations should not destroy reusable evidence. |
| Unrelated evidence retention | ≥95% | Selectivity is measured against independently proven unrelated changes. |
| Normalized re-verification work | ≤0.70 | The candidate aims for at least 30% less work than matched global rerun after all safety gates pass. |
| Shifted-task success | ≥3 percentage-point improvement, or preregistered non-inferiority with materially lower cost | A faster verifier that makes tasks fail is not a win. |

All are proposed targets, not measured results or external standards. The existing benchmark includes 12 metrics; v2 adds false-attribution precision and four named industry arms.

### Statistical standard

The current 38-case catalogue is a breadth/coverage checklist, not 38 independent samples. Repeated syscalls, events, mutations and retries inside one task lineage do not increase the independent sample count.

For a one-sided 95% exact-binomial upper bound below 0.1% after observing zero false accepts, the protocol requires at least 2,995 independent task lineages. For a 0.01% bound, the corresponding order of magnitude is 29,956 independent lineages. These are finite-sample bounds, not proof of zero risk. Split development, qualification and untouched confirmation lineages before unblinding; stratify by build system, kernel, architecture, sandbox mode, observation channel, mutation family and task-effect class.

### Fair execution protocol

- Use the same immutable input snapshots, source revisions, toolchains, task definitions, mutation schedule and resource limits across arms.
- Pin exact tool and platform versions, configuration digests, kernel/build identity, namespace and privilege profile.
- Run paired scenarios with both consumed and unrelated mutations, plus negative controls and injected observer faults.
- Preserve all unsupported/failed cases and reasons. Do not silently remove hard strata from denominators.
- Report per-family/per-channel confusion matrices, not just an overall mean.
- Measure wall time, CPU, I/O bytes, trace bytes, snapshot bytes, memory, cache hits, denied undeclared accesses and re-verification work.
- Evaluate safety gates first. Report efficiency and retention only for arms that pass safety.
- Publish exact command/configuration digests and raw receipts for each arm. No best-effort or excluded baseline can be represented as SLSA Build L4.

## How BASALT can become better than today's best systems

The credible path is a composition win, not an unsupported claim that one Rust observer is better than all of Bazel, Nix, SLSA and security telemetry:

1. Reuse the strongest available action sandbox (Bazel/Nix or a BASALT sandbox) for enforcement.
2. Add multiple independent observation lanes and record their exact scope/loss state.
3. Use the independent challenge harness to establish consumed/not-consumed ground truth and adversarially mutate resources.
4. Bind task output and every dependency to content identity and the snapshot used at read time.
5. Make a deterministic reuse decision: reuse only when every dependency and every required observation source remains valid; otherwise invalidate or reverify the smallest affected scope.
6. Emit SLSA-compatible provenance with complete resolved dependency digests and separately authenticated builder evidence.

Only matched benchmark results can establish superiority. The proof threshold is: pass every mandatory safety gate, demonstrate the target false-acceptance bound on held-out lineages, and then show better selectivity/overhead while preserving task success. If BASALT only wins on overhead but misses hidden inputs, it loses.

## Next engineering order

1. Execute the current authored tests on an exact, recorded head and repair compile/rustfmt failures without weakening a gate.
2. Complete an independent Linux observer capture producer for the existing 38 cases, with a separate oracle writer and explicit loss counters.
3. Add SLSA `resolvedDependencies` mapping and an independent verifier test for transitive fetched artifacts.
4. Add Bazel/Nix/sccache harness adapters and pin their exact execution profiles.
5. Run development, qualification and untouched confirmation splits; do not call the 38-case catalog a statistical qualification sample.
6. Only after the safety gates pass, optimize incremental retention and observation overhead.

## Upstream reference points

- Bazel sandboxing: https://bazel.build/docs/sandboxing
- NixOS reproducible builds: https://reproducible.nixos.org/
- SLSA Build track basics: https://slsa.dev/spec/v1.2/build-track-basics
- SLSA build provenance: https://slsa.dev/spec/v1.2/build-provenance
- sccache cache keys: https://github.com/mozilla/sccache/blob/main/docs/Caching.md
- Reproducible Builds definition: https://reproducible-builds.org/docs/definition/
- Falco event sources: https://falco.org/docs/concepts/event-sources/
- Tetragon tracing policies: https://tetragon.io/docs/concepts/tracing-policy/
