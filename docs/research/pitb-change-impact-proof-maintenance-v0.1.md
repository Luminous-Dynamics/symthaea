# PITB Change-Impact and Proof-Maintenance Benchmark (CIPM) v0.1

**Status:** proposed original benchmark; design only, no tasks scored.  
**Priority:** P1 research track; build from the Aegis and Vero adapters after their frozen-input contracts stabilize.  
**Tracking:** [PITB Aegis integration #10272](https://github.com/Luminous-Dynamics/symthaea/issues/10272)

## 1. Research gap

Aegis measures proof obligations over implementation-derived tasks. VeriSoftBench measures repository context and dependencies. Vero measures whether an agent can complete an entire code-and-proof repository. CoCo-Prover studies cost-aware scheduling over proof graphs.

A further operational problem remains under-measured: **when verified software changes, can the system identify exactly which claims and artifacts are invalidated, preserve what remains valid, and repair the minimum necessary proof set?**

A system that proves a repository once but cannot safely maintain its guarantees across changes is not yet a reliable engineering system.

CIPM evaluates proof maintenance as a first-class capability.

## 2. Task construction

Start from a frozen, fully buildable and independently replayed repository snapshot. Apply a deterministic, versioned change to one or more source declarations, specifications, helper lemmas, imports, toolchain inputs, or build configuration.

Each change has a gold dependency-impact set established by independent graph analysis plus a separately validated oracle. Keep direct and transitive impact distinct.

Change families:
- implementation-preserving refactor;
- behavior-preserving representation change;
- local behavior change with narrow proof impact;
- shared helper lemma change;
- type or API signature change;
- invariant strengthening or weakening;
- specification-only change;
- dependency/import/toolchain change;
- deliberate stale-proof or stale-receipt replay;
- semantic mutation that should invalidate a previously true claim.

Include no-op and irrelevant-file changes as negative controls. The generator must preserve parent/child snapshot provenance and exact patch bytes.

## 3. Required system actions

For each changed snapshot, the system must:
1. identify changed source and proof artifacts;
2. compute the affected dependency closure;
3. classify claims as still valid, invalidated, unknown, or requiring revalidation;
4. reuse unaffected proofs only when their dependency and environment evidence remains valid;
5. regenerate or repair affected proofs;
6. rebuild and replay the complete required repository state;
7. produce a receipt that binds old snapshot, patch, new snapshot, impact set, retained proofs, invalidated proofs, repairs, and final verifier evidence.

The system must fail closed when dependency information is missing or stale. Unknown impact is not the same as unaffected.

## 4. Core metrics

### Impact precision and recall

Let G be the oracle set of artifacts/claims that truly require invalidation, and P the system-predicted invalidation set.

- Precision = |P ∩ G| / |P|. Measures unnecessary invalidations.
- Recall = |P ∩ G| / |G|. Measures missed invalidations.

Report both per change family. A high precision system that misses stale proofs is unsafe; a high recall system that invalidates everything is uneconomical.

### Maintenance and integrity

- stale-proof acceptance rate (must be zero on seeded controls);
- stale-receipt replay acceptance rate (must be zero);
- unaffected-proof preservation rate;
- repair success rate on affected obligations;
- full repository build/replay completion;
- proof repair cost and total change-to-green latency;
- number of revalidated declarations and proofs;
- unnecessary rebuild/reproof work;
- dependency-graph update correctness;
- exact-head reproducibility.

### Economic measures

- compute and API cost per successfully maintained change;
- p50/p95 change-to-verified-green latency;
- cost of conservative full revalidation vs selective revalidation;
- incremental maintenance cost as a function of graph size and change radius.

## 5. Safety gates

- 100% rejection of seeded stale proof and stale receipt controls.
- 100% recall on curated critical semantic changes before expanding the hidden suite.
- No proof may be retained solely because its text is unchanged; its transitive dependencies and environment must also remain compatible.
- No proof may be marked unaffected when the impact graph is incomplete.
- All retained proofs must be replayed under the declared trust policy before release.
- Unknown/ambiguous impact must trigger conservative revalidation, not optimistic reuse.

The finite-suite 100% targets are gates for the curated suite, not universal safety guarantees. Report sample sizes and confidence bounds.

## 6. Paradigm-shift hypothesis

Move from **proof generation** to **proof lifecycle management**.

A verified software repository should carry a live, machine-readable assurance graph. Each claim points to:
- source and specification identities;
- translation/model identity;
- supporting lemmas and imported declarations;
- toolchain and axiom policy;
- proof artifact and independent replay receipt.

A change becomes an explicit graph transformation. The system computes a conservative impact frontier, selectively repairs it, and independently verifies the resulting repository state.

The target is not simply fewer rebuilds. It is a calibrated minimum repair set with zero stale-proof acceptance.

## 7. Experimental comparisons

Compare:
1. full repository rebuild/reproof after every change;
2. text-based changed-file heuristics;
3. dependency-graph invalidation without semantic classification;
4. PITB dependency-aware impact analysis with conservative unknown handling;
5. full PITB maintenance controller with proof reuse and repair.

Use identical repository snapshots, patches, toolchains, and compute budgets. Report paired task-level outcomes. Include benign changes, semantic changes, and adversarial stale-evidence controls.

## 8. Relationship to other benchmarks

- Aegis supplies implementation-derived program/contract tasks.
- Vero supplies multi-module repositories and complete-instance acceptance.
- VeriSoftBench supplies repository dependency-context tasks.
- SorryDB supplies reusable proof insight and integrity-aware retrieval.
- CoCo-Prover supplies a cost-aware proof-graph scheduling comparator.
- SIB supplies adversarial semantic and receipt mutations.
- CIPM evaluates whether verified claims survive change safely and economically.

Keep separate benchmark identities and scorecards. CIPM may use subsets or derivatives of public benchmarks only where licensing and task terms permit; never imply that its generated mutations are official Aegis or Vero tasks.
