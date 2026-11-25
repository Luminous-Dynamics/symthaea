# Lean checker diversity and common-mode assurance — v1

**Issue:** [#13637 — CHECKER-DIVERSITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/13637)  
**Parent assurance capsule:** [#13606 — LEAN-TRUST-CLOSURE-CAPSULE-001](https://github.com/Luminous-Dynamics/symthaea/pull/13606)  
**Status:** source-grounded registry, strict run-profile schema, synthetic fixture, and Rust semantic-validator prototype. No checker was executed by this work and no proof is qualified by the synthetic profile.

## 1. The paradigm shift: checker count is not checker diversity

A naive assurance report says “six checkers passed.” That is not enough.

The current Lean high-assurance toolchain includes the official kernel plus the bundled `lean4lean`, `nanoda`, `con-leche`, `con-ron`, and `leanchecker-paranoid` execution paths. But these paths do not represent six independent algorithms:

- `lean4lean` is derived directly from the official C++ kernel and may share algorithmic bugs.
- `leanchecker-paranoid` is an allocator-hardened build of the official checker, not a new kernel algorithm.
- `con-ron` is a Rust port/refinement of `con-leche`; it introduces implementation, language and runtime diversity while sharing the core checking algorithm lineage.
- `con-leche` has its own term representation and a machine-checked consistency argument.
- `nanoda` is an independently implemented Rust checker.

This yields a conservative source-grounded topology:

| Execution path | Language | Algorithm lineage | What diversity it adds |
|---|---|---|---|
| Official Lean kernel | C++ | Lean core kernel | Reference implementation |
| lean4lean | Lean | Lean core kernel lineage | Different language / partial formal verification; not counted as a separate algorithm family |
| leanchecker-paranoid | C++ | Lean core kernel lineage | Hardened allocator/runtime variant; no new algorithm family |
| nanoda | Rust | nanoda | Independent checker implementation and algorithm lineage |
| con-leche | Lean | con-leche model checker | Independent representation and machine-checked consistency basis |
| con-ron | Rust | con-leche model checker | Verified Rust refinement and separate runtime; not counted as a separate algorithm family |

**Conservative diversity vector for the full six-path set:** six execution records; six implementation families; three algorithm-lineage families; three implementation languages; two independently implemented external algorithm lineages (nanoda and con-leche). Known term-representation and parser-family counts are reported only where the registry has evidence. Unknown is never silently converted into diversity.

The registry is not a ranking and does not produce an independence score.

## 2. Source-grounded registry

The versioned reference is [LEAN_CHECKER_DIVERSITY_REGISTRY_V1.json](LEAN_CHECKER_DIVERSITY_REGISTRY_V1.json), validated by [LEAN_CHECKER_DIVERSITY_REGISTRY_V1.schema.json](LEAN_CHECKER_DIVERSITY_REGISTRY_V1.schema.json).

It records for each checker:
- implementation family and source-revision pinning policy;
- algorithm lineage;
- implementation language;
- known term representation and parser family, or explicit unknown;
- assurance basis and evidence references;
- residual trust assumptions.

It also records every pairwise checker relationship. There are 15 pairs for six checkers. A pair with insufficient evidence is explicitly marked `unknown` and is not counted as independent. The graph distinguishes:
- derived from;
- hardened build of;
- verified refinement / port of;
- independent implementation of;
- shared algorithm lineage;
- unknown relationship.

### 2.1 Why con-ron is not a second con-leche algorithm

The Lean Kernel Arena documents that con-ron is con-leche ported to Rust, with its own hash-consed term representation. A Lean twin of the pure checker is proved to refine con-leche; Aeneas then translates the Rust checker back to Lean and proves refinement to that twin. This means con-ron gives useful protection against a subset of compiler/runtime/bignum failures that con-leche and Lean might share, while retaining the con-leche checking algorithm lineage. It is implementation diversity, not a second independent algorithm family.

### 2.2 Why lean4lean and leanchecker-paranoid need different labels

The Lean Kernel Arena describes lean4lean as derived directly from the C++ kernel, and warns that it may share implementation bugs. The Lean release tooling describes `leanchecker-paranoid` as a hardened checker build. Both are useful, but neither should inflate the count of independent algorithm families.

### 2.3 Why con-leche and nanoda matter

The con-leche project documents its own term representation and a model-based consistency argument: accepted declaration sets have a set-theoretic model, implying that it does not accept a proof of False. Nanoda is described as an independently implemented Rust checker. They provide two external algorithm lineages distinct from the official Lean kernel lineage.

These are not interchangeable assurance bases:
- an implementation described as independently implemented is still trusted unless its own soundness theorem is available;
- a machine-checked consistency argument has its own proof, parser, driver and metatheory assumptions;
- a verified refinement transfers a stated property subject to the correctness of the refinement proof and unverified driver boundaries.

## 3. Same-export run profile

[LEAN_CHECKER_DIVERSITY_RUN_V1.schema.json](LEAN_CHECKER_DIVERSITY_RUN_V1.schema.json) defines a task-specific run profile. A profile binds:
- exact exported proof bytes by SHA-256;
- Lean toolchain and export format;
- challenge and solution source digests;
- comparator version, config digest, paranoid-mode flag and sandbox profile;
- axiom policy identity and digest;
- each requested checker, exact checker/binary revision, input export digest, result and raw receipt digest;
- a recomputed diversity vector;
- an explicit profile disposition and claim ceiling.

The synthetic fixture [LEAN_CHECKER_DIVERSITY_RUN_SYNTHETIC_V1.json](LEAN_CHECKER_DIVERSITY_RUN_SYNTHETIC_V1.json) demonstrates the expected six-path vector. It is deliberately labelled `synthetic_fixture_not_execution`; its disposition is `draft`. It is not a comparator receipt.

### 3.1 Outcome semantics

The run model does not force every exit code into pass/fail.

| Outcome | Meaning | Counts as rejection? |
|---|---|---:|
| `accepted` | Checker accepted the exact export under the recorded policy | No |
| `rejected` | Checker explicitly rejected the export | Yes, for checker concordance |
| `declined` | Checker cannot handle the artifact/feature | No |
| `error` | Checker or wrapper failed | No |
| `timeout` | Execution exceeded its limit | No |
| `resource_exhausted` | Memory/process/resource limit prevented a result | No |
| `not_run` | Checker was requested but not executed | No |

A checker decline, panic, timeout or OOM is not mathematical evidence that a theorem is false. It is a missing assurance result. Preserve exit codes, signals and raw stderr separately.

This distinction is operationally important: the Lean Kernel Arena's scoring model separates incorrect answers from declines/timeouts/crashes, and the nanoda Arena integration explicitly notes a limitation where rejected proofs and other failure modes are conflated. Symthaea's receipt must preserve the distinction instead of copying an ambiguous wrapper summary.

### 3.2 Qualification semantics

A profile may report `all_required_accepted` only if:
1. every requested checker was actually run;
2. every run checked the exact same export digest;
3. every required checker accepted under the same versioned axiom policy;
4. the comparator/orchestrator reports success;
5. the minimum diversity vector is met;
6. no checker relationship is being silently treated as independent.

For a `--paranoid` bundle profile, the requested set must equal the full six-path registry. A custom subset may be useful, but it is not reported as a full paranoid bundle.

If any checker rejects, the profile is `rejected` for checker concordance. That does **not** automatically mean the informal theorem is false; the rejection may concern the artifact, axiom policy, unsupported feature, or a checker defect and must be investigated. Any decline, error, timeout, resource exhaustion or missing run produces `inconclusive`.

## 4. Rust semantic validator

Added to the existing `scripts/research/proof-plan-contract` crate:

- `src/checker_diversity.rs`: registry, pairwise-lineage and same-export profile validator.
- `src/bin/lean-checker-diversity-contract.rs`: CLI.

The validator checks:
- exact six-checker registry identity;
- complete pairwise relationship coverage (15 pairs);
- known lineage constraints (lean4lean and paranoid with official; con-ron with con-leche);
- unknown relationships remain uncounted;
- profile digest binds the exact registry bytes;
- unique requested checker IDs and one run record per request;
- same export digest for every executed checker;
- exact recomputation of all diversity counts;
- checker outcome semantics and profile disposition;
- full bundle coverage whenever paranoid mode is declared.

The CLI validates **record consistency only**. It does not execute Lean, invoke comparator, or establish theorem correctness.

Example invocation:

```bash
cargo run --locked \
  --manifest-path scripts/research/proof-plan-contract/Cargo.toml \
  --bin lean-checker-diversity-contract -- \
  docs/research/LEAN_CHECKER_DIVERSITY_REGISTRY_V1.json \
  docs/research/LEAN_CHECKER_DIVERSITY_RUN_SYNTHETIC_V1.json
```

This example should report a structurally valid *synthetic draft* profile, not a checker execution or proof pass.

## 5. Benchmark: from checker agreement to differential fault discovery

Use the Lean Kernel Arena as an external regression source, but freeze each round, checker revision, Lean version, export version and test artifact. Results from different rounds are not directly comparable.

The benchmark should report:
- false acceptance on known-invalid exports;
- false rejection on known-valid exports;
- decline rate;
- timeout rate;
- resource exhaustion / crash rate;
- agreement and disagreement matrix;
- parser/serialization coverage;
- mutual-inductive, nested-inductive, quotient, recursor, definitional-equality and proof-irrelevance cases;
- median/p95 runtime, instruction count where available, memory and throughput;
- cost per independently accepted export.

### Prospective targets

| Metric | Initial target |
|---|---:|
| Wrong acceptance on frozen soundness controls | 0 observed |
| Wrong rejection on supported positive controls | 0 observed |
| Unclassified process outcomes | 0 |
| Same-export digest binding | 100% |
| Full paranoid bundle coverage when claimed | 100% |
| Pairwise lineage classifications | 100% explicit; unknown remains a valid classification |
| Independent external algorithm lineages in full bundle | At least 2 |
| Algorithm lineages in full bundle | At least 3 |
| Languages in full bundle | At least 3 |
| Differential disagreement triage | 100% of observed disagreements get minimized/reproduced or remain explicitly unresolved |

Zero observed wrong acceptances is not proof of zero error probability. Publish corpus size, coverage, confidence bounds and every declined case.

## 6. Large-artifact scalability case study

The public verification record for the recently announced Lean formalization of the Riemann Hypothesis / Dirichlet GRH is a valuable scale case, separate from the small adversarial corpus. As of the MIT verification page's 24 November 2026 update, it reports:
- 175,187 modules compiled;
- a 61,570,595,788-byte exported proof;
- a sandboxed rebuild reproducing the export byte-for-byte;
- nanoda and con-leche accepting the full export;
- con-ron and con-leche challenge/solution comparisons;
- lean4lean accepting every compiled module;
- very different throughput and resource profiles, including multi-day replay by the Lean default and paranoid kernels.

The same page says the end-to-end `lake comparator` run was still in progress at that timestamp. Its stated theorem scope is the Riemann zeta function and Dirichlet L-functions; it explicitly does not cover Dedekind, Hecke or general automorphic L-functions. Treat this as a published external verification/scalability record, not a Symthaea result and not a blanket claim about every generalized Riemann hypothesis.

This case suggests an important next systems problem: **resource-aware verification orchestration**. A high-assurance profile should record checker-specific memory/time budgets, allow compatible independent checking on separate hardware, and preserve the distinction between accepted, declined and unfinished checks. Do not use one global timeout that systematically removes the slower checker from the assurance set.

## 7. Next research increments

1. Run the registry/profile semantic tests and JSON schema validation once a Rust toolchain is available; no Rust execution is claimed by this commit.
2. Add a first observed small `And.comm` challenge/export run through the pinned full comparator bundle.
3. Preserve per-checker raw exit status/logs and confirm byte-identical export digests.
4. Add Lean Kernel Arena soundness, completeness and corner-case fixtures under a frozen round snapshot.
5. Add export mutation and parser differential fuzzing; minimize every disagreement.
6. Add a scale lane using public large-artifact manifests, with resource budgets and explicit pending/declined states.
7. Integrate the run-profile digest into LeanTrustClosureCapsule and ECPCB task receipts.

## 8. Claim ceiling

This work records checker provenance and measures the diversity that is actually evidenced. It does not prove checker soundness beyond each checker's own assurance basis, remove common-mode bugs, prove export format correctness, validate the human meaning of the challenge statement, or certify OS/hardware/sandbox security.

**The objective is not to maximize the number of green checkmarks. It is to make each additional check reduce a named, evidenced failure domain.**
