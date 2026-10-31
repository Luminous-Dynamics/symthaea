# PITB SorryDB toolchain stratification and smoke cohort

**Date:** 2026-10-31  
**Status:** source-inspected candidate selection; a pinned source verifier is implemented and wired into CI, but no project has been rebuilt and no proof has been attempted.

## Environment is an independent experimental variable

The frozen 1,000-task SorryDB_2601 cohort spans 14 Lean toolchain tags:

| Lean toolchain | Tasks |
|---|---:|
| v4.27.0-rc1 | 318 |
| v4.26.0 | 218 |
| v4.24.0 | 145 |
| v4.26.0-rc2 | 97 |
| v4.22.0 | 72 |
| v4.23.0-rc2 | 37 |
| v4.24.0-rc1 | 36 |
| v4.25.0-rc2 | 22 |
| v4.18.0-rc1 | 14 |
| v4.25.0 | 12 |
| v4.18.0 | 11 |
| v4.26.0-rc1 | 10 |
| v4.21.0 | 7 |
| v4.22.0-rc4 | 1 |
| **Total** | **1,000** |

This is not one homogeneous Lean environment. A toolchain or dependency-cache failure is not a proof-search failure. Every task should have two independently reported phases:

1. **Environment preflight:** source commit, source path/span, Lean binary/toolchain availability, Lake manifest dependency pins, project reconstruction, and clean baseline build.
2. **Proof intervention:** only entered after the original project can be reconstructed and checked at the pinned commit.

Do not merge toolchain_unavailable, source_commit_unavailable, dependency_fetch_failed, project_build_failed, or unknown_preflight into verified_failure. They remain part of the operational assignment denominator, but the theorem-proving Pass@k denominator should be reported only for rows whose preflight reached ready. Publish readiness coverage by toolchain and repository beside proof success. Also report a full-assignment operational yield where preflight failures are non-solves, so systems cannot inflate results by silently dropping difficult environments.

A crude textual scan found 426 debug_info.goal strings with instance-like binder spellings such as inst✝ or bracketed class binders. This is a screening heuristic, not a verified count of LocalInstances registry entries. The exact registry must be exported from Lean's metavariable declaration; a debug goal string is insufficient.

## Three candidate smoke tasks

The versioned manifest is data/benchmarks/proof-insight-transfer-v0/sorrydb-track/smoke-cohort-candidates.v0.json.

| Role | Project / Lean | Task | Why it is included |
|---|---|---|---|
| Stable baseline | LeanCourse25 / v4.24.0 | nullmeasurable_measurable_null | Small project with pinned Mathlib manifest; a basic proof-hole reconstruction control |
| Stable dependency context | PrimeNumberTheoremAnd / v4.26.0 | table_1_prop | Blueprint/Mathlib dependency environment and an analytic-number-theory obligation |
| Local instance challenge | clap-lean / v4.26.0-rc2 | fail₁ | Debug-context preview includes inst✝ : Fact (Nat.Prime p); specifically exercises preservation and capture of local instance synthesis resources |

For all three, the sorry token span was checked directly against the source file at the exact task repository commit. The manifest also pins the source file, lean-toolchain, lakefile.toml, and lake-manifest.json blob identities. The candidate tasks are deliberately from three distinct Lean toolchain tags.

**None of the projects has been rebuilt.** source_inspected: true means only that the file/manifest bytes and exact source token span were retrieved at the recorded commit. The fields environment_reconstructed, project_build_status, lean_evidence_status, and proof_attempt_status remain false/not-run. The debug context is a task-selection hint and must not be used as a replacement for actual Lean state.

## Immediate smoke sequence

1. Fetch each pinned repository commit with no branch-tip substitution.
2. Verify the lean-toolchain, lakefile.toml, and lake-manifest.json bytes against the manifest blob pins.
3. Provision the exact Lean version and all pinned Lake dependencies. Record build-cache presence as an environment characteristic; do not change the dependency graph.
4. Build the untouched project in a clean checkout. If it cannot build, classify the preflight failure and preserve logs.
5. Export the actual metavariable state before solving: target expression, ordered LocalContext, BinderInfo, local declaration values/provenance, LocalInstances registry, and environment/declaration digests.
6. Replace only the designated sorry token in an isolated checkout, run an independent pinned Lean check, and compare the target declaration type digest and axiom audit.
7. Only after the smoke artifacts are reproducible should S0-S4 budgets and model configs be frozen.

## Why this is a better first test

The three tasks vary along orthogonal axes: stable vs release-candidate Lean toolchains, a comparatively small Mathlib course project vs a project with extra blueprint dependencies, and a proof state with an explicit typeclass-instance binder. This is a useful contract/integration smoke test, not a statistically representative accuracy sample. Do not publish Pass@k on three hand-selected tasks.

The full SorryDB report must retain all toolchain strata. The matched causal comparison remains S0 direct baseline, S1 retrieval, S2 context/instance evidence, S3 typed insights, and S4 the combined system. A context or insight arm only demonstrates value if it improves proof completion or cost over the same model and environment; merely retaining LocalInstances is a correctness requirement, not a benchmark win.

## Source pins

- SorryDB snapshot commit: 74b3b63f8ba890d675f362924b21ac55db410383
- Evaluation JSON Git blob SHA-1: af3b663429dc4689ee728d2fb5c7d8d5034476f7
- Raw bytes SHA-256: 26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062

The exact per-task repository/config/source-file pins are recorded in the candidate JSON. This is provenance, not a claim that any repository has been built or any SorryDB row solved.


## Independent checkout verification

The Rust binary \`verify_sorrydb_smoke_cohort\` validates each materialized repository before any build:
- exact repository origin and HEAD commit;
- clean tracked worktree;
- Git blob SHA-1 for the actual \`lean-toolchain\`, \`lakefile.toml\`, \`lake-manifest.json\`, and target Lean source file;
- exact Lean toolchain text and Mathlib revision from the pinned Lake manifest;
- repository-relative path safety and the exact \`sorry\` token span using zero-based Unicode codepoint columns.

The focused workflow checks out each project at its recorded commit and invokes the verifier. Its result is only a source-integrity receipt: it does not establish that dependencies are available, the unchanged project builds, or Lean's actual metavariable state has been exported.

The three sorry spans were rechecked against their exact source commits. Each source-file blob matches its manifest pin, and zero-based character slicing selects exactly \`sorry\` in all three files. This agrees with SorryDB's \`position_to_index\` implementation, which uses Unicode character columns; raw UTF-8 byte offsets would be incorrect for these Lean files.


## Lean exporter compatibility is a separate blocker

The current PITB v2 Lean pilot package is pinned to Lean 4.34.1, whereas the first three SorryDB candidates use Lean 4.24.0, 4.26.0, and 4.26.0-rc2. The v2 JSON evidence contract can be shared across versions, but a compiled Lean exporter cannot be assumed source-compatible across these toolchains. Do not run the current 4.34.1 exporter against older projects by swapping a version string.

The next implementation should split the evidence pipeline into two layers:

1. **Version-local Lean capture shim:** compile inside each task's own Lake environment. It emits only versioned primitive data: target expression, ordered local declarations, binder info, declaration type/value availability, auxiliary-declaration flags, local-instance registry class/type expressions, and direct elaborated proof-term support when a proof exists.
2. **Version-neutral PITB canonicalizer/verifier:** validates the raw shim payload against a strict adapter schema, normalizes stable tags and expression serialization, computes content digests, derives role-aware dependency edges, and verifies that every retained instance reference resolves in the projected context.

Each raw receipt must record Lean version/build identity, Mathlib/Lake dependency revisions, serializer ID, expression options, universe level representation, and source commit. A digest from one serializer/toolchain is not a cross-version theorem identity. The target must be captured and compared within the same pinned environment.

Do not create a lowest-common-denominator exporter that drops local instances or declaration values. Version-specific shims may have different capture internals, but the canonical semantic fields must be complete. If an older Lean version cannot expose a field with the required fidelity, mark that evidence as unavailable and block the corresponding v2 qualification rather than silently defaulting it.

### Compatibility matrix to qualify

| Environment | Candidate role | Required qualification |
|---|---|---|
| Lean 4.24.0 | Stable baseline | Compile capture shim; local context/instance round-trip fixture; exact target digest |
| Lean 4.26.0 | Project/dependency context | Same tests plus Lake/Mathlib manifest binding |
| Lean 4.26.0-rc2 | Explicit instance challenge | Confirm registry membership for Fact (Nat.Prime p) and prove projection digest changes when removed |
| Lean 4.34.1 | Existing PITB pilot | Keep as the development reference; do not treat it as compatibility evidence for older releases |

The first real LocalInstances test must come from Lean's actual metavariable declaration in clap-lean, not the SorryDB pretty-printed goal. The debug text only tells us which task is promising. It does not prove that the printed binder maps one-to-one to a registry entry.

A failed build or unsupported API is a preflight/adapter-compatibility outcome. It is not a proof-search failure and must remain in the operational readiness report.
