# MAG-QUAL-001 — Sealed-target materials qualification protocol

**Status:** preregistered protocol / implementation draft  
**Date:** 2026-09-19  
**Authority:** benchmark qualification only; no MAT-001 materials-discovery promotion

## Purpose

MAG-001 is intended to test whether Symthaea can generalize into held-out composition/structure regions rather than reproduce values it has already seen. A hidden-target API is insufficient by itself: the generation process must be unable to read target bytes before its submission is frozen.

MAG-QUAL-001 therefore separates the campaign into two authority domains.

## Phase A — target-blind generation

The generator may receive only the public benchmark search view:

- exact search-space artifact;
- exact allowed training artifacts;
- exact family/structure split artifact;
- public metric definitions;
- public SHA-256 commitment to the sealed evaluator targets.

The strong profile additionally requires:

- prediction generation runs with network access disabled by the orchestrator;
- no reusable cache is mounted;
- no evaluator-only credential or secret is present;
- no evaluator target path/file is present;
- the mounted benchmark-data set equals the manifest-derived set exactly, with no extra data artifact;
- exact execution-environment, workspace-inventory, and sanitized process-environment artifacts are SHA-bound.

The `symthaea-materials-qualification` crate freezes the exact serialized submission bytes and emits a `BlindGenerationReceipt` containing the manifest, target commitment, generator artifact, search trace, public search view, and isolation attestation identities.

The byte-level submission digest is deliberate. Re-serializing semantically identical predictions after target disclosure is a different Phase-A subject and must fail qualification.

## Workspace leak canary

`mag_qual_inventory` recursively inventories a designated Phase-A workspace, hashes each regular file, and fails if any file is byte-identical to the public sealed-target digest.

The canary also rejects symlinks and special files in the inventoried root so a target cannot be hidden behind an unbound filesystem redirect.

This inventory is a **canary and provenance artifact**, not proof that no target information exists anywhere on the host. Strong qualification still requires the orchestrator to expose only the intended mounts to the generation process.

## Phase B — sealed evaluation

Only after the Phase-A submission and receipt digests are frozen may the evaluator receive the sealed target artifact.

The evaluator must:

1. revalidate the public manifest;
2. revalidate the complete generation receipt;
3. verify that the submission bytes still match the frozen Phase-A digest;
4. verify the sealed target bytes against the public target SHA-256;
5. score the frozen submission using the MAG-001 evaluator;
6. bind the exact evaluator artifact and execution environment;
7. emit a `BlindEvaluationReceipt` containing the scorecard and its digest.

Any post-disclosure submission mutation is a hard failure.

## Target escrow

Sealed targets **must not be committed to this public repository** and must not be uploaded to any artifact store accessible to Phase A.

A real campaign therefore requires an evaluator-only delivery mechanism such as a protected runner/environment or a separately controlled artifact escrow. The repository currently defines the cryptographic/semantic receipt contract but does not pretend that a repository boolean can enforce host/network isolation.

Before the first real MAG-001 qualification run, the chosen escrow/orchestrator must demonstrate:

- Phase A cannot authenticate to the target store;
- Phase A cannot access a target-containing cache or previous evaluator artifact;
- target disclosure occurs only after the exact generation receipt is immutable;
- Phase B cannot replace the frozen submission;
- evaluator outputs containing target values cannot enter future generator training/cache state.

## Reported metrics

A qualified run should report, separately:

- family/structure-holdout coverage;
- scalar MAE and RMSE by property;
- promising-candidate precision, recall, and F1;
- OOD detection performance;
- cross-fidelity calibration performance;
- unresolved source/target contradictions;
- generator compute/resource cost;
- number and fraction of candidates escalated to higher fidelity.

A single aggregate leaderboard score is not authoritative.

## Interpretation

Possible outcomes include `PASS`, `MIXED`, `NULL`, or `NEGATIVE`; qualification infrastructure must preserve all of them.

Even a strong MAG-QUAL-001 PASS means only that Symthaea reproduced/generalized against the preregistered hidden benchmark under the bound protocol. It does **not** mean that Symthaea discovered a new material, established synthesis, demonstrated coercivity, proved economic relevance, or advanced a candidate on the MAT-001 evidence ladder.

Prospective materials claims require their own literature-novelty, local first-principles, synthesis, characterization, replication, and technoeconomic evidence.
