# Nix evaluation diagnostics and counterfactual repair

Status: initial deterministic diagnostic core proposed in v0.1. This document
sets the evidence contract and benchmark plan; it does not claim automatic
repair or physical-host qualification.

## Why this layer exists

Nix already provides a reproducible evaluator and build graph. Nixward should
not attempt to replace either. Its differentiated job is to connect evaluator
evidence to configuration provenance, causal diagnosis, bounded repair
experiments, authorization, and post-change verification.

The incident that motivated this work had two independent signals:

- Fatal evaluation failure: a local overlay dereferenced prev.brave, while
  the pinned nixpkgs revision exposed a throwing alias because Brave had been
  removed.
- Non-fatal cache warning: cuda-maintainers.cachix.org returned HTTP 401.
  That warning affected substituter access; it was not the cause of the failed
  NixOS evaluation.

A diagnostic engine must preserve that distinction. Treating every line in
stderr as one undifferentiated failure creates bad repairs (for example,
changing CUDA configuration to address a package-alias exception).

## v0.1 scope

The first implementation is a deterministic, non-executing parser over a
captured stderr string. It recognizes removed/renamed packages, unknown NixOS
options, module assertions, type conflicts, fixed-output hash mismatches, and
cache authorization/availability warnings.

Every finding carries:

- a stable issue kind and severity;
- the original evidence line and one-based line number;
- a source frame when it can be parsed from the evaluator trace;
- review prompts, never executable patches.

Unknown errors remain unknown. Cache authentication and availability findings
are warnings and do not become the primary failure. The parser does not invoke
Nix, read configuration files, inspect credentials, write files, update
flake.lock, or authorize system changes.

## Paradigm shift: diagnose a system transition, not a terminal message

The intended control loop is:

1. **Observe**: capture the exact command, exit status, stderr, flake reference,
   working-tree identity, and lockfile digest.
2. **Normalize**: retain structured Nix log events when available and parse
   human-readable evaluator traces without discarding original bytes.
3. **Localize**: map evaluator frames to source files, option definitions,
   overlay expressions, flake inputs, and locked revisions.
4. **Explain**: build an evidence-linked causal graph. Distinguish direct
   observations from inferred edges and unknowns.
5. **Generate counterfactuals**: create candidate patches in isolated
   worktrees/staging trees. Never edit the operator's live checkout in place.
6. **Evaluate**: run read-only syntax/evaluation checks and build the expected
   system derivation. Keep the original and candidate input graphs pinned.
7. **Challenge**: run negative controls and regression checks; reject a
   candidate if it weakens unrelated assertions, signatures, hashes, or policy.
8. **Authorize**: bind an owner-approved capability to the exact source diff,
   lock graph, command, target machine, and rollback binding.
9. **Realize and verify**: build the immutable closure, activate only through
   the existing transaction boundary, and verify observed post-state before
   promotion.

Cognitive confidence, HDC similarity, generated explanations, and model
preferences are advisory. None is a machine-mutation authority.

## Complementary ecosystem tools

Nixward should use the best existing tools as execution and verification
substrates:

- Nix flake check and nix build remain the native validation/build interface.
- nix-eval-jobs can parallelize attribute evaluation and stream JSON result
  records; it is a useful adapter for large multi-host or flake output sweeps.
- NixOS testers.runNixOSTest provides QEMU-backed integration tests for service
  health, activation behavior, networking, and recovery.
- nil / nixd, nixfmt, statix, and deadnix are complementary developer tools for
  language intelligence, formatting, and static checks.
- Deployment tools such as Colmena, deploy-rs, and Clan address orchestration
  and machine management. Nixward's opportunity is the evidence-linked
  diagnosis/repair and owner-authority boundary across those workflows.

We should not claim a universal performance win over Nix's evaluator. A better
result is fewer unsafe repairs, faster root-cause localization, and stronger
evidence for each host transition.

## Benchmark proposal: NixWard-EvalRepair

### Dataset

Start with a versioned corpus of at least 200 sanitized incidents, expanding
toward 1,000. Include real traces where redistribution is permitted and
synthetic fixtures with explicit provenance. Cover:

1. removed or renamed nixpkgs attributes;
2. missing NixOS options and missing module imports;
3. module type/priority conflicts;
4. failed module assertions;
5. Nix syntax and parse failures;
6. fixed-output hash mismatches;
7. flake input fetch/lock failures;
8. cache HTTP 401/403/404 and network unavailable;
9. derivation build failures (kept distinct from evaluation failures);
10. post-activation service/boot health failures.

Each case should have a primary cause, optional secondary warnings, source
location where available, severity, expected evidence, and a minimal valid
repair when one exists. Split train/dev/test by incident family and
upstream revision to avoid near-duplicate leakage.

### Metrics and target gates

These are research targets, not measured results:

| Metric | v0.1 target | Why it matters |
|---|---:|---|
| Fatal-vs-warning separation | 99% | Avoid repairs aimed at irrelevant cache warnings |
| Top-1 diagnostic class | 95% | Ground the explanation in known failure modes |
| Exact package/option extraction | 98% | Make localization actionable |
| Source-file localization | 95% | Find the owning configuration layer |
| Exact line localization | 85% | Useful precision without overclaiming |
| Unknown-error abstention | 100% on held-out unknown classes | Prevent plausible hallucinated fixes |
| Unauthorized host mutations | 0 | Hard safety invariant |
| Candidate regression rate | 0 in the qualification corpus | A repair must not degrade protected invariants |
| Parser latency | under 10 ms per 10,000-line trace | Keep local diagnosis interactive |

For counterfactual repair, measure more than pass rate:

- candidate evaluation success on exact pinned inputs;
- independent reproduction of the candidate derivation;
- protected-option and policy invariant preservation;
- negative-control rejection rate;
- rollback/recovery success in VM tests;
- time-to-diagnosis and time-to-verified-recovery;
- proportion of cases correctly refused because evidence is insufficient.

### Comparisons

| System/tooling | Strong at | Nixward differentiation to prove |
|---|---|---|
| Native Nix evaluator / nix flake check | Authoritative evaluation and output checks | Explain why a failure occurred and preserve causal evidence |
| nix-eval-jobs | Parallel evaluation, JSON streaming, per-attribute failures | Correlate failures across source, lock graph, and target intent |
| NixOS VM tests | Reproducible behavioral/integration verification | Generate targeted counterfactual tests and bind receipts to the exact repair |
| nil / nixd and static linters | Editor diagnostics, language intelligence, static hygiene | Connect static findings with runtime evaluator evidence |
| Colmena / deploy-rs / Clan | Deployment orchestration and fleet operations | Provide a portable evidence/authority/recovery envelope around a change |

The competitive claim should be about end-to-end assurance, not raw evaluation
speed. We must publish matched baselines, datasets, hardware, exact Nix and
nixpkgs revisions, and failed cases.

## Immediate engineering sequence

1. Land the deterministic stderr classifier with regression fixtures.
2. Add a JSON report command that only consumes an explicitly supplied log;
   no implicit host reads.
3. Add a source/flake evidence collector that is read-only and content-addressed.
4. Add the first counterfactual runner using a temporary Git worktree and a
   constrained command allowlist.
5. Use nix build for closure realization and testers.runNixOSTest for
   service/boot behavior before any physical activation.
6. Bind candidate diff + lock graph + command + validation receipts into the
   existing ChangePlan/SystemTransaction protocol.
7. Add detached Ed25519 owner authority and consume-once replay evidence at the
   mutation boundary; never treat CLI --approve or Phi as equivalent.
8. Expand the corpus and publish reproducible benchmark reports.

## Known limitations of v0.1

- It consumes captured stderr; it does not yet ingest Nix's internal JSON event
  stream or nix-eval-jobs JSON records.
- It does not inspect local source files. A parsed stack frame is a location,
  not proof of the source contents.
- Error classification is rule-based and intentionally incomplete.
- Recommendations are static prompts, not generated patches.
- No live host, QEMU, CI, or upstream nixpkgs build has been run by this
  authoring environment. Qualification remains pending.
