# GCS Topology Signals: Advisory-Only Contract (2026-10-22)

## Status and claim boundary

This change adjusts how existing Geodesic Code Synthesis (GCS) diagnostics influence the coding agent. It is a conservative control-flow and epistemic-boundary change, not a claim that GCS improves correctness or that topology fingerprints are useless. Source-level read-back only has been performed in the authoring environment: Rust/Cargo, feature-gated compilation, unit tests, and end-to-end behavioral runs have not been executed here.

## Problem

The prior gate converted a nonzero second Betti number into an assertion that unreachable code regions existed, and it described abstract operation-sequence non-convergence as a possible infinite loop. Those signals may motivate investigation, but neither is an independent proof of control-flow reachability or program termination. More importantly, any finding sent a candidate back to the Fixing phase and was inserted into the next prompt as **MUST FIX**, before independent behavioral testing could run.

## New contract

1. GCS emits hypotheses, explicitly labelled as advisory rather than verified defects.
2. Successful generation/compilation proceeds to the independent Testing phase even when topology advisories exist.
3. Advisories may inform repair only when another validation signal warrants repair; they do not veto a test-passing candidate.
4. Prompt text explicitly says not to change correct behavior solely to satisfy a heuristic and that compiler diagnostics and behavioral tests take precedence.
5. The no-feature implementation remains a no-op; ordinary builds should retain the same phase transition except for renamed diagnostic state.

## Falsifiable qualification plan

Run the existing coding-agent task suite with the same fixed tasks, model/backend, seed policy, timeout, repository commit, and resource budget in paired modes: (A) topology diagnostics disabled, (B) diagnostics enabled but advisory-only. Do not mix this experiment with changes to the code generator or test oracle.

| Metric | Why it matters | Proposed acceptance rule |
|---|---|---|
| Held-out behavioral pass@1 | Primary user-visible correctness outcome | No statistically supported regression; seek improvement, not a cosmetic topology score |
| Test-passing candidate false-repair rate | Measures whether valid output is harmed by a heuristic | Zero hard rejections caused solely by GCS |
| Extra repair iterations per task | Measures whether advisories waste work | Report paired median and p95; lower is better at equal quality |
| Verified fixes per unit cost | Measures practical capability | Include model calls/tokens, wall time, compile/test time, and compute |
| GCS signal precision/recall against confirmed defects | Validates the heuristic itself | Label findings independently; require confidence intervals before considering any hard gate |
| Regression rate on held-out tests | Detects destructive fixes | No increase in regression; record pass-to-pass and fail-to-pass outcomes |

Stratify by task family and feature configuration. Use bootstrap confidence intervals or paired randomization, publish the task-manifest digest, record exact code/model/runtime revisions, and keep a private held-out split. Never turn a queued check or unexecuted local test into a pass.

## Broader paradigm

Treat Symthaea's signals as a hierarchy of evidence. Static structure and HDC/topological similarity propose hypotheses; compiler/type checking validates language constraints; a complete behavioral oracle validates specified examples; property/metamorphic tests expand coverage; fuzzing and differential tests probe broader behavior; formal proofs cover only the claims and semantics actually modeled. A weaker layer may prioritize exploration but must not masquerade as a stronger layer.

## Validation commands

From a checked-out branch, run the default code-generation prompt test and the feature-gated code-generation suite in the pinned Nix environment. Also test both with and without `geodesic_synthesis`, then execute paired fixed-task experiments and inspect failures manually.

```bash
cargo test --features code_generation --lib gcs_advisory_tests
cargo test --features 'code_generation geodesic_synthesis' --lib
```

These are proposed qualification steps, not recorded passes. Confirm exact feature names and the repository's canonical Nix shell before using them; no result should be reported until run on the exact source head.

## Related work

- [Coding benchmark oracle integrity PR](https://github.com/Luminous-Dynamics/symthaea/pull/8310): fail-closed test-oracle scoring.
- [Mutation-control benchmark PR](https://github.com/Luminous-Dynamics/symthaea/pull/8313): reference/mutant controls and reproducible receipt format.
- [SWE-Bench Pro V2 task-quality controls](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2): coding benchmark tasks need validation and a reference-patch-pass / empty-patch-fail gate.