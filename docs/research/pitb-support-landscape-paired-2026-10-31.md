# PITB matched-seed support comparisons (2026-10-31)

**Status:** supplemental analysis track and synthetic self-test committed. No real Lean trial ledger, p-value, or empirical performance gain is claimed.

## Why add paired analysis?

The v3 support-landscape ledger schedules the same seed at each replicate index across every context. The exact reliability analyzer uses this to enforce pairing and distinct seeds across replicate indices. But marginal per-context rates alone discard information in the matched design.

For every well-formed candidate context S, this analyzer compares its result with the full mutable context under the same replicate seed and reports the four matched cells:

- both succeed;
- candidate only succeeds;
- full context only succeeds;
- both fail.

It also reports candidate-minus-full paired differences for model calls, tokens, wall time, checker time, and measured cost.

## Statistical procedure

The exact two-sided McNemar test conditions on the number of discordant pairs and uses the Binomial(d, 0.5) null distribution. The script evaluates the tail with exact integer combinations and rational arithmetic, then applies Holm step-down adjustment across all tested candidate contexts within each study group.

This is a matched comparison, not an unpaired difference in two success rates. Its inferential interpretation requires independent replicate pairs, a valid predeclared seed pairing protocol, identical frozen model/policy/toolchain/budget identities, and no hidden cross-run state. Distinct seed digests are necessary but do not by themselves establish independence.

The report is diagnostic. It does not prove that a declaration is logically necessary, that an ablation caused a difference, or that the context is reliable in an absolute sense. The separate v3 Clopper-Pearson/Bonferroni analysis remains the reliability gate. McNemar results are complementary evidence about matched changes versus the full-context baseline.

## Files

- Analyzer: scripts/research/pitb_support_landscape_paired_v0.py
- Workflow: .github/workflows/pitb-support-landscape-paired.yml

The self-test constructs ten paired replicates in which the full context succeeds ten times and a candidate context succeeds eight times. It checks all four contingency cells, the exact McNemar p-value of 1/2 for two discordant pairs in one direction, Holm output, and paired resource-delta generation.

## Next research step

Once a real 2–4 mutable-declaration Lean calibration exists, compare:
1. exact per-context reliability and the reliable minimal-support antichain;
2. paired candidate-vs-full outcome differences;
3. paired resource savings;
4. the same metrics for MVarId.cleanup, dependency closure, random valid contexts, direct/graph retrieval, adaptive search, and HDC/LTC-ranked proposals.

Do not select or report the lowest p-value context. Freeze candidates and contrasts before confirmation, use the Holm-adjusted family, and retain the raw paired contingency table.
