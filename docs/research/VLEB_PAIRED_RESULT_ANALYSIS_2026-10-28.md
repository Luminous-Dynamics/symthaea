# VLEB paired-result analysis protocol — 2026-10-28

## Why this exists

A benchmark can have trustworthy receipts and still overstate progress if its analysis changes the denominator after outcomes are known, treats attempts on one task as independent observations, reports only raw resolution, or compares two systems on different tasks. This module adds deterministic analysis of a locked, task-paired results file.

It is an analysis layer, not an evidence-authentication gate. A result labeled assured_pass must already have passed the independent behavioral oracle and all VLEB evidence gates. The input file must be pinned by the protected evaluator; this tool verifies the exact input SHA-256 supplied by the evaluator at the CLI boundary.

## Required reporting

For each candidate and each task lane, the tool reports:

- assurance-adjusted semantic success rate (AASSR), with a Wilson confidence interval;
- raw resolution rate, including raw passes whose evidence is not yet sufficient;
- counts/rates for pass, raw-but-unassured pass, fail, unresolved, and abstained;
- unresolved and abstention rates;
- total cost, conventional median and p90 duration, and cost per independently verified pass; excluded-only lanes remain visible with zero scored tasks and their incurred costs;
- an exact two-sided McNemar p-value and task-paired AASSR risk difference for the preregistered A/B comparison.

The primary denominator is the locked cohort after only documented task exclusions adjudicated before outcome unblinding. Every included task must have results for both candidates. Costs and elapsed time already incurred on excluded-but-executed assignments remain in the economics totals; exclusion must not make a system appear cheaper. Missing results are invalid input, not an implicit drop. Exclusions require a reason and a digest-pinned adjudication record; the script can verify presence and shape of that record but cannot prove that blinding was genuine.

## Statistical interpretation

The Wilson interval is used instead of a naive normal interval for each binomial rate. The exact McNemar test compares paired binary assured-pass outcomes and conditions on discordant pairs. A fixed task-level aggregation rule must be preregistered (for example, three attempts summarized as pass@3); attempts must not be counted as separate independent tasks.

The script reports effect size and p-value, not only statistical significance. A nonsignificant result is not evidence of equivalence. Equivalence or non-inferiority requires its own preregistered margin and a suitable confidence-interval procedure. If multiple lanes, model variants, or hypotheses are examined, the report must disclose them and address multiplicity; the primary test should remain preregistered.

## Security and validity boundaries

- --expected-input-sha256 is mandatory: an input mismatch exits nonzero before analysis.
- Candidate configuration digests and task/protocol identities are recorded; the upstream evaluator must independently pin them.
- Non-finite costs, negative costs, invalid durations, duplicate task IDs, missing paired outcomes, unpinned protocol digests, and post-unblinding exclusions are rejected.
- Failed, unresolved, abstained, and raw-but-unassured results remain non-passes for AASSR.
- If there are zero independently verified passes, cost per verified pass is undefined (null), not zero.
- The tool does not itself validate GitHub attestations, witness receipts, task oracles, or semantic truth. Those controls must run first.

## External methodological grounding

SWE-bench Pro V2's September 22, 2026 refresh removed 89 invalid tasks, fixed 69 instruction/test contradictions, regraded diffs on pristine images and added model-endpoint-only network access (Scale Labs methodology: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full). This is evidence that task validity and clean regrading materially affect benchmark trustworthiness.

A 2026 audit of SWE-bench submissions reported that top-ranked differences could be statistically unstable and scaffold choice could coincide with substantial score variation (paper: https://arxiv.org/abs/2609.17394). This is motivation for paired task-level comparisons and disclosure, not proof that candidate systems are equivalent.

LegacyCodeBench adopts execution-based checking of claims about original COBOL programs rather than using generated code as its own oracle (project and paper: https://kalmantic.github.io/legacycodebench/). VLEB should combine that idea with independent provenance and locked statistical analysis.

## Reproducible invocation

From a trusted evaluator checkout, pin the complete results file out of band and run:

    python tools/vleb/analyze_paired_results.py \
      --input /protected-evidence/paired-results.json \
      --expected-input-sha256 "$TRUSTED_PAIRED_RESULTS_SHA256" \
      --output analysis.json

Retain the exact input, analysis output, tool source digest, protocol digest, model/scaffold configuration digests, and task-level results with the confirmatory evidence bundle.
