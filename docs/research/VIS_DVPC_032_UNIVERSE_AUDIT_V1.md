# VIS-DVPC-032 — Candidate-action universe completeness audit v1

**Status:** structural preflight. It does not estimate reward, certify support, or establish policy-value identification.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8762

## Why this exists

A trace can contain probabilities that sum to one while still omitting actions from the candidate universe if a preprocessing step renormalized probabilities over a truncated list. A digest of the trace proves the bytes are unchanged, not that those bytes describe every action the policy could select.

This verifier compares all unique \`(context_id, action_id)\` pairs in the observed trace against a separate evaluator-supplied candidate manifest. A match is required before the downstream support audit or OPE estimator is run. Missing contexts, missing action pairs, unexpected contexts, extra pairs, duplicate pairs, malformed rows, and digest mismatches are explicit failure cases.

## Trust contract

The candidate manifest must come from an independent trusted source: e.g. the evaluator that produced the logging-policy/candidate set, a signed campaign receipt, or a versioned catalogue/availability manifest. The expected manifest digest must come from a trusted upstream receipt, not be computed from the same untrusted file at the moment of checking. A SHA-256 digest provides byte identity but not authenticity.

The tool checks candidate-universe completeness relative to the supplied manifest. It does **not** prove that the manifest itself is complete, that propensities are true, that rewards are missing at random, or that policy value is statistically identified.

## Inputs

- Manifest TSV: \`context_id<TAB>action_id\` for every action the evaluation and logging policies may select in that context.
- Trace TSV: the same context/action pairs with the seven-column VIS-DVPC-027 trace header. The tool only checks the pair universe; VIS-DVPC-027 separately validates propensities, selected-action reward visibility, overlap, and cluster diagnostics.
- Expected SHA-256 digest for each file.

The candidate set per context is hashed over action IDs sorted lexicographically and joined by newline, with a final newline. Therefore, row reordering does not change the candidate-set digest.

## Verdicts

- \`COMPLETE_ACTION_UNIVERSE\`: trace and trusted manifest contain exactly the same unique context/action pairs.
- \`INCOMPLETE_ACTION_UNIVERSE\`: there are missing/extra contexts or action pairs.
- Malformed input, duplicate pairs and digest mismatch are hard errors (non-zero exit from CLI). Do not continue to support/OPE evaluation on either condition.

## Run

\`\`\`sh
rustc --edition=2024 --test tools/vision-dvpc-universe-audit.rs -o /tmp/vis-dvpc-032-tests
/tmp/vis-dvpc-032-tests
rustc --edition=2024 -O tools/vision-dvpc-universe-audit.rs -o /tmp/vis-dvpc-032
/tmp/vis-dvpc-032 --verify MANIFEST.tsv TRACE.tsv EXPECTED_MANIFEST_SHA256 EXPECTED_TRACE_SHA256
\`\`\`

## Known-answer controls

- Exact 2-context / 2-action manifest-to-trace match.
- Missing action pair and extra action pair controls.
- Entire missing context gets a distinct reason.
- Duplicate pair, malformed headers, wrong manifest/trace digest all fail closed.
- Reordered rows preserve the verdict and canonical candidate-set hashes.

No calibration, distribution-shift, risk, or SOTA claim follows from a completeness pass. It is one prerequisite in a layered chain: byte identity -> action-universe completeness -> support/identification -> valid inference -> decision gate.
