# Melothaea RMI-Bench Track A — offline runner v1

**Date:** 21 November 2026  
**Parent protocol:** [#10088 — causal reciprocal musical interaction benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/10088)  
**Runner gap:** [#10159 — offline Track A fixture runner and replayable reports](https://github.com/Luminous-Dynamics/symthaea/issues/10159)  
**Implementation:** \`crates/domains/symthaea-muse/src/rmi_benchmark.rs\` and \`src/bin/rmi_bench_track_a.rs\`

## Purpose

Make the existing symbolic RMI intervention oracle executable as a small, reproducible, offline Rust benchmark. Track A measures whether a policy's next symbolic action changes when one controlled musical feature in the partner's current observation changes, and whether a controlled partner policy changes its next symbolic action in response to the agent's intervention.

This is a benchmark mechanism check. It does not run the production Melothaea planner, use audio, prove musical relevance, or measure human collaboration.

## Frozen trial design

The runner enumerates:

- four intervention dimensions: motif intervals, an eight-step onset mask, tempo in milli-BPM, and symbolic ensemble role;
- five agent policies: fixed response, one-way prompt, yoked replay, imitation-only, and a synthetic target-contingent positive control;
- two partner policies: fixed partner and a synthetic target-responsive positive control;
- a deterministic seed schedule \`0..N\`, where \`N\` is selected by \`--seed-count\` (default 16, maximum 1,024).

Each pair uses the same pre-intervention history and source-event identity in both arms. Exactly one declared feature changes between control and treatment. The input validator rejects changed source identity, a no-op target intervention, out-of-range values, stale/invalid history digests, and uncontrolled changes to another feature.

For each agent/partner combination the report stores:
- the complete intervention pair, not merely its label;
- the paired agent evaluation envelope and digest;
- the reciprocal evaluation envelope and digest;
- the agent and partner policy identities;
- a unique receipt ID, pair digest, seed and target feature.

The output also includes the exact source commit supplied to the runner and SHA-256 of the workspace \`Cargo.lock\`. The commit is operator-supplied; the runner does not independently query Git to authenticate it. Invoke it using \`git rev-parse HEAD\` from the checkout being evaluated.

## Baselines and falsifiers

| Policy | Expected property | What a positive result means |
|---|---|---|
| Fixed response | No partner-to-agent sensitivity | Negative control |
| One-way prompt | No current-input sensitivity | Negative control |
| Yoked replay | Same replay in both arms | Negative control |
| Imitation-only | Surface input sensitivity may be present | Sensitivity is not collaboration quality |
| Synthetic contingent | Target sensitivity, zero collateral feature changes | Positive control for the evaluator |
| Fixed partner | No agent-to-partner sensitivity | Negative control |
| Synthetic responsive partner | Target sensitivity to changed agent actions | Positive control for the reciprocal evaluator |

The runner reports both directions independently. It does not combine them into a scalar. It records exact counts and collateral-change totals; it intentionally does not attach inferential confidence intervals to a small deterministic synthetic suite.

A failed intervention, invalid policy response, or inconsistent reciprocal evaluation aborts the run. Trial receipts are not silently dropped. The expected denominator is:

\`seed_count × 4 targets × 5 agent policies × 2 partner policies\`

At the default 16 seeds, this is 640 receipts.

## Run locally without network or external models

From the repository root, after a Rust toolchain is installed:

\`\`\`bash
cargo fmt --manifest-path Cargo.toml --package symthaea-muse -- --check
cargo test -p symthaea-muse --bin rmi_bench_track_a
cargo run -p symthaea-muse --bin rmi_bench_track_a -- \\
  --source-commit "$(git rev-parse HEAD)" \\
  --seed-count 16 \\
  --output /tmp/melothaea-rmi-track-a.json \\
  --self-check
\`\`\`

\`--self-check\` invokes the same compiled executable in a fresh process with the same seed schedule and source commit, then compares the independently produced report digest. The report is domain-separated and SHA-256 sealed over its canonical deterministic payload. BTreeMap aggregation and fixed field/array order keep serialization stable for a fixed executable, source identity, lockfile and manifest.

The CLI requires \`--source-commit\`, bounds the seed count, and rejects malformed commit identifiers. It reads the workspace \`Cargo.lock\` relative to \`CARGO_MANIFEST_DIR\`; a missing lockfile is an error rather than a silently omitted dependency identity.

## Interpretation and claim ceiling

A successful run would establish only that:
1. the frozen synthetic intervention fixtures satisfy the pair contract;
2. the evaluator distinguishes the registered synthetic positive and negative controls;
3. paired records and aggregation replay deterministically in a fresh process.

It would **not** establish that:
- a live Melothaea policy is reciprocally adaptive;
- its responses are musically appropriate, novel, complementary or preferred;
- a musician changes their behavior because of Muse;
- the system performs reliably on live audio, MIDI devices or acoustic instruments;
- a generated piece is artistically superior to Suno, Lyria, Mureka, JAM_BOT or another system;
- a person has granted consent, authorship, or export authority;
- the system has consciousness or subjective experience.

## Next research increments

1. **Production adapter:** bind the real Melothaea observation/action boundary to the RMI traits without letting the evaluator inject the intervention label or synthetic target.
2. **Musical relevance:** score response relevance and preservation with source-bound score/event relations and blinded musician annotations. Keep this distinct from input sensitivity.
3. **Streaming track:** combine RMI with #10307's WAIT / RESPOND / CONTINUE / YIELD semantics, chunk coverage, no-lookahead and event-clock constraints.
4. **Recovery track:** combine with #10390's interruption, veto, tempo/meter-shift, renderer-failure and output-underrun interventions.
5. **Human reciprocal study:** preregister randomized/counterbalanced performer sessions and participant/ensemble-clustered statistics. Simulator influence must never be represented as human causal evidence.
6. **Agency track:** connect #11319's proposal/preview/approval/reversal states and explicit protection of user-authored material.

Do not report a synthetic positive-control result as a Melothaea result. Report each cohort (synthetic symbolic, rendered audio, MIDI device, acoustic performance and human study) separately with its own evidence ceiling.
