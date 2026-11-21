# Melothaea RMI-Bench Track A — snapshot-isolated offline runner v1

**Date:** 21 November 2026  
**Parent protocol:** [#10088 — causal reciprocal musical interaction benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/10088)  
**Runner:** [#10159 — offline Track A fixture runner and replayable reports](https://github.com/Luminous-Dynamics/symthaea/issues/10159)  
**Paired-state blocker:** [#10155 — isolate policy state across paired counterfactual arms](https://github.com/Luminous-Dynamics/symthaea/issues/10155)  
**Immutable-model identity follow-up:** [#10206](https://github.com/Luminous-Dynamics/symthaea/issues/10206)  
**Implementation:** \`src/rmi_benchmark.rs\`, \`src/rmi_track_a_runner.rs\`, \`src/bin/rmi_track_a.rs\`

## Purpose

Execute the synthetic symbolic Track A protocol offline with counterfactual state isolation, registered baselines, exact raw trial records and independent fresh-process replay. Track A tests whether an agent's symbolic action changes in response to a controlled partner intervention, and whether a controlled partner's next symbolic response changes as a function of the agent's action.

This is a mechanism/evaluator qualification suite. It does not run a production Melothaea policy, render audio, adjudicate musical relevance, or establish human collaboration.

## Counterfactual state integrity

The original evaluator had a serious confound: it invoked one mutable policy instance in this order:

1. control;
2. control replay;
3. treatment;
4. treatment replay.

An input-insensitive policy with an internal counter could emit A/A then B/B and appear treatment-sensitive. This was documented in #10155. Repeating each arm twice does not solve state contamination.

The updated policy contracts require:
- a versioned, serializable mutable episode snapshot;
- a content digest over that exact snapshot;
- independent restores for control, control replay, treatment and treatment replay;
- policy ID/version equality and exact restored-snapshot equality before invocation;
- arm-specific output and post-state digests;
- a reverse-order probe (treatment, then control) that must reproduce the original arm-specific actions and post-state digests.

The partner policy uses the same snapshot/restore, deterministic replay and arm-order requirements. Regressions cover a two-call mutable counter that ignores input, plus an untracked thread-local counter that cannot be recovered from the declared snapshot and must be rejected by the reverse-order pass.

The snapshot contract is a declared purity/state boundary, not proof against every conceivable external side effect. A production adapter must prohibit or account for shared global state, remote hidden state, external mutations and nondeterministic infrastructure.

## Frozen suite

The runner enumerates four intervention dimensions:

- motif: signed semitone-interval tuple;
- rhythm: eight-step onset/rest mask;
- tempo: integer milli-BPM;
- role: one of four symbolic ensemble roles.

The default run uses eight trials per target (32 paired trials). Each paired trial runs nine registered evaluations:

| Evaluation | Purpose |
|---|---|
| Fixed-response agent | Negative control for partner-to-agent sensitivity |
| One-way prompt agent | Negative control for current-input contingency |
| Yoked replay agent | Replays a distinct donor fixture; tests replay confounding |
| Unrelated-feature agent | Responds to a non-manipulated dimension; must not pass target sensitivity |
| Imitation-only agent | Input-sensitive control; copying is not collaboration |
| Synthetic contingent agent | Positive control for the manipulated target only |
| Fixed agent + responsive partner | Negative control for agent-to-partner influence |
| Synthetic agent + fixed partner | Negative control for partner response |
| Synthetic agent + responsive partner | Positive control for the two-direction evaluator |

Every paired fixture uses the same frozen history and source-event identity in both arms. Exactly one declared musical feature changes. Trial seeds are disjoint by target and bounded by the runner contract. Yoked replay binds its donor trial ID and fixture digest; the donor cannot be the active trial itself.

The runner records all raw evaluation results or explicit errors. It does not silently drop a failed case. The manifest records suite/version identity, target and policy inventory, seed schedule, source revision (when supplied), Rust compiler version (when supplied), target OS/architecture and claim ceiling. It also binds the isolated dependency lockfile digest.

## Report and replay

The JSON report contains:
- frozen manifest and manifest digest;
- one record per paired trial;
- exact intervention-pair digest;
- yoked donor identity and digest;
- all nine registered evaluation outcomes/errors;
- separate negative/positive-control counts;
- collateral-change counts;
- agent and partner arm-order verification counts;
- synthetic fixture expectation result;
- domain-local payload SHA-256.

The standalone integration test launches the compiled CLI twice in fresh processes with identical trial count, seed root and output destination, then requires byte-identical JSON reports.

No inferential confidence intervals are attached to these small deterministic synthetic controls. They are known-answer tests, not a representative sample of music or people.

## Run locally

The root workspace currently contains a wildcard member glob that also matches a retired \`crates/core/symthaea\` path with no \`Cargo.toml\`. Cargo loads the workspace before honoring \`-p symthaea-muse\`, so a normal package-scoped command cannot currently start. Do not add a fake member or weaken the repository-wide workspace rule to work around this.

The focused qualification script stages the exact production RMI module, runner, CLI and replay test into a temporary isolated Rust 2024 package with only \`serde\`, \`serde_json\` and \`sha2\`. It first preserves the original source bytes and hashes, runs Rustfmt normalization only on the temporary staging copy, checks the normalized copy, and records both original and formatted source digests. It then generates and hashes the isolated lockfile, embeds the exact repository source revision/compiler/lock identity into the runner manifest, and runs evaluator tests, runner tests, CLI tests, fresh-process replay, warnings-denied Clippy and the fixture suite.

From a clean checkout on the pinned Rust 1.96.0 toolchain:

\`\`\`bash
bash scripts/mel_rmi_track_a_qualification.sh "$(git rev-parse HEAD)"
\`\`\`

The default run is eight trials per target (32 total). A larger run can be requested:

\`\`\`bash
MEL_RMI_TRIALS_PER_TARGET=32 \\
MEL_RMI_SEED_ROOT=42 \\
MEL_RMI_EVIDENCE_DIR=/tmp/mel-rmi-track-a-evidence \\
bash scripts/mel_rmi_track_a_qualification.sh "$(git rev-parse HEAD)"
\`\`\`

The script rejects a dirty checkout or any source SHA that does not equal the exact checked-out \`HEAD\`. It preserves the isolated manifest, generated lockfile, original source copies, rustfmt-normalized test copies, JSON report, and SHA-256 evidence list. Formatting is normalized only in the temporary staging directory; the repository sources are not silently rewritten by qualification.

**Qualification boundary:** this validates the exact RMI sources and runner in an isolated crate. It does not compile all of \`symthaea-muse\` or the full monorepo workspace, because the unrelated missing workspace member currently prevents Cargo from loading the main package. The actual package module/bin registration remains an integration seam until that workspace issue is repaired and full-package compilation succeeds.

## Remaining limitation: immutable policy identity

#10155's mutable episode-state isolation is implemented in the updated oracle. #10206 remains a distinct requirement before a production or external model adapter is evaluated.

The current RMI trait records policy ID and human-readable version, but these labels are not by themselves proof of exact immutable model identity. A production adapter must bind, separately from mutable episode state:
- checkpoint/weights or immutable executable artifact digest;
- adapter and configuration digest;
- tokenizer, grammar, prompt template or other static assets that affect behavior;
- runtime/build identity and device/backend mode;
- deterministic/stochastic source and seed authority;
- independently resolvable evidence for each digest.

Missing model artifacts must make replay unavailable; no silent replacement by a newer model under the same label is permitted. The synthetic controls in Track A have no external model weights, so their current qualification does not resolve #10206.

## Claim ceiling

A successful exact-head run would establish only that:
1. registered synthetic interventions are valid and isolated to one declared feature;
2. the evaluator forks each arm from the same declared mutable state and catches registered state/order confounds;
3. known fixed, one-way, yoked and unrelated-feature negatives do not pass target sensitivity;
4. imitation can be input-sensitive without being labelled collaboration;
5. synthetic target-specific positive controls pass without collateral feature changes;
6. the reciprocal synthetic partner controls are distinguished;
7. the frozen report replays byte-identically in fresh processes.

It would **not** establish that:
- a live Melothaea policy is reciprocally adaptive;
- response changes are musically relevant, valuable, original or preferred;
- a human musician is causally influenced by Muse;
- the system performs reliably on live audio, MIDI hardware or acoustic instruments;
- a generated work is better than Suno, Lyria, Mureka, JAM_BOT or another system;
- model/checkpoint identity has been independently qualified under #10206;
- any consent, authorship, publication or export authority exists;
- the system has consciousness or subjective experience.

## Next sequence

1. Complete exact-head qualification of the isolated state-aware evaluator and report.
2. Implement the immutable model/checkpoint/config/adapter identity receipt from #10206 before adding a live or proprietary policy.
3. Define a production Muse adapter over the actual observation/action boundaries; do not synthesize missing music-theory state from the small RMI feature tuple.
4. Bind the adapter's proposals to source Score/event identity, preservation contracts and candidate-specific receipts.
5. Add blinded musician relevance annotations and held-out material; keep musical quality separate from raw sensitivity.
6. Combine with #10307 streaming WAIT/RESPOND/CONTINUE/YIELD and #10390 timing/recovery/grounded outcomes.
7. Only then run randomized, ethically approved human reciprocal interaction under #10088 and agency/consent checks under #11319.

No synthetic result should be represented as a Melothaea result.
