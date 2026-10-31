# AR-5B5 semantic-alias and verifier-corruption benchmark v0.1

**Status:** synthetic negative-control harness committed; tests authored but not compiled or run. The fixture is a control instrument, not a natural-language semantic benchmark and not evidence of verifier performance on real research proposals.

## Objective

A direction-search system must distinguish genuinely new mechanisms from syntactically different descriptions of an existing mechanism. False merges suppress useful directions; false splits and false-new admissions waste scarce evaluation budget; false novelty rejections prevent useful work from ever entering the search.

The benchmark deliberately separates policy-visible proposals from evaluator-only truth labels.

File: \`crates/domains/symthaea-architecture-engineering/src/semantic_alias_benchmark.rs\`

## Information separation

### Policy-visible challenge

\`PolicyVisibleAliasChallenge\` contains:
- proposal ID;
- human-readable surface form;
- descriptor SHA-256.

It contains no ground-truth identity or utility.

### Evaluator-only adjudication key

\`AliasAdjudicationKey\` holds:
- proposal-to-truth identity mapping;
- per-truth utility value;
- digest of the exact policy-visible challenge.

The key has private fields and is not serialized into the challenge. The evaluator report publishes only a digest of the adjudication key, not its labels or utilities. The policy under test must receive only the challenge and return decisions.

The included fixture has three latent directions, two syntactic aliases, and utilities in [0,1]. It is deliberately small for unit tests. The benchmark needs larger hidden suites with paraphrase families, structural rewrites, operator aliases, near-neighbour non-aliases and domain-specific adjudication.

## Verifier output and corruption injection

For every proposal, the verifier emits exactly one of:
- \`Admitted { canonical_direction_id }\`;
- \`Duplicate { canonical_direction_id }\`;
- \`Rejected\`;
- \`Inconclusive\`.

A Duplicate must refer to a canonical ID admitted earlier in the trial. This validates the basic registry protocol while still allowing a genuinely new proposal to be falsely collapsed into an existing canonical ID.

\`inject_alias_corruptions\` supports deterministic negative controls:
- false merge: a new semantic identity is incorrectly returned as Duplicate of an existing direction;
- false-new admission: a previously seen semantic identity is admitted under a new canonical ID;
- false novelty rejection: the first proposal for a new semantic identity is rejected;
- inconclusive: a proposal does not receive a determinate verifier result.

The corruption plan is explicit and applied to selected proposal IDs. The resulting report is bound to the challenge, verifier trial, and evaluator-only adjudication-key digest. A deterministic report verifier can recompute all confusion counts and regret from those exact inputs. A confirmatory campaign should additionally freeze a randomized, seeded corruption schedule, rates and selection policy before exposing outcomes.

## Metrics

The report includes:
- pairwise true duplicate, predicted duplicate, true positive, false-merge, false-split and true-negative counts;
- duplicate precision and recall (undefined rather than silently zero when the relevant denominator is zero);
- false-new admissions;
- false novelty rejections;
- admitted, duplicate, rejected and inconclusive decision counts;
- number of canonical slots created and distinct semantic directions discovered;
- downstream discovery regret under a fixed canonical-direction allocation budget;
- challenge, trial, evaluator-only adjudication and final receipt digests.

Pairwise labels are derived from the hidden truth mapping. A false merge occurs when distinct truth identities are assigned the same canonical ID. A false split occurs when the same truth identity maps to different canonical IDs or one member of a duplicate pair is rejected/inconclusive.

### Downstream regret

The fixture assigns each distinct truth identity a utility. The oracle selects the best distinct identities up to the allocation budget. The verifier-controlled system selects canonical slots in first-admission order. Repeated canonical slots that map to the same true identity do not earn utility twice, but still consume a budget slot. This makes the penalty for false splits and false merges explicit.

The current regret is a deterministic synthetic utility calculation, not a statistical estimate of real-world value. Real evaluations should derive utility from independently verified downstream candidate outcomes and use matched budget constraints.

## Tests authored

- clean trial has perfect duplicate precision/recall and zero discovery regret;
- false merge creates pairwise false positives and downstream regret;
- false-new admission creates a false split and wastes a canonical allocation slot;
- false novelty rejection is detected and reduces reachable utility;
- duplicate decisions pointing to unregistered canonical IDs are rejected;
- a substituted policy-visible challenge cannot reuse another challenge's adjudication key;
- deterministic report verification rejects a tampered regret or confusion summary.

No tests have been executed in this environment.

## Next research steps

1. Expand to hundreds/thousands of hidden semantic clusters across multiple operator families, with independently adjudicated paraphrases and hard near-neighbour negatives.
2. Use multiple independent adjudicators and resolve disagreements before freezing ground truth.
3. Add false-merge, false-split, false-new and false-reject rates across preregistered corruption levels.
4. Add seeded randomized corruption schedules and retain the exact schedule digest in each report.
5. Connect canonical direction IDs to EventTape proposals and candidate evaluations; compute regret from measured downstream outcomes rather than synthetic utilities.
6. Run a calibration study: compare verifier confidence against empirical identity correctness, including out-of-distribution descriptor families.
7. Test adversarial proposal attacks: order changes, surface perturbations, token aliases, structural rewrites and deliberately misleading descriptors.

No verifier accuracy or SOTA claim is made by this synthetic fixture.
