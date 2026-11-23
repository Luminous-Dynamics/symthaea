# ProofPlan Local-to-Global Consistency Protocol v1

**Parent:** [#13326 — MATH-PROOF-PLAN-001](https://github.com/Luminous-Dynamics/symthaea/issues/13326)  
**Implementation contract:** [#13338 — MATH-PROOF-PLAN-SCHEMA-001](https://github.com/Luminous-Dynamics/symthaea/issues/13338)  
**Counterfactual semantics:** [#13309 — MATH-COUNTERFACTUAL-001](https://github.com/Luminous-Dynamics/symthaea/issues/13309)  
**Programme:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)

**Status:** research protocol / prospective benchmark design. No Symthaea local-to-global consistency result has been collected.

## 1. Research question

Can a model correctly reject an invalid inference when evaluated in isolation, yet later reuse that same inference when constructing or evaluating a complete proof plan under contextual pressure?

This is a local-to-global consistency question. It is related to ProofGrid's epistemic-stability discussion, but the measure proposed here is distinct and must not be labelled or substituted for ProofGrid's Epistemic Stability Index.

The experiment targets a concrete failure mode: local correctness judgements do not reliably constrain global proof construction.

## 2. Critical scope control

The local and global trial must bind to the same:
- theorem identity and source revision;
- elaborated target proposition identity;
- exact hypothesis/context set available to the proof;
- domain, binder, universe and variable scope;
- warrant/inference input and output proposition identities.

The global prompt may change presentation, order, verbosity, surrounding valid steps, distractor material, time pressure or instruction framing. It must **not** add a new mathematical assumption that makes the previously invalid warrant valid.

If the mathematical context changes, that is a separate scope-mutation control and is not part of the primary local-to-global denominator.

This prevents a false inconsistency label when the local inference is invalid under one hypothesis set but valid under a stronger global context.

## 3. Unit of evaluation

The primary unit is a paired tuple:

`(theorem_family, task_id, warrant_id, model_snapshot, seed, context_condition)`

A trial contains two stages with a shared immutable proposition-and-hypothesis contract.

### Stage L — local inference assessment

Present the model with:
- exact premise proposition(s);
- exact conclusion proposition;
- explicit hypothesis set;
- proposed inference/warrant;
- theorem/source identity;
- no global proof-plan narrative.

Require a structured response:
- `accept`;
- `reject`;
- `unknown`;
- `ill_formed`.

Also require a short reason and the proposition/assumption identities the model relied upon. Preserve raw text, structured output, token budget, latency and tool calls.

The local label is not its own gold authority. Ground truth must be established independently by:
- Lean type-checking or kernel-checked proof when the warrant is formal;
- checked finite countermodel for finite propositions where applicable;
- a sound decision procedure with a validated encoding;
- independent mathematical annotation for cases not decidable by the selected formal mechanism.

### Stage G — global proof-plan reconstruction

Present the same theorem and exact hypothesis-set digest, plus:
- the complete proof-plan context;
- a set of valid and invalid candidate actions/warrants;
- controlled distractors and surrounding valid actions;
- the same target proposition and permitted tool budget.

Ask for a complete `ProofPlanV1` or a structured review of the candidate plan. Record whether the target warrant is:
- explicitly used as a justification;
- used implicitly by a derived step;
- cited but rejected;
- omitted;
- marked unknown;
- replaced by a valid alternative.

A global output counts as reuse only if the invalid warrant materially supports a claimed transition, subgoal discharge, theorem derivation or terminal conclusion. Mere textual mention does not count.

## 4. Primary metric

Define the eligible local-rejection set:

`E = { paired trials where the warrant is independently gold-labelled invalid and Stage L returns reject }`

Define global reuse set:

`R = { trials in E where Stage G materially reuses the same invalid warrant }`

The primary metric is:

`Context-Induced Reuse Rate (CIRR) = |R| / |E|`

Interpretation:
- lower is better;
- CIRR is undefined when `|E| = 0`;
- always report numerator and denominator;
- do not report zero when no eligible local rejections exist;
- a low CIRR is not proof of general mathematical understanding.

A companion measure may be reported:

`Local-to-Global Consistency = 1 - CIRR`

This is only a companion complement, not a replacement for the primary count. Report it only when CIRR is defined.

## 5. Essential secondary metrics

Report separately:
- local invalid-warrant rejection precision and recall;
- local false acceptance / false rejection;
- global invalid-warrant reuse rate over all independently invalid warrants;
- global omission and abstention rates;
- global valid-warrant omission / substitution rate;
- exact first-invalid-step localization accuracy;
- distance between predicted and gold first invalid step;
- theorem statement and hypothesis-set identity preservation;
- proof-plan side-condition coverage;
- alternative valid proof-route recognition;
- unsupported claim and fabricated dependency rate;
- confidence calibration and Brier score;
- wall time, tokens, model/tool calls and reviewer minutes.

Do not collapse these into a single score that could hide false acceptance.

## 6. Controlled conditions

For each theorem/warrant family, use paired variants:

1. **Neutral global context:** concise complete plan with no distractors.
2. **Position pressure:** place the invalid warrant among many valid steps or late in the proof.
3. **Authority cue:** a surrounding explanation claims that a named helper supports the warrant, while the formal evidence does not.
4. **Lexical lure:** replace the warrant with a superficially similar but propositionally mismatched statement.
5. **Narrative pressure:** make the warrant appear to be the natural next step in a plausible proof plan.
6. **Instruction pressure:** explicitly instruct the model to complete the plan efficiently, without changing the theorem or assumptions.
7. **Order control:** shuffle valid and invalid candidate ordering while preserving identities and content.
8. **Scope-change control:** explicitly add a new assumption that makes a previously invalid warrant valid; this is excluded from the primary CIRR denominator and checks whether the system notices the changed scope.
9. **Valid-warrant control:** evaluate valid inferences under the same contextual conditions to detect over-rejection or generic abstention.

Randomization and seed must be frozen. Do not use the same response as both treatment input and evaluator truth.

## 7. Baselines and interventions

At matched source, model, context and compute budgets:

- local judgment only;
- generic self-critique followed by global plan generation;
- two-pass independent local/global model instances;
- graph-grounded proof plan without explicit local rejection memory;
- graph-grounded plan with an explicit, immutable local verdict/evidence envelope;
- independent verifier result injected as a typed warrant receipt;
- full Symthaea plan pipeline.

The central intervention is whether the global planner receives the **same signed/immutable local verdict plus evidence envelope**, or only a natural-language summary of the local judgement.

This tests whether a structured evidence constraint prevents context-induced regression better than conversational memory alone.

## 8. Authority and evaluator separation

- Lean kernel checks formal proof terms and exact target identities.
- A checked finite model supports only the specific finite proposition and witness it evaluates.
- A validated decision procedure supports only claims within its soundness assumptions.
- Independent mathematicians adjudicate semantic plan roles and non-mechanized interpretations.
- LLM judges may assist triage, but they are not the sole authority for labels or theorem correctness.

Store evaluator truth and witness artifacts separately from candidate-visible inputs. The candidate must not receive hidden gold labels.

## 9. Statistical design

- Pair Stage L and Stage G by theorem/task/warrant/model snapshot/seed.
- Cluster resampling by theorem family or independent task/world blocks; do not count repeated prompts for the same warrant as independent theorem instances.
- Report cluster-bootstrap or other preregistered confidence intervals for CIRR and secondary metrics.
- Stratify by context condition and theorem family.
- Report all denominators, missing/invalid trials, timeouts, abstentions and exclusions.
- Control multiplicity for confirmatory context families; label per-warrant analyses exploratory unless preregistered.
- Keep model seeds nested within theorem/task clusters if the design uses multiple seeds.
- Freeze prompts, task splits, model snapshots, budgets, context conditions and scoring rubrics before confirmatory runs.

## 10. Prospective target ladder

These targets are not current Symthaea results:

- T0: 100% exact theorem/hypothesis identity binding across paired stages.
- T1: zero global acceptance of a warrant independently established as invalid without a checked counterevidence event or explicit unknown.
- T2: CIRR at most 5% on held-out controlled invalid-warrant cases, with a confidence interval and nonzero eligible denominator.
- T3: no more than a 2 percentage-point increase in false rejection of valid warrants relative to the strongest matched baseline.
- T4: at least a 10 percentage-point reduction in CIRR versus the strongest matched baseline under context-pressure conditions.
- T5: preserve the improvement on held-out theorem families and model snapshots.
- T6: publish quality–compute–latency–review-cost frontiers.

First freeze the corpus and measure baselines; do not tune on the confirmatory holdout.

## 11. Claim ceiling

- Local rejection does not prove global correctness.
- Global reuse does not prove the original theorem false.
- Contextual consistency is not equivalent to mathematical understanding.
- A failed proof route is not a counterexample.
- ProofGrid's Epistemic Stability Index is not CIRR.
- A low CIRR on a small synthetic set is not general mathematical reliability.
- A proof plan is not a kernel-checked proof.

The intended claim, if supported by independent results, is narrow: a structured evidence envelope can reduce context-induced reuse of previously rejected invalid warrants in proof-plan reconstruction, while preserving correct acceptance of valid warrants.

## 13. Executable LocalGlobalPairV1 receipt and CIRR semantics

The first executable receipt contract is specified by:

- docs/research/LOCAL_GLOBAL_PAIR_V1.schema.json
- docs/research/LOCAL_GLOBAL_PAIR_SYNTHETIC_V1.json
- scripts/research/proof-plan-contract/src/local_global.rs
- scripts/research/proof-plan-contract/src/bin/local-global.rs

### Candidate/evaluator separation

LocalGlobalPairV1 has two top-level partitions:

1. candidate: source/theorem/warrant scope bindings, model and decoding identity, local verdict, global plan digest, candidate-side warrant mentions, prompt/input/output digests, artifact references and resource budgets.
2. evaluator_only: eligibility, independent warrant validity, local-verdict correctness, adjudicated global disposition, adjudication authority and supporting references.

Only the serialized candidate projection is suitable for a model prompt or candidate handoff. The Rust candidate_view() method returns that projection without evaluator-only fields. The combined receipt is a post-hoc record; never provide its evaluator_only partition to the candidate.

### Scope equality is a hard primary-denominator gate

Both stages independently repeat the full scope binding: source repository/revision/path/declaration, theorem ID, root-goal digest, hypothesis-set digest, premise-set digest, conclusion digest, warrant ID and warrant digest. The primary CIRR denominator excludes any pair whose local and global bindings differ. An evaluator receipt that labels a scope-mismatched pair eligible is rejected.

Changes to wording, ordering, distractors, narrative framing and plan length may be allowed by a task protocol, but any change to source, proposition, assumptions, premises, conclusion or focal warrant must create a separate control stratum rather than silently remain in the primary denominator.

### Three-state global adjudication; no silent imputation

CIRR continues to mean:

CIRR = definite material reuse of a locally rejected, independently invalid warrant / eligible locally rejected, independently invalid warrants.

Global adjudication has five outcomes: used_directly, used_implicitly, cited_rejected, omitted, and unknown. Only direct and implicit use count in the numerator. Citing a rejected warrant as rejected, or omitting it, is not material reuse.

An unknown global outcome remains in the denominator but is also reported separately. The tool returns a lower bound (unknown treated as no reuse), an upper bound (unknown treated as reuse), and a point estimate only when there are no unresolved outcomes. A zero denominator produces an undefined point estimate, never zero.

### Executable checks and claim ceiling

The Rust validator rejects unknown fields, malformed SHA-256 digests, execution budget overruns, duplicate warrant-mention IDs, unsupported eligible labels, and eligible scope mismatches. Unit tests cover candidate/evaluator separation, finite Boolean countermodel evaluation and corruption controls, scope mutations, denominator-zero behavior, unknown-outcome bounds, direct reuse, cited-rejected non-reuse and budget violations.

The synthetic fixture now carries a recursively typed propositional formula and a finite Boolean valuation. It checks the converse failure: from the premise `p → q`, the conclusion `q → p` does not follow; the valuation `p=false, q=true` makes the premise true and the conclusion false. The Rust validator evaluates every premise and the conclusion, rejects duplicate or missing atom assignments, and does not trust a witness string. This is a finite semantic check for propositional formulas only, not Lean-kernel verification or a general mathematical proof. The paired candidate outputs and resulting 1/1 CIRR arithmetic remain synthetic contract fixtures, not an LLM evaluation or observed benchmark result.

Run locally:

    cargo test --locked --manifest-path scripts/research/proof-plan-contract/Cargo.toml
    cargo run --locked --manifest-path scripts/research/proof-plan-contract/Cargo.toml --bin local-global -- docs/research/LOCAL_GLOBAL_PAIR_SYNTHETIC_V1.json

A successful run means structural receipt checks and deterministic metric arithmetic passed. It does not mean Lean replay, expert adjudication, model testing or a human study has occurred.
## 15. Canonical scope-preimage integrity (partial v1 implementation)

The local/global receipt now carries literal theorem-scope preimages alongside their SHA-256 digests: root_goal_text, hypotheses, premises, conclusion_text and warrant_text.

Rust recomputes each digest using SHA-256 over compact serde_json serialization, prefixed by the domain separator symthaea.local_global_pair.v1:{domain} followed by a NUL byte. Distinct domains are used for root goal, hypotheses, premises, conclusion and warrant. A changed proposition, assumption, premise, conclusion or warrant is rejected even when the supplied digest still has valid hexadecimal shape.

The hypothesis and premise arrays are treated as sorted, duplicate-free sets. Scalar mathematical text is exact UTF-8 after JSON string serialization. No implicit whitespace, Unicode or notation normalization is performed; normalization rules must be a separately versioned protocol decision.

Remaining integrity gap: model decoding configuration and execution prompt/input/output digests are still only syntax-checked. Their canonical preimages and artifact contents are not yet included/resolved. A digest shape is not a trusted execution attestation. Artifact URI authorization and digest verification remain follow-up work.

The SHA-256 dependency is introduced in this commit. The focused workflow temporarily regenerates Cargo.lock and uploads it as an artifact. The generated lock must be committed in a follow-up before restoring --locked everywhere.
