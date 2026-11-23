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
