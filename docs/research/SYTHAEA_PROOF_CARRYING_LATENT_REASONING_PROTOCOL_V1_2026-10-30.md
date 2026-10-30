# Proof-Carrying Latent Reasoning: protocol and benchmark V1

**Date:** 2026-10-30  
**Status:** research protocol / implementation proposal. No new runtime behavior, benchmark result, or SOTA claim is asserted here.

## 1. Thesis

The next step is not to make Symthaea emit a longer hidden or visible chain of thought. It is to make reasoning a **budgeted sequence of state transitions that must discharge explicit proof obligations**.

Use two coupled workspaces:

1. **Latent workspace:** HDC/CfC state proposes the next useful operation, retrieves context, maintains alternatives, and estimates which unresolved question is worth more computation.
2. **Proof workspace:** a typed, append-only graph records claims, premises, rules, evidence references, contradictions, tool results, and independent verifier receipts.

The latent workspace may propose. It cannot, by itself, promote a proposition to verified fact, authorize a tool action, or certify its own correctness. The proof workspace controls what may be stated as established. A human-readable explanation is compiled from that verified graph; it is not presented as a transcript of hidden cognition.

This changes the optimization target from *produce plausible intermediate text* to *resolve the smallest set of obligations needed for a correct, supported decision*.

## 2. What current research changes about the plan

The 2026 comparison set is stronger than a binary comparison between Transformer CoT and Symthaea's native generator.

| Approach | Reported contribution | Implication for Symthaea |
|---|---|---|
| Explicit CoT | Serial textual steps can improve reasoning, but a plausible explanation is not necessarily faithful to the causal computation | Keep it as a strong baseline and a possible teacher, not as ground truth |
| LOTUS / looped Transformers | Supervising recurrent latent blocks with gold CoT-step tokens is reported to bridge the latent/explicit gap at 3B in tested settings, with 2.5–6.9× lower thought-phase latency | Latent state needs task-relevant supervision; recurrence alone is not a training objective |
| Latent Recurrent Thoughts (LRT) | A task proposer plus a recurrent latent reasoner and frozen LLM decoder outperforms non-thinking CoT prompting on several reported tasks under a matched decoder, prompt, data and budget | This is a close architectural comparator; the same-decoder experiment is mandatory |
| Think-at-Hard | Always iterating can overthink: its selective iteration policy reports 3.8–4.4% gains over always-iterate baselines while skipping iterations on 93% of tokens in its evaluated settings | Do not run CfC refinement on every token or thought step; gate compute on demonstrated need |
| LongCoT | 2,500 expert problems test dependency management across long reasoning horizons; paper-release best results were below 10% | Long-range state and dependency control remain useful stress tests, but baseline/version must be reproduced |
| GRACE | Human-annotated step-level faithfulness separates deductive errors from grounding errors | Score every support step, not only the final answer |

Sources:
- Fan et al., *Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers* (LOTUS), June 2026: https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/
- Chen & Fu, *Latent Recurrent Thoughts*, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.578/
- Fu et al., *Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning*, ICML 2026: https://proceedings.mlr.press/v306/fu26t.html
- Motwani et al., *LongCoT*, ICML 2026: https://proceedings.mlr.press/v306/motwani26a.html
- Pham et al., *GRACE: Step-Level Benchmark for Faithful Reasoning over Context*, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1582/

These are paper-reported results on their own tasks and settings. They are not head-to-head results against Symthaea. LongCoT's published below-10% figures describe results at that paper's release, not a verified October 2026 leaderboard.

## 3. A suitable substrate already exists

The first pilot should use the existing zero-dependency *symthaea-legal-reasoning* kernel as a controlled proof substrate, not create a second proof engine.

Its documented contract already includes:
- recursive typed grounded inference and a ProofGraph containing initial assertion provenance, undefeated rule applications, premise-to-conclusion edges, and explicit-exception guards;
- transitive proof slicing for a queried conclusion;
- FactAssertion classifications for observed, stipulated, and assumed facts, with source-provision and asserting-authority fields;
- explicit support/opposition and conflict outcomes rather than silently choosing one side;
- temporal rule-pack selection using decision time and event time;
- canonical evidence manifests binding the result to the crate version, semantic profile, rule pack, and query;
- fail-explicitly behavior for non-stratified theories, oscillation, temporal overlap, and configured limits.

The kernel also explicitly defines its limits. It does not interpret natural language, authenticate sources, establish evidence sufficiency, choose jurisdiction-specific source hierarchies, or prove that a supplied formalization is legally correct. Its canonical bytes are suitable for external hashing/signing, but are not themselves a cryptographic signature or authentication. Its local why-not explanation is not a minimal counterfactual proof. Those boundaries must remain visible in the experiment.

This is an excellent pilot because the checker is deterministic and independent of the candidate planner. It lets us first answer: *can the planner construct and maintain correct proof obligations and dependencies when the semantics are known?* Only after that should a natural-language front end be added.

## 4. Contract between the latent and proof workspaces

Treat the following as logical record types; this document does not claim they are already implemented as Rust types.

| Record | Required content | Invariant |
|---|---|---|
| Claim | Stable ID, proposition, polarity, status, scope/time | Unknown, assumed, supported, refuted, conflicted, and verified are distinct states |
| Evidence reference | Source ID, source version/time, locator, content digest, authority/scope metadata | A digest establishes byte identity, not source authenticity or truth |
| Rule application | Stable rule ID/version, ordered or canonical premise IDs, conclusion ID, exception guards | Every derived claim has a complete, reproducible dependency set |
| Computation event | Input-state digest, selected operation, consumed IDs, produced IDs, cost, outcome | Events are append-only; a failed or timed-out operation cannot silently produce support |
| Verifier receipt | Verifier identity/version, input digest, result, failure details | The verifier must be independent of the proposer for the claim being checked |
| Decision | Answer or abstention, supported claim IDs, policy decision, budget spent | The answer cannot cite a claim that lacks the required accepted support |

### Proof-graph invariants

1. Every verified derived claim is reachable from accepted premises or declared formal axioms.
2. Every non-axiom derivation carries a rule identity and version.
3. Cycles are rejected unless the selected formal semantics explicitly supports and checks them.
4. Alternative supports are represented as alternatives, not flattened into one conjunction.
5. Opposing support remains visible as conflict; a scalar confidence must not erase it.
6. Removing or invalidating a premise invalidates only conclusions that depend on it, including transitive descendants.
7. Adding irrelevant evidence or permuting independent input order must not change unrelated conclusions.
8. A claim without adequate support remains a hypothesis or triggers abstention; it is never upgraded by a high latent activation, psi, or caller-provided confidence value.

### A critical distinction

There are four separate questions:

- **Answer correctness:** is the final answer correct?
- **Derivation validity:** do the rule applications follow from their stated premises?
- **Evidence grounding:** are the premises supported by admissible, correctly scoped sources?
- **Faithfulness:** does an explanation accurately describe the causal computation that produced the answer?

A formal checker can establish derivation validity relative to a formalization. It cannot by itself establish evidence grounding or latent-computation faithfulness. Do not report one as a proxy for the others.

## 5. Adaptive reasoning: make every extra step earn its cost

The current research frontier warns against unconditional recurrence. The controller should choose a refinement step only when a measurable unresolved obligation justifies it.

For each open obligation, record:
- consequence severity if wrong;
- present support state (missing, weak, disputed, contradicted, stale);
- expected probability that another operation will resolve it;
- expected reduction in answer risk or uncertainty;
- estimated latency, energy, token, and tool cost.

Prioritize by **expected reduction in decision risk per unit cost**, not by raw latent activation or an uncalibrated scalar. The estimate must be calibrated on held-out decisions. Start with deterministic rules for obvious hard cases (missing proof edge, contradiction, expired source, failed verifier); only learn a scheduler after collecting trustworthy outcomes.

Suggested trigger classes:
- a required premise is missing;
- two sources or rules conflict;
- a premise is outside its temporal or authority scope;
- an independent checker rejects a derivation;
- a high-impact conclusion has only assumed or weak support;
- a counterexample or mutation invalidates a dependency;
- the remaining expected value of computation is positive under the fixed budget.

Stop when required obligations are discharged, the answer is sufficiently supported under a calibrated risk threshold, or the budget is exhausted. The last case should return an explicit partial answer or abstention, not an invented completion.

## 6. First benchmark: Proof-Graph Integrity and Counterfactual Maintenance (PGI-CM)

This is the next custom benchmark to author. It measures whether a system can construct, verify, and maintain a proof graph under controlled changes. It is deliberately narrower than general intelligence and can be scored without an LLM judge.

### Task families

Generate typed fact bases, rule packs, query propositions, and expected proof graphs with an independent reference evaluator. Hold out entire rule-template families, not merely random instances.

| Family | Perturbation or difficulty | Required behavior |
|---|---|---|
| Dependency depth | Chains of 2, 4, 8, 16, 32 derivation edges | Preserve exact dependency order and derive only supported conclusions |
| Branching | Multiple premises and multiple independent support paths | Preserve all support paths; do not collapse alternatives |
| Exceptions and defeaters | Explicit exceptions, strict/defeasible conflict, incomparable support | Apply the selected semantics and expose conflict |
| Temporal revisions | Change event date or decision date across rule validity intervals | Select the governing revision or return explicit absence/overlap |
| Premise retraction | Remove or invalidate one leaf source | Retract every dependent conclusion; preserve unrelated claims |
| Counterfactual mutation | Replace one premise with its opposite or alter a rule version | Recompute affected descendants and preserve unaffected branches |
| Distractor growth | Add 10× and 100× irrelevant facts/rules | Query results and proof slices remain unchanged |
| Order/duplication | Permute facts/rules and add duplicate rules/evidence | Canonical result remains invariant |
| Open-world uncertainty | Remove a necessary premise or provide insufficient provenance | Return unknown/abstain; do not infer the opposite |
| Contradictory evidence | Add both positive and negative support | Preserve the conflict; do not average it away |

Each task must include a deterministic oracle and a machine-checkable expected result. The generator must reject tasks where the answer can be recovered from shallow shortcuts or where the supposed counterfactual change does not affect the expected answer.

### Baselines

For the typed-kernel track:
1. direct kernel query;
2. query plus local explanation;
3. query plus proof-graph slice;
4. deliberately shallow current-state / one-hop baseline;
5. candidate planner plus independent kernel verifier.

For the language-mediated track, freeze one decoder and compare:
1. direct answer;
2. explicit textual CoT;
3. explicit CoT plus the same independent verifier;
4. structured proof planner without latent refinement;
5. latent refinement without proof workspace;
6. latent refinement plus proof workspace;
7. the best matched same-decoder latent baseline that can be reproduced (including LRT/LOTUS where code, weights, and licenses permit).

All language-mediated arms must receive the same facts, rules, decoder, tokenizer, context, answer format, tool permissions, and total inference budget. Do not compare a pretrained LLM to the Genesis-seeded native Broca generator and label the result a substrate-only comparison.

### Metrics (report independently; no single blended score)

- Final answer accuracy and abstention coverage.
- Proof precision, recall, and F1 at node and edge level.
- Derivation soundness: fraction of emitted derivations accepted by the independent checker.
- Support completeness: fraction of non-axiom conclusion paths that terminate in the required accepted premises.
- Unsupported-claim rate, including answers that happen to be right for the wrong reason.
- Dependency-retraction recall: affected descendants correctly invalidated after premise mutation.
- Unaffected-stability rate: unrelated conclusions preserved after mutation.
- Counterfactual locality: fraction of changed outputs confined to the expected dependency closure.
- Temporal-scope accuracy and conflict-detection recall.
- GRACE-style grounding and inference error rates for natural-language explanations.
- Calibration: Brier score, expected calibration error, and risk-versus-coverage.
- Compute: wall-clock p50/p95, decoder tokens, recurrent iterations, tool calls, memory, accelerator time, and energy where available.
- Overthinking rate: fraction of previously correct supported claims made incorrect by an additional refinement step.

### Initial gates to preregister before evaluation

These are proposed gates, not achieved measurements. They must be fixed in the benchmark manifest before seeing candidate-system results.

| Gate | Initial target |
|---|---:|
| Machine-checkable derivation acceptance | 100% of proofs labelled verified |
| Unsupported conclusion escapes on synthetic closed-world cases | 0 |
| Unauthorized evidence/tool actions | 0 |
| Premise-retraction recall | ≥ 99% |
| Unaffected conclusion stability | ≥ 99% |
| Proof-edge F1 on held-out rule templates | ≥ 0.95 |
| Same-decoder answer accuracy vs explicit CoT | within 2 absolute percentage points |
| Same-decoder inference cost | ≤ 70% of explicit-CoT cost at matched accuracy, as a stretch gate |
| Explanation grounding | ≥ 90% supported steps on the held-out human-annotated track |

Do not relax failed gates after inspecting results. Report confidence intervals and per-family failures. Bootstrap by held-out rule-template family (not by individual generated instance) to avoid treating near-duplicates as independent evidence. Use at least eight independently generated seeds for stochastic planners, and publish deterministic oracle results separately.

The cost-parity target is deliberately a stretch target. If it fails, publish the accuracy/cost Pareto curve rather than changing the budget after the fact. The same-decoder target applies only to the language-mediated experiment, not to native Broca generation.

## 7. LongCoT and GRACE: use them correctly

- **LongCoT:** first reproduce a versioned baseline with the official harness and deterministic answer extraction. Report accuracy by domain and dependency horizon, token budget, latency, and failure mode. Then evaluate whether proof-graph maintenance prevents state drift and dependency loss. Do not claim a LongCoT win from a small custom synthetic graph benchmark.
- **GRACE:** evaluate the natural-language explanation compiled from the proof graph. Annotate whether each sentence is grounded in a premise, a valid inference, unsupported, or contradictory. Separate grounding errors from inference errors. A correct final answer with an unsupported explanation still fails the explanation track.
- **Formal proof suites:** where the task is theorem proving, require acceptance by an independent kernel (for example, Lean) rather than model-graded plausibility. Keep theorem proving separate from source-grounded empirical reasoning.

## 8. Implementation sequence

1. **Freeze this protocol and task manifest.** Version the rule templates, generators, oracle, splits, and initial gates. Record a cryptographic digest of every dataset artifact.
2. **Build the typed PGI-CM harness around symthaea-legal-reasoning.** Reuse its existing proof graph, fact provenance, temporal profiles, and canonical manifests. Do not implement a duplicate inference engine.
3. **Add adversarial mutation tests.** Start with premise deletion/replacement, unrelated-evidence insertion, rule/fact permutation, duplicate rules, temporal boundary changes, and explicit conflicts. Assert exact descendant invalidation and unrelated stability.
4. **Add a thin planner interface.** The planner proposes a query, next obligation, or evidence request. The existing kernel independently decides whether a derivation is valid. No free-form model output can mark a proof verified.
5. **Run deterministic kernel and shallow baselines first.** Only after the harness passes its own mutation controls should LLM and latent planners be evaluated.
6. **Run the same-decoder comparison.** Keep the answer decoder frozen across direct, CoT, proof-only, latent-only, and hybrid arms. Publish the full cost/accuracy curve.
7. **Connect to Broca only after the typed protocol works.** First measure the existing NORMAL-vs-RESET recurrent-state ablation. Temporal traces/e-prop remain gated by the already frozen Temporal Benchmark V2 preregistration; do not use this protocol to bypass it.
8. **Expand to native generation and real-world source grounding.** Add authentication, role/scope authorization, source revocation, and source-quality review as separate layers. A correct proof over a false or unauthorized premise is not a correct factual answer.

## 9. What would count as a real advantage

A convincing result is not “Symthaea has a proof graph” or “latent vectors are more efficient.” It is a measured system that, under the same decoder and total budget:
- reaches the same or better final-answer accuracy;
- emits fewer unsupported derivations;
- survives premise deletion, contradiction, reordering, and rule updates;
- knows when to abstain;
- spends extra compute only on unresolved high-impact obligations;
- and produces explanations whose claims can be independently checked.

The most plausible differentiation is **latent adaptive computation + typed proof maintenance + independent verification + evidence revocation**. That is a more defensible target than attempting to beat frontier LLMs at free-form CoT with the current native Broca generator.

## 10. Explicit non-claims

This protocol does not claim that the current Symthaea implementation already performs proof-carrying latent reasoning, that psi or caller-provided confidence measures answer correctness, that the legal kernel authenticates sources, that proof graphs expose all latent computation, or that Symthaea currently matches or beats frontier reasoning models. It defines the experiment needed to establish or falsify those claims.
