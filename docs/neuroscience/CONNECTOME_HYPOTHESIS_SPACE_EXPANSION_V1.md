# Adversarial hypothesis-space expansion for connectome query policies V1

**Tracking:** [#10603 — adversarial hypothesis-space expansion](https://github.com/Luminous-Dynamics/symthaea/issues/10603)  
**Implementation branch:** \`research/connectome-hypothesis-space-expansion-v1\`  
**Parent assurance layers:** [#10528](https://github.com/Luminous-Dynamics/symthaea/pull/10528), [#10568](https://github.com/Luminous-Dynamics/symthaea/pull/10568), [#10588](https://github.com/Luminous-Dynamics/symthaea/pull/10588), [#10590](https://github.com/Luminous-Dynamics/symthaea/pull/10590).

## 1. The remaining blind spot

The adaptive policy can now preserve correlated observations, accept additive support challenges within an existing world, and pass an exhaustive small support-inclusion lattice. Yet all of those guarantees are still conditional on the *declared scenario universe*.

A biologically plausible reconstruction can be absent entirely. That missing world may imply the opposite query decision. Expanding the outcome support within the current worlds cannot expose a world that has never been represented.

The next paradigm shift is to treat the hypothesis universe itself as challengeable.

## 2. Append-only world expansion

A scenario challenge adds one or more new worlds, each containing:
- stable scenario ID;
- graph/reconstruction SHA-256;
- frozen query decision;
- source-evidence and joint-observation-model SHA-256;
- dependence ancestry;
- at least one complete joint outcome tuple over the exact frozen action universe.

Each challenge binds a challenger evidence digest and challenger model digest, along with the base scenario-set and joint-support digests. It is evaluated independently against the same base; additions from one challenge do not silently flow into another.

The challenge may not alter:
- any base scenario ID, graph digest or query decision;
- any base joint-support tuple or provenance;
- any action ID, measurement model, effort cost or action ancestry;
- the frozen query specification or dataset revision.

The expanded input is re-canonicalized and passed back through the same strict validator used for the base. Its scenario-set and joint-support digests are recomputed, not copied from challenger input.

## 3. Monotonicity property

Let the base set of worlds be \(W\), with fixed query function \(q\), action universe \(A\), action costs \(c\), and per-world joint outcome relations \(J(w)\).

A scenario challenge produces:

\`W' = W \\cup W_{new}, \\qquad W\\cap W_{new}=\\varnothing\`

Every old world and every old tuple remains unchanged. New worlds carry their own non-empty joint support over the same actions.

Any policy that resolves all worlds in \(W'\) also resolves all worlds in the subset \(W\). The adaptive policy's adversarial outcome tree therefore has no fewer obligations after expansion. Its exact minimum worst-case cost must obey:

\`C^*_{adaptive}(W') \\ge C^*_{adaptive}(W)\`

The fixed-bundle feasible set also cannot grow: each old opposing-decision pair remains, and new worlds may introduce additional conflict pairs. Therefore:

\`C^*_{fixed}(W') \\ge C^*_{fixed}(W)\`

An infeasible base policy or fixed bundle cannot become feasible by appending a world. Adding a same-decision world may leave cost unchanged; adding an opposing-decision world may increase cost or destroy feasibility.

This is a computational theorem over the declared finite worlds and their supplied supports. It does not prove that the base or expanded world set spans biological reality.

## 4. Synthetic known-answer controls

### 4.1 Newly discovered opposing world

The base five-world example has an exact adaptive optimum of 4 review-minutes and a fixed-bundle optimum of 5.

The challenge appends a sixth, refuting world whose joint observation tuple is identical to an existing supporting world. The new world cannot be distinguished by any declared action from the existing opposite-decision world.

Expected result:
- before: adaptive policy exists at cost 4;
- after: no adaptive policy exists within the declared action universe;
- fixed plan: no robust bundle exists after expansion;
- monotonicity passes because feasibility is lost, not created.

This is the critical scientific counterexample: a policy can be exactly optimal and pass every within-world support test while remaining invalid under a previously omitted opposing reconstruction.

### 4.2 Additional same-decision world

A second independent challenge appends a new supporting world with a joint tuple already represented by an existing supporting world.

Expected result:
- adaptive cost remains 4;
- fixed cost remains 5;
- query unanimity and feasibility are unchanged;
- monotonicity passes.

### 4.3 Invalid expansion controls

Reject duplicate/base-world IDs, scenario/support identity mismatch, invalid graph/source/model digests, malformed tuple rows, wrong action coverage and world-cap overflow.

## 5. Exact outputs

For each scenario challenge the report records:
- challenger and derived challenge digest;
- expanded scenario-set and joint-support digests;
- added world count;
- adaptive and fixed optimum before/after;
- feasibility before/after;
- adaptive and fixed monotonicity verdicts;
- overall fail-closed monotonicity state.

Any decrease in a previously feasible optimum or resurrection of an infeasible plan yields \`HYPOTHESIS_EXPANSION_MONOTONICITY_VIOLATION\` and a non-zero CLI exit after report serialization.

The expanded scenario and joint-support digests use the normal canonical digest functions. The result remains tied to the exact dataset revision, query spec and action-set digest.

## 6. Benchmark and promotion targets

| Metric | Target |
|---|---:|
| Adaptive effort decreases after append-only world expansion | 0 |
| Fixed-plan effort decreases after append-only world expansion | 0 |
| Policy resurrection after adding worlds | 0 |
| Base world mutation/deletion accepted | 0 |
| Scenario/support identity or action-universe mismatch accepted | 0 |
| Independent oracle agreement for expanded digests and costs | 100% |
| Query-critical worlds added by blinded expert challenge | Report recall and provenance |
| Real-world scenario support completeness | Independently adjudicated; never inferred from synthetic passes |

The next scale-up is adversarial scenario generation: experts propose alternate merges, splits, synapse polarity assignments, missing branches and morphology-consistent alternatives. For every proposal, report its effect on query answer, policy feasibility, worst-case review effort and selected policy tree.

## 7. Industry benchmark comparison: where this layer belongs

**Predictive proofreading SOTA.** ConnectomeBench2 is the direct classifier benchmark. The official project page reports more than 716,485 expert-labelled proofreading decisions and over 4.5 million images across mouse, human, zebrafish and fly. Its multi-modal ViT uses both EM and mesh geometry, and reports human-level split/merge proofreading performance. This is a much stronger baseline than generic VLM claims and must be reproduced before claiming predictive superiority. [Project page](https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/).

**Dataset revision discipline matters.** The current Hugging Face card explicitly says \`main\` is the initially submitted V1 and instructs users to load revision \`v2\` for the dataset used by the arXiv paper. It reports 401,170 dataset rows and 63% EM coverage overall; real merge-edit rows have EM coverage of 38% and split-edit rows 23%. The paper's 716,485 number is expert proofreading decisions, not the same denominator as dataset rows. Benchmark reproduction must pin the V2 revision and report per-task/per-modality denominators rather than mixing these counts. [Dataset card, current release note and modality table](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2).

**VLM comparison.** The September 2026 preprint benchmarks 19 open and 2 closed VLMs for synapse presence/polarity and evaluates open VLMs on ConnectomeBench2 proofreading. It reports that most zero-shot VLMs are near chance, while LoRA on a few thousand labels moves open models near specialist performance; it also reports cross-species transfer results. These are relevant comparisons for perception and candidate generation, not for the reliability certificate itself. [arXiv:2609.36492](https://arxiv.org/abs/2609.36492).

**Decision robustness research.** The 2026 UAI paper *Robust Bayesian Decision Making under Adversarial Uncertainty* argues that information gain and nominal decision confidence can yield fragile downstream decisions under hidden/weakly modeled effects, and proposes sequential adversarially robust decision-aware design. Our finite-set approach is complementary: it makes no prior or posterior assumption and focuses on exact support-set additions and proof-carrying query decisions. It should be compared with that paper's Bayesian adversarial approach only on matched tasks where its priors and adversary model are explicitly instantiated. [PMLR paper](https://proceedings.mlr.press/v337/harikumar26a.html).

## 8. Practical route to beating the field

Do not claim better split/merge balanced accuracy because a policy planner is not a segmenter. Compete on a different, downstream failure mode:

1. Reproduce the strongest ConnectomeBench2 V2 perceptual baseline on its official split, with exact species/task/modality denominators.
2. Bind model outputs to immutable data, model, preprocessing, spatial frame and calibration digests.
3. Generate and adjudicate query-relevant alternative reconstruction worlds.
4. Preserve correlated joint outcome support rather than multiplying marginal possibilities.
5. Find exact adaptive review policies and compare them with the optimal fixed bundle.
6. Challenge both within-world outcome support and the world set itself.
7. Issue a certificate only when every surviving hypothesis gives the same frozen scientific answer, and expose when an added world destroys that guarantee.

Primary prospective metrics: false query certificates per 1,000 blinded adjudications; query-critical error recall at matched expert minutes; adaptive effort vs optimal fixed bundle; world-expansion fragility rate; selective risk at declared coverage; calibration under spatial and species shift; and the time required for independent reviewers to replay a certificate.

Targets of zero false promotions in synthetic exhaustive controls, exact oracle agreement on bounded cases, and at least 25% lower expert effort at matched query reliability are goals for prospective evaluation—not achieved biological results.

## 9. Claim boundary

This layer proves only a computational property of an append-only finite world expansion. It cannot prove that the challenger world is physically possible, that the support includes every possible outcome, that a source is authentic or independent, or that the resulting query answer is biologically true. No probabilities, posterior distributions, entropy or expected value of information are calculated.
