# Symthaea Research Intelligence and Formal Discovery Programme

**Version:** v0.1  
**Date:** 2026-11-14  
**Status:** Research specification and source audit; implementation and benchmark execution not yet claimed  
**Owner:** Luminous Dynamics  
**Parent work:** [SYM-RSI-TASTE-001 #10877](https://github.com/Luminous-Dynamics/symthaea/issues/10877)  
**RH workstream:** [MATH-RH-001 #12129](https://github.com/Luminous-Dynamics/symthaea/issues/12129)

## 1. Executive position

The next improvement is not another general-purpose agent wrapper. Symthaea needs a research system that becomes better at deciding **what to investigate, how to investigate it, what the result actually establishes, and when to stop or change direction**.

Treat research as a closed, evidence-bearing decision process:

Question → Evidence state → Hypotheses → Discriminating action → Observation → Independent assessment → Updated research policy.

The important unit is not a generated paper or a plausible explanation. It is a prospective decision with a frozen information state, a falsifiable prediction, an admissible experiment, a realized outcome, and independent scoring.

This programme composes existing owners instead of creating parallel versions of their infrastructure:

- #10877 owns research taste and its prospective evaluation.
- #10879 / draft PR #10879 owns typed research decisions and the initial synthetic fixtures.
- #12046 owns decision-sufficient evidence acquisition and value of information.
- #12075 owns evidence-bound scientific anomaly/hypothesis/investigation composition.
- #6862 owns cold mathematical discovery, holdout and novelty controls.
- #10104 owns mathematical discovery/proof-integrity task construction.
- #10810 owns SAT/UNSAT/theorem-proof semantic separation.
- #5370 and #12126–#12128 own web-research epistemic repairs.
- #11920 owns benchmark/grader integrity.

No new scheduler, memory store, universal anomaly detector, evidence ledger, proof authority, or promotion authority should be introduced by this document.

## 2. Paradigm shifts

### 2.1 From information retrieval to decision-sufficient research

Do not maximize papers read, URLs collected, citations, context length, or confidence. Identify the smallest competent evidence or experiment that could change a material research decision.

For each candidate action, estimate:
- expected reduction in decision loss;
- probability that competing hypotheses make distinguishable predictions;
- acquisition/compute/time cost;
- source dependence and model misspecification risk;
- privacy, rights and operational constraints;
- expected value of the next observation, not merely raw entropy reduction.

Hard rights, safety and authority constraints remain non-compensable gates. When value cannot be estimated credibly, abstain or request review.

### 2.2 From idea volume to learned research taste

Research taste is a vector, not a scalar:
1. **Direction:** choose problems with importance, neglectedness and tractability.
2. **Approach:** define a concrete mechanism or mathematical construction that could advance the problem.
3. **Experimental:** choose the best discriminating test under a fully inclusive budget.
4. **Interpretive:** distinguish positive, null, negative, ambiguous and invalid results; update proportionately.
5. **Transfer:** improve decisions on new families without task-specific leakage.

Store accepted and rejected directions, null results, counterexamples, pivots and reasons for stopping. Do not learn only from successful demonstrations. Test whether the successor research policy improves from the same starting state and matched total budget.

### 2.3 From one favored explanation to open-world hypothesis competition

Maintain multiple explanations plus an explicit OTHER / model-inadequacy hypothesis. Each hypothesis must predict observations that differ from its competitors.

A candidate that makes no discriminating predictions is not a useful explanation. If no admissible observation can distinguish two candidates, report non-identifiability rather than inventing confidence.

### 2.4 From plausibility to proof-carrying research

Every important output should distinguish:
- source assertion;
- extracted claim;
- model hypothesis;
- numerical observation;
- simulation result;
- symbolic derivation;
- proof candidate;
- kernel-accepted proof;
- independent replication;
- human/domain review.

The transition between these states requires a typed, inspectable artifact. Agent consensus, confidence values, high Phi, plausible prose, or successful code execution do not authorize promotion to a stronger epistemic state.

### 2.5 From HDC as assumed advantage to HDC as falsifiable research hypothesis

HDC may be valuable for associative retrieval, structural analogy, compact episodic indexing and cross-domain relationship search. CfC/LTC dynamics may be useful for evolving evidence state and time-dependent hypotheses.

Those are hypotheses to test, not privileges to assume. Compare HDC/CfC against:
- lexical/BM25 retrieval;
- dense embedding retrieval;
- citation and theorem-dependency graph traversal;
- symbolic indexing and unification;
- hybrid retrieval;
- strong general-purpose model plus the same external tools.

Ablate HDC, temporal dynamics, episodic memory, graph search and active inference separately. The HDC/CfC path is retained only where it produces a repeatable causal improvement at matched quality, total cost and compute.

### 2.6 From model-generated papers to reproducible discoveries

A research report is a rendering of a research record, not the record itself. Each numerical claim links to a raw execution or measurement artifact; each cited claim links to a source span and stance; each proof links to an exact statement and checker result; each method description matches the actual code/configuration.

Failed, incomplete, refused, negative and invalid runs remain in the denominator.

## 3. Current SOTA and industry comparison (2026-11-14 snapshot)

These benchmarks measure different constructs. They must not be added into one leaderboard or compared as if their scores were interchangeable.

| Comparator | What it measures | Public reference point | Symthaea use |
|---|---|---|---|
| AstaBench | Broad scientific workflow across literature, code, analysis and discovery | Stanford AI Index 2026 reports 2,400+ problems, 57 agents / 22 agent classes; best average score about 0.53 at about $3.40/problem | Broad workflow baseline; report per-task score and inclusive cost |
| FIRE-Bench | Full-cycle rediscovery of findings from published ML papers | Current public leaderboard: best F1 46.7; 30 tasks; 73.6% of errors attributed to planning failures | Test research planning, execution and evidence-backed conclusions |
| ASI-Bench | Project-level research as methodological guidance is withdrawn | Leaderboard updated Nov 10, 2026; top reported overall 63.90, second 62.23 | Test whether Symthaea can choose methods rather than follow a supplied recipe |
| IdeaTaste | Novel idea / approach matching against later research | Models remain far below human-community reference, especially on concrete ideas and plans | Measure feasibility and approach specificity, while protecting genuine novel discoveries |
| TasteVal | Experimental planning and compute-to-target under a fixed coder | Reported 2.30x experimental-compute multiplier and 1.14x normalized final-performance multiplier for one evaluated system; researcher inference cost is excluded from the compute multiplier | Separate experimental compute, final quality and fully inclusive total cost |
| Formal Conjectures | Formal research conjectures in Lean | Immutable benchmark snapshots; May 2026 paper describes 2,615 statements including 1,029 open research conjectures | Pin a stable snapshot; measure statement fidelity, proof discovery and kernel acceptance |
| ETH-ResearchBench / Aegis | Hard mathematical discovery, counterexamples, Rust-to-Lean verification and proof integrity | #10104 contains the November 2026 comparison and task design | Prioritize discovery and correctness over saturated theorem-completion scores |
| Bounded zeta certificate project | Kernel-checked finite-height verification for zeta zeros | Public project reports T = 100,000,300 and 248,008,818 zeros within that finite range | A demanding domain-specific reproduction and certificate-integrity baseline |

Sources:
- AstaBench overview: https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf
- FIRE-Bench paper: https://proceedings.mlr.press/v306/wang26d.html
- FIRE-Bench leaderboard: https://firebench.github.io/
- ASI-Bench: https://asibench.apexin.ai/
- IdeaTaste: https://arxiv.org/abs/2611.03424
- TasteVal: https://arxiv.org/abs/2610.06824
- Formal Conjectures paper: https://arxiv.org/abs/2605.13171
- Formal Conjectures stable snapshots: https://github.com/google-deepmind/formal-conjectures
- Mathematical discovery/proof integrity comparator inventory: #10104
- Bounded zeta certificate project: https://github.com/smmercuri/zeta

### What the comparison tells us

The current frontier does not justify claiming that autonomous scientific research is solved. AstaBench's best reported overall score is still around 0.53; FIRE-Bench's leading agent is below 50 F1 and identifies planning as a major bottleneck. ASI-Bench's leading result is useful as a project-level methodological-autonomy reference, not a measure of mathematical truth. Formal mathematics has high scores on some statement-supplied benchmarks, but discovery, theorem-statement fidelity, open-ended proof obligations and proof integrity remain separate tasks.

The opportunity is to outperform on a **matched capability vector**:
- more correct decisions per total dollar and hour;
- fewer invalid or wasteful experiments;
- better prospective prediction calibration;
- stronger open-world and out-of-domain performance;
- more independently verified intermediate discoveries;
- zero accepted false proofs or corrupted evidence claims.

Do not claim “beats SOTA” from a new score on a different task, budget, model, split or grader.

## 4. Research architecture and boundaries

Use a three-role architecture, while preserving distinct epistemic authority:

| Role | Responsibility | Must not claim |
|---|---|---|
| Symthaea | Retrieve and connect knowledge, form hypotheses, choose research directions, design experiments, interpret results and update strategy | That a generated idea or high confidence is a discovery |
| Symtropy | Execute frozen experiments, simulations, computations and reproducible replays; return raw and normalized observations | That synthetic simulation is physical measurement or solver convergence is model validity |
| SAM | Independently inspect evidence lineage, verifier outputs, negative controls, statement fidelity and claim ceilings | That a structurally valid receipt proves the underlying physical or mathematical claim |

The independent evaluator and protected ground truth must not be authored or mutable by the candidate being evaluated. Use #11920 and existing evaluator/receipt authorities.

## 5. Research benchmark design

### 5.1 Core experimental arms

For each task family, compare:
1. no-search / static evidence packet;
2. active literature search;
3. cost-matched random retrieval;
4. lexical and dense retrieval baselines;
5. theorem/citation graph retrieval;
6. Symthaea full research stack;
7. expert researchers with individual pre-discussion judgments and adjudicated panel review.

Use the same model where possible to isolate the value of the Symthaea components. Match source cutoff, task, tools, compute, token budget, retry policy, total cost and evaluation protocol.

### 5.2 Required scorecard (do not collapse)

**Research direction**
- expected value predicted before action;
- realized value and opportunity cost;
- calibration and selection diversity;
- performance on fresh problem families.

**Approach construction**
- exactness of hypothesis/mechanism;
- predicted intermediate effects;
- feasibility, falsifiability and relevance;
- source grounding and novelty kept distinct.

**Experiment selection**
- discrimination among competing hypotheses;
- expected and realized value of information;
- cost/latency/reviewer burden;
- failed or invalid experiment rate;
- matched-compute result quality.

**Interpretation**
- proper prediction scores (Brier/log/CRPS where appropriate);
- calibration and selective risk/coverage;
- correct handling of null, conflicting and missing evidence;
- continue / pivot / stop / replicate / escalate accuracy;
- overclaim and false-clear rate.

**Mathematical discovery**
- statement fidelity;
- lemma and conjecture discovery;
- exact counterexample validity;
- candidate proof rate;
- Lean kernel acceptance;
- independent checker acceptance;
- axiom profile and sorry/admission scan;
- holdout theorem-family transfer;
- total CPU-hours, wall time, memory, model/tool tokens and cost.

**Evidence integrity (hard gates)**
- zero phantom citations in the audited set;
- zero unsupported numerical claims;
- zero post-cutoff leakage;
- zero hidden evaluator access or grader mutation;
- zero rights/authority violations;
- zero false proof acceptances;
- exact source, code, configuration and artifact identity.

### 5.3 Prospective targets (not achieved results)

After a pilot establishes variance and statistical power:
- at least 20% lower realized investigation loss at matched eligible resources on two independent task strata;
- at least 20% reduction in unnecessary evidence requests with no increase in missed critical evidence;
- at least 20% fewer wasted proof-search/experiment resources than the matched baseline;
- positive prospective calibration skill against base-rate and simple-rule baselines;
- positive transfer to unseen domains/families;
- 100% registered positive/negative control outcomes;
- zero accepted false claims, false proofs, weakened theorem statements or source-lineage bypasses.

Targets are goals, not measured performance. Report confidence intervals, worst-stratum performance and full denominators.

## 6. Riemann Hypothesis workstream

The Riemann Hypothesis asserts that every nontrivial zero of the Riemann zeta function has real part 1/2. It remains an open problem. RH is a moonshot, not the first acceptance criterion.

### 6.1 Start with the proof boundary

The public smmercuri/zeta project formalizes a theorem RiemannHypothesisUpTo(T): all nontrivial zeros with imaginary part bounded by T lie on the critical line. It reports a finite ceiling of T = 100,000,300 and 248,008,818 zeros. This is a substantial formal-verification achievement, but it is not the universal Riemann Hypothesis.

The project describes different trust layers (Lean kernel, native checker and Python) and notes that its global comparator has practical limitations at large heights; sharded checks and published build artifacts are part of the large-scale trust story. Do not flatten those trust distinctions into “Lean proved RH”.

Start with a small self-contained height (10^4), then 10^5 only after the complete build and comparator reproduction is understood. Record external zero search, data source, certificate generator, source tag, Lean version, axioms, shard receipts, proof retention/compression, independent comparator, CPU hours and all excluded outputs. Never jump directly to the 10^8 headline.

### 6.2 Mathematical map

Build a versioned theorem/dependency graph for:
- analytic continuation, functional equation, trivial and nontrivial zeros;
- zero-free regions and zero counting;
- Riemann–von Mangoldt and explicit formula;
- zero-density estimates and moments;
- Li, Weil and Nyman–Beurling criteria;
- spectral / Hilbert–Pólya approaches and their missing implications;
- known positive-proportion and finite-height results.

Every node includes an exact statement, assumptions, primary source, formal status, dependencies, counterexamples to common overgeneralizations and unresolved obligations.

### 6.3 Research directions worth pursuing

Do not ask the system to “solve RH” as one unconstrained prompt. Generate and evaluate concrete intermediate targets:
- sharpened zero-density or moment inequalities;
- new estimates that constrain possible off-critical-line zeros;
- formalization of known results and hidden assumptions;
- equivalence transformations that expose new tractable subproblems;
- explicit error bounds and certified numerical lemmas;
- proof-search and certificate-compression improvements transferable to other theorem families.

Each proposal must name the exact statement, assumptions, proof sketch, expected intermediate lemma, fastest falsifier, resource budget, prior-art coverage and stop/pivot rule.

### 6.4 Formal proof acceptance

A research system may emit a conjecture, a tactic script or a proof candidate. Only the exact theorem statement accepted by the declared Lean kernel and independently audited statement/axiom profile can be called a formally verified theorem.

Bind each accepted artifact to:
- theorem statement digest and declaration name;
- Lean and Mathlib versions;
- import/dependency closure;
- kernel/checker identity;
- #print axioms result;
- absence of sorry/admitted proof;
- proof artifact digest and reproducible build manifest;
- independent statement-fidelity review.

A bounded-height theorem, a positive-proportion theorem, a conditional theorem and RH itself are different results. Never promote one into another.

## 7. Immediate implementation order

| Priority | Work | Owner | Runner-independent output |
|---|---|---|---|
| P0 | Fix source-stance and confidence semantics | #5370, #12126, #12127 | Typed contract, migration design, adversarial fixtures |
| P0 | Close web-to-grammar authority gap | #12128 | Fail-closed admission contract and negative controls |
| P0 | Correct proof outcome semantics in math service | #12129 / #10810 | Typed proof-state design and tests for non-proof outputs |
| P1 | Build the zeta source/theorem map | #12129, #6862 | Versioned map, frozen literature corpus, exact statements |
| P1 | Reproduce bounded-height baseline at 10^4 | #12129 | Pinned instructions, trust model, resource ledger |
| P1 | Refresh benchmark adapters and matched baselines | #10877, #10104, #11920 | Frozen benchmark manifest and comparison matrix |
| P2 | Run cold discovery on tractable RH-adjacent subproblems | #6862, #10104, #12129 | Candidate archive, holdouts, counterexamples, formal certificates |
| P2 | Evaluate HDC/CfC research contribution | #10877, #12075 | Ablation report against lexical, dense, graph and symbolic baselines |

### What can be done without waiting for CI

- Inventory source claims and exact code paths.
- Freeze literature and benchmark snapshots, licenses and cutoff dates.
- Draft JSON schemas and exact typed state transitions.
- Author adversarial and positive-control fixtures.
- Build a manual theorem/dependency map and proof boundary register.
- Specify the bounded-height reproduction and cost ledger.
- Prepare human reviewer rubrics and preregistered statistical plans.
- Review the exact code and create minimal patches after contracts are agreed.

No build, clippy, test, Lean proof, benchmark run or independent replay is claimed by authoring these plans. When runners become available, run them on the exact PR head and keep queued/skipped distinct from pass.

## 8. Stop conditions and claim discipline

Stop or pivot when:
- sources do not support the cited claim;
- source independence is unknown or contradicted;
- a candidate lacks a discriminating prediction;
- a required experiment is out-of-domain or inadmissible;
- the model is not identifiable from available evidence;
- the proof statement differs from the intended result;
- the formal artifact depends on unreviewed axioms or admissions;
- the claimed improvement disappears under matched budgets, holdouts or independent grading.

The project succeeds when it produces more independently validated, useful discoveries per unit of total resources while becoming better calibrated about what it does not know—not when it produces more confident text, more agents, or more benchmark claims.


## 9. Benchmark and industry comparison refresh — 2026-11-14

This section makes the external comparison more precise. The figures below are snapshots of different benchmark constructs; they are not one common leaderboard and are not direct evidence of Symthaea capability.

### 9.1 What current public systems actually measure

| Benchmark | Latest relevant public snapshot | What it demonstrates | What it does not demonstrate |
|---|---|---|---|
| **AstaBench** | 2,400+ problems across 11 benchmark components; 57 agents tested across 22 classes, with 18 overall entries capable of completing the full suite; leading overall score approximately 0.53 at roughly US$3.40/problem in the Stanford AI Index 2026 snapshot | Breadth across literature understanding, code/execution, data analysis and end-to-end discovery | A single continuous scientific campaign; a complete 57 × 11 evaluation matrix; universal expert-level research competence |
| **FIRE-Bench** | 30 ML-paper rediscovery tasks; Claude Code Sonnet-4 at 46.7 ± 23.4 claim-level F1; Codex GPT-5-medium at 41.9 ± 25.4; three runs per task; benchmark reports 73.6% of errors in planning | Full-cycle rediscovery with a verifiable empirical claim target; very large run-to-run instability | Original scientific discovery, universal research ability, or a stable single-run score |
| **Autonomous Science Bench 0.1** | Six laboratory-derived pilot tasks; ten configurations completed all six, three trials/task; GPT-6 Astra + Codex 72.2% at US$190.23 total; Opus 5.5 + Claude Code 66.7% at US$172.91 | Budgeted design–measure–learn campaigns with separate discovery and held-out-learning gates | A statistically decisive winner from a three-trial/task pilot; physical laboratory performance for replay/twin backends |
| **ASI-Bench** | 60 tasks, B1–B4 methodological-guidance ladder; leaderboard updated 10 November 2026; leading reported aggregate 63.90 | Whether the system can choose methods as procedural guidance is withdrawn | Empirical truth, independent discovery, or proof validity |
| **IdeaTaste** | More than 90,000 proposals across six models and four research areas; models average about 26% of human level on citation matching and 18% on idea matching | A major gap between recognizing research directions and constructing the concrete approaches later pursued by communities | Scientific truth or a fair penalty for a genuinely original result that has not yet appeared in a paper |
| **TasteVal** | Eight private AI-R&D tasks; 24 recruited experts; Opus 5.5 reported 2.30× experimental-compute multiplier (95% CI 1.15–4.37) and 1.14× normalized final-performance multiplier (95% CI 1.01–1.31); one refusal was missing on one task | Experimental planning and interpretation can reduce experimental compute to reach a target | A 2.30× quality gain or an inclusive total-cost gain; researcher inference-token costs are excluded from the compute multiplier |
| **Formal Conjectures** | 2,615 Lean 4 statements; 1,029 open research conjectures and 836 solved problems for autoformalization in the May 2026 paper snapshot | A path to zero-contamination formal research discovery and theorem-level evaluation | Proof search on one mutable snapshot as a stable, timeless leaderboard |
| **smmercuri/zeta** | Kernel-checked RiemannHypothesisUpTo(T) reported through T = 100,000,300, covering 248,008,818 nontrivial zeros in the declared finite range | A rigorous bounded-height verification pipeline, with explicit kernel/native/Python trust layers | The universal Riemann Hypothesis |

Primary sources:
- AstaBench / Stanford AI Index: https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf
- AstaBench project: https://allenai.org/blog/astabench
- FIRE-Bench leaderboard and task definitions: https://firebench.github.io/
- FIRE-Bench paper: https://arxiv.org/abs/2602.02905
- Autonomous Science Bench 0.1: https://www.autonomous-science-bench.ai/about and https://www.autonomous-science-bench.ai/leaderboard
- ASI-Bench leaderboard: https://asibench.apexin.ai/
- IdeaTaste paper and leaderboard: https://arxiv.org/abs/2611.03424 and https://cxcscmu.github.io/IdeaTaste/
- TasteVal paper: https://arxiv.org/abs/2610.06824
- Formal Conjectures: https://arxiv.org/abs/2605.13171 and https://github.com/google-deepmind/formal-conjectures
- Bounded-height zeta verification and trust model: https://github.com/smmercuri/zeta

### 9.2 The paradigm shift these results suggest

The bottleneck is increasingly **research decision quality**, not simply code generation or search-result volume.

- IdeaTaste suggests that broader retrieval and more generated proposals do not automatically yield concrete, feasible approaches.
- TasteVal demonstrates why experiment efficiency and final scientific performance must be separate outcomes.
- FIRE-Bench shows both a planning bottleneck and high variance. A single successful run is not a reliable research capability claim.
- Autonomous Science Bench makes experiment selection, outcome interpretation, discovery and held-out learning jointly observable. Its six-task pilot is useful, but the current three-trial/task design cannot support strong claims from small score gaps.
- AstaBench is a valuable broad compatibility suite, but its overall score is a macro aggregate across components rather than one end-to-end campaign success probability.
- ASI-Bench adds an important methodological-autonomy axis. It should complement, not replace, outcome-based scientific validity.
- Formal Conjectures and the zeta project demonstrate why exact theorem statements, frozen source versions, kernel acceptance and external trust boundaries matter.

### 9.3 How Symthaea should compete

Do not build one agent and tune it against one headline number. Run a preregistered, matched tournament with controlled factors:

1. **Information mode:** fixed source packet; active literature retrieval; cost-matched random retrieval; theorem/citation graph; HDC-assisted retrieval.
2. **Reasoning mode:** fixed model and prompt; model plus typed Research State; model plus counterfactual hypothesis set; full Symthaea research policy.
3. **Execution mode:** fixed Coder; Symtropy replay; Symtropy digital twin; human-supervised physical campaign only where authorized.
4. **Evaluation mode:** independent claim/source adjudication; hidden outcome oracle; kernel checker; blinded domain-expert review.

Hold the model, task snapshot, tool access, context/token budget, retries, wall-time limit, total compute, evaluator and data cutoff constant when isolating a component. Test the interaction between base-model capability and active exploration rather than assuming retrieval helps every model equally.

### 9.4 Scoreboard: separate quality, cost, reliability and integrity

For each benchmark family, publish the full vector:

- task-native result and per-task outcomes;
- prospective prediction calibration;
- discovery quality on measured items;
- learning/generalization on held-out unmeasured items;
- claim-level precision/recall and false-positive/false-negative rates where applicable;
- correct abstention and non-identifiability recognition;
- mean, median, worst-stratum and run-to-run dispersion;
- total model tokens, tool calls, Coder tokens, CPU/GPU hours, peak memory, energy where measured, API/tool cost, human review time and retry cost;
- independent source/statement/measurement verification;
- rights, evaluator, security and authority violations as hard gates.

A benchmark that omits model inference cost, human review, refusals or invalid attempts must be described with those omissions. Preserve every run and denominator. Do not convert a missing or refused run into a pass.

### 9.5 Prospective stretch goals after baseline reproduction

These are **targets, not achieved results**. Freeze the evaluation protocol and conduct a powered pilot before using them for promotion:

| Track | First milestone | Stretch target |
|---|---|---|
| Broad workflow (AstaBench-compatible) | Reproduce the strongest comparable public baseline on the same benchmark revision | At least 10% relative score improvement at no greater fully inclusive cost, with the paired confidence interval excluding zero |
| Empirical rediscovery (FIRE-Bench-compatible) | Reproduce claim-level scoring and three-run variance on a pinned 30-task subset | Mean F1 above 55 with materially lower run variance, without increasing unsupported-claim false positives |
| Scientific campaigns (AS-Bench-compatible) | Pass both discovery and held-out-learning gates on a pinned replay campaign | At least 10% lower fully inclusive cost-to-target than the strongest matched baseline, with no reduction in either final gate |
| Research taste (IdeaTaste-compatible) | Improve concrete-approach specificity and feasibility against a blinded human reference | Reach at least 30% of the human reference on idea matching while separately retaining reviewed, independently valuable novel discoveries |
| Experimental taste (TasteVal-compatible) | Measure experimental-compute-to-target and normalized final score separately | Exceed the 2.30× reported compute multiplier on a genuinely comparable suite while maintaining final quality; also lower inclusive cost per confirmed improvement |
| Methodological autonomy (ASI-Bench-compatible) | Demonstrate valid method selection at B3 and distraction resistance at B4 | Improve the matched B3/B4 results without degrading outcome validity or transfer |
| Formal mathematics | Reproduce a frozen Formal Conjectures snapshot and bounded-height zeta baseline | More independently accepted hard-mode discoveries per total cost, with zero false theorem acceptances and positive transfer to unseen theorem families |

The stretch targets are not comparable across different benchmark task sets. “SOTA” is claimed only on an exact task/split, benchmark version, model/harness, budget and evaluator match.

### 9.6 Zeta verification: preserve the trust boundary

The smmercuri/zeta project reports kernel, native-checker and Python results separately. For the 10^8 record it reports the Lean kernel theorem for the finite interval, but its large build pipeline may discard proof terms after checking and retain hashes/build logs. The README reports successful comparator checks at 10^4 and sharded comparator checks at 10^5 and 10^6; it warns that the global comparator is impractical beyond 10^4 under the current Lean literal-hashing issue.

Therefore:
- reproduce the small 10^4 configuration first;
- distinguish a kernel build, native precheck and independent comparator run;
- do not imply the 10^8 proof terms remain available for inspection if they were discarded;
- do not imply a sampled/sharded independent comparator is the same as a complete global re-check;
- do not promote any bounded-height result to full RH.

This programme should outperform industry by being more reliable, more cost-efficient and more epistemically disciplined—not by claiming that different benchmark scores prove one system is generally superior.
