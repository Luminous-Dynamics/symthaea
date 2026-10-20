# Proof Insight Transfer: research design (2026-10-19)

**Status:** research proposal / benchmark contract; no performance claim.  
**Symthaea base:** `41e968095ed71170ac29f6a78155e2094245a452`  
**Upstream OpenAI math snapshot inspected:** `1581d4cbdd69a0faf34b2089614fe2a814316a6a`  
**Design principle:** learn from proofs by testing their structure and consequences, not by treating vector similarity as mathematical understanding.

## Executive decision

Build an evidence-bounded **Proof Insight Transfer** capability around the existing `symthaea-lean-bridge` and `symthaea-proof-audit`. Do not build a second theorem prover, another untyped belief store, or an HDC-only proof classifier.

The measurable target is:

> Given a verified or carefully scoped proof of theorem A, extract a typed, falsifiable strategy record; retrieve an independently relevant theorem B; formulate a transfer attempt with explicit preconditions; then establish whether that transfer creates verified progress on B.

A result counts as useful only when it beats non-neural / standard retrieval baselines on held-out theorem families and the formal checker validates the exact target statement. Similarity scores, plausible explanations, and generated conjectures do not count as mathematical discoveries on their own.

## Why this is the next step

### Current Symthaea evidence

The repository already has the right trust boundary: a Rust-to-Lean bridge and a proof-audit crate that checks axiom provenance and pinned-statement conformance. The April 2026 mathematical-reasoning report provides a critical negative control:

- The proof bridge reported 31/32 accepted on a **hand-curated**, in-scope miniF2F subset (96.9%).
- Its auto-ingested subset reported a 46% median acceptance rate across three 50-item seeds (44–50%).
- A surface-token HDC signature classifier achieved 22/31 accuracy, exactly matching the majority-class baseline. It had no lift.
- The same report explicitly concluded that token-presence similarity did not capture enough semantic structure to select tactic cascades.

These numbers are not directly comparable to leaderboard miniF2F scores: the hand-curated subset was selected for the bridge's supported AST and tactic scope, while the auto-ingested harness has its own parser and translation failures. The work must preserve this distinction in every report.

### What the field teaches us

- Retrieval matters in formal proving, but random theorem splits can leak nearby proof patterns. LeanDojo documents family-/premise-aware generalization concerns and a Lean 4 benchmark with 122,517 theorems/proofs.
- DeepSeek-Prover-V2 reports 88.9% pass ratio on miniF2F-test and 49/658 PutnamBench problems in its 2025 paper. Treat those as **published results under that paper's setup**, not a guarantee about the October 2026 global frontier.
- The 2026 ACL paper *Discover and Prove* introduces hard-mode variants where the model must first discover an answer instead of being handed the answer in the formal statement. It reports 36 formally proved PutnamBench theorems in hard mode and emphasizes the gap between informal answer discovery and formal proof completion.
- **Constructive solving is a separate, harder task:** ICML 2026's FPS/D-FPS framework makes the answer an unknown witness tied to proof obligations and uses Restricted Propositional Equivalence (RPE) to score semantic answer equivalence. Its released benchmarks contain 387 FormalMath500, 375 MiniF2F-Solving, and 324 PutnamBench-Solving items. Their published D-FPS Hybrid-CoT/DeepSeek-V3.2 solved rates are 48.58%, 57.33%, and 1.54%, respectively; these are setup-specific reference rows, not directly comparable to ordinary theorem-proving scores. ([paper](https://proceedings.mlr.press/v306/liu26gl.html), [implementation/results](https://github.com/Purewhite2019/formal_problem_solving_main))
- **Latest research-grade comparator (14 October 2026):** First Proof Batch 3 tested five systems on ten specialist research-math problems. Thirty expert reviewers scored correctness, exposition, and attribution. ProofCouncil reached 9/10 correct and GPT-6 Astra 7/10; only 4/10 Astra solutions had adequate citations, versus 7/10 for ProofCouncil. Negative scoring penalized wrong answers, and the open-weight entrant's submissions all contained major errors. This is not numerically comparable to Lean benchmarks, but shows why research-grade evaluation must measure correctness, exposition, attribution, and abstention separately. (https://1stproof.org/third-batch-press-release.html)
- OpenAI's pinned public `math` snapshot (`1581d4cbdd69a0faf34b2089614fe2a814316a6a`) has 719 manuscripts in 372 related-result families and reports roughly 42% of top-line results formalized. The indexer cross-checks the README's declared family count against parsed catalog headers and fingerprints the README, family map, formalization catalog, and Comparator scope notes. Its deterministic family split is a pre-screen assignment—not a guarantee against cross-family mathematical dependence or pretraining contamination. The repository maintainers warn that some Comparator configurations validate supporting results rather than the paper's headline theorem.
- **A critical evaluation finding:** the ACL 2026 industry-track paper *Benchmarking Testing in Automated Theorem Proving* proposes T², evaluating generated theorem statements by compiling dependent successor theorems. In its 2,206-problem setup, the best reported model reached 38.9% testing accuracy despite strong compilation results. This is a reason to measure semantic/downstream correctness, not just “Lean accepted.” (https://aclanthology.org/2026.acl-industry.150/)

Consequently, Symthaea should not pitch itself as competing on raw theorem throughput yet. Its differentiator should be measured *cross-family insight transfer, failure explanation, and evidence quality*.

## Paradigm shifts

### 1. From proof text to typed, executable insight contracts

Represent a proof idea as a structured record, not only a paragraph or vector:

- **Obstacle:** what the current goal makes hard.
- **Move:** a mathematical transformation or strategy (e.g. expose an invariant, decompose a conjunction, normalize a denominator, reduce to a subproblem, introduce a dual formulation).
- **Preconditions:** hypotheses, domains, regularity, non-zero conditions, finiteness, or other required conditions.
- **State transition:** goal/hypotheses before and after the move.
- **Justification:** exact manuscript spans or formal declarations supporting the record.
- **Failure boundary:** counterexamples, violated preconditions, or known non-applicability.
- **Verification:** checker, toolchain, theorem identity, assumptions/axioms, and exact artifact revision.
- **Transfer candidate:** the target problem and why the typed structures align.

This allows the system to ask not only “what looks similar?” but “does the target actually satisfy this strategy's preconditions?”

### 2. From passive similarity to controlled counterfactuals

For selected formalized results, create controlled variants: remove or weaken a hypothesis, alter a sublemma, swap a proof transformation, or try a nearby false conjecture. Ask the checker and bounded counterexample search what changes.

The intervention is evidence about the dependency of this proof artifact—not a universal proof that a lemma is mathematically necessary. Distinguish:
- artifact dependency (what this proof uses);
- theorem dependence (whether the theorem remains true without an assumption);
- strategy usefulness (whether a move improves proof search or transfer).

Do not infer the latter two from the first.

### 3. From “retrieve related proofs” to “retrieve discriminating experiments”

The best next proof to inspect is not necessarily the nearest one. Prefer examples that separate competing explanations: positive transfer, a near-miss with one violated precondition, and a counterexample. That turns retrieval into a small scientific experiment.

### 4. From proof success to a two-stage research loop

**Discovery stage:** propose lemmas, transformations, proof sketches, and candidate generalizations.  
**Falsification stage:** run counterexample search, check type/domain conditions, verify the exact statement in Lean, audit axioms and assumptions, and record negative outcomes.

A plausible conjecture that fails a counterexample test is a useful negative result. A candidate that compiles but proves a different or vacuous statement is a rejection.

**Add semantic validation:** Lean acceptance proves the exact encoded proposition under its configured axioms and imports; it does not prove the encoded statement faithfully represents the intended manuscript claim. Independently review the informal-to-formal statement mapping and add T²-style *dependent-successor checks* where the theorem has meaningful downstream consumers. Include mutation controls that deliberately alter the target statement or a critical hypothesis, then verify that the expected downstream checks detect the change. A successor test that does not depend on the candidate theorem is not evidence; neither is an unrelated global build failure.

### 5. From one global “math ability” score to a capability vector

Report separate results for:
- parsing/formalization coverage;
- proof completion;
- proof explanation fidelity;
- proof-strategy retrieval;
- cross-family transfer;
- answer discovery in hard mode;
- counterexample detection;
- assumption/axiom auditing;
- human-rated explanatory utility;
- formal-statement fidelity and dependent-successor semantic checks;
- citation/attribution quality and calibrated abstention on research-grade claims;
- compute and wall-clock cost.

This prevents a gain in one easy-to-measure metric from disguising regressions elsewhere.

### 6. From supplied propositions to constructive mathematical problem-solving

In ordinary theorem proving, the proposition is already known and the system searches for a proof. In constructive problem-solving, the answer itself is an unknown witness: Symthaea must derive an object and then prove that it satisfies every stated conclusion. Treat this as a distinct benchmark track rather than conflating “found a plausible answer,” “typechecked a term,” and “proved the requested conditions.”

Use FPS/D-FPS-style task records with explicit unknowns, formal types, candidate witnesses, per-conclusion proof obligations, and RPE or stronger formal equivalence checks. A witness that typechecks but fails the requested conclusions is not solved; two differently represented answers may still be equivalent. This track directly tests whether proof-derived insights help *discover* useful mathematics rather than only close a pre-specified goal.

## Proposed architecture

Use existing components before adding new infrastructure:

1. **Source adapter (new):** ingest `openai/math/CONTENTS.md`, `lean/formalization.yaml`, selected source artifacts, and reasoning-summary metadata from the pinned upstream snapshot. Treat upstream text and theorem outputs as untrusted data, never as instructions to the agent.
2. **Provenance layer (reuse / extend):** preserve repository commit, path, file digest, family, theorem identifier, artifact scope, license, verification commands, and status.
3. **Structural extractor (new):** parse supported Lean declarations/ASTs and manually or model-assisted annotate a small audited pilot. Do not initially attempt to parse every mathematical manuscript or to infer semantic proof structure from token hashes.
4. **Insight record store (new, typed JSON schema):** store strategy contracts and evidence anchors. Keep the graph authoritative; vector embeddings are optional retrieval indexes.
5. **Candidate retrieval (baseline portfolio):** compare BM25/lexical, typed AST features, dependency graph, available embedding search, and HDC representation. No representation wins by architectural preference; it must win a frozen benchmark.
6. **Transfer planner (new):** state a target lemma/goal, map preconditions, predict which proof obligation should improve, include a negative control, and produce Lean code only through a pinned bridge.
7. **Constructive task adapter (new):** represent unknown witnesses and their formal types, enumerate the target conclusions that the witness must satisfy, and record a separate status for candidate generation, typechecking, each proof obligation, and answer equivalence (RPE or a formal equivalence proof).
8. **Trust gate (reuse):** check the exact theorem and expected proposition, audit `sorryAx` and declared axioms under an explicit policy, and preserve checker/toolchain receipts. Failed, timed-out, not-attempted, and accepted are distinct outcomes.
9. **Evaluator (new):** run family-held-out comparisons, report confidence intervals and resource use, archive raw result files, and fail closed when provenance or evaluation data are missing.

Potential new code should start under the existing Lean bridge / proof-audit boundaries. Do not pull heavyweight main-crate cognitive dependencies into the bridge merely to access a 16K HDC vector. The previous cascade-selection null is a warning against architecture-led integration.

## Benchmark proposal: PITB v0

**Name:** Proof Insight Transfer Benchmark (PITB).  
**Purpose:** evaluate whether proof-derived strategy records improve transfer to a different theorem family.  
**Scope:** begin with a small, manually audited pilot; expand only after records and splits are reproducible.

### Dataset construction

1. Pin the OpenAI math repo commit and archive an index of paths, hashes, family identifiers, manuscript status, formalization status, and available verification instructions.
2. Start with 10–20 *candidate* families selected to span arithmetic/number theory, combinatorics/optimization, and analysis/ mathematical physics. Actual inclusion depends on usable evidence, licenses, and independently verifiable scope; do not promise that every family has a formalized headline theorem.
3. Use the ten published reasoning-summary families as an **exploratory development set only** where the summaries are available. They are too few and too exposed to support the headline holdout result.
4. Use the indexer's deterministic 70/15/15 pre-screen split, derived by SHA-256 ranking of a frozen seed and family ID. Preserve assignments and do not reshuffle after seeing task difficulty or model outcomes. After catalog screening, retain explicit eligibility/exclusion/unknown labels within each split.
5. Keep companion papers, alternative proofs, consequences, and derived lemmas in the same source-family partition. The family boundary is only a first defense: audit cross-family mathematical dependencies and disclose possible pretraining exposure before claiming generalization.
6. Add a hard-mode track in which the target answer or key construction is not supplied to the proving agent. Freeze targets before model calls.
6. Include verified positive cases, near-miss cases, weakened-assumption variants, deliberately false conjectures, malformed or mismatched theorem specifications, and timeout/tool-unavailable cases.

Use source links and short evidence spans rather than wholesale replication of manuscript text. Recheck individual license and attribution obligations before redistributing derived artifacts; a repo-wide license alone does not establish every third-party artifact's rights.

### Required baselines

| Baseline | Question |
|---|---|
| Lexical/BM25 retrieval | Do proof words alone help? |
| Typed AST / structural retrieval | Does theorem and term structure improve over lexical retrieval? |
| Dependency-graph retrieval | Do theorem dependencies expose useful neighbors? |
| LLM with retrieved proofs, no insight records | Do typed records add value beyond context stuffing? |
| LLM/prover with no retrieval | Is retrieval useful at all on these items? |
| Majority/constant strategy policy | Does an adaptive choice beat the trivial policy? |
| Symthaea HDC-enabled variant | Does HDC improve outcomes beyond the symbolic and graph baselines? |
| Full pipeline with proof-audit gate | Does the end-to-end result remain correct and checkable? |

Match model, prompt, token/compute budget, tool access, Mathlib revision, retry policy, and timeout across comparisons. Report Pass@1 and bounded Pass@k separately. Record whether a result is from a model's internal knowledge, retrieved evidence, or formal checking.

### Metrics and initial acceptance thresholds

Thresholds below are **proposed go/no-go goals**, not measured results. Pilot estimates should be reported with confidence intervals and sample sizes; do not claim statistical victory from a tiny sample.

| Metric | Definition | Initial target |
|---|---|---|
| Insight evidence fidelity | Fraction of factual statements in sampled records supported by cited proof/manuscript evidence, scored by two reviewers | ≥90%; disagreements retained |
| Research explanation quality | Blinded expert ratings of correctness, exposition, and attribution; false claims are separately penalized | Report per-axis scores and inter-rater agreement; do not hide incorrect results in an average |
| Abstention quality | Risk-versus-coverage curve: incorrect or unsupported assertions as the system answers more transfer requests | At matched coverage, beat the no-abstention baseline on unsupported-claim rate; report false-abstention rate separately |
| Structural retrieval | Expert-labelled relevant strategy in Recall@10, grouped by theorem family | ≥15% relative lift over strongest simple baseline before scale-up |
| Verified transfer lift | Absolute change in target theorems with checker-accepted, exact-statement proofs vs strongest matched baseline | ≥10 percentage points on a sufficiently powered family-held-out evaluation; 95% interval for lift excludes zero |
| Constructive witness discovery | Fraction of tasks where a derived witness is type-correct, every requested conclusion is proved, and the answer passes RPE/formal equivalence | Establish baseline separately on FormalMath500, MiniF2F-Solving, and PutnamBench-Solving; report “candidate”, “typechecked”, and “solved” as separate stages |
| Falsification controls | Rejection of seeded wrong-target, vacuous, `sorry`, undeclared-axiom, and violated-precondition cases | 100% of frozen deterministic gate fixtures rejected |
| Semantic/downstream correctness | Fraction of generated theorem statements that pass reviewed statement correspondence plus relevant dependent-successor tests | Establish a T²-style baseline first; target ≥10 percentage-point lift over a matched baseline, with confidence intervals and mutation-control results |
| Counterexample utility | Rate of seeded false conjectures found within a declared finite search budget | Establish baseline first; report precision/recall separately |
| Calibration | Relation between predicted transfer success and verified outcomes | Brier score + reliability plot; establish baseline before setting a hard threshold |
| Reproducibility | Re-run on pinned artifacts produces same data partition and same deterministic gate outcomes | 100% deterministic replay for supported checks |
| Cost | CPU/GPU time, tool calls, tokens where available, and Lean attempts per verified transfer | Report distributions and quality-vs-budget curves; no single averaged number |

A gate reaching 100% on its deterministic synthetic cases is only a harness correctness condition, not proof of mathematical soundness for arbitrary generated claims. Formal correctness still rests on exact proposition checking and the trusted Lean kernel/dependency configuration.

## Relationship to existing and industry benchmarks

Use established suites for proving competence, but do not confuse them with the new research question:

- **miniF2F / FIMO / PutnamBench:** conventional formal proof completion; include hard-mode or answer-discovery variants where available.
- **LeanDojo Benchmark 4:** premise selection and generalization to new premises in Mathlib.
- **OpenAI `math` artifacts:** source material for proof-insight provenance and cross-family analysis, not a standardized leaderboard by themselves.
- **PITB:** cross-family strategy transfer and quality of mathematical insight records.
- **FPS / D-FPS:** constructive witness discovery with RPE answer equivalence; use as a separate track because solving is not equivalent to proving a supplied proposition.

The dated public comparison points located for this design include DeepSeek-Prover-V2 (its paper reports 88.9% miniF2F-test pass ratio and 49/658 PutnamBench problems), the 2026 *Discover and Prove* hard-mode results, T² semantic testing, and First Proof Batch 3's human-reviewed research-level evaluation. These are different tasks and must remain separate rows in reports.

**Important current-frontier correction (checked 2026-10-19):** a current public status page reports that PutnamBench's 672 Lean statements are now saturated, with a four-way tie at 672/672 and substantially different reported costs across systems (the cost scopes are not directly comparable). See [Humanfia's October 2026 overview](https://humanfia.ai/projects/hoa) and the [PutnamBench project](https://github.com/trishullab/PutnamBench). Treat this as a dated report of the public leaderboard, not a result reproduced by Symthaea. Raw PutnamBench pass count is consequently a weak differentiator at the top of the leaderboard. PITB should instead distinguish itself on **family-held-out transfer, hard-mode answer discovery, semantic/downstream correctness, source attribution, calibrated abstention, independent reproduction, and cost per independently verified result**. First Proof Batch 3's reported results (up to 9/10 correct for a frontier system) are not directly comparable to Lean pass rates; its separate axes—correctness, exposition, attribution, and penalties for wrong answers—should influence PITB's research track.

Before publishing any claim of “state of the art,” perform a dated source review and reproduce a directly comparable baseline on the exact benchmark, split, Lean/Mathlib versions, compute budget, and Pass@k protocol. Otherwise phrase results as “our measured result on PITB v0,” not SOTA.

## No-runner-first execution plan

### Milestone 0 — research contract and source indexer (started)
- Freeze the research contract, baselines, metrics, risks, and no-claim boundaries in this document.
- Add `symthaea-proof-insights`, a Rust-first indexer that pins the OpenAI math checkout to an expected commit, rejects dirty worktrees, fingerprints the source catalog and Comparator scope notes, extracts family links, and emits explicit matched/unmatched formalization-source accounting.
- Add `data/benchmarks/proof-insight-transfer-v0/splits.json`: frozen deterministic 70/15/15 family assignments for the pinned 372-family catalog (260/56/56), 961 unique internal paths across family records, source SHA-256 fingerprints, and a partition digest. The SHA-256 implementation was checked against the standard `abc` test vector; the split is an independent manifest until matched against Rust-generated output.
- Add a `--verify-splits` mode to the Rust indexer to compare generated and frozen family IDs/assignments, source commit, and partition digest.
- Unit tests are authored for family extraction, path normalization, duplicate-family rejection, catalogue-claim consistency, and deterministic split assignment.
- **Verification boundary:** the authoring environment lacked a local Rust toolchain, so the Rust tests and indexer have not yet been executed. The independent static split manifest's hash implementation passes the standard SHA-256 `abc` vector, but the manifest has not yet been matched against Rust output. The JSON Schema has been JSON-parsed and structurally smoke-checked, not passed through an independent JSON Schema validator.

### Milestone 1 — validate and close data-integrity gaps
- Run `cargo test -p symthaea-proof-insights` in a Rust-enabled checkout, then execute the indexer against the pinned upstream snapshot.
- Inspect every unmatched formalization-source path, and compare output family/source/declaration counts against the upstream files; add negative fixtures for missing artifacts, duplicate paths, unknown manifest versions, dirty worktrees, and mismatched commits.
- Independently validate the insight-record JSON Schema and validate generated records against it.
- Acceptance: all catalog entries accounted for as included/excluded/unknown; no silent omissions; exact source commit and file digests present; clean-room rerun yields identical output bytes.

### Milestone 2 — manually audited pilot
- Select 10–20 candidate families after inspecting formalization scope.
- Produce a modest gold set of insight records with two independent annotations on a subset, including failure boundaries.
- Keep published summaries development-only; family-hold out evaluation cases.

### Milestone 3 — baseline tournament
- Establish BM25, structural, graph, and no-retrieval results before enabling HDC.
- Use the same targets and budget, exact artifact pins, a predeclared random seed, and saved raw outcomes.
- If HDC or cognitive-loop features fail to improve the strongest baseline, retain the negative result and do not add integration complexity.

### Milestone 4 — transfer and falsification
- Generate explicit candidate transfers plus near-miss / negative controls.
- Validate exact Lean propositions and record axiom/dependency policy, formalization scope, and checker command.
- Ask mathematicians to score a blinded sample of explanation utility and insight fidelity.

### Milestone 5 — scale only on evidence
- Expand to more families only after provenance is complete and transfer lift is credible under grouped holdout.
- Publish benchmark version, splits, source commits, resource budgets, raw failure classes, and known contamination risks.

## Stop conditions

Stop or redesign if:
- structural/graph retrieval does not beat lexical retrieval;
- insight records cannot reliably cite specific source evidence;
- transfer gains vanish on family-held-out data;
- gains rely on supplying the answer or theorem-specific hints to the prover;
- incorrect target statements can pass the acceptance gate;
- HDC adds complexity without measurable incremental value;
- formalization coverage is misrepresented as headline-theorem verification;
- the benchmark has too few independent families for a supported generalization claim.

Negative results must be preserved. A system that accurately says “no transferable principle was demonstrated” is more valuable than one that emits an attractive but unsupported analogy.

## Research references

- OpenAI math artifacts (snapshot inspected: `1581d4cbdd69a0faf34b2089614fe2a814316a6a`): https://github.com/openai/math
- OpenAI math Lean Comparator scope notes: https://github.com/openai/math/blob/main/lean/ComparatorChallenges/README.md
- Symthaea mathematics arc (April 2026): https://github.com/Luminous-Dynamics/symthaea/blob/main/papers/symthaea-math-arc-2026-04.md
- Symthaea proof-audit gate: https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/bridges/symthaea-proof-audit
- LeanDojo: *Theorem Proving with Retrieval-Augmented Language Models*, NeurIPS 2023: https://arxiv.org/abs/2306.15626
- DeepSeek-Prover-V2, 2025: https://arxiv.org/abs/2504.21801
- Liu et al., *Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4*, ACL 2026: https://aclanthology.org/2026.acl-long.3/
- Liu et al., *Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving*, ICML 2026: https://proceedings.mlr.press/v306/liu26gl.html
- FPS/D-FPS source code, data, and evaluation tables: https://github.com/Purewhite2019/formal_problem_solving_main
- Kim et al., *Benchmarking Testing in Automated Theorem Proving*, ACL 2026 Industry Track (T² downstream semantic tests): https://aclanthology.org/2026.acl-industry.150/
- First Proof Project, *Third Batch Press Release*, 14 October 2026: https://1stproof.org/third-batch-press-release.html
- OpenAI, *Our First Proof submissions*, 20 February 2026: https://openai.com/index/first-proof-submissions/

## Claim ceiling

This proposal does not claim that Symthaea has discovered new mathematics, understands mathematical intuition, beats the best prover, or improves general proof search today. It establishes a falsifiable path to test whether typed, provenance-linked proof insights enable verified cross-family transfer. Only published measurements and independently checked results may raise that claim ceiling.
