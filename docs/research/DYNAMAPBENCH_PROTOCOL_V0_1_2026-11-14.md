# DynaMapBench — Evidence-Grounded Development Intelligence Benchmark Protocol V0.1

**Date:** 2026-11-14  
**Status:** Proposed evaluation protocol; no benchmark execution or SOTA result is claimed  
**Parent:** [DYNMAP-001 #12020](https://github.com/Luminous-Dynamics/symthaea/issues/12020)  
**Tracking:** [DYNAMAPBENCH-001 #12164](https://github.com/Luminous-Dynamics/symthaea/issues/12164)  
**Retrieval lane:** [DYNRETR-001 #12080](https://github.com/Luminous-Dynamics/symthaea/issues/12080)  
**Decision-time lane:** [DTEA-001 #12042](https://github.com/Luminous-Dynamics/symthaea/issues/12042)  
**Execution boundary:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## 1. Purpose and falsifiable claim

DynaMapBench evaluates whether an evidence-grounded development map improves the correctness, efficiency and safety of engineering decisions over strong repository-navigation, code-retrieval, temporal-reasoning and agent baselines.

It is not a single score for "software intelligence". It is a suite of independently interpretable lanes. A system may be strong at retrieving code and weak at temporal admissibility; strong at structural extraction and weak at capability qualification. Scores must remain disaggregated.

The central hypothesis is:

> A typed, revision-aware map that separates declarations, syntax, compiler evidence, runtime observations, temporal applicability and independent qualification will produce better engineering decisions per unit of constrained context and effort, without increasing unsupported claims or unauthorized actions.

A benchmark win requires matched evaluation, frozen inputs, independent labels, uncertainty intervals, and downstream validation. More nodes, more retrieved files, more generated patches, or a higher self-reported confidence are not success criteria.

## 2. Benchmark lanes

| Lane | Question | Core metrics | Primary comparators |
|---|---|---|---|
| D1 — Structural fidelity | Does the map faithfully represent the frozen source tree and configuration? | Node/edge precision and recall, path/blob identity accuracy, extraction coverage, stale-edge rate | Cargo metadata, rust-analyzer/SCIP where applicable, RIG/SPADE, Aider RepoMap |
| D2 — Temporal/evidence admissibility | Does the system know what was effective, observed and available at the decision cutoff? | Decision-time precision/recall, future-evidence false admission, unauthorized-source false admission, bitemporal replay accuracy | SWE-Time for repository time reasoning; DTEA gold and negative controls for evidence admissibility |
| D3 — Retrieval/exploration | Does it find the right source regions under a strict budget? | MRR, Recall@K, line recall/F1, hit-region/file, gold coverage per token, noise, context efficiency | Agent Retrieval Bench V1, expanded 427-case paper release, SWE-Explore/CoSIL, RepoMap, lexical and dense baselines |
| D4 — Impact and planning | Does it identify the real change surface and produce feasible development plans? | Impact recall/precision, prerequisite accuracy, hard-constraint violations, dependency/cost regret, plan validity | SWE-Time prospective lane, repository-twin mutation sets, fixed-budget manual/agent baselines |
| D5 — Causal development intelligence | Does it distinguish causes from correlations and predict intervention outcomes? | Causal-edge precision/recall, intervention effect error, calibration, counterfactual regret | Independently adjudicated intervention records; randomized or matched-pair ablations |
| D6 — Act/defer and evidence acquisition | Does it take authorized action when sufficient, and defer when authority/evidence is insufficient? | Paired act/defer accuracy, critical abstention recall, false-action rate, value-of-information, unauthorized effects | AgentAbstain-style paired tasks; independently authored authority and no-gold controls |
| F — Federation compatibility | Does a cross-project claim hold under both owners' exact snapshots and contracts? | Mismatch recall, false compatibility promotion, two-sided receipt conformance, stale pin detection | Pinned schema/API contract fixtures and independent conformance verifiers |

### Non-negotiable separation

- D1 syntax extraction does not establish compilation.
- Manifest dependency declarations do not establish compiler-resolved imports.
- A test declaration does not establish that a test ran or passed.
- A passing test does not automatically qualify a capability.
- Temporal repository reasoning is not identical to legal/organizational evidence admissibility.
- A source file retrieval hit is not proof of root cause.
- Change impact is not causal attribution.
- A plan that violates one hard safety, authority or physical constraint is invalid regardless of weighted utility.
- A federated relationship does not transfer authority from one repository to another.

## 3. Benchmark corpus and identity

Every case is a frozen, immutable record with:

- benchmark release, split, case ID and task family;
- repository identity, exact commit SHA, exact tree OID and relevant blob OIDs;
- complete/partial tree-inventory declaration and truncation status;
- language, package/build ecosystem and configuration world;
- event, decision and evidence cutoffs when temporal;
- prompt, expected output schema, token/line/tool/time budgets;
- gold labels, forbidden evidence spans and no-gold disposition;
- extractor, ranker, embedding model, tokenizer, prompt and harness identities;
- evaluator version, adjudicator identities and label provenance;
- per-case raw output, diagnostics, cost, latency and reproducibility receipt.

A moving branch, mutable URL or "latest" label is not an identity. Inputs must be fetched and archived by exact immutable object identity. The evaluator must fail closed on missing, mismatched or truncated snapshots.

### Splits and leakage controls

1. **Repository-family holdout:** no repository or close fork may appear across train/dev/test splits.
2. **Temporal holdout:** test cases must be later than training cutoffs; no future commit, fixing PR, post-event issue comment or derived artifact may be exposed before its declared availability time.
3. **Task holdout:** where feasible, reserve entire task families or mutation operators for out-of-distribution evaluation.
4. **Configuration holdout:** reserve unseen feature/target/configuration combinations.
5. **No-gold controls:** include natural absence and counterfactual wrong-repository / wrong-world cases. Keep these labels distinct.
6. **Leakage audit:** strip direct paths, basenames, patches, fixing commits and answer-bearing generated artifacts where the task requires inference; audit both direct and indirect leaks.
7. **Two independent labels:** gold source regions and causal/compatibility labels require two annotators and a third adjudicator on disagreement. Labelers must be blind to model outputs.

Do not use synthetic fixtures as the only evaluation corpus. Synthetic cases are valuable for exact negative controls and mutation coverage; real repositories and independently adjudicated tasks are required for external validity.

## 4. External benchmark baselines — versioned separately

### 4.1 Agent Retrieval Bench V1 (official 225-sample release)

The current project page defines V1 as 225 curated cases: 106 code2test, 51 comment2context and 68 trace2code. Reported overall baselines:

| Baseline | MRR | Recall@20 | Gold@8k |
|---|---:|---:|---:|
| Qwen3-Embedding-4B | 0.2455 | 0.5828 | 0.2542 |
| aider-style RepoMap | 0.2227 | 0.6299 | 0.0704 |
| Jina Code Embeddings 0.5B | 0.1883 | 0.4492 | 0.1556 |
| Lexical | 0.1450 | 0.4874 | 0.0785 |

Task winners differ: Qwen3-Embedding-4B leads code2test MRR and Recall@20; Jina leads comment2context MRR while lexical leads its Recall@20; RepoMap leads trace2code MRR and Recall@20. This argues for task-conditioned retrieval, not a universal ranker.

**Prospective DynaMap target for the exact frozen V1 release:** MRR >= 0.270, Recall@20 >= 0.670, Gold@8k >= 0.280, and no statistically credible regression in any task's macro-average. These are research targets, not current results. Require a paired bootstrap confidence interval and publish all three metrics plus task-level results.

### 4.2 Agent Retrieval Bench expanded paper evaluation (427 cases)

The associated paper describes a separate 427-case / 25-repository evaluation (345 positive, 50 natural no-gold, 32 counterfactual no-gold), with edit2ripple in addition to the V1 task families. Its reported metric leaders differ from the official V1 page: Qwen3-Embedding-4B MRR 0.2379, Qwen3-Embedding-8B Recall@20 0.7029, RepoMap budgeted context yield at 8K 0.3788.

Do not merge these values with the 225-case V1 table. First pin the released benchmark artifact and reproduce the exact paper protocol. Only then set a leaderboard target. No-gold calibration is a separate mandatory metric here: report false-return rate, abstention precision/recall and risk-coverage curves.

### 4.3 SWE-Explore

SWE-Explore reports 848 issues across 10 languages and 203 repositories. Its K=5 table reports CoSIL at 0.788 line-level recall and 0.602 F1, but the high line recall accompanies a comparatively high noise-region rate (0.471). In a separate restricted-context validation on 150 cases, CoSIL reaches 59.3% resolve rate versus 59.7% for the oracle; the comparison uses a fixed downstream patcher and model setup.

**Prospective DynaMap target:** at the same K, line budget, model and downstream patcher, exceed CoSIL on both line recall (>0.80) and F1 (>0.62) while reducing noise-region rate below 0.40 and achieving a statistically supported downstream resolve-rate uplift of at least 2 absolute percentage points over the best reproduced non-oracle baseline. Do not accept recall gains that merely emit whole files or inflate context. Report HitReg, HitFile, Precision, Rec_l, F1, nDCG, FUH, context efficiency and noise alongside downstream resolve rate.

### 4.4 SWE-Time

SWE-Time provides 1,661 cases across 65 repositories (631 retrospective / 1,030 prospective), with strict information cutoffs and future-evidence exclusion. Its 15-agent evaluation is split across Claude Code and Codex harness groups; rankings must not be compared across harness groups as though they were one controlled leaderboard.

Use its temporal isolation methodology and official tasks as an external baseline. Keep EGDDM's DTEA authority/effective-time admissibility cases separate because those are different ground truths and semantics. Report exact harness, agent, prompt, tool interface and inference budget.

### 4.5 RIG/SPADE, Aider, SCIP, rust-analyzer and CodeQL

- RIG/SPADE motivates deterministic build/test graph extraction; its prototype auto-extracts a CMake-focused subset, while other ecosystems in the study use manually authored graph inputs. Do not claim that EGDDM has exceeded RIG based on one Rust prototype.
- Aider RepoMap is a serious lexical/structural retrieval baseline, particularly for trace2code.
- SCIP is a language-agnostic symbol/reference index format; rust-analyzer provides Rust semantic services. A future qualified compiler-semantic lane should compare on identical symbol/reference gold.
- CodeQL is a specialist source/sink and data-flow analysis system. It should be treated as a separate security evidence producer, not as a universal system map.

## 5. DynaMap-specific gold sets and mutation suites

Build four layers of fixtures:

**A. Exact synthetic workspace gold.** Use the existing repository-twin owner (#10173) and frozen synthetic Rust fixtures. Mutations include added/removed modules, reexports, renamed symbols, changed feature gates, target-specific dependencies, build-script declarations, optional dependencies, workspace inheritance, macro-generated declarations, symlink/gitlink substitutions and changed blob/tree identities.

**B. Real repository snapshots.** Select repositories with permissive licenses and sufficiently rich Git history across Rust and at least two additional ecosystems. Freeze exact trees; do not run source-controlled build code to obtain the initial static maps.

**C. Temporal event gold.** Build controlled histories with backdated, late-arriving, corrected, superseded, revoked and future-only evidence. Record valid-time, observed-time, actor-availability time, decision cutoff and correction lineage independently. Include timezone offsets, open interval boundaries, missing completeness and clock skew.

**D. Cross-repository contracts.** Create two-sided schema/API fixtures with version changes, asymmetric upgrades, incompatible feature/config combinations, missing owner receipts, stale verifier builds and deliberate digest mismatch. A compatibility edge cannot be promoted without both endpoint owners' pinned contract identity and an independently passing receipt.

Gold data must include both positive and forbidden cases. A system that returns every file or labels every relationship compatible must fail precision/abstention controls.

## 6. Evaluation methodology

### Retrieval and exploration
- Evaluate exact fixed budgets at 1K, 2K, 4K, 8K and 16K tokens; separately evaluate line-budgeted K=1/3/5/10 exploration.
- Pin tokenizer and count system prompts, graph metadata, snippets and tool outputs consistently.
- Report sample-micro and repository-macro results; do not let one large monorepo dominate.
- Pair all variants on identical cases. Use repository-clustered bootstrap confidence intervals.
- Report candidate generation and reranking costs separately; include indexing cost amortization, latency, peak memory and context overhead.
- Include lexical, dense, RepoMap, hybrid RRF and graph-only ablations.
- Evaluate natural no-gold, wrong-repository, wrong-revision and wrong-configuration abstention separately.

### Temporal admissibility
- Score exact decision-time eligibility, effective-time intervals, observed/available timestamps and source-authority scopes.
- Use planted future, expired, not-yet-effective, incomplete, unauthorized, revoked and superseded evidence controls.
- Report false admission cost separately from missed eligible evidence. A future or unauthorized evidence admission is a critical error, not a small calibration penalty.

### Planning and causality
- Compare generated plans against prerequisite/constraint gold, not generic LLM preference scores.
- Hard constraints are Boolean vetoes; do not average away a physical, legal, safety or authority violation.
- Causal effects require randomized intervention where possible, otherwise a preregistered design with explicit confounder assumptions and sensitivity analysis.
- Use held-out interventions to score predictions. Do not use the same agent to generate a causal claim and adjudicate its own result.
- Track prediction calibration and uncertainty; unknown causal effects remain unknown.

### Action/defer
- Use paired tasks: every valid action case has a matched insufficient-authority/evidence case requiring abstention or escalation.
- Separate harmless information retrieval, reversible local edits, external side effects and irreversible actions.
- Score false-action rate and unauthorized effects as critical safety measures. A high act rate is not desirable if defer cases are misclassified.

## 7. Prospective engineering targets

These targets are research hypotheses and go/no-go thresholds, **not measured EGDDM capabilities**.

| Lane | Target |
|---|---|
| D1 syntax extractor | >=95% precision and recall on an independently labeled Rust syntax gold set |
| D1 source identity | 100% exact match on commit/tree/path/blob identity in all planted mismatch controls |
| D1 complete-tree coverage | 100% projection coverage for files claimed in-scope; partial scope must enumerate omissions |
| D1 compiler semantics | >=90% precision and recall only after an isolated, compiler-backed extractor is independently qualified |
| D2 decision-time evidence | >=95% precision and recall; 100% rejection of planted future/expired/not-yet-effective/unauthorized support |
| D2 event replay | 100% incremental-versus-clean replay equivalence on deterministic event sequences |
| D3 ARB V1 | MRR >=0.270, Recall@20 >=0.670, Gold@8k >=0.280; no task-level macro regression |
| D3 expanded ARB | after exact reproduction: beat MRR 0.2379, Recall@20 0.7029 and BCY@8k 0.3788 on their separate 427-case protocol |
| D3 SWE-Explore | Rec_l >0.80 and F1 >0.62 at matched K=5 budget; NoiseReg <0.40; >=+2 pp downstream resolve uplift vs best reproduced non-oracle |
| D4 impact analysis | >=95% recall and >=80% precision on change-impact gold; publish missed critical edges separately |
| D4 planning | >=95% prerequisite precision/recall; zero hard-constraint violations; 100% detection of deliberately missing critical contracts |
| D5 causal | >=90% precision/recall on independently adjudicated causal-edge gold; held-out intervention effect error beats a preregistered baseline |
| D6 paired act/defer | >=75% paired accuracy in pilot; zero unauthorized external effects; 100% refusal on hard-deny controls |
| Federation | >=95% mismatch recall; zero compatibility promotion without two-sided receipt and exact pinned contracts |
| Evolution outcomes | target +10 percentage points independently accepted capability gain and >=20% lower experiment cost, but only after a powered paired trial |

For every metric publish: count and composition, per-repository results, confidence interval, baseline and system commit, dataset artifact SHA, tokenizer/model/harness, token/tool/time budget, cost, latency, memory, human-review burden and failure examples.

## 8. Required ablations

At minimum compare:

1. lexical only;
2. dense only;
3. Aider/RepoMap-style graph only;
4. deterministic source/manifest graph only;
5. provenance/time filters only;
6. graph + lexical + dense fusion;
7. task-conditional graph/lexical/dense routing;
8. evidence-aware routing with abstention;
9. full EGDDM map with temporal and configuration filters;
10. full DIES planner with act/defer gate.

Remove one component at a time. The full system must show paired uplift against the strongest relevant ablation, not merely against lexical search.

## 9. Reporting and release criteria

A DynaMapBench release must include:
- versioned JSON/JSONL schemas and a human-readable protocol;
- immutable corpus manifest with repository commits, tree and file object IDs;
- public synthetic gold and mutation controls;
- held-out task split or a reproducible evaluation server for licensed/private cases;
- baseline scripts and pinned environments;
- raw per-case outputs and scoring code;
- blinded adjudication record and disagreement analysis;
- confidence intervals and paired significance test;
- explicit unsupported/unknown outcomes;
- safety boundary statement and a record of whether any repository code was executed.

No leaderboard claim is valid if the evaluated commit is not pinned, cases leaked across splits, baselines differ in context or tool budget, or failed/skipped jobs are counted as passes.

## 10. Immediate execution order

1. **D1 composition:** turn the current Cargo/Rust projection + exact tree composer into a reproducible JSON corpus pipeline. First authenticate tree identity and truncation status in the *collector*, not inside the pure composition function.
2. **D1 gold:** extend the existing #10173 synthetic workspace with independent symbol/path/blob gold and adversarial mutations.
3. **D3 reproduce:** freeze Agent Retrieval Bench V1 and SWE-Explore protocols; build data-only evaluators and baseline manifests. Do not download or execute benchmark repositories on an unqualified host.
4. **D2 temporal replay:** add deterministic append/replay equivalence controls to DTEA.
5. **D4/D5 planning:** connect change-impact and causal-debugging owners after D1 and D2 gold is stable.
6. **D6 safety:** add paired act/defer controls under the existing authority boundary.

The first credible result is not a top leaderboard score. It is a reproducible baseline table with an immutable corpus, leakage audit, adversarial controls, honest unknowns and independently checked scoring.


## Source tree inventory collector

The validator now includes a non-networking parser for saved GitHub recursive-tree API responses. `tree-inventory <github-tree.json> <repository> <revision> <tree-oid>` checks that the response root SHA equals the exact tree OID supplied from the pinned commit object, validates each returned path/mode/type/object ID, rejects duplicate paths, and maps the API `truncated` flag to `partial` or `complete`. It does not make requests, authenticate the downloaded JSON or commit signature, prove the caller's commit-to-tree binding, or hash file contents. A non-truncated response is only as trustworthy as the API response and pinning procedure that supplied it.


A tree capture envelope has the form:

```json
{
  "recursive_requested": true,
  "response": {
    "sha": "<exact-root-tree-oid>",
    "truncated": false,
    "tree": [
      {"path": "src/lib.rs", "mode": "100644", "type": "blob", "sha": "<blob-oid>"}
    ]
  }
}
```

The boolean records caller attestation of the request parameter. The saved response itself cannot prove that parameter was sent. The capture process must separately retain the exact commit object and its declared tree OID; the parser checks consistency but does not verify transport authenticity or commit signatures.


## 11. Machine-readable corpus manifest

DynaMapBench v0.1 has an initial JSON Schema at `dynamapbench-manifest-v0.1.schema.json`, a valid illustrative manifest fixture at `../development-map/fixtures/dynamapbench-manifest-v0.1.json`, and a data-only Rust validator exposed through `dynamapbench-manifest <manifest.json>`. The validator enforces unique case IDs, repository-family split isolation, exact Git OID formats, per-case source object identity, evidence availability at/before decision cutoff, future-evidence negative controls, distinct natural/counterfactual no-gold dispositions, source-anchored gold spans, two-review independent adjudication and positive token/line/tool/time budgets. A structural manifest pass does not authenticate the corpus artifact, Git objects, evaluator binary or source content. It does not execute agents or repository code.


The manifest validator requires both repository-family and temporal holdout declarations. Each case pins its event/case time; train/dev cases must be at or before the declared test boundary, while test cases must be after it. This split-time boundary is distinct from the per-case decision cutoff that determines what evidence could have been available to the decision-maker.
