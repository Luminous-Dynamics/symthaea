# Broca: Language, Literature, Creativity, and Philosophical Reasoning
## Evidence-first research charter — 2026-11-18

**Status:** research and evaluation plan; targets below are proposed, not achieved results.  
**Scope:** public Symthaea/Broca work.  
**Principle:** distinguish implemented, structurally inspected, locally tested, CI-tested, and independently evaluated claims.

## Executive thesis

Broca should not be positioned as a smaller imitation of a frontier transformer. The research opportunity is a **meaning-centred, evidence-carrying, multilingual creative system** that combines:

- a pretrained sequence model where it is useful;
- Symthaea's HDC representations and CfC/LTC temporal dynamics as candidate semantic-state mechanisms;
- explicit discourse, narrative, argument, and provenance structures;
- retrieval of licensed source material with evidence spans;
- human-led evaluation for literature, poetry, interpretation, and cultural adequacy.

This is a research hypothesis, not evidence that HDC or continuous-time dynamics already outperform transformers. Each component must earn its place through matched ablations.

## 1. Source inspection: current engineering facts

The following observations were made against public repository head `43a97f3f8857c12b87ce619c3f772bd338b14ec3` on 2026-11-18.

### 1.1 Supervision defect corrected in this branch

In `crates/domains/symthaea-broca/src/bin/broca_projection_train.rs`, the batch loop generated a sequence and passed `result.token_ids` into `local_fep_distill`. That made the projection's target the model's own sampled completion rather than the pair's supplied `target_text`.

The branch changes this path to:
1. encode `pair.target_text` with the Mamba backend's own tokenizer;
2. reject empty or unencodable targets;
3. reset recurrent Mamba state before the teacher-forced target pass;
4. distill against those teacher token IDs.

It intentionally does **not** trust `TrainingPair::target_ids`, because that field may be populated using Broca's separate BPE vocabulary. Three small regression tests cover text forwarding, empty tokenizer output, and whitespace-only input.

**Important limit:** this repairs which token sequence supervises the local projection update. It does not establish that the complete HDC-to-Mamba training objective is optimal, that the projection has converged, or that generation quality improved. The exact-head build/tests are still required.

### 1.2 Existing work to reuse rather than duplicate

- `training.rs` already has `TrainingPair`, `target_text`, tokenization helpers, and a CfC/HDC BPTT path.
- `liquid_mamba.rs` contains Mamba distillation, temporal projection, and local predictive-coding updates.
- `broca_poetry_eval.rs` already generates haiku/tanka/free-verse samples, checks form constraints, and optionally uses a local LLM judge. Its own source correctly warns that the judge has not been validated against human aesthetic ratings.
- `creative_mode.rs` has poem forms and mechanical validation.
- `evaluation.rs` contains perplexity, diversity, target overlap, intent slices, and collapse diagnostics.
- The built-in Broca tokenizer describes itself as a minimal ASCII/common-English tokenizer with byte fallback. A real multilingual programme must test the actual encoding path, not assume that byte fallback equals competent multilingual language modelling.

### 1.3 Immediate audit questions

These are next investigations, not conclusions:
- Verify that the HDC projection output is actually injected into the teacher-forced sequence in the intended order.
- Audit reset/state semantics for all teacher-forcing and generation paths, including negative controls.
- Confirm that all datasets record tokenizer identity, vocabulary digest, normalization policy, language, source license, and split provenance.
- Test whether each named evaluation metric changes under a known positive and negative control.
- Separate projection reconstruction error from language-model likelihood, target adherence, and human-perceived quality.

## 2. Paradigm shifts worth testing

### A. From prompt context to persistent semantic state

Treat a conversation as a versioned semantic state, not just a long string. Maintain:
- entities and referents;
- claims, evidence, uncertainty, and provenance;
- goals and constraints;
- temporal relations and discourse commitments;
- alternative interpretations.

HDC could provide compact distributed representations; CfC/LTC could model temporal evolution. A symbolic or graph layer should preserve exact propositions and evidence where vector similarity is insufficient. Evaluate whether this hybrid state improves long-horizon consistency over a prompt-only baseline.

### B. From one correct reading to evidence-supported plural interpretation

Literature and philosophy often admit multiple defensible readings. Broca should return a small set of interpretations, each with:
- textual evidence spans;
- assumptions and interpretive lens;
- counter-reading;
- confidence and unresolved ambiguity.

Do not reward confident single-answer collapse where the text genuinely underdetermines the answer. Score evidence coverage, interpretive coherence, and recognition of ambiguity separately.

### C. From next-token imitation to a compose–criticize–revise process

Creative generation should be a controlled search over drafts:
1. represent intent, form, audience, voice, and constraints;
2. generate candidate structures and lines;
3. evaluate meter, sound, imagery, novelty, and coherence independently;
4. revise only the dimensions that need revision;
5. preserve authorial intent and the human's right to reject every suggestion.

A critic must not be the same uncalibrated model used as the sole judge of its own output.

### D. From English-first NLP to language-family-aware composition

Do not treat South African languages as English with words swapped in. Model language-specific morphology, orthography, idiom, code-switching, register, and oral/literary forms. Start with English, Afrikaans, isiZulu, isiXhosa, and Sesotho, while designing the dataset and schema to extend to all 11 South African official written languages.

Use byte/grapheme fallbacks as safety nets, not as proof of morphological or semantic competence. Measure round-trip fidelity, morphology-sensitive generalization, and native-speaker preference.

### E. From benchmark scores to reproducible evidence

Every result should carry:
- exact repository commit and checkpoint digest;
- model/tokenizer identity and vocabulary digest;
- dataset version, license, split and contamination policy;
- prompt, decoding settings, seeds, and tool permissions;
- hardware, latency, tokens/sec, peak memory, and estimated energy;
- raw predictions, scoring code, judge identities/roles, and uncertainty intervals.

A queued, skipped, or missing CI job is not a pass. A fluent sample is not a benchmark result.

## 3. Benchmark landscape and what it tells us

Use existing suites as anchors; create new tasks only where a documented gap remains.

| Capability | Existing evaluation anchor | What it contributes | Broca extension |
|---|---|---|---|
| General language/reasoning | MMLU-Pro, GPQA Diamond, LiveBench, HELM | Broad capability sanity checks and comparable baselines | Report separately; these do not establish literary or creative competence |
| Long narrative | LiteraryQA, NarrativeQA | Literary long-document question answering; LiteraryQA improves NarrativeQA's noisy literary subset | Evidence-grounded plot, character, chronology, and causal questions |
| Narrative structure | NarraBench, LitVISTA, Too Long, Didn't Model (TLDM) | Narrative task taxonomy, orchestration, events, perspective, long-range dependencies | Track focalization, revelation, pacing, narrative time, and global structure |
| Creative writing | LitBench | Pairwise story preference data and reward-model evaluation | Poetry-specific human evaluation; calibrated critic; author revision utility |
| Argumentation | ArgBench | Broad argument mining, quality assessment, perspective, reasoning, and generation | Philosophy-specific reconstruction, objection, reply, and steelmanning |
| Conceptual reasoning | Conceptual Reasoning Index (CRI): LMCA, ACCoRD, DTBench | Expert-rated arguments where empirical ground truth may be unavailable | Multi-perspective philosophical reasoning and explicit uncertainty |
| African languages | NGLUEni, AfroBench, MzansiBench, plus relevant African-language task datasets | Existing language-family and South African evaluation infrastructure | Add literature, idiom, morphology, code-switching, and native-speaker review |

### Key findings from the literature

- **NarraBench (EACL 2026)** surveys 78 narrative benchmarks and estimates that only 27% of narrative tasks are well captured by existing evaluations. Narrative events, style, perspective, and revelation are among the under-covered areas.
- **LiteraryQA (EMNLP 2025)** finds weaknesses in the quality of some NarrativeQA examples and reports that n-gram metrics correlate poorly with human system rankings on its literary subset.
- **LitVISTA (ACL 2026)** finds that frontier models still struggle to jointly capture narrative function and structure; even oracle event anchors leave important narrative orchestration failures visible.
- **Too Long, Didn't Model (2026)** reports unstable novel understanding beyond 64k tokens for the seven frontier models tested. Long context capacity should not be confused with stable global narrative comprehension.
- **LitBench (EACL 2026)** reports 73% human-preference agreement for its strongest off-the-shelf LLM judge, versus 78% for specialized reward models. Automated judges need calibration; they are not human ground truth.
- **ArgBench (Findings of ACL 2026)** unifies 33 datasets across 46 computational argumentation tasks.
- **Anthropic's 2026 Conceptual Reasoning Index** provides a useful example of expert-rated argument evaluation for questions with limited empirical feedback.
- **NGLUEni** covers four Nguni languages across six tasks and eleven datasets. The 2025 AAAI African-language study reports measurable gaps and gains from language-specific data and cross-lingual transfer.
- **MzansiBench (UCT 2026 project)** is an especially relevant emerging evaluation resource: 13 tasks and 16 datasets across South Africa's 11 official written languages. Verify its repository, data licenses, and exact protocol before adopting it as a formal baseline.

## 4. Proposed Broca benchmark suite

Create a versioned suite with six independently reported tracks. Keep training, development, public test, and private challenge items disjoint at the work/author/source-family level.

### Track L — Language grounding and composition

Tasks:
- paraphrase under meaning preservation;
- ambiguity detection and resolution with context;
- compositional generalization to unseen combinations;
- contradiction and entailment;
- register and audience adaptation;
- instruction/intent following without loss of constraints.

Metrics: exact task accuracy where applicable, semantic consistency, constraint satisfaction, calibration, and evidence precision. Keep perplexity as a diagnostic, not the headline quality metric.

### Track N — Narrative and literary understanding

Tasks:
- event ordering and causal dependency;
- character belief versus narrator belief;
- focalization and perspective shifts;
- temporal ellipsis and unreliable narration;
- motif and symbol tracking across a full work;
- competing interpretations supported by evidence;
- revelation and pacing structure.

Metrics: evidence-span precision/recall, event-graph F1, temporal relation accuracy, interpretation coverage, contradiction rate, and blind expert ratings.

### Track P — Poetry and formal craft

Start with English forms already supported by Broca, then add language-specific forms only with competent native-language annotation.

Tasks:
- constrained composition;
- meter/syllable and line-form adherence;
- sound, cadence, repetition, and enjambment;
- imagery and metaphor interpretation;
- revision under a specific editorial request;
- collaborative continuation that preserves a human-authored voice.

Metrics: formal compliance, blinded pairwise preference, originality (with duplicate/near-duplicate checks), revision usefulness, and human ratings of craft and resonance. Do not use BLEU or a single LLM score as a substitute for aesthetic evaluation.

### Track A — Philosophy and argumentation

Tasks:
- reconstruct an argument into premises and conclusion;
- identify hidden assumptions and equivocations;
- formulate the strongest charitable objection;
- steelman opposing views;
- distinguish empirical, conceptual, normative, and metaphysical claims;
- preserve disagreements between schools of thought;
- reason through counterfactuals and thought experiments.

Metrics: premise/claim coverage, inference validity, objection relevance, steelman fidelity, source fidelity, calibration, and blinded expert ratings. Report agreement and disagreement between human evaluators.

### Track M — Multilingual and cultural competence

Evaluate English, Afrikaans, isiZulu, isiXhosa, and Sesotho first, with a path to all 11 official written languages.

Tasks: translation, QA, summarization, morphology-sensitive minimal pairs, idiom, code-switching, register, cultural references, literary passage analysis, and original composition.

Metrics: per-language scores, worst-language score, macro-average, English-to-language parity ratio, code-switching fidelity, and native-speaker preference. Never hide weak-language results inside a macro-average.

### Track D — Dialogue and collaboration

Tasks:
- long-horizon reference tracking;
- stable commitments and correction of mistaken memories;
- collaborative drafting over multiple turns;
- response to critique without losing author intent;
- graceful uncertainty and disagreement;
- distinguish what the system knows, infers, retrieves, and invents.

Metrics: task completion, consistency, correction success, user-rated agency, revision acceptance, and unsupported-claim rate.

## 5. Proposed targets — explicitly aspirational

Do not present these as current results. Establish a measured baseline first, then pursue the targets below on held-out data.

| Measure | First target | Stretch target |
|---|---:|---:|
| Mamba projection supervision | 100% of accepted examples encode from `target_text`; 0 empty targets accepted | Tokenizer/split/provenance validated for every sample |
| Formal poetry constraints | ≥90% compliance on small, explicitly supported forms | ≥97% with human review of false positives |
| Literary evidence attribution | ≥90% precision on supported evidence spans | ≥95% precision and ≥90% recall |
| Argument reconstruction | ≥85% expert-rated completeness on a narrow rubric | Match or exceed best matched open baseline |
| Human pairwise creative preference | Statistically non-inferior to the best matched small/open baseline | >55% win rate against that baseline with 95% CI above 50% |
| Judge reliability | Establish inter-rater agreement and judge-vs-human correlation | ≥0.80 pairwise agreement on a held-out, human-rated set |
| Language parity | Publish results for every language, no hidden missingness | Reduce the baseline English/low-resource gap by ≥25% relative |
| Dialogue consistency | 30-turn test with explicit correction probes | 100-turn tests with measured degradation curve |

The targets are deliberately track-specific. Beating a general-purpose frontier model on an easy or tiny custom slice would not establish superiority. A result counts as a meaningful win only with matched prompts, budgets, access to tools, decoding rules, data splits, and statistically supported comparisons.

## 6. Comparison protocol: how to compete honestly with industry

Run at least four baselines:
1. Broca's current native CfC/HDC generator, without Mamba.
2. Mamba teacher alone, with no HDC projection conditioning.
3. The complete hybrid Broca path.
4. Strong external models: one or more open-weight models in the feasible compute band and frontier API models under an identical prompt/evaluation contract.

Required ablations:
- remove HDC conditioning;
- remove temporal projection;
- remove retrieval/evidence structures;
- remove the critic/revision stage;
- compare one-shot versus multi-candidate revision;
- compare language-specific tokenization/normalization choices;
- compare teacher-forced training with the corrected supervised path.

Reporting:
- exact versioned data and code;
- at least three seeds where stochastic training is involved;
- confidence intervals and paired significance tests;
- clustered resampling by book/author/source, not by adjacent passage;
- native-speaker and domain-expert review for language and philosophy tracks;
- API cost, wall-clock latency, tokens/sec, peak RAM/VRAM, and estimated energy;
- all refusals, missing judge outputs, parsing failures, and unscored cases reported explicitly.

Do not collapse these tracks into one league score. A model can win language correctness and lose literary perspective-taking; that distinction is information, not an inconvenience.

## 7. Immediate execution plan

### P0 — repair the learning signal
- [x] Change projection training to encode the supplied `target_text` with the Mamba teacher tokenizer.
- [x] Reject empty/whitespace-only targets and empty tokenization.
- [x] Reset recurrent teacher state before teacher-forced target passes.
- [x] Add small regression tests for those invariants.
- [ ] Run formatter, compile the `mamba-cpu` binary, and run the new tests on the exact branch head.
- [ ] Run a tiny controlled training/evaluation fixture and compare before/after using frozen inputs.

### P1 — establish measurement integrity
- Add a machine-readable benchmark manifest with dataset/version/license/split/metric metadata.
- Audit every relational test for a real positive and negative control.
- Store raw predictions and complete run receipts.
- Ensure the same work/author never appears across training and test splits.

### P2 — build the language and culture evaluation corpus
- Start from licensed, published task sets (NGLUEni, AfroBench, MzansiBench where license permits).
- Recruit native-language reviewers before scaling.
- Add original, consented literature and poetry examples with rights and attribution recorded.
- Create a private challenge split to reduce benchmark contamination.

### P3 — narrative, poetry, and philosophy
- Add evidence-span and event-graph evaluation to the existing poetry and narrative tools.
- Calibrate the poetry judge against blind human pairwise ratings.
- Add argument reconstruction and competing-interpretation datasets.
- Track whether model revisions actually improve expert and author preference.

### P4 — publish matched comparisons
- Freeze code, checkpoints, splits, prompts, and evaluation container.
- Run the full baseline matrix, disclose all resource use, and release results with receipts.
- Claim superiority only on tracks where the evidence supports it.

## 8. Research references

1. NarraBench, EACL 2026: https://aclanthology.org/2026.eacl-long.176/
2. LiteraryQA, EMNLP 2025: https://aclanthology.org/2025.emnlp-main.1729/
3. LitVISTA, ACL 2026: https://aclanthology.org/2026.acl-long.1024/
4. Too Long, Didn't Model, 2026: https://aclanthology.org/2026.latechclfl-1.28/
5. LitBench, EACL 2026: https://aclanthology.org/2026.eacl-long.362/
6. ArgBench, Findings of ACL 2026: https://aclanthology.org/2026.findings-acl.1099/
7. Conceptual Reasoning Index, 2026: https://alignment.anthropic.com/2026/conceptual-reasoning-index/
8. NGLUEni, LREC-COLING 2024: https://aclanthology.org/2024.lrec-main.1071/
9. African low-resource language benchmarks, AAAI 2025: https://mlanthology.org/aaai/2025/alhanai2025aaai-bridging/
10. MzansiBench, UCT 2026 project: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
11. AI Index Report 2026: https://hai.stanford.edu/ai-index/2026-ai-index-report

---

**Decision rule:** the next paradigm shift is only real when it changes an independently measured outcome. Symthaea should be ambitious in its hypotheses and unusually strict about its evidence.


## 9. Execution update — 2026-11-18 (source changes, tests pending)

### Teacher-target and temporal supervision
The research branch now changes `broca_projection_train` to encode each supplied `target_text` with the Mamba teacher tokenizer. It rejects empty/whitespace-only targets and zero-token encodings. Cached `TrainingPair::target_ids` are not reused because they can belong to Broca's independent BPE vocabulary.

The local FEP temporal path now:
- starts teacher forcing from a reset Mamba state;
- records one teacher hidden state per target position;
- aligns the teacher trajectory to the temporal projection chunk count using endpoint-preserving linear interpolation;
- validates every teacher state's dimension and finiteness;
- checks that prediction and observation lengths match before chunk updates;
- rejects empty target sequences.

This interpolation is a **shape-alignment hypothesis**, not a validated semantic alignment. It preserves first/last teacher states and samples intermediate states evenly. The benchmark work must determine whether this is useful compared with alternatives such as average pooling, learned monotonic alignment, cross-attention, or a global pooled target. It must not be described as established temporal correspondence.

Regression coverage authored:
- target text is the actual text passed to the teacher tokenizer;
- whitespace-only and empty tokenizer outputs fail closed;
- one-chunk and one-token alignment;
- 64-chunk output shape and endpoint preservation;
- a reversed-time negative control;
- malformed/non-finite trajectories;
- offline mock-backend local-FEP temporal integration and empty-target rejection.

### Original Broca literary/philosophy seed
Added `data/evals/broca-literary-philosophy-seed-v1.jsonl` with six original-authored, CC0-1.0 development fixtures: three literary interpretation cases and three philosophical argument cases.

Added the Rust CLI `broca-culture-eval` and registered it in `Cargo.toml`.

Example dataset-only validation:

```bash
cargo run -p symthaea-broca --bin broca-culture-eval -- \
  --dataset crates/domains/symthaea-broca/data/evals/broca-literary-philosophy-seed-v1.jsonl
```

Example scored predictions:

```bash
cargo run -p symthaea-broca --bin broca-culture-eval -- \
  --dataset crates/domains/symthaea-broca/data/evals/broca-literary-philosophy-seed-v1.jsonl \
  --predictions predictions.jsonl --output report.json
```

The scorer reports exact quote grounding, gold evidence-anchor recall, claim-ID precision/recall/F1, acceptable interpretation coverage, and ambiguity accuracy. These are **structural checks only**. They do not measure prose quality, originality, truth, or expert philosophical adequacy. Six cases are a smoke-test seed, not a benchmark population, and must not be used for SOTA claims.

### Updated industry comparison reference
The live Conceptual Reasoning Index page, retrieved 2026-11-18, currently lists Claude Opus 5.5 at 77.9, GPT-6 Astra at 73.8, and Gemini 4 Argon at 72.3, followed by Muse Wonder at 67.7. These are live leaderboard snapshot values, not immutable benchmark releases. CRI evaluates expert-rated conceptual argument quality (LMCA), selected logical-consistency constraints (ACCoRD), and decision-theory cases (DTBench). A Broca comparison must pin the exact prompt, model version, benchmark data release, token/effort budget, refusals, and confidence intervals before drawing conclusions.

### Current evidence status
- Source changes and tests have been authored on the research branch.
- No Rust compilation, formatter, unit test, integration test, training run, or SOTA comparison has been completed in this environment.
- The latest GitHub Actions workflow runs exposed for the branch report `skipped` for CI, Broca Measurements, Broca Feature Matrix, and Symthaea Benchmark Suite. `skipped` is not pass evidence.
- The branch is currently two commits behind `main`; the upstream movement is unrelated to Broca, but the PR base/mergeability must be refreshed before merging.
- The temporal-alignment issue remains open until exact-head tests execute and a reviewer accepts the alignment objective.

No benchmark score or quality improvement is claimed by these changes.
