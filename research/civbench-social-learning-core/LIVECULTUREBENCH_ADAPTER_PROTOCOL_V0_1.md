# LiveCultureBench adapter and cross-cultural evaluation protocol v0.1

**Status:** benchmark adapter design plus reusable Rust metric layer; official execution artifacts not yet available in the upstream repository.  
**Reference:** Pham et al. (2026), *LiveCultureBench: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations*, ACL 2026, DOI 10.18653/v1/2026.acl-long.1247.  
**Official paper:** https://aclanthology.org/2026.acl-long.1247/  
**Official repository:** https://github.com/thanhpv2102/LiveCultureBench

## 1. Why this is a distinct CivBench track

LiveCultureBench studies a different but adjacent problem from Reward Network III. It places an LLM target agent in a simulated town represented as a location graph. Synthetic residents have diverse cultural/profile information; the target receives a daily goal and supporting agents provide social context. A separate LLM verifier evaluates task progress and socio-cultural norm violations. The paper focuses on cross-cultural robustness, task/norm trade-offs, and reliability of LLM-based judging.

Reward Network III asks whether a strategy discovered by a machine or person spreads and persists through a population. The exploratory Reward Network ABM asks how discovery, learnability, demonstration advantage and social selection affect persistence. LiveCultureBench asks how an agent behaves in a situated social environment with declared local norms and how reliably that behavior can be evaluated.

These are complementary benchmark families. Their scores must not be pooled into one leaderboard.

## 2. Current upstream availability gate

At the time this protocol was written, the official GitHub repository README contained only the note "To be updated with data download link". The ACL Anthology provides the paper and bibliographic record, but the official repository did not yet expose a usable runner/data interface through its README.

Therefore:

- No official simulator has been executed from this repository.
- No LiveCultureBench dataset or prompt has been imported.
- No model score, verifier score or result from the paper is reproduced here.
- The current Rust module `src/cultural_situation.rs` is only an adapter-neutral metric implementation.
- Do not claim a LiveCultureBench implementation, reproduction or SOTA comparison until the official code/data/prompt release is pinned and run.

## 3. Adapter acceptance gates

When upstream artifacts become available:

1. Pin the ACL paper version, official repository commit, release/archive identifier, dataset digest, task/config files, model/provider IDs, prompt templates and verifier prompt/version.
2. Inspect the official license and data terms before mirroring or transforming any artifacts.
3. Reproduce the authors' original scenario-generation path and baseline metrics before adding new agents or norms.
4. Preserve the small-town location graph, time/step semantics, movement/action rules, resident profiles, daily goals, social support-agent messages and local norm context exactly as supplied.
5. Record target agent, supporting agent and verifier model identities separately. A verifier is not ground truth.
6. Preserve raw simulator traces and raw verifier responses as source artifacts; create derived metrics in separate deterministic transformations.
7. Import the authors' human-annotated verifier calibration/test sets without leakage. Preserve their split IDs and selection rules.
8. Add independent human audits for any new context, norm family, language or region not represented by the original annotations.
9. Bind each result to exact environment, agent, prompt, policy, profile, norm-set and verifier digests.
10. Do not tune prompts, context labels, score thresholds or verifier calibration on the definitive held-out split.

## 4. Metric mapping in CivBench

The Rust layer in `src/cultural_situation.rs` accepts these five episode-level metrics separately:

| Dimension | Direction | Interpretation |
|---|---|---|
| Goal completion | Higher is better | Fraction of declared daily-goal subtasks completed |
| Norm violation rate | Lower is better | Fraction of applicable, context-local norms judged violated |
| Profile faithfulness | Higher is better | Consistency with explicitly supplied agent-profile constraints |
| Contextual awareness | Higher is better | Recognition/use of relevant local context |
| Coherence | Higher is better | Consistency and plausibility across the episode |

The metrics are reported both overall and stratified by declared scenario/context IDs. This is not a license to infer cultural identity from a person's race, ethnicity, religion or other sensitive trait. Context IDs should refer to explicit task conditions, source-defined cultural/norm packages, or opt-in scenario labels.

Never combine goal completion and norm violation into one raw mean. A high task-completion score with high norm violations is not automatically better than a lower-scoring but safer run. Use a Pareto frontier with norm violation minimized and other dimensions maximized.

## 5. Verifier reliability is its own benchmark

Every verifier prediction should carry:

- task and episode ID;
- dimension and declared context ID;
- predicted label and probability/confidence;
- verifier model, prompt and runtime versions;
- whether the target action was judged, not applicable, unknown, or abstained;
- human label, annotator ID/panel and adjudication state where available;
- audit-selection probability or sampling stratum.

For human-audited cases, CivBench reports per-dimension and per-context:

- Brier score and log loss;
- fixed-bin expected calibration error;
- precision, recall, F1 and confusion counts;
- abstention rate and human-audit coverage;
- reliability curves and uncertainty intervals in the eventual adapter.

The V0 Rust metric helper computes per-context Brier, log loss, ECE, confusion metrics, audit coverage and abstention rate. It does not implement conformal risk control or claim to reproduce the paper's verifier.

### Calibration and human oversight

- Keep calibration and final test annotations disjoint.
- Stratify the audit by task dimension, declared context, language/norm family and violation severity.
- Report annotator agreement and adjudication rules.
- Measure false negatives separately for high-severity norm violations.
- Report selective risk as a function of verifier coverage/abstention.
- Do not use overall ECE to hide poor calibration in a smaller context group.
- Use confidence intervals and report group sample sizes; small or absent groups must remain explicitly underpowered/unevaluated.
- LLM-verifier outputs are claims requiring audit, not truth labels.

## 6. CivBench paradigm extensions

After the official benchmark is reproduced, add separate, controlled conditions:

1. **No-context baseline:** target sees the task but not the local norm/context package.
2. **Context-only baseline:** target receives norms and local context, but no inherited skill.
3. **Skill-only baseline:** target receives a skill capsule but not the culture-specific operating envelope.
4. **Scoped proof-carrying skill:** capsule binds preconditions, allowed actions, local norms, exclusions and evidence receipts.
5. **Plural-norm condition:** multiple declared norm packages may legitimately conflict; the agent must recognize conflict, seek clarification or abstain rather than silently collapse them.
6. **Teacher-removal condition:** target must retain and execute the skill after the originating demonstrator and support agents disappear.
7. **Corrupted-context condition:** inject stale, contradictory or out-of-scope norms and measure detection, false adoption and correction.
8. **Cross-context transfer:** hold out location graph, norm package, agent profile family and goal structure separately. Do not leak near-duplicate contexts across splits.
9. **Cultural-lineage condition:** transmit a scoped capsule through several generations, retain its source lineage, and test whether each descendant preserves its limits and can reject invalid transfers.
10. **Verifier independence:** evaluate with a second verifier model and a human panel; measure whether conclusions change with judge identity.

A high norm-adherence rate alone is not proof of understanding. Measure task outcome, context recognition, profile fidelity, correct abstention, teach-back and held-out transfer separately.

## 7. Proposed benchmark gates

The following are engineering targets for pilot calibration and pre-registration, not achieved results or claims about the paper:

| Gate | Proposed measure | Initial target / requirement |
|---|---|---|
| Artifact reproducibility | Exact scenario and prompt rerun | 100% identical task/config digests; stochastic traces reproduce under pinned seed and runtime where the simulator promises determinism |
| Action validity | Legal action and environment-contract violations | 0 invalid actions in deterministic smoke fixtures; report rates on stochastic tests |
| Norm safety | High-severity false-negative rate | Report per severity and context; freeze a risk threshold only after human calibration |
| Verifier calibration | Brier, log loss, ECE, reliability plots | Proposed pilot gate ECE ≤0.05 where human-audit n is adequate; do not impose on underpowered strata |
| Verifier selectivity | Risk versus coverage | Report full curve; no single operating point without an explicit risk budget |
| Cross-context robustness | Worst-context performance and dispersion | Report worst-case and distribution; do not hide context regressions behind a global mean |
| Task/norm trade-off | Pareto dominance/frontier | No universal scalar score; publish goal completion against norm violations |
| Skill transfer | Held-out location/norm/profile/task families | Beat matched no-skill/context-only baselines with a pre-registered interval excluding zero, or report no gain |
| Cultural sovereignty | Challenge/refusal/correction/alternative retention | Learner must retain the ability to challenge, refuse, fork or request evidence; adoption rate is not the objective |
| Cost | Total tokens, latency, tool calls, memory and estimated cost | Report per successful task and per audited-safe task |

## 8. Fair comparison to the field

LiveCultureBench compares LLM agents operating in a situated social simulation. Reward Network III measures human population transmission of machine-discovered strategies. DeepMind MEDAL-ADR examines few-shot sensorimotor imitation and held-out task transfer. CivBench should compare the appropriate systems on each task family and resist a single composite ranking.

The likely differentiator for Symthaea is not a claim that HDC or LTC automatically beats an LLM. It is the combination of:

- explicit scoped capabilities and knowledge lineage;
- context-sensitive retrieval rather than global prestige alone;
- separate task, safety, verifier and transfer metrics;
- knowledge that can be challenged, corrected, forked and revoked;
- retained skills that still work after the teacher is removed;
- independent verification bound to the exact policy and context digests.

That combination is a hypothesis to test. It becomes a superiority claim only after exact upstream reproduction and matched held-out experiments.

## 9. Integration boundaries

- **Symthaea:** situational reasoning, memory, skill retrieval, teach-back, action selection and abstention.
- **Mycelix/shared evidence plane:** source provenance, identity assertions, signatures, dispute and revocation history.
- **CivBench:** scenario adapters, split policy, scoring, human-audit protocol, verifier calibration and independent qualification.
- **Symtropy:** synthetic population/network stress tests and multi-generation social simulations.
- **Sol-Atlas:** optional view of task context, norm scope, capability lineage and evaluation frontier; visualization is not validation.

The LiveCultureBench adapter should import the official environment rather than silently replace it with a CivBench lookalike. The Rust metric layer is reusable but does not claim official task compatibility until the upstream schemas are available.
