# IG-PHANTOM-CONSENSUS-001 v0.1

**Status:** synthetic fixture design / structural qualification only  
**Owning issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11682  
**Research reference:** https://arxiv.org/abs/2611.08971

## Research question

Does a system's estimate of what peers privately judged match the independently recorded private baseline, and do ambiguous public signals cause public decisions to diverge from that baseline?

This is distinct from:
- self-confidence calibration (#11367);
- an agent's own stance transition (#11320);
- evidence-carrying revision provenance (#11340);
- communication topology and actual message delivery (#11337).

This artifact composes with those owners; it does not implement a new agent runtime or graph engine.

## Files

- PHANTOM_CONSENSUS_PROTOCOL_V0_1.json and its schema: four-stage experiment contract, intervention arms, metrics and claim ceiling.
- PHANTOM_CONSENSUS_FIXTURES_V0_1.json and its schema: 12 deterministic synthetic controls, including a phantom-consensus case, transparent recovery controls, fabricated tally, missing record, duplicate identity, abstention/tie and normative-pluralism case.
- qualifier/: standalone Rust structural and oracle validator.
- .github/workflows/ig-phantom-consensus-qualifier.yml: path-filtered exact-revision workflow.

## Local qualification

From the repository root:

cargo test --manifest-path research/ig-phantom-consensus-001/qualifier/Cargo.toml

cargo run --manifest-path research/ig-phantom-consensus-001/qualifier/Cargo.toml -- \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_PROTOCOL_V0_1.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_PROTOCOL_V0_1.schema.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_FIXTURES_V0_1.json \
  research/ig-phantom-consensus-001/PHANTOM_CONSENSUS_FIXTURES_V0_1.schema.json

The qualifier checks a bounded schema header and explicit structural invariants. It does not claim complete JSON Schema implementation. It computes peer distributions from the recorded baseline, checks probability-vector completeness, calculates squared L2 probability-vector error against the recorded empirical peer distribution, and verifies the expected phantom-consensus oracle. Baseline and intervention payload digests are emitted for reproducible run receipts.

## Measurement semantics

- Each private baseline row is a recorded elicited output, not hidden model state.
- Peer estimates exclude the estimating agent; the denominator includes missing records as UNKNOWN and abstentions as ABSTAIN.
- ABSTAIN is a recorded response excluded by the declared aggregation rule. UNKNOWN is a missing private record: it remains in the scoring denominator and blocks inference of the complete private plurality.
- A phantom-consensus oracle is true only when an agent's estimated unique peer plurality differs from the unique plurality in the other agents' recorded private baseline. Ties / no unique plurality do not become an artificial majority.
- Verifiable task accuracy is separate from peer-belief accuracy.
- Normative-pluralism tasks have no gold preference. The benchmark checks representation, not agreement with a preferred policy.
- Transparency arms reveal genuine recorded private judgments but never the answer key.

## Implemented versus planned metrics

The protocol names eight outcome families, but v0.1 computes three synthetic fixture-level quantities:

- `peer_private_belief_l2_error`: squared L2 error between a fixture's estimated peer-choice probability vector and the empirical peer distribution computed from the committed private baseline.
- `phantom_consensus_incidence`: a bounded known-answer oracle indicating whether any agent's estimated unique peer plurality differs from the unique recorded peer plurality.
- `synthetic_transparency_recovery_l2_improvement`: a paired fixture-level check that the peer-belief error decreases after authentic tally/ledger disclosure in the two frozen recovery controls. This is not a causal effect estimate from model-agent runs.

The remaining six metrics are research targets, not implemented results in this slice: private/public decision gap, collective-rationality failure, model-agent transparency-recovery effect, signal susceptibility, pluralism preservation and full run-validity reporting. The fixture validator checks selected structural invariants for those protocols but does not claim to estimate their causal effects. Future work must not report these as benchmark outputs until their scoring and analysis implementations exist.

## Claim ceiling

A successful synthetic qualification establishes only that these 12 fixture contracts and bounded oracles behave as declared. No LLM agents were executed by this corpus. No PHANTOM-Bench reproduction, human effect, democratic legitimacy, public-opinion simulation or deployment claim is made. Model evaluation requires exact artifact/method review, matched baselines, pinned prompts/models, randomized treatment blocks, multiple seeds, held-out tasks and independent analysis.

## Industry and research crosswalk

This benchmark should be evaluated as an **assurance and measurement layer**, not as a replacement deliberation interface.

- **PHANTOM-Bench (arXiv:2611.08971, 6 November 2026)** is the direct method comparator: isolated private judgments, ambiguous public signals, elicited peer-private estimates, transparency recovery, neutral-text and explicit-vote controls. First reproduce its method and controls where artifacts permit; do not inherit its reported numbers as our baseline.
- **Misalignment Contagion (COLM 2026)** is the adjacent comparator for persistent public/private stance shifts and communication topology. Compose its causal design through Symthaea #11320 rather than duplicate it.
- **Polis** is a large-scale opinion-mapping workflow: participants submit short statements and vote agree/disagree/pass; opinion groups are inferred from voting patterns. CivOS should complement such workflows with independent evidence, exposure, authority and outcome records rather than recreate opinion clustering.
- **Decidim** provides participatory spaces and components including proposals, amendments, votes, meetings, results and an accountability component that links proposals to manually managed implementation results, milestones and progress. CivOS's opportunity is machine-verifiable, bitemporal evidence and independent execution/appeal receipts that can interoperate with these records.
- **Loomio** provides discussion, polls and proposal templates (sense-check, advice, consent and consensus), with published outcomes and review dates. It is a workflow competitor, not a substitute for the independent qualification layer.
- **Stanford's Online Deliberation Platform** supplies structured small-group video discussions, speaker queues/time limits, agenda management and abuse prevention; Stanford describes operation across more than 40 languages and 40+ countries as of fall 2026. This is a meaningful participation/facilitation comparator. CivOS should evaluate whether facilitation preserves private voice and epistemic accuracy, not just whether turns are distributed.

Primary product/method references:
- PHANTOM-Bench: https://arxiv.org/abs/2611.08971
- Polis overview: https://knowledge-for-policy.ec.europa.eu/participatory-deliberative-democracy/ldi/polis_en
- Decidim accountability: https://docs.decidim.org/en/develop/admin/components/accountability.html
- Decidim components: https://docs.decidim.org/en/develop/features/components.html
- Loomio proposals and polls: https://www.loomio.com/docs/en/user_manual/polls/intro_to_decisions
- Stanford Online Deliberation Platform: https://deliberation.stanford.edu/tools-resources/online-deliberation-platform

### Direct benchmark and methodological comparators

| Comparator | What it measures | How Symthaea / CivOS should compare |
|---|---|---|
| [PHANTOM-Bench (6 Nov 2026)](https://arxiv.org/abs/2611.08971) | Private judgments, ambiguous public signals, estimated peer-private beliefs, transparency recovery, and neutral / explicit-vote controls. | Exact methodological reproduction first. Then extend with missingness, abstention, authentic roster provenance, communication topology, pluralistic tasks and independently replayable receipts. |
| [Beyond Correctness (ICML 2026)](https://proceedings.mlr.press/v306/ha26a.html) | Direction and magnitude of solution revision during multi-agent debate, including regression from correct answers under social pressure. | Add revision-distance and warranted-update outcomes by composing #11320 and #11340. Do not treat peer-belief accuracy as a substitute for reasoning correctness. |
| [Free-MAD (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1600/) | Consensus-free debate, whole-trajectory scoring and anti-conformity; reports results across eight reasoning datasets and reduced communication cost. | Include as a decision-method baseline on matched verifiable tasks, tracking token/latency cost and full-trajectory evidence. It does not by itself measure actual private peer beliefs or human legitimacy. |
| [Demystifying Multi-Agent Debate (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1694/) | Initial perspective diversity and confidence-modulated debate updates across reasoning QA benchmarks. | Compare diversity and confidence-aware interventions, while keeping individual confidence (#11367) distinct from estimates about peers' private choices (#11682). |
| [Misalignment Contagion (COLM 2026)](https://proceedings.colm.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html) | Persistent stance shifts, public/private re-elicitation and communication-topology effects. | Compose through #11320; preserve evidence-supported revision as a positive outcome and test topology without transferring authority semantics. |
| [DeliberationBank / DeliberationJudge (COLM 2026)](https://arxiv.org/abs/2510.05154) | Participant-conditioned opinion-summary judgments on representativeness, informativeness, neutrality and policy approval. | Use for the Mycelix #6011 summary track, with independent rights/consent and checkpoint-integrity gates. It is not a group-intervention benchmark. |
| [Real-Time Group Dynamics with LLM Facilitation (FAccT 2026)](https://deepmind.google/research/publications/224297/) | Two human-group experiments (N=879) on facilitation preference, allocation influence, consensus and participation equity. | Human-reference comparator for Mycelix; keep perceived satisfaction, trust, real allocation outcomes and participation equality separate. |

These are complementary evidence types, not interchangeable leaderboard rows. PHANTOM-Bench is the closest same-task reference; Free-MAD and confidence/diversity-aware debate are candidate decision algorithms; Beyond Correctness and Misalignment Contagion inform trajectory metrics; DeliberationBank informs participant-conditioned voice fidelity; FAccT informs human-group causal outcomes.

## What "better than SOTA" should mean here

Do not claim superiority from synthetic fixtures or a lower aggregate agreement error. The confirmatory comparison should include:

1. **Exact reproduction:** recreate PHANTOM-Bench's task, signals, stages, prompts, models and scoring wherever the source artifacts and terms permit; publish artifact gaps and any deviations.
2. **Matched extensions:** add peer-roster integrity, explicit UNKNOWN/ABSTAIN handling, communication-topology and centrality contrasts, evidence-bearing corrections, and separate normative-pluralism tasks.
3. **Intervention efficacy:** test whether anonymous tally or peer-ledger disclosure repairs peer-belief estimates without exposing the answer key; distinguish improved social inference from answer leakage.
4. **Safety non-regression:** verify improved social-belief calibration does not reduce warranted evidence updates, increase stubbornness, erase minority positions, or distort confidence calibration.
5. **Robustness and cost:** report model-family × task × signal × topology strata, multiple seeds, confidence intervals, worst-condition outcomes, token/latency/compute budgets and abstention coverage.

A defensible improvement is a better calibrated estimate of peers' recorded private positions and reduced ambiguity-induced decision degradation **at matched task coverage and compute**, with no loss of evidence responsiveness or normative pluralism. Set numeric effect-size targets only after reproducing the reference and conducting power analysis.

