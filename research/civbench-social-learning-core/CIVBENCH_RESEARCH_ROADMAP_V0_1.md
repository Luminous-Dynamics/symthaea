# CivBench research roadmap v0.1: from diffusion to reliable cumulative culture

**Status:** research and engineering roadmap. All numerical targets below are proposed gates, not achieved results.  
**Principle:** compare exact tasks and budgets; do not merge unlike studies into a single leaderboard.

## Executive thesis

The next paradigm shift is to stop treating “social learning” as a single event. Reliable cultural accumulation is a chain of distinct operations:

1. **Discover** a useful capability.
2. **Represent** it in a way that another agent can retrieve and reconstruct.
3. **Select** a demonstrator using evidence, not prestige alone.
4. **Observe** a demonstration or receive a capsule.
5. **Acquire** the information.
6. **Integrate** it into active behaviour without erasing better prior knowledge.
7. **Retain** it through memory pressure, teacher removal and generational change.
8. **Correct** it when corrupted, outdated or contradicted by evidence.
9. **Transfer** it to a new task/context.
10. **Preserve plurality** where multiple valid solutions or cultural lineages have different operating envelopes.

A message being delivered, a graph edge being created, a hash being stored, a demonstration being watched, a strategy being copied and a strategy being successfully executed are not interchangeable observations.

The field-level objective should be **reliable cumulative capability under bounded resources and preserved epistemic diversity**—not raw adoption rate, popularity or mean reward alone.

## Benchmark portfolio and industry comparison

| Track | Strong external reference | What it measures | Current CivBench contribution | What is required before a superiority claim |
|---|---|---|---|---|
| Human population-level transmission | Brinkmann et al. (2026), Reward Network III | Machine discovery, human demonstrator selection, skill adoption and persistence across generations | Local-observation Reward Network engine, synthetic task fixtures, baselines and generational reference model | Ingest exact Zenodo v2.0 data, network splits, DQN weights/config and author analysis; rerun on matched tasks |
| Networked cultural accumulation | Czaplicka, Baumann & Rahwan (2025) | Interaction of local peer learning, innovation and global algorithmic mediation under varying network density | Ring-plus-shortcut mediation model, acquisition/adoption telemetry, lineage concentration and performance-vs-diversity recommender sweep | Port the published ER network, trait branches/levels, effort budget, costs, payoff distribution and death-birth semantics; reproduce paper curves first |
| Agent-to-agent few-shot transfer | Platonov et al. (2023), MEDAL-ADR | Demonstration fidelity, recall after expert dropout, generalization to held-out tasks | Not yet implemented as an exact benchmark; current task protocol names the comparison | Use equivalent procedural task families and budgets; compare memory, expert dropout, attention, ADR and teach-back with held-out context splits |
| Iterated content transformation | Perez et al. (ICLR 2025), telephone game | How content attributes drift and converge across repeated model-to-model handoffs | Four-attribute synthetic channel and attractor/convergence sweep | Run pinned real model/provider chains and independently scored text outputs; report each attribute separately |
| Individual-learning controls | 2024 combinatorial innovation experiments | Whether social learning adds value over repeated individual effort | Explicit matched-control requirements in protocols | Equalize individual trials/time/cost, fatigue and feedback; compare on same task pools |
| Proof-carrying skill exchange | Proposed CivBench design | Scope, contract correctness, provenance, qualification, correction and revocation | Capsule proposal and JSON Schema | Implement strict Rust validation, independent verifier, safe policy IR and held-out teach-back evaluation |

### Source references

- Brinkmann et al., *Propagation and preservation of AI-discovered problem-solving strategies in human culture*, Nature Communications (2026): https://www.nature.com/articles/s41467-026-76113-2
- Czaplicka et al., *Mutual benefits of social learning and algorithmic mediation for cumulative culture*, Journal of the Royal Society Interface (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11978438/
- Platonov et al., *Learning few-shot imitation as cultural transmission*, Nature Communications (2023): https://www.nature.com/articles/s41467-023-42875-2
- Perez et al., *When LLMs Play the Telephone Game*, ICLR (2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html

These are complementary benchmark families, not commensurate scores. Reward Network III is a human population task; MEDAL-ADR is sensorimotor few-shot imitation; the telephone game measures text transformation; Czaplicka et al. model agent-based accumulation under a different task abstraction.

## Paradigm shifts

### 1. From diffusion to capability transactions

Transmission should be a typed transaction with explicit observations and receipts:
- content bytes arrived;
- publisher identity was authenticated;
- evidence and scope were validated;
- learner reconstructed the rule;
- learner executed it successfully;
- learner retained it;
- learner transferred it to a held-out context.

Record each transition separately. The current code already separates observation/acquisition/adoption in the networked model and observation/acquisition/corruption/integration in the generational model.

### 2. From one prestige score to evidence-calibrated plural selection

A global top performer can accelerate capability diffusion but can also make a culture dependent on one lineage. Compare:
- performance-only ranking;
- performance plus lineage rarity;
- evidence reliability;
- uncertainty/calibration;
- context-fit to the learner's state;
- human/agent choice and explicit refusal.

The performance-vs-lineage-diversity recommender in the current branch is a first controlled intervention. The result must be presented as a Pareto frontier over capability and diversity, not as an automatic endorsement of the novelty bonus.

### 3. From adoption to cumulative improvement

A transmitted strategy counts as a cultural contribution when it improves a learner that did not independently discover it, survives transmission and helps on a pre-registered held-out task. The generational model therefore reports social-only optimal acquisition/integration separately from individual discovery and overall prevalence.

### 4. From message integrity to semantic fidelity

Cryptographic integrity proves that bytes were not modified. It does not prove that meaning, evidence, assumptions or safety boundaries survived the transformation. The iterated-transmission track should measure:
- attribute-level drift;
- evidence/citation retention;
- uncertainty retention;
- precondition/postcondition recall;
- factual contradiction;
- task success after teach-back;
- source attribution and lineage.

### 5. From one global culture to a portfolio of context-bound skills

Different solutions may be correct in different environments. The best collective memory should preserve alternatives, scope, assumptions and evidence, and select among them based on context. Convergence is not always success; diversity can be protective when environments shift.

### 6. From static trust to continuous qualification

A skill capsule should start discoverable, not trusted. Qualification must be scoped to task family and operating envelope, rechecked after dependency/runtime changes, and revocable when evidence changes. Signed authorship, semantic retrieval, executable validity, scientific validity and authorization are separate claims.

## Proposed benchmark targets

Targets are initial engineering gates and should be frozen after pilot runs before definitive held-out evaluation.

| Area | Proposed target |
|---|---|
| Reproducibility | Reproduce each reference paper's published curves/summary statistics within its reported uncertainty before extending the model. Pin code, data, config, seeds and output digest. |
| Baseline superiority | On a pre-registered held-out split, beat the strongest matched simple baseline with a 95% confidence interval for improvement that excludes zero, or clearly report non-inferiority plus a quantified reliability/diversity benefit. |
| Transfer | Report performance on unseen network identities and held-out topology/reward families; no near-duplicate network family leakage. |
| Cultural contribution | Report socially acquired-and-integrated capability among agents that did not individually discover it; never substitute total prevalence. |
| Retention | Estimate survival curves after teacher removal, memory pressure, restart and multiple generations; include no-social and matched individual-learning controls. |
| Error correction | Measure false adoption, descendant contamination, time-to-detection and recovery after corrupted demonstrations and contradictory evidence. |
| Content drift | Report signed and absolute drift by attribute, evidence retention, uncertainty retention and task success; never collapse safety-critical drift into semantic similarity. |
| Diversity | Report unique lineages, effective diversity and HHI alongside capability. A candidate target is no more than 5% reduction in held-out capability while achieving a meaningful, pre-registered reduction in lineage concentration, but the final trade-off must be pilot-calibrated. |
| Capsule qualification | Reject 100% of invalid digest/signature/scope/policy/expired/revoked fixtures in the adversarial test suite; independently qualify only exact pinned digests. |
| Resource efficiency | Report CPU/GPU time, memory, energy if measurable, sample/interaction budgets and cost per successfully transferred held-out capability. |

No target should be declared achieved from synthetic unit fixtures alone.

## Evaluation discipline: how to beat state of the art credibly

1. **Reproduce first.** A paper's reference code and data should be pinned, imported and independently rerun. A hand-built synthetic graph is a unit-test fixture, not a reproduced benchmark.
2. **Match budgets.** Equalize demonstrations, individual-learning trials, time/compute, number of interactions and memory.
3. **Hold out the right thing.** Split by network identity and generator family; for LLM chains also split by source text, language, domain and model/prompt identity as appropriate.
4. **Use matched randomness.** Use common seed sets for counterfactual conditions where possible and publish the seed schedule.
5. **Report distributions.** Show means, intervals, tails and failure rates across independent full-system seeds; do not present the best seed as the model's performance.
6. **Preserve negative controls.** No-social, individual-only, random demonstrator, performance-only recommender, replay/no-op, and adversarial/corruption controls must remain in the benchmark.
7. **Ablate every new mechanism.** If HDC retrieval, memory, proof-carrying capsules, diversity ranking or error correction is added, compare it with the same architecture without that feature.
8. **Separate training/selection/evaluation.** No tuning on test data. An independent verifier consumes exact artifact and input digests.
9. **Measure capability and agency.** Learners should be able to understand, reject, fork, correct and retain alternative skills; adoption is not proof of flourishing.
10. **Publish a task-specific claim.** Say which benchmark, split, budgets, baselines and uncertainty support the claimed gain. Do not make a universal "better than SOTA" statement.

## Immediate implementation roadmap

### P0: exact source reproduction
- Reward Network III final Zenodo v2.0 ingest and paper reanalysis.
- Rust port of the Czaplicka et al. model, kept separate from the simplified networked mediation mechanism.
- Real-LLM telephone-game source/model/prompt chain runner, with a metadata and retention policy.

### P1: controlled paradigm experiments
- Performance-only versus lineage-diversity recommender frontier.
- Density × mediation fraction × recommender objective sweep.
- Error-correcting and proof-carrying demonstrations.
- Teach-back and strategy compression under fixed information budgets.
- Recall/transfer after teacher removal and memory pressure.

### P2: Symthaea integration
- Replace scalar synthetic quality with the existing task-specific evaluators.
- Add HDC retrieval as an explicitly non-authoritative search layer; test whether it improves retrieval, not truth.
- Use CfC/LTC/FEP cognitive mechanisms as contenders under the same frozen tasks and budgets.
- Preserve provenance and uncertainty with existing Mycelix evidence contracts.
- Use Symtropy for larger network and environment stress tests; use Sol-Atlas only for evidence/lineage visualization, not validation.

## Current status and limits

This roadmap is included in a research PR. The Rust modules, sweeps and synthetic fixtures are authored, but their compile/test status is pending the dedicated workflow. The final Reward Network III Zenodo archive has not yet been ingested, the published Czaplicka model has not yet been ported, and the telephone-game channel does not call an LLM. No SOTA superiority, empirical human effect or exact paper reproduction is claimed.
