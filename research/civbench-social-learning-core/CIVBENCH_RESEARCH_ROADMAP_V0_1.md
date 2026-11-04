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
| Situated multi-cultural agent behavior | Pham et al. (2026), LiveCultureBench (ACL 2026) | Dynamic-town goal completion, local norm adherence, profile faithfulness, contextual awareness, coherence and verifier reliability | Adapter-neutral Rust metrics, context-stratified aggregation, verifier calibration and Pareto comparison | Official repository currently lacks its data download link; pin released runner/data/prompts, reproduce paper and human-audited verifier split before comparison |
| Exploratory generational ABM | Brinkmann et al. (2026), archived `abm/abm.ipynb` in Zenodo v2.0 | Binomial allocation of exploration effort, strategy discovery, stochastic candidate-set softmax teacher selection, Bernoulli transmission, max-based integration, noisy demonstration rewards and machine-availability conditions | New `reward_network_abm_reference.rs` semantic port and five notebook-aligned grid profiles; the old `paper_abm` remains separately labelled as a simplified strategy-adoption ablation | Validate every state transition against the notebook and archived outputs; add NumPy stream/conformance tests where needed; hydrate original network/weight LFS assets and reproduce paper curves before superiority claims |
| Networked cultural accumulation | Czaplicka, Baumann & Rahwan (2025) | Interaction of local peer learning, innovation and global algorithmic mediation under varying network density | Source-pinned Rust semantic port of the authors' C model plus a separate ring/shortcut mechanism extension; reference global-best policy and exposure/adoption-count-aware recommender treatment | Compare Rust transitions against an instrumented C build; reproduce paper curves first; then evaluate diversity interventions on common seeds |
| Agent-to-agent few-shot transfer | Platonov et al. (2023), MEDAL-ADR | Following with an expert present, acquisition, score after a half-episode expert dropout, and transfer to held-out tasks | New `few_shot_transfer.rs` adapter implements the published CT score, following/half-exposure decomposition, expert-normalized half-exposure score, macro aggregation, uncertainty and task/family split checks | The metric is implemented, but GoalCycle3D and MEDAL-ADR are not ported; compare actual agents only after pinned task families, expert policies, budgets and held-out contexts are available |
| Iterated content transformation | Perez et al. (ICLR 2025), telephone game | How content attributes drift and converge across repeated model-to-model handoffs | Four-attribute synthetic channel and attractor/convergence sweep | Run pinned real model/provider chains and independently scored text outputs; report each attribute separately |
| Individual-learning controls | 2024 combinatorial innovation experiments | Whether social learning adds value over repeated individual effort | Explicit matched-control requirements in protocols | Equalize individual trials/time/cost, fatigue and feedback; compare on same task pools |
| Proof-carrying skill exchange | Proposed CivBench design | Scope, contract correctness, provenance, qualification, correction and revocation | Capsule proposal and JSON Schema | Implement strict Rust validation, independent verifier, safe policy IR and held-out teach-back evaluation |

### Source references

- Brinkmann et al., *Propagation and preservation of AI-discovered problem-solving strategies in human culture*, Nature Communications (2026): https://www.nature.com/articles/s41467-026-76113-2
- Pham et al., *LiveCultureBench: a Multi-Agent, Multi-Cultural Benchmark for Large Language Models in Dynamic Social Simulations*, ACL 2026: https://aclanthology.org/2026.acl-long.1247/ (DOI 10.18653/v1/2026.acl-long.1247)
- Czaplicka et al., *Mutual benefits of social learning and algorithmic mediation for cumulative culture*, Journal of the Royal Society Interface (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11978438/
- Platonov et al., *Learning few-shot imitation as cultural transmission*, Nature Communications (2023): https://www.nature.com/articles/s41467-023-42875-2
- Perez et al., *When LLMs Play the Telephone Game*, ICLR (2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html

These are complementary benchmark families, not commensurate scores. Reward Network III is a human population task; MEDAL-ADR is sensorimotor few-shot imitation; the telephone game measures text transformation; Czaplicka et al. model agent-based accumulation under a different task abstraction.


### Archived ABM source and parameter boundary (updated 2 November 2026)

The Zenodo v2.0 archive and the self-contained `abm/abm.ipynb` source are now the reference for the exploratory Reward Network III ABM. The notebook sets (R=50), (G=20), and normally (N_{gen}=20); it samples (k_{opt}\sim\mathrm{Binomial}(K_{ind},q_{opt})), computes cumulative optimal/myopic discovery probabilities, samples ten demonstrators without replacement, selects one by softmax of prior demonstration reward, transmits with probability \(\lambda\), maps failed transmission to strategy 0 (myopic), integrates with \(\max(s_{ind},s_{social})\), and adds Gaussian demonstration noise with standard deviation \(\epsilon\sqrt{K_{demo}}\). Machine availability is none, seed generation only, or all generations.

The extracted notebook profiles contain **4,400 parameter cases** across five experiment families: human-only discovery (800), seed-machine discoverability (600), teacher-selection temperature (900), demonstration reward noise (900), and population/machine-count robustness (1,200). Each uses 50 replications in the archived notebook. The new `reward_network_abm_reference.rs` port analytically marginalizes the latent binomial allocation to sample the exact terminal strategy distribution; an exhaustive small-(K) enumeration test checks equivalence to explicit binomial summation. The Rust RNG is deterministic but not NumPy bit-stream compatible. The separate `paper_abm.rs` remains a compact causal/ablation model and is **not** the archived paper ABM.

The exact notebook source hash, archive hash and LFS object inventory are recorded in `REWARD_NETWORK_III_PROTOCOL_V0_1.md`. The archive contains 42 LFS pointers. Hydrating original network and model artifacts with digest verification remains P0; no participant/move rows should be copied into the public repository. The Czaplicka 2025 track also has a pinned Rust semantic port and a distinct recommender extension, but C/Rust conformance and published-curve replication remain pending.

### New empirical cautions for cumulative culture

- Walker & Fay (2026) report two large experiments (combined (N=941)): larger populations enhanced cumulative technological improvement only under particular information-filtering or external-representation conditions. Population size alone is not a reliable intervention.
- A 2026 transmission-chain experiment finds subjective appeal can cause ineffective solutions to emerge and persist even when payoff is unchanged. A benchmark must distinguish perceived appeal, actual payoff and functional effectiveness, and must measure cultural persistence of inefficient variants.
- LiveCultureBench (ACL 2026) adds multi-cultural task/norm trade-offs and verifier uncertainty; static cultural-trivia scores are not a substitute for situated behaviour.

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
- Reward Network III final Zenodo v2.0 ingest and paper reanalysis, separately for the human population experiment and the 20-generation exploratory ABM.
- LiveCultureBench official artifact ingestion and exact ACL 2026 reproduction once its public data/runner release is available; keep adapter-neutral situation metrics separate from official-paper results.
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

This roadmap is included in draft PR #9970. The standalone Rust crate passed its exact-head `cargo check --locked --all-targets` and `cargo test --locked --all-targets` gate on the earlier run #222 (105 tests); the newer archived-notebook ABM module and sweep were added after that run and require their own exact-head validation. The Reward Network III Zenodo v2.0 ZIP (28,279,082 bytes) passed the published MD5 and ZIP checks; its SHA-256 and the extracted ABM notebook source digest are recorded in the protocol. The ZIP contains 306 files, including 42 Git LFS pointers; the pointer objects have recorded OIDs/sizes but are not hydrated. No participant-data reanalysis, DQN reproduction, NumPy RNG parity, or full paper-result replication is claimed. The Czaplicka model now has a source-pinned Rust semantic port, but no C/Rust conformance or paper-result replication is claimed. The telephone-game channel does not call an LLM. The official LiveCultureBench repository currently lacks its data-download link, so its simulator is not yet reproduced; only adapter-neutral metrics are implemented. No SOTA superiority, empirical human effect or exact paper reproduction is claimed.


## New evidence and benchmark track: intentional transmission and causal assistance

Two 2026 human studies sharpen the next research question:
- Mackintosh et al., *Intentional transmission of knowledge optimises cumulative cultural evolution*, Scientific Reports (19 September 2026), DOI 10.1038/s41598-026-62734-6. Adults strategically preferred beneficial information when teaching; this selective transmission improved cumulative cultural accumulation compared with random selection. The pattern was not uniform across age groups.
- Saral, Ibáñez de Aldecoa & Derex, *The roles of cultural transmission and causal reasoning in the cultural evolution of technology*, PNAS Nexus 5(4), pgag090 (26 March 2026), DOI 10.1093/pnasnexus/pgag090. In two transmission-chain experiments (n=900), cumulative improvement occurred even without causal structure; causal reasoning helped early search but its incremental contribution diminished over generations; technical/physical framing did not add advantage over the abstract causal framing.

This supports a separate **intentional teaching** benchmark: teacher selection (who), content selection (what), learner-context adaptation (for whom), representation (how) and budget allocation (how much) are independently controlled. A teacher with a high overall score may still transmit the wrong subset of its repertoire.

The branch now includes `src/intentional_teaching.rs` and `INTENTIONAL_TEACHING_PROTOCOL_V0_1.md`. The model compares random, declared-benefit, evidence-calibrated, learner-context-aware and diversity-aware packet selection under a fixed bandwidth budget, plus a matched individual-only condition. It explicitly separates learning-resource cost from intrinsic item difficulty and reports observation, acquisition, integration, correct execution, retention and held-out transfer separately. The six-condition sweep uses 256 fixed synthetic seeds; it does not reproduce either paper or estimate human effects.

New experiments to prioritize:
1. Compare example replay, causal explanation, executable rule and proof-carrying capsule under matched learner-time and token/bandwidth budgets.
2. Test strategic teaching for beneficial but difficult items versus easy/high-reward items.
3. Test robustness to inflated claims, weak evidence, context shifts and rare high-benefit skills.
4. Test whether teaching policies improve held-out cumulative utility over matched individual learning.
5. Report capability, error adoption, knowledge diversity and learner autonomy as separate outcomes.


## New external benchmark target: state-aware adaptive transmission (2026)

Colas et al. (2026), *Discovering Adaptive Transmission Programs for Collective Innovation* (arXiv:2608.24545; CogSci 2026), formalizes transmission as executable state-aware programs that choose who shares what with whom. The authors report up to 37% higher collective performance than standard baselines in their Collective LittleAlchemy simulation and use a decisive ablation: preserve the exact sender/receiver schedule and timing but shuffle the message content. The paper also reports transfer across task/domain and population variations. These are the authors' simulation results, not CivBench results and not demonstrated human effects. The paper's centralized full-state observation is an important privacy/autonomy boundary.

The branch now includes a new synthetic DAG baseline in `src/adaptive_transmission.rs` and a 324-condition, 128-seed sweep in `src/bin/adaptive_transmission_sweep.rs`. It compares NoSocial, FixedRing, FixedRingContentShuffled, RandomNetworkContent, StateAwareFrontier and StateAwareContentShuffled across Layered/Branching/Convergent DAGs, population sizes 8/16/32, shortcut rates 0/0.2 and attention capacities 1/2/4. Each task has six base elements and 14 synthetic recipes. This is a mechanism baseline, not the authors' 720-element task and not LLM-evolved program search.

**Critical evaluation rule:** the state-aware content-shuffled condition must preserve sender/receiver pairs while changing only the content. This separates "who/when" scheduling from "what" is transmitted. Report collective result coverage, maximum depth, per-agent knowledge, individual discoveries, social acquisitions, redundancy, attention use and new results per transmission. Do not use a single adoption metric.

Next stages:
1. Pin and ingest the authors' actual task/runner/search code and license before a reproduction attempt.
2. Replace the tiny synthetic DAG with a held-out task family and a train/validation/test split by recipe family, not just random seed.
3. Add exact matched topology/schedule content-shuffle tests and common-random-number streams.
4. Compare hand-coded frontier heuristics with bounded DSL policies and LLM/evolutionary search only after the baseline is reproducible.
5. Test privacy-preserving local/consented state features against centralized full observability.
6. Add dropout, corruption, delayed prerequisites, adversarial withholding, lineage concentration and alternative-route recovery.
7. Make a held-out generalization claim only if state-aware routing beats the strongest matched static baseline on unseen DAG families/population sizes with pre-registered uncertainty and equal message/attention budgets.

References:
- Paper: https://arxiv.org/abs/2608.24545
- Author publication record: https://www.cedriccolas.com/publications/


### Cross-project adaptive-transmission interoperability

A proposed event contract now links the state-aware transmission benchmark with the existing civilizational claim, evidence, run and skill-capsule envelopes:
- `ADAPTIVE_TRANSMISSION_INTERCHANGE_V0_1.md`
- `ADAPTIVE_TRANSMISSION_EVENT_V0_1.schema.json`

The contract binds the exact payload and task scope to a program/version digest, sender/recipient references, consent mode, minimized recipient-need signal, route and resource budget, and separate delivery/observation/acquisition/integration/execution/retention/correction/transfer outcomes. It explicitly supports positive skills, scoped negative results, corrections, requests and contextual warnings.

This is not a new evidence ledger and does not require a central collector to ingest full recipient inventories. Symthaea owns cognitive selection and learning; Symtropy owns population-scale simulation; Mycelix owns identity/rights/provenance/disputes; Sol-Atlas owns projection and spatial-temporal inspection. Runtime enforcement is not yet implemented.


## New implementation update — recipient-state disclosure frontier (4 November 2026)

The adaptive-transmission engine now records an explicit `RecipientStateDisclosure` on each run:
- `CentralFullInventory` is the centralized research upper bound.
- `CoarseRecipeReadiness` releases at most two prioritized unheld-recipe request/readiness signals per recipient per step (recipe ID plus prerequisite-match fraction) rather than the full recipient inventory. Publishing every unheld recipe would reveal the complement of the held-recipe inventory; the cap blocks that direct reconstruction but does not eliminate inference from repeated signals.

The coarse mode rejects collective-novelty policy features that require global inventory knowledge. A multi-seed runner compares centralized, coarse and no-social conditions across Layered, Branching and Convergent DAG families (256 fixed seeds per family) under identical population, message and attention budgets. CI executes the sweep and stores a CSV artifact after the compile/test job.

**Interpretation boundary:** the coarse signal can still reveal which recipe is missing and how ready the recipient is. This is not differential privacy, anonymity, secure aggregation, cryptographic secrecy, or consent enforcement. Its purpose is to measure the utility cost of a minimized signal and establish a baseline for explicit leakage attacks. Runtime consent/rights enforcement and inference attacks are tracked in [Symthaea issue #10413](https://github.com/Luminous-Dynamics/symthaea/issues/10413).

This completes a first synthetic vertical slice across the architecture: Symthaea emits a bounded state-aware decision; Mycelix's proposed event envelope references consent/rights/provenance; Symtropy can study population effects and shocks; Sol-Atlas can visualize spatial and temporal transmission paths. These integrations remain interface-level research work, not deployed cross-project runtime capabilities.
