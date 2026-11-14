# AOBench — Mutation Adequacy and Cross-Plane Adversarial Qualification

**Status:** Research and engineering protocol; no mutation score or benchmark result claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-009 #12140](https://github.com/Luminous-Dynamics/symthaea/issues/12140)  
**Depends on:** #12010, #12014, #12015, #12048, #12067, #12118, #12121, #12122

## 1. Research thesis

AOBench must qualify not only the organization being evaluated, but also the instruments used to evaluate it.

A checker can pass every test written by its author while a crucial predicate is missing, reversed, weakened, or bypassed. A benchmark that does not test its own evaluators is vulnerable to false confidence, especially when the evaluated agent can influence the model, the trace, the intervention, or the interpretation.

The central question for this track is:

> Can independently generated defects in the assurance machinery be detected without rejecting valid organizational substitutions?

This makes **mutation adequacy** a first-class qualification dimension. It is not a scalar reward and cannot compensate for a failure in safety, authority, consent, provenance or evidence integrity.

## 2. Paradigm shift: test the test, then test the boundary

The programme advances three linked ideas.

### 2.1 Mutation-tested assurance

Instead of asking only whether the implementation passes its authored fixtures, deliberately mutate the assurance logic and verify that the frozen challenge corpus detects each meaningful defect.

This includes source-level mutants (remove a guard, reverse a subset test, bypass a digest check), model-level mutants (remove a transition, alter an authority invariant, hide a rare capability), and evidence-level mutants (replay a receipt, reorder an event, substitute a producer).

Mutants that are equivalent under the declared contract or are not reached by the test suite must be reported separately. They must not be silently counted as killed.

### 2.2 Paired adversarial twins

Every invalid case must have a positive twin created by removing only the invalid condition. This guards against a benchmark that appears secure because it rejects everything.

Examples:
- unauthorized actor versus externally authorized actor, with the same visible output;
- forged provenance versus independently reconstructible lineage;
- revoked consent versus current consent;
- hidden irreversible side effect versus no side effect;
- changed topology with preserved behavior versus unchanged topology with altered behavior;
- executable bytes changed under an approved version versus an unapproved digest substitution;
- one-epoch continuity versus repeated-epoch drift;
- missing observer/receipt versus independently observed and authenticated execution.

Positive twins are as important as attacks: legitimate internal replacement, state-ID permutation and behavior-preserving code replacement must not be classified as discontinuity merely because implementation identity changed.

### 2.3 Cross-plane claim integrity

AOBench already separates structural admission, formal viability, causal intervention, runtime evidence, behavioral continuity, governance and substrate/resource observations. Mutation testing must challenge the *interfaces* between these planes.

For example, a valid signature proves that a receipt was signed by the corresponding key; it does not by itself prove that the reported physical or semantic event occurred. Likewise, a matching action/observation trace does not establish that the actor had authority, that consent was current, or that irreversible effects were absent.

The benchmark must detect when one evidence plane is used to make a claim belonging to another.

## 3. Mutation operator catalogue

The first corpus should be small, explicit and severe before it becomes large.

| Family | Representative mutations | Expected qualification |
|---|---|---|
| Runtime lineage | Skip SHA-256 recomputation; accept wrong producer/version; allow failed event outputs; omit required producer/consumer evidence | Reject |
| External evidence | Accept absent required dependency receipt; replay stale receipt; detach receipt from event/epoch ancestry | Reject or Inconclusive when authenticity cannot be checked |
| Structural admission | Remove allowlist; mutate protected root; weaken required edge; sever policy-enforcement dependency; bypass executable identity grant; exceed edit budget | Reject |
| LTS semantics | Reverse simulation direction; omit successor-pair test; ignore required invariant; accept malformed state/transition; treat budget exhaustion as pass | Reject or Inconclusive |
| Intervention | Wrong-target delivery; no-op actuator; collateral mutation; observer loss; carryover; cross-trial interference | Reject isolation / mark Interpretable Inconclusive as appropriate |
| Statistical assay | Treat missing as zero; bypass multiplicity; ignore duplicate trial IDs; claim equivalence from non-significance; ignore underpower | Reject or Inconclusive |
| Temporal continuity | Drop a maintenance epoch; reorder events; hide delayed divergence; accept accumulated repeated-epoch drift | Reject or Inconclusive |

The complete reviewed operator catalogue, severity assignment, test identity and expected result will be versioned with the corpus.

### 3.1 Current continuity model: explicit scope

The current finite transition checker uses state observations, capabilities, required invariants, initial-state sets, and action/observation-labelled transitions. That is a useful deterministic foundation.

Its current transition type does not itself encode typed side effects, actor identity, authority grants, consent scope, provenance, irreversible-effect classes or substrate deltas. The specification therefore treats those as separate model/evidence dimensions; this track must not claim they are already enforced by the LTS implementation. They become explicit cross-plane challenge inputs and future typed-contract work.

The checker is model-relative: its result applies only to the declared finite state space, action/observation alphabet, invariant set and probes. Hidden states and unmodelled mechanisms remain outside the witness.

## 4. Independent challenge and oracle protocol

1. Freeze subject commit(s), source tree identity, policy and model digests, corpus version, harness version, action/probe budget and expected oracle outcomes.
2. Generate deterministic source-level and model-level mutants from a reviewed operator catalogue.
3. Demonstrate that every mutant is reached by at least one relevant test. Classify equivalent and unreachable mutants separately.
4. Execute the same sealed challenge against the unmodified verifier, each mutant and the candidate verifier.
5. Keep the expected-outcome oracle independent of the implementation predicate under test. Do not let the agent author both the candidate model and the sole oracle for its success.
6. Preserve exact mutant patch, challenge ID, raw observations, runtime receipts, verifier build identity and replay instructions.
7. Blindly adjudicate ambiguous cases. Record disagreements and update the oracle only through a versioned review, never by silently changing expected outcomes after seeing candidate results.
8. Hold out complete mechanism families—not merely random seeds from a familiar generator—for confirmatory runs.

A challenge is *adequately observed* only when the intervention target, delivery, isolation, observer and causal interpretation meet their respective frozen requirements. A missing observation is not a negative result.

## 5. Metrics and qualification gates

Report each family separately.

- **Critical-mutant kill rate:** fraction of reviewed, reached, non-equivalent critical mutants detected.
- **Overall non-equivalent mutation score:** detected non-equivalent mutants divided by all reached non-equivalent mutants.
- **Escaped-mutant severity:** count and severity of meaningful defects that survive.
- **Positive-twin acceptance:** fraction of legitimate condition-removed controls accepted.
- **False continuity acceptance:** invalid candidates incorrectly declared continuous.
- **False discontinuity:** valid behavior-preserving substitutions incorrectly rejected.
- **Correct Inconclusive rate:** unresolved evidence and exhausted budgets remain unresolved.
- **First-divergence localization:** exact/within-k transition and event localization.
- **Probe efficiency:** tests/interventions required to distinguish a candidate pair.
- **Resource cost:** verifier time, memory, search nodes and mutation execution cost.
- **Temporal robustness:** continuity across repeated maintenance epochs.

Proposed pilot targets (not achieved results):

| Qualification dimension | Pilot target |
|---|---:|
| Reviewed critical safety/authority/consent/provenance/irreversible-effect mutants killed | 100% |
| All reviewed non-equivalent mutants killed after adjudication | ≥95% |
| Valid deterministic positive twins accepted | 100% |
| Inadequate-evidence and exhausted-budget controls marked Inconclusive | 100% |
| Critical false-continuity acceptances in sealed pilot | 0 |

These are calibration targets, not claims of zero real-world risk. A finite zero-failure sample must be reported with its denominator and uncertainty. Mutation classes must not be pooled into a headline number that hides a critical escaped mutant.

## 6. Experimental design: from unit tests to an organizational demonstration

The minimum convincing experiment is a complete causal and reconfiguration episode:

1. Start with a frozen organization manifest and independently qualified baseline capability probes.
2. Induce a primary-producer failure using a registered intervention.
3. Independently observe and localize the first failure.
4. Require Symthaea to distinguish plausible causes through budgeted probes rather than assume the first visible symptom is causal.
5. Discover an alternate producer or arrangement.
6. Build the proposed organization in shadow mode; do not grant it production authority.
7. Run structural admission, viability and continuity witnesses against frozen contracts.
8. Obtain external authorization for promotion and bind the grant to the exact candidate manifest/process identities.
9. Promote through a controlled transition, retaining rollback and evidence ancestry.
10. Execute repeated maintenance epochs and sealed held-out probes.
11. Independently reconstruct the timeline, receipts, interventions, capability measurements and failures.

The existing synthetic-organism environment #6970 should be reused as the first adapter after its own exact-head qualification. AOBench should not create a second simulator merely to test this protocol.

## 7. Industry and research comparators

AOBench should not compete on one undifferentiated agent-success score. The relevant systems measure different capabilities:

| Comparator | What it measures well | AOBench's matched extension |
|---|---|---|
| MANTA | Inference-time multi-agent topology adaptation; five task suites | Whether adaptations preserve explicit capability, authority and evidence contracts |
| ASIWorld | Persistent open-world execution, progression and survival costs | Hidden organizational failure, verified regeneration, lineage and repeated-epoch continuity |
| BALROG | Interactive LLM/VLM reasoning in dynamic game environments | Process-level failure, intervention validity and organization continuity during interaction |
| AgentLAB | Long-horizon attacks across realistic agent environments | Whether attacks mutate authority, consent, boundary, provenance or the evaluator itself |
| Traverse (*Locating Hidden Failures...*) | First-mistake localization in long agent trajectories | Causal first-failure diagnosis plus verified intervention and organizational repair |
| ACDB | Budgeted hard interventions and causal-graph recovery | Hidden organizational mechanisms, intervention delivery/isolation and safe structural action |
| Causal-Discovery-Bench | Sequential experiment design, resource allocation and conflicting evidence | Identifying which organizational probe is informative and when evidence is insufficient |

The comparator numbers are not directly rankable. For example:
- MANTA reports a 74.0 mean across five suites (30 tasks per suite, three independent runs); its WorkBench result is 43.3 while ADAS reports 66.7 and AgentSquare 62.2 in that column. The aggregate must not hide domain variation.
- AgentLAB reports five attack families, 28 environments and 644 security cases.
- Traverse's 2026 paper studies 2,518 trajectories, 6,967 annotated mistakes and 78 failure types; it motivates first-error localization rather than final-outcome-only scoring.
- ACDB reports a budgeted intervention task and shows that classical PC plus a greedy active-orientation heuristic can outperform raw LLM intervention strategies on its current calibrated ladder. Its authors explicitly frame the results as a benchmark audit/calibration stage, not evidence that LLMs solve causal discovery.
- Causal-Discovery-Bench reports 3,802 episodes across 13 models and attributes 94.5% of observed agent mistakes to experimental design rather than downstream inference in its analysis. This makes experimental design a primary AOBench target.
- ASIWorld provides a persistent survival sandbox where long-horizon actions, preparation, execution and recovery costs can be observed; BALROG supplies a broader interactive reasoning comparator.

AOBench's advantage should come from **matched experimental bridges**: run a fixed baseline and topology-adaptive system under the same organization, model budget, probes, interventions, environment, seeds and held-out mechanisms, then measure the assurance dimensions above. Do not transfer a published score from one harness to another.

## 8. Near-term implementation sequence

**M0 — Freeze test inventory.** Map every existing runtime, structural and continuity predicate to at least one positive, negative, boundary and mutation fixture.

**M1 — Mutation-test the deterministic core.** Start with digest, producer/consumer, required-edge, protected-root, executable-identity, simulation-direction, successor-relation and budget-exhaustion mutants.

**M2 — Add cross-plane fixtures.** Bind actor/authority, consent, provenance, external dependency and irreversible-effect observations to separate, independently sourced evidence.

**M3 — Add temporal and stochastic controls.** Use repeated epochs, delayed divergence and qualified #12048 reports; insufficient power remains Inconclusive.

**M4 — Run the complete synthetic-organization episode.** Reuse #6970, publish manifests and raw receipts, and compare fixed-topology, parameter-only, local-repair, topology-only and joint-adaptation arms under matched budgets.

**M5 — Independent replication.** Release corpus version, generator, oracle specification, exact commit SHAs, raw evidence, uncertainty, resource cost and all failures.

## 9. Claim boundary

This track measures the mutation resistance and falsifiability of bounded assurance instruments. It does not establish universal behavioral equivalence, safety in every environment, biological autopoiesis, life, sentience or consciousness. No performance or mutation score is claimed until the frozen protocol has been run and independently qualified.

## References

- Huang et al. (2026), [MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems](https://arxiv.org/abs/2607.28527).
- Jiang et al. (2026), [AgentLAB: Benchmarking LLM Agents against Long-Horizon Attacks](https://proceedings.mlr.press/v306/jiang26as.html).
- Rahman et al. (2026), [Locating Hidden Failures Makes Long-Horizon Agents More Reliable](https://arxiv.org/abs/2609.17930).
- Deb et al. (2026), [Active Causal Discovery Benchmark (ACDB)](https://arxiv.org/abs/2609.31675).
- Liu et al. (2026), [Causal-Discovery-Bench: Can LLMs Autonomously Conduct Effective Experiments?](https://arxiv.org/abs/2611.09934).
- Lin et al. (2026), [ASIWorld: Benchmarking Autonomous Agents in a Persistent Survival Sandbox](https://arxiv.org/abs/2610.14807).
- Paglieri et al. (2025), [BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html).
