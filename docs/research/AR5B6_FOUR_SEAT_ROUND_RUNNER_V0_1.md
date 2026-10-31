# AR-5B6 — four-seat fixed-set round runner v0.1

**Status:** executable synthetic four-seat round driver implemented; six tests authored but not compiled or executed.  
**Tracking issue:** [#9393](https://github.com/Luminous-Dynamics/symthaea/issues/9393)  
**Implementation:** \`crates/domains/symthaea-architecture-engineering/src/round_campaign_runner.rs\`  
**Contract:** [AR-5B4](./AR5B4_MATCHED_ROUND_SEMANTICS_V0_1.md)  
**Adaptive baselines:** [AR-5B5](./AR5B5_ADAPTIVE_BANDIT_BASELINES_V0_1.md)

## Why this is a separate benchmark

The original policy campaign selected one arm per loop iteration. Coral Discovery is a multi-agent, round-based scheduler. Comparing those units directly would be misleading.

AR-5B6 executes four seats in every round through the AR-5B4 contract. It fixes four directions before any policy sees an outcome. This isolates the direction-allocation mechanism from the separate problem of proposing and verifying genuinely new research directions.

## Track A protocol

- Four pre-registered directions: A, B, C and D.
- Four evaluation seats per round.
- Common maximum of two seats per direction per round.
- The first round is the same bootstrap for every policy: A, B, C and D receive one evaluation each.
- All subsequent assignments are computed from the same pre-round history snapshot.
- Candidate outcomes are calculated independently from that snapshot's per-direction evaluation count; no result changes the assignment of another seat in the same round.
- Every seat has a unique action ID, immutable work-item digest, result/verifier receipt reference and cost receipt.
- Round close requires four valid results and four cost receipts. The ledger enforces contiguous rounds and a close-digest parent chain.

The four-seat action budget is exactly \(4R\) for \(R\) rounds. Each candidate execution charges one campaign unit.

## Policies implemented

1. **Fixed round robin:** one seat per direction each round.
2. **Seeded random:** randomly choose two distinct directions and assign two seats to each.
3. **Score only:** rank directions by best observed score and allocate two seats to each of the top two.
4. **Global UCB:** rank by mean observed score plus an exploration bonus.
5. **Discounted UCB:** use the AR-5B5 exponentially discounted history and frozen \(\gamma=0.90\).
6. **Gaussian Thompson sampling:** use the AR-5B5 Normal-Normal posterior and deterministic seeded sampling.
7. **DirectionBoard fixed-set ablation:** rank the actual DirectionBoard direction statistics by local-record improvement yield per cost plus UCB exploration; the NEW arm is excluded.
8. **Coral direction-only ablation:** rank directions using the published Coral windowed-yield, historical \(y_+\), and exploration-potential equations; the NEW arm and infancy estimator are excluded.
9. **Oracle upper bound:** access the future synthetic payoff schedule. This is not deployable and must not be included in deployable-policy averages.

The DirectionBoard and Coral policies here are deliberately named *fixed-set* / *direction-only* ablations. They do not claim to reproduce either complete system. All policies obey the same two-claims-per-direction cap in this mechanism-matched track. Native system claim caps and NEW discovery belong in a separate end-to-end system comparison.

## Development landscapes

The runner reuses four deterministic payoff families:

- **Delayed breakthrough:** an early leader saturates while a low-start direction has a later payoff.
- **No new direction:** pre-registered existing directions contain the best outcomes; there is no hidden new operator.
- **New operator as a pre-registered arm:** the delayed high-payoff direction D is visible from the start. This is a scheduler control, not a direction-discovery result.
- **Rank reversal:** the early leader declines while another direction improves later.

These are development fixtures only. Confirmatory claims require at least 100 paired payoff instances across six or more independent landscape families, including deceptive proxies, resource trade-offs, cross-task regressions and controlled drift.

## Evidence and cost model

The runner emits a full \`RoundPlan\` and \`RoundCloseReceipt\` per round and appends them to \`RoundLedger\`. The plan binds the campaign, round, policy, seed, parent close digest, pre-round snapshot digest, fixed cost-profile digest, claim caps, work-item digests and action IDs.

The synthetic resource-cost profile charges every evaluation:

- 100 wall-time milliseconds;
- 80 CPU-time milliseconds;
- one candidate execution;
- one campaign budget unit.

These are **synthetic fixture costs**, not measured hardware performance. They are held equal across policies to isolate allocation effects. Real CPU/GPU, memory, energy, token, tool and evaluator costs must be collected in the best-practical-system track.

Verifier receipts in this simulator are deterministic content references. They are not signatures, independent external attestations or proof that an external verifier operated honestly. The contract validates structural linkage and cost/result conservation.

## Authored tests

Six tests are authored for:

- four-seat conservation and the common per-direction cap across all policies and four landscapes;
- identical bootstrap assignments for all policies;
- complete deterministic replay of the round/receipt chain for a fixed seed;
- frozen pre-round snapshot binding on every result receipt;
- fixed-set treatment of the high-payoff D arm without a free NEW proposal;
- no-new-direction negative controls for DirectionBoard and Coral directions-only ablations.

The tests exercise the simulator and contract together, but they have **not** been compiled or executed in the authoring environment.

## Important limitations

- This is Track A only: directions A–D are known before the first evaluation.
- It does not measure semantic novelty detection or the quality/cost of NEW proposals.
- It does not implement Coral's agent workspaces, asynchronous execution protocol, full verifier, NEW estimator or native claim caps.
- It does not integrate CodeEvolve's island-based search, CVT-MAP-Elites archive, inspiration crossover or weighted LLM ensemble.
- Every development evaluation is verified-success; deterministic failure/timeout injection and retry accounting should be a separate stress campaign.
- Resource costs are synthetic and single-worker. Atomic multi-worker reservations, GPU concurrency and memory peak aggregation are not implemented.
- The synthetic oracle knows the payoff schedule and is an upper bound only.

## What the runner enables next

1. Add at least two additional independent landscape families and preregister 100+ paired seeds.
2. Add serialized-run replay that checks the complete chain from plan and work-item digest through close receipts and ledger totals.
3. Add failure/timeout stress conditions with identical predeclared fault schedules across policies.
4. Build Track B with a separate paid \`ProposeDirection\` action, independently adjudicated novelty, and a separately charged later evaluation.
5. Run Coral-native NEW and claim-cap behavior as a faithful-system ablation.
6. Only after those tracks, connect CodeEvolve/CVT-MAP-Elites and actual LLM/agent systems, pinning model, prompt, orchestration, sandbox, verifier and all resource budgets.

## Claim ceiling

This tranche establishes the executable structure of a four-seat fixed-set synthetic campaign and connects policy selection to round plan/close validation. It does not demonstrate better-than-SOTA scheduling, successful real-agent discovery, novel operator invention, scientific validity or production readiness.
