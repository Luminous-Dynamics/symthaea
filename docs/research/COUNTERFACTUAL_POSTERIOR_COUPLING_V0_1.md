# Posterior-Coupled Individual Rollouts (PCIR) v0.1

**Status:** research protocol and semantic fixtures; no trainer or model result is implemented.  
**Parent:** #13530 — DMC individual-effect pilot.  
**Consumes:** #13520 individual-effect arithmetic; #13255/#13258 temporal representation; #13239/#13152 decision sufficiency; #11655/#11659 simultaneous episode calibration; #13375/#13377 and Symtropy #3741/#3743 where their contracts apply; #9437 evaluator isolation.

## 1. Research question

Can a world model represent uncertainty about the *same hidden individual* and carry that uncertainty fairly through every alternative action, so that an imagined treatment effect is not contaminated by comparing different hypothetical individuals?

The current individual-effect programme correctly requires a common factual history, restored branch point, individual identity, shared exogenous-forcing commitment and paired action branches. It separately measures realized, frozen-latent and imagined-rollout contrasts in #13520.

This protocol makes one missing model-side object explicit: a **posterior over stable individual characteristics**, together with a coupling rule that reuses each posterior draw across all sibling interventions.

Examples of stable characteristics include mass, pole length, friction, actuator gain, morphology, sensor bias and persistent material properties. These are not the same thing as the branch-local dynamic state, nor the same thing as time-varying exogenous disturbances.

The hypothesis is that an explicit stable-instance / branch-dynamics factorization will reduce counterfactual comparison noise and preserve heterogeneous individual effects better than (a) a single point-estimate latent or (b) independent posterior sampling for each action branch. This is an experiment hypothesis, not an established property of HDC, CfC or LTC.

## 2. The critical distinction: one individual, multiple hypotheses

Let h_i be the factual history of individual i. Let u_i denote its unknown but stable characteristics, and let:

q_phi(u_i | h_i)

be the model's posterior over those characteristics at the frozen branch point.

A Monte Carlo draw u_(i,k) is one *hypothesis about the same individual*. It is not a new individual. For every paired rollout draw k, the exact same draw must be used for all intervention branches:

u_(i,k) sampled from q_phi(u_i | h_i)

x_(t+1)^(a,k) = F_theta(x_t^(a,k), a_t, u_(i,k), epsilon_t^(a,k))

y_t^(a,k) = G_theta(x_t^(a,k), u_(i,k))

Here:
- u_(i,k) is stable and shared across sibling actions within draw k;
- x_t^(a,k) is branch-local and evolves under that branch's action and observations;
- epsilon_t^(a,k) is process noise or an exogenous event stream, with sharing permitted only where a declared source-to-event correspondence says the events are the same;
- the factual prefix and restored branch point are common; post-intervention endogenous states are not forced to remain equal.

For action pair (a,b), compute the paired effect for each posterior draw:

D_(i,k)(a,b) = U(Y_(i,k)^a) - U(Y_(i,k)^b)

Then aggregate the distribution of D_(i,k) over k to describe epistemic uncertainty about this individual's effect. Do not replace this distribution with its mean when the distribution is multimodal or crosses zero.

### Why independent branch sampling is invalid for this estimand

If branch a samples u_(i,k) and branch b independently samples u_(i,j), the contrast becomes a comparison between two different hypotheses about the individual's stable characteristics. It adds between-draw variation to the action contrast and can create, inflate or hide apparent effects.

This is not a claim that independent samples are invalid for every predictive-generation task. It is specifically invalid for the paired estimand that asks: *what changes when the action changes while the individual is held fixed?*

## 3. Keep three kinds of uncertainty separate

| Quantity | Meaning | Coupling rule |
|---|---|---|
| Stable-individual epistemic uncertainty | Which fixed physical/structural characteristics does this same unit have? | Sample once per paired rollout bundle; reuse across sibling actions |
| Branch-local endogenous uncertainty | How does the state evolve after each alternative action? | Evolve independently under each branch's action and observations |
| Exogenous/process stochasticity | Which external disturbances/events occur? | Share only when the task's event-correspondence contract proves they are the same event |

A common seed is not by itself proof of common exogenous events. Use the existing source-to-address-to-draw and replay contracts where the task is stochastic. For deterministic DMC, do not invent a stochastic correspondence requirement.

Likewise, a common individual draw does not mean sharing the entire post-action state trajectory. That would erase the very causal consequences being measured.

## 4. Model architecture hypothesis

The intended division of responsibility is a testable design, not a prescribed implementation:

1. **History inference:** infer a posterior q_phi(u_i | h_i) over stable individual factors, plus a separate branch-point dynamic state.
2. **Typed stable representation:** test HDC for representing typed physical/structural factors and a set or distribution of posterior hypotheses. A single bundled hypervector is not automatically a calibrated posterior.
3. **Branch-local dynamics:** test CfC/LTC for evolving each action-conditioned dynamic state while receiving the same stable individual draw for paired branches.
4. **Outcome and utility heads:** predict observables and the frozen task utility for each branch; derive paired differences within each shared posterior draw.
5. **Evidence layer:** bind history, branch point, posterior-draw identity, model/configuration, action, branch trajectory and independently admitted replay artifacts.

Do not imply that HDC is inherently Bayesian, that CfC/LTC intrinsically factorizes stable and dynamic state, or that an uncertainty vector is a calibrated posterior. The implementation must expose and test the semantics.

## 5. PCIR v0.1 admission contract

Each paired-rollout bundle must freeze:
- individual identity and factual-history digests;
- restored branch-point state and task/dynamics configuration digests;
- posterior model version, posterior profile and posterior sample-set commitment;
- unique posterior draw IDs and a canonical digest for each draw;
- the exact action set and intervention-pair IDs;
- one branch record for every required action under every posterior draw;
- action-specific branch-local initial state, rollout trajectory and outcome commitments;
- the shared exogenous manifest and the applicable event-correspondence evidence;
- evaluation split, seed schedule, utility/loss scale and evaluator profile.

For every (individual, posterior_draw_id):
- exactly one stable-parameter draw commitment exists;
- every required sibling action references that same draw ID and draw digest;
- the factual history, restored branch point and model/configuration are identical;
- action IDs are distinct and complete;
- branch-local trajectories are allowed to diverge after action application;
- a missing, duplicate or mismatched branch remains a failed/unknown case, not a zero effect.

The adapter must validate these records before calling #13520. #13520 remains the finite effect-arithmetic consumer; this protocol does not create a second effect scorer or authenticate receipts by hashing alone.

### Posterior update rule

The initial posterior draw is frozen at the shared branch point for a paired rollout bundle. Branch-specific observations may update a branch-local posterior *belief* if the tested model is explicitly filtering online, but they must not silently replace the bundle's sampled stable individual with a different draw. If the experiment uses branch-specific re-sampling, it must define a different estimand and must not enter this PCIR paired-effect result.

## 6. Required ablations and negative controls

All comparisons use identical individuals, histories, branch points, interventions, horizons, utility definitions, posterior sample counts and compute accounting.

| Arm | Stable individual representation | Cross-action posterior coupling |
|---|---|---|
| A0 | Zero-effect / population-mean reference | Not applicable |
| A1 | Branch-point-only predictor | Not applicable |
| A2 | Point-estimate stable latent | Deterministic common point |
| A3 | Posterior stable latent | Correct shared draw per sibling set |
| A4 | Posterior stable latent | Deliberately independent draw per action (negative control) |
| A5 | Full hidden-parameter oracle in simulator | Exact common parameter |
| A6 | HDC-only representation | Shared posterior draw |
| A7 | CfC/LTC-only representation | Shared posterior draw |
| A8 | HDC + CfC/LTC factorization | Shared posterior draw |

Where faithful and budget-matched implementations exist, add GRU/RSSM and other strong recurrent world-model baselines. Do not treat a paper's RSSM or another environment's score as a matched result.

Required semantic controls:
- **Same-individual control:** changing a posterior draw ID for one sibling action rejects the pair.
- **Branch-divergence control:** same stable draw with different post-action states is accepted.
- **Independent-sampling negative control:** distinct draw IDs across siblings are rejected for the paired estimand.
- **Multimodality control:** equal-weight positive and negative effect modes must not be collapsed into a misleading zero-effect point estimate.
- **Aleatoric-separation control:** event correspondence is checked independently of stable posterior coupling.
- **No-support control:** missing or out-of-support posterior hypotheses produce UNKNOWN / insufficient-support, not a confident point prediction.
- **Identity control:** a full-state or known-parameter simulator oracle provides an upper reference, not a deployable model.
- **Split control:** all branches and posterior draws for one individual remain in the same train/validation/test individual partition.

## 7. Metrics: retain distributions, not only means

Report the following separately:
- Posterior quality for hidden stable parameters where simulator truth is available: coverage, bias, dispersion and posterior predictive checks.
- Individual-effect fidelity: within-pair-centred MAE/RMSE, signed bias, sign accuracy, within-pair correlation, slope/intercept, worst-bin bias and performance against the zero-effect baseline.
- Uncertainty quality: interval coverage and width for the individual effect, calibration by predeclared individual strata, and the fraction of posterior mass on each side of zero.
- Latent-to-rollout retention: A_lat, A_pred, and losses at each transition where the Okita decomposition is identifiable above its estimator floor.
- Decision quality: worst-case and expected regret, action-ranking stability, and common epsilon-good action-set preservation through #13239/#13152.
- Closed-loop value: task success, progress, recovery, constraint violations, failed/late application denominator and performance under distribution shift.
- Cost: actual model bytes, posterior sample count, simulator interactions, p50/p95/p99 inference/application latency, accelerator/CPU time and measured energy where calibrated instrumentation exists.

Do not create one weighted composite leaderboard score. Report the Pareto frontier over effect fidelity, uncertainty quality, decision regret, closed-loop outcome and resource cost.

## 8. Benchmark sequence

### Stage 0 — semantic and arithmetic qualification

Run the machine-readable fixtures in the companion JSON file. These validate pairing semantics only; they are not empirical evidence.

### Stage 1 — DMC mechanism extension

Keep #13530 Track A separate: reproduce the paper's DMC environment-eligibility and masking diagnostic only after recovering its exact supplementary configuration and estimator.

Track B adds posterior coupling as a new Symthaea experiment on the recovered DMC individual parameterization. Compare A0–A8 on held-out individuals, held-out intervention pairs, and the joint holdout. The paper did not report a trained DMC world-model checkpoint; no Track B result can be called a DMC model replication.

### Stage 2 — second simulator family

Only after the DMC mechanism is qualified, select a second branchable task with a different causal structure (for example, a contact-rich manipulation task) and re-audit action efficacy, restoration, individual dependence, measurement floor and convergence. Do not assume all tasks in a simulator family qualify.

### Stage 3 — closed-loop embodied value

Use matched implementations and task protocols for World-In-World, RoboDyna/LIBERO-Dynamic and/or WorldArena 2.0 where the official assets and execution path can be recovered. Compare direct policy, full-state reference, point-estimate world model, posterior-coupled model and compute-matched strong baselines. Include MRS.Q-style search-overestimation controls. Rollback is prior art for rewind/branch/action-chunk search; it is a comparator, not a novelty claim.

### Stage 4 — hardware-in-loop

Require measured restoration residuals, independently checked reset evidence, actuator application timestamps and a declared physical tolerance. A single physical unit cannot reveal both exact individual potential outcomes without a valid intervention/restoration design. Where exact unit-level counterfactual restoration is impossible, use randomized matched units and report population/subgroup effects instead of asserting exact per-unit ground truth.

## 9. Statistical and calibration ownership

- Use #8578 for individual/episode cluster-level inference; fifteen action pairs from one DMC individual are not fifteen independent subjects.
- Reuse #11655/#11659 for simultaneous calibration of expected-loss intervals across the frozen action × horizon scope when its assumptions and reference quantities match. Do not create a second conformal calibration implementation here.
- Keep posterior parameter coverage distinct from expected-loss interval coverage. One does not imply the other.
- Reuse #13239/#13152 for finite-set decision sufficiency and regret. Do not duplicate its minimax-regret oracle.
- Reuse #13202 for robust outcome-support bounds where the uncertainty is over candidate-visible observation outcomes. PCIR's stable-parameter posterior is a different uncertainty object; the two may be composed, not conflated.
- Reuse #9029/#13413 for evidence-gated model use and dimension-wise adequacy. A PCIR posterior is not self-authorizing.
- Reuse #13375/#13377 and Symtropy #3741/#3743 only under their actual branch/stochastic contracts. #13377's four-arm DVPC validator is not a generic two-action verifier.
- Keep #9437 evaluator isolation mandatory.

## 10. Prior art and novelty boundary

This protocol does not claim novelty for:
- latent-variable world models or posterior inference;
- contrast-aware counterfactual training objectives;
- paired intervention comparisons or common random numbers in simulation;
- branching, rewind or alternative-action search;
- uncertainty-aware HDC in general.

Relevant prior art includes Okita's distinction between individual effects retained in a latent and those expressed in imagined rollouts; RSSM-style deterministic/stochastic state factorization; contrast-aware intervention loss; Causal-JEPA's object-level latent intervention; MRS.Q's warning that search can underperform despite model accuracy; World-In-World's closed-loop embodied evaluation; and Rollback's branch/backtracking experience. PCIR's contribution is a specific **testable protocol and architecture hypothesis** for coupling posterior hypotheses about one stable individual across sibling action rollouts, keeping that posterior separate from branch-local dynamics and event-level stochastic correspondence, and measuring the effect on decisions.

That claim remains provisional until the negative controls and held-out ablations demonstrate a measurable advantage over point-estimate and independently sampled baselines.

## 11. Supplementary-source recovery for DMC Track A

The Okita article links a supplementary file that includes section **S3, Simulator configuration**. This is a concrete recovery lead and should be inspected before treating simulator settings as irrecoverable. The file is linked from the article's Supplementary Materials section (https://www.mdpi.com/2504-4990/8/11/363#supplementary; supplementary asset https://www.mdpi.com/article/10.3390/make8110363/s1).

The current tool-side retrieval attempt could confirm the published link and S3 description but could not retrieve the binary asset. Therefore this protocol does **not** claim that S3 contains every missing DMC input or that the exact DMC formulation is recovered. After obtaining it, make a field-by-field coverage table against #13530's unresolved-input list: exact values, partially specified values, and still-missing values must be separated. Author correspondence may still be required for undocumented seeds, mask ordering, action sequence serialization or estimator code.

## 12. Explicit non-claims

This v0.1 change adds only a research contract and semantic fixtures. It does not implement:
- posterior inference, a Bayesian HDC encoder or a CfC/LTC trainer;
- paired posterior sampling in an executing model;
- simulator adapters, branch execution or restoration;
- a generic two-branch verifier or a stochastic receipt verifier;
- effect scoring, calibration, statistical inference or a decision authority.

No DMC replication, trained model, HDC/CfC benefit, closed-loop gain, physical result, resource advantage or SOTA improvement is claimed.

## Sources

- Okita, *Where Individual-Effect Information Survives: A Diagnostic Ladder for World Models* (2026), https://doi.org/10.3390/make8110363
- Okita supplementary material (includes S3 Simulator configuration), https://www.mdpi.com/article/10.3390/make8110363/s1
- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet), https://arxiv.org/abs/1811.04551
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), https://arxiv.org/abs/2301.04104
- *Causal-JEPA: Learning World Models through Object-Level Latent Masking* (ICML 2026), https://proceedings.mlr.press/v306/
- *The Surprising Difficulty of Search in Model-Based RL* (MRS.Q, ICML 2026), https://proceedings.mlr.press/v306/chang26a.html
- *World-In-World* (ICLR 2026), https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- Rollback (2026), https://arxiv.org/abs/2610.36068
- WorldArena 2.0, https://world-arena.ai/
