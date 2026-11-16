# EGDDM ShiftBench D6.16 — Adaptive risk-constrained evidence-acquisition policies v0.1

**Date:** 2026-11-16  
**Status:** evaluator-only exact finite-POMDP prototype; synthetic known-answer fixture; no external agent-performance claim  
**Related work:** [Issue #12486](https://github.com/Luminous-Dynamics/symthaea/issues/12486), D6.7 impact/cost oracle, D6.11 identifiability, D6.13 whole-DAG worlds, D6.14 evidence handoff, D6.15 fixed-trace acquisition.

## Why this exists

D6.15 scores a supplied fixed acquisition trace and a fixed-outcome minimum-sufficient set. It cannot score a conditional policy such as "run check A; if it passes, act; if it fails, query B; choose a branch-specific repair."

Adaptive Feature Acquisition is established research, not a new invention. The [IJCAI 2026 survey](https://ijcai.org/proceedings/2026/860) formulates active feature acquisition as a POMDP and organizes cost-sensitive trees, model-based, model-free and hybrid methods. [NM-PPG](https://arxiv.org/abs/2605.05511) is a recent non-myopic pathwise policy-gradient baseline. [RigorBench](https://arxiv.org/abs/2606.22678) already covers process planning, verification, recovery, abstention and atomic transitions; D6.16 must report those adjacent axes rather than claiming process-discipline novelty. [Informed abstention evaluation](https://arxiv.org/abs/2606.02965) establishes positive usability controls alongside hazardous-action blocking.

D6.16's bounded research distinction is the engineering assurance contract around policy branches: evaluator-authoritative hard risk, authority and budget; time-admissible observations; exact posterior reconstruction; complete branches; independently supplied cost distributions; and an exact small-instance policy oracle.

## v0.1 model and implementation

- Hidden world is fixed during one episode with evaluator-owned prior weights.
- Each acquisition has a world-conditional observation-likelihood distribution in integer basis points. Sequential observations are conditionally independent given the hidden world in v0.1.
- Query costs use evaluator-supplied cold-run samples and nearest-rank p95.
- Candidate policies are finite trees with \`Acquire(option)\`, \`Act(action)\` and \`Defer(reason)\` nodes.
- The evaluator reconstructs the posterior at every branch; candidate branch keys must exactly cover all outcomes possible under the current belief.
- Hard-gated and irreversible actions require zero posterior mass on unsafe worlds. Reversible actions may use a trusted residual-risk envelope. Deferral carries an evaluator-trusted opportunity-cost penalty, so always-abstain is not optimal by default.
- Exact finite-horizon dynamic programming searches acquisition choices for at most eight worlds and eight options, respecting per-path budgets and a maximum acquisition depth. It is exact only for this bounded model and supplied likelihoods/costs.

## Synthetic known-answer fixture

Three worlds: safe (60%), hazard-A (20%) and hazard-B (20%). Each of three consequential actions is safe in exactly one world. Four queries:
- \`verify-release\`: p95 cost 500; perfectly separates safe from hazard worlds.
- \`classify\`: p95 cost 1,500; identifies safe / A / B.
- \`screen\`: p95 cost 300; noisy and insufficient for irreversible release.
- \`redundant-audit\`: p95 cost 200; produces the same result in every world.

Six policy controls:
- **Adaptive verify → classify:** on pass, release; on fail, classify and choose repair A/B. Expected cost 1,100.
- **Fixed classifier:** always use the single classifier. Expected cost 1,500.
- **Greedy screen-only:** cheap screen, then release on green despite 5,800 bps of branch probability still containing unsafe worlds.
- **Full static acquisition:** screen, verify and classify on every path. Expected cost 2,300.
- **Always-act:** release with 4,000 bps ex-ante actual unsafe probability and a 10,000-bps hard-gate violation probability.
- **Always-defer:** safe but pays the synthetic 10,000-unit deferral penalty.

The exact oracle should choose \`verify-release\` first and acquire \`classify\` only on the fail branch. The adaptive policy reaches the oracle's expected cost of 1,100, a 400-unit improvement over the fixed classifier and 1,200 below full static acquisition. These are synthetic known-answer calculations, not runner timings, real harm probabilities or agent results.

## Baselines

D6.15 fixed traces / fixed sets; cheapest-first; cost-per-information-gain greedy; myopic value-of-information; cost-sensitive decision trees; exact finite-horizon POMDP / dynamic programming; model-based, model-free and hybrid policies; NM-PPG or faithful implementation; zero-, one-, two- and full-round adaptivity; expert/operator; full revalidation; no revalidation; always-act and always-defer.

Adaptive-greedy may be included only when the objective and observation model are shown to satisfy the relevant adaptive-submodularity assumptions. Complementary checks, common-cause failures, hard risk gates and state-dependent costs can invalidate casual use of those guarantees.

## Metrics and future targets

Publish independent axes: exact-oracle agreement and safe-policy regret; unsupported/unauthorized/actual unsafe action probability (hard exclusions); risk-envelope violations and calibration; expected and p50/p95/p99 acquisition cost; defer penalty; expected query count and maximum depth; adaptivity gap; justified action coverage and unnecessary deferral; budget overrun; branch coverage/dead ends; future-evidence leakage; robustness to noise, missingness, correlation, common cause and distribution shift.

Future goals, not current results:
- Exact-oracle agreement on all registered small enumerable controls.
- Zero hard-gate violations on registered known-answer controls.
- At least 95% justified action coverage on fully evidenced positive controls.
- At least 10% lower expected measured cost than the strongest non-adaptive baseline on held-out cases where adaptivity has demonstrated value.
- At least 10% lower p95 cost than full revalidation at matched safety.
- Independent gold annotation, confidence intervals, cost provenance and negative results published alongside any wins.

## Limits and next gates

v0.1 assumes evaluator-provided priors and likelihoods are admissible, observations are conditionally independent given the hidden world, and query costs do not interact. It is an offline evaluator, not a planner executing real checks. It does not learn likelihoods, calibrate harm, authenticate receipts, enforce production authority, or execute/reconcile side effects. Next: common-cause hidden-state controls, correlated-cost models, blinded agent trials, independent annotation, Rust formatting/build/test/Clippy/CI and external replication. Keep the PR draft until those gates pass.
