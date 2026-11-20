# SC-ACTIVE-RCT-ALLOCATION-0.1

**Status:** exact bounded synthetic allocator for selecting candidate matched aliquot blocks for future randomized experiments. It does not select real specimens or execute randomization.  
**Research issue:** [#12987](https://github.com/Luminous-Dynamics/symthaea/issues/12987)  
**Causal-identification contract:** [#12969](https://github.com/Luminous-Dynamics/symthaea/issues/12969)  
**Intervention planner:** [#12968](https://github.com/Luminous-Dynamics/symthaea/pull/12968)  
**Risk/resource planner:** [#12965](https://github.com/Luminous-Dynamics/symthaea/pull/12965)

## The research shift

The allocator answers a different question from experiment selection:

- #12966 / #12968: which intervention should be applied, and in what sequence, to change a specimen state?
- #12969: when is a causal effect actually identified by treatment assignment, controls and specimen/batch evidence?
- **This contract:** which eligible matched specimen pairs should receive scarce randomized treatment/control allocation next, to improve the causal estimand and downstream decisions?

The target is not generic predictive uncertainty. It is uncertainty in the family-specific causal response and the application-weighted treatment effect, subject to randomization positivity, family representation, batch interference limits, specimen consumption and instrument/budget capacity.

## Implemented objective set

The Rust binary \`sc_active_rct_allocation_audit\` exhaustively enumerates feasible subsets of candidate matched-pair blocks and evaluates:

1. **Joint family-CATE-state information** — mutual information between the selected paired outcomes and the joint discrete family-effect state.
2. **Focused application-weighted ATE information** — mutual information between selected paired outcomes and the preregistered application-weighted finite-pool ATE.
3. **ATE precision** — expected reduction in posterior variance of that ATE.
4. **Decision EVSI** — reduction in Bayes-optimal downstream decision loss under the explicit application decision catalogue.
5. **Generic predictive entropy** — sum of each candidate's marginal outcome entropy, a deliberately simple generic-uncertainty comparator that can prefer noisy outcomes.

The first two are **finite discrete analogues inspired by the objective alignment in Causal-EPIG**, not faithful reproductions of its posterior-predictive joint-potential-outcome or focused-CATE acquisition algorithms. Here each family has a small discrete response-state posterior and each candidate provides a three-outcome likelihood conditional on its family state.

Greedy baselines use marginal objective gain per financial cost. A separate budgeted-active heuristic combines focused ATE information with a fixed 0.20 overlap-deficit bonus and a 0.20 domain-discrepancy bonus per selected block. This is a transparent **Gao-inspired surrogate**, not a reimplementation of the ICML 2026 algorithm or its theory.

The auditor also reports a deterministic 64-draw family-stratified random-feasible baseline and exact oracles for every objective.

## Candidate block and randomization semantics

An eligible candidate is a prequalified matched pair of two distinct aliquots. It must have:
- a treatment-capable aliquot and a control-capable aliquot;
- a qualified matching declaration;
- a declared assignment probability inside the frozen [0.20, 0.80] positivity gate;
- specimen, batch and material-family identity;
- historical treatment propensity and domain discrepancy as observed-design diagnostics;
- complete resource consumption and a likelihood row for every possible family response state.

The allocator chooses pairs only. The receipt describes a within-pair design that randomizes which aliquot receives treatment (and which control) using the declared probability. Every allocation record is marked \`design_only_not_executed\`. The synthetic auditor does not draw a real-world allocation, reserve a facility, direct a laboratory or authorize intervention.

Historical propensity and new randomized assignment probability are separate fields. In the fixture, historical propensities can be extreme while the new pairwise randomized assignment remains 0.5. The former measures historical support deficit; it must never silently become the new assignment probability.

## Hard constraints

The exact subset oracle enforces:
- minimum and maximum selected block count;
- minimum and maximum blocks per material family;
- maximum blocks per batch;
- the preregistered randomization probability range;
- distinct specimen identities across candidate blocks;
- matched-pair, treatment and control eligibility;
- every resource capacity independently (financial budget, cryostat minutes, specimen micrograms and magnetometer minutes).

The three family representation minima are hard constraints, not soft reward terms. If one family has no eligible matched pair, the auditor returns no feasible allocation rather than spending the entire budget on easier families.

## Frozen fixture

The main fixture has:
- 3 material families;
- 3 discrete causal-response states per family;
- 27 joint family-response hypotheses;
- application weights 0.60 / 0.30 / 0.10;
- 9 eligible matched-aliquot blocks;
- one explicit no-control negative control;
- 4 vector-resource limits;
- minimum one randomized block per family and maximum two per family;
- a maximum of two selected blocks from any one synthesis batch;
- a maximum of four selected blocks overall.

Three acquisition objectives deliberately select different plans. An independent arithmetic cross-check of the frozen fixture gives:

| Exact objective | Selected blocks | Objective value |
|---|---|---:|
| Joint family-CATE-state information | alpha high, alpha medium, beta medium, gamma high | 1.04316 nats |
| ATE variance reduction | alpha high, alpha medium, beta high, gamma medium | 0.95604 loss-free variance units |
| Decision EVSI | alpha low, beta high, beta medium, gamma high | 0.58065 loss units |

The joint-information plan reduces the weighted ATE posterior variance by about 0.88407. The ATE-precision plan reduces it by about 0.95604. The decision-EVSI plan produces the largest decision-loss improvement in the fixture. These are calculations from invented likelihoods, priors, losses and capacities—not superconductivity evidence.

The held-out joint-world identifier exists only for benchmark integrity. It is validated against the world registry but is not stored in the planning model and is never used in acquisition. A test changes the held-out identifier to a different valid world and asserts that all acquisition plans remain identical.

## Frozen negative controls

The registered suite is \`data/research/superconductivity_active_rct_allocation_suite_v0_1.json\`:

1. **Main objective-divergence fixture** — distinct information, precision and decision objectives; one unmatched no-control candidate is excluded.
2. **Randomization positivity violation** — one otherwise-eligible pair has assignment probability 0.90, outside the frozen [0.20,0.80] gate. The auditor must reject the instance.
3. **Missing family support** — every candidate in one family is marked ineligible. The auditor must return no feasible allocation rather than violate the family minimum.

All family-effect states, world priors, candidate likelihoods, propensities, losses and costs are synthetic.

## Literature and matched SOTA comparisons

- **Causal-EPIG** (Gao, Fawkes & Sejdinovic, ICML 2026) explicitly aligns active acquisition to potential outcomes / CATE and distinguishes joint potential-outcome information from focused CATE information. The present finite discrete implementation is a benchmark analogue, not a reproduction: https://proceedings.mlr.press/v306/gao26i.html
- **Budgeted Active Experimentation** (Gao et al., ICML 2026) combines uplift uncertainty, domain discrepancy and overlap deficits to select scarce randomized samples, and reports theoretical bounds plus synthetic and industrial neural-uplift studies. The current overlap/discrepancy greedy score is only a transparent comparator surrogate: https://proceedings.mlr.press/v306/gao26z.html
- **GO-CBED** (Zhang et al., TMLR 2026) learns non-myopic intervention sequences for a specified causal query. It is relevant to intervention planning (#12968), but it selects interventions, not candidate matched aliquot blocks: https://uq.engin.umich.edu/causal-learning.html

The next matched benchmark should implement or independently reproduce the published acquisition functions and compare them on identical finite SCMs, candidate pools, assignment budgets, randomization seeds, block constraints, held-out worlds and causal-estimation procedures. This v0.1 does not claim superiority over these papers or over an industrial lab-allocation system.

## Next benchmarks

Expand from 9 candidate blocks / 27 worlds to 40–100 synthetic aliquot units over 3–5 families and at least 4 batches. Add:
- heterogeneity in treatment effect by family and baseline phase;
- pressure × strain interaction and potential-outcome covariance;
- source/lab/instrument domain shift and historical targeting;
- shared-batch interference and cluster randomization;
- missing-not-at-random outcomes and attrition;
- a randomized audit fraction and sequential allocation rounds;
- exact small-pool allocation regret plus Monte Carlo evaluation on larger pools;
- ATE/CATE bias, RMSE, coverage, randomization type-I error/power, family support, cost per qualified contrast and downstream regret.

Aspirational targets (not achievements): exact-oracle agreement on every bounded instance; zero positivity or representation violations; at least 20% lower ATE/CATE RMSE or interval width at matched RCT budget without coverage loss; and at least 20% lower downstream decision regret against the strongest matched published baseline.

## Claim ceiling

This is a software and synthetic benchmark only. No real specimen is selected, no randomization is carried out, and no superconductivity intervention effect, material discovery, instrument reservation, laboratory procedure or safety approval is established.