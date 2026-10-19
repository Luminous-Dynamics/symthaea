# OpenAI Mathematics Transfer Map for Symthaea

**Status:** research plan and source triage; not a proof certificate or completed benchmark.  
**Date:** 2026-10-19  
**Upstream snapshot:** `openai/math@1581d4cbdd69a0faf34b2089614fe2a814316a6a`  
**Existing work to extend:** [proof-guided math corpus RFC #7585](https://github.com/Luminous-Dynamics/symthaea/pull/7585), [Gaussian action-information reference #6210](https://github.com/Luminous-Dynamics/symthaea/issues/6210), [action-conditioned uncertainty #6216](https://github.com/Luminous-Dynamics/symthaea/issues/6216), [LTC stability/free-energy prerequisites #5723](https://github.com/Luminous-Dynamics/symthaea/issues/5723).

## Why this work

The opportunity is to turn mathematical knowledge into better decision semantics and independently checkable software contracts. The unit of progress is not imported theorems or lines of Lean. It is a reproducible chain:

`candidate theorem -> exact assumptions -> Lean kernel check -> reference fixtures -> production correspondence -> measured behavior`

Each arrow is a distinct obligation. Retrieval, numerical agreement, a plausible manuscript, or a queued workflow never substitutes for a kernel-checked proof or implementation test.

OpenAI's public collection itself warns that its manuscripts have different verification status and that not all are formalized. Treat the upstream repository as a versioned source of candidates, not a trust oracle.

## Prioritized workstreams

### P0 — Action-conditioned Gaussian entropy (extend the active pilot)

**Target:** #6210/#6211/#6212/#6214/#6216 and the existing proof-corpus proposal. Extend the current work; do not create a second oracle or proof lineage.

For a positive finite diagonal precision vector `pi` of dimension `d`, freeze the software reference profile:

`H(pi) = 1/2 * (d + d*ln(2*pi_const) - sum_i ln(pi_i))`

`E(a) = H(pi_pred[a]) - H(pi_current)`

Under the declared minimization convention, more predicted-entropy reduction gives a more negative `E(a)`. The source profile must be named `DiagonalGaussianPredictedEntropyDeltaV1`, not expected information gain. This is a reference-model proxy, not a universal definition of epistemic value.

The existing [source PR #6211](https://github.com/Luminous-Dynamics/symthaea/pull/6211) correctly distinguishes predicted hidden-state entropy change from expected information gain / mutual information and uses `CommittedAction` for generic-FEP novelty. The independent qualifier lineage [#6214](https://github.com/Luminous-Dynamics/symthaea/pull/6214) and hardening follow-up [#7367](https://github.com/Luminous-Dynamics/symthaea/pull/7367) already cover the reference semantics and hostile controls. They remain qualification work: an authored validator or queued workflow is not a hosted PASS.

Candidate source files at the immutable upstream revision:

- [`OAI/Analysis/LpDimension/Entropy.lean`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Analysis/LpDimension/Entropy.lean), blob `b2b32a01166f7990231a1d16240b1f80269d9114`.
- [`OAI/Probability/SKRatio/Entropy/Entropy.lean`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/OAI/Probability/SKRatio/Entropy/Entropy.lean), blob `76e87cc6ef21186511d27d932cff0cb6e303a304`.

Adapt only the lemmas that match the exact assumptions and the already-approved Lean environment; do not copy an upstream file wholesale or add a second Lake/Mathlib project. Required theorem families: equal precision gives equal entropy; componentwise precision increase is non-increasing; at least one strict increase yields strict decrease; finite-sum/permutation invariance.

**Falsification fixtures:** wrong sign; omitted positivity; zero/negative/non-finite precision; equality incorrectly accepted as strict decrease; wrong EFE orientation; accidental generalization to arbitrary covariance; altered theorem statement; stale environment. Test the real-number theorem and f64 implementation separately.

**Gate:** zero accepted proof holes/unauthorized axioms; every artifact binds theorem/source/environment/checker identities; all reference cases and mutation controls checked. A Lean PASS alone does not show uncertainty calibration or action optimality.

### P1 — True finite categorical expected information gain (research-only)

**Target:** [FEP-EFE-ACTION-002 / #6223](https://github.com/Luminous-Dynamics/symthaea/issues/6223). This is a distinct model profile—not a rename or silent upgrade of the P0 entropy proxy.

For latent state S, future observation O, and candidate action a, freeze both equivalent finite-model definitions:

`EIG(a) = sum_o q(o|a) * KL(q(s|o,a) || q(s))`

`EIG(a) = H(q(s)) - sum_o q(o|a) * H(q(s|o,a))`

where `q(o|a)=sum_s q(s)q(o|s,a)` and a posterior is formed only when `q(o|a)>0`. In the EFE minimization convention, an information contribution can be `E_info(a)=-EIG(a)`, which must be explicitly recorded rather than inferred from the name “epistemic”.

**Candidate corpus:** [`fep-efe-action-002a-reference-candidate-v1.json`](./data/fep-efe-action-002a-reference-candidate-v1.json), with 34 cases (22 valid models, 12 invalid-domain/shape cases) and 10 explicit semantic mutation controls. It covers uninformative and revealing observations, noisy sensors, skewed priors, many-to-one partitions, impossible branches, alphabet relabeling, normalized-distribution failures, non-finite values, and dimensional mismatch.

**Exploratory arithmetic check:** on the fetched branch bytes, an independent transient JavaScript check evaluated EIG through both expected-KL and entropy-reduction forms. Their maximum absolute difference was approximately 1.12e-16; the maximum discrepancy from the authored simple symbolic expressions was approximately 2.23e-16. All six encoded metamorphic comparisons held, and all 12 invalid fixtures exposed a domain/shape rejection. This is only an exploratory arithmetic/source check—not the #6223 independent stdlib oracle, Lean proof, hosted qualification, Rust execution, or production result.

**Qualification boundary:** follow #6223's existing order. Freeze/review the candidate profile and corpus; qualify it with an independent oracle that computes both forms without case-ID answer lookup; add hostile semantic mutations; then formalize theorem-sized properties in the approved SYM-FV environment. Do not begin production integration merely because the fixtures look right. Hypothetical posterior branches must not mutate live beliefs or become evidence, and model calibration / physical information gain remain separate.

### P1 — Relative entropy as a discrepancy contract for beliefs

The pinned finite-law file defines relative entropy and proves non-negativity for finite normalized distributions with nonnegative numerator mass and strictly positive comparison mass. It also contains entropy-variational and Hellinger/total-variation comparison results.

Explore these as contracts for comparing finite belief distributions, policy counterfactuals, posterior approximations, and transition summaries. First establish which distributions each production API actually emits and whether normalization/strict-positivity preconditions hold. Do not label KL divergence as calibrated uncertainty or as mutual information without proving the required interpretation.

**Experiments:**
- exact identical distributions -> zero KL and identical distribution identity;
- non-identical normalized distributions -> nonnegative KL under valid support assumptions;
- zero/negative mass, missing support, NaN/Inf, normalization drift -> typed rejection or an explicitly different supported profile;
- compare Rust output with independent high-precision reference vectors;
- compare policy ranking with raw-component and counterfactual-weight ablations.

Prefer relation/interval assertions over brittle last-bit transcendental equality.

### P2 — Proof-guided retrieval as an empirical capability

Continue #7585 rather than making an unrelated retrieval subsystem. Benchmark the frozen corpus and held-out theorem-family split using:
1. no-retrieval baseline;
2. deterministic lexical/type/operator retrieval;
3. the pinned LeanSearch v2 reference where a fair pinned run is feasible;
4. HDC reranking;
5. hybrid retrieval.

Measure premise-group Recall@1/5/10, nDCG@10 where relevant, downstream proof closure, p50/p95 runtime, timeout rate, peak memory, dependencies, and exact checker results. Split by theorem family and source to prevent leakage. The proposed initial goal is **+10 percentage points absolute held-out proof closure** over a frozen baseline, with unchanged proof-acceptance gates. This is a target, not a result.

LeanSearch v2 reports results on a 69-query research-level Mathlib benchmark; that is a published external result, not reproduced evidence for Symthaea. [Paper](https://arxiv.org/abs/2605.13137) · [code and benchmark](https://github.com/frenzymath/LeanSearch-v2).

### P3 — LTC stability and candidate Lyapunov functionals

Use the existing SYM-FV/LTC work and #5723. For the exact idealized vector field, define a candidate functional `V(x)`, derive `dV/dt = grad(V) dot f(x)`, and prove the sign only under explicit parameter and state-domain assumptions. Establish invariant sets and convergence only when the proved result permits it.

Do **not** infer LTC stability from an entropy theorem for a different graph heat-flow system. Keep these claims distinct:

- candidate functional exists != Lyapunov function;
- non-increasing != strictly decreasing;
- strict decrease != unique equilibrium;
- idealized ODE stability != discrete f32/f64 integration stability;
- mathematical convergence != scientific/cognitive validity.

**Benchmarks:** stable/unstable parameter sweeps around analytic boundaries; boundedness and invariant-domain checks; step-size refinement; divergence and equilibrium mutation controls; compare analytic sign predictions against independent numerical trajectories.

### P4 — Numerical certificates for transcendental functions

Extend existing interval/certificate work (including the exp-related direction in [PR #6865](https://github.com/Luminous-Dynamics/symthaea/pull/6865)) toward the `ln` terms used in entropy scores. A useful result is a certified interval for each score and a ranking only when intervals separate; overlap means `TieOrUndetermined`, not a forced winner.

**Benchmarks:** sound enclosure on a frozen input grid; zero false-separated rankings; seeded near-ties; underflow/overflow and extreme-domain rejection; certificate replay with exact source/rounding identities. Keep the interval algorithm itself under test and do not use approximate agreement as proof authority.

## Second-pass scan: additional formalized upstream families

The pinned upstream formalization catalogue at [`lean/formalization.yaml`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/formalization.yaml) lists further candidates beyond entropy:

- **Memory–sample lower bounds for noiseless Gaussian regression** ([formalization scope](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/docs/140.md), blob `dbabe31e00bf52ef24a4633980ff3794fb1e8127`). The formalized model studies exact noiseless Gaussian observations and bounds how many observations are required at a given memory budget and angular accuracy. Potential use: an impossibility/feasibility lens for memory-constrained inference and representation compression. Its assumptions are too specific to directly establish lower bounds for Symthaea's HDC encodings or learned, noisy, non-Gaussian inputs. First task: translate a narrowly matched synthetic observation model and test whether our memory accounting matches the theorem's bit model.
- **Subpolynomial query complexity for well-conditioned log-concave sampling** ([formalization scope](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/docs/139.md), blob `323bb69fdd61aaa00418330c794a19f6b2efc368`). Its conditions include a potential with `V(0)=0`, `grad V(0)=0`, and `I <= Hessian(V) <= 2I`; the stated result concerns value/gradient query complexity and total-variation accuracy, not wall-clock cost or bit complexity. Potential use: study sampling-based inference only for explicitly convex/well-conditioned synthetic models. First task: check whether a target distribution actually satisfies those assumptions before comparing query budgets.
- **Posterior replicas and conditional information in Gaussian regression** (catalogued with the same family as the memory–sample lower bounds) may inform independent-sample and information-accounting tests. It does not establish posterior calibration for an arbitrary production agent.

These are useful because they expose resource limits and model assumptions—not because their conclusions automatically transfer. They remain lower priority than fixing #6216's already-characterized action-invariant epistemic score.

## Broader math-transfer opportunities

| Priority | Candidate mathematics | Symthaea capability | First falsifiable result |
|---|---|---|---|
| P0 | Log monotonicity, concavity, finite sums | Action-conditioned uncertainty | Entropy ordering and strictness fixtures |
| P1 | Relative entropy and probability inequalities | Belief comparison and policy evaluation | Non-negativity plus support/normalization hostile cases |
| P1 | Global premise retrieval and proof planning | More proofs closed per compute budget | Held-out retrieval-to-proof benchmark |
| P2 | Lyapunov functions, invariant sets, differential inequalities | LTC stability claims | Exact sign theorem and parameter-boundary counterexamples |
| P2 | Interval arithmetic and remainder bounds | Trustworthy real-number rankings in f64 code | Sound score enclosures and abstention on overlap |
| P3 | Matrix norms, conditioning and perturbation bounds | Linear algebra, model sensitivity, geometry/robotics interfaces | Bound output drift under bounded input/model perturbation |
| P3 | Concentration and posterior information bounds | Uncertainty-aware inference under sampling | Coverage/calibration study against a stated data-generating model |
| P3 | Convexity, minimax and potential functions | Constrained planning/resource allocation | Feasibility and optimality gaps on small exact instances |

P3 items are research candidates, not claims that the upstream OpenAI collection has already supplied drop-in formal proofs for Symthaea. Search and review a specific pinned theorem before promoting any item to an implementation requirement.

## Shared benchmark contract

Every transfer should have:
- an immutable upstream commit, path, blob digest and citation;
- a written target statement independent of the upstream proof;
- explicit domain, units, conventions, and strict/weak inequalities;
- an approved Lean environment identity and kernel output;
- known-answer cases plus hostile mutations;
- an independent reference where numerical behavior is involved;
- exact source/head and test/tool evidence;
- a claim ceiling stating what the result does *not* establish.

No `sorry`, `admit`, unauthorized axiom, changed statement, hidden premise, numerical-only proof substitute, or promotion from queued/partial execution is accepted. The qualification pipeline is not complete until it executes on the exact source head.

## Execution order

1. Reconcile this map with #7585 and #6210–#6217; do not duplicate their proof environments or rewrite unqualified source.
2. Finish the entropy-delta theorem and reference fixtures first because the downstream behavior defect is already characterized.
3. Add distribution-divergence contracts using the same approved substrate.
4. Run the retrieval ablation on held-out families.
5. Pursue Lyapunov and interval-certificate tracks only with frozen statements and explicit negative controls.
6. Promote nothing to production authority based on this document alone.

## Source register

- OpenAI math snapshot: [repository at pinned revision](https://github.com/openai/math/tree/1581d4cbdd69a0faf34b2089614fe2a814316a6a).
- Formalized-result catalogue at the same revision: [`lean/formalization.yaml`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/formalization.yaml), blob `275eee8cfab7ce49d3a5e88f02cbb8f28fe6fee6`.
- Gaussian memory/sample and sampling assumptions: [`lean/docs/140.md`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/docs/140.md), [`lean/docs/139.md`](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6/lean/docs/139.md).
- Existing Symthaea research: [#7585](https://github.com/Luminous-Dynamics/symthaea/pull/7585), [#6210](https://github.com/Luminous-Dynamics/symthaea/issues/6210), [#6216](https://github.com/Luminous-Dynamics/symthaea/issues/6216), [#5723](https://github.com/Luminous-Dynamics/symthaea/issues/5723).
- Premise retrieval baseline: [LeanSearch v2 paper](https://arxiv.org/abs/2605.13137), [implementation/benchmarks](https://github.com/frenzymath/LeanSearch-v2).

**Validation status:** document/source triage only. This commit does not execute Lean, run retrieval, compile Rust, verify production correspondence, or establish any theorem or benchmark PASS.
