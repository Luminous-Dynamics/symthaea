# Question-specific connectome proofreading stopping benchmark V1

**Tracking:** #10720 — SCI-CONNECTOMICS-094
**Composes:** #10656, #10661, #10670, #9204, #10676, #9522, #9587
**Status:** synthetic contract definition; no biological stopping decision or proofreading-cost saving is claimed.

## 1. Thesis

A connectome can be globally stable while a rare, low-degree reconstruction alternative still changes a scientific conclusion. The converse is also possible: graph-wide statistics may continue moving while every admissible reconstruction gives the same answer to a specific question.

**Global completion and question-specific sufficiency are different estimands.**

V1 therefore defines a two-level stopping contract. Global graph-invariant convergence remains a useful planning signal. It is neither necessary nor sufficient for stopping work on a particular scientific question.

## 2. Evidence distinction

Keep four states separate:

- **Global convergence:** the declared graph-invariant trajectories satisfy a frozen asymptotic-change tolerance for the required number of consecutive checkpoint transitions.
- **Admitted-world agreement:** every independently admissible reconstruction world in the declared scope produces the same frozen query decision; disputed and indeterminate worlds prevent agreement.
- **Residual query-flip bound:** a probability sample from a closed, finite candidate frame yields a one-sided upper confidence bound on the fraction of frame units whose adjudicated alternative changes the query decision.
- **Question-specific stop:** the admitted worlds agree, the residual frame is closed, the sampling design is estimable and its upper risk bound is within the pre-registered tolerance.

Never substitute one state for another. A stable global metric is not a query certificate. An admitted set that agrees is not proof that omitted worlds do not exist. A sample bound applies only to its declared frame and sampling design.

## 3. Global-invariant arm

Given checkpoint values `g[j,t]` for invariant `j` at checkpoint `t`, V1 defines the transition drift as:

`d[j,t] = abs(g[j,t] - g[j,t-1]) / max(abs(g[j,t-1]), scale_floor)`.

A transition is globally stable only if every declared invariant has drift at or below the pre-registered tolerance. Global stopping occurs at the earliest checkpoint ending the required number of consecutive stable transitions.

Report invariant identities, units, normalizers, missingness, checkpoint edit counts, tolerance, stable-transition count, and exact source digests. Do not silently remove an inconvenient invariant or restart the window after an unfavourable checkpoint.

## 4. Query-specific arm

Freeze the query specification and decision-loss digests before sampling. For each checkpoint:

1. Preserve every reconstruction world and its independent disposition.
2. Exclude only worlds explicitly adjudicated `INADMISSIBLE`; `DISPUTED` and `INDETERMINATE` remain unresolved and block a stop.
3. Require all `ADMISSIBLE` worlds to agree on the query decision.
4. Freeze a finite residual-candidate frame with source revision, region/task scope, population count, membership digest and an explicit coverage declaration.
5. Draw a probability sample with recorded unit IDs, outcomes, design and inclusion probabilities.
6. Compute a one-sided finite-population upper confidence bound on residual query-flip prevalence.
7. Pass only when world agreement holds, frame coverage is closed, all sample outcomes are adjudicated, and the upper bound is at or below the pre-registered tolerance.

An incomplete frame returns `FRAME_INCOMPLETE`, even when its observed sample has zero query flips. A missing/zero inclusion probability or unknown sample disposition is `NOT_ESTIMABLE`, not a negative result.

## 5. V1 exact finite-population bound

For a closed finite frame of size `N`, a simple random sample without replacement of size `n`, and `x` observed query-flipping units, use the hypergeometric distribution:

`P(X <= x | N, K, n) = sum_h C(K,h) C(N-K,n-h) / C(N,n)`.

The one-sided `(1-alpha)` upper count is the largest integer `K` whose cumulative probability is at least `alpha`. The upper prevalence bound is `K/N`.

For zero observed flips this reduces to:

`P(X = 0 | N, K, n) = C(N-K,n) / C(N,n)`.

Known answer: `N=1000`, `n=300`, `x=0`, `alpha=0.05` gives upper count `K=8`, or `0.8%`. This is below a `1%` tolerance only for a **closed** frame, with no opposing admitted world and no unresolved adjudication.

V1 supports only simple random sampling without replacement. The production adapter must compose #9204's stratified/clustered design and use its actual independent sampling unit, inclusion probabilities, variance estimator and finite-population correction. Do not label the V1 bound as a biological population guarantee.

## 6. Required dispositions

- `GLOBAL_ONLY`: global invariants meet the stopping rule, but the query-specific requirements do not.
- `QUERY_ROBUST_WITHIN_DECLARED_FRAME`: query-world agreement and the closed-frame residual-risk bound both pass; global convergence may still be false.
- `QUERY_UNRESOLVED`: opposing admissible worlds remain, or residual query-flip risk exceeds tolerance.
- `FRAME_INCOMPLETE`: residual candidate-frame coverage is not closed; no query-specific stop is allowed.
- `NOT_ESTIMABLE`: disputed/indeterminate worlds, unresolved sample outcomes, missing/zero inclusion probabilities or invalid design prevent a defensible estimate.

The two stop flags (`global_stop` and `query_stop`) must be preserved independently. The joint disposition is not a composite score.

## 7. Synthetic adversarial corpus

V1 includes four exact cases:

1. **Global convergence, query disagreement:** global metrics stabilize; two admissible worlds disagree. Global-only stopping passes, question-specific stopping fails.
2. **Query sufficiency before global convergence:** graph invariants keep changing; all admissible worlds agree; a closed `N=1000`, `n=300`, zero-flip sample gives a 0.8% upper bound under a 1% tolerance. Query-specific stopping passes while global stopping does not.
3. **Global convergence, incomplete frame:** global metrics stabilize and observed samples are clean, but the candidate frame is incomplete. The result must be `FRAME_INCOMPLETE`.
4. **Indeterminate alternative:** an unresolved reconstruction world remains. It cannot be silently treated as agreeing or non-query-flipping.

Mutation tests must reject duplicate/non-frame sample IDs, mismatched population counts, changed query/loss digests, malformed source hashes, invalid inclusion probabilities, unresolved sampled units and a post-outcome preregistration.

## 8. Competitive benchmark matrix

| Comparator | What it measures | Role in this benchmark |
|---|---|---|
| 2026 global graph-invariant stopping preprint | Asymptotic global graph statistics and edit-count forecasts | Global convergence baseline; not a question-specific certificate |
| ConnectomeBench2 | Split/merge proofreading decisions and model calibration across species | Candidate/edit classification and reviewer-assistance baseline; not scientific-query sufficiency |
| NEURD / RoboEM / VLM proofreaders | Morphology, neurite and edit candidate generation | Candidate-frame coverage and cost baselines |
| Query-critical discovery (#10656) | Whether omitted alternatives change a frozen query | Source of query-changing residual alternatives |
| Adaptive intervention selection (#10670) | Which feasible observation best separates critical worlds | How unresolved worlds may be discriminated before stopping |
| #9204 sampling design | Residual-alternative discovery under a declared sampling frame | Production estimator owner; V1 is only the simple-random finite-population oracle |

Do not compare raw accuracy, graph convergence, query-flip prevalence, reviewer time or causal validity as if they were interchangeable.

## 9. Metrics and promotion targets

Report separately:
- first global-stop checkpoint and proofreading edit count;
- first query-stop checkpoint and proofreading edit count;
- admitted-world agreement and unresolved-world count;
- finite-frame query-flip upper bound and confidence level;
- false-stop rate (a query stop declared despite an admitted opposing world or known query-flipping frame unit);
- frame-coverage and adjudication completeness;
- reviewer-hours/edits avoided after query sufficiency;
- residual query-changing exposure at stop;
- subgroup/region/task failure rates and compute/acquisition cost.

Aspirational promotion targets (not results):
- 100% exact hypergeometric upper-bound agreement with brute-force enumeration on bounded synthetic cases;
- zero false query-stop passes under the four adversarial cases;
- zero passes with incomplete frames, disputed/indeterminate worlds, unresolved outcomes or invalid inclusion probabilities;
- lower reviewer cost than global-only stopping at matched residual query-risk bounds, or lower query risk at matched cost, on a blinded held-out corpus;
- prospective biological qualification only with a source-bound frame and a pre-registered independent-unit design.

Show the Pareto frontier of under-proofreading risk versus reviewer/experiment cost. Do not hide the trade-off in a single score.

## 10. Claim ceiling

A pass establishes question-specific robustness only within the frozen query, loss, admissible world set, closed residual frame, evidence tier and sampling design. It does not establish that the frame contains every biologically possible alternative, that the connectome is globally complete, or that a fitted mechanism is causally correct. Global convergence, admitted-world agreement and residual sampling bounds remain separate evidence.