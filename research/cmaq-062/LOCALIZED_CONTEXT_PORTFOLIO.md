# CMAQ-062 — Localized conditional evidence with predictable context trees

Status: design and proof sketch. No implementation or power result is claimed.

## 1. Objective

Detect conditional distribution changes that occur only on a small or structured subset of covariate space. A single global conditional witness can dilute a sparse effect. Fixed KNN neighborhoods help but require high-dimensional distance choices and can be expensive.

CMAQ-062 investigates a hierarchical portfolio of local conditional e-processes. Each node has a region indicator and a local predictable witness/betting policy. The parent combines nodes using a fixed normalized prior, so one global alpha threshold controls the entire countable family.

## 2. Validity sketch

Let the direct null hold:

    Law(X_t | F_(t-1), Z_t) = Law(Y_t | F_(t-1), Z_t).

For a registered measurable region R_j, define w_j(Z_t)=1{Z_t in R_j}. This is known before the current response pair. Let r_t be any bounded predictable paired feature difference with

    E[r_t | F_(t-1), Z_t] = 0,      ||r_t|| <= 1.

Let u_(j,t) be a unit direction selected from F_(t-1), and let 0 <= lambda_(j,t) < 1 be predictable. Then

    f_(j,t) = 1 + lambda_(j,t) w_j(Z_t) <u_(j,t), r_t>

is positive and has conditional expectation one. Hence each node process

    M_(j,t) = product_(s<=t) f_(j,s)

is a nonnegative martingale/e-process.

For any fixed prior weights pi_j >= 0 with sum_j pi_j=1,

    E_t = sum_j pi_j M_(j,t)

is also an e-process. No independence between node processes is needed. Ville's inequality gives the single global threshold 1/alpha.

The mixture prior must be registered before confirmatory monitoring. A data-adaptive region may be used when its definition is predictable before the current pair, but changing the region inventory or post-hoc reallocating prior mass across results is not allowed in the fixed-mixture construction.

## 3. Context-tree construction

Candidate inventories:
- Root region covering the complete registered covariate support.
- Dyadic splits of standardized continuous coordinates.
- Quantile partitions fitted only on a separate calibration split, then frozen.
- Registered conjunctions for important context dimensions.
- Multi-resolution overlapping regions for boundary robustness.

Each observation belongs to one node per depth along a partition path. Updating only those nodes yields O(depth) node updates rather than O(number_of_nodes), before local feature costs.

The prior should reserve mass by level and node:
- level prior: a summable dyadic or polynomial tail;
- within-level prior: equal or volume-aware weights;
- fixed exact rational representation where feasible;
- no post-hoc prior adaptation based on observed e-values.

A node's local payoff can use fixed Fourier maps, a learned predictable representation, SCPT-style KNN context witnesses, or a small bounded multi-scale kernel bank. All node factors must share the same conditional paired null and norm contract.

## 4. Local compute and restart

Potential design:
1. Predictable node membership from current pre-pair Z_t.
2. Update local witnesses for the root-to-leaf path.
3. Produce a bounded factor per active node.
4. Use fixed prior mixture or one predictable path-selected factor.
5. Apply a scalar or per-node capital restart only where it is mathematically composed and preregistered.

An O(depth) implementation is a target, not a demonstrated complexity result. If each node has K local kernels and D feature dimensions, full active-path processing may still cost O(depth*K*D).

## 5. Failure modes and controls

- Sparse changes that straddle cell boundaries.
- Rare regions with insufficient historical support.
- Empty or near-empty child nodes.
- Covariate drift that moves data between regions under a valid null.
- Partition leakage from current/future responses.
- Data-dependent prior weights chosen after seeing evidence.
- Extremely deep trees whose prior mass makes detection too slow.
- Overlapping node dependence; this is not a validity problem for a fixed mixture, but it affects compute and interpretation.

Cold-start handling must be explicit. One option is a node-specific predictable betting fraction of zero until sufficient local history exists; its e-process remains valid but has no early power in that node.

## 6. Benchmark grid

Primary sparse-support alternatives should affect 0.5%, 1%, 5%, 10%, or 25% of covariate support. Include:
- axis-aligned boxes and smooth local bumps;
- sparse conjunctions and oblique regions;
- effects on partition boundaries;
- rare contexts whose frequency decays over time;
- local scale and tail changes;
- local equal-marginal conditional-pairing shifts;
- valid nulls with covariate drift and serially dependent responses.

Compare CMAQ-054/057/058 global monitors, faithful SCPT, fixed grid monitors, and the context-tree portfolio on identical generated streams.

Report:
- pointwise and simultaneous null calibration using CMAQ-059;
- unconditional power and restricted delay with misses/early alarms H+1;
- power by support size and distance to a cell boundary;
- prior mass assigned to the realized affected region;
- number of active node updates and feature-map evaluations;
- p50/p99 latency, memory, and warm-up requirements;
- worst-case as well as average performance.

## 7. Acceptance gates

1. Formal conditional-centering and mixture argument reviewed independently.
2. Frozen partition and exact prior manifest.
3. Tests for tree membership, boundary handling, prior normalization and empty-node behavior.
4. Mutation controls for current-pair partition leakage and post-hoc prior changes.
5. Exact-head Rust fmt/test/Clippy.
6. Matched-stream Monte Carlo against global and SCPT baselines.

Prospective target: reduce restricted mean delay by at least 20% on three sparse-support alternatives versus global CMAQ-057 at matched alpha, without a simultaneous null-bound violation or more than 2 percentage points of worst-family power regression. This is a target, not a result.

No SOTA, statistical qualification or deployment claim is made.
