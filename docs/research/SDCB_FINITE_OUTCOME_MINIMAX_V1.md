# SDCB finite-outcome minimax oracle M0

**Status:** exact finite-outcome reference authored; independent rational replay completed; Rust compilation/tests and hosted CI remain unqualified.

Machine-readable fixtures: `docs/research/SDCB_FINITE_OUTCOME_MINIMAX_V1.json`.

## Why this is the next step

The convex-hull witness in PR #14188 proved that pairwise disjoint members can still have intersecting convex hulls. The next requirement is an executable oracle that computes the finite-family randomized test, supplies a matching least-favourable-mixture dual certificate, and rejects unresolved arithmetic or search failures.

This implementation solves two separately formulated rational LPs:

### Primal randomized test

\[
\begin{aligned}
\max_{\phi,\beta}\quad &\beta\\
\text{s.t.}\quad
&\sum_k p_{ik}\phi_k\geq\beta &&\forall i,\\
&\sum_k q_{jk}\phi_k\leq\alpha &&\forall j,\\
&0\leq\phi_k\leq1,\quad \beta\geq0.
\end{aligned}
\]

### Dual least-favourable mixtures

\[
\begin{aligned}
\min_{\lambda,w,z}\quad
&\alpha\sum_jw_j+\sum_kz_k\\
\text{s.t.}\quad
&\lambda_i\geq0,\quad\sum_i\lambda_i=1,\\
&w_j\geq0,\quad z_k\geq0,\\
&\sum_i\lambda_ip_{ik}-\sum_jw_jq_{jk}\leq z_k
&&\forall k.
\end{aligned}
\]

For any finite-outcome instance, this is a finite-dimensional LP pair. The exact solver returns a lower bound from a feasible primal randomized test and an upper bound from a feasible dual mixture certificate. Matching rational objectives prove the exact maximin value for that instance.

## Implemented reference

- `scripts/research/sdcb_finite_outcome_minimax.rs` — exact `i128` rational LP construction and vertex/basis enumeration for primal and dual formulations.
- `scripts/research/run_sdcb_finite_outcome_minimax_tests.sh` — standalone `rustc --test` and executable harness.
- The solver checks every candidate basis exactly, validates all constraints, and refuses arithmetic overflow, malformed probability laws, enumeration-limit exhaustion, or a nonzero primal-dual gap.
- Enumeration is exponential and deliberately capped at 2,000,000 bases per LP. This is a small-instance oracle, not a scalable optimizer.

## Frozen known-answer fixtures

| Fixture | alpha | Exact maximin power | Interpretation |
|---|---:|---:|---|
| Convex-hull overlap | 0.10 | **0.10** | Pairwise disjoint members; mixture overlap caps robust power at alpha |
| Partial separation | 0.10 | **0.50** | Nontrivial robust power between alpha and one |
| Complete separation | 0.10 | **1.00** | Disjoint support permits perfect discrimination |

### Partial-separation certificate

\[
A_1=(1,0,0),\quad A_2=(0,1,0),\quad B_1=(0.1,0.1,0.8).
\]

The primal test \(\phi=(0.5,0.5,0)\) has size 0.1 and power 0.5 under each alternative. A matching dual certificate is \(\lambda=(0.5,0.5)\), \(w=5\), \(z=0\). Its objective is \(0.1\cdot5=0.5\). Thus the exact maximin power is 0.5.

This case is useful because it proves the method is not merely an overlap detector: it returns a nontrivial robust operating point with matching upper and lower certificates.

## Adversarial mutations

The Rust suite covers:
- adding a null member that is a 50/50 alternative mixture: power must fall from 0.5 to 0.1;
- adding an alternative member supported only on the third outcome: worst-case power must fall from 0.5 to 0.1;
- permuting family members without changing the optimum;
- rejecting malformed probability mass;
- failing closed when the basis-enumeration budget is zero.

The test helper independently re-checks primal size and power constraints, dual nonnegativity, dual mixture constraints, dual objective, and exact zero gap.

## Validation and boundaries

- A separate Python `Fraction` vertex-enumeration replay established exact primal/dual values for all three fixtures and both family-expansion mutations.
- Eight Rust tests are authored, but the current authoring environment has no `rustc`; they are **not compiled or executed**.
- The shell harness is authored but likewise not run.
- This increment does not supply a continuous-Gaussian solver, continuous nuisance-family guarantee, physical sensor test, or SOTA/industry superiority claim.

The next mathematical increment after M0 is M1: continuous one-dimensional Gaussian families with certified integration and root-isolation enclosures. Do not move to vector Gaussian optimization before the M0 oracle is executed and checked on the target toolchain.
