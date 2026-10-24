# PHOT-OPT-001A: canonical MIS-to-QUBO/Ising mapping contract

Status: **research specification; no solver or photonic execution result claimed**

Parent work: [PHOT-OPT-001 #8617](https://github.com/Luminous-Dynamics/symthaea/issues/8617). Compose with [PHOT-ENG-000 #5668](https://github.com/Luminous-Dynamics/symthaea/issues/5668), [PHOT-001 #5679](https://github.com/Luminous-Dynamics/symthaea/issues/5679), [MIS verifier PR #8324](https://github.com/Luminous-Dynamics/symthaea/pull/8324), and [QCOMP-REDUCTION-ABLATION-001 #8590](https://github.com/Luminous-Dynamics/symthaea/issues/8590).

## 1. Scope

Define one exact, testable mapping from a canonical simple undirected graph to a binary quadratic objective, plus a matching spin/Ising representation and reconstruction contract. This is a specification for later code and hostile tests, not a claim that any optical hardware uses this exact convention.

The first implementation should be unweighted MIS. Weighted MIS is admitted only with explicit non-negative vertex weights, a proven penalty condition, and overflow/precision checks. Reject unsupported inputs instead of silently coercing them.

## 2. Canonical problem

Input graph: (G=(V,E)), with vertices canonically indexed (0,ldots,n-1), unique undirected edges ((u,v)) where (u<v), no loops, and no duplicate edges.

Binary variable (x_i\in\{0,1\}) indicates whether vertex (i) is selected. For non-negative weights (w_i), maximum-weight independent set is

\[
\max_x\quad \sum_{i\in V} w_i x_i
\qquad\text{subject to}\qquad
x_u+x_v\le 1\quad\forall (u,v)\in E.
\]

For unweighted MIS set (w_i=1). The independent-set verifier—not the energy function—remains authoritative about feasibility.

## 3. Canonical QUBO energy and penalty proof

Use the minimization convention

\[
E_{Q}(x)=-\sum_i w_i x_i
+A\sum_{(u,v)\in E}x_u x_v.
\]

Each undirected edge appears exactly once in the edge-list sum. The objective constant/offset is zero in this convention.

For unweighted MIS, (A>1) is sufficient for every global minimum to be feasible. More generally, for finite non-negative weights, (A>\max_i w_i) is sufficient:

If a candidate is infeasible, at least one selected vertex (i) has (k\ge1) selected neighbors. Removing (i) changes its energy by

\[
\Delta E_Q=w_i-Ak<0.
\]

Therefore an infeasible assignment cannot be a global minimizer. For non-negative integer weights, (A=1+\max_i w_i) is a convenient exact choice. For floating weights, the representable margin above the largest weight must be explicit and tested; do not assume an arbitrarily small epsilon survives coefficient scaling or hardware quantization.

This is a mathematical bound for the stated ideal binary objective, **not** a promise that an analog device or finite-precision solver will return a feasible assignment.

### Matrix-convention trap

Different libraries use different matrix conventions. Define the canonical edge-list objective above first, then test any serializer against its own documented convention:

- Upper-triangular energy \(\sum_i Q_{ii}x_i+\sum_{i<j}Q_{ij}x_ix_j\): (Q_{ii}=-w_i), and (Q_{ij}=A) for an edge.
- Symmetric matrix energy \(x^TQx\): (Q_{ii}=-w_i); for each edge, set (Q_{ij}=Q_{ji}=A/2). Using (A) in *both* symmetric positions doubles the penalty and is a mapping bug.
- Any constant offset must be carried explicitly and must not affect the decoded argmin.

The serialized manifest must state the exact energy convention and offset. Do not infer a convention from a field named merely `Q`, `matrix`, `bias`, or `coupler`.

## 4. Ising / spin convention

For spins (s_i\in\{-1,+1\}), define (x_i=(1+s_i)/2) and minimize

\[
E_I(s)=C+\sum_i h_i s_i+\sum_{(i,j)\in E}J_{ij}s_i s_j,
\]

with

\[
C=-\frac12\sum_i w_i+\frac{A}{4}|E|,
\qquad
h_i=-\frac{w_i}{2}+\frac{A}{4}\deg(i),
\qquad
J_{ij}=\frac{A}{4}\quad ((i,j)\in E).
\]

Only existing graph edges receive couplings in this ideal mapping. If a target device requires additional couplers, embedding, dense completion, gauge transforms, or coefficient rescaling, that is a separate transformation with separate provenance and cost; it must not overwrite the canonical source problem.

A positive global scale \(\alpha>0\) preserves the ideal ordering of configurations but changes the relationship between coefficients, analog noise, precision, and device range. Record it. If an API maximizes energy or defines (s_i) with the opposite sign, implement an explicit boundary conversion and verify the result against the canonical energy oracle.

## 5. Hand-checkable fixtures

Use (A=2) for unweighted examples unless otherwise stated. The cardinality is independently measured from the decoded bit vector and rechecked by the graph verifier. Energies below use the canonical minimization convention above.

| Graph | Vertices / edges | MIS cardinality | Minimum energy |
|---|---:|---:|---:|
| Empty graph on 3 vertices | (n=3,m=0) | 3 | -3 |
| One edge (K_2) | (n=2,m=1) | 1 | -1 |
| Path (P_3) | (n=3,m=2) | 2 | -2 |
| Triangle (K_3) | (n=3,m=3) | 1 | -1 |
| One edge plus one isolated vertex | (n=3,m=1) | 2 | -2 |
| Weighted single edge, weights ((3,1)) | (n=2,m=1,A=4) | max weight 3 | -3 |

There may be multiple minimizers (for example, either endpoint of (K_2)); tests compare the set of admissible minimizers/objective values, not an arbitrarily selected vertex ordering.

### Deliberately bad-penalty negative control

For (K_2) with unweighted objective and (A=0.5), selecting both endpoints has energy \(-2+0.5=-1.5\), lower than either feasible singleton at (-1). This fixture proves the test harness detects a penalty that is too small. Its expected behavior is to flag the configuration as an invalid MIS and flag the penalty contract as violated—not to silently repair the canonical mapping or count the invalid set as success.

## 6. Required independent checks

1. Enumerate every binary assignment for tiny graphs, compute canonical QUBO energy directly from the edge list, and compare with any matrix/spin conversion.
2. Check that every global energy minimizer is feasible and has maximum independent-set weight for the declared valid penalty.
3. Cross-check spin energy against binary energy for every assignment on tiny graphs; test all-zeros/all-ones and every one-hot assignment explicitly.
4. Run the independent graph solution verifier on *every* decoded proposal, before and after any repair.
5. Ensure the parser rejects loops, duplicate edges, invalid endpoints, inconsistent vertex counts, unknown encodings, NaN/infinite weights or coefficients, integer overflow, and non-canonical input when canonical bytes are required.
6. Test edge-order and vertex-label permutations: after applying and then reversing the explicit index map, the optimum objective and feasibility must be unchanged.
7. Test coefficient scaling, offsets, upper-triangle versus symmetric-matrix export, and sign-convention adapters against the canonical oracle.
8. Preserve infeasible, undecodable, timed-out, and failed candidates. They must not disappear from success-rate denominators.

## 7. Canonical input/output receipt

Before an external or simulated backend is integrated, the run manifest should pin at least:

- schema and mapping version;
- canonical graph bytes digest and vertex-index mapping;
- sorted unique edge-list digest;
- weight-vector digest and penalty (A), including the proof/selection rule;
- energy convention, offset, QUBO/Ising coefficients and positive scale;
- backend class: digital simulation, photonic simulation/emulation, optical analog device, or photonic quantum circuit;
- exact solver/device/software version and source/license reference where applicable;
- embedding/gauge/layout and calibration/noise references if applicable;
- raw bitstring/spin outputs, decode status, canonical energy, independent feasibility result;
- repaired solution and repair cost as separate fields;
- per-stage duration, retries/failures, energy/power measurement or labelled estimate, and end-to-end wall time;
- receipt schema version and digest of the complete normalized record.

A graph digest, seed, or claimed solver name alone is not a reproducibility receipt.

## 8. Research boundary and next step

First implement the canonical objective and tiny exhaustive oracle in the existing MIS workstream, then add property/hostile tests. Only after mapping invariants pass should a digital Ising simulator or external photonic adapter be considered.

Keep these claims distinct:

- a valid MIS proposal;
- a QUBO minimizer under the specified mapping;
- a photonic-system simulation;
- a measured optical analog result;
- a photonic quantum-circuit execution;
- an advantage over a strong classical solver under a matched full-workflow budget.

No level implies the next. In particular, this specification provides no photonic execution, speedup, SOTA, or quantum-advantage evidence.

Related published context:
- [Coherent Ising machine for large-scale MIS, Science Advances (2025)](https://doi.org/10.1126/sciadv.ads7223).
- [Programmable 200 GOPS Hopfield-inspired photonic Ising machine, Nature (2025)](https://www.nature.com/articles/s41586-025-09838-7).
- [Analog optical computer for AI inference and combinatorial optimization, Nature (2025)](https://doi.org/10.1038/s41586-025-09430-z).
- [QOBLIB](https://zib-aopt.github.io/QOBLIB/), for a separate public-instance comparison track.
