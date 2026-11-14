# AIRROAD Clarabel SDP smoke tests

This is a nested Cargo workspace so its build does not require parsing the
large Symthaea root workspace. It compiles the checked-in
`symthaea-airroad-iqc/src/lib.rs` directly as a module and pins Clarabel
`=0.11.1` with the `sdp-openblas` feature.

## Test 1: feasible analytic SDP

Let

```
H = [ 2   1+2i ]
    [ 1-2i  5  ]
```

and let `R(H)` be AIRROAD's realification. The problem is

```
minimize    trace(X)
subject to  X - R(H) is positive semidefinite.
```

The optimum is `X = R(H)`, with objective `trace(R(H)) = 14`. The test
checks Clarabel's strict `Solved` status, the primal svec vector, the decoded
matrix, primal and dual residuals, and primal/dual objective agreement.

The conic formulation uses Clarabel's standard form `A x + s = b, s in K`,
with `x = svec(X)`, `A = -I`, and `b = -svec(R(H))`.

## Test 2: infeasible problem and Farkas witness

The second problem is

```
minimize    trace(X)
subject to  X - I is positive semidefinite
            X_11 = 0.
```

The PSD constraint implies `X_11 >= 1`, so the equality makes the problem
infeasible. The test requires Clarabel's strict `PrimalInfeasible` status and
checks the returned ray `z` against the standard-form certificate conditions:

- `A^T z = 0`;
- the PSD portion of `z` belongs to the dual PSD cone;
- `b^T z < 0`.

A solver status without these witness checks is not accepted by this smoke
test.

## Independent SCS cross-check

The workflow also runs a pinned SCS 3.3.1 research-only reference. SCS uses
the **lower** triangle stacked column-wise, whereas Clarabel's Rust adapter
uses the **upper** triangle stacked column-wise. Both use sqrt(2)-scaled
off-diagonal entries. The Python cross-check therefore builds and decodes
SCS vectors independently instead of reusing Clarabel's packed vector.

The SCS job solves both analytic problems again and independently checks its
primal solution and infeasibility certificate. It also constructs the trace
objective in SCS's lower-triangle ordering and checks the diagonal positions
for matrix orders 2, 3 and 4. The upper- and lower-triangle objective vectors
must not be reused interchangeably. NumPy, SciPy and SCS are validation-only
dependencies; the production AIRROAD crate remains Rust-only.

## Scope

These are analytic conic-backend tests. They do not implement the published
sampled-FRF IQC theorem, consume physical FRF uncertainty, establish robust
stability, validate aircraft networks, or provide airworthiness evidence.
