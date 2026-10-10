# Independent Verification — Ramanujan Protocol

This document tells a reader who has **not** cloned the full `symthaea` repo, or who wants a second-party check, exactly how to verify the paper's claims.

## What is claimed

The showcase distinguishes bounded candidate assessment from independently checked formal obligations:

- **Symbolic + sampled** — symbolic chain-rule differentiation followed by finite-point residual sampling. This is supporting evidence, not a universal proof.
- **Numeric** — trajectory variance is below threshold, but the symbolic/sample assessment did not pass.
- **Approximate** — the candidate remains a best-effort fit with variance above threshold.
- **Formally verified (SMT)** — a separate supported polynomial obligation returns unsat from Z3. This applies only to the exact statement and witness checked.

Candidate-search claims use deterministic seed 42. The checked-in stdout and LaTeX table are historical snapshots from before this evidence-label hardening; their old PROVEN labels are not execution evidence for the current source. Regenerate and qualify the outputs against the exact head before publication or reproducibility claims.

## Verification paths, in order of effort

### Path 1: Docker (lowest effort)

Install Docker. Then:

```bash
cd papers/ramanujan
docker build -t ramanujan-repro .
docker run --rm ramanujan-repro
```

The container produces `showcase_stdout.txt`, `showcase_stderr.txt`, and `results_table.tex`. Diff against the versions committed in this directory:

```bash
diff <(docker run --rm ramanujan-repro cat /work/showcase_stdout.txt) showcase_stdout.txt
```

Any diff other than timing lines (`real`, `user`, `sys`) is a reproducibility failure that should be reported.

### Path 2: Local toolchain

Requires Rust stable ≥ 1.75 and Z3 ≥ 4.13 on `PATH`.

```bash
cd papers/ramanujan
./reproduce.sh
```

### Path 3: SMT-only (no Rust required)

Every `.smt2` file in `proofs/` is a standalone formal proof obligation checkable by any SMT-LIB2-compliant solver. With Z3:

```bash
for f in papers/ramanujan/proofs/*.smt2; do
  printf "%-40s " "$(basename $f)"
  z3 -smt2 "$f" | tail -1
done
```

Expected output: `unsat` on every line. CVC5 and MathSAT also close these; tested with Z3 4.13+. The `./reproduce.sh --verify-proofs` flag automates this loop.

## Scope of formal verification

| Status | Meaning |
|--------|---------|
| **Symbolic + sampled** | A symbolic derivative was constructed and its residual passed at six fixed points. This is finite evidence only; a nonzero derivative can vanish at those points. |
| **Numeric** | Trajectory variance is below threshold, but the symbolic/sample assessment did not pass. |
| **Approximate** | Best-effort candidate, variance above threshold. |
| **Formally verified (SMT)** | A separately committed supported proof obligation returned unsat; see the exact witness files below. |

Results from the committed baseline run (see `showcase_stdout.txt`):

| Row | Discovery | Status |
|-----|-----------|--------|
| Harmonic oscillator | `x² + v²` | **Symbolic + sampled** |
| Lotka–Volterra | `x − ln x + y − ln y` | **Symbolic + sampled** |
| Kepler two-body (angular momentum) | `xv_y − yv_x` | **Symbolic + sampled** |
| Hénon–Heiles | full 4D Hamiltonian | **Symbolic + sampled** |
| PCR3BP Jacobi | `cos(y/e)^(x³)` | **Numeric** (low variance, wrong formula — honest) |
| Mystery ODE (anisotropic oscillator) | `½(pₓ²+pᵧ²) + x² + y² + xy` | **Symbolic + sampled** |
| Triangular numbers | `n(n+1)/2` | Identity |

The PCR3BP row deserves attention: the discovered expression has variance $2.7 \times 10^{-10}$ but is transparently unrelated to the Jacobi integral. The pipeline reports \texttt{Numeric}, not \texttt{Symbolic + sampled}, which is the correct honest signal. A reader should read this as "the engine found something that happens to be low-variance on this trajectory, not a conservation law."

## SMT proof witness availability

**As of the formal-verify commit, four `.smt2` witness files are committed under `proofs/`**:

- `harmonic_oscillator.smt2` — `E = x² + v²`
- `kepler_angular_momentum.smt2` — `L = xvy − yvx`
- `henon_heiles_6H.smt2` — `6H = 3(px² + py²) + 3(x² + y²) + 6x²y − 2y³` (scaled Hénon-Heiles; see note)
- `mystery_ode.smt2` — `H = ½(px² + py²) + x² + y² + xy`

All four return `unsat` under Z3 4.13+ (tested), independent re-verification confirmed.

### Symbolic/sample assessment versus formal SMT verification

Two stacked layers of evidence exist. The paper reports both:

| Status tag in `showcase_stdout.txt` | Method | Reach |
|-------------------------------------|--------|-------|
| `Symbolic check + 6-point residual` (showcase status) | Symbolic chain-rule derivation via `SymExpr::diff` + `SymExpr::simplify`, then numerical residual check at 6 sample trajectory points | Handles polynomial and transcendental invariants; strong evidence but not a formal proof |
| `unsat` (shown in `proofs/*.smt2` after `verify_invariants_formal`) | Z3 UNSAT on the obligation `∃x : dE/dt ≠ 0` encoded in `QF_NRA` | Polynomial invariants only; this IS a formal proof |

The Lotka–Volterra invariant is **Symbolic + sampled** evidence but **not** `unsat` (its log term is transcendental, outside `QF_NRA`). This is the correct honest pair of labels: we have strong evidence of conservation (showcase) and explicitly cannot formalize it within Z3's algebraic fragment (proofs/).

### Why Hénon-Heiles uses 6H

IEEE-754 `f64` cannot represent `1/3` exactly. The serializer emits `0.3333333333333333`, which Z3 reads as the literal rational `3333333333333333/10000000000000000`, making `dE/dt` nonzero as a rational expression (sat, wrongly). Multiplying `H` by 6 clears all fractional coefficients; since conservation is preserved under constant rescaling, proving `d(6H)/dt = 0` proves `dH/dt = 0`. Phase 2 can upgrade the serializer to emit exact rationals (`(/ 1 3)` in SMT-LIB2) and avoid the rescale.

## Cross-host determinism

Bit-identical LaTeX output requires matching:

- Rust compiler major version (1.75+; any patch level within 1.75.x works)
- Z3 major version (tested with 4.12 and 4.13)
- CPU architecture (x86_64 vs aarch64 can change floating-point summation order, which can tip variance above/below the `1e-6` threshold for borderline candidates)

The Docker image pins all three. The local `reproduce.sh` pipeline is deterministic within a single host but may produce cosmetic diffs across different architectures.

## What fails verification

- After exact-head regeneration, any unexplained status or numerical change is a reproducibility failure. The intentional removal of legacy PROVEN labels is expected.
- If the Docker container returns a non-zero exit code, reproduction failed.
- If `./reproduce.sh --verify-proofs` reports any `FAIL`, that specific claim is not independently verifiable on the verifying host's SMT solver; it may reflect a solver-version difference. Report the full `z3 -v` version along with the failure.

## What does not fail verification

- Wall-clock timing differences.
- Minor formatting of `showcase_stderr.txt` (compile warnings change between compiler versions).
- Numerical jitter in the 7th+ significant figure of reported variance; exact table cell strings are deterministic but the underlying `f64` arithmetic is IEEE-754 associativity-dependent.
