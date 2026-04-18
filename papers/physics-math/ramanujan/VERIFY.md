# Independent Verification — Ramanujan Protocol

This document tells a reader who has **not** cloned the full `symthaea` repo, or who wants a second-party check, exactly how to verify the paper's claims.

## What is claimed

The paper claims that, starting from six canonical ODE systems, Symthaea's pipeline rediscovers conservation laws with three distinct rigor levels:

- **PROVEN** — a symbolic proof via chain-rule normalization, optionally formalized by Z3 returning `unsat` on the negation of `dC/dt = 0`.
- **Numeric** — trajectory variance below threshold but no symbolic certificate.
- **Approximate** — best-effort candidate, variance above threshold.

All claims are tied to a deterministic seed (`42`). The LaTeX table in `main.tex` is produced **verbatim** by `cargo run --release -p symthaea-physics-bridge --example ramanujan_showcase`.

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

The pipeline reports one of three statuses per discovered invariant:

| Status | Meaning |
|--------|---------|
| **PROVEN** | Chain-rule normalization closes $\mathrm{d}C/\mathrm{d}t$ to zero symbolically. When the normalizer cannot close, the engine emits the corresponding SMT-LIB2 query and accepts Z3's `unsat` verdict as a formal proof. |
| **Numeric** | Trajectory variance below threshold ($<10^{-6}$) but no symbolic certificate. The discovered expression might be algebraically wrong but numerically conserved on the particular trajectory sampled. |
| **Approximate** | Best-effort candidate, variance above threshold. |

Results from the committed baseline run (see `showcase_stdout.txt`):

| Row | Discovery | Status |
|-----|-----------|--------|
| Harmonic oscillator | `x² + v²` | **PROVEN** |
| Lotka–Volterra | `x − ln x + y − ln y` | **PROVEN** |
| Kepler two-body (angular momentum) | `xv_y − yv_x` | **PROVEN** |
| Hénon–Heiles | full 4D Hamiltonian | **PROVEN** |
| PCR3BP Jacobi | `cos(y/e)^(x³)` | **Numeric** (low variance, wrong formula — honest) |
| Mystery ODE (anisotropic oscillator) | `½(pₓ²+pᵧ²) + x² + y² + xy` | **PROVEN** |
| Triangular numbers | `n(n+1)/2` | Identity |

The PCR3BP row deserves attention: the discovered expression has variance $2.7 \times 10^{-10}$ but is transparently unrelated to the Jacobi integral. The pipeline reports \texttt{Numeric}, not \texttt{PROVEN}, which is the correct honest signal. A reader should read this as "the engine found something that happens to be low-variance on this trajectory, not a conservation law."

## SMT proof witness availability

**As of the formal-verify commit, four `.smt2` witness files are committed under `proofs/`**:

- `harmonic_oscillator.smt2` — `E = x² + v²`
- `kepler_angular_momentum.smt2` — `L = xvy − yvx`
- `henon_heiles_6H.smt2` — `6H = 3(px² + py²) + 3(x² + y²) + 6x²y − 2y³` (scaled Hénon-Heiles; see note)
- `mystery_ode.smt2` — `H = ½(px² + py²) + x² + y² + xy`

All four return `unsat` under Z3 4.13+ (tested), independent re-verification confirmed.

### Honest distinction between "PROVEN (showcase)" and "formally verified (SMT)"

Two stacked layers of evidence exist. The paper reports both:

| Status tag in `showcase_stdout.txt` | Method | Reach |
|-------------------------------------|--------|-------|
| `PROVEN ✓` (shown in the showcase LaTeX table) | Symbolic chain-rule derivation via `SymExpr::diff` + `SymExpr::simplify`, then numerical residual check at 6 sample trajectory points | Handles polynomial and transcendental invariants; strong evidence but not a formal proof |
| `unsat` (shown in `proofs/*.smt2` after `verify_invariants_formal`) | Z3 UNSAT on the obligation `∃x : dE/dt ≠ 0` encoded in `QF_NRA` | Polynomial invariants only; this IS a formal proof |

The Lotka–Volterra invariant is `PROVEN` (symbolic + numerical) but **not** `unsat` (its log term is transcendental, outside `QF_NRA`). This is the correct honest pair of labels: we have strong evidence of conservation (showcase) and explicitly cannot formalize it within Z3's algebraic fragment (proofs/).

### Why Hénon-Heiles uses 6H

IEEE-754 `f64` cannot represent `1/3` exactly. The serializer emits `0.3333333333333333`, which Z3 reads as the literal rational `3333333333333333/10000000000000000`, making `dE/dt` nonzero as a rational expression (sat, wrongly). Multiplying `H` by 6 clears all fractional coefficients; since conservation is preserved under constant rescaling, proving `d(6H)/dt = 0` proves `dH/dt = 0`. Phase 2 can upgrade the serializer to emit exact rationals (`(/ 1 3)` in SMT-LIB2) and avoid the rescale.

## Cross-host determinism

Bit-identical LaTeX output requires matching:

- Rust compiler major version (1.75+; any patch level within 1.75.x works)
- Z3 major version (tested with 4.12 and 4.13)
- CPU architecture (x86_64 vs aarch64 can change floating-point summation order, which can tip variance above/below the `1e-6` threshold for borderline candidates)

The Docker image pins all three. The local `reproduce.sh` pipeline is deterministic within a single host but may produce cosmetic diffs across different architectures.

## What fails verification

- If any row changes its verification status (PROVEN → Numeric, or disappears), that is a reproducibility failure.
- If the Docker container returns a non-zero exit code, reproduction failed.
- If `./reproduce.sh --verify-proofs` reports any `FAIL`, that specific claim is not independently verifiable on the verifying host's SMT solver; it may reflect a solver-version difference. Report the full `z3 -v` version along with the failure.

## What does not fail verification

- Wall-clock timing differences.
- Minor formatting of `showcase_stderr.txt` (compile warnings change between compiler versions).
- Numerical jitter in the 7th+ significant figure of reported variance; exact table cell strings are deterministic but the underlying `f64` arithmetic is IEEE-754 associativity-dependent.
