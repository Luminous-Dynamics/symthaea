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

### Path 3: SMT-only (deferred until Phase 2)

At Phase 1 we do not commit `.smt2` witnesses (see "SMT proof witness availability" below). Once the engine instrumentation lands, the workflow will be: `./reproduce.sh --verify-proofs` re-runs every committed `.smt2` file through the user's local SMT solver (Z3, CVC5, MathSAT). Until then, this path falls back to Path 1 or Path 2.

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

**Phase 1 caveat:** the \texttt{conjecture\_engine} calls Z3 as a subprocess with SMT-LIB2 piped via stdin. It does not currently write the SMT-LIB2 to disk. Committing \texttt{.smt2} witnesses under \texttt{proofs/} requires a small instrumentation change (Phase 2 work). Until then, verification options are:

1. Run \texttt{./reproduce.sh} --- the engine re-invokes Z3 on your host with the same SMT-LIB2 it used originally.
2. Trust the committed \texttt{PROVEN} status string. This is a weaker form of reproducibility --- it reproduces only if your host agrees with ours on what Z3 returns.

Readers who need stronger reproducibility should wait for the Phase 2 engine change or run \texttt{reproduce.sh} themselves.

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
