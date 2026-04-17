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

Every `PROVEN` row in the results table is backed by an SMT-LIB2 proof obligation committed under `papers/ramanujan/proofs/`. Each file encodes `dC/dt = 0` on the supplied dynamics; Z3's certificate of `unsat` on the negation is the formal proof.

Any SMT-LIB2-compliant solver works: Z3, CVC5, MathSAT, etc. To re-verify every committed proof:

```bash
cd papers/ramanujan
./reproduce.sh --verify-proofs
```

Without `reproduce.sh`:

```bash
for f in papers/ramanujan/proofs/*.smt2; do
  printf '%s: ' "$(basename "$f")"
  z3 -smt2 "$f" | tail -1
done
```

Expected output: `unsat` for every file.

## Scope of formal verification

| Discovery | Verification modality |
|-----------|-----------------------|
| Harmonic oscillator `E = x² + v²` | Chain-rule + Z3 `unsat` on `QF_NRA` negation |
| Kepler energy `½\|v\|² − 1/r` | Chain-rule symbolic cancellation; Z3 unreliable because of the `1/r` singularity |
| Kepler angular momentum `xv_y − yv_x` | Chain-rule + Z3 `unsat` |
| Hénon–Heiles energy | Chain-rule + Z3 `unsat` |
| "Mystery ODE" anisotropic oscillator | Chain-rule + Z3 `unsat` |
| Lotka–Volterra `x − ln x + y − ln y` | **Numeric only** — normalizer handles polynomials but not logarithmic identities; Z3 cannot decide transcendentals |
| PCR3BP Jacobi | **Approximate only** — our grammar doesn't reach it exactly |

Readers should accept `PROVEN` claims only for rows where a committed `.smt2` witness exists.

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
