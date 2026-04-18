# Ramanujan Protocol — arXiv Submission Metadata

Copy-paste into the arXiv submission form fields.

## Title

```
Ramanujan Protocol: Autonomous Conservation-Law Discovery via Grammar-Guided
Genetic Symbolic Regression with Chain-Rule Verification and Z3 Formal Proofs
```

(If the full title exceeds arXiv's character limit, use:)

```
Ramanujan Protocol: Autonomous Conservation-Law Discovery with Z3 Formal Proofs
```

## Authors

```
Tristan Stoltz
```

Affiliation:

```
Luminous Dynamics
```

Email (corresponding):

```
tristan.stoltz@evolvingresonantcocreationism.com
```

## Abstract

```
We present the Ramanujan Protocol, a pipeline that rediscovers conservation
laws of canonical dynamical systems directly from their differential
equations, without access to analytic solutions, textbook hints, or numerical
labels. The method combines grammar-guided genetic symbolic regression over
expression trees, symbolic verification of conservation via the chain rule,
and formal proof delegation to the Z3 SMT solver for the invariants that
admit polynomial or polynomial-with-reciprocal encoding. Applied to seven
canonical problems (harmonic oscillator, Lotka-Volterra, Kepler two-body
angular momentum, Henon-Heiles chaotic Hamiltonian, anisotropic coupled
oscillator, the circular restricted three-body problem, and triangular
numbers), the pipeline recovers the exact analytic conservation law in six
cases with a verified symbolic proof, and a numerically-conserved-but-
semantically-wrong expression in the seventh (PCR3BP). Against a 221-equation
physics catalog, every recovered invariant matches its canonical entry at
>= 99% structural similarity. All runs are deterministic under a fixed seed
(=42) and machine-checkable; any reader with Docker or a local Rust + Z3
toolchain can regenerate the full results table end-to-end in under thirty
minutes.
```

## Primary category

```
cs.SC   (Symbolic Computation)
```

## Cross-list categories

```
cs.LG   (Machine Learning)
math.DS (Dynamical Systems)
cs.AI   (Artificial Intelligence)
```

Rationale: the core contribution is a symbolic-regression+SMT pipeline
(cs.SC primary). It uses genetic programming (cs.LG). The target domain is
conservation laws of dynamical systems (math.DS). The "autonomous
discovery" framing fits the AI-for-science literature (cs.AI).

## MSC 2020 codes

```
68W30    Symbolic computation and algebraic computation
37J05    Relations with symplectic geometry and topology (Hamiltonian systems)
68T20    Artificial intelligence -> Problem solving
```

## License for arXiv

```
arXiv.org perpetual, non-exclusive license
```

Source license (separate from arXiv terms): see LICENSE file in the
Symthaea repository (AGPL-3.0-or-later).

## Comments (the free-form field)

```
12 pages, 1 table, 0 figures. All results are deterministic under fixed
seed. Full reproduction harness (Docker + Rust + Z3) in the ancillary
files; see VERIFY.md. Companion validation experiments for the
integration-measure estimator in papers/consciousness-theory/stochastic-
resonance/ (separate submission).
```

## Ancillary files

Upload separately via the arXiv ancillary-files interface (these do not
count toward the main source package):

- `proofs/*.smt2` — 9 Z3 witness files, each independently verifiable
- `proofs/README.md` — how to verify each proof
- `reproduce.sh` — one-shot reproduction script
- `Dockerfile` — containerised reproduction (lowest-effort verification path)
- `VERIFY.md` — full verification walkthrough
- `showcase_stdout.txt` / `showcase_stderr.txt` — reference outputs for diff

See `SUBMIT.md` for the step-by-step runbook.
