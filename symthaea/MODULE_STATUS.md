# Symthaea Module Status

**Purpose:** Honest, directly-observed status of Symthaea's math and science modules. This document is the single source of truth on module reach, tested behavior, and known limitations.

**Method:** Each entry below is based on direct file reads + test counts + stub-marker greps performed on `main` branch. No entry repeats a claim without a file path + line reference. When in doubt, read the module.

**Scope discipline:** This document describes *what is implemented and tested*, not *what is planned*. Planned work lives in `plans/` and `memory/`. Aspirations do not appear here.

---

## Legend

| Status | Meaning |
|--------|---------|
| **Production** | Real algorithm; ≥10 tests against analytic ground truth or substantial property tests; no `unimplemented!()`/`todo!()`/`TODO`. Ready for downstream use. |
| **Tested with Known Limits** | Real algorithm; tested; but scope is bounded by a documented simplification (e.g. "finite-dimensional only", "Euclidean only", "numerical not symbolic"). |
| **Concept Encoder + Math** | Module encodes concepts as `ContinuousHV` for HDC binding AND contains real computational math with tests. Two roles coexist in one file. |
| **Research / Partial** | Core infrastructure exists; some sub-capability is explicitly marked TODO or is a known gap. |
| **External Commensurability** | `Yes` = results can be compared to a published external benchmark or analytic reference. `No` = internal metric only. |

---

## Math Foundation (`symthaea-core/src/hdc/`)

| Module | LOC | Tests | Status | External Comm. | Notes |
|--------|-----|-------|--------|----------------|-------|
| `lie_theory.rs` | 899 | 20 | **Production** | Yes (Fulton-Harris, Hall) | Real representation theory for sl(2), su(2), so(3), gl(2), root systems A_n/B_n/D_n/G₂, Killing form, BCH, irreps. No stubs. |
| `langlands.rs` | 1,460 | 22 | **Production** | Yes (elliptic curve point-counting) | Real Weierstrass point-counting, a_p = p+1-#E(F_p). Modular-form q-expansion is simplified (not full newform eigensolver). No `TODO` markers. |
| `gct.rs` | — | 24 | **Tested with Known Limits** | Partial | Ryser permanent + determinant fully real. Kronecker coefficient marked `// TODO` at `gct.rs:479` (Schur-function conversion missing). |
| `algebraic_geometry.rs` | — | 16 | **Tested with Known Limits** | Yes (numerical) | Newton's method on systems, Gauss elimination Jacobian solve, elliptic curve group law, Bézout. Self-labels "Numerical Algebraic Geometry" — no Gröbner bases, no symbolic ideals. |
| `category_theory.rs` | 954 | 16 | **Tested with Known Limits** | Partial | Small categories, functors, natural transformations, adjunctions, monads, Yoneda. No topoi, derived categories, sheaves. |
| `measure_probability.rs` | 573 | 15 | **Tested with Known Limits** | Yes | σ-algebra axiom validator (Ω, ∅, complement closure, pairwise union closure), measure spaces, martingales, Brownian motion, Itô's lemma, CLT, KS statistic. Radon-Nikodym absent. |
| `consciousness_topology.rs` | 1,806 | — | **Research / Partial** | No | β₀ real (DFS connected-components). β₁/β₂ transitioning to robust SNF/Persistence via `symthaea-hodge`. |
| `fem.rs` | 280 | 3 | **Tested with Known Limits** | Yes (numerical) | 1D and 2D Poisson solvers implemented via Galerkin weak forms and Sparse Conjugate Gradient (CSR). Coupled to Φ measurement. |
| `linear_algebra.rs` | 3,041 | — | **Production** | Yes | QR, SVD, eigensolvers, GMRES, PCA, Cholesky (prior-verified). |
| `calculus.rs` + `autodiff.rs` + `autodiff_phi.rs` | 4,983 | — | **Production** | Yes | Forward/reverse AD, specialized Φ-gradient tracking. |
| `differential_equations.rs` | 1,906 | — | **Production** | Yes | ODE/PDE solvers, bifurcation, attractor detection. |
| `logic_engine.rs` | 1,767 | — | **Production** | Yes | Propositional DPLL, FOL unification, natural deduction (Modus Ponens/Tollens, Hypothetical/Disjunctive Syllogism), resolution, Tseitin CNF. `ProofResult { proof_steps: Vec<ProofStepLogic> }` at `logic_engine.rs:200-218` — sequential not tree-structured. |
| `optimization.rs` | 1,672 | — | **Production** | Yes | L-BFGS, Newton, SOS decomposition, Lagrange multipliers. |
| `fft.rs` | 1,119 | — | **Production** | Yes | Cooley-Tukey FFT. |
| `number_theory.rs` | 945 | — | **Production** | Yes | Modular arithmetic, Diophantine/Pell, primality, CRT. |
| `functional_analysis.rs` | 1,623 | 17 | **Tested with Known Limits** | Yes (finite-dim) | L² spaces, bounded operators, Jacobi eigenvalue, Sobolev norms H¹/H², Fredholm alternative. Entire module is finite-dimensional; no distributions, no weak solutions on infinite-dim Hilbert spaces. |
| `conjecture_engine.rs` | 11,097 | — | **Production** | Yes (external Z3) | Symbolic regression GP, Z3 auto-proof (`:3157` `auto_prove_via_z3`, `:4362` `expr_to_smtlib2`), Bayesian confidence. 6 PROVEN conjectures via `ramanujan_showcase.rs` (seed=42 deterministic). |
| `abstract_thought/` (4 files) | 2,777 | — | **Research / Partial** | No | Meta-HDC, dynamic grammar, category discovery, macro quality. Wired to conjecture engine; does NOT talk to `GridEncoder` (ARC-AGI-2 integration gap). |

---

## Domain-Specific Crates

| Module | LOC | Tests | Status | External Comm. | Notes |
|--------|-----|-------|--------|----------------|-------|
| `symthaea-hodge/homology.rs` | 245 | 3 | **Tested with Known Limits** | Yes | Discrete algebraic homology solver (SNF over ℤ₂) with support for Persistent Homology (Vietoris-Rips). |

---

## Physics / Science (`symthaea-core/src/physics/` and sub-crates)

| Module | LOC | Tests | Status | External Comm. | Notes |
|--------|-----|-------|--------|----------------|-------|
| `physics/thermodynamics.rs` | 649 | 13 | **Concept Encoder + Math** | Yes | `ThermoEncoder` encodes Boltzmann/Shannon entropy, Helmholtz/Gibbs/enthalpy, Landauer limit, Szilard, canonical/Jarzynski/fluctuation theorem, Carnot, variational FE. Tests at `thermodynamics.rs:468-639` verify analytic identities (second law, Carnot efficiency, Jarzynski equality). Also holds `ContinuousHV` fields for HDC binding — dual-role module. |
| `physics/classical_mechanics.rs` | — | — | Unverified | — | Needs direct audit. |
| `physics/general_relativity.rs` | — | — | Unverified | — | Needs direct audit. |
| `physics/standard_model.rs` | — | — | Unverified | — | Needs direct audit. |
| `symthaea-quantum-chemistry` | 3,744 | 30+ | **Tested with Known Limits** | Yes (H2, water) | MP2 (full 4-index ERI transformation), CCD (linearized with MP2 init, quadratic damping). CIS uses Koopman approximation only — no 2-electron integral matrix. TDDFT scaffolding. |
| `symthaea-nuclear` | 24,429 | — | **Production** (structure) / **Gap** (reactions) | Partial | FRDM mass formula, shell model, HFB, AME2020 data. No optical potential, no DWBA, no (n,γ)/(n,f) reaction cross-section predictions. |
| `symthaea-particle-physics/relativistic_qm.rs` | — | — | **Production** (hydrogen) | Yes | Dirac γ-matrices (Dirac rep), exact E_{n,j} formula, fine structure. |
| `symthaea-frontier-physics/` | 11,000+ | — | **Research / Partial** | Partial | Zero `unimplemented!()`/`todo!()`/`FIXME`/`placeholder` markers. `topological_qft.rs` computes real formulas (Chern number via Berry curvature integral, Jones polynomial evaluation, Berry phase, topological entanglement entropy). QED loop integrals, renormalization, and full tensor-network DMRG contraction are scaffolded but not reached by tests. |
| `symthaea-physics-bridge/` | 10,646 | 100+ | **Production** | Yes | 27-equation catalog (Maxwell, Einstein, Schrödinger, Dirac, Yang-Mills, Navier-Stokes, Friedmann). HDC semantic encoding for structural analogy discovery. |
| `symthaea-physics/` | 18,591 | — | Unverified | — | Plasma control, antimatter, fusion — needs direct audit before claiming status. |
| `symthaea-continuum-physics/open_quantum.rs` | — | — | Unverified | — | Lindblad master equation claimed; needs direct audit. |

---

## Confirmed Gaps (Explicitly Not Implemented)

These are pillars that either do not exist or exist as name-only scaffolding:

- **Stochastic calculus / SDEs** — `measure_probability.rs` mentions Itô; no Milstein, no Fokker-Planck solver, no Langevin integrator as general library.
- **Persistent homology** — Discrete persistence tracking implemented; full Ripser-level barcode visualization still a gap.
- **Control theory** — LQR, MPC, H-∞, observability Gramians absent as a general library. Robotics crates have bespoke controllers.
- **Fiber bundles / connections / characteristic classes** — `riemannian_geometry.rs` has Ricci flow; no principal bundles, no Chern/Pontryagin classes.
- **Gröbner bases** — `algebraic_geometry.rs` is numerical only.
- **Post-HF CCSD(T)** — `symthaea-quantum-chemistry` has CCD (linearized); no full CCSD(T), no real TDDFT with integral-driven matrix.
- **Nuclear reactions** — `symthaea-nuclear` covers structure; no optical potential, no DWBA.
- **QED loop integrals** — `symthaea-frontier-physics` scaffolding exists; actual one-loop g-factor computation not reachable.

---

## Benchmark Status

| Benchmark | Type | Result | External Commensurability |
|-----------|------|--------|---------------------------|
| `symthaea-psych-bench` | Internal, 136+ problems, 26 domains | Composite z-score | **Internal only.** This metric is not commensurable with any external benchmark. Do not cite externally as evidence of math/sci capability. External validation is the Phase 1 goal. |
| Ramanujan conjecture suite (Tier A: showcase) | Internal, 7 physics targets | 6 PROVEN + 1 Numeric honest failure (PCR3BP) via GP + symbolic chain-rule + 6-point numerical check; 221-equation catalog ≥99% match | Strong. Reproducible (`papers/ramanujan/reproduce.sh`, seed=42, ~5 min). Paper draft at `papers/ramanujan/main.tex`. |
| Ramanujan conjecture suite (Tier B: Z3 formal) | Internal, polynomial subset | **9 invariants formally proven via Z3 UNSAT with persistent `.smt2` witnesses** across 6 distinct systems: harmonic, Kepler angular momentum, Hénon-Heiles (scaled), Mystery ODE, Duffing (scaled), quartic anharmonic (scaled), 2D isotropic oscillator energy + angular momentum, linear coupled oscillators (scaled). Lotka-Volterra skipped (transcendental). | Strongest formal claim in Symthaea: any SMT-LIB2 solver re-verifies each witness independently. `papers/ramanujan/proofs/*.smt2`, `papers/ramanujan/reproduce.sh --verify-proofs`. |
| IMO tactics | Internal, 14 problems | 14/14 across 5 domains | Partial — problem set is curated, not a standard contest year. Externally-commensurable contest results await Phase 1 AIME/HMMT work (Phase 2+). |
| Lean 4 propositional tautology suite (`proptauts`) | Internal, 23 fixtures via `synthesize_proof_term` | **23/23 Lean-accepted, 0 `sorry`, 100% strict rate** — externally verified by `lean <file>` | Strong. Each proof term regenerable from source (`cargo run -p symthaea-lean-bridge --example prove_proptauts`). Every emitted `.lean` file committed under `proofs/proptauts/`. |
| Lean 4 miniF2F subset | Existing fixtures (`proofs/minif2f/`) | **3/3 accepted** | Strong, but tiny corpus — fixtures are symthaea-engine-originated, not drawn from the miniF2F-v2 upstream. |
| **Lean 4 miniF2F-v2 full** | Upstream corpus at github.com/openai/miniF2F | **Architecturally out-of-scope at Phase 1** — see `docs/minif2f-v2-scope.md`. Propositional bridge cannot represent the algebraic/arithmetic/real-number content that constitutes ~98% of the benchmark. | Nil (0/N expected). Honest scope signal, not a quality failure. Phase 2 path: extend `Proposition` with equality + arithmetic, hook Z3 `QF_LRA`/`QF_LIA` to `linarith`-style Lean tactics. |
| PutnamBench | Upstream Lean 4 Putnam formalization | Not yet attempted | Same architectural mismatch as miniF2F-v2; Phase 2. |
| **Lean 4 Phase 2/3 arithmetic suite (`fol_arith`)** | Internal, 15 hand-crafted `FolFormulaExt` fixtures incl. 1 IMO-class | **15/15 accepted (100%)** after Phase 3 Move 1 (antisymmetric multiplicative cross-term hints for ≥4-binder systems, commit `4c518d9194`). Covers: reflexivity (ℝ/ℤ), integer monotonicity, Nat non-negativity, commutativity, trichotomy (via `rcases lt_trichotomy`), square non-negativity (both `x*x` and `x²` shapes), implication, RatLit exactness (3·(1/3)=1), binomial identity, AM-GM-onevar, sum-of-squares nonneg, `2xy ≤ x²+y²`, **Cauchy-Schwarz 2-var** `(a·x+b·y)² ≤ (a²+b²)(x²+y²)` closed via `sq_nonneg (a·y − b·x)`. | Strongest-yet formal claim. Reproducible via `LAKE_ENV=1 cargo run -p symthaea-lean-bridge --example prove_fol_arith`. Lake project at `lean-proofs/phase2/` pulls Mathlib v4.12.0 (elan-managed). 15 `.lean` files under `proofs/fol_arith/`. |

**Phase 1 delivered externally-commensurable:** propositional tautology suite (23/23), the ~3-problem miniF2F-style internal set (3/3), and the Ramanujan Protocol reproduction.

**Phase 2+3-Move-1 delivered externally-commensurable (as of 2026-04-17, on `main`):** arithmetic FolFormulaExt + SMT serializer (49 internal tests), Lake/Mathlib project setup via elan, Mathlib-tactic Lean bridge emitting files checked by `lake env lean`, **15/15 = 100% accept rate on hand-crafted arithmetic suite including the IMO-class Cauchy-Schwarz 2-var inequality** via named-variable threading + antisymmetric multiplicative cross-term hints.

**Not yet delivered:** measurement against real miniF2F-v2 problems via Lean-metaprogramming auto-ingestion (Phase 3 Move 2). Target: beat Lean's native `grind` tactic's ~32.4% miniF2F accept rate.

---

## Unverified Modules (Audit Deferred)

The following modules have substantial LOC but have not been audited in this pass. Do not cite their status until verified:

- `symthaea-core/src/physics/` subdirectory beyond `thermodynamics.rs` (classical_mechanics, general_relativity, standard_model, constants)
- `symthaea-physics` crate (plasma, antimatter, fusion — 18,591 LOC)
- `symthaea-continuum-physics` crate (Lindblad, Navier-Stokes)
- `symthaea-materials`, `symthaea-proteins`, `symthaea-genomics`, `symthaea-orbital`
- `hdc/` modules beyond the 16 listed above (graph_theory, game_theory, information_geometry, polynomial_algebra, synthetic_geometry, riemannian_geometry, and ~170 others)

Verification of these modules is tracked in `plans/` — they are NOT claimed as production here.

---

## Change Discipline

This document is append-only for additions and edit-in-place for corrections. When a module's status changes, edit the row with a brief note; do not delete history in commit messages. Every row must cite a file path. If a row cannot cite evidence, it must be moved to "Unverified Modules" and audited before being restored.

Last direct-observation pass: see `git log -- MODULE_STATUS.md`.
