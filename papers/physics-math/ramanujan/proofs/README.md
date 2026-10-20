# SMT-LIB2 Proof Witnesses

Each file in this directory is a formal proof obligation for one Ramanujan-Protocol invariant. Each asserts `∃x : dE/dt ≠ 0` and expects Z3 to return `unsat` — which is a formal proof that the invariant is conserved on the encoded dynamics.

## How they're built

```bash
cargo run -p symthaea-physics-bridge --example verify_invariants_formal
```

The example hand-derives `dE/dt = Σ (∂E/∂xᵢ)·(dxᵢ/dt)` via `SymExpr::diff` + `SymExpr::simplify`, serializes the resulting expression to SMT-LIB2 (polynomial subset of `QF_NRA`), and writes one `.smt2` per invariant. Output directory is `papers/physics-math/ramanujan/proofs/` by default, overridable via `SYMTHAEA_Z3_DUMP_DIR`.

## Current witnesses (all Z3-unsat as of this commit)

| File | Invariant | Status |
|------|-----------|--------|
| `harmonic_oscillator.smt2` | `E = x² + v²` | **unsat** |
| `kepler_angular_momentum.smt2` | `L = x·vy − y·vx` | **unsat** |
| `henon_heiles_6H.smt2` | `6H` (Hénon-Heiles ×6 rescale) | **unsat** |
| `mystery_ode.smt2` | `H = ½(px² + py²) + x² + y² + xy` | **unsat** |
| `duffing_4E.smt2` | `4E = 2v² + 2x² + x⁴` (Duffing conservative, ×4) | **unsat** |
| `quartic_anharmonic_4E.smt2` | `4E = 2v² + x⁴` (pure quartic, ×4) | **unsat** |
| `isotropic_2d_energy.smt2` | `2E = vx² + vy² + x² + y²` (2D isotropic, ×2) | **unsat** |
| `isotropic_2d_angular_momentum.smt2` | `L = x·vy − y·vx` (same system, 2nd invariant) | **unsat** |
| `linear_coupled_2E.smt2` | `2E = v1² + v2² + 2x1² + 2x2² − 2·x1·x2` (2-mode, ×2) | **unsat** |

Nine committed formal proof witnesses across six distinct dynamical systems, including a system with two independent invariants (2D isotropic harmonic oscillator: energy + angular momentum). The draft source additionally emits an exact-rational Hénon-Heiles obligation; it is pending exact-head solver verification and is not counted in these nine yet.

## Why the committed Hénon-Heiles witness uses 6H

The standard Hamiltonian is `H = ½(px² + py²) + ½(x² + y²) + x²y − y³/3`. The legacy serializer stores constants as `f64`; `1/3` becomes `0.3333333333333333`, which SMT-LIB2 interprets as a different exact rational. The committed `henon_heiles_6H.smt2` therefore clears the fractional coefficients by proving the equivalent scaled invariant.

The current source now adds an explicit `SymExpr::Rational(numerator, denominator)` representation and emits a second, unscaled `henon_heiles_exact_H` proof obligation. **That new obligation is not counted in the verified witness table above until a fresh exact-head run returns `unsat` and the generated SMT-LIB2 file is committed and independently re-checked.** The existing scaled witness remains the currently committed Hénon-Heiles proof artifact.

## Independent re-verification

Any SMT-LIB2-compliant solver closes each file. With Z3:

```bash
# verify one:
z3 -smt2 papers/physics-math/ramanujan/proofs/harmonic_oscillator.smt2
# expected output: unsat

# verify all:
for f in papers/physics-math/ramanujan/proofs/*.smt2; do
  printf "%-40s " "$(basename $f)"
  z3 -smt2 "$f" | tail -1
done
```

CVC5 and MathSAT should also return `unsat`. Different solvers may take different times (Z3 is fast on these: ~50 ms each).

## Problems NOT in this directory and why

| Problem | Reason |
|---------|--------|
| Lotka–Volterra `x − ln x + y − ln y` | Transcendental; outside `QF_NRA`. `verify_invariants_formal` reports `skipped` for it. |
| PCR3BP Jacobi | Showcase rediscovered a wrong expression (`cos(y/e)^(x³)`); the honest-numeric-failure row has no provable invariant. |
| Triangular numbers `n(n+1)/2` | Sequence identity, not a trajectory invariant; verified at each `n` by the showcase's per-point `verify_formal`, not as a single `dE/dt = 0` obligation. |

These gaps are documented honestly in `../VERIFY.md` and `../main.tex`.
