# miniF2F Proof Fixtures

Auto-generated Lean 4 proof scripts emitted by `symthaea-lean-bridge`.

## Provenance

These files are produced by:

```bash
cargo run -p symthaea-lean-bridge --example prove_minif2f
```

The example drives `symthaea_core::hdc::logic_engine` (DPLL, natural deduction, modus ponens) for each fixture and translates the resulting `ProofResult` into Lean 4 via the bridge module. **These are not hand-written proofs.** They are the first concrete output of the Phase 1 Lean pipeline per `plans/2-please-make-precious-fairy.md` WS-B.

## Fixtures

| File | Theorem | Expected status |
|------|---------|-----------------|
| `minif2f_identity_impl.lean` | `∀ P: Prop, P → P` | Accepted by core Lean 4 `tauto` (intuitionistic) |
| `minif2f_excluded_middle.lean` | `∀ P: Prop, P ∨ ¬P` | Requires classical — Mathlib `tauto` or manual `Classical.em` |
| `minif2f_modus_ponens_deducibility.lean` | `∀ P Q: Prop, (P → Q) → P → Q` | Accepted by core Lean 4 `tauto` |

## External verification

The Phase 1 target is `lean4 --check` acceptance. Install Lean via [`elan`](https://github.com/leanprover/elan):

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
elan toolchain install leanprover/lean4:v4.12.0   # or current stable
```

Then:

```bash
# Emit + verify in one shot:
LEAN_CHECK=1 cargo run -p symthaea-lean-bridge --example prove_minif2f

# Or check an individual file:
lean --check proofs/minif2f/minif2f_identity_impl.lean
```

## Known limitations (Phase 1)

- **Core Lean 4 vs Mathlib `tauto`:** core Lean 4's `tauto` is intuitionistic. Fixtures that require classical reasoning (e.g. excluded middle) may fail without a Mathlib import. Current emitter does not add `import Mathlib.Tactic.Tauto` — that's Phase 2 work once we measure how many miniF2F problems need it.
- **Nonlinear real arithmetic:** Z3-produced `QF_NRA` witnesses are emitted as `sorry`-tagged Lean and count as failures in the external-verify gate (not false positives).
- **Branching reconstruction:** `ProofStepLogic` is a flat `Vec`, not a tree. For case-split proofs, the current bridge falls back to `tauto` rather than reconstructing branches. If miniF2F error-triage shows >20% of failures come from this, we'll refactor `ProofTree` in Phase 2.

## What success looks like

At Phase 1 week 6: `cargo run --example prove_minif2f` over the full miniF2F-v2 set produces `proofs/minif2f/*.lean` files; `lean4 --check` accepts ≥ 30%. The `lean_check=accepted` column in the emitted CSV is the metric of record.
