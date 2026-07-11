# symthaea-modal

Propositional **modal logic** — the formal, testable core of metaphysics
(modality: necessity □ / possibility ◇). Kripke-model evaluation plus bounded
validity / countermodel search across **K / T / S4 / S5**.

Second of the five "hard" knowledge domains
(`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`). Extends the logic substrate
(DPLL/FOL engine, `symthaea-proof-audit`) into modal reasoning. It gives you the
*machinery* metaphysicians argue with, not answers to metaphysical questions.

Pure `std`, zero deps, no `symthaea-core` link.

## Capabilities

- `kripke::KripkeModel` — worlds + accessibility + valuation; `satisfies`.
- `validity::{is_valid, find_countermodel}` — bounded search over frames
  satisfying a system's conditions (K any / T reflexive / S4 +transitive /
  S5 +symmetric). Finding a countermodel soundly disproves validity.

The tests demonstrate the classic axiom separations: `□p→p` (T) splits K/T,
`□p→□□p` (4) splits T/S4, `◇p→□◇p` (5) splits S4/S5.

## Example

```rust
use symthaea_modal::kripke::{implies, necessarily, var};
use symthaea_modal::validity::{is_valid, System};
let t = implies(necessarily(var("p")), var("p")); // □p → p
assert!(!is_valid(&t, System::K));
assert!(is_valid(&t, System::T));
```

```bash
cargo test -p symthaea-modal
```
