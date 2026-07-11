# symthaea-legal-reasoning

The **formal, testable core of law** — deontic logic, defeasible rules, and
Hohfeldian jural relations. Third of the five "hard" knowledge domains
(`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`); hooks into governance/civic +
`EthicsEngine`.

Pure `std`, zero deps, no `symthaea-core` link. It *applies and checks* rules —
it does **not** interpret statutes, reason from precedent, or decide what the law
should be.

## Capabilities

- `deontic` — obligation/permission/prohibition (`Norm`), norm-set consistency
  (can't be both required and forbidden), permission derivation (`O→P`, `F→¬P`).
- `defeasible` — `Rule` (conditions + exceptions → conclusion), forward-chained
  to a fixpoint. The "birds fly unless penguin" pattern for statutes-with-exemptions.
- `hohfeld` — the eight jural positions with correlatives (Right↔Duty,
  Power↔Liability, …) and opposites.

## Example

```rust
use symthaea_legal_reasoning::deontic::{Norm, is_consistent};
let norms = vec![Norm::Obligatory("testify".into()), Norm::Forbidden("testify".into())];
assert!(!is_consistent(&norms)); // can't be both required and forbidden
```

```bash
cargo test -p symthaea-legal-reasoning
```
