# symthaea-kinship

The **formal, testable core of anthropology**: a kinship algebra. Given a
genealogy, compute the kin term between any two people (English/Eskimo system).

First of the five "hard" knowledge domains
(`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`) — proof that each contains a
deterministic, testable core. Hooks into `mycelix-hearth`'s kinship model.
Culture/symbolism/ethnography are deliberately out of scope: this *computes
relations*, it does not *interpret culture*.

Pure `std`, zero deps, no `symthaea-core` link.

## Capabilities

- Consanguineal: self, parent…great-grandparent, child…great-grandchild,
  sibling, uncle/aunt…granduncle, niece/nephew…grandniece, cousins to arbitrary
  degree + removal ("second cousin once removed").
- Affinal: spouse (husband/wife), parent/sibling/child-in-law.

## Example

```rust
use symthaea_kinship::{Genealogy, Sex};
let mut g = Genealogy::new();
g.person("grandpa", Sex::Male).person("dad", Sex::Male).person("ego", Sex::Male);
g.parent_of("grandpa", "dad").parent_of("dad", "ego");
assert_eq!(g.relation("ego", "grandpa").unwrap(), "grandfather");
```

```bash
cargo test -p symthaea-kinship
```

## Not yet

Descent/exogamy rules, half- vs full-sibling, non-English terminology systems
(Sudanese/Hawaiian/Iroquois).
