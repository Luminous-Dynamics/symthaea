# symthaea-chronicle

The **formal, testable core of history-as-narrative** — historical events, causal
chains, and anachronism detection. Fourth of the five "hard" knowledge domains
(`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`).

Pure `std`, zero deps, no `symthaea-core` link. **Non-duplication:** rich
Allen-interval temporal reasoning lives in the main crate
(`src/consciousness/temporal/`) and should be used when integrated — this crate
uses only minimal date comparison and adds the genuinely new *events + causation
+ anachronism* layer. It computes ordering/causation/anachronism; it does not
judge significance or synthesize narrative.

## Capabilities

`Chronicle` →
- `chronological()` — events in date order.
- `precedes` / `contemporaneous` — interval comparison.
- `causally_leads_to` / `causal_chain` — transitive causal reachability + path.
- `is_anachronistic(entity, year)` — reference outside a lifespan (Napoleon +
  smartphone → true).

## Example

```rust
use symthaea_chronicle::Chronicle;
let mut c = Chronicle::new();
c.event("press", 1440, None).event("reformation", 1517, None)
 .causation("press", "reformation").entity("napoleon", 1769, 1821);
assert!(c.causally_leads_to("press", "reformation"));
assert_eq!(c.is_anachronistic("napoleon", 2007), Some(true));
```

```bash
cargo test -p symthaea-chronicle
```
