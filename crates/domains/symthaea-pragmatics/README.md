# symthaea-pragmatics

The **formal, rule-based core of linguistic pragmatics** — speech-act
classification, presupposition-trigger detection, and deixis resolution. Fifth
of the five "hard" knowledge domains
(`symthaea/HARD_DOMAINS_PLAN_2026-07-07.md`).

Pure `std`, zero deps, no `symthaea-core` link. **Scope:** pragmatics is applied
theory-of-mind; rich intent/implicature *inference* belongs to the main crate's
ToM + NSM + Broca, which this crate does NOT reimplement. It provides the
decidable layer beneath that. Open-ended conversational implicature is out of
scope.

## Capabilities

- `speech_act::classify` → Searle's five types (assertive/directive/commissive/
  expressive/declarative).
- `presupposition::detect` → definite-description / factive / aspectual / cleft
  triggers ("John stopped smoking" presupposes he smoked).
- `deixis::resolve` → I/you/here/now against a `Context`.

## Example

```rust
use symthaea_pragmatics::speech_act::{classify, SpeechAct};
assert_eq!(classify("I promise to help."), SpeechAct::Commissive);
```

```bash
cargo test -p symthaea-pragmatics
```
