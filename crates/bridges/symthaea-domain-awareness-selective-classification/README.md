# symthaea-domain-awareness-selective-classification

Explicit adapter from `symthaea-selective-classification` into domain-awareness
identity hypotheses.

Only `EvidenceUsable` classifier output can produce hypotheses. `Abstain` and
`Incomplete` outputs produce no identity claim.

Set-valued predictions remain set-valued:

```text
{ small-aircraft, bird }
```

becomes two competing `IdentityHypothesis` values rather than a forced winner.
The existing domain-awareness epistemic logic then determines whether the result
is conflicting, insufficient, or sufficiently supported to be `Known`.

OOD abstention is preserved as an `OutOfDistribution` epistemic hint.

Even a high-support singleton classification can only become identity evidence.
It does not establish intent, risk, or physical authority.

## Verification

```bash
cargo test -p symthaea-domain-awareness-selective-classification
```
