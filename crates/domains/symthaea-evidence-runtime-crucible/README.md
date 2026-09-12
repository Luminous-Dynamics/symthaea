# symthaea-evidence-runtime-crucible

Adversarial integration crucible for the safety-evidence readiness stack.

The crucible deliberately starts from a synthetic safety case that is genuinely `Ready`, then perturbs one reviewed assumption at a time and requires readiness to fall away.

Covered perturbations include:

- evidence expiration
- explicit revocation
- unresolved contradiction
- deployment configuration drift
- model-manifest drift
- calibration-manifest drift
- verifier-wide quarantine
- trusted-time uncertainty straddling evidence expiry
- excessive wall-clock rollback

It also verifies reviewed recovery paths:

- replacement evidence for a new deployment configuration can restore readiness
- `RequireReplacement` quarantine remediation keeps pre-remediation evidence blocked while allowing newly verified replacement evidence
- contradiction resolution does not reactivate contradicted evidence, but a separate replacement receipt can restore readiness

The fixture intentionally does **not** instantiate the canonical DomainAwareness template. That avoids circularly assuming the DA-024..DA-027 lifecycle obligations are already discharged in the very test intended to qualify them.

A passing crucible report is test evidence only. It never grants physical authority and must still be independently verified before use in a formal safety case.

```bash
cargo test -p symthaea-evidence-runtime-crucible
```
