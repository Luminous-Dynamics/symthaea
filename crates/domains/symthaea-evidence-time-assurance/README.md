# symthaea-evidence-time-assurance

Trusted clock/uncertainty assurance for lifecycle-managed safety evidence.

Evidence validity, expiry, revocation, supersession, quarantine, and requalification all depend on time. A bare integer timestamp is therefore not sufficient evidence for deployment readiness.

`EvidenceTimeGuard` requires a deployment-reviewed clock policy with an exact clock source, clock domain, epoch reference, maximum allowed clock uncertainty, and bounded backward wall-clock tolerance. It also requires strictly increasing monotonic time and replay-resistant sample ids.

Any time-integrity violation latches the guard `Untrusted`; constructing a new guard after a reviewed clock/epoch recovery is required before readiness can be trusted again.

A trusted sample yields an uncertainty interval `[earliest, latest]`, not just a point estimate. The provided readiness wrapper evaluates the complete quarantine/deployment/lifecycle stack at both interval endpoints. Readiness is `Ready` only when both endpoints are `Ready`.

This prevents evidence from being accepted when it may already be expired or may not yet be valid anywhere inside the trusted clock uncertainty window.

The time-assurance layer never grants physical authority.

```bash
cargo test -p symthaea-evidence-time-assurance
```
