# symthaea-evidence-policy-readiness

Composes the versioned assurance-policy manifest with Symthaea's existing deep evidence-readiness pipeline.

Only safety receipts bound to the exact current assurance-manifest digest are eligible for deep readiness. Historical receipts from an older policy revision remain auditable but are excluded from the current active evidence set.

The current manifest must also match the exact safety-contract digest and deployment/configuration/model/calibration context, and its external signature-verification receipt must validate against the exact manifest digest.

The remaining matched receipts are then evaluated through the existing trusted-time/lifecycle/quarantine, verifier-diversity, and atomic-coverage gates.

This means changing the rules for trusted time, quarantine, verifier diversity, atomic coverage, requalification, deployment scope, lifecycle handling, or evidence dependency invalidates prior policy-scoped readiness evidence automatically.

This crate never grants physical authority.
