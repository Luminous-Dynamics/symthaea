# Claim Boundaries

This crate is a research scaffold. It must preserve a clear separation between measured local behavior and interpretation.

## Allowed claims

The current crate can support claims like:

- a local implementation check passed
- a local simulation was reproducible under the same seed and configuration
- a negative control separated matched-key recovery from wrong-key recovery
- a noise sweep degraded under a specified noise model
- a toy circuit artifact was exported for external inspection

## Disallowed claims

The current crate cannot support claims like:

- quantum consciousness was demonstrated
- quantum advantage was demonstrated
- physical entanglement was executed
- a hardware quantum backend validated the result
- a topology proxy proves consciousness
- a non-cryptographic fingerprint is a security receipt

## Audit helpers

The `audit` module provides local guardrails:

- `audit_binding_probe`
- `audit_negative_control`
- `audit_robustness`

These helpers are not peer review, formal verification, statistical validation, or security audit. They are small checks to prevent obvious over-interpretation in examples and automated reports.
