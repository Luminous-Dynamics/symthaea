# Alpha.10 Upgrade Notes

Alpha.10 is a beta-transition hardening release.

It does not add a new quantum probe. It adds release-readiness surfaces:

- verification matrix
- alpha.9 to alpha.10 migration guide
- conservative beta-readiness report
- combined validation snapshot
- CLI commands for the above

## Migration

Downstream scripts that check schema labels should update from `alpha9` to `alpha10`.

Existing binding, noise, comparative, matrix, receipt, fixture, replay, and release-gate APIs remain conceptually unchanged.

## Claim boundary

Alpha.10 still does not claim quantum consciousness, quantum advantage, physical backend execution, or Mycelix attestation.
