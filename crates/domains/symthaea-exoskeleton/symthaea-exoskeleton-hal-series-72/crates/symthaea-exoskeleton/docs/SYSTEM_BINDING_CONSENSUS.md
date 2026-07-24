# System Binding Consensus

Actuation requires every safety domain to agree on one release, deployment,
calibration, hardware manifest, persistent journal head, boot epoch, calibration
revision, and contract version.

A collection of individually valid permits is not sufficient. Missing, stale,
replayed, or differently bound attestations yield zero authority. A mismatch is
latched because it may indicate rollback, partial update, stale process state,
or substitution across qualification artifacts.
