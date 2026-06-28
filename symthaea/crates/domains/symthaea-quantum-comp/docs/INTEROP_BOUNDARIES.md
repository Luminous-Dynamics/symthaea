# Integration Boundaries

Alpha.9 adds integration-boundary declarations for future adapters.

These declarations are documentation-shaped data. They do not perform external execution or attestation.

## Targets

- `LocalLab`: local-only reports and examples.
- `Symthaea`: future substrate registry or cognition runtime.
- `Mycelix`: future source-chain or governance receipt layer.
- `ExternalQuantumBackend`: future QASM/backend observation path.

## Authorities

- `ExportOnly`: may export reports or artifacts only.
- `ObserveOnly`: may observe external outputs but not certify them.
- `AttestationRequest`: may request signing from another system.
- `BlockedInAlpha`: not allowed in alpha without review.

## Rule

The alpha crate does not sign Mycelix source-chain entries and does not validate hardware backend runs.
