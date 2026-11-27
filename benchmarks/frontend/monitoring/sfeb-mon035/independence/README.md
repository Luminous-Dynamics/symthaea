# MON-035 — Observer Common-Cause Independence

MON-035 checks whether two recovery-evidence observers have separate transitive failure-domain dependencies, rather than merely different names or process IDs.

```sh
cd benchmarks/frontend/monitoring/sfeb-mon035/independence
go run ./cmd/qualify --input INPUT-example.json
bash CHECKS.sh
```

The v1 profile checks eight domains: operator, identity provider, control plane, compute host, storage, network, time source and signer. Any shared domain blocks independence. Missing dependency evidence is indeterminate.

This is an offline semantic check, not infrastructure attestation. The inventory SHA-256 is not authenticated here; a trusted external collector and custody verifier must establish inventory completeness before MON-024 consumes the result.

The benchmark uses two 1,000-node dependency closures. No live infrastructure, cloud account, KMS/HSM or observer control plane is queried.
