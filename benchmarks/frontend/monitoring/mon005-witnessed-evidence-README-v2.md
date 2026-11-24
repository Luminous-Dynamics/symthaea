# SFEB MON-005 — witnessed evidence checkpoints v2

## Executed locally

- Node.js v22.16.0.
- **45/45 deterministic cases passed**.
- **496 Merkle inclusion proofs** verified for every leaf position across tree sizes 1–31.
- **2,016 RFC 9162-style consistency proofs** verified across every prior tree size for every new tree size 2–64.
- Four Ed25519 test witnesses; **3-of-4 quorum** signed two tenant-scoped checkpoints containing 12 then 18 records.
- The witness accepts an update using only the new six-record suffix, the prior checkpoint, one consistency proof and inclusion proofs for each suffix record.
- One unavailable signer is tolerated; two signatures are insufficient.
- The exact raw-source and gzip executions produce byte-identical output; raw and compressed stdin syntax checks pass.

## Run

```sh
node --check benchmarks/frontend/monitoring/mon005-witnessed-evidence-v2.mjs
node benchmarks/frontend/monitoring/mon005-witnessed-evidence-v2.mjs

gzip -dc benchmarks/frontend/monitoring/mon005-witnessed-evidence-v2.mjs.gz | node --input-type=module --check -
gzip -dc benchmarks/frontend/monitoring/mon005-witnessed-evidence-v2.mjs.gz | node --input-type=module
```

Only Node.js built-in modules are used.

## What changed from v1

V1 asked a witness to recompute the entire historical prefix at each checkpoint. V2 uses the RFC 9162 consistency-proof algorithm to prove that a new Merkle root extends the previously trusted root. The witness validates the appended record hash chain from the prior terminal record hash and checks each appended record's inclusion proof against the new root. The witness API receives only the new suffix and proof bundle; it does not receive the complete historical event list.

The proof generator in this model still uses a full local event array to produce proof bundles. This is acceptable for a reference test but is not the scalable production log architecture. A production log service needs efficient range/subtree retrieval and consistency-proof generation; the witnesses need only the compact proof and suffix.

## Important limitations

This is not a distributed deployment or Byzantine-fault-tolerance demonstration. All witness instances are simulated in one process and keep in-memory state. Test keys are deterministic fixtures. No independent service domains, durable compare-and-set, PostgreSQL adapter, network protocol, HSM/KMS, production key rotation/revocation, full RFC 8785 JCS implementation, external transparency-log adapter, timestamp authority, real sink or independent readback exists.

The 3-of-4 quorum argument assumes at most one Byzantine witness and at least one honest member of every two-quorum intersection that durably refuses to sign different heads for the same tenant/epoch. Simulating four witnesses in one process does not establish this assumption operationally.

A signed checkpoint certifies that test keys signed a checkpoint. It does not establish the truth of source, authorization, execution or sink receipts. Those still require independent validation. No throughput or SOTA superiority result is claimed.
