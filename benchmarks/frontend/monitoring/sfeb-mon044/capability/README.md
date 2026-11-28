# MON-044 Capture Capability Evaluator v0.1

A dependency-free Go evaluator for version- and configuration-bound capture capability and negative-space coverage.

## What it checks

- Connector distribution, server/plugin, broker/client, configuration, publication, schema-history, snapshot, transaction metadata, operation-filter, column-policy and topic-routing identity.
- Independent required-scope source identity and digest.
- Exact required-item and capability-item scope matching.
- Supported claims bound to passing conformance profile/report and the exact capability identity.
- Conditional claims with every named prerequisite independently evidenced.
- Unsupported and excluded required scope blocked.
- Not-applicable claims with a justification digest.
- Unknown and missing required capability held indeterminate.
- Duplicate and malformed declarations blocked.
- Optional negative-space and out-of-scope declarations remain visible.

## Verdicts

- capability_verdict: quality of the capability statements and evidence supplied for declared items.
- coverage_verdict: closure of the mandatory recovery scope.
- campaign_verdict: always indeterminate in v1 because source inventory, capability issuer, conformance report and runtime receipts are not authenticated.
- production_action_permitted: always false.

An optional capability can remain unknown without invalidating a fully covered required scope, but it is reported. An unknown or unsupported required item never qualifies.

## Important limitation

SHA-256 digests bind bytes; they do not authenticate the connector vendor, conformance runner, inventory source, runtime configuration or evidence issuer. A passing test demonstrates only the tested fixture and pinned identity. MON-041 custody and independent source capture remain mandatory.

The public Debezium/PostgreSQL fixture is synthetic. It is not evidence that the named version combination was deployed or live-tested. The deterministic fixture values and digests are not a vendor capability registry.

## CLI

`go run ./cmd/qualify --input INPUT.json`

Exit codes: 0 required scope covered offline (campaign still indeterminate); 2 invalid/unsupported required coverage; 3 insufficient evidence; 64 usage; 65 malformed input; 66 input unavailable; 74 output failure.

Run `bash CHECKS.sh` for formatting, vet, unit, race, CLI, schema and benchmark checks.
