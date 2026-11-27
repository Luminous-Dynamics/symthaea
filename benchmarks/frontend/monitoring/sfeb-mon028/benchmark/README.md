# MON-028 scorer module

Go 1.23, standard-library benchmark logic plus the MON-026 verifier as a local module dependency. No network access or third-party package is needed.


## 0. Generate the suite from MON-025 and the independent oracle

The MON-025 fault matrix is the scenario schedule, not the sole source of expected answers. MON-028 keeps those answers in the separately maintained `ORACLE-v1.json`. The generator checks exact case coverage and rejects:
- missing or extra oracle cases;
- duplicate IDs;
- outcome disagreement between the matrix and independent oracle;
- positive controls not marked qualified;
- an unsafe control marked qualified;
- workload/environment/resource profiles with the wrong profile identity.

The default 30 replicates over 41 MON-025 cases produces 1,230 planned observations. The oracle has a different role from the fault matrix: it independently declares the expected verdict and family. The generator requires all matrix IDs to exist in the oracle, rejects extra oracle entries, and checks the matrix's embedded expected_outcome against the separately authored oracle.

From this module directory:

```sh
go run ./cmd/suite-from-mon025 \
  --fault-matrix ../../sfeb-mon025/FAULT-MATRIX-v1.json \
  --oracle ../ORACLE-v1.json \
  --workload-profile ../WORKLOAD-example.json \
  --environment-profile ../ENVIRONMENT-example.json \
  --resource-budget-profile ../RESOURCE-BUDGET-example.json \
  --campaign-id recovery-campaign-2026-11-27-001 \
  --benchmark-id mon-recovery-e2e-v1 \
  --replicates 30 \
  --output ../SUITE-generated.json
```

The example environment profile contains replacement placeholders and the sample budget is illustrative. Replace them with actual pinned versions/image digests, host and enforced cgroup limits before a confirmatory run. The generator hashes exact raw bytes for all five input documents into the resulting suite.

Package `SUITE-generated.json`, `ORACLE-v1.json`, the raw MON-025 fault matrix and the workload/environment/budget profiles through MON-027. The benchmark authority signs that exact allowlist. The scorer verifies the signed bundle before using any expected answers.

## 1. Prepare and sign the independent suite

The benchmark authority generates a suite JSON conforming to `../SUITE-v1.schema.json` using `cmd/suite-from-mon025`. It fixes all 41 fault case identities, oracle-derived families and outcomes, replicate counts, and the raw fault-matrix/oracle/workload/environment/resource digests.

Compute its canonical digest:

```sh
go run ./cmd/suite-digest --suite ../SUITE-generated.json
```

Package the suite as a raw artifact through MON-027, sign it using a dedicated suite-authority key, and retain the signed MON-026 manifest separately from the exact artifact root. Distribute that same custody-verified suite to every operator.

## 2. Produce and independently attest result artifacts

The SUT emits a result JSON conforming to `../RESULT-v1.schema.json`. It contains observed verdicts only; it must not contain expected outcomes. Every assigned case/replicate appears exactly once. Skipped, timed-out and infrastructure-error outcomes remain represented and do not qualify.

An independent benchmark observer—not the SUT—reviews the run receipts and packages the result file through MON-027 under a separate custody identity. The suite-authority and result-observer public keys must be distinct. The result bundle may also contain the baseline result as a second declared artifact.

## 3. Score and compare

The scorer first verifies both custody bundles with independently supplied public keys, exact run IDs, sequences and predecessor roots. It rejects raw, unsigned suite/result JSON as scoring authority.

```sh
go run ./cmd/score \
  --suite-bundle-root ./suite-bundle/artifacts \
  --suite-manifest ./suite-bundle/manifest.signed.json \
  --suite-artifact SUITE.json \
  --suite-key-id benchmark-authority-2026 \
  --suite-public-key-hex "$BENCHMARK_AUTHORITY_ED25519_PUBLIC_KEY_HEX" \
  --suite-run-id benchmark-suite-release-001 \
  --suite-sequence 1 \
  --suite-clock-domain benchmark-authority-clock \
  --suite-max-age 8760h \
  --suite-max-clock-uncertainty 100ms \
  --result-bundle-root ./result-bundle/artifacts \
  --result-manifest ./result-bundle/manifest.signed.json \
  --result-artifact candidate-result.json \
  --result-key-id independent-benchmark-observer-2026 \
  --result-public-key-hex "$BENCHMARK_OBSERVER_ED25519_PUBLIC_KEY_HEX" \
  --result-run-id benchmark-observation-001 \
  --result-sequence 1 \
  --result-clock-domain benchmark-observer-clock \
  --result-max-age 24h \
  --result-max-clock-uncertainty 100ms

# Optional paired baseline comparison. The baseline must be another declared
# artifact in the same signed result bundle and must use the same suite,
# workload, fault matrix, environment, resource budget and replicate set.
# Add:
#   --baseline-artifact baseline-result.json
```

For sequence 2 or later, also supply `--suite-previous-root` and `--result-previous-root` with independently retained chain heads.

Exit codes: 0 safety gate passed, 2 malformed/unsafe result or gate failure, 3 incomplete/indeterminate custody, 64 CLI usage error, 65 JSON parse error, 70 output error.

## 4. Verification commands

```sh
bash ../CHECKS.sh
```

The script runs gofmt, vet, unit tests, race tests, fuzzing and the scorer benchmark. It does not run PostgreSQL/Kafka/PITR experiments.

## Statistical interpretation

Aggregate and per-family false-qualified rates use the completed unsafe-case denominator. Planned unsafe observations are reported separately. Incomplete campaigns cannot pass the safety gate. The one-sided Wilson 95% upper bound is an approximate screening interval; common-host, common-reset and common-injector clustering requires cluster-aware confirmatory analysis.

A score report is a derived artifact. Preserve the suite custody bundle, result custody bundle, raw SUT histories, per-case receipts and score report together under MON-026/027 custody.

## Suite identity

`ComputeSuiteDigest` uses a MON-028 domain separator and a length-prefixed binary encoding with fixed field order. Cases are sorted by CaseID before hashing. The encoding is independent of JSON key order, whitespace and serializer-specific escaping. The digest does not replace an external MON-026/027 custody signature.
