# MON-028 scorer module

Go 1.23, standard-library benchmark logic plus the MON-026 verifier as a local module dependency. No network access or third-party package is needed.

## 1. Prepare and sign the independent suite

The benchmark authority authors a suite JSON conforming to `../SUITE-v1.schema.json`. It fixes case identities, fault families, expected verdicts, replicate counts, fault-matrix/oracle/workload/environment/resource digests.

Compute its canonical digest:

```sh
go run ./cmd/suite-digest --suite ../SUITE-example.json
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
