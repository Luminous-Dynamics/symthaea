# Authenticated REA Evidence

This manual workflow adds an authenticated signature/provenance layer to the portable Unblob evidence bundle.

## Run

1. Confirm a successful **REA Pinned Provider Smoke** run was produced by a push to `main` (not by a pull request or another branch).
2. On the `main` branch, run **Attest REA Firmware Evidence** from GitHub Actions and provide that provider-smoke run ID.
3. The workflow checks the source run's identity and success, downloads its evidence artifact, checks out the exact source revision that produced it, and runs `symthaea-rea-evidence-verifier`.
4. It signs the bundle manifest as the in-toto statement subject, using GitHub's OIDC-backed attestation action with a custom, versioned evidence predicate. That predicate binds the manifest digest to the outer receipt digest, the provider run, source revision, semantic digests and the explicit claim boundary.
5. The workflow immediately verifies the signature bundle against the expected repository, signer workflow, source ref, source commit and predicate type before uploading the portable attestation artifact.

The workflow intentionally refuses provider runs whose event is not a push to `main` or whose conclusion is not success. It does not sign pull-request outputs by default.

## Verify as a consumer

Download both the original provider-evidence artifact and the `rea-provider-smoke-signed-attestation` artifact from the attestation workflow run. The latter contains the Sigstore bundle plus predicate and verification receipt.

Verify the signed subject using GitHub CLI:

    gh attestation verify bundle-manifest.json \
      --bundle /path/to/attestation.bundle.jsonl \
      --repo Luminous-Dynamics/symthaea \
      --signer-workflow Luminous-Dynamics/symthaea/.github/workflows/rea-evidence-attestation.yml \
      --predicate-type https://github.com/Luminous-Dynamics/symthaea/attestations/rea-provider-smoke/v1 \
      --source-ref refs/heads/main \
      --source-digest EXPECTED_ATTESTATION_WORKFLOW_COMMIT \
      --format json

Use the source digest/ref corresponding to the **attestation workflow run**, not the earlier provider-smoke run. The custom predicate separately identifies the source revision and run that produced the evidence.

Then check that:
- the statement subject SHA-256 equals the downloaded `bundle-manifest.json` digest;
- the signed predicate's `subject.receiptSha256` equals the downloaded `receipt.json` digest;
- `providerRun.sourceHeadSha` equals the expected source revision associated with the successful main push;
- the evidence tier is exactly `unblob-provider-smoke`;
- REA CLI/MCP parity, firmware execution, emulation and device qualification remain false.

The Rust receipt verifier should then be invoked with the expected provider source head and the receipt SHA-256 extracted from this **verified, signed predicate** using `--expected-receipt-sha256`. Do not supply a digest calculated solely from the untrusted evidence bundle as an “independent” anchor.

## Claim boundary

The subject is a manifest whose entries bind all raw reports, semantic projections, inputs, expected payloads and receipts. The signed predicate binds the outer receipt, which is excluded from the manifest to avoid a circular digest. GitHub's attestation certificate and transparency proof authenticate the statement under its signer/source policy; the custom predicate supplies the domain facts, which still need policy verification.

This workflow does not claim SLSA level or full in-toto conformance. It authenticates only a narrow provider-smoke evidence record; it does not imply REA CLI/MCP parity, arbitrary-firmware extraction completeness, recovered firmware behavior, emulation, or device qualification. A valid signature says who attested to the statement and what digest they bound—not that the underlying firmware analysis is safe or correct.
