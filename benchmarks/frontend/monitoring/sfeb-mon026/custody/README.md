# MON-026 custody package

Go 1.23, standard-library-only reference implementation.

## API

- `ComputeRoot(manifest)` computes the deterministic semantic manifest root.
- `SignManifest(&manifest, keyID, ed25519PrivateKey)` fills the root and signs it. The private key belongs in a separate signing service, not in the evidence bundle.
- `VerifyBundle(bundleRoot, manifest, policy)` checks the manifest, signature, artifacts, source boundary, clock policy and expected predecessor root.

The verifier requires the expected run ID, exact sequence and (for later sequences) expected predecessor root plus a trusted public key in `Policy.TrustedKeys`. The bundle's key ID selects a configured trust anchor; the public key is never accepted from the bundle itself.

## Example

```go
root, err := custody.ComputeRoot(manifest)
if err != nil {
    return err
}
_ = root

result := custody.VerifyBundle(bundleDir, manifest, custody.Policy{
    Now:                   time.Now().UTC(),
    MaxAge:                24 * time.Hour,
    ClockDomain:           "attestation-clock",
    MaxClockUncertaintyNS: int64(100 * time.Millisecond),
    ExpectedRunID:         "run-2026-11-27-001",
    ExpectedSequence:      2,
    ExpectedPreviousRoot:  previousBundleRoot,
    TrustedKeys: map[string]ed25519.PublicKey{
        "custody-key-2026-01": trustedPublicKey,
    },
})
```

A caller must persist the expected run ID, exact sequence and chain head independently. Missing expected identity/sequence or an empty predecessor for sequence 2 or later yields `indeterminate`; mismatches are blocked. The artifact directory is treated as an exact inventory: unlisted files and symlinks are rejected.

## Tests and benchmark

```sh
go test ./...
go test -race ./...
go vet ./...
go test -run '^$' -fuzz FuzzValidateRelativePathNeverPanics -fuzztime=30s
go test -run '^$' -bench BenchmarkVerifyBundle -benchmem -count=10
```

## Security boundary

The library rejects absolute/traversal/drive-qualified paths, duplicate paths, symlinks and non-regular files. It streams file hashing and compares the opened file's size and modification time before and after reading.

This is a portable reference, not a hardened hostile-filesystem implementation: a malicious actor with concurrent write access to the bundle directory may exploit filesystem races. Production use should verify against an immutable snapshot or use platform-specific no-follow/openat APIs, then retain the bundle in access-controlled immutable storage.

## Offline CLI

Build and run from this module directory:

```sh
go build -o mon026-verify ./cmd/verify
./mon026-verify \
  --bundle-root ./bundle/artifacts \
  --manifest ./bundle/manifest.json \
  --key-id custody-key-2026-01 \
  --public-key-hex "$TRUSTED_ED25519_PUBLIC_KEY_HEX" \
  --expected-run-id run-2026-11-27-001 \
  --expected-sequence 1 \
  --clock-domain attestation-clock \
  --max-age 24h \
  --max-clock-uncertainty 100ms
```

For sequence 2 or later, also supply `--expected-previous-root SHA256`. The manifest file must be outside `bundle-root`; the bundle root is an exact inventory and rejects unlisted files. Exit codes: 0 qualified, 2 blocked, 3 indeterminate, 64 usage/configuration error, 65 manifest parse error, 66 file access error.
