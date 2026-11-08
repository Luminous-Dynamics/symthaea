# REA package signature evaluation contract

This isolated Rust harness validates a versioned PackageSignatureEvaluation statement. It is a data-contract and semantic-validation slice only. It does not parse APK signing blocks, verify Apple CMS signatures, build certificate chains, query revocation sources, invoke Android Package Manager, or call Apple Security.framework.

## Six independent claim results

Every statement carries separate statuses for:
1. package content integrity;
2. scheme-specific cryptographic signature validity;
3. signer public-key/certificate byte identity;
4. trust-policy acceptance under a named context;
5. package/distribution authorization;
6. runtime behavior.

Statuses are pass, fail, unknown, unsupported or not_evaluated. No aggregate trusted/safe/valid_package field exists.

## Semantic gates

- Package digest and byte length are explicit; potentially 64-bit lengths are canonical decimal strings.
- A content-integrity pass requires a separate member receipt whose package digest exactly matches the package digest.
- Cryptographic signature pass requires a scheme-specific passing result and exact signer SPKI/certificate DER SHA-256 records for that same scheme.
- Trust-policy pass requires a named policy/version, trust-anchor-set digest, time, target platform, signer-identity pass and revocation treatment.
- Valid signature by an unexpected signer is representable as signature_crypto=pass with trust_policy=fail and application_authorization=fail.
- APK v1/JAR cannot claim whole-package coverage. APK v4 pass requires a bound .idsig digest plus a passing v2/v3 result.
- Apple code-signing pass requires CodeDirectory and CMS evidence. Apple application authorization requires matching bundle ID, Team ID and signer key, a provisioning-profile digest and passing entitlement evaluation.
- Runtime pass requires a separately named suite, execution-evidence digest, platform build and device class.
- in-toto/DSSE attestation is a separate optional envelope; it does not upgrade the evaluation claims.
- The claim ceiling always preserves publisher benevolence, human/organization attribution, absence of malicious code, safety, intent, uninspected semantics, future updates and complete policy compliance as not established.

## Build and test

```bash
cargo fmt --manifest-path tools/rea-package-signature-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-package-signature-harness/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-package-signature-harness/Cargo.toml --locked --all-targets -- -D warnings
```

The tests use synthetic statements and negative controls; a green run qualifies the schema and semantic validator, not any real platform signature verifier.
