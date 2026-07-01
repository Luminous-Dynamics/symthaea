# Alpha.9 Release Checklist

Before publishing or importing alpha.9 into a larger Symthaea workspace, run the local verification script on a machine with Rust installed:

`./scripts/verify-local.sh`

Manual checks:

- Confirm README still states non-claims clearly.
- Confirm schema labels end in `alpha9`.
- Confirm CLI help lists `inventory` and `manifest`.
- Confirm release manifest blocks quantum consciousness and quantum advantage claims.
- Confirm receipts still state they are non-cryptographic and not Mycelix source-chain entries.
- Confirm any external backend adapter remains outside this crate or is feature-gated with raw metadata requirements.

Publishing caveat:

Alpha.9 is still an alpha research scaffold. Do not present it as a validated quantum backend, medical/safety system, or production engineering decision tool.
