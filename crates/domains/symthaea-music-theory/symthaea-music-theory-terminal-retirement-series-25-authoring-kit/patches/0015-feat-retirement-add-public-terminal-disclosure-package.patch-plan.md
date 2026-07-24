# Patch 0015: feat retirement add public terminal disclosure package

**Series:** 25

## Objective

Publish a privacy-bounded, independently verifiable statement that mutation authority has ended.

## Intended changes

- Bundle terminal checkpoint, retirement plan and receipt, signer and policy evidence, revocation summaries, archive completeness, successor status, and mandatory limitations.
- Exclude private keys, credentials, pseudonyms, listener records, and private governance proofs.
- Support deterministic offline verification.

## Required tests

- Missing revocation, wrong head, policy substitution, or secret-field inclusion fails.
- Package reproduces byte-for-byte.
- Verification reports authority retirement separately from archive completeness.

## Non-claims

- Does not disclose every private incident detail.
- Does not claim physical destruction of all key copies.
