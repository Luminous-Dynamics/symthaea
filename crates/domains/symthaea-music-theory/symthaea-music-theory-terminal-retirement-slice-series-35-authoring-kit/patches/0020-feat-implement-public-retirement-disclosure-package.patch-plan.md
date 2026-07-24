# Patch 0020: feat implement public retirement disclosure package

**Series:** 35

## Objective

Produce a privacy-bounded, independently verifiable terminal disclosure.

## Intended changes

- Bundle terminal checkpoint, plan, authorization, receipt, revocation summaries, archive status, custody summary, successor status, manifests, and limitations.
- Exclude credentials, private proofs, listener records, pseudonyms, and raw signing secrets.
- Support deterministic offline verification.

## Acceptance evidence

- Missing revocation, wrong checkpoint, policy substitution, or private-field leakage fails.
- The package rebuilds byte-for-byte.
- Retirement and archive completeness are reported separately.

## Non-claims

- Does not disclose every private incident detail.
- Does not prove physical destruction of all keys.
