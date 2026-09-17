# EKM-044 — Restart V2 Validation Receipt

## Purpose

EKM-044 adds a deterministic, read-only audit receipt for an exact decoded EKM-042 restart-v2 snapshot after the EKM-035 base validator, EKM-041 typed-schema validator, and EKM-043 cross-component validator all accept it.

The receipt is deliberately **not** a restore token. Validation success does not grant quarantine-construction or activation authority.

## Bound state

`EpistemicRestartValidationReceiptV1` binds:

- restart capture cycle;
- EKM-042 outer checksum;
- EKM-034 base-wire checksum;
- EKM-040 typed-schema-wire checksum;
- embedded historical V1 manifest digest;
- claimed EKM-039 V2 digest;
- complete EKM-035 validation report;
- complete EKM-041 validation report;
- complete EKM-043 cross-component validation report;
- explicit legacy-manifest assurance classification;
- explicit non-authority state;
- a domain-separated BLAKE3 receipt digest over all of the above.

`verify_against()` re-runs all validators and requires exact receipt equality with the supplied snapshot.

## Machine-readable authority boundary

Every receipt contains `RestartValidationAuthorityV1` with:

- `quarantine_construction_authorized = false`
- `activation_authorized = false`

These fields are private and have no public constructor or setter. The receipt therefore cannot be used through this API to manufacture restore authority.

The receipt digest is an integrity/reproducibility identifier only. It is **not** a digital signature, authorization, trusted timestamp, proof of origin, or proof that the bytes came from a trusted machine.

## Legacy V1 assurance

The receipt explicitly records:

`LegacyManifestAssuranceV1::EmbeddedDigestAnchorOnly`

This means EKM-043 independently re-runs typed belief decisions and binds the independently canonical typed-schema digest to the embedded V1 manifest digest. It does **not** mean the historical Debug-based V1 manifest digest has been independently reconstructed from untrusted bytes.

## Negative controls

The module includes controls showing that:

- a valid snapshot produces a deterministic receipt and verifies against itself;
- changing only the outer checksum changes receipt identity even when semantic DTO fields remain unchanged;
- tampering with the claimed V2 digest prevents receipt creation because EKM-043 fails first;
- the receipt exposes no quarantine or activation authority.

## Non-claims

EKM-044 does not:

- construct `EpistemicRestartCapsuleV1` or V2;
- construct a quarantine image;
- hydrate writable support or revision history;
- authorize or activate restored state;
- mutate evidence, belief, causal, world-model, or action state;
- authenticate a snapshot or receipt;
- perform file/database/network I/O;
- qualify any parent PR while GitHub Actions remains queued.

## Qualification boundary

CI is authoritative. At authoring time the parent EKM-043 exact-head CI run #7065 was still queued. No format, compile, Clippy, test, runtime, or wire-to-quarantine qualification is inferred from queued jobs.
