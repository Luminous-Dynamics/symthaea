# MON-014 version-pinned interoperability inventory v1

**Snapshot date:** 2026-11-26. Resolve every pin to a commit, release or explicit Internet-Draft version in benchmark output.

| Artifact | Snapshot pin/status | MON use | Boundary |
|---|---|---|---|
| OAR source repository | openactionrecord/oar commit dd1dc0393385048c26b2b47ac543859a1ba6cdd2 | Maintained reference verifier and official vectors | OAR -01 additions are not yet implemented in this code |
| OAR TypeScript core / CLI | 0.6.0; Node >=20.12; pnpm 10.28.0 lock | Primary maintained offline verifier | Not an independent implementation from the same repository |
| OAR Python | 0.6.0 | Second language implementation | Python dependency versions are not pinned by a lockfile in the repository snapshot; record resolved versions |
| OAR vectors | 18 main + 6 compatibility | Expected results and backward compatibility | Published keys and TSA roots are test-only; never production trust anchors |
| OAR format | OAR 0.3; accepts legacy 0.2-mvp and aar-evidence/0.1 compatibility | Action lifecycle, approval, authority check, state commitments, readback confirmation | Not source provenance or cross-layer causal join by itself |
| OAR -01 | Active individual Internet-Draft, 25 Nov 2026 | Versioned future mapping | -01 additions are planned in current OAR implementation |
| Agent Action Capsule | draft-mih-scitt-agent-action-capsule-06, 24 Nov 2026 | Producer action/effect and cross-party evidence | Individual Internet-Draft; not published RFC or endorsed IETF standard |
| COSE Receipts | RFC 9942 | Maintained COSE receipt substrate | Select and pin a VDS profile; proof modes differ |
| SCITT architecture | RFC 9943 | Signed statement transparency / registration | Registration is not source truth or completeness proof |
| COSE Receipts for MMRs | draft-ietf-scitt-receipts-mmr-profile-00, 21 Nov 2026 | MMR-specific receipt adapter | Active WG Internet-Draft, not RFC |

## OAR source and vector identities

- Repository: https://github.com/openactionrecord/oar
- OAR main source commit: dd1dc0393385048c26b2b47ac543859a1ba6cdd2
- TypeScript core package manifest blob: 2cfd96a619aee94c22309b3aadb7d8ca2cf01e89
- TypeScript verifier source blob: 48ebbfb27842176424bc3f76c94eb53b3c1852e6
- Main manifest blob: a9c880c1e6d66c5d04b60802e8362475d388a475
- Compatibility manifest blob: 2564393bb52253bc6a8235535cb436ceb2dd723a
- Current core / CLI / Python package version: 0.6.0
- The main manifest enumerates 18 cases; the compatibility manifest enumerates 6.

The blob identifiers above are Git object IDs, not SHA-256 content digests. The source commit is the immutable repository pin.

## Outcome projection rules

OAR's lifecycle is phase-aware. Do not map every occurrence of the word "confirmed" or "executed" without checking its phase and predecessor chain.

- confirmation / confirmed: OAR outcome confirmed, subject to full upstream verification and lifecycle linkage.
- confirmation / not_confirmed: not a committed outcome; retain unresolved.
- execution / refused: refused action if the lifecycle and decision chain are consistent.
- execution / executed or failed: execution was reported, but not independently confirmed by OAR confirmation.
- decision / denied: no action should execute; a later execution record conflicts.
- decision / held or allowed, approval / approved: nonterminal; they are not committed effects.
- approval rejected or expired: refusal candidate if no conflicting execution.
- missing or unknown lifecycle predecessor: indeterminate; preserve diagnostic.

The MON adapter preserves the complete upstream report and per-record checks. It does not replace OAR's own outcome and evidence-level vocabulary.
