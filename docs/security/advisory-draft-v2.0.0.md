# Draft Security Advisory: Experimental HDC Claims and Mesh Authentication

**Draft date:** 2026-07-13
**Status:** **N/A — not published.** There are no known users of this repository
or of any release built from it. CVE/GHSA assignment and coordinated disclosure
exist to inform affected third parties; with no confirmed consumers, there is
no one to coordinate with, and formal disclosure machinery would be
disproportionate to the actual exposure. This draft is retained only as the
historical record of what was found and considered, in case a future user
report changes that assessment (see "If a user ever reports reliance" below).
**Affected release:** Symthaea `v2.0.0`
**Fixed release:** contained in the same working tree; no separate advisory
release planned

## Summary

Symthaea `v2.0.0` contains experimental HDC constructions whose historical
names imply cryptographic properties they do not provide. It also uses an
eight-bit authentication tag in version-1 `WisdomPacket` messages.

The affected HDC constructions include a forgeable linear tag, records labelled
as threshold shares that individually reveal the secret, a reversible rotation
labelled as a commitment, shared-mask aggregation labelled as FHE, and
deterministic sensor context labelled as secret key material.

## Impact

Impact depends on downstream use:

- Any workflow treating `HdcMac` as an integrity boundary permits chosen-message
  forgery after one known message/tag pair.
- Any workflow treating `HdcThresholdSharing` as `k`-of-`n` protection permits
  recovery from one share.
- `HdcCommitment` provides neither hiding nor binding.
- Shared-mask aggregation reveals pairwise plaintext XORs and Hamming distances.
- Sensor-only context values may be enumerable and must not be treated as secret
  key entropy.
- Version-1 mesh authentication can be guessed with probability `1/256` per
  attempt.

No repository evidence establishes that these APIs protected deployed keys,
funds, or third-party data. Maintainers must not interpret that absence as proof
that no downstream deployment exists.

## Containment in the unreleased fix

- Version-2 `WisdomPacket` uses an untruncated HMAC-SHA-256 tag over the complete
  versioned packet, including TTL.
- Legacy/unknown packet wire versions are rejected rather than silently treated
  as authenticated.
- Forgeable HDC packet helpers require an explicit
  `insecure-experimental-crypto` feature.
- Shared-mask network aggregation is removed from the distributed feature
  profile.
- Affected compatibility APIs are deprecated and explicitly documented as
  insecure demonstrations.
- Sensor context is no longer installed automatically as a session key.
- Adversarial regression tests preserve each counterexample.

These changes do not make the quarantined HDC constructions secure.

## Required user action

Before releasing this advisory, contact known `v2.0.0` users privately and ask
whether they enabled or relied upon:

1. mesh packet authentication;
2. `fhe-wisdom` or `CollectiveWisdomPool`;
3. `HdcMac`, `HdcThresholdSharing`, `HdcCommitment`, or `HdcTreasuryPool`;
4. sensor-derived context as session-key material;
5. serialized version-1 `WisdomPacket` data that must interoperate after upgrade.

Users should stop treating all affected HDC APIs as security boundaries. Mesh
peers must upgrade together to the version-2 wire format; version-1 packets must
not be accepted as authenticated.

## Compatibility

The authenticated packet layout changes from 2,072 to 2,104 bytes and adds an
explicit wire-version byte plus a 32-byte tag. This is intentionally not
authentication-compatible with version 1.

## Credit and evidence

The issue was found during Symthaea's cryptographic claim-integrity pass. The
full internal evidence, consumer inventory, attack descriptions, and validation
record are in `2026-07-13-cryptographic-claim-integrity.md` and
`crypto-claim-inventory.json` in this directory.

## Decision record (2026-07-13)

- **CVE/GHSA: not assigned.** No known users exist. Assigning a tracking
  identifier for a finding with no confirmed downstream consumer does not
  inform anyone and was judged disproportionate. If a user is ever
  discovered, revisit this decision before any further exposure and consider
  filing at that time instead of retroactively.
- **Independent professional review of the version-2 mesh protocol: not
  commissioned.** No budget exists for a paid audit, and the actual novel
  surface is thin — HMAC-SHA-256 is a standard, heavily analyzed primitive;
  the only thing that could be wrong is *how it is applied*. A self-review
  checklist substituted for a paid audit (see
  `2026-07-13-cryptographic-claim-integrity.md`, "Self-review checklist"):
  all security-relevant fields are inside the tag (including TTL), there is
  no downgrade/legacy-acceptance path, comparison is constant-time, and key
  establishment is explicitly out of scope for this pass. One real gap was
  found this way: no replay/freshness check on `sequence` at the
  authentication layer (see CI-007). Free/no-cost outside review (e.g.
  posting the wire-format spec to a public cryptography forum) remains an
  option for later, not a blocker now.
- **Downstream coordination: not applicable.** No known users to contact.
- **Publication: this advisory stays unpublished.** The internal findings
  document and inventory are the durable record instead.

## If a user is ever reported

Re-open this draft, fill in the checklist below, and treat "no known users"
as no longer true for severity purposes:

- [ ] Reporting user's actual reliance identified (mesh auth? `fhe-wisdom`?
  `HdcMac`/`HdcThresholdSharing`/`HdcCommitment`/`HdcTreasuryPool`? sensor-derived
  session keys? raw v1 `WisdomPacket` bytes needing to interoperate?).
- [ ] Deployment impact and severity reassessed given the actual reliance.
- [ ] CVE/GHSA decision reconsidered.
- [ ] Coordinated upgrade path communicated to that user.
- [ ] This document updated or published as appropriate.
