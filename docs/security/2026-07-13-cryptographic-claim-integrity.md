# Symthaea Cryptographic Claim Integrity Pass

**Date:** 2026-07-13, follow-up review pass 2026-07-14  
**Status:** internal containment complete, including a second independent
review pass (see "Follow-up review pass" below)  
**Deployment decision:** prohibited  
**Disclosure:** N/A — no known users (see `advisory-draft-v2.0.0.md`)

## Decision

Cryptographic-discovery work is paused until the repository's existing
security-labelled constructions are inventoried, adversarially tested,
quarantined, and removed from every real security boundary.

This is a claim-integrity failure: functional algebra and successful round-trip
tests were treated as evidence for security notions they do not establish.

## Confirmed findings

### CI-001 — Linear HDC tag is forgeable

Historical claim: `HdcMac` authenticates a `BinaryHV`.

Construction:

```text
tag = message XOR rotate(key, 7)
```

Given one known `(message, tag)` pair, an attacker computes
`mask = message XOR tag`. For any chosen `message2`, the forged tag is
`message2 XOR mask`. This succeeds without the key.

**Security notion violated:** existential unforgeability under chosen-message
attack.  
**Severity:** high where used as an integrity boundary.  
**Containment:** types and mesh helper methods deprecated; replacement required.

### CI-002 — Each threshold share reveals the secret

`HdcShare` stores both:

```text
share = secret XOR mask
mask
```

Therefore `share XOR mask = secret` for every individual share. Recovery does
not enforce or use `k`; it accepts any non-empty slice.

**Security notion violated:** privacy of every set of fewer than `k` shares.  
**Severity:** critical if used to protect real secrets.  
**Containment:** construction deprecated; treasury/collective uses classified as
insecure experiments.

### CI-003 — Rotation is not a commitment scheme

Construction:

```text
commitment = rotate(secret, offset)
```

The offset has only 16,384 effective values, rotation preserves visible
structure, and a commitment can be opened as different rotated messages by
choosing matching offsets.

**Security notions violated:** computational/statistical hiding and binding.  
**Severity:** high if used for commitments.  
**Containment:** type deprecated; no non-test production consumer found.

### CI-004 — Shared-mask aggregation leaks relationships and is not FHE

For ciphertexts `c1 = p1 XOR mask` and `c2 = p2 XOR mask`:

```text
c1 XOR c2 = p1 XOR p2
```

Hamming distance is also preserved. The implementation deliberately depends on
this common-mask behavior for classification and majority aggregation. It is an
algebra demonstration, not semantic-security-preserving FHE.

**Security notions violated:** joint-message confidentiality and the claimed
privacy-preserving protocol.  
**Severity:** high in the `fhe-wisdom` network profile; research-only elsewhere.  
**Containment:** pool and mask-sharing helpers deprecated and relabelled;
`fhe-wisdom` removed from the distributed-network feature profile.

### CI-005 — Sensor context is enumerable, not secret entropy

`symthaea-soma` quantizes a small set of sensor readings to deterministic seeds,
maps them to public HDC vectors, combines them, and hashes the result. An
attacker can enumerate plausible sensor states and compare candidate keys.

**Security notion violated:** unpredictability required of secret key material.  
**Severity:** latent high. The value was installed as a Holon session key, but
outbound ChaCha20-Poly1305 encryption is not yet wired.  
**Containment:** automatic installation removed; the API is deprecated and may
only be treated as a public context fingerprint.

### CI-006 — Mesh packet tag was only eight bits

The released version-1 mesh layout stored only the first byte of a keyed hash.
An attacker could guess a valid tag with probability `1/256` per attempt. Rate
limiting does not turn an eight-bit authenticator into an acceptable integrity
boundary.

**Security notion violated:** meaningful message-authentication strength.  
**Severity:** high for authenticated mesh workflows. Safety-critical messages
were fail-closed when a key was configured, but the tag itself was inadequate.  
**Containment:** wire format version 2 carries an untruncated HMAC-SHA-256 tag,
authenticates the TTL, and rejects legacy/unknown packet versions. Forwarders
must re-sign a changed TTL with a configured group key; otherwise an
authenticated packet is not mutated and relayed under the original identity.

### CI-007 — No replay/freshness check on authenticated mesh packets

`WisdomPacket` carries a monotonic `sequence` field and the version-2 tag
authenticates it (it is inside the HMAC input), but nothing on the receive
path rejects a previously-seen, still-validly-tagged packet. The mesh's
existing "dedup" counters (`packets_deduplicated`, `packets_replayed`) are
about gossip-loop suppression and late-joiner catch-up, not adversarial replay
rejection — a captured valid packet can be re-sent later and will verify.

**Security notion violated:** freshness / replay resistance, distinct from
the authenticity property HMAC alone provides.  
**Severity:** low-to-moderate; a replayed wisdom vector is stale data, not a
forgery, and no severity-escalating consumer of stale mesh data was found.  
**Containment:** none yet — this is a protocol-logic gap, not a cryptographic
one. A fix needs a per-source sliding window or last-seen-sequence table,
which is a small stateful feature, not a MAC change. Documented here as a
follow-up rather than blocking WP−1, since HMAC-SHA-256 itself is not at
fault.

## Consumer inventory

| Consumer | Classification | Security boundary | Required action |
|---|---|---:|---|
| `symthaea-hdc-crypto` | exported compatibility crate | potential | keep quarantined or remove from publication |
| `symthaea-core::hdc::{hdc_crypto,hdc_fhe,hdc_treasury}` | duplicated implementation | potential | consolidate, default-disable, then remove/replace |
| `src/swarm/mesh::WisdomPacket` authentication | callable network API | yes | migrated to version-2 HMAC-SHA-256; legacy HDC helpers require explicit `insecure-experimental-crypto` opt-in |
| `src/cognitive_loop/managers/swarm_manager.rs` under `fhe-wisdom` | feature-gated network workflow | yes when enabled | disable for privacy claims; retain only under explicitly insecure experiment naming |
| `symthaea-soma` sensor → Holon key path | future transport path | latent | establish an independent secret; bind context through a standard KDF |
| `hdc_treasury` | tests/internal module only | no current external caller found | prohibit financial/security use; replace before any integration |
| `symthaea-psych-bench` security benchmarks | research benchmark | no | rename domain/results so they measure algebra, not FHE/security |
| `examples/fhe_collective_intelligence_demo.rs` | demonstration | no | rename and rewrite claims |
| GIS design documents | speculative documentation | no | remove privacy/security claims tied to these types |

The machine-readable companion is `crypto-claim-inventory.json` in this
directory.

## Follow-up review pass (2026-07-14)

An independent reviewer examined the containment work above and found the
quarantine itself was incomplete in several precise ways — not new insecure
claims, but places where the write-up or the code still overstated precision,
left a discoverable unmarked API, or missed a genuine secondary failure mode.
All were verified against the actual code/math before fixing (one, the
even/odd bundling claim, was confirmed by hand-deriving a concrete
counterexample; the Hoeffding-bound and sigma figures were independently
recomputed). Applied to both duplicated implementations
(`symthaea-hdc-crypto` and `symthaea-core::hdc::{hdc_crypto,hdc_fhe,binary_hv}`)
unless noted:

- **Quarantine boundary was permeable**: `EncryptedHV` and its methods
  (`encrypt`/`decrypt`/`hom_bind`/`encrypted_similarity`) had no
  `#[deprecated]` attribute, unlike every other type in these modules — a
  consumer could discover it and reasonably assume it was a legitimate,
  unflagged encryption primitive. Fixed: deprecated with an explicit note
  that it provides no authentication/integrity/misuse-resistance.
  Additionally, `symthaea-hdc-crypto`'s crate-root re-exports of all
  MAC/threshold/commitment/FHE types now require an
  `insecure-experimental-crypto` feature (default off) — the underlying
  `crypto`/`fhe` modules and their tests are unaffected, only the
  no-warning root-level import path is gated. No other workspace crate
  imports these types via the root path, so this is a zero-impact change.
- **Bundling "approximate" claim was imprecise**: shared-mask majority-vote
  bundling is **exact** for an odd contributor count (ties are structurally
  impossible) and only **approximate** for an even count, where a tied local
  tally resolves to 0 in both plaintext and ciphertext, but decrypting the
  ciphertext tally can flip that 0 to 1 if the mask bit is 1 — disagreeing
  with the plaintext bundle. Verified by hand for a 2-contributor case and
  captured as `legacy_attack_even_count_bundle_disagrees_with_plaintext`. The
  existing 5-contributor aggregate test (odd, hence exact) was loosened to a
  similarity threshold >0.85 for no reason — tightened to `assert_eq!`.
- **`compute_with_offset`'s "domain separation" claim was false**: since
  `permute` is a cyclic shift, one known (message, tag) pair at any offset
  lets an attacker derive the key's permutation at *every* other offset by a
  further rotation, forging tags across "domains." Removed the claim, added
  `legacy_attack_cross_offset_forgery` demonstrating the cross-offset forge
  exactly (replacing a prior test that only checked two offsets produce
  different tags — true, but not evidence of domain separation).
- **`HdcContextKey` has a second, independent failure**: beyond low entropy
  (CI-005), the pre-hash derivation is linearly malleable. Because
  XOR/permute is linear, `sensors[0] -> sensors[0] XOR delta` and
  `sensors[1] -> sensors[1] XOR permute(delta, -1)` cancel exactly, so two
  different sensor tuples derive an identical fingerprint — hashing cannot
  recover a distinction already destroyed upstream. Added
  `legacy_attack_context_key_collision` and documented the mechanism.
- **Statistical claims were wrong, not just imprecise**: `sim(a,b) ~ 0.5 +/-
  0.0039` was labeled "(3 sigma)" but 0.0039 is 1 sigma (sigma = 1/(2*sqrt(D)));
  3 sigma is ~0.0117. Separately, the noisy-MAC false-positive rate at
  threshold 0.95 was stated as `2^-4700`; recomputing the stated Hoeffding
  bound (`exp(-2*D*(tau-0.5)^2)` at D=16384, tau=0.95) gives ~6635.5 nats,
  which converts to ~2^-9573, not 2^-4700 — the original figure did not
  convert the exponential bound to base 2. Both fixed with the derivation
  shown inline so the numbers are checkable, not just asserted.
- **`permute()` used native-endian byte conversion** (`from_ne_bytes`/
  `to_ne_bytes`) in both the standalone crate's `BinaryHV` and, more
  significantly, in **`symthaea-core`'s production `BinaryHV`** used
  throughout the entire Symthaea codebase (not just this quarantine). Two
  hosts permuting identical input bytes by the same shift were not
  guaranteed to get the same output bytes on a big-endian machine. Fixed to
  explicit `from_le_bytes`/`to_le_bytes` in both crates; this is a no-op on
  every currently deployed target (x86_64 and aarch64 are both
  little-endian), so no observed behavior changed today — it only pins the
  specification for portability. Added known-answer permutation vectors
  (independently computed in Python) to both crates to catch a future
  regression back to native-endian conversion.
- **`BinaryHV` cannot safely hold secrets**: it is `Copy`, exposes its bytes
  publicly, and its `Debug` impl prints a byte prefix, with no zeroization on
  drop or move. Documented this directly on the type rather than
  introducing a new non-`Copy` zeroizing secret wrapper — the latter would be
  real architecture work with no current consumer to justify it, since
  nothing here should be handling real secrets in the first place. If real
  secret material is ever needed, a dedicated wrapper is the right answer,
  not `BinaryHV`.
- **`CollectiveWisdomPool` had unlisted failed invariants**: no check that
  contributions share a mask/round, duplicate `contributor_ids` accepted (one
  participant can weight the vote multiple times), no authentication or
  provenance, and `with_capacity(n)` does not actually cap `max_size` at the
  documented 256 default (only the internal `Vec`'s pre-allocation is
  capped). Documented as known failed invariants on the type rather than
  "fixed" — enforcing them would make a deliberately-broken toy model look
  more trustworthy than it is, which cuts against the quarantine's purpose.
- **Two tests were misleadingly named**: `test_otp_ciphertext_hides_plaintext`
  (claims "hiding" from one similarity sample under a still-guessable
  deterministic mask — precisely the false claim this whole pass exists to
  retract) renamed to `deterministic_xor_mask_decorrelates_one_sample`.
  `test_hdc_mac_domain_separation` was replaced outright (see above) rather
  than merely renamed, since its content was also wrong, not just its name.

Verified: `cargo test -p symthaea-hdc-crypto` and the equivalent
`symthaea-core --lib` tests pass with all new attack/KAT tests included (see
"Validation performed" below for the exact commands and results).

## Self-review checklist (substitute for a paid protocol audit)

No budget exists for a professional cryptographic review of the version-2
mesh protocol. HMAC-SHA-256 itself needs no re-review — it is one of the most
analyzed MAC constructions in use. The only thing that could actually be
wrong here is *how it was applied*, which is a much cheaper thing to check
than "is this a good cipher":

- [x] All security-relevant fields are inside the tag input, including the
  fields that change on forwarding (TTL). Verified in `compute_packet_mac`/
  `verify_packet_mac` (`src/swarm/mesh/mod.rs`).
- [x] No downgrade path: `from_bytes`/`verify_packet_mac` reject any wire
  version other than the current one, and reject any length other than
  exactly one version-2 packet. Verified by
  `wisdom_packet_rejects_legacy_or_unknown_wire_version` and
  `test_packet_mac_rejects_wrong_length_or_wire_version`.
- [x] Constant-time comparison: verification uses `hmac::Mac::verify_slice`,
  which is constant-time by construction (not a manual `==` on the tag).
- [ ] Replay/freshness: **gap found**, see CI-007. Not a MAC weakness — a
  missing protocol-level check.
- [ ] Key establishment/rotation for `mesh_auth_key`: explicitly out of scope
  for this pass; whatever distributes that key is a separate design that
  still needs its own review whenever it is built.

This checklist is not a substitute for independent review if a real
deployment and threat model ever exist. It is a proportionate response to
"no users, no budget, and the primitive itself is standard" — see the
decision recorded in `advisory-draft-v2.0.0.md`.

## Replacement decisions

| Need | Decision for normal code | Status |
|---|---|---|
| Mesh message authentication | Untruncated HMAC-SHA-256 with a 256-bit independently established key | implemented; external protocol review pending |
| Threshold secret sharing | No replacement until a real consumer and threat model exist; then use an independently reviewed Shamir/VSS implementation | insecure implementation quarantined |
| Commitments | No bespoke HDC construction; use a reviewed commitment protocol with explicit domain separation and fresh random opening material | no current production consumer |
| Confidential aggregation/FHE | No replacement by analogy; select a reviewed secure-aggregation or HE protocol only for a concrete requirement | shared-mask demo quarantined |
| Sensor-bound derivation | HKDF-SHA-256 from independent secret input; sensor fingerprint may appear only as public context/info | automatic key installation removed |
| Treasury confidentiality | Prohibited until conventional authenticated encryption and an independently reviewed threshold-control design are specified | no current external consumer found |

These are architecture decisions, not independent cryptographic review. The
external-review gate remains open.

## Terminology remediation

Quarantine includes the source-level API documentation, not only deprecation
attributes. Both duplicated implementations now describe the historical types
as compatibility transforms and state their executable attacks. Old claims of
an HDC-MAC security proof, a secure threshold-sharing entry point,
privacy-preserving nearest-neighbor search, threshold-controlled treasury
decryption, and sensor-derived secret keys were removed. The original type
names remain only where changing them would break compatibility; warnings and
explicit insecure labels make that debt visible.

## Released-version disclosure assessment

Repository tag inspection found the affected HDC constructions, shared-mask
benchmarks, forgeable HDC packet helpers, and eight-bit mesh tag in `v2.0.0`.
The same searches found none of those symbols in `v1.9.0`.

**Decision (2026-07-13): no known users exist for this repository or any
release built from it.** Given that, formal coordinated disclosure (CVE/GHSA,
private outreach, staged advisory publication) is disproportionate — it
exists to inform third parties, and there are none to inform. The advisory
draft is kept as a historical record and template, marked not-for-publication,
rather than executed. See `docs/security/advisory-draft-v2.0.0.md` for the
full decision record and the checklist to use if a user is ever reported.

This is a point-in-time judgment, not a permanent exemption: if a user of any
affected release is ever discovered, re-open that document, reassess
severity against their actual reliance, and reconsider CVE/GHSA at that time.

## Evidence-level vocabulary

The following labels are distinct and must not be promoted implicitly:

1. **Observed on samples** — agrees with sampled points.
2. **Cross-validated** — agrees with data held out from fitting.
3. **Exhaustively checked** — evaluated over every member of a stated finite domain.
4. **SMT bounded/sample checked** — solver established a fixed-width or explicitly
   bounded statement; the formula and bounds must be recorded.
5. **Symbolically proved** — a universal symbolic argument was checked under
   stated assumptions.
6. **Proof-assistant verified** — a pinned theorem was checked by a named proof
   assistant, with axiom provenance recorded.

Passing unit/property tests establishes implementation behavior only. It does
not establish confidentiality, unforgeability, hiding, binding, threshold
privacy, or general cryptographic security.

## Validation performed

The following checks were run during this pass:

- `cargo check -p symthaea-core` — passed.
- `cargo test -p symthaea-core --lib legacy_attack` — 4 attack regressions
  passed (forgery, one-share recovery, alternative commitment opening, and
  shared-mask relation recovery).
- `cargo test -p symthaea-soma legacy_attack_enumerates_quantized_sensor_context`
  — passed.
- `cargo check -p symthaea-soma` — passed.
- `cargo check -p symthaea-psych-bench` — passed.
- `cargo check -p symthaea --features mesh` — passed with the version-2
  HMAC-SHA-256 packet layout.
- `cargo check -p symthaea --tests --features mesh` — passed, including all
  mesh-enabled test targets.
- `cargo test -p symthaea --features mesh test_packet_mac -- --nocapture` — all
  8 packet-authentication tests passed, including the independently generated
  HMAC known-answer fixture, first-byte-only regression, tampering, wrong-key,
  wrong-length/version, serialization, and fragmentation cases.
- `cargo test -p symthaea-hdc-crypto` — all 58 unit tests passed, including
  the attack regressions; the crate's three pre-existing doctests then failed
  because the doctest linker could not locate `blake3` in rlib form.

### Follow-up review pass validation (2026-07-14)

- `cargo test -p symthaea-hdc-crypto` — **61/61 unit tests pass** (58 prior +
  3 new: `legacy_attack_cross_offset_forgery`,
  `legacy_attack_context_key_collision`,
  `legacy_attack_even_count_bundle_disagrees_with_plaintext`, plus
  `test_permute_known_answer_vectors` and the renamed/tightened tests) and
  **3/3 doctests pass** — the earlier blake3-rlib doctest-linker failure did
  not recur on this build.
- `cargo test -p symthaea-core --lib hdc_crypto` — **36/36 pass**, including
  the cross-offset forgery and context-key collision attacks.
- `cargo test -p symthaea-core --lib hdc_fhe` — **14/14 pass**, including the
  even-count bundling counterexample, confirmed against the real production
  SIMD `bundle()` path (not a separate toy implementation).
- `cargo test -p symthaea-core --lib test_permute_known_answer` — **1/1
  pass**, confirming the little-endian fix in the production
  `hdc::binary_hv::BinaryHV::permute()` (used throughout the whole Symthaea
  codebase, not just this quarantine) against independently-computed KAT
  vectors.

Deprecation warnings from quarantined APIs are expected and intentional. They
make remaining consumers visible until replacement or deletion.

## Exit gates for WP−1

- [x] Initial API and consumer inventory.
- [x] Confirmed constructions labelled insecure and deprecated.
- [x] Sensor-derived value removed from automatic session-key installation.
- [x] Attack demonstrations cover both duplicated implementations, plus sensor
  context enumeration in `symthaea-soma`.
- [x] `fhe-wisdom` removed from network-oriented feature profiles and labelled
  as a quarantined legacy experiment.
- [x] Mesh authentication migrated from an eight-bit tag to full
  HMAC-SHA-256; forgeable HDC helpers are hard-gated behind an explicit
  insecure feature.
- [x] Psych-bench result namespaces and the historical FHE example relabelled
  as insecure shared-mask algebra demonstrations.
- [x] Conventional replacement decisions documented.
- [x] Mesh protocol reviewed via self-review checklist in lieu of a paid
  audit (no users, no budget — see "Self-review checklist" above); real gap
  found and documented (CI-007, replay/freshness). Any future
  threshold/commitment/aggregation construction still needs independent
  review before real use.
- [x] Verification statuses distinguish cross-validation, bounded data checks,
  fixed-input SMT checks, symbolic checks, and actual formal verification.
- [x] Released-version exposure assessed locally (`v2.0.0` affected;
  repository evidence for `v1.9.0` not found).
- [x] Maintainer/downstream coordination and advisory publication: **not
  applicable — no known users.** Decision and reopen-checklist recorded in
  `advisory-draft-v2.0.0.md`.

WP−1 internal containment is complete. Cryptographic component-discovery work
must not resume before a fresh assessment if any of the following becomes
true: a user of an affected API is reported, or a real consumer/threat model
emerges for threshold sharing, commitments, or confidential aggregation.
