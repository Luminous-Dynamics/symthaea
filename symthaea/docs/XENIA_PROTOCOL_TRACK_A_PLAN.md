# Xenia Protocol — Track A Plan (Library + Paper)

**Name approved 2026-04-17.** Xenia (ξενία) is the ancient Greek covenant
between guest and host — the moral logic of remote access made
cryptographic. A technician is a *guest* in a client's machine; the
client extends bounded hospitality; the protocol codifies the terms.
Violating xenia was, to the Greeks, one of the gravest sins. The
consent ceremony, sealed-replay recording, and attestation log are
all expressions of xenia. Pronunciation: `KSEN-ya` (classical) or
`ZEN-ya` (anglicized) — the paper introduces the term in one sentence.

**Naming boundary**: "Soma" stays as the internal Symthaea substrate
name (`SomaRdpServer`, `HOLON_SOMA_ROADMAP.md` research roadmap).
"Xenia" is the external public protocol name — this plan, the crate,
the paper, the spec. Same bytes on the wire; only the naming boundary
differs.

**Prior-art note**: a popular Xbox 360 emulator project uses
`xenia-project`. It's an executable, not a library — crates.io
namespace overlap is manageable. If the team wants extra clearance,
`xenia-wire` (library) + `xenia-protocol` (umbrella) stay clear of the
emulator's brand space.

**Scope**: 4-6 weeks of focused work to extract the Holon-Soma wire
from symthaea into a standalone open-source crate, write a protocol
specification, publish a paper, and seed initial adoption. Low-floor,
high-ceiling — even total failure leaves behind a research substrate
for Phase III/IV.

**Status**: planning (2026-04-17, draft 2 with Xenia name approved +
four frontier features added). No implementation until this doc is
approved.

**Relationship to existing roadmap**: `HOLON_SOMA_ROADMAP.md` is the
research roadmap (Phase I.A through V, all Symthaea-internal). This
plan is the **commercial/academic off-ramp**: the same wire as a
general-purpose protocol, extracted and published under the Xenia
name. The two reinforce each other — paper credibility funds grant
applications; grant funding keeps the research going; research
numbers are the paper's empirical section.

---

## One-paragraph summary

Extract `symthaea/src/swarm/rdp_wire.rs`, `rdp_session.rs`,
`replay_window.rs`, and the minimal `rdp_protocol.rs` types needed to
use them, into a standalone `xenia-wire` crate. Publish to crates.io
under Apache-2.0 OR MIT dual. Write a protocol specification with
reference test vectors. Write a short (10-15 page) academic paper
documenting the design rationale, empirical numbers from Phase I.A
through Phase II.A, and the decentralized-identity coupling angle.
Ship a WASM browser viewer so reviewers can see the protocol work
without building a native client. Add three spec-level features that
must be in v1 (consent ceremony, sealed-replay recording, attestation
log) so the initial release establishes the differentiators, not just
the commoditized parts. Submit the paper to a security venue; post the
release on HN / lobste.rs / r/rust / r/sysadmin; email ~20 MSP-focused
people with the protocol link. Total budget: 4-6 focused weeks.

---

## Scope gate

### In Track A
- Extracting the wire into a standalone crate (`xenia-wire`).
- Protocol specification document (`SPEC.md`).
- Academic paper (10-15 pages).
- WASM browser viewer (minimum-viable verifier).
- Three v1 spec features: consent ceremony, sealed-replay recording,
  attestation-chained action log.
- Publishing to crates.io.
- Release announcement + blog post + consulting outreach list.

### Explicitly out of Track A (belongs in Track B/C or beyond)
- Windows / macOS / iOS capture agents.
- MSP tenant architecture on Holochain.
- Web dashboard or admin UI.
- Ticketing integration (Zammad, itop, etc.).
- Billing / subscription management.
- Commercial license tier.
- Sales / distribution channels.
- Natural-language remediation, mesh fallback, cross-MSP federation —
  these are v2+ spec extensions.
- Real ML-KEM handshake — Track 2.5 in the research roadmap. Track A
  ships the current placeholder key setup, with the handshake as a
  clearly-flagged open problem in the spec.

### Explicitly deferred decisions
- Whether to pursue Track B/C at all. That decision is gated on Track
  A outcomes (paper acceptance, crate adoption, inbound consulting
  inquiries). No commitment to product-company-scale work until then.
- Which venue to submit the paper to. Candidates listed in §Paper
  section; decision waits on timing of the actual CFPs.

---

## Week-by-week milestones

Each week caps at ~30 focused hours. If a week slips, later weeks
compress or move to Week 7 (buffer).

### Week 1 — Crate extraction

**Deliverables**:
- `xenia-wire/` at repo root, published to crates.io as `xenia-wire
  0.1.0-alpha.1` (or similar pre-release).
- `Cargo.toml` with minimal deps: `chacha20poly1305`, `zeroize`,
  `bincode`, `serde`, `lz4_flex` (optional), `thiserror`. No `tokio`,
  no `axum`, no `quinn`. Wire is transport-agnostic.
- Core types re-exported: `RdpSession`, `ReplayWindow`, `WireError`,
  `PAYLOAD_TYPE_RDP_FRAME`, `PAYLOAD_TYPE_RDP_INPUT`,
  `PAYLOAD_TYPE_RDP_FRAME_LZ4`.
- `seal_frame` / `open_frame` / `seal_input` / `open_input` /
  `seal_frame_lz4` / `open_frame_lz4` all exported.
- Generic `Sealable` trait so consumers can bring their own frame
  types; `RdpFrame` becomes an optional reference implementation
  behind a feature flag.
- CI via the standalone repo's own workflow: fmt + clippy + test + doc
  build on every push.
- README with 30-line Quick Start + link to `SPEC.md`.
- `LICENSE-APACHE` and `LICENSE-MIT`.
- `CHANGELOG.md` with the 0.1.0-alpha.1 entry.

**Exit criterion**: `cargo add xenia-wire` in a fresh crate compiles +
runs a working seal-open-roundtrip example. A contributor with no
symthaea context can use the crate.

**Risks**: `RdpSession`'s AEAD may have transitive symthaea deps that
aren't obvious. Budget 20% of the week for surprises.

### Week 2 — Specification document

**Deliverables**:
- `SPEC.md` in the standalone repo, version-stamped.
- Sections:
  1. Introduction + non-goals.
  2. Wire format (envelope byte layout).
  3. Nonce construction (source_id + payload_type + epoch + sequence).
  4. Payload-type registry (0x10–0x13 assigned + reserved ranges).
  5. Replay window semantics (64-slot sliding window per
     `(source_id, payload_type)`).
  6. Epoch rotation + key lifecycle (current-key / previous-key
     grace period).
  7. LZ4-before-AEAD rule (with the measurement citation).
  8. Handshake (current placeholder; real ML-KEM deferred — Track 2.5).
  9. Error taxonomy (each `WireError` variant, when it fires,
     recommended caller response).
  10. Security properties (confidentiality, integrity, replay
      resistance, forward secrecy via epoch rotation, domain
      separation via payload_type).
  11. Non-goals (not a TLS replacement, not an MSP workflow, not a
      general AEAD library).
- `test-vectors/` directory with hex-encoded input/output pairs so
  other implementations can validate interop.
- Spec version is `draft-01`. Future spec changes bump `draft-02`,
  etc. until stabilization.

**Exit criterion**: the spec is specific enough that an independent
implementer could write an interoperable client in a different
language (e.g., Go, Swift) using only `SPEC.md` and the test vectors.
No reading of symthaea source required.

**Risks**: test vectors are fiddly. Budget half a day for generating +
verifying them against the Rust reference.

### Week 3 — Paper draft

**Deliverables**:
- `papers/xenia-paper.md` (or LaTeX in the repo), 10-15 pages.
- Structure:
  1. Abstract (~200 words).
  2. Introduction: why remote-control protocols matter; why the
     ConnectWise Feb-2024 CVE-2024-1709 breach motivates a
     decentralized-trust topology; why post-quantum matters.
  3. Related work: RDP, VNC, Apache Guacamole, ScreenConnect's
     architecture (public info), QUIC, Signal's ratchet, Holochain.
  4. Protocol (recap of the spec, with design rationale callouts).
  5. Empirical evaluation:
     - Phase I.A: 3.27-3.52× bandwidth reduction vs JSON baseline on
       real Pixel 8 Pro hardware (already measured).
     - Phase I.C: WS-vs-QUIC head-of-line blocking comparison (table
       from commit `e019c03e62` — WS tail inflates 4.7× at 1% loss,
       QUIC stays ≤2×).
     - Phase II.A: LZ4-before-seal 2.12× ratio on live capture (commit
       `b5c685b37a`); 30 fps budget analysis (commit `2986545136`);
       discussion of why compression must precede AEAD.
     - Phase I.B: mobile-first support via scrcpy + Tensor G3 HEVC
       (canonical sustain run: 16 fps mean / 23 fps peak on
       single-CPU decode).
  6. Discussion: comparison against ConnectWise, TeamViewer, AnyDesk
     on five axes (post-quantum readiness, centralized vs
     decentralized trust, mobile-first support, compression under
     loss, open vs proprietary).
  7. Future work:
     - Real ML-KEM handshake (Track 2.5 in the research roadmap).
     - Consciousness-gated session oversight (a second paper, when
       Symthaea side-car integration is done).
     - WAN split-cognition experiments (Phase IV Markov blanket).
  8. Conclusion.
- BibTeX file with all references.
- At least one figure comparing the HoL-blocking latency distributions
  (box plot or CDF). Produced from the netem data already on main.
- At least one architecture diagram (ASCII or tikz).

**Exit criterion**: the draft is sendable to a friendly reviewer (a
security researcher, not a security user) for comments. Expect 1-2
revision cycles before submission.

**Target venues** (candidates, decide at end of Week 3 based on CFP
timing):
- **USENIX Security 2027** — top-tier, deadlines typically Aug/Feb.
- **NDSS 2027** — top-tier, deadlines typically Apr/Jul.
- **ACM CCS 2027** — top-tier, deadline typically May.
- **IEEE S&P 2027** — top-tier, rolling.
- **EuroSys / ACM SOSP** if we lean systems over security.
- **Arxiv + conference-specific workshop** as a fallback (arxiv is
  always available; workshops are easier to get into if the main
  track misses).
- **Rustconf 2027 talk proposal** as a secondary distribution path.

### Week 4 — WASM browser viewer

**Deliverables**:
- `xenia-viewer-web/` subcrate with `wasm-bindgen` + `web-sys`.
- HTML/JS shell that:
  - Connects to a Xenia server via WebSocket or WebTransport.
  - Displays received frames via canvas.
  - Sends mouse/keyboard events as sealed `InputFrame`s.
  - For the demo, session-key bootstrap is via PAKE or QR code —
    NOT a real ML-KEM handshake.
- Deployed to a static site (GitHub Pages, Cloudflare Pages, or
  `luminousdynamics.io/xenia`) so paper reviewers can click a link,
  scan a QR, and see the protocol working.
- Live demo harness: a small server (Docker image) MSPs can run with
  `docker run -p 7778:7778 luminousdynamics/xenia-demo` and connect
  to from the web viewer.

**Exit criterion**: a reader of the paper clicks a URL, scans a QR
code from the live demo server, and sees a working sealed remote
session render in their browser. End-to-end without native install.

**Risks**: WASM + AEAD + rendering is the highest-risk week.
`chacha20poly1305` works fine on wasm32 targets, but bincode's
serde-derive-at-runtime may need work. Budget slack.

### Week 5 — v1 spec differentiators

Three features that must be in v1 so the initial release establishes
the differentiators, not just the commoditized parts. Without these
the crate is just "PQC remote desktop," which is a crowded market.
With them it's "decentralized-trust remote control with
cryptographically verifiable sessions," which is unclaimed.

**5a. Consent ceremony** (~2 days):
- New payload types:
  - `0x20` `ConsentRequest` — tech asks end user to approve session.
    Contains tech identity fingerprint, claimed reason (free-text
    ticket reference), requested scope (screen / keyboard / files /
    shell), time limit, **`causal_binding: Option<CausalPredicate>`
    (always `None` in v1 — reserved for v1.1 Ricardian extension)**,
    **MSP attestation chain (see 5d-3 below)**.
  - `0x21` `ConsentResponse` — end user signs approval or denial
    with their device key.
  - **`0x22` `ConsentRevocation` — end user terminates session
    asymmetrically, mid-stream. Recorded as a terminal event in
    the attestation chain. See 5d-2 below.**
- Session establishment requires a valid `ConsentResponse` before any
  `RdpFrame` is accepted. Server-side enforcement: if missing,
  `session.open()` on an `RdpFrame` returns `WireError::NoConsent`.
  After a valid `ConsentRevocation`, subsequent frames return
  `WireError::ConsentRevoked`.
- Spec section documents the flow + threat model (what MITM looks
  like, why device key signing prevents it).
- Test vectors for consent request + response + revocation.

**5b. Sealed-replay recording** (~1 day):
- New file format `.xenia-session` — a simple container: metadata
  header (session parties, start time, protocol version, signatures)
  followed by the sealed envelopes in arrival order, each with a
  length prefix.
- `XeniaReplay` API: `open(path)`, `next_frame() -> Option<(ts,
  RdpFrame)>`, `seek(ts)`.
- Spec section describes the format. Security property: tamper
  evidence — any modification breaks the AEAD chain.
- Reference CLI: `xenia-replay <file.xenia-session>` plays back into
  the viewer.

**5c. Attestation-chained action log** (~2 days):
- Every command/input the tech issues is signed by the tech's device
  key and logged with monotonic sequence number + hash of prior log
  entry (blockchain-of-one-tech).
- New payload type `0x23` `AttestedAction`.
- End-user client can verify the chain retroactively to prove no
  tamper.
- Spec section documents it.

**5d. Design considerations locked in before implementation**

Three decisions pre-made here so Week 5's implementer isn't deriving
them under time pressure. Each is cheap to make now, painful to
retrofit once the 0.1 wire format ships.

**5d-1. Causal-binding forward compatibility**. The v1
`ConsentRequest` payload MUST include a `causal_binding:
Option<CausalPredicate>` field, always set to `None` in v1. This
reserves the wire-format slot for the v1.1 Ricardian ticket-state
binding extension. Receivers that don't understand the predicate
(old clients) still parse the message correctly; v1.1-aware
receivers honor it. Zero-cost optionality in v1; breaking-change
avoided in v1.1.

**5d-2. ConsentRevocation is v1, not later**. Reserve payload type
`0x22` for revocation shipped FROM end user TO server. Semantics:
arriving revocation immediately terminates the session; all
subsequent `RdpFrame`s from the tech return
`WireError::ConsentRevoked`; the revocation is recorded in the
attestation chain as a terminal event (monotonic sequence +
hash-chain entry identical to an `AttestedAction` but with a
distinct payload_type so auditors can grep for it). Defining it now
means the revocation format is stable from day 1; retrofitting
later creates a "some revocations verified, some not" mess. One
payload type + one sentence in the spec; half-day of
implementation.

**5d-3. Tech-credential attestation chain** — the "attested by your
MSP" phrase in 5a is made concrete: MSP runs a Holochain agent;
MSP's agent key signs the tech's device public key (canonical DID
format `did:key:...` or `did:holo:...`); the signature is stored in
a Holochain directory addressable by the MSP's public key; the
`ConsentRequest.msp_attestation` field carries the (tech_key,
msp_key, signature, optional expiry) tuple. End user's client
queries the Holochain network to verify the signature matches the
MSP's published agent key. This makes the decentralized-trust
claim concrete rather than hand-wavy — and it matches the Holochain
architecture Luminous Dynamics is already building. If the
Holochain directory is unreachable at consent time, the client can
fall back to a cached signature with a user-visible "offline
verification" indicator (threat-model decision: prefer soft-fail
over hard-fail for availability, with explicit UX to communicate
the downgrade).

**Exit criterion**: each of 5a/5b/5c is spec'd, implemented with
tests, and the spec cross-references them as v1-required features.
5d-1/5d-2/5d-3 are documented as design decisions in the spec
rationale section, not optional appendices — they're load-bearing
for v1.1 and beyond.

**Why in Track A** (not deferred): retrofitting security-critical
protocol features after v1 is painful and often impossible. Ship them
now so the 0.1.0 release is the real thing.

### Week 6 — Polish + release + outreach

**Deliverables**:
- Bump `xenia-wire` to `0.1.0` (stable).
- Release binary for `xenia-viewer-web` at the deploy URL.
- Paper submitted (or arxiv + preprint blog post if CFPs don't align).
- `BLOG_POST_1.md`: "Announcing Xenia — a PQC-sealed remote-control
  protocol you can actually audit." Target: HN, lobste.rs, r/rust.
- `BLOG_POST_2.md`: "Why ConnectWise's architecture is a single
  point of failure." Target: r/sysadmin, r/msp, LinkedIn. Anchor
  the decentralized-trust pitch.
- Consulting outreach: email ~20 MSP-focused people (independent
  consultants, small MSP owners, MSP-tooling founders) with a short
  "here's the crate, here's the paper, here's the demo URL, here's my
  calendar" note. Not a sales pitch, an invitation to evaluate.
- `FOLLOW_UPS.md` in the repo: a running list of concrete next steps
  for Track B/C prioritized by signals received during Track A.

**Exit criterion**: the release is announced, the paper is under
review or on arxiv, three MSP-adjacent people have read the paper or
tried the demo.

---

## Crate structure (Week 1 target)

```
xenia-wire/                              (published crate, version 0.1.0)
├── Cargo.toml                          (no tokio, no axum, no quinn)
├── README.md
├── CHANGELOG.md
├── LICENSE-APACHE
├── LICENSE-MIT
├── SPEC.md                             (draft-01 at Week 2, v1 at Week 6)
├── src/
│   ├── lib.rs                          (re-exports + crate docs)
│   ├── session.rs                      (from rdp_session.rs; renamed `RdpSession` → `Session`)
│   ├── replay_window.rs                (as-is)
│   ├── wire.rs                         (from rdp_wire.rs — sealing functions)
│   ├── payload_types.rs                (const registry: 0x10-0x22)
│   ├── error.rs                        (WireError taxonomy)
│   ├── frame.rs                        (`Sealable` trait; optional reference RdpFrame behind `reference-frame` feature)
│   ├── consent.rs                      (Week 5a)
│   ├── replay_recorder.rs              (Week 5b)
│   └── attestation.rs                  (Week 5c)
├── test-vectors/                       (Week 2 target)
│   ├── seal-frame-sample-1.hex
│   ├── open-frame-sample-1.hex
│   ├── consent-request.hex
│   └── ...
├── examples/
│   ├── hello_xenia.rs                  (30-line quick-start)
│   ├── replay_session.rs
│   └── attest_action.rs
├── benches/
│   └── seal_open_throughput.rs
└── tests/
    ├── integration_roundtrip.rs
    ├── integration_consent.rs
    └── integration_replay.rs
```

`xenia-viewer-web/` is a separate crate in the same workspace —
optional, only built with the `wasm` feature set.

---

## Success criteria

**Green (unambiguous hit)** — any two of:
- Paper accepted at a Tier-1 security venue.
- `xenia-wire` accumulates ≥100 GitHub stars in the first month or
  ≥1000 crates.io downloads in the first quarter.
- Three or more inbound consulting inquiries from MSP-world people
  within the first 60 days of launch.
- One of the existing OSS MSP tools (Tactical RMM, MeshCentral,
  RustDesk) expresses interest in adopting the wire or the
  consent-ceremony spec.

**Yellow (noble partial)** — any one of:
- Paper on arxiv + cited by one other paper in the following 12
  months.
- A single-digit number of real users who aren't us.
- The research program (Phase III/IV) benefits from the extracted
  crate being cleaner than the in-tree version.

**Red (nothing stuck)** — none of the above. In this case Track A
still leaves behind: a cleaner extracted crate, a documented spec, a
paper on arxiv, and the infrastructure for Phase III/IV. Not zero
value; just commercially null.

---

## Failure modes and mitigations

| Failure | Likelihood | Mitigation |
|---|---|---|
| Crate extraction reveals hidden symthaea dependencies | Medium | Budget 20% of Week 1 for surprises. Fall back: vendor the specific types rather than re-deriving. |
| WASM AEAD has perf / compat issues | Medium | Week 4 is the highest-risk week. Fallback: ship a native-only viewer and mark the WASM viewer as experimental. |
| Paper rejected at top venue | High | Plan B = arxiv + a systems-conf workshop. Plan C = it's still a real document on our site. |
| No crate adoption | High | Track A's value doesn't depend on adoption (research substrate + paper + consulting material). Adoption is upside. |
| MSP outreach returns zero replies | Medium | The 20-person list is low-cost. If zero, try a second round after the paper lands (more credibility). |
| Scope creep — we try to ship Track B features in Track A | High | **This is the real risk.** The scope gate is the discipline. Every decision "should this go in 0.1?" defaults to NO unless it's in the 3 v1 differentiators (5a/5b/5c). |
| Phase III/IV research eats Track A's schedule | Medium | Track A is 4-6 weeks of focused work. Phase III pre-registration is already frozen; Phase IV is weeks of code. They shouldn't overlap in the same week. If they do, Track A slips; don't sacrifice the research. |

---

## Prerequisites before Week 1 starts

- [x] **Name approved 2026-04-17: Xenia.** Crate: `xenia-wire` (core
      byte protocol) + `xenia-viewer-web` (WASM demo viewer). Future
      higher-level crates TBD (`xenia-agent`, `xenia-tenant`).
- [x] **Repo location: `luminousdynamics/xenia-wire` on GitHub,
      public from day 1 with a conspicuous `PRE-ALPHA — DO NOT USE
      IN PRODUCTION` banner in the README.** Rationale: private-
      until-release means no community feedback during development;
      public-from-day-1 matches the Rust ecosystem norm and lets
      feedback arrive when it's actionable.
- [x] **crates.io publishing identity: reuse the existing
      BWS-stored Luminous Dynamics token** (secret
      `736da236-a95f-4dd2-8efc-b42800c9106a` per project CLAUDE.md —
      same org that publishes `symtropy-*`, `symthaea-core`,
      `sovereign-profile`). Aligns with the GitHub org choice; no
      new token provisioning needed.
- [ ] 4-6 weeks of calendar time blocked out — Track A does not
      succeed if fragmented across 12 weeks of part-time work. **This
      is the last remaining prerequisite before Week 1 can begin.**

---

## Dependencies

**Internal** (already done):
- `symthaea/src/swarm/rdp_wire.rs` + `rdp_session.rs` +
  `replay_window.rs` (Phase I.A).
- `seal_frame_lz4` / `open_frame_lz4` (Phase II.A, commit
  `44c6b5e55f` — just landed).
- Phase I.C QUIC transport (commit `66d28825de` and follow-ups) as
  the transport-layer reference.
- Empirical numbers from Phase I.A (3.27-3.52×), I.C (HoL table),
  II.A (2.12×).

**External** (crates used):
- `chacha20poly1305` 0.10 (already a symthaea dep).
- `zeroize` 1.x (already).
- `bincode` 1.x (already).
- `lz4_flex` 0.11 (already).
- `wasm-bindgen` 0.2 (new, for Week 4).
- `web-sys` 0.3 (new, for Week 4).

**None of these are paid or restricted. The whole track costs zero
dollars in software licensing.**

---

## Relationship to the rest of the program

- **Phase III Φ-sweep** (research roadmap): the `xenia-wire` crate
  becomes the wire for the 360-trial matrix. The extracted crate is
  cleaner than the current in-tree version, which helps the research.
- **Phase IV Markov blanket**: the split-cognition experiments need
  WAN-capable transport. Xenia's QUIC + LZ4 is the right substrate.
  Track A effort directly serves Phase IV.
- **Consciousness-gated session oversight** (future paper): a second
  paper, after Symthaea side-car integration is done. Track A makes
  this paper possible by giving the wire a clean public surface to
  integrate against.
- **Grant applications**: Track A's outputs (spec, paper, demo) are
  concrete deliverables for PQC + decentralized-trust-focused
  grants (NIST PQC transition, NSF Secure Computing, etc.).
- **Consulting revenue**: the paper + crate + demo are the pitch for
  "hire me to integrate Xenia into your stack." No product-company
  overhead.

---

## What Track A does NOT commit to

- A Track B/C go/no-go decision. That's a post-Week-6 conversation.
- A business entity, trademark, or commercial licensing discussion.
- Windows/macOS/iOS agents (Track B).
- MSP tenant dashboard (Track C).
- Any named pricing, any sales funnel, any SaaS.

---

## Future spec extensions — beyond ConnectWise, beyond v1

Four proposed extensions beyond the Week-5 v1 differentiators. These
are **not** Track A work; they go in the spec as "reserved payload
type ranges + future-work section," and implementation happens in
Track B+ contingent on v1 adoption. Engineering assessment for each:

### 1. Ticket-bound authority via Ricardian contracts (**highest priority, v1.1**)

Authority tied to external causal state: "This session is valid only
while Mycelix DHT says ticket #1234 is in state `In-Progress`." When
the ticket transitions (closed, escalated, reassigned), the authority
revokes automatically. No human forgets to end the session; no
session outlives its justification.

Implementation sketch: extend the `ConsentRequest` payload (0x20)
with a `causal_binding: Option<CausalPredicate>` field. The
predicate is a Ricardian contract — human-readable prose
("Authority valid while ticket #1234 is In-Progress") plus
machine-readable condition (Mycelix DHT query + expected value).
Server-side open_frame checks the DHT binding before accepting each
frame; if the predicate becomes false, subsequent frames return
`WireError::AuthorityExpired`.

**Why this is the strongest of the four**: it composes naturally with
the v1 consent ceremony (5a) and attestation log (5c); it's a
genuinely novel contribution (ConnectWise has time-based expiry but
not state-bound expiry); implementation is bounded; it exercises the
Mycelix DHT integration Luminous Dynamics is already building. v1.1
spec extension; target 6-8 weeks post-v1 release.

### 2. Privacy masking via on-device Symthaea (**high priority, v2**)

An on-device Symthaea side-car observes the outbound frame stream and
overlays masks on tiles containing sensitive content — passwords
being typed, bank account numbers in a browser, medical records,
anything matching a consciousness-gated rule set. The tech sees the
masked regions as redacted blocks.

Implementation sketch: new side-car process runs alongside the
server, reads outbound frame bytes before AEAD seal, applies
vision-manifold + text-detection (OCR via existing Symthaea HDC
vision pipeline), masks tile regions matching a policy rule set, then
re-sealsin to payload_type 0x20. The policy is per-client (the client
decides what to mask) and per-ticket (e.g., "for this ticket, mask
all banking fields except the one referenced in the ticket
description").

**Why v2 not v1.1**: depends on Symthaea integration being stable in
the Xenia stack (currently Symthaea is internal research substrate,
not a public dep). Shipping this requires a clean `symthaea-vision`
or similar public crate. That's its own Track.

**False-negative / false-positive risks**: either is a failure mode.
FN leaks sensitive data to the tech; FP obscures legitimate work.
Need a user-override UI ("I need to see this field for the
troubleshooting" → tech requests, end user approves, request signed
and logged in the attestation chain).

### 3. Biometric Intent / Proof-of-Presence (**medium priority, v2**)

Tech maintains an ongoing proof that they (specifically they, not
someone holding their session token) are at the keyboard. Options:
- Active webcam with periodic liveness frames signed by the tech's
  device key
- Biometric heartbeat via smartwatch / paired device
- Typing-cadence match against a learned profile
- Periodic challenge-response ("Read this phrase aloud")

If the proof lapses for N seconds, session inputs are paused
(display continues but no writes). A second lapse requires re-consent
via the v1 ceremony.

Implementation sketch: new payload type `0x23 PresenceHeartbeat`,
sent from tech's device at configurable interval (default 30 s). Each
heartbeat carries a biometric claim + device-key signature. Server
rejects frames after N missed heartbeats.

**Honest caveats**:
- **Accessibility**: techs with visual impairments or motor
  disabilities need a way to opt out of specific modalities. The
  default heartbeat should be minimally demanding (e.g., typing
  cadence) not maximally (eye-tracking).
- **Privacy**: biometric data must stay on the tech's device; the
  heartbeat carries a boolean attestation, not the biometric itself.
- **Consent**: the tech's employer can't force biometric monitoring
  without the tech's explicit agreement; tech-side consent ceremony
  required in addition to the existing client-side one.
- **Fatigue**: video-based liveness over long sessions is exhausting.
  Keep modalities lightweight.

v2 target. Requires user-research with working MSP techs before it
ships in a real spec — we'd be shipping an accessibility problem if
we got the modality set wrong.

### 4. Holographic Replay Audit (**low priority, speculative**)

Auditors replay a technician's session in a 3D/VR environment using
Symtropy-Bevy as the renderer. Files the tech touched appear as
objects in space; commands they issued as arcs connecting objects;
the temporal evolution plays back as the auditor walks through.

**Honest engineering take**: this is **research-curious**, not
**product-urgent**. The core claim — that 3D representation is more
intuitive for non-technical overseers than flat session replay — is
testable but not obvious. Compliance auditors are often legal and
accounting people who don't own VR headsets. Before investing, we'd
want a 1-week prototype with 3-5 real auditors evaluating it against
the flat replay. If the usability signal is strong, build it; if it's
weak, park it.

Placement: **not a spec extension** (the replay format stays flat;
VR is a rendering layer on top). Lives as a separate companion
product `xenia-audit-studio` if it ever ships.

### Summary: recommended prioritization

| # | Feature | Priority | Target | Risk |
|---|---|---|---|---|
| 1 | Ticket-bound authority (Ricardian) | **High** | v1.1, ~2 months post-v1 | Low (composes naturally with v1) |
| 2 | Privacy masking (Symthaea-gated) | **High** | v2 | Medium (depends on Symthaea public API) |
| 3 | Biometric proof-of-presence | **Medium** | v2 | Medium (accessibility + privacy needs user research) |
| 4 | Holographic replay audit | **Low** | Speculative companion product | High (usability claim unverified) |

Ship (1) in the **first minor-version spec bump after v1 release**
if the ticket-state binding gets positive signal from the
decentralized-trust pitch. Defer (2) and (3) to v2. Treat (4) as a
research-curiosity side project.

---

## Post-Track-A decision tree

Based on the signal picked up during Weeks 4-6 + the 30 days after
release:

```
paper accepted + crate adopted + inbound inquiries
    → Track B is worth starting (Windows agent first, scope-capped)
paper accepted OR crate adopted (but not both)
    → Publish v0.2 with spec feedback incorporated;
      defer Track B until a second signal appears
nothing stuck
    → Track A outputs stand on their own; focus back on
      Phase III/IV research without distraction
```

Explicitly: no commitment to Track B/C upfront. The plan is a
**feedback-gated option**, not a sequence.

---

## Open questions — all decided as of 2026-04-17 (draft 3)

1. ~~Crate name~~ **DECIDED**: Xenia (`xenia-wire` +
   `xenia-viewer-web`). Prior-art: Xbox emulator `xenia-project`
   (executable, not a library — namespace clean).
2. ~~Paper target~~ **DECIDED**: decide at end of Week 3 based on CFP
   timing. Candidates remain USENIX Security, NDSS, ACM CCS,
   IEEE S&P. Arxiv + workshop as fallback.
3. ~~Repo location~~ **DECIDED**: `luminousdynamics/xenia-wire` on
   GitHub, public from day 1 with a PRE-ALPHA banner.
4. ~~License~~ **DECIDED**: Apache-2.0 OR MIT dual (the symtropy
   pattern). Both license files ship in the repo root.
5. ~~Budget~~ **DECIDED**: plan for 6 weeks, aim for 4. The 6-week
   schedule includes a buffer for the high-risk WASM+AEAD work in
   Week 4; the core deliverables fit a 4-week sprint if Week 4 is
   clean.
6. ~~Consulting outreach list~~ **DECIDED**: draft the ~20-name
   research list during Week 6 polish, targeting MSP owners,
   cybersecurity consultants, decentralized-tech founders, and
   independent security researchers.
7. ~~Frontier features ordering~~ **DECIDED**: v1.1/v2 split per the
   engineering-take table in the Future spec extensions section.
   Ticket-bound authority (Ricardian) in v1.1; privacy masking and
   biometric presence in v2; holographic audit as a speculative
   companion product (not a spec extension).

The only remaining prerequisite before Week 1 begins is the
4-6-week calendar-time commitment in the Prerequisites section
above. Once that's blocked out, Week 1 can begin in a fresh,
focused session.

---

*Plan authored 2026-04-17. Revisions track the change log below as
the plan gets feedback before Week 1 starts.*

## Change log

- **2026-04-17** (draft 3): **Prerequisites resolved + Week-5 design
  considerations locked in.** All seven open questions answered (see
  Open questions section). Repo set to `luminousdynamics/xenia-wire`,
  public from day 1 with PRE-ALPHA banner. crates.io publisher: reuse
  BWS-stored Luminous Dynamics token. License Apache-2.0 OR MIT dual.
  Paper venue decision deferred to end of Week 3 based on CFP timing.
  Consulting outreach list drafted in Week 6. Frontier feature
  ordering per the engineering-take table. Week 5 gains new subsection
  "5d. Design considerations locked in before implementation":
  (5d-1) `causal_binding: Option<CausalPredicate>` reserved in v1
  `ConsentRequest` so the v1.1 Ricardian extension doesn't break
  wire compat; (5d-2) `ConsentRevocation` payload type `0x22` is a
  v1 feature not later, with `WireError::ConsentRevoked` and
  terminal-event recording in the attestation chain; (5d-3)
  tech-credential attestation chain made concrete — MSP runs a
  Holochain agent, MSP-key signs tech-device-key in canonical DID
  format, stored in a Holochain directory, carried in
  `ConsentRequest.msp_attestation`, fall-back to cached signature
  with user-visible offline indicator for availability. Payload-type
  rebase: `AttestedAction` moves from `0x22` → `0x23` to make room
  for `ConsentRevocation`. Only remaining prerequisite before Week 1
  is the 4-6-week calendar-time commitment.
- **2026-04-17** (draft 2): **Name approved — Xenia.** Doc renamed
  from `SOMA_PROTOCOL_TRACK_A_PLAN.md` to
  `XENIA_PROTOCOL_TRACK_A_PLAN.md`. All external-facing references
  updated: `xenia-wire` crate, `xenia-viewer-web` viewer,
  `.xenia-session` replay format, `XeniaReplay` API, `xenia-replay`
  CLI, `hello_xenia.rs` example, `luminousdynamics/xenia-demo`
  Docker image. Internal Symthaea names (`SomaRdpServer`,
  `HOLON_SOMA_ROADMAP.md`) unchanged — that's the naming boundary.
  Added etymology + pronunciation note at the top. Added prior-art
  note on the Xbox emulator `xenia-project` (executable, not a
  library — namespace clean). Open-question #1 marked decided.
  Added new "Future spec extensions — beyond ConnectWise, beyond
  v1" section with four features (ticket-bound authority via
  Ricardian contracts, privacy masking via Symthaea, biometric
  proof-of-presence, holographic replay audit) prioritized with
  engineering commentary and honest caveats. Summary table ranks
  them v1.1 / v2 / speculative.
- **2026-04-17** (draft 1): Initial plan document under the Soma
  name, scope gate, six-week milestones, three spec-level
  differentiators (consent ceremony, sealed-replay recording,
  attestation log), success criteria, failure-mode table,
  post-Track-A decision tree. No implementation work yet — this
  is the plan, not the work.
