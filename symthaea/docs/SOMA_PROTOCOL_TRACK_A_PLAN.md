# Soma Protocol — Track A Plan (Library + Paper)

**Scope**: 4-6 weeks of focused work to extract the Holon-Soma wire from
symthaea into a standalone open-source crate, write a protocol
specification, publish a paper, and seed initial adoption. Low-floor,
high-ceiling — even total failure leaves behind a research substrate
for Phase III/IV.

**Status**: planning (2026-04-17). No implementation until this doc is
approved.

**Relationship to existing roadmap**: `HOLON_SOMA_ROADMAP.md` is the
research roadmap (Phase I.A through V, all Symthaea-internal). This
plan is the **commercial/academic off-ramp**: the same wire as a
general-purpose protocol, extracted and published. The two reinforce
each other — paper credibility funds grant applications; grant funding
keeps the research going; research numbers are the paper's empirical
section.

---

## One-paragraph summary

Extract `symthaea/src/swarm/rdp_wire.rs`, `rdp_session.rs`,
`replay_window.rs`, and the minimal `rdp_protocol.rs` types needed to
use them, into a standalone `soma-wire` crate. Publish to crates.io
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
- Extracting the Soma wire into a standalone crate (`soma-wire`).
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
- `soma-wire/` at repo root, published to crates.io as `soma-wire
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

**Exit criterion**: `cargo add soma-wire` in a fresh crate compiles +
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
- `papers/soma-paper.md` (or LaTeX in the repo), 10-15 pages.
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
- `soma-viewer-web/` subcrate with `wasm-bindgen` + `web-sys`.
- HTML/JS shell that:
  - Connects to a Soma server via WebSocket or WebTransport.
  - Displays received frames via canvas.
  - Sends mouse/keyboard events as sealed `InputFrame`s.
  - For the demo, session-key bootstrap is via PAKE or QR code —
    NOT a real ML-KEM handshake.
- Deployed to a static site (GitHub Pages, Cloudflare Pages, or
  `luminousdynamics.io/soma`) so paper reviewers can click a link,
  scan a QR, and see the protocol working.
- Live demo harness: a small server (Docker image) MSPs can run with
  `docker run -p 7778:7778 luminousdynamics/soma-demo` and connect
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
    shell), time limit.
  - `0x21` `ConsentResponse` — end user signs approval or denial
    with their device key.
- Session establishment requires a valid `ConsentResponse` before any
  `RdpFrame` is accepted. Server-side enforcement: if missing,
  `session.open()` on an `RdpFrame` returns `WireError::NoConsent`.
- Spec section documents the flow + threat model (what MITM looks
  like, why device key signing prevents it).
- Test vectors for consent request + response.

**5b. Sealed-replay recording** (~1 day):
- New file format `.soma-session` — a simple container: metadata
  header (session parties, start time, protocol version, signatures)
  followed by the sealed envelopes in arrival order, each with a
  length prefix.
- `SomaReplay` API: `open(path)`, `next_frame() -> Option<(ts,
  RdpFrame)>`, `seek(ts)`.
- Spec section describes the format. Security property: tamper
  evidence — any modification breaks the AEAD chain.
- Reference CLI: `soma-replay <file.soma-session>` plays back into
  the viewer.

**5c. Attestation-chained action log** (~2 days):
- Every command/input the tech issues is signed by the tech's device
  key and logged with monotonic sequence number + hash of prior log
  entry (blockchain-of-one-tech).
- New payload type `0x22` `AttestedAction`.
- End-user client can verify the chain retroactively to prove no
  tamper.
- Spec section documents it.

**Exit criterion**: each of 5a/5b/5c is spec'd, implemented with
tests, and the spec cross-references them as v1-required features.

**Why in Track A** (not deferred): retrofitting security-critical
protocol features after v1 is painful and often impossible. Ship them
now so the 0.1.0 release is the real thing.

### Week 6 — Polish + release + outreach

**Deliverables**:
- Bump `soma-wire` to `0.1.0` (stable).
- Release binary for `soma-viewer-web` at the deploy URL.
- Paper submitted (or arxiv + preprint blog post if CFPs don't align).
- `BLOG_POST_1.md`: "Announcing Soma — a PQC-sealed remote-control
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
soma-wire/                              (published crate, version 0.1.0)
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
│   ├── hello_soma.rs                   (30-line quick-start)
│   ├── replay_session.rs
│   └── attest_action.rs
├── benches/
│   └── seal_open_throughput.rs
└── tests/
    ├── integration_roundtrip.rs
    ├── integration_consent.rs
    └── integration_replay.rs
```

`soma-viewer-web/` is a separate crate in the same workspace —
optional, only built with the `wasm` feature set.

---

## Success criteria

**Green (unambiguous hit)** — any two of:
- Paper accepted at a Tier-1 security venue.
- `soma-wire` accumulates ≥100 GitHub stars in the first month or
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

- [ ] User approves this plan (scope, branding, license).
- [ ] Decision on crate name: `soma-wire` (default), `holon-rdp`,
      `luminous-soma`, or user's preferred name.
- [ ] Decision on GitHub repo location: `luminousdynamics/soma-wire`,
      or under tstoltz's personal org, or a new org.
- [ ] Decision on which user identity publishes to crates.io (reuses
      the existing BWS-stored token or a new per-crate token).
- [ ] 4-6 weeks of calendar time blocked out — Track A does not
      succeed if fragmented across 12 weeks of part-time work.

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

- **Phase III Φ-sweep** (research roadmap): the `soma-wire` crate
  becomes the wire for the 360-trial matrix. The extracted crate is
  cleaner than the current in-tree version, which helps the research.
- **Phase IV Markov blanket**: the split-cognition experiments need
  WAN-capable transport. Soma's QUIC + LZ4 is the right substrate.
  Track A effort directly serves Phase IV.
- **Consciousness-gated session oversight** (future paper): a second
  paper, after Symthaea side-car integration is done. Track A makes
  this paper possible by giving the wire a clean public surface to
  integrate against.
- **Grant applications**: Track A's outputs (spec, paper, demo) are
  concrete deliverables for PQC + decentralized-trust-focused
  grants (NIST PQC transition, NSF Secure Computing, etc.).
- **Consulting revenue**: the paper + crate + demo are the pitch for
  "hire me to integrate Soma into your stack." No product-company
  overhead.

---

## What Track A does NOT commit to

- A Track B/C go/no-go decision. That's a post-Week-6 conversation.
- A business entity, trademark, or commercial licensing discussion.
- Windows/macOS/iOS agents (Track B).
- MSP tenant dashboard (Track C).
- Any named pricing, any sales funnel, any SaaS.

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

## Open questions for the user

1. **Crate name**: `soma-wire` default; user preference?
2. **Paper target**: do we pre-commit to a specific venue now, or
   decide at end of Week 3 based on CFP timing?
3. **Repo location**: under `luminousdynamics` org on GitHub? Public
   from day 1? Or private until Week 6 release?
4. **License**: Apache-2.0 OR MIT (the symtropy pattern) confirmed?
5. **Branding**: Soma is a Holochain cell concept + a Luminous
   Dynamics project name — any overlap to worry about?
6. **Budget**: 4 weeks or 6 weeks? The plan is written at 6 but the
   core deliverables fit in 4 with tight scope control.
7. **Consulting outreach list**: user has the ~20 names in mind or
   wants me to draft a short research list?

---

*Plan authored 2026-04-17. Revisions track the change log below as
the plan gets feedback before Week 1 starts.*

## Change log

- **2026-04-17** (draft 1): Initial plan document, scope gate, six-week
  milestones, three spec-level differentiators (consent ceremony,
  sealed-replay recording, attestation log), success criteria,
  failure-mode table, post-Track-A decision tree. No implementation
  work yet — this is the plan, not the work.
