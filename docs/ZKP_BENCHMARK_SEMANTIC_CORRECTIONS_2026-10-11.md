# ZKP benchmark semantic-correction receipt

**Date:** 2026-10-11  
**Branch:** `audit/correct-hdc-zkp-circuits`  
**Scope:** Public Symthaea repository, packed-word Binius proof demonstrations and published claims.

## Corrections in this change

- Hamming distance now constrains a real 64-bit population count for every XOR word and sums those counts into the public output. The prior circuit summed/truncated XOR-word values; historical Hamming measurements are superseded.
- Multi-input bundling and opaque-word aggregation now use a bit-sliced counter and strict-majority comparison. Ties resolve to zero. The prior (n>3) AND tree was not majority; historical 8- and 16-input results are superseded.
- Rust tests cover boundary popcounts, every single-bit position, incorrect count claims, multiple majority arities, explicit threshold/tie cases, and the four-input case that distinguishes majority from all-input AND.
- The former federated-learning demo is re-scoped as a proof of strict bitwise majority over opaque witness words; Binius64 does not guarantee their confidentiality. It does not prove OTP encryption, participant identity, numeric gradient aggregation, or an end-to-end federated-learning protocol.
- The former triple-stack output is labeled as a proof of an XOR relation over opaque words, not proof of an encryption pipeline.
- The public reproduction script resolves the repository root correctly and only claims to run artifacts present in the public standalone repository. It no longer invokes absent health/Winterfell paths.
- README and paper claims that depended on invalid circuit semantics or an unmeasured same-scale Winterfell comparison are narrowed or withdrawn.

## Evidence status

- **Source reviewed:** Binius64 frontend methods for bitwise operations, shifts, 64-bit integer addition/subtraction, and constraint verification.
- **Automated tests:** authored in Rust; execution is pending the repository CI/build environment.
- **Not claimed:** fresh benchmark timings, an independent cryptographic review, a measured cross-backend speedup, numerical CfC correctness, end-to-end health attestation, or encrypted federated learning.
- **Remaining research work:** implement a numerical fixed-point CfC circuit with constrained gating; publish and run a semantically matched Winterfell XOR baseline; independently validate any separate health/credential proofs before making public quantitative claims.


## Follow-up source and CI review (2026-10-12)

- A second-pass inspection found a duplicated token sequence in `bundling.rs`'s witness-population call that would have prevented compilation. The malformed line is corrected.
- The legacy temporal modules now describe what their circuits actually constrain: a private, unconstrained gate word driving a packed-bit recurrence. The former `sigmoid` path is explicitly a low-byte mask; it has no sigmoid, numeric activation, clamp, or range check. Labels and call sites were aligned with that semantics.
- Added a dedicated `HDC Binius proof-circuit tests` CI job. It checks formatting for the isolated crate, runs the crate tests, and executes the public reproduction script. The normal root-workspace checks did not explicitly cover this standalone crate.

## Current validation status

- **Static review:** the duplicated bundling call is fixed; the standalone HDC test/reproduction job is now wired into CI.
- **Automated execution:** pending. No passing Rust build, circuit test, reproduction, or workflow-syntax result has been observed yet for the latest head.
- **Still not claimed:** numeric CfC proof, sigmoid correctness, OTP encryption, numeric federated-gradient aggregation, measured Winterfell baseline, or fresh benchmark metrics.


## Binius64 privacy boundary (2026-10-12)

- The vendored `vendor/binius64/ARCHITECTURE.md` explicitly lists the standard `binius-prover` / `binius-verifier` backend as **not zero-knowledge** and lists Iron Spartan as the separate ZK backend. The HDC benchmark depends on the standard Binius64 crates, not Iron Spartan.
- Therefore the current HDC circuits demonstrate constraint satisfaction / computational integrity only. They do **not** guarantee that witness inputs remain confidential. Calling a wire a witness or omitting it from `public_words` is not, by itself, a zero-knowledge guarantee.
- Replaced the misleading triple-stack runtime demo with an opaque-word XOR integrity proof. It no longer simulates host-side OTP values and then reports encryption or perfect secrecy as verified.
- Re-scoped README and paper language to remove any implication that the bundled Binius64 HDC circuit is privacy-preserving. A ZK-capable backend and privacy/security review are prerequisites for those claims.
- Added `spartan_privacy.rs`, a minimal Iron Spartan proof smoke test with two private Boolean witness bits, a public XOR output, and a regression check rejecting a non-Boolean field value. This is not yet a migration of the large HDC circuits and has not passed CI in this session.
- The vendored Iron Spartan verifier declares `SECURITY_BITS = 96`; evaluate the target security policy and parameters before production adoption.

## CI follow-up

- Registered the new HDC test job in the top-level draft-safety census.
- Fixed the workflow-safety checker so manual-only runner roots are not incorrectly required to include `ready_for_review`; its existing self-test covers this branch.
- The latest automated HDC build, test, reproduction, and workflow checks still need a completed run. No green result is claimed yet.

## Benchmark metric and architecture follow-up (2026-10-12)

- The main XOR harness reads CircuitStat::n_and_constraints and CircuitStat::n_mul_constraints; its reported count is specifically the nonlinear AND/MUL count. It must not be described as a count of all equality wiring or all circuit rows.
- Removed stale prose that called an older 256-count “structural AND constraints” and implied it was the current nonlinear metric. Earlier timing/proof-size values remain historical only until the corrected branch is executed and exact output is retained.
- Reframed the multi-backend section as a proposal, not an automatically routed or deployed system. The public harness uses the standard non-zero-knowledge Binius64 backend; the separate Iron Spartan check is only a tiny Boolean-XOR smoke test, and does not show that the HDC circuits provide witness privacy.
- Replaced outdated limitations text that still asserted an unmeasured Winterfell proof-size/speed advantage and referred to the packed-bit recurrence as a numerical CfC/sigmoid proof.
- These documentation/source-comment changes do not substitute for the still-pending Rust build, proof tests, or fresh measurements.


## Algebraic-cost and permutation-scope follow-up (2026-10-12)

- Rewrote the XOR theorem to distinguish the absence of nonlinear AND/MUL gates from the wordwise equality assertions that bind the public output. A nonlinear-gate count is not a count of all circuit structure or a performance result.
- Withdrawn the unqualified claim that a general HDC cyclic permutation requires zero constraints. Whole-word reordering can be an index permutation; general bit-level shifts across packed-word boundaries require separate rotation/shift logic and a measured circuit.
- Removed the stale source-header claim that an executable Winterfell XOR circuit is available at a monorepo path absent from this standalone artifact. The benchmark now describes the Winterfell side only as an analytic estimate.


- Follow-up source scan removed a second stale comment that referenced a non-existent standalone path for a separately measured Winterfell circuit. Both module-level and function-level descriptions now consistently state that this crate contains no Winterfell XOR prover.


## Iron Spartan statement-binding regression (2026-10-12)

- Extended the minimal Iron Spartan smoke test to serialize the proof and verify it through a fresh verifier transcript.
- Added a negative case that changes the public XOR output from one to zero and requires verification to fail. This tests proof binding to the public statement rather than only checking that a valid proof verifies.
- The regression is authored but remains unconfirmed until the current CI head runs the Rust test successfully.


## Upstream Binius64 ZK API research (2026-10-12)

- Current upstream `binius-zk/binius64` exposes `binius_prover::zk_config::ZKProver` and `binius_verifier::zk_config::ZKVerifier`, which wrap the word-oriented Binius64 IOP proof with an Iron Spartan-based ZK wrapper. Official upstream source: [prover](https://github.com/binius-zk/binius64/blob/main/crates/prover/src/zk_config.rs), [verifier](https://github.com/binius-zk/binius64/blob/main/crates/verifier/src/zk_config.rs).
- The vendored Symthaea snapshot lacks those wrapper modules, while retaining older non-ZK standard APIs plus separate Spartan crates. Accordingly, the current public benchmark's non-ZK limitation is real, but must be scoped to the vendored API snapshot rather than generalized to the entire Binius64 project.
- Preferred next experiment: pin a known upstream revision (or perform a reviewed backport) and prove the existing packed-word XOR statement with the word-oriented ZK wrapper. Compare security parameters, transcript/public-input binding, proof verification, build reproducibility, and performance against the existing non-ZK path. Avoid reimplementing all HDC circuits over the older GF(2^128) frontend unless the wrapper route proves infeasible.
- Research finding only: this integration/migration is not part of this PR and no current upstream security claim substitutes for testing and reviewing the exact code vendored in Symthaea.


## Standalone package metadata cleanup (2026-10-12)

- Removed the remaining `Cargo.toml` comment claiming a measured Winterfell XOR circuit lives at a monorepo-only path. The standalone package now consistently states that it only prints an analytical estimate for Winterfell and does not provide a measured same-scale comparison.


## Upstream Binius64 wrapper disclosure report (2026-10-12)

- A fresh upstream review identified [binius-zk/binius64 PR #2505](https://github.com/binius-zk/binius64/pull/2505), titled “Reject private words in the ZK prover and verifier, since proofs leak them.” At the time inspected, this security-relevant PR was open and unmerged.
- The report describes a composed-wrapper leak: `ZKWrappedProverChannel::prove_oracle_relation` sends inner oracle-relation claims through the inner channel rather than the wrapper path that applies its one-time-pad key. The verifier can compute the relevant linear functional from public data / Fiat–Shamir challenges. The report provides a public-API reproduction on its stated base revision and records disclosure in the stated test cases.
- The proposed upstream patch rejects constraints/witnesses with private words as a stopgap; it is not the cryptographic protocol repair. The open report also says hiding of witness-dependent oracle openings has not been established. Do not treat the existence or naming of `ZKProver` / `ZKVerifier` as evidence that private HDC witnesses are hidden.
- Scope discipline: this report concerns the composed Binius64 ZK wrapper. It does not establish that the separate, direct Iron Spartan proof system exercised by this repository's tiny smoke test has the same flaw. That smoke test itself remains too small to establish privacy for the HDC relations and is not an independent cryptographic audit.
- Migration gate is strengthened: do not integrate the current upstream wrapper as a private-witness solution until a protocol fix is reviewed, the disclosure reproduction is converted into a failing regression test against the vulnerable code / passing test against the fix, tests cover transcript noninterference and hidden-word layouts, and independent review supports the claimed zero-knowledge property. Public-only wrapper operation does not meet the private-witness HDC requirement.
- This is an upstream security finding and an integration blocker. The standard non-ZK Binius64 harness in this PR remains accurately described as computational-integrity-only. No new claim about standalone Iron Spartan is made.
