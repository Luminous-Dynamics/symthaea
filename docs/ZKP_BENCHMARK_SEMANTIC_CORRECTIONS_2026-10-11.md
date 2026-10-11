# ZKP benchmark semantic-correction receipt

**Date:** 2026-10-11  
**Branch:** `audit/correct-hdc-zkp-circuits`  
**Scope:** Public Symthaea repository, packed-word Binius proof demonstrations and published claims.

## Corrections in this change

- Hamming distance now constrains a real 64-bit population count for every XOR word and sums those counts into the public output. The prior circuit summed/truncated XOR-word values; historical Hamming measurements are superseded.
- Multi-input bundling and opaque-word aggregation now use a bit-sliced counter and strict-majority comparison. Ties resolve to zero. The prior (n>3) AND tree was not majority; historical 8- and 16-input results are superseded.
- Rust tests cover boundary popcounts, incorrect count claims, multiple majority arities, and the four-input case that distinguishes majority from all-input AND.
- The former federated-learning demo is re-scoped as a proof of strict bitwise majority over opaque private words. It does not prove OTP encryption, participant identity, numeric gradient aggregation, or an end-to-end federated-learning protocol.
- The former triple-stack output is labeled as a proof of an XOR relation over opaque words, not proof of an encryption pipeline.
- The public reproduction script resolves the repository root correctly and only claims to run artifacts present in the public standalone repository. It no longer invokes absent health/Winterfell paths.
- README and paper claims that depended on invalid circuit semantics or an unmeasured same-scale Winterfell comparison are narrowed or withdrawn.

## Evidence status

- **Source reviewed:** Binius64 frontend methods for bitwise operations, shifts, 64-bit integer addition/subtraction, and constraint verification.
- **Automated tests:** authored in Rust; execution is pending the repository CI/build environment.
- **Not claimed:** fresh benchmark timings, an independent cryptographic review, a measured cross-backend speedup, numerical CfC correctness, end-to-end health attestation, or encrypted federated learning.
- **Remaining research work:** implement a numerical fixed-point CfC circuit with constrained gating; publish and run a semantically matched Winterfell XOR baseline; independently validate any separate health/credential proofs before making public quantitative claims.
