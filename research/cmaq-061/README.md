# CMAQ-061 — Previsible learned representations

This crate supplies a bounded unary encoder primitive for sequential conditional paired monitoring. It is intentionally separate from C058/C059 and does not claim a complete end-to-end learned e-process.

## Contract

For pair t, the encoder state `theta_t` is frozen before observing `(X_t,Y_t)). The same unary map is applied separately to both responses:

`h_t^X = clip_B(g_{theta_t}(X_t))`, `h_t^Y = clip_B(g_{theta_t}(Y_t))`.

Under direct conditional paired equality and measurability of `theta_t` with respect to the full pre-pair filtration and current pre-pair covariate, the transformed difference is conditionally centered. A parent betting layer may therefore construct its bounded predictable factor from this difference.

The API separates:
1. `encode_pair`: transform current X/Y with the current frozen version; no mutation.
2. `prepare_next_version`: stage a candidate version from the symmetric pair mean; no mutation.
3. Parent evidence commit.
4. `commit_next_version`: advance the encoder version only after the parent commits the current evidence transaction.

The update rule in this first implementation is a bounded deterministic Hebbian-style adaptation using only the unordered pair mean. It is a research baseline, not claimed to be an optimal learned representation. Its value is that it exercises the predictability boundary with a small, auditable implementation.

## Validity limits

- This crate does not itself create an e-process; the parent must use a predictable bounded direction, positive factor, and correct null theorem.
- The same unary map must be applied to X and Y. Arbitrary joint pair encoders are outside this contract.
- No current-pair information may affect the factor applied to that same pair. Staged updates are only eligible for the next pair.
- The deterministic output clipping gives a hard norm bound, but clipping may collapse directions and reduce power; clipping rates must be reported.
- The encoder's input preprocessing, parameters, version, digest, optimizer/update rule, and update order must be bound into the parent policy trace.
- The current online update uses both X and Y symmetrically. It does not guarantee that representation adaptation improves power; it only preserves the conditional-centering argument under the stated null.

## Tests

Deterministic tests cover same-map use, pair-order sign symmetry, pair-order-invariant next-version updates, output norm clipping, non-mutating preparation, version commit/replay rejection, sequence monotonicity, and invalid/non-finite inputs.

## Qualification still required

No local Rust compiler was available during authoring. The exact-head workflow runs formatting, tests, and strict Clippy. Before integrating with CMAQ-057/058, add an end-to-end test that commits evidence first, then commits the encoder version, and a negative control that intentionally uses the next version on the current pair and is rejected by the transaction coordinator.

The next benchmark should compare fixed Fourier, fixed random neural, and previsible-adaptive encoders on identical streams. Report null calibration, power, restricted delay, clipping rate, feature compute, update cost, memory, and worst-family regressions. Do not claim SOTA based on this API or synthetic wiring tests.
