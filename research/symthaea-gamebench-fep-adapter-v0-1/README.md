# Symthaea GameBench candidate adapters

This is an isolated research workspace for real candidate baselines built on Symthaea's existing cognitive submodules. It composes with [GameBench #14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137) and the observation/action-only interface in `symthaea-gamebench-v0-1`.

## Implemented candidate arms

| Arm | Existing Symthaea component | Mechanism evaluated | Limit |
|---|---|---|---|
| FEP navigation | `symthaea-fep::ActiveInferenceAgent` | Belief update, expected-free-energy action selection, transition learning from next observation | F1 O1 full map only; simple two-dimensional transition prior |
| Spatial HDC | `symthaea-core::hdc::unified_hv::ContinuousHV` | 16,384D spatial hypervectors, role binding, bundling and cosine similarity | One-step spatial policy; no temporal state or episodic memory |
| HDC-LTC spatial | `symthaea-core::hdc::hdc_ltc_unified::HdcLtcUnifiedNeuron` | HDC representation with closed-form temporal evolution | Small stateful temporal baseline; not the CfC implementation and not a full cognitive loop |

All arms use the same observation/action-only candidate contract. The candidate receives only the current declared observation, offered action attempts, remaining budget and step count; it receives transition feedback as previous/next candidate-visible observations. No task seed, evaluator ID, full environment, hidden state, exact planner or evaluator outcome is passed to reset or policy methods.

## Confidence and prediction reporting

The FEP action softmax is a relative policy-selection distribution, not a calibrated probability of task success. HDC cosine similarity is a representation score, not success confidence. All three adapters therefore report `confidence: None` until a separate calibration map is fitted and tested on held-out tasks.

The FEP arm emits a discrete coordinate forecast based on its predicted state mean and precision. HDC/HDC-LTC emit point coordinate forecasts derived from the candidate's representation/selected transition. These forecasts are scored independently of task success.

## What this does not establish

These are component-level candidate arms, not a fully integrated Symthaea HDC+CfC system. The current adapter package does not yet wire the main `src/dynamics/cfc` network, production episodic memory, global workspace or multi-stage planning into one candidate. It does not establish visual game competence, transfer, SOTA performance or robotics transfer.

The HDC-LTC implementation uses `HdcLtcUnifiedNeuron`; do not label it CfC. The FEP implementation uses `ActiveInferenceAgent`; do not label it full Symthaea.

## Isolation

This package has its own Cargo workspace and is not a production Symthaea workspace member. It path-depends on the actual sibling `symthaea-fep`, `symthaea-core`, and standalone GameBench crates. No production Cargo manifest or runtime dependency has changed.

## Validation

From this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Tests are source-authored, not yet executed in this environment. The dedicated adapter workflow is the qualification gate.