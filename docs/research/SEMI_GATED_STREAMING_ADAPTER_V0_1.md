# SEMI-GATED-STREAM-001 — evidence-gated streaming VM adapters v0.1

**Issue:** #13470  
**Implementation:** `symthaea-semiconductor::benchmark::gated_streaming_adapter`  
**Status:** Rust streaming-adapter engine with authored unit tests; exact-head validation pending.

## Research anchor: GRAPE

The public author repository is `https://github.com/cheonwoolee/GRAPE` (Apache-2.0). It contains anonymized derived feature/label bundles for two semiconductor datasets, EQP/chamber context IDs, a PyTorch global predictor and two interpretable online adapters.

The upstream runner's prequential path predicts a sample and immediately updates from that same sample's true label, while carrying mutable model state through validation, test and unseen streams. This is a zero-delay label assumption. Its unseen branch also selects the first `nf` columns rather than the training-derived ranked indices used by train/validation/test.

Our benchmark must keep two tracks distinct:
1. **Official-code-faithful GRAPE:** reproduce the released code's exact preprocessing, splits, feature indices, update order and metrics, including its zero-delay assumption and the unseen-index behavior.
2. **Corrected/operational track:** same frozen global baseline and declared adapters, but with consistent training-derived feature indices, explicit label availability and evidence-gated delayed updates.

Never silently repair the official baseline and then claim paper reproduction.

## Paradigm: prediction and learning are separate events

The streaming protocol is now:

`feature observation -> immutable prediction receipt -> pending label -> label available -> K1 qualification -> adapter update receipt`

The prediction records the exact selected feature vector, global baseline prediction, feature/context/temporal correction decomposition, context key, global-model artifact digest, feature-schema digest, policy digest and adapter-state digest.

A later-arriving label does not recompute the old residual against the current adapter. The update uses the frozen prediction-time selected features and prediction value, which keeps delayed-label learning causally anchored to the original forecast.

This v0.1 engine implements deterministic **arrival-order updates**. Event-time replay with a new immutable state chain is a separate future mode; it must not rewrite the arrival-order receipts.

## Implemented adapter state

The global VM predictor is supplied as an immutable baseline prediction in v0.1; this module does not fit its weights.

Online state contains:
- feature-level normalized LMS (NLMS) weights;
- a global short-term residual EMA;
- a global environment/context residual EMA;
- per-context residual EMA and support count;
- accepted update count and idempotency set.

The environment correction shrinks local estimates toward the global correction until the context has sufficient support. A new tool/chamber context starts from global context state; it cannot inherit another context's local correction.

NLMS uses a finite norm floor, maximum feature/error limits and bounded per-coordinate weight deltas. Zero feature norm is safe and does not divide by zero or cause unbounded updates.

## Evidence and availability gate

A label delivery is accepted only when:
- the schema and idempotency key are valid;
- the sample has a pending prediction;
- the label's full payload digest matches the K1 eligibility report's label digest;
- the label ID, revision, measurement-event ID and measurement value match;
- the report digest independently replays;
- the report frontier is `CurrentAtK1`;
- eligibility is `EligibleUnderDeclaredPolicy` and downstream consumption is permitted;
- label availability, report availability and delivery times are ordered correctly;
- the clock domain and feature-schema/global-model identities match;
- the label revision has not already been accepted.

UNKNOWN is held for possible explicit re-presentation with a later qualified K1 report. BLOCKED never updates. Any identity, payload, digest, timing or policy mismatch is rejected. A denied attempt emits a receipt but does not mutate adapter parameters, update count, idempotency state or the pending prediction.

The report's `source_authenticity_proven=false` is not treated as an automatic block: eligibility here means the declared software evidence contract passed; this module makes no issuer-authenticity claim.

## Atomicity and receipt chain

All successful candidate updates execute against a cloned adapter state. The engine computes the candidate state digest and update receipt first, then commits the state, consumes the pending prediction and records the accepted label revision. Any overflow or invalid intermediate value leaves live state unchanged.

The update receipt binds:
- prediction receipt;
- label and revision;
- label payload digest;
- K1 eligibility receipt;
- policy digest;
- adapter state before/after;
- update sequence and disposition.

Prediction receipts remain immutable after label delivery. Repeated accepted label ID/revision produces a DUPLICATE no-update receipt.

## Authored adversarial tests

The module currently includes tests for:
1. prediction is read-only and freezes selected features/state digest;
2. qualified zero-delay update;
3. delayed label after an intervening update uses prediction-time data;
4. UNKNOWN is held and can later be explicitly re-presented as qualified;
5. BLOCKED label never updates;
6. accepted revision is idempotent;
7. label/event/value/payload mismatch is rejected;
8. label/report availability and clock checks;
9. K0 or stale eligibility report is rejected;
10. zero feature vector is safe;
11. context correction is local and shrinks to global support;
12. feature-schema mismatch and duplicate measurement-event prediction reject;
13. fixed arrival schedule replays deterministically;
14. unavailable K1 eligibility report cannot be used before report availability;
15. prediction sequence overflow does not partially register a sample;
16. context-support overflow leaves the entire update transaction unchanged.

These tests are authored, not yet reported as passing. Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference runs are required.

## Benchmark tournament

Compare on identical chronological streams and equal **qualified-label** budgets:
- official-code-faithful immediate GRAPE;
- delayed raw GRAPE-style adapters with no evidence gate;
- evidence-gated delayed adapters (this module);
- frozen global/no-adaptation baseline;
- rolling OLS/online linear and AdaOTP where the task protocol allows.

Report:
- per-target RMSE, MAE and R² by chronological window/context;
- worst-window bias and negative-transfer frequency;
- shift recovery wafers, recovery delay and overshoot;
- qualified/UNKNOWN/BLOCKED/rejected/duplicate labels;
- delay backlog and label age;
- updates withheld by reason;
- per-wafer p50/p95 update latency, memory/state size and throughput;
- interval coverage separately from point accuracy.

Hard software targets:
- 100% no-update for UNKNOWN/BLOCKED, mismatched, stale or not-yet-available labels;
- 0 duplicate accepted updates;
- 0 prediction receipt mutation after label delivery;
- 100% deterministic replay for a frozen delivery schedule;
- 0 partial state mutation when an update fails.

A predictive gain target is set only after faithful GRAPE reproduction; compare at equal qualified-label budgets and report paired confidence intervals.

## Claim ceiling

This is a synthetic Rust adapter/event replay contract. It is not a faithful full GRAPE model reproduction, does not authenticate source issuers or reference chains, does not prove physical root cause, does not demonstrate real-fab prediction gains or manufacturing savings, and does not authorize APC/R2R, recipe updates, wafer disposition, rollback, equipment commands or process control.
