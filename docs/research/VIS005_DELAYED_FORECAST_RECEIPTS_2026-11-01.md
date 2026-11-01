# VIS-005 foundation: delayed visual forecast receipts v1

**Status:** implementation branch; intended as a source-level evidence primitive for autonomous vision-manifold forecasts. Not yet action-conditioned. No qualification or SOTA result is claimed.

## Why this increment exists

The vision manifold already issues multi-horizon CfC forecasts and later evaluates them against encoded real observations. Before this change, the evaluator retained aggregate mean error, persistence error, dispersion, lateness, sample counts, dropped forecasts and expirations, but discarded the per-forecast linkage after updating those accumulators.

This increment adds a bounded and checkpointed per-forecast receipt ledger. Each forecast receives a monotonic ID at issuance. When a later observation matures or expires that forecast, the evaluator appends an immutable receipt with the source frame, due time, scoring frame, observed time, lateness policy, content fingerprints and outcome.

## Receipt lifecycle

For each accepted observation in a sensor modality:

1. The evaluator advances its modality-local elapsed-time clock.
2. Every pending forecast whose due time has arrived is evaluated against the current encoded observation.
3. A forecast within the configured lateness tolerance emits a "scored" receipt with prediction error and persistence-baseline error.
4. A forecast outside that tolerance emits an "expired" receipt with no error values; expired forecasts are not silently scored against a stale frame.
5. New autonomous CfC forecasts are issued for each configured horizon, subject to the per-horizon pending queue bound.

A receipt identifies both:
- source_frame_sequence: the frame from which the prediction was issued;
- target_frame_sequence: the actual observation used when the pending forecast matured or expired.

It also records issued_at_seconds, due_at_seconds, observed_at_seconds, requested_horizon_seconds, the exact lateness factor applied, and lateness_seconds. The target is the first accepted observation at or after the due time that reaches the evaluator, provided it satisfies the lateness policy. The current system does not interpolate an exact-time observation between frames. A non-zero lateness must remain visible in every report; do not label it exact-horizon ground truth.

## Fingerprint semantics

Every fingerprint is BLAKE3 over a domain-separated, deterministic byte encoding.

### Visual hypervector fingerprint

The symthaea-vision-hv-v1 domain is followed by:
- the HDC dimension as little-endian u64;
- every component's exact IEEE-754 f32 bit pattern in little-endian order.

This yields fingerprints for:
- the encoded source observation hypervector;
- the source visual-manifold latent state;
- the prediction hypervector compared by the evaluator;
- the persistence hypervector compared by the evaluator;
- the actual target observation hypervector.

These fingerprints bind the encoded vectors, not the raw RGB/depth frame bytes, camera device, sensor calibration, or storage object. The raw observation's exact byte digest is not currently included in this evaluator receipt.

Pending forecasts are deterministically dilated if the owning manifold changes HDC dimension before the forecast matures. In that case, the prediction and persistence fingerprints are recomputed immediately after the dilation, so the receipt identifies the exact vectors the evaluator compares. It does not retain a second digest of the pre-dilation representation; the source frame/belief and source transition-parameter fingerprints remain bound to issuance time.

### Transition parameter fingerprint

The symthaea-vision-cfc-transition-params-v1 domain is followed by:
- the source weight_hv fingerprint;
- VisionConfig.tau_base exact f32 bits;
- VisionConfig.input_blend exact f32 bits.

This is intentionally named a transition parameter fingerprint, not a full checkpoint digest. It does not cover the optimizer, training history, encoder weights, all visual configuration fields, compiler/runtime, binary, or the complete model artifact. A production model identity requires a separate complete checkpoint/source/build receipt.

## Bounded storage and checkpoint migration

DelayedHorizonEvaluator::with_limits lets the caller specify:
- a maximum pending forecast count per horizon;
- a maximum retained scoring-receipt count.

The default retained receipt limit is 4,096. When the ledger is full, the oldest receipt is evicted and evicted_score_receipts increments. The aggregate sample/drop/expiry counters remain independent of receipt retention.

The serialized evaluator state is now schema 4. The pending forecast and scoring receipts are included in checkpoints. The validator rejects:
- zero or duplicate IDs;
- IDs that are not below next_forecast_id;
- over-capacity pending and receipt collections;
- invalid horizons or inconsistent due/issue/observation/lateness arithmetic;
- malformed BLAKE3 fingerprints;
- partially populated source provenance;
- non-finite or out-of-range errors;
- a scored receipt whose lateness violates its stored policy;
- an expired receipt that contains errors or should have been scoreable.

Schema 1–3 checkpoints are migrated. Legacy pending forecasts receive new internal IDs and recomputed prediction/persistence fingerprints from the vectors retained in the old checkpoint. Their source frame, source observation, source belief and transition parameter provenance remain None. The loader does not invent retrospective provenance. Their saved aggregates and existing pending vectors are retained.

## Relationship to CMAQ-018 and the joint action bank

The receipt ledger establishes a reliable audit path for the existing autonomous visual forecasting baseline. The CMAQ-018 joint action bank separately establishes a structure for same-history, multi-action joint samples and paired effect intervals.

A future action-conditioned adapter must add explicit:
- canonical action-manifest digest and action ID;
- hypothetical-action identity in the predictor request;
- initial observed-history digest;
- simulator branch/fork ID and same-reset evidence;
- disturbance-tape identity when using common random numbers;
- model/checkpoint/build identity;
- branch-specific forecast receipt, with the predictor sealed before hidden outcomes are revealed.

Do not reinterpret the current autonomous receipt as a counterfactual branch receipt. It contains no hypothetical action ID, action manifest, branch ID or disturbance tape.

## Claim ceiling

This increment provides:
- monotonic forecast identity;
- bounded per-forecast issuance/scoring/expiry evidence;
- source/target frame and evaluator-clock linkage;
- BLAKE3 fingerprints for encoded source/target/prediction vectors and selected transition parameters;
- checkpoint persistence, schema validation and legacy migration.

It does not establish:
- action-conditioned visual prediction;
- exact-time observation interpolation;
- raw sensor-byte integrity or hardware provenance;
- complete predictor/checkpoint identity;
- truthful stochastic coupling;
- calibration under adaptive action selection;
- physical safety, SOTA or downstream decision advantage.

The next step is to connect a real hypothetical-action-conditioned predictor to this receipt protocol and use Symtropy's sealed exact-state branch evaluator to produce independent realized outcomes for each candidate action.
