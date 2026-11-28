# CMAQ-061 — Previsible learned representations

Status: mathematical design and implementation contract. No learned encoder or benchmark result is implemented by this document.

## 1. Core opportunity

The direct conditional paired-equality null is

    Law(X_t | Z_t, F_(t-1)) = Law(Y_t | Z_t, F_(t-1)),

where Z_t is observed before the response pair. It does not require the response representation to be fixed for all time. It requires the representation used at time t to be measurable with respect to information available before X_t and Y_t.

This creates a route to online representation adaptation without a terminal split: learn from the past, freeze the current model version, apply it identically to X_t and Y_t, create the current evidence factor, and update the learner only afterward.

## 2. Conditional-centering lemma

Let:
- F_(t-1) be the full past filtration, including previous pairs, optimizer state, representation versions, and monitor decisions;
- Z_t be the current covariate, available before X_t and Y_t;
- Theta_t be a representation parameter measurable with respect to sigma(F_(t-1), Z_t, R_t), where R_t is optional randomness independent of (X_t,Y_t) conditional on F_(t-1),Z_t;
- phi_(Theta_t)(x;Z_t) be a measurable map from response space into a separable Hilbert space H;
- psi_t(Z_t) be a measurable context feature with norm at most one;
- u_t be an H-valued or tensor-space direction, measurable before the current pair and with norm at most one.

Assume under H0:

    Law(X_t | F_(t-1), Z_t) = Law(Y_t | F_(t-1), Z_t).

Apply the same realized map phi_(Theta_t)(.;Z_t) to both current responses. Conditional on F_(t-1), Z_t, Theta_t and R_t, the transformed responses have equal laws. Therefore, whenever the expectations exist,

    E[ phi_(Theta_t)(Y_t;Z_t)
       - phi_(Theta_t)(X_t;Z_t)
       | F_(t-1), Z_t, Theta_t, R_t ] = 0.

Multiplying by the predictable context feature preserves centering:

    r_t = 1/2 psi_t(Z_t) tensor
         [phi_(Theta_t)(Y_t;Z_t)-phi_(Theta_t)(X_t;Z_t)]

has conditional mean zero.

If ||phi_(Theta_t)(x;Z_t)|| <= 1 for every admissible x and ||psi_t(Z_t)|| <= 1, then ||r_t|| <= 1. For a predictable ||u_t|| <= 1 and fixed 0 < lambda < 1,

    f_t = 1 + lambda <u_t, r_t>

satisfies f_t >= 1-lambda > 0 and

    E[f_t | F_(t-1), Z_t, Theta_t, R_t] = 1.

Thus M_t = product_(s<=t) f_s is a nonnegative martingale under the direct paired null, relative to the filtration that includes the selected representation and selection randomness. Ville's inequality supplies the single alpha threshold.

The proof uses equality of conditional marginal laws, not exchangeability of the pair (X_t,Y_t). Do not replace the difference of two identical unary transforms by an arbitrary jointly learned pair transform: arbitrary pairwise cross-attention or order-dependent transforms need not have zero conditional expectation under equal marginals.

## 3. When learning is allowed

The following order is mandatory:

1. Observe Z_t.
2. Select/load the versioned representation checkpoint using only past evidence, Z_t, and predeclared independent randomness.
3. Freeze all trainable parameters, normalization statistics, vocabulary/tokenizer state, projections and feature schema for the current pair.
4. Apply the same frozen response map to X_t and Y_t.
5. Enforce the declared output norm bound.
6. Compute the betting direction, rate and factor from pre-pair history.
7. Commit the current e-factor and trace.
8. Only then update the learner, optimizer, running normalization and representation-selection state for time t+1.

A current pair may update the representation for the *next* observation after its own factor is committed. It may not train, choose a checkpoint, fit a scaler, update batch-normalization statistics, choose a dimension, or select a feature map that is then used to compute its own factor.

## 4. Norm enforcement

A learned encoder is not automatically bounded. The monitor must enforce a mathematically defined map such as:

    phi_bar(x) = q(x) / max(1, ||q(x)||_2),

with a safe zero-vector convention and finite-value checks. The normalization itself is part of the registered feature map and must be applied identically to both responses. The implementation must detect overflow/nonfinite pre-normalization outputs, not let NaNs silently pass through a comparison.

For context-dependent encoders, use a fixed bounded context factor psi_t(Z_t) or a deterministic projection to the unit ball. The product feature norm must be proven <=1, not merely checked on observed training data.

## 5. What can invalidate the argument

- The current X_t or Y_t influences Theta_t, u_t, lambda_t, feature normalization, or checkpoint choice before the factor is evaluated.
- X_t and Y_t are passed through different checkpoints or different mutable normalization states.
- The encoder has unbounded outputs or the actual runtime map differs from the registered bounded map.
- Current-pair labels, pair ordering or a downstream evaluator can influence the model update before evidence is committed.
- Selection randomness is correlated with current response values.
- The model uses future labels, asynchronous training updates, batch statistics containing the current pair, or cross-observation operations whose state has not been included in F_(t-1).

These are integrity/proof violations, not ordinary false-alarm events.

## 6. Architecture proposal

A new PredictableRepresentation trait should expose:

- immutable content-addressed checkpoint and feature-schema digest;
- deterministic apply-frozen(x, z) -> bounded feature;
- explicit output dimension and norm-bound semantics;
- staged_update(pair, optimizer_state) returning a next-version candidate but never mutating the currently frozen version;
- commit_next_version() permitted only after the evidence transaction commits;
- trace receipts binding checkpoint, optimizer state, training-pair range/digest, preprocessing, normalization, context map and update order.

The monitor should retain a small fixed Fourier/RBF bank alongside learned representations. A fixed characteristic reference guards against learned-encoder blind spots; a predeclared mixture can combine each valid expert with one global threshold. The learned expert improves power where its representation is useful; the fixed family preserves coverage across registered feature views.

## 7. Benchmark plan

Primary matched-null baseline:
- faithful SCPT, including past-only KNN, Equation (16) betting fractions, and its equal-weight a={2,4,6} mixture;
- C054 fixed Fourier family;
- C057 predictable per-expert Fourier betting;
- C058 capital-only restart;
- learned-predictable encoder with fixed betting;
- learned-predictable encoder with predictable adaptive betting;
- fixed + learned mixture.

High-dimensional tracks:
- covariate dimension p in {2, 8, 32, 128};
- sparse/local effects, conditional variance changes, equal-marginal pairing changes, high-frequency and oscillating shifts;
- low-smoothness and rare-region alternatives;
- temporal drift with conditional equality preserved;
- finite-map blind spots and representation-collapse controls.

Report:
- simultaneous Type-I upper bounds over the complete frozen family;
- unconditional power and restricted delay with early alarms/misses at H+1;
- worst-family and worst-dimension power;
- paired differences on identical source streams;
- p50/p99 encoder, transform, monitor and learner-update latency;
- memory, output dimension, checkpoint bytes and update frequency;
- fraction of time spent in each checkpoint and the exact version trace.

## 8. Acceptance gates

1. Independent review of the conditional-centering lemma and exact filtration.
2. Bounded-encoder implementation and frozen/next-version state machine.
3. Negative controls that inject same-pair leakage through checkpoint selection, normalization, and update order.
4. Mutation tests for checkpoint, preprocessing and optimizer trace.
5. Exact-head Rust fmt/test/Clippy.
6. Preregistered null and alternative Monte Carlo grid using the CMAQ-059 calibration primitives.
7. At least one faithful learned-feature baseline and one fixed-feature fallback.

Prospective target: improve restricted mean delay by at least 20% on three predeclared high-dimensional/localized alternatives versus C057, without more than a 2 percentage-point worst-family power regression or a simultaneous null-bound failure. This is a target, not a result.

No SOTA, general learned-testing theorem beyond the assumptions above, Type-I calibration, safety or deployment claim is made.
