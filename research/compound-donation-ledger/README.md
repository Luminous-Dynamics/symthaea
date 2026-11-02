# Compound donation ledger — reference core

This is a standalone, dependency-free Rust research crate for Issue #9894.
It is intentionally outside the main Cargo workspace until the mathematical
contract and API stabilize.

## Scope

The crate implements three bounded research primitives:

1. A fixed-point, append-only donation ledger. Each donor automatically
   reserves a precommitted fraction of its **own new adjusted-evidence
   increments**. Incoming donations cannot be re-donated. Reservations become
   source-tagged escrow tokens; each token records its precommitted destination
   ID and can be credited once (or split without exceeding the original
   amount) only to that destination after it is active and has positive weight.
2. A finite staircase calibrator. It evaluates the threshold-budget function
   A_pi(x) = sum_j pi_j K_j 1{x >= K_j} using integer arithmetic and a
   precommitted prior-mass budget. It also supplies a finite prefix of the
   universal dyadic prior pi_j = 1/(j(j+1)), K_j = 2^j, and an
   e-BH rank-geometric constructors: a family-size-specialized version and
   a horizon-independent candidate-weighted rank prior that works across
   family-size growth without changing earlier candidates' calibrators.


3. An exact persistent SCORE-style overshoot-refund account for one
   already-valid, time-monotone e-process. It tracks a capped cumulative refund,
   releases only the newly earned increment, stores exact rational numerators,
   and reports the effective test cost and threshold-crossing bit. It does not
   implement the global alpha-wealth/FDR spending algorithm.

The ledger receives already-calibrated, nondecreasing evidence values. It does
not compute likelihood ratios, test martingales, e-processes, null models,
p-values, or multiple-testing rejections.

## Accounting units

- Candidate weights are integer ticks on WEIGHT_SCALE; their sum may not
  exceed one.
- Evidence values are integer ticks on EVIDENCE_ONE; EVIDENCE_ONE represents
  the value one.
- Weighted capital is weight_ticks * evidence_ticks.
- Public scores are stored as weighted scores gamma_i * C_i, so the
  implementation never divides by a candidate weight.
- Reserve rates are fixed rational values in each candidate specification.
  The rate remainder is carried forward to avoid systematic per-round
  truncation. Outgoing capital is additionally capped by
  weight * min(adjusted_evidence, 1).

A reserve is generated deterministically on an evidence update. This makes
the donor's withdrawal path independent of the currently declared future
recipient roster. Each donor's destination slot is also committed in its
candidate specification before evidence is observed. If that destination is
outside the current horizon or remains dormant, the token stays in escrow.
Routing never rewrites a previous donor withdrawal or recipient credit.
This fixed-destination rule is the exact projective reference policy; adaptive
recipient selection remains a separate research problem.

## Event order and predictability

Activation events must be supplied before evidence updates for that candidate.
The ledger checks event order and monotone round numbers, but a serialized
event log cannot prove that a real-world arrival decision was genuinely made
before an observation. Production use must bind each round's roster to the
pre-observation arrival-policy receipt and source-event attestation; see
Symthaea Issue #9809. This crate does not authenticate timestamps or source
completeness.

## Replay and trust boundary

The writer's `verify_receipt` method replays through the same reference
transition engine; it is self-replay, not independent verification. The
separate `verifier/` crate uses its own BTreeMap state model and linear prefix
scan. It does not call the writer's ledger or transition helpers; it does
reuse the public event types as the receipt wire schema. Reservation receipts
and snapshots carry the precommitted destination ID, which the independent
verifier checks against the candidate policy.

Both crates are authored but not compiled or run in this environment. The
optimized writer can later use an augmented prefix-sum tree while the verifier
retains a linear scan. Neither crate is a digital signature, Merkle proof,
source attestation, or proof that the evidence itself is statistically valid.
Those layers require separate authenticated envelopes and independently
verified global-filtration evidence inputs.

## Run

From this directory, run the writer/reference crate and the independent
verifier separately:

~~~sh
cargo test --offline
cargo test --manifest-path verifier/Cargo.toml --offline
~~~

The writer/reference crate has no external dependencies. The verifier depends
only on the local reference crate. The main Symthaea workspace explicitly
excludes this research directory so the standalone crates can be built without
joining the production workspace. No repository CI result is claimed.

## Exploratory simulation harness

A NumPy-only, research-only script is provided at
`benchmarks/staircase_rank_geometric_exploratory.py`. It reproduces the
fixed-family Bernoulli likelihood-ratio screening described in RESEARCH.md,
including an independent mode and a balanced-negative-dependence null mode.
It reports fixed-time FDR, SupFDR, terminal power and probability of any false
discovery separately. Example commands:

~~~sh
python3 benchmarks/staircase_rank_geometric_exploratory.py --dependence independent
python3 benchmarks/staircase_rank_geometric_exploratory.py --dependence balanced_negative_null

# Fixed candidate weights across m=10,100,1000 prefixes
python3 benchmarks/projective_rank_horizon_transfer.py
~~~


The recorded run summary is
`benchmarks/staircase_rank_geometric_exploratory_run_2026-11-02.json`.
It stores the exact seed for each cell, per-method FDR/SupFDR/power/FWER
estimates, and one-sided Hoeffding upper bounds for SupFDR. The results were
generated by a Python reimplementation matching the committed harness logic;
the JSON states this execution limitation explicitly. They are exploratory
only and should be independently reproduced from the committed script before
being used as confirmatory evidence.

This is not production code and does not compute evidence processes from real
data. It is an exploratory simulation harness; frozen confirmatory manifests,
held-out seeds, broader dependence and alternative families are required
before performance claims.

## Horizon-independent rank calibration

The `StaircaseAdjuster::projective_weighted_e_bh_rank_prior` constructor is
independent of the active family size. For a fixed candidate weight
`weight_numerator / weight_denominator` and rational alpha, it places prior
mass `1 / ((j + 1)(j + 2))` at the dyadic rank `2^j`. The masses telescope
to one. The 64-point telescoping implementation covers every possible finite rank;
the remaining prior tail is unspent. A second hybrid constructor mixes half
of this prior with a uniform half over the 64 u64 rank scales, giving a
constant 256x worst-rank threshold overhead. The candidate's weight must be fixed as
part of the global summable weight sequence, not renormalized when the active
roster grows. See RESEARCH.md for the validity proof and O((log k)^2)
rank-threshold overhead.

The fixed-family `weighted_e_bh_rank_geometric` constructor can be more
powerful when the family size is genuinely known in advance, because it
spends more prior mass on that finite rank range. Do not recalculate that
constructor as the active family grows. Use the projective constructor when
horizon extension is a requirement.


The projective transfer runner uses a fixed global weight sequence
`gamma_i = 1 / max_family_size` and does not renormalize shorter prefixes.
Its recorded exploratory output is
`benchmarks/projective_rank_horizon_transfer_2026-11-02.json`. That run used
300 streams per cell, so its SupFDR point estimates have wide Monte Carlo
uncertainty; the main value of the experiment is the power-transfer comparison
under one horizon-coherent weight sequence.

## Persistent SCORE-style overshoot refund

For a precommitted rational level alpha and an already-valid time-monotone
e-process V_t, the Rust reference defines

~~~text
O_t = min(max(alpha * V_t - 1, 0), alpha)
C_t = alpha * V_t - O_t
~~~

Only the difference O_t - O_(t-1) is released on an update, so old overshoot
is never credited again. The writer uses a common exact rational denominator
for the refund and effective cost; it rejects evidence or round rollback and
commits each update atomically.

The separate verifier recomputes the update from the precommitted alpha,
previous evidence/refund state and new evidence value. A sequence verifier
starts at V_0=1 and independently checks all cumulative refunds, increments,
costs and decisions. It rejects a forged or repeated refund increment.

This is a primitive, not a completed online FDR algorithm. The process-level
inequalities are documented in RESEARCH.md. A global alpha-wealth theorem is
still required before released refunds can be spent across hypotheses. Do not
mix these alpha-wealth units with the weighted evidence-capital units of the
donation ledger without a joint proof.

