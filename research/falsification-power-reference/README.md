# Falsification power reference v0

A dependency-free Rust reference crate for the statistical power ceiling created by a fixed experimental design.

This crate is a narrow work item for Symthaea issue #12375. It does not add a model generator, experiment selector, posterior engine, model IR, evidence ledger, or production model-adequacy decision.

## What it computes

For each evaluator-declared alternative, provide its expected residual at every observation in the fixed design:

- residual = alternative-model prediction minus working-model prediction;
- sigma = known standard deviation of independent Gaussian observation noise;
- alpha = predeclared one-sided per-alternative false-alarm level.

The crate computes the alternative-specific Gaussian matched-filter noncentrality and best-case power. It then reports every alternative separately, the least-detectable alternative, and whether all alternatives meet the requested power target.

The power envelope refuses alternatives with different vector lengths, preventing comparisons with different observation counts. The v0 API does not receive observation IDs, timestamps, sensor identity or cost weights: the calling evaluator must bind identical design IDs/order/units and account for acquisition and execution costs separately.

## Assumptions and claim ceiling

This is an **evaluator-side oracle diagnostic**, not an operational adequacy test:

- independent observations;
- common, known Gaussian noise standard deviation;
- a fixed design and fixed stopping rule;
- known discrepancy direction for each declared alternative;
- one-sided matched-filter test aligned to that known alternative;
- no nuisance-parameter estimation, sequential peeking, multiplicity correction, correlated noise, censoring, missingness or sensor uncertainty.

Those assumptions make the calculation optimistic. When the model parameters or noise process must be estimated, the actual test can have lower power. Adaptive policies require an appropriate sequentially valid test or predeclared alpha-spending rule. Correlated and heteroskedastic noise require a covariance-aware extension and independent validation.

A failed target is reported as **underpowered**, not as evidence for or against the working model. Meeting the target only refers to the finite, declared alternative envelope. It does not prove that the envelope contains every scientifically material alternative.

## API

- **known_direction_power** — best-case power for one declared discrepancy.
- **evaluate_power_envelope** — retain alternative-specific rows and compute worst-case power.
- **minimum_replicates_for_power** — solve the constant-effect known-variance sample-size reference.

## Run

From the repository root:

    cargo fmt --manifest-path research/falsification-power-reference/Cargo.toml -- --check
    cargo test --manifest-path research/falsification-power-reference/Cargo.toml
    cargo clippy --manifest-path research/falsification-power-reference/Cargo.toml --all-targets -- -D warnings

No external Rust crates are required.

## Known-answer controls

The focused Rust suite includes 20 unit tests covering:
- null power equals nominal alpha;
- the 5% one-sided normal critical value;
- an external normal-tail known answer (three observations, delta=0.24, sigma=0.15, alpha=0.05);
- sign symmetry of two-sided power;
- monotonicity with additional informative observations;
- the minimum-replicate boundary for delta=0.24, sigma=0.15, alpha=0.05 and 80% target (three replicates);
- alternative-by-alternative reporting and worst-case identification;
- rejection of mismatched budgets, duplicate identifiers and malformed inputs;
- no finite sample-size claim for a zero discrepancy.

These are authored Rust tests. They must be compiled and executed on the exact candidate head before any test-pass claim is made.


The alpha level is per declared alternative. A campaign-level claim that searches across alternatives must separately control family-wise error or use a qualified joint/sequential test; this reference does not do that.


### Symmetry-discovery interpretation

For one proposed transformation or group action, take the working null to be **zero mean paired residual** under the declared transformation. Each registered break alternative supplies a predictable nonzero mean residual (d_{j,t}) for the next paired measurement. The evaluator commits the candidate/action identity, probe/observation identity, decision digest, discrepancy and known (sigma_t) before observing the residual (Y_t).

A fixed mixture over registered break alternatives can then accumulate evidence against the proposed symmetry. Crossing the anytime boundary is evidence to **falsify the proposed invariance under this observation model**. Not crossing is *unresolved*, not evidence that the transformation is a symmetry. The mixture prevents choosing the most favorable break alternative after looking at the data.

This does not automatically test a whole group: required generators/elements and probe strata must be declared, and the campaign must control multiplicity across candidate actions and probe families. A successful statistical test is still only a proposal/falsification result; it does not replace exact group closure, Lie-algebra, complete-task or physical step-map qualification, and it never authorizes quotient pruning.

The implementation uses binary64 arithmetic and is a research reference, not a formally verified floating-point e-process certificate. Conditional-model assumptions, numerical error, null calibration and actual adaptive action logs require independent review. Persistent or correlated nuisance effects are not covered merely because the caller uses a mixture; use the explicit persistent-noise model or a separately justified conditional law.


## Sequential null calibration harness

Run:

    cargo run --locked --manifest-path research/falsification-power-reference/Cargo.toml --example sequential_null_calibration

The deterministic harness simulates 10,000 independent null paths with 50 looks each. Both valid policies choose their discrepancy using only the previous residual and commit it before sampling the current observation. The known standard deviation changes according to a fixed action-dependent schedule. It reports the empirical probability of ever crossing the 5% boundary and 95% Wilson intervals for:
- one predictable alternative;
- a fixed two-alternative mixture.

An intentionally invalid negative control chooses its discrepancy after observing the current residual. It should cross frequently, demonstrating that the local commit/observe API cannot prevent a caller from leaking the outcome through an out-of-band path.

The harness is a deterministic software regression and a useful calibration warning light. It does not prove the martingale theorem, establish type-I error for real sensors or physical trajectories, or qualify persistent/correlated noise. The valid processes rely on the correctly specified simple conditional Gaussian null with predictable (d_t) and known (sigma_t). A real research campaign still needs independent null calibration, sealed alternatives, full action logs, and multiplicity control across candidate transformations and probe families.


## Finite-reference conformal e-process for simulator nulls

The crate also contains a separate finite-reference conformal process in `conformal`:

[
e_t = (K_t+1) \frac{S_t(Y_t)}{S_t(Y_t)+\sum_{j=1}^{K_t} S_t(\widetilde{Y}_{t,j})}.
]

Here (S_t) is a positive score selected using only information available before the current observation, and the (widetilde{Y}_{t,j}) are fresh conditional draws from the candidate null simulator. Under the null, the observed value and the reference draws must be conditionally exchangeable. Then the normalized observed score has conditional expectation (1/(K_t+1)), so the factor has conditional mean one. Multiplying these factors gives an e-process.

The implementation:
- commits candidate, observation, decision, score-profile and reference-batch identities before consuming the score values;
- rejects reused observation/reference-batch IDs, score/profile mismatches, invalid scores and reference-count mismatches;
- evaluates the ratio in log space;
- retains raw score inputs in a versioned receipt and independently replays the complete transcript.

This is a **finite-reference conformal e-process core**, inspired by the construction in SLICE. It is not a full SLICE reproduction: it does not train a neural ratio estimator, invert a continuous parameter-indexed family into confidence sequences, implement certified continuous-parameter outer approximations, or address every nuisance-parameter construction in the paper. It also cannot prove that the caller generated fresh independent simulations or applied the same score implementation to all (K+1) samples; the caller must preserve those execution records independently. Score quality affects power and detection delay, not validity under the stated exchangeability assumptions.

For symmetry testing, one candidate transformation defines the null simulator for paired residuals or another declared symmetry diagnostic. A conformal e-process can falsify that candidate under the simulator's conditional observation model. Non-rejection remains unresolved; it is not an exact symmetry proof, physical qualification, or permission to prune. Testing many candidate transformations requires a declared family-wise or simultaneous-confidence-set construction; this single-candidate class does not silently provide multiplicity control.

### Current sequential-inference comparators

- **Fixed-horizon terminal test:** the matched Neyman–Pearson or application-specific test at a frozen horizon.
- **Induced anytime-valid terminal test:** Koning & van Meer show an arbitrary valid terminal test can be sequentialized while matching its terminal-horizon decision, so an e-process is not the only anytime-valid route.
- **Gaussian LR e-process:** direct likelihood-ratio reference, with a transparent conditional Gaussian assumption.
- **Finite-reference conformal e-process:** simulator-backed candidate nulls without tractable likelihoods; reference simulator calls are part of the acquisition cost.
- **Faithful SLICE implementation:** neural ratio score, fresh null references, candidate-grid inversion and any continuous-grid certification required by the evaluated setting. This remains a future baseline, not an implemented feature here.
- **Invalid post-hoc leakage:** explicit negative control showing that selecting a discrepancy after observing the current outcome destroys the predictable-betting argument.

### Expanded calibration output

The deterministic calibration example now evaluates three valid null procedures (single Gaussian, fixed Gaussian mixture, and finite-reference conformal) plus the invalid post-hoc leakage control. It reports empirical ever-crossing rates and Wilson 95% intervals for 10,000 seeded paths with 50 looks. The conformal lane uses five fresh standard-normal reference draws per look and a positive history-adaptive score. It is a reproducible regression diagnostic, not a scientific type-I calibration for real systems.
