# ROB-CAPABILITY-RETENTION-001

Status: first executable descriptive comparison of quality-qualified productive work and terminal physical recoverability. The reference consumes externally qualified per-cluster work, reserve and option-ledger summaries. It does not calculate physical reserves, infer viability, issue actuation authority, or replace statistics.

Parent: [ROB-OPERATIONAL-UTILITY-001 #11607](https://github.com/Luminous-Dynamics/symthaea/issues/11607). Compose with:
- Symtropy [#4918 — future-option loss ledger](https://github.com/Luminous-Dynamics/symtropy/pull/4918).
- Symtropy [#4588 — observation-constrained future mobility reserve](https://github.com/Luminous-Dynamics/symtropy/issues/4588).
- Symtropy [#4549 — sequential mission disturbances and recovery reserve](https://github.com/Luminous-Dynamics/symtropy/issues/4549).
- Symtropy [#4783 — instrumented single-joint metrology](https://github.com/Luminous-Dynamics/symtropy/issues/4783).
- Symthaea [#11807 — paired independent-cluster inference](https://github.com/Luminous-Dynamics/symthaea/pull/11807).
- Symthaea [#11829 — shortcut / privileged-input probes](https://github.com/Luminous-Dynamics/symthaea/pull/11829).
- Symthaea [#11830 — provenance-group holdouts](https://github.com/Luminous-Dynamics/symthaea/issues/11830).

## Research question

Does an autonomy or productivity improvement deliver more accepted, safe work while preserving enough independently evidenced physical capability to continue, recover or complete the next mission?

A policy may increase accepted output by consuming battery reserve, thermal headroom, actuator authority, mechanical life, sensing margin or recovery routes. More current output is not automatically more sustainable capability. The comparison must report work and terminal reserve together.

AGIBOT's October 2026 Worken proposal measures active robots × autonomous working hours × output efficiency as a proposed productive-work indicator. Symthaea #11607 already owns Luminous's accepted quality-qualified work/hour contract. This reference is a complement: it adds the post-work capability/reserve vector instead of creating another productivity ledger.

## Work output contract

Every scheduled work item must end in exactly one terminal category:
- Accepted: independently inspected and quality-qualified output;
- Rejected;
- Reworked;
- Aborted;
- Held for review;
- Unknown.

All categories remain in the scheduled denominator. Only Accepted counts as productive output. The work-rate denominator is the declared scheduled available robot time, not powered-on time. Human intervention, electrical energy and compute energy are reported separately.

## Terminal capability vector

The reference preserves independent reserve dimensions, where applicable:
- battery energy reserve;
- thermal headroom;
- usable actuator torque headroom;
- qualified mechanical-condition indicator;
- sensor eligibility / calibration;
- qualified recovery-option count;
- remaining deadline slack;
- service-free operating allowance.

Every Required dimension has a frozen minimum floor. The measured lower bound is supplied by the owning evidence system with source receipt and, where required, calibration receipt. A value below its floor vetoes positive disposition. An Unknown value makes the comparison inconclusive, not zero.

A NotApplicable dimension is a distinct typed state with an explicit reason. It is not Unknown and cannot carry a measurement.

The reference does not combine joules, degrees, torque, route counts, hours or service debt into one scalar.

## Option ledger composition

The option-set record binds:
- exact option-loss ledger receipt;
- predecessor physical/reserve state;
- resulting physical/reserve state;
- starting qualified option-set digest;
- terminal qualified option-set digest;
- terminal qualified option count.

The starting option-set digest must match across the paired baseline/candidate cluster. The reference consumes these receipts and reports terminal option counts; it does not recompute option membership or future viability.

## Comparison and hard gates

Candidate and baseline bind the same task/work mix, robot configuration, evaluator, operating envelope, resource budget and reserve policy.

The report emits per independent cluster:
- accepted quality-qualified work and work rate;
- human intervention minutes;
- electrical and compute resource use;
- baseline and candidate terminal option counts;
- each reserve lower bound, candidate-minus-baseline delta and floor status;
- critical safety-event counts;
- evidence-completeness state.

Any observed candidate critical safety event vetoes positive disposition. Any qualified candidate reserve lower bound below a frozen floor vetoes positive disposition. Unknown/censored work, Unknown reserve evidence, Unknown option evidence or Unknown independence makes the comparison Inconclusive.

A reserve dimension may decline while staying above its floor; such a trade-off is reported rather than hidden. No average can compensate for a floor breach.

## Statistical boundary

The output is descriptive per independent cluster. This crate emits no p-values, confidence intervals, sample-size claims, significance decisions or SOTA rankings. It does not use repeated episodes or control ticks as independent statistical units. Confirmatory inference remains with #11807.

A passing synthetic reference is not an operational productivity gain, physical reserve measurement, mission qualification or safety certification.

## Running the reference

The package is dependency-free and isolated from the Symthaea root Cargo workspace:

    cargo fmt --manifest-path research/rob-capability-retention-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-capability-retention-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-capability-retention-001/reference/Cargo.toml --all-targets -- -D warnings

The exact-head Rust 1.96 GitHub Actions workflow is the qualification gate. Authored tests are not PASS until the dedicated workflow succeeds.
