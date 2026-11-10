# Vocal correction trajectory evaluator (VCCR-001)

A dependency-free Rust evaluator for checking whether a sequence of vocal corrections is bounded, provenance-consistent, uncertainty-aware and convergent.

## Why this exists

An individual edit can look successful while a feedback loop oscillates, overshoots, stalls, damages a protected property or continues mutating after it has already converged. VCCR-001 evaluates the trajectory across successive renders.

It does not render audio, choose corrective actions, estimate acoustic metrics or generate uncertainty intervals.

## Outcomes

- **CONVERGED** — every target interval is wholly inside its accepted range and every protected metric interval is wholly inside its collateral tolerance.
- **NEEDS_CORRECTION** — measurements are decisive, protected constraints hold, and at least one target remains outside its accepted range while the correction sequence makes bounded progress.
- **INDETERMINATE** — an interval overlaps a decision boundary. Further mutation is not allowed from that evidence.
- **NONCOMPLIANT** — a protected limit is definitely violated, an action exceeds its declared bound, or a target regresses beyond its declared allowance.
- **STALLED** — a correction produces no new output or fails every target's minimum-progress rule.
- **ITERATION_LIMIT_REACHED** — the correction budget is exhausted without verified convergence.

A trajectory with NEEDS_CORRECTION is not a pass. It means only that the supplied partial trace is internally admissible so far.

## Per-iteration contract

Iteration zero is the frozen baseline:
- no parent output;
- no action digest;
- zero action norm;
- output digest exactly matches the baseline digest in the specification.

Each correction iteration:
- has a sequential index;
- references the exact previous output digest;
- carries a non-empty action digest and positive finite action norm;
- stays within the declared maximum action norm;
- binds the same source case and frozen context digest;
- binds a new output digest and complete metric family.

The frozen context digest must commit to the source case, lyrics, pronunciation plan, score/alignment, voice identity and authorization, renderer and voicebank artifacts, analyzer artifacts, preprocessing, calibration profile and other fixed inputs. A changed context starts a new trajectory.

## Target and collateral rules

Every target has an accepted interval, minimum required progress and maximum allowed regression.

For a target measurement interval:
- wholly inside the accepted interval: compliant;
- wholly outside: needs correction;
- overlapping a boundary: indeterminate.

The distance-to-target is conservative: for a measurement wholly below the accepted range, it uses the lower endpoint; for a measurement wholly above the accepted range, it uses the upper endpoint. Every unsatisfied target must stay within its declared regression allowance, and at least one unsatisfied target must improve by its declared minimum or become compliant. Previously compliant targets may never regress.

Every protected metric must remain wholly inside its symmetric absolute tolerance on every iteration. A definite violation is noncompliant; a boundary overlap is indeterminate. Neither is hidden by another metric's improvement.

The evaluator stops at the first converged, indeterminate, noncompliant or stalled iteration. Supplying additional mutation evidence after any terminal outcome is rejected.

## Uncertainty and provenance

Every iteration requires exactly one observation for every declared target and collateral metric. Each interval binds:
- metric ID and unit;
- finite, ordered lower and upper bounds;
- interval-family ID and family-wise coverage declaration;
- uncertainty method ID;
- calibration receipt ID.

All metrics in one iteration must share the same interval-family ID and coverage declaration. Coverage must meet the trajectory's minimum. The evaluator checks metadata consistency; it cannot prove the interval producer achieves its claimed statistical coverage.

Every output and action digest is required to be a 64-character lowercase hexadecimal digest. The frozen-context digest is fixed across the complete trajectory.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-correction-trajectory-eval/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-correction-trajectory-eval/Cargo.toml
~~~

## Integration

- #11402 supplies the planned versioned acoustic observations.
- #11420 calibrates no-op collateral tolerances.
- #11456 / PR #11461 supplies interval-aware measurement decisions.
- #11401 and VELB-001 measure target response and edit locality.
- #11356 will own the renderer-neutral local patch and preservation receipt.

## Claim boundary

Synthetic tests qualify only trajectory validation logic. A passing receipt does not prove that the renderer or controller is good, that the upstream intervals are calibrated, or that the audio sounds natural. Real closed-loop claims require held-out licensed/consented singing material, independent analyzers, blinded listening and matched baseline controllers.
