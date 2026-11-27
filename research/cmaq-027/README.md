# CMAQ-027 — Recoverability-Qualified Commitment

**Status:** dependency-free Rust evaluator and deterministic tests authored; exact-head formatting, tests and strict Clippy pending.  
**Relationship:** CMAQ-027 follows CMAQ-026 and connects decision-sufficient evidence acquisition to pre-action runtime assurance.  
**Claim ceiling:** finite scalar recoverability-contract checking. No verified physical certificate, reachability proof, formal safety guarantee, or SOTA claim.

## Research shift: recoverability rather than deployed-mode invariance

Classical invariance-style shielding requires the deployed behavior itself to remain inside a safe set. Recent work by Berrueta et al. instead studies whether an aggressive deployed mode can be switched back into a trusted recovery mode from a certified tube. Their theorem places the contraction requirement on the recovery mode, not on the unverified deployed behavior, while requiring a valid switch to remain reachable. Their robust condition explicitly bounds disturbances against the tube's contraction margin.

A separate 2026 multi-candidate Simplex study constructs context-dependent recovery regions and profiles that include admissible state/parameter cells and a maximum dwell time. This supports a useful Symthaea concept: a **recoverability-qualified commitment envelope**.

CMAQ-027 does not re-prove those results. It creates a finite execution contract for consuming a supplied recovery profile and candidate reachable-prefix bounds.

## Recovery profile v0.1

The profile binds:
- profile, plant dynamics, context scope, and recovery-controller identities;
- an external certificate-receipt digest;
- a scalar recoverability entry radius;
- a contraction ratio and per-step disturbance bound;
- a recovery target radius and maximum recovery steps;
- a maximum high-performance dwell time.

The deterministic recurrence is:

`e_{k+1} \le \lceil e_k \cdot \alpha / \beta \rceil + d`

where (0 \le \alpha < \beta), (d) is the declared disturbance bound, and (e_0) is the entry-radius bound. The profile is admitted only when:
1. the first disturbed recovery step remains inside the entry radius;
2. repeated recovery steps reach the target radius by the declared deadline.

**Important mathematical boundary:** this recurrence is a discrete scalar benchmark surrogate, not a direct transcription or proof of the continuous-time contraction theorem in Berrueta et al. For a continuous certificate with contraction rate \(\gamma\), an implementation must conservatively derive the discrete contraction factor and per-step disturbance accumulation for its actual sampling interval and disturbance model (for example, from a bound involving \(e^{-\gamma\Delta t}\)). The conversion and the underlying metric/tube proof must come from the independent verifier. CMAQ-027 only checks the supplied discrete profile arithmetic.

Integer arithmetic and ceiling division make the finite witness deterministic and conservative with respect to this declared discrete recurrence.

## Candidate reachable-prefix evidence

Each prefix binds four non-negative contributions:
- predicted deviation from the recovery reference;
- state-estimation error;
- transition-model error;
- disturbance/reachability error.

The evaluator sums these bounds with checked arithmetic. Prefix zero is the current state. Prefixes must be contiguous from zero.

The evaluator admits commitment only while every prefix through the chosen commitment horizon remains inside the recovery entry radius. It stops before the first predicted boundary crossing and never commits beyond the maximum dwell time. It also reports the last prefix that a simple forward-invariance-only baseline would admit, for comparison.

If the previous execution outcome is unknown, the evaluator returns BlockUntilReconciliation before examining a proposed new mutation. If the current state is already outside the recovery envelope, it returns NoCertifiedRecoveryFromCurrentState.

## Important trust boundary

The crate checks digest syntax and profile arithmetic. It does **not** authenticate the certificate receipt, verify a contraction metric, derive a reachability tube, prove the recovery controller's safety, or establish the physical plant's disturbance bound. Those artifacts must be produced by an independent verifier and authenticated at the evidence-ingestion boundary.

The scalar radius is a finite benchmark abstraction. Real high-dimensional certificates may use Riemannian metrics, ellipsoids, polytopes, Taylor models, interval reachability, or other set representations. A later adapter should preserve the proof artifact and its native set semantics rather than flattening every certificate to one scalar.

## Deterministic witnesses

1. A candidate path leaves a conventional forward-invariant safe set but remains inside the certified recovery entry envelope; qualified commitment can continue.
2. The first prefix outside the recovery envelope forces switching before that prefix.
3. Maximum dwell caps commitment even when the envelope margin remains positive.
4. A disturbance that exceeds the contraction margin rejects the recovery profile.
5. Recovery must reach its declared target inside the deadline.
6. Plant/context scope mismatch fails closed.
7. An unknown prior execution result blocks the next mutation.
8. Non-contiguous prefixes, current-state envelope violations, arithmetic overflow and malformed certificate digests are rejected.

## Benchmark plan

Matched controls:
- invariance-only shield;
- Simplex switch at fixed threshold;
- uncertainty-calibrated pause/reobserve/fallback gate;
- backup-controller predictive safety filter;
- recoverability-aware commitment with bounded reachability;
- direct high-performance policy without a recovery supervisor (negative control).

Report:
- high-performance utilization and mission utility;
- commitment horizon and recovery margin;
- missed-switch / envelope-exit counts;
- recovery success and recovery time under profile-covered disturbances;
- false intervention and unnecessary switch rate;
- profile coverage by context/plant parameter cell;
- switch latency, inference latency and total compute;
- worst-context results and independent scenario-cluster confidence intervals.

Start with deterministic toy witnesses, then high-fidelity simulation with injected friction, mass, latency, sensor and actuation mismatch, and only then hardware-in-the-loop and physical runs. Keep each evidence tier separate.

## Research references

- Berrueta et al., [Recoverability, Not Invariance: Certified Safety for Robots at the Limits of Performance](https://arxiv.org/abs/2611.23589), 17 Nov 2026.
- Jäger et al., [Context-dependent Runtime Construction of Safe Regions of Attraction from Design-Time Identification for Multi-Candidate Simplex Architectures](https://arxiv.org/abs/2610.35040), 26 Oct 2026.
- Hsu, Hu and Fisac, [Safety Filters for Dynamic Decision-Making](https://www.annualreviews.org/doi/10.1146/annurev-control-071723-102940), 2024.

## Next integration

Connect this contract to Symtropy's execution and recovery adapters. A production runtime must consume authenticated recovery-profile receipts, fresh context and plant-scope evidence, and actual measured state bounds. CMAQ-026 determines whether to acquire evidence; CMAQ-027 determines how long the system can safely continue before it must hand back to a trusted recovery mode.
