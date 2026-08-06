//! Phase 4 characterization — what is the scalar that gates robotics motor safety?
//!
//! First empirical deliverable of `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`.
//! Everything that program landed before this was structural (a contract, a trait, a rename,
//! two deletions); nothing had actually *measured* the system. This does.
//!
//! Governing protocol: `SYMTHAEA_PHASE4_CHARACTERIZATION_PROTOCOL_2026-07-29.md`.
//!
//! ## The question
//!
//! `EmbodimentBridge::step(&mut self, thought_hv, dt, phi: f64)` hands every one of the 15
//! `EmbodimentBridge` robotics platforms a bare `f64` named `phi`, and
//! `MotorSafetyLevel::from_phi` thresholds it into Green/Yellow/Orange/Red — which physically
//! gate torque scaling, admittance compliance gain, and grip-force caps. But that argument is
//! **not raw Φ**. On the `cycle(&str)` path this harness drives
//! (`cycle.rs:313` → `integration.rs:1150` → `integration.rs:960-983`) it is:
//!
//! ```text
//! phi_input = max(unified_psi, spectral_boost, structural_boost)
//!   unified_psi      — live Ψ, recomputed EVERY cycle
//!   spectral_boost   — cached spectral-MIP Φ × 0.6, refreshed every 67 cycles
//!   structural_boost — cached (micro+meso+macro)/3 × 0.5, also every 67 cycles
//! ```
//!
//! then fed as one of ten inputs to `ConsciousnessMasterEquation`, then `max`'d against a
//! cached `ConsciousnessEquationV2` reading × 0.8. So on ~98.5% of cycles no refresh occurs and
//! whichever of a *stale* Φ or the *live* Ψ is larger drives the composite.
//!
//! **This harness answers, with real numbers: which component actually dominates, what the
//! composite's distribution looks like, and how often each safety tier is occupied.**
//!
//! ## Scope: measurement only
//!
//! Read-only. No production code path is modified; this reads already-public accessors. Run:
//!
//! ```text
//! cargo run --release --example safety_signal_characterization           # 3-arm default
//! cargo run --release --example safety_signal_characterization -- --bands 3
//! ```
//!
//! ## Three limitations, disclosed rather than papered over
//!
//! 1. ~~**`consciousness_level` from telemetry is NOT bit-identical to the gate value.**~~
//!    **RESOLVED 2026-07-29** — `CognitiveLoopService::safety_gate_phi()` was added (read-only,
//!    no behavior change) and this harness now reports the true gate value and its tiers
//!    directly, alongside telemetry so the two can be compared. The original limitation text is
//!    kept below because it explains *why* they differ and why the telemetry rows are still
//!    printed as diagnostics rather than removed.
//!
//!    Original text:
//!    `integration.rs:1274` applies a `social_mod` (±5%) *after* the carryover write at `:1142`,
//!    and on non-MCE cycles `:1265` returns `carryover × 0.98` without updating carryover. So
//!    tier occupancy near the 0.3/0.6 boundaries is biased by up to ~5% plus a decay tail.
//!    (That accessor has since been added, which is why this limitation is marked resolved
//!    above; the telemetry rows remain as a diagnostic contrast.)
//! 2. **`metadata.structural.*_phi` are scale-boosted** by `(1 + scale_boost)`
//!    (`cycle_phase_feedback.rs:1205-1210`) when HierarchicalCfC is on. Exact only when it's
//!    off, which is the default config used here.
//! 3. **`unified_psi` is autoregressive.** `cycle_extracted.rs:588-592` computes it as
//!    `0.65×component_psi + 0.35×phi_level` — partly a function of the *previous* cycle's
//!    `consciousness_level`. So "Ψ dominates" partly means "the composite feeds itself"; read
//!    the argmax tally with that feedback loop in mind rather than as clean attribution.
//!
//! ## ⚠️ THE BIGGEST LIMITATION, promoted from a footnote (2026-07-29)
//!
//! The gate field has **two writers with unrelated formulas**, and this harness only exercises
//! one of them:
//!
//! | Writer | Entry point | Formula |
//! |---|---|---|
//! | `integration.rs:1150` | `cycle(&str)` — **what this harness drives** | the Φ/Ψ composite above |
//! | `helpers/mod.rs:419` | `cycle_with_hv()` | `compute_consciousness_level(prediction_confidence, coherence, flow_intensity, pattern_confidence)` — no Φ, no Ψ, no spectral MIP |
//!
//! So every number this harness produces characterizes **the text-path writer**, not "the
//! embodiment safety input" in general. A system interleaving both entry points gets whichever
//! wrote last. The harness now prints `gate writer provenance` per regime so this limitation is
//! visible in the OUTPUT rather than only here — and so a future arm driving `cycle_with_hv()`
//! must show a different writer, or it is not exercising the path it claims to.
//!
//! See `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`, Phase 4 correction box.
//!
//! ## Two confounds in THIS harness, disclosed
//!
//! 4. **Regimes run sequentially on ONE service instance**, so regime 2 and 3 inherit all state
//!    from regime 1 — the boosts in particular are caches that never reset. Regime ordering is
//!    therefore part of the result, not controlled for. (`exp_loop_ablation.rs --section e2` has
//!    the same design.) A cleaner future version would construct a fresh service per regime, at
//!    the cost of losing the warm-up the later regimes currently inherit.
//!
//!    *(Scope note added 2026-07-30: this remains an accurate description of arms A and B.
//!    Arm C, and the `--bands` reference condition, construct a fresh service per regime and
//!    are not subject to it. The confound is not resolved — it is deliberately retained in
//!    A/B as the order-counterbalance control required by protocol §3.)*
//! 5. **The argmax tie-breaks to the earlier branch** (`psi`, then `spectral`, then
//!    `structural`). This matters enormously in practice: when `spectral_boost` and
//!    `structural_boost` are BOTH clamped to exactly 1.0 — which the 2026-07-28 run shows is the
//!    common case — structural can never be credited even though it is *co*-maximal. So a
//!    "structural 0.0%" reading does **not** mean structural is ignored; it means **structural
//!    never strictly exceeded spectral**. Do not report the former.
//!
//! ## Why ≥400 cycles per regime
//!
//! The 67-cycle refresh must fire several times or both boosts stay pinned at 0.0 and the
//! attribution degenerates to "Ψ always wins" — a measurement artifact, not a finding.
//!
//! ## Three arms (added 2026-07-29) — falsifying this harness's own first result
//!
//! The 2026-07-28 run reported: repetitive → spectral mean 0.3371, varied → 0.9975,
//! alarming → 1.0000, and concluded that the safety gate goes constant under *alarming* input.
//! **That conclusion did not survive scrutiny of confound 4 above.** Saturation increased
//! monotonically with *run order*, and run order was perfectly confounded with regime (one
//! shared service, caches never reset). "Alarming input saturates the gate" was
//! indistinguishable from "800 cycles of any input saturates the gate."
//!
//! So this now runs three arms:
//!
//! - **A: shared/forward** — repetitive, varied, alarming on one service. Reproduces the
//!   2026-07-28 run exactly; also a determinism check, since the config is seeded and
//!   `async_training` is off, so arm A must match the committed numbers or something is
//!   non-deterministic that shouldn't be.
//! - **B: shared/reverse** — alarming, varied, repetitive on one service. **The decisive test.**
//!   If `alarming` still saturates when it runs *first* on cold caches, the regime effect is
//!   real. If `repetitive` saturates when it runs *last* on warm caches, the original headline
//!   was a cache-warming artifact.
//! - **C: fresh-per-regime** — each regime on its own newly-constructed service. Removes the
//!   confound entirely rather than merely testing for it; the cleanest per-regime numbers,
//!   at the cost of losing inherited warm-up.
//!
//! Each regime also reports a **first-half vs second-half** split of `spectral_boost`, which
//! shows warming *within* a regime directly rather than by inference across regimes.
//!
//! ## The F2 exclusion rule (added 2026-07-30, protocol Amendment 2)
//!
//! Amendment 2's finding **F1** established that `carryover.history.consciousness_level`
//! initialises to `0.05` — a *prior*, commented in `carryover.rs` as "Floor: prevents fully
//! unconscious cold-start" — and stays there until the first real write.
//! `MotorSafetyLevel::from_phi(0.05)` is **Red**, so a freshly constructed service reports Red
//! tier occupancy for the whole cold-start window from a value that was never a measurement.
//!
//! Finding **F2** then retroactively qualified the 2026-07-29 true-gate data: the `min=0.0500`
//! that appeared five times in it was this floor, not a measured distribution minimum. The
//! protocol consequence is **binding on every run**:
//!
//! > every run under this protocol must **exclude or separately report** cycles preceding the
//! > first non-`ColdStartFloor` write. Mixing prior-driven and measurement-driven cycles in one
//! > distribution is not admissible.
//!
//! This harness takes the *separately report* option, deliberately: every metric is printed
//! twice — **(a) ALL CYCLES** and **(b) POST-FIRST-WRITE** — because the difference between
//! them is itself the size of the F2 artifact, and silently replacing (a) with (b) would hide
//! how large it was. Only **(b)** is admissible as a measurement distribution.
//!
//! Detection is mechanical, via `safety_gate_provenance()`: the first cycle whose writer is not
//! `GateWriter::ColdStartFloor`. Its index and the excluded-cycle count are printed per regime.
//!
//! Finding **F3** bears on how to read the staleness row below: the first real write lands at
//! cycle 5 (consumed at 6), so startup urgency is Critical rather than Normal, and the
//! observed produced→consumed lag is ~5 cycles because the MCE's own 5/10/20-cycle cadence
//! dominates it. **The gate routinely acts on a value several cycles old by design.** A
//! staleness figure here is therefore not a defect indicator on its own, and any staleness
//! criterion must exceed both the 67-cycle boost refresh and the MCE write cadence.
//!
//! ## Per-metric reproducibility bands: `--bands N` (added 2026-07-30, protocol A1.3)
//!
//! The frozen §3 "~8%" reproducibility floor was estimated from n=3 runs of **one** metric
//! (`consciousness_level` mean). A1.3 forbids transferring it across metric types — "8% of a
//! percentage" is ambiguous between percentage points and relative change — and fixes these
//! rules, implemented here exactly as written:
//!
//! | Metric type | Band unit | Source |
//! |---|---|---|
//! | Continuous scalars (means of `phi_input`, gate value, components) | **relative**, as % of the A1 mean | re-estimated |
//! | Proportions (tier occupancy, dominance share, saturation fraction) | **percentage points** | re-estimated, *never* inherited from the scalar band |
//! | Counts (tier transitions, invalid cycles) | **absolute count** | re-estimated |
//!
//! The denominator is **always the A1 reference arm** (the present composite) — never
//! each-arm-relative — so between-arm comparisons stay commensurable.
//!
//! `--bands N` (default 3, hard minimum 3) runs **only** the reference condition, N times, on a
//! **fresh service each time**, and prints a band table per regime with the correct unit per
//! metric. Band = the observed spread across those N runs, reported both as range (max − min)
//! and as max deviation from the mean. A1.3's marginal-exceedance rules (≤band = non-finding;
//! ≤2× band = candidate requiring replication; >2× band = effect) are restated in the output
//! footer so they can be applied mechanically.
//!
//! **Judgement call, stated rather than buried:** "the A1 reference condition" is operationalised
//! here as the *fresh-service-per-regime* run of the present composite — i.e. arm C's instance
//! policy, which protocol §3 calls non-negotiable — replicated N times, per regime. Bands are
//! reported per regime because every metric in this harness is computed per regime, so a
//! regime-pooled band would not be comparable to any figure a decision criterion actually uses.
//! Arms A and B (the shared-instance order controls) are deliberately *not* replicated by this
//! mode: their run-order confound is the thing they exist to expose, so spread across their
//! replicates would not estimate reproducibility noise.
//!
//! Bands are printed for both the all-cycles and post-first-write subsets, for the same reason
//! the metrics themselves are: the post-first-write band is the admissible one, and showing
//! both makes the F2 artifact's contribution to apparent noise visible.

use symthaea::cognitive_loop::types::GateWriter;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::embodiment::MotorSafetyLevel;

const GENESIS: &str = "safety-signal-characterization-2026-07-28";
/// ≥400 so the 67-cycle spectral/structural refresh fires ~6×. See module docs.
const CYCLES_PER_REGIME: usize = 400;

/// Smoke-test override for [`CYCLES_PER_REGIME`], set only by `--cycles N`.
///
/// **A run below `CYCLES_PER_REGIME` is protocol-INADMISSIBLE** and the output says so on
/// every affected table. It exists solely to exercise the band arithmetic and table
/// formatting cheaply (the independent verifier of the `--bands` implementation noted the
/// code path had been compile-verified but never executed, and recommended exactly this).
/// Below 400 the 67-cycle refresh fires fewer than 5 times, so both boosts stay near 0.0 and
/// component attribution degenerates — see the module docs. Never report numbers from such a
/// run.
static CYCLE_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Effective cycles per regime for this process.
fn cycles_per_regime() -> usize {
    let o = CYCLE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if o == 0 { CYCLES_PER_REGIME } else { o }
}

/// True when running below the protocol floor, i.e. output is smoke-test only.
fn is_inadmissible_run() -> bool {
    cycles_per_regime() < CYCLES_PER_REGIME
}
/// A1.3 requires ≥3 reference runs to estimate a band. 3 estimates a floor; it does not
/// characterize a noise distribution — that limit is inherited from the protocol, not chosen.
const MIN_BAND_RUNS: usize = 3;
const DEFAULT_BAND_RUNS: usize = MIN_BAND_RUNS;
/// The clamp ceiling the components are `.clamp(0.0, 1.0)`-ed to. A value at or above this is
/// "saturated" for the purposes of the saturation-fraction metric. Note the gate field itself
/// is written **unclamped** (protocol §1), which is why the test is `>=` and not `==`.
const SATURATION_CEILING: f64 = 1.0;

fn base_config() -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::default();
    c.genesis_phrase = Some(GENESIS.to_string());
    c.async_training = false;
    c
}

// ── per-cycle record ─────────────────────────────────────────────────────────────────────

/// Which of the three `max` branches produced `phi_input` on a given cycle.
///
/// Tie-breaks to the earlier branch — see module-docs limitation 5. A `Structural` count of
/// zero means structural never *strictly exceeded* spectral, not that it was ignored.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dominant {
    Psi,
    Spectral,
    Structural,
}

/// One cycle's full observation. Kept as records rather than parallel vectors so that the F2
/// subset split (all cycles vs. post-first-write) is a single slice operation and cannot
/// desynchronise one metric from another.
#[derive(Clone, Copy)]
struct CycleRecord {
    /// Regime-local index, 0-based.
    local_index: usize,
    /// `stats.total_cycles` observed after this cycle completed — the harness's absolute clock,
    /// i.e. the *consumed* side of the produced/consumed join.
    absolute_cycle: usize,
    phi_input: f64,
    psi: f64,
    spectral: f64,
    structural: f64,
    consciousness_level: f64,
    /// The TRUE gate value: exactly what `EmbodimentBridge::step()` receives.
    gate: f64,
    /// Which formula produced the value consumed on this cycle.
    writer: GateWriter,
    /// The writer's own recorded cycle index — the *produced* side of the join.
    written_at: usize,
    dominant: Dominant,
    /// Wall-clock microseconds this cycle took (Amendment 5 / A5.0 covariate).
    ///
    /// Recorded because the Jul-28 E1 re-run found telemetry `consciousness_level` splitting
    /// into two sharp regimes (0.064 vs 0.646) that tracked cycle wall-time, not the ablation
    /// under test — an artifact invisible at the time precisely because nothing recorded this
    /// number. It is a covariate, not a metric: nothing in the exceedance rules reads it. Its
    /// job is to make a load artifact *falsifiable* in any future run rather than discoverable
    /// only three hours in.
    cycle_micros: f64,
}

// ── distribution / tier helpers ──────────────────────────────────────────────────────────

fn mean_of(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn dist(label: &str, xs: &[f64]) {
    if xs.is_empty() {
        println!("    {label:<34} (no samples)");
        return;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f64| s[((s.len() - 1) as f64 * f).round() as usize];
    println!(
        "    {label:<34} n={:<5} min={:.4} q25={:.4} med={:.4} q75={:.4} max={:.4} mean={:.4}",
        s.len(),
        s[0],
        q(0.25),
        q(0.50),
        q(0.75),
        s[s.len() - 1],
        mean_of(&s)
    );
}

/// Tier occupancy and transition count, computed by calling the REAL production classifier
/// rather than a local copy of the 0.6/0.3/0.1 thresholds — so this measurement cannot
/// silently drift from what ships. Every tier figure in this file routes through here.
#[derive(Default, Clone, Copy)]
struct TierStats {
    green: usize,
    yellow: usize,
    orange: usize,
    red: usize,
    transitions: usize,
    n: usize,
}

impl TierStats {
    fn pct(&self, n: usize) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        100.0 * n as f64 / self.n as f64
    }
}

fn tier_stats(xs: &[f64]) -> TierStats {
    let mut t = TierStats {
        n: xs.len(),
        ..Default::default()
    };
    let mut prev: Option<MotorSafetyLevel> = None;
    for &x in xs {
        let level = MotorSafetyLevel::from_phi(x);
        match level {
            MotorSafetyLevel::Green => t.green += 1,
            MotorSafetyLevel::Yellow => t.yellow += 1,
            MotorSafetyLevel::Orange => t.orange += 1,
            MotorSafetyLevel::Red => t.red += 1,
        }
        if prev.is_some_and(|p| p != level) {
            t.transitions += 1;
        }
        prev = Some(level);
    }
    t
}

fn tier_report(label: &str, xs: &[f64]) {
    if xs.is_empty() {
        println!("    {label:<34} (no samples)");
        return;
    }
    let t = tier_stats(xs);
    println!(
        "    {label:<34} Green {:.1}% | Yellow {:.1}% | Orange {:.1}% | Red {:.1}%  ({} transitions)",
        t.pct(t.green),
        t.pct(t.yellow),
        t.pct(t.orange),
        t.pct(t.red),
        t.transitions
    );
}

/// Cycles whose gate value is in the set the protocol calls invalid: non-finite, or negative.
///
/// `from_phi` fail-safes non-finite values to Red explicitly; negatives merely fall through the
/// ladder to Red. Both are counted here because protocol §1 treats them as one category, but
/// they are not the same mechanism.
fn invalid_count(xs: &[f64]) -> usize {
    xs.iter().filter(|x| !x.is_finite() || **x < 0.0).count()
}

fn saturation_count(xs: &[f64]) -> usize {
    xs.iter()
        .filter(|x| x.is_finite() && **x >= SATURATION_CEILING)
        .count()
}

// ── metric vector (the unit of band estimation) ──────────────────────────────────────────

/// Fixes how a reproducibility band for a metric must be expressed. Protocol A1.3: continuous
/// scalars are RELATIVE (% of the A1 mean); proportions are in PERCENTAGE POINTS and must be
/// re-estimated rather than inherited from the scalar band; counts are ABSOLUTE.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MetricKind {
    Scalar,
    Proportion,
    Count,
}

impl MetricKind {
    fn unit(self) -> &'static str {
        match self {
            MetricKind::Scalar => "rel-%",
            MetricKind::Proportion => "pp",
            MetricKind::Count => "count",
        }
    }
}

struct Metric {
    name: &'static str,
    kind: MetricKind,
    /// Scalars in native units; proportions already in percentage points (0..100); counts as
    /// whole numbers held in an `f64` so one table can carry all three.
    value: f64,
}

/// The canonical, ordered metric vector for one subset of cycles. Order is stable across calls
/// so band estimation can index positionally.
///
/// An empty subset yields `NaN` values rather than being skipped, so the table keeps its shape
/// and an empty post-first-write subset is visible as `n/a` instead of vanishing.
fn metrics(records: &[CycleRecord]) -> Vec<Metric> {
    let col = |f: fn(&CycleRecord) -> f64| -> Vec<f64> { records.iter().map(f).collect() };
    let phi_input = col(|r| r.phi_input);
    let gate = col(|r| r.gate);
    let n = records.len();
    let share = |k: usize| {
        if n == 0 {
            f64::NAN
        } else {
            100.0 * k as f64 / n as f64
        }
    };
    let dom = |d: Dominant| share(records.iter().filter(|r| r.dominant == d).count());
    let t = tier_stats(&gate);

    vec![
        Metric {
            name: "mean phi_input",
            kind: MetricKind::Scalar,
            value: mean_of(&phi_input),
        },
        Metric {
            name: "mean unified_psi",
            kind: MetricKind::Scalar,
            value: mean_of(&col(|r| r.psi)),
        },
        Metric {
            name: "mean spectral_boost",
            kind: MetricKind::Scalar,
            value: mean_of(&col(|r| r.spectral)),
        },
        Metric {
            name: "mean structural_boost",
            kind: MetricKind::Scalar,
            value: mean_of(&col(|r| r.structural)),
        },
        Metric {
            name: "mean consciousness_level",
            kind: MetricKind::Scalar,
            value: mean_of(&col(|r| r.consciousness_level)),
        },
        Metric {
            name: "mean safety_gate_phi",
            kind: MetricKind::Scalar,
            value: mean_of(&gate),
        },
        Metric {
            name: "gate tier Green",
            kind: MetricKind::Proportion,
            value: t.pct(t.green),
        },
        Metric {
            name: "gate tier Yellow",
            kind: MetricKind::Proportion,
            value: t.pct(t.yellow),
        },
        Metric {
            name: "gate tier Orange",
            kind: MetricKind::Proportion,
            value: t.pct(t.orange),
        },
        Metric {
            name: "gate tier Red",
            kind: MetricKind::Proportion,
            value: t.pct(t.red),
        },
        Metric {
            name: "dominance Psi",
            kind: MetricKind::Proportion,
            value: dom(Dominant::Psi),
        },
        Metric {
            name: "dominance spectral",
            kind: MetricKind::Proportion,
            value: dom(Dominant::Spectral),
        },
        Metric {
            name: "dominance structural",
            kind: MetricKind::Proportion,
            value: dom(Dominant::Structural),
        },
        Metric {
            name: "phi_input saturation",
            kind: MetricKind::Proportion,
            value: share(saturation_count(&phi_input)),
        },
        Metric {
            name: "gate saturation",
            kind: MetricKind::Proportion,
            value: share(saturation_count(&gate)),
        },
        Metric {
            name: "gate tier transitions",
            kind: MetricKind::Count,
            value: t.transitions as f64,
        },
        Metric {
            name: "invalid gate cycles",
            kind: MetricKind::Count,
            value: invalid_count(&gate) as f64,
        },
    ]
}

// ── F2: prior-driven vs measurement-driven cycles ────────────────────────────────────────

/// Index of the first record whose gate provenance writer is NOT `ColdStartFloor`.
///
/// `None` means the gate was never written during this regime: every cycle is prior-driven and
/// the entire subset is inadmissible as a measurement distribution under Amendment 2.
fn first_real_write(records: &[CycleRecord]) -> Option<usize> {
    records
        .iter()
        .position(|r| r.writer != GateWriter::ColdStartFloor)
}

/// The admissible subset: cycles from the first non-`ColdStartFloor` write onward.
fn post_first_write(records: &[CycleRecord]) -> &[CycleRecord] {
    match first_real_write(records) {
        Some(i) => &records[i..],
        None => &[],
    }
}

fn print_f2_line(records: &[CycleRecord]) {
    match first_real_write(records) {
        Some(i) => {
            let r = &records[i];
            println!(
                "    {:<34} regime-local cycle {} (abs total_cycles={}, writer written_at={}, \
                 writer={}); {} prior-driven cycles excluded (n {} -> {})",
                "F2 first real write",
                r.local_index,
                r.absolute_cycle,
                r.written_at,
                r.writer.label(),
                i,
                records.len(),
                records.len() - i
            );
        }
        None => {
            println!(
                "    {:<34} NEVER — all {} cycles are still on the {} prior (0.05). The whole \
                 subset is INADMISSIBLE as a measurement distribution (Amendment 2, F2).",
                "F2 first real write",
                records.len(),
                GateWriter::ColdStartFloor.label()
            );
        }
    }
}

// ── driving one regime ───────────────────────────────────────────────────────────────────

type Regime = (&'static str, Box<dyn Fn(usize) -> &'static str>);

/// Drive one regime on the given service, collecting per-cycle records. Pure data collection —
/// no printing — so the same records can be reported in full (default mode) or reduced to a
/// metric vector (bands mode) without re-running anything.
fn collect_regime(
    svc: &mut CognitiveLoopService,
    make_input: &dyn Fn(usize) -> &'static str,
) -> Vec<CycleRecord> {
    let mut records = Vec::with_capacity(cycles_per_regime());

    for i in 0..cycles_per_regime() {
        let t0 = std::time::Instant::now();
        let r = svc.cycle(make_input(i));
        let cycle_micros = t0.elapsed().as_secs_f64() * 1e6;

        // Reconstruct the three components exactly as integration.rs:960-983 does.
        let psi = svc.stats().unified_psi as f64;
        let spectral = svc
            .consciousness_snapshot()
            .spectral_mip_phi
            .map(|p| (p * 0.6).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let m = &r.metadata.structural;
        let structural =
            ((m.structural_micro_phi + m.structural_meso_phi + m.structural_macro_phi) / 3.0 * 0.5)
                .clamp(0.0, 1.0);

        let phi_input = psi.max(spectral).max(structural);
        // Ties resolve to the earlier branch. That matters: when all three are 0.0 (before
        // the first 67-cycle refresh) every cycle counts as Psi-dominant, which is correct
        // for what the `max` actually returns but should not be read as Psi "winning".
        let dominant = if phi_input == psi {
            Dominant::Psi
        } else if phi_input == spectral {
            Dominant::Spectral
        } else {
            Dominant::Structural
        };

        // WHICH writer produced the value we just consumed, and how stale it is.
        // This harness drives cycle(&str) only, so it should report 100%
        // text-composite -- i.e. it makes its own scope limitation visible in the
        // output rather than only in the module docs. A future arm driving
        // cycle_with_hv() must show a different writer, or the experiment is not
        // exercising the path it claims to.
        //
        // It is also the F2 exclusion mechanism (Amendment 2): a `ColdStartFloor` writer
        // means the consumed value is the 0.05 prior, not a measurement.
        let (writer, written_at) = svc.safety_gate_provenance();

        records.push(CycleRecord {
            local_index: i,
            absolute_cycle: svc.stats().total_cycles,
            phi_input,
            psi,
            spectral,
            structural,
            consciousness_level: r.metadata.consciousness.consciousness_level,
            // The TRUE gate value: exactly what EmbodimentBridge::step() receives and
            // MotorSafetyLevel::from_phi() thresholds. Distinct from the telemetry value
            // above, which has social_mod and a 0.98 non-MCE decay applied after the gate
            // field is written.
            gate: svc.safety_gate_phi(),
            writer,
            written_at,
            dominant,
            cycle_micros,
        });
    }

    records
}

/// Print the full metric report for one subset of a regime's cycles.
fn report_subset(title: &str, records: &[CycleRecord]) {
    println!("    ── {title} (n={}) ──", records.len());
    if records.is_empty() {
        println!("    (no samples in this subset)");
        println!();
        return;
    }

    let col = |f: fn(&CycleRecord) -> f64| -> Vec<f64> { records.iter().map(f).collect() };
    let phi_input = col(|r| r.phi_input);
    let specs = col(|r| r.spectral);
    let cls = col(|r| r.consciousness_level);
    let gates = col(|r| r.gate);
    let n = records.len() as f64;

    dist("phi_input (the composite)", &phi_input);
    dist("  component: unified_psi", &col(|r| r.psi));
    dist("  component: spectral_boost", &specs);
    dist("  component: structural_boost", &col(|r| r.structural));
    dist("consciousness_level (telemetry)", &cls);
    dist("*** safety_gate_phi (TRUE gate) ***", &gates);

    let dom = |d: Dominant| 100.0 * records.iter().filter(|r| r.dominant == d).count() as f64 / n;
    println!(
        "    {:<34} Psi {:.1}% | spectral {:.1}% | structural {:.1}%",
        "argmax of the three-way max",
        dom(Dominant::Psi),
        dom(Dominant::Spectral),
        dom(Dominant::Structural)
    );

    // Warming diagnostic: within-subset drift, measured directly rather than inferred by
    // comparing across regimes (which is exactly the confound that broke the first run).
    let half = specs.len() / 2;
    println!(
        "    {:<34} 1st half {:.4} -> 2nd half {:.4}",
        "spectral_boost warming (in-subset)",
        mean_of(&specs[..half]),
        mean_of(&specs[half..])
    );

    tier_report("tiers of phi_input", &phi_input);
    tier_report("tiers of consciousness_level", &cls);
    // The only tier row that describes real robot behavior. The two above are diagnostics:
    // phi_input is an intermediate fed to the master equation, and consciousness_level is
    // post-social_mod telemetry. This one is what MotorSafetyLevel actually sees.
    tier_report("*** TIERS OF TRUE GATE ***", &gates);

    println!(
        "    {:<34} phi_input {:.1}% | gate {:.1}% (value >= {:.1})",
        "saturation fraction",
        100.0 * saturation_count(&phi_input) as f64 / n,
        100.0 * saturation_count(&gates) as f64 / n,
        SATURATION_CEILING
    );
    println!(
        "    {:<34} {} of {} gate cycles (non-finite or negative)",
        "invalid cycles",
        invalid_count(&gates),
        records.len()
    );

    let mut writer_counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    let mut max_staleness = 0usize;
    for r in records {
        *writer_counts.entry(r.writer.label()).or_insert(0) += 1;
        max_staleness = max_staleness.max(r.absolute_cycle.saturating_sub(r.written_at));
    }
    let writers: Vec<String> = writer_counts
        .iter()
        .map(|(k, v)| format!("{k} {:.1}%", 100.0 * *v as f64 / n))
        .collect();
    println!(
        "    {:<34} {} | max staleness {} cycles",
        "gate writer provenance",
        writers.join(" | "),
        max_staleness
    );
    println!();
}

/// Full per-regime report: the F2 line, then EVERY metric twice — all cycles, then
/// post-first-write only. Amendment 2 permits "exclude OR separately report"; this takes the
/// second option so the size of the F2 artifact stays visible.
fn report_regime(label: &str, records: &[CycleRecord]) {
    println!("  regime: {label}");
    print_f2_line(records);
    println!();
    report_subset(
        "(a) ALL CYCLES — includes prior-driven, NOT admissible",
        records,
    );
    report_subset(
        "(b) POST-FIRST-WRITE — the admissible measurement distribution",
        post_first_write(records),
    );
}

/// `arm` identifies the service-instance/order condition (A/B/C). It is threaded through
/// solely so the covariate line can record it, and that is not cosmetic: the first A5.0
/// analysis grouped covariate rows by regime alone and obtained r=+0.917 between cycle time
/// and gate value in the `alarming` regime. That correlation was entirely the arm confound —
/// arm A happens to be both slower and higher-gate than arms B/C — and it vanishes once arm
/// is held fixed. A covariate that cannot be conditioned on the dominant lurking variable
/// manufactures exactly the kind of finding this protocol exists to prevent.
fn run_regime(
    svc: &mut CognitiveLoopService,
    arm: &str,
    label: &str,
    make_input: &dyn Fn(usize) -> &'static str,
) {
    let load_start = read_loadavg();
    let records = collect_regime(svc, make_input);
    report_regime(label, &records);
    append_covariate_line(&format!("{arm}/{label}"), &records, load_start);
}

// ── A5.0: host-load covariate ────────────────────────────────────────────────────────────
//
// Amendment 5 defers the A1.3 methodology choice pending A5.0, which asks a prior question:
// **does the true gate value depend on host load?** If it does, motor authority is partly a
// function of CPU contention, which is a safety finding rather than a measurement nuisance,
// and it decides which of Amendment 4's three options is even available.
//
// The decisive statistic is deliberately NOT computed here, and the reason matters. Within a
// single run, cycle time and gate value both trend with cycle index — warmup is slow and the
// gate starts at a prior — so a within-run correlation between them is confounded by
// time-in-run and would manufacture a relationship out of two independent trends. The
// within-run figure is still printed below, explicitly labelled, so that nobody computes it
// later and mistakes it for the answer.
//
// A5.0's actual unit of observation is **the run**: one point per process, correlating that
// run's mean cycle time against its mean gate value, across runs sampling the box's natural
// load swing (9-61 on this host). That is exactly the shape of the E1 finding — 15 arms, each
// a point — and it is why each run appends one line to a durable log rather than printing and
// exiting. Analysis happens across the accumulated file, not inside any single invocation.

/// 1-minute load average, or NaN where unavailable (non-Linux, or `/proc` not mounted).
fn read_loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|f| f.parse().ok()))
        .unwrap_or(f64::NAN)
}

/// Pearson r. Returns NaN for n < 2 or a degenerate (zero-variance) input, rather than
/// silently reporting 0.0 — a flat series and an uncorrelated one are different findings.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let (mx, my) = (mean_of(&xs[..n]), mean_of(&ys[..n]));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (dx, dy) = (xs[i] - mx, ys[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    sxy / (sxx * syy).sqrt()
}

/// Where run-level covariate lines accumulate across invocations.
///
/// Defaults to a durable location rather than the session scratchpad: A5.0 needs runs spread
/// over hours to sample the load swing, and scratchpad does not survive that
/// (`feedback_scratchpad_is_ephemeral_use_var_lib_symthaea`). Override with
/// `SAFETY_COVARIATE_LOG`.
fn covariate_log_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SAFETY_COVARIATE_LOG") {
        return std::path::PathBuf::from(p);
    }
    let mut p = std::env::var("HOME")
        .map_or_else(|_| std::path::PathBuf::from("."), std::path::PathBuf::from);
    p.push("longrun-logs");
    p.push("phase4-a50-covariate.tsv");
    p
}

/// One run = one observation. Appends a TSV line and reports what it wrote.
///
/// Only admissible (post-first-write) cycles feed the summary: including prior-driven cycles
/// would mix the 0.05 cold-start constant into the gate mean and blunt exactly the dependence
/// A5.0 is looking for.
fn append_covariate_line(regime_label: &str, records: &[CycleRecord], load_start: f64) {
    let admissible = post_first_write(records);
    if admissible.is_empty() {
        println!("  [A5.0] no admissible cycles in this regime — nothing logged");
        return;
    }
    let micros: Vec<f64> = admissible.iter().map(|r| r.cycle_micros).collect();
    let gates: Vec<f64> = admissible.iter().map(|r| r.gate).collect();
    let (mean_micros, mean_gate) = (mean_of(&micros), mean_of(&gates));
    let within_run_r = pearson(&micros, &gates);
    let load_end = read_loadavg();

    println!(
        "  [A5.0 covariate] mean cycle {mean_micros:.0} us | mean gate {mean_gate:.6} | load {load_start:.2} -> {load_end:.2}"
    );
    println!(
        "      within-run r(cycle_us, gate) = {within_run_r:+.4}  <- CONFOUNDED by time-in-run, \
         not the A5.0 statistic; see this section's header"
    );

    let path = covariate_log_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let is_new = !path.exists();
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            use std::io::Write;
            if is_new {
                let _ = writeln!(
                    f,
                    "# Phase 4 A5.0 host-load covariate. One line per regime per process.\n\
                     # Unit of observation is the RUN. Correlate mean_cycle_us vs mean_gate\n\
                     # ACROSS lines from separate processes; within_run_r is confounded.\n\
                     #regime\tn_admissible\tmean_cycle_us\tmean_gate\tload_start\tload_end\twithin_run_r"
                );
            }
            let _ = writeln!(
                f,
                "{regime_label}\t{}\t{mean_micros:.1}\t{mean_gate:.6}\t{load_start:.2}\t{load_end:.2}\t{within_run_r:.4}",
                admissible.len()
            );
            println!("      appended to {}", path.display());
        }
        Err(e) => println!("      [A5.0] could not write {}: {e}", path.display()),
    }
}

// ── band estimation (A1.3) ───────────────────────────────────────────────────────────────

/// Spread of one metric across the N reference replicates, in that metric's own unit.
struct Band {
    name: &'static str,
    kind: MetricKind,
    /// The A1 mean — the single denominator for every relative comparison (A1.3).
    a1_mean: f64,
    /// max − min across the replicates, in native units.
    range_native: f64,
    /// max |value − mean| across the replicates, in native units.
    max_dev_native: f64,
}

impl Band {
    /// Convert a native-unit spread into the unit A1.3 mandates for this metric type.
    fn in_band_unit(&self, native: f64) -> Option<f64> {
        match self.kind {
            // Relative, as a percentage of the A1 mean. Undefined when that mean is ~0, and
            // reported as such rather than silently producing a huge or infinite percentage.
            MetricKind::Scalar => {
                if self.a1_mean.is_finite() && self.a1_mean.abs() > f64::EPSILON {
                    Some(100.0 * native / self.a1_mean.abs())
                } else {
                    None
                }
            }
            // Already percentage points; already re-estimated (not inherited).
            MetricKind::Proportion => Some(native),
            // Absolute count.
            MetricKind::Count => Some(native),
        }
    }
}

fn estimate_bands(runs: &[Vec<Metric>]) -> Vec<Band> {
    assert!(
        runs.len() >= MIN_BAND_RUNS,
        "band estimation needs >= {MIN_BAND_RUNS} reference runs, got {}",
        runs.len()
    );
    let width = runs[0].len();
    (0..width)
        .map(|i| {
            let vals: Vec<f64> = runs.iter().map(|r| r[i].value).collect();
            let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
            let (a1_mean, range_native, max_dev_native) = if finite.len() == vals.len() {
                let mean = mean_of(&finite);
                let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
                let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let dev = finite
                    .iter()
                    .map(|v| (v - mean).abs())
                    .fold(0.0f64, f64::max);
                (mean, max - min, dev)
            } else {
                // Any non-finite replicate makes the band unestimable; do not paper over it
                // by dropping that run from the denominator.
                (f64::NAN, f64::NAN, f64::NAN)
            };
            Band {
                name: runs[0][i].name,
                kind: runs[0][i].kind,
                a1_mean,
                range_native,
                max_dev_native,
            }
        })
        .collect()
}

/// Loud, repeated warning printed on every affected table when running below the protocol
/// floor. Deliberately not a single header line: a smoke run's output must be unmistakable
/// even when someone pastes one table out of context.
fn inadmissible_banner() {
    if is_inadmissible_run() {
        println!(
            "    !! INADMISSIBLE ({} cycles < {} protocol floor) — SMOKE TEST ONLY, DO NOT REPORT",
            cycles_per_regime(),
            CYCLES_PER_REGIME
        );
    }
}

fn print_band_table(title: &str, bands: &[Band]) {
    println!("    ── {title} ──");
    inadmissible_banner();
    println!(
        "    {:<26} {:<7} {:>12} {:>12} {:>12} {:>12}",
        "metric", "unit", "A1 mean", "BAND*", "range(diag)", "2x BAND"
    );
    for b in bands {
        let fmt = |native: f64| -> String {
            match b.in_band_unit(native) {
                Some(v) if v.is_finite() => match b.kind {
                    MetricKind::Scalar => format!("{v:.2}%"),
                    MetricKind::Proportion => format!("{v:.2}pp"),
                    MetricKind::Count => format!("{v:.0}"),
                },
                _ => "n/a".to_string(),
            }
        };
        let a1 = if b.a1_mean.is_finite() {
            format!("{:.4}", b.a1_mean)
        } else {
            "n/a".to_string()
        };
        println!(
            "    {:<26} {:<7} {:>12} {:>12} {:>12} {:>12}",
            b.name,
            b.kind.unit(),
            a1,
            // BAND* = max deviation from the A1 mean. THIS is the canonical band figure
            // that A1.3's exceedance rules apply against. `range` is printed only as a
            // diagnostic (it is ~2x max-dev for symmetric spread) and must NOT be used as
            // the band -- three candidate figures with no designated one would reintroduce
            // exactly the analysis-time choice A1.3 exists to remove.
            fmt(b.max_dev_native),
            fmt(b.range_native),
            fmt(2.0 * b.max_dev_native),
        );
    }
    println!();
}

/// Run ONLY the A1 reference condition, `n_runs` times, fresh service each time, and print a
/// per-metric band table per regime. See the module docs for how "reference condition" is
/// operationalised and why bands are per-regime.
fn run_band_estimation(regimes: &[Regime], n_runs: usize) {
    println!("── PER-METRIC REPRODUCIBILITY BANDS (protocol A1.3) ──");
    println!(
        "   reference condition: A1 present composite, FRESH service per regime (arm C policy)"
    );
    println!("   N = {n_runs} replicate runs; denominator = the A1 mean across those runs");
    println!("   arms A/B (shared-instance order controls) deliberately NOT replicated here\n");

    for (label, make_input) in regimes {
        println!("  regime: {label}");
        let mut all_runs: Vec<Vec<Metric>> = Vec::with_capacity(n_runs);
        let mut post_runs: Vec<Vec<Metric>> = Vec::with_capacity(n_runs);

        for run in 0..n_runs {
            let mut svc =
                CognitiveLoopService::new(base_config()).expect("band-estimation service");
            let records = collect_regime(&mut svc, make_input.as_ref());
            let post = post_first_write(&records);
            let first = first_real_write(&records);
            let gate_mean = mean_of(&post.iter().map(|r| r.gate).collect::<Vec<_>>());
            println!(
                "    run {}/{}  first-real-write@local {} | excluded {} | post-write n={} | \
                 mean gate {:.4}",
                run + 1,
                n_runs,
                first
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "never".to_string()),
                first.unwrap_or(records.len()),
                post.len(),
                gate_mean
            );
            all_runs.push(metrics(&records));
            post_runs.push(metrics(post));
        }
        println!();

        print_band_table(
            "(a) ALL CYCLES — includes prior-driven, NOT admissible",
            &estimate_bands(&all_runs),
        );
        print_band_table(
            "(b) POST-FIRST-WRITE — the admissible band",
            &estimate_bands(&post_runs),
        );
    }

    println!("A1.3 canonical band = the BAND* column (max deviation from the A1 mean) of the");
    println!("(b) POST-FIRST-WRITE table. The `range(diag)` column is a DIAGNOSTIC ONLY and is");
    println!("not the band; using it would inflate every band by ~2x. Designated here rather");
    println!("than left to the analyst, because A1.3 exists to remove analysis-time choice.");
    println!();
    println!("A1.3 marginal exceedance, fixed in advance — apply against the (b) BAND* column:");
    println!("  observed difference vs A1  <= band        -> NON-FINDING, not reportable");
    println!(
        "  observed difference vs A1  > band, <= 2x   -> CANDIDATE, needs a replication count"
    );
    println!("  observed difference vs A1  > 2x band       -> EFFECT, still report replications");
    println!("Band unit is fixed by metric type and is NOT transferable between types:");
    println!("  scalars = % of the A1 mean | proportions = percentage points | counts = absolute");
}

// ── Amendment 6 follow-up: decompose "position" into cycle count vs preceding content ────
//
// Amendment 6 measured that ~80% of gate variation is sequence position, and stated as an
// explicit limitation that "position" conflates TWO things: how many cycles have accumulated,
// and what content those cycles carried. No arm in the 3-arm design separates them — arm A's
// alarming-at-position-3 was preceded by repetitive+varied, arm B's repetitive-at-3 by
// alarming+varied, and arm C has no history at all.
//
// This mode separates them by holding one fixed while varying the other. The measured regime
// is ALWAYS `alarming`, so the measurement target never moves; only the history before it does.
//
//   Factor 1 — CYCLE COUNT (preceding content fixed at `repetitive`):
//     D0: 0 preceding cycles          (= position 1)
//     D1: 1x preceding, repetitive    (= position 2)
//     D2: 2x preceding, repetitive    (= position 3)
//
//   Factor 2 — CONTENT (preceding cycle count fixed at 2x):
//     D2 (repetitive) · E-varied · E-alarming
//
// Reading it: if the gate climbs across D0→D1→D2, cycle count drives the position effect. If it
// differs across D2/E-varied/E-alarming, content drives it. Both can be true, and the design
// gives their relative size directly.
//
// Every condition gets a FRESH service (protocol R1). Preceding levels scale with
// `cycles_per_regime()` so `--cycles` reproduces positions 1/2/3 exactly at any window size.

/// Run `n` cycles purely to accumulate state. Return values are deliberately discarded — this
/// is history, not measurement.
fn warm(svc: &mut CognitiveLoopService, make_input: &dyn Fn(usize) -> &'static str, n: usize) {
    for i in 0..n {
        let _ = svc.cycle(make_input(i));
    }
}

/// Distance to the nearest `MotorSafetyLevel::from_phi` boundary (protocol R3).
///
/// Reported for every condition because a value sitting near a boundary is operationally
/// fragile independent of any comparison: a perturbation far smaller than any effect under
/// study flips its tier, and therefore its torque cap.
fn boundary_distance(gate: f64) -> f64 {
    [0.1_f64, 0.3, 0.6]
        .iter()
        .map(|b| (gate - b).abs())
        .fold(f64::INFINITY, f64::min)
}

fn run_decomposition(regimes: &[Regime]) {
    let find = |name: &str| -> &Regime {
        regimes
            .iter()
            .find(|(l, _)| *l == name)
            .unwrap_or_else(|| panic!("regime {name} must exist"))
    };
    let (_, alarming) = find("alarming");
    let w = cycles_per_regime();

    println!("── DECOMPOSITION: cycle count vs preceding content (Amendment 6 limitation) ──");
    println!("   Measured regime is ALWAYS `alarming`; only the history before it varies.");
    println!("   Fresh service per condition (R1). Preceding window = {w} cycles.\n");

    // (condition label, preceding regime name or None, preceding multiplier)
    let conditions: Vec<(&str, Option<&str>, usize)> = vec![
        ("D0_pre0", None, 0),
        ("D1_pre1x_repetitive", Some("repetitive"), 1),
        ("D2_pre2x_repetitive", Some("repetitive"), 2),
        ("E_pre2x_varied", Some("varied"), 2),
        ("E_pre2x_alarming", Some("alarming"), 2),
    ];

    let mut results: Vec<(&str, f64)> = Vec::new();
    for (label, pre_name, mult) in &conditions {
        let mut svc = CognitiveLoopService::new(base_config()).expect("decomposition service");
        if let Some(pre) = pre_name {
            let (_, pre_input) = find(pre);
            warm(&mut svc, pre_input.as_ref(), w * mult);
        }
        let load_start = read_loadavg();
        let records = collect_regime(&mut svc, alarming.as_ref());
        let admissible = post_first_write(&records);
        let gate = mean_of(&admissible.iter().map(|r| r.gate).collect::<Vec<_>>());
        let tier = symthaea_core::embodiment::MotorSafetyLevel::from_phi(gate);
        println!(
            "  {label:<22} pre={:>4} cyc {:<11} -> gate {gate:.6}  tier {tier:?}  \
             boundary dist {:.4}",
            w * mult,
            pre_name.unwrap_or("(none)"),
            boundary_distance(gate),
        );
        append_covariate_line(&format!("DEC/{label}"), &records, load_start);
        results.push((label, gate));
    }

    let get = |k: &str| results.iter().find(|(l, _)| *l == k).map(|(_, v)| *v);
    println!("\n  ── effect sizes (R5: both reported, always) ──");
    if let (Some(d0), Some(d1), Some(d2)) = (
        get("D0_pre0"),
        get("D1_pre1x_repetitive"),
        get("D2_pre2x_repetitive"),
    ) {
        let cyc = [d0, d1, d2];
        let span = cyc.iter().cloned().fold(f64::MIN, f64::max)
            - cyc.iter().cloned().fold(f64::MAX, f64::min);
        println!("  CYCLE-COUNT effect (content held = repetitive): {span:.4}");
        println!("    D0 {d0:.6} -> D1 {d1:.6} -> D2 {d2:.6}");
    }
    if let (Some(r), Some(v), Some(a)) = (
        get("D2_pre2x_repetitive"),
        get("E_pre2x_varied"),
        get("E_pre2x_alarming"),
    ) {
        let c = [r, v, a];
        let span =
            c.iter().cloned().fold(f64::MIN, f64::max) - c.iter().cloned().fold(f64::MAX, f64::min);
        println!("  CONTENT effect (cycle count held = 2x):        {span:.4}");
        println!("    repetitive {r:.6} | varied {v:.6} | alarming {a:.6}");
    }
    println!(
        "\n  R4: re-run this in a SEPARATE PROCESS. Conditions must agree to <=1e-4. A larger\n  \
         disagreement means an uncontrolled variable entered the design — investigate it, do\n  \
         not average over it."
    );
    println!("  R2: differences matter when they change the TIER column above, not by magnitude.");
}

// ── CLI ──────────────────────────────────────────────────────────────────────────────────

enum Mode {
    /// The default 3-arm characterization (A shared/forward, B shared/reverse, C fresh).
    ThreeArm,
    /// A1.3 band estimation over N reference replicates.
    Bands(usize),
    /// Amendment 6 follow-up: cycle count vs preceding content.
    Decompose,
}

/// Parse `--bands [N]` / `--bands=N`. Absent → the 3-arm default.
///
/// An N below the protocol's hard floor is a hard error, not a silent clamp: §7 requires an
/// undeclared fallback to abort the run rather than emit plausible numbers from a substituted
/// mechanism, and quietly promoting `--bands 1` to 3 would be exactly that.
fn parse_mode<I: Iterator<Item = String>>(args: I) -> Result<Mode, String> {
    let args: Vec<String> = args.collect();
    let mut i = 0;
    let mut mode = Mode::ThreeArm;
    while i < args.len() {
        let a = &args[i];
        if let Some(rest) = a.strip_prefix("--bands=") {
            let n: usize = rest
                .parse()
                .map_err(|_| format!("--bands: not a number: {rest:?}"))?;
            mode = Mode::Bands(n);
        } else if a == "--bands" {
            let n = match args.get(i + 1) {
                Some(v) if !v.starts_with('-') => {
                    i += 1;
                    v.parse::<usize>()
                        .map_err(|_| format!("--bands: not a number: {v:?}"))?
                }
                _ => DEFAULT_BAND_RUNS,
            };
            mode = Mode::Bands(n);
        } else if let Some(rest) = a.strip_prefix("--cycles=") {
            let n: usize = rest
                .parse()
                .map_err(|_| format!("--cycles: not a number: {rest:?}"))?;
            if n == 0 {
                return Err("--cycles 0 is meaningless".to_string());
            }
            CYCLE_OVERRIDE.store(n, std::sync::atomic::Ordering::Relaxed);
        } else if a == "--cycles" {
            let v = args
                .get(i + 1)
                .ok_or_else(|| "--cycles needs a value".to_string())?;
            i += 1;
            let n: usize = v
                .parse()
                .map_err(|_| format!("--cycles: not a number: {v:?}"))?;
            if n == 0 {
                return Err("--cycles 0 is meaningless".to_string());
            }
            CYCLE_OVERRIDE.store(n, std::sync::atomic::Ordering::Relaxed);
        } else if a == "--decompose" {
            mode = Mode::Decompose;
        } else {
            return Err(format!("unrecognized argument: {a:?}"));
        }
        i += 1;
    }
    if let Mode::Bands(n) = mode {
        if n < MIN_BAND_RUNS {
            return Err(format!(
                "--bands {n}: protocol A1.3 requires at least {MIN_BAND_RUNS} reference runs to \
                 estimate a band. Refusing to substitute a smaller N."
            ));
        }
    }
    Ok(mode)
}

fn main() {
    let mode = match parse_mode(std::env::args().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "usage: safety_signal_characterization [--bands N | --decompose] [--cycles N]\n\
                 \t--bands N     N >= {MIN_BAND_RUNS}, default {DEFAULT_BAND_RUNS}\n\
                 \t--decompose   Amendment 6: cycle count vs preceding content"
            );
            std::process::exit(2);
        }
    };

    println!("=== Phase 4 safety-signal characterization ===");
    println!("Protocol: SYMTHAEA_PHASE4_CHARACTERIZATION_PROTOCOL_2026-07-29.md");
    println!("Measurement only; no production path modified.");
    println!(
        "{CYCLES_PER_REGIME} cycles/regime (67-cycle refresh fires ~{}x)",
        CYCLES_PER_REGIME / 67
    );
    println!(
        "F2 exclusion rule ACTIVE: every metric reported twice (all cycles / post-first-write).\n"
    );

    let regimes: Vec<Regime> = vec![
        (
            "repetitive",
            Box::new(|_| "the system hums quietly in the background"),
        ),
        (
            "varied",
            Box::new(|i| {
                [
                    "a bird lands on the windowsill and looks around",
                    "the pressure reading has drifted two points since noon",
                    "someone is asking whether the schedule changed",
                    "rain begins, lightly, on the far side of the yard",
                    "a cable was left unplugged behind the rack",
                    "the tone of the room shifted after that remark",
                ][i % 6]
            }),
        ),
        (
            "alarming",
            Box::new(|i| {
                [
                    "URGENT: fire detected in the server room, evacuate immediately!",
                    "Critical failure: coolant pressure dropping, meltdown risk rising!",
                    "Intruder alert: perimeter breach at the north gate right now!",
                ][i % 3]
            }),
        ),
    ];

    if let Mode::Bands(n) = mode {
        run_band_estimation(&regimes, n);
        return;
    }
    if matches!(mode, Mode::Decompose) {
        run_decomposition(&regimes);
        return;
    }

    println!("3 arms: shared/forward, shared/reverse, fresh-per-regime. See module docs.\n");

    // ── Arm A: shared service, forward order ──────────────────────────────────────
    // Reproduces the 2026-07-28 run. Doubles as a determinism check: seeded genesis +
    // async_training off means these numbers MUST match the committed ones.
    println!("── ARM A: shared service, FORWARD order (repetitive → varied → alarming) ──");
    println!("   (reproduces the 2026-07-28 committed run; also a determinism check)\n");
    {
        let mut svc = CognitiveLoopService::new(base_config()).expect("arm A service");
        for (label, make_input) in &regimes {
            run_regime(&mut svc, "A", label, make_input.as_ref());
        }
    }

    // ── Arm B: shared service, reverse order ──────────────────────────────────────
    // THE DECISIVE TEST. If `alarming` still saturates running FIRST on cold caches, the
    // regime effect is real. If `repetitive` saturates running LAST on warm caches, the
    // 2026-07-28 headline was a cache-warming artifact.
    println!("── ARM B: shared service, REVERSE order (alarming → varied → repetitive) ──");
    println!("   (decisive test: does saturation follow the REGIME or the run ORDER?)\n");
    {
        let mut svc = CognitiveLoopService::new(base_config()).expect("arm B service");
        for (label, make_input) in regimes.iter().rev() {
            run_regime(&mut svc, "B", label, make_input.as_ref());
        }
    }

    // ── Arm C: fresh service per regime ───────────────────────────────────────────
    // Removes the confound rather than testing for it. Cleanest per-regime numbers, and the
    // instance policy the --bands reference condition replicates.
    println!("── ARM C: FRESH service per regime (confound removed, no inherited warm-up) ──\n");
    for (label, make_input) in &regimes {
        let mut svc = CognitiveLoopService::new(base_config()).expect("arm C service");
        run_regime(&mut svc, "C", label, make_input.as_ref());
    }

    println!("Reminders when reading the above:");
    println!(" - Only the (b) POST-FIRST-WRITE subset is an admissible measurement distribution.");
    println!("   The (a) rows include cold-start cycles whose gate value is the 0.05 prior, which");
    println!("   from_phi maps to Red. That is what invalidated the 2026-07-29 minima (F2).");
    println!(" - `consciousness_level` is the telemetry value, biased vs. the true gate value by");
    println!("   social_mod (~±5%) and a 0.98 decay on non-MCE cycles. Tier percentages near the");
    println!("   0.3/0.6 boundaries inherit that bias. See module docs, limitation 1.");
    println!(" - unified_psi is autoregressive (0.35 weight on the previous consciousness_level),");
    println!("   so a high Psi-dominance share is partly the composite feeding itself.");
    println!(" - Staleness of several cycles is BY DESIGN (F3): the MCE's 5/10/20-cycle write");
    println!("   cadence dominates the produced->consumed lag. It is not a defect indicator.");
    println!(" - No band is reported here. Run with --bands N first; per A1.3 a between-arm");
    println!("   difference is uninterpretable until its metric's band is estimated.");
}
