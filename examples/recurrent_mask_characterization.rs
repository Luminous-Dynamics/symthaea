// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Step 1: characterize the recurrent-dimension masking mechanism itself.
//!
//! Step 0 (commit 132b919be1) made the lever reachable and honestly labelled.
//! This measures what the lever actually *does*, before any controller is built
//! on top of it. It is a characterization report, not a gate — it prints
//! measurements and asserts almost nothing, following the precedent of
//! `symthaea-core`'s `binding_algebra_characterization`.
//!
//! # Sections
//!
//! - **A. Reachability** — enumerate configs and record whether a mask actually
//!   executes. Confirms the three independent no-op routes and that the
//!   override is the only way in.
//! - **B. Suffix dependence** — the mask always amputates the same trailing
//!   dimensions. Compare that choice against leading and random subsets of the
//!   same size, to separate "capacity removed" from "this particular ordering".
//! - **C. Redistribution** — does the surviving prefix compensate over
//!   subsequent cycles, or does the state decay?
//! - **D. Stability** — is the energy profile a fixed property, or does it move
//!   between runs? Relevant because `consciousness_level` was shown to be
//!   wall-clock dependent (2026-07-28), so loop state may not be run-invariant.
//! - **E. Overhead** — measured to be robust against host load, which is the
//!   normal condition on this box (12-16 concurrent sessions).
//!
//! # Measuring under load
//!
//! Contention only ever *adds* time. So:
//! - **interleave** the conditions round-robin inside one process rather than
//!   running them back to back. Load drifts slowly relative to a cycle, so
//!   consecutive cycles of different conditions see near-identical contention;
//!   it becomes a common-mode term that cancels in the paired difference. This
//!   is the strongest of these techniques and the reason the section is built
//!   around a round-robin driver instead of sequential runs.
//! - report the **minimum** over N cycles (the uncontended cost), not the mean,
//!   whose right tail is pure interference;
//! - aggregate the microsecond-quantized overhead over many cycles and report a
//!   *ratio* to summed step time, so quantization averages out;
//! - warm up first, so first-cycle allocation does not land in the minimum.
//!
//! CPU time (`CLOCK_PROCESS_CPUTIME_ID`) would be a further improvement, but
//! `libc` is only a workspace dependency here, not one of this crate's own, and
//! adding a dependency to the flagship crate to time an example is not a good
//! trade. Interleaving plus minimum-of-N covers the same question.
//!
//! Run: `cargo run --release --example recurrent_mask_characterization`

// Deliberately no `Instant` here: every timing number in section E comes from
// the per-cycle instrumentation on `RecurrentMaskEvent`, compared in paired
// round-robin form. Wrapping whole runs in a wall-clock timer is what made the
// Step 0 numbers unusable.
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::SubstrateType;

/// Cycles per measured condition. Large enough that the minimum is a decent
/// uncontended estimate and the summed overhead beats microsecond quantization.
const CYCLES: usize = 200;
/// Independent repeats for the stability section.
const STABILITY_RUNS: usize = 3;
/// Random subsets sampled per fraction in the suffix-dependence section.
const RANDOM_SUBSETS: usize = 16;

const PROBE: &str = "recurrent mask characterization probe";

fn base_config() -> CognitiveLoopConfig {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.enable_validation_overlay = false;
    config
}

fn masked_config(frac: f32) -> CognitiveLoopConfig {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(frac);
    config
}

fn l2(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Energy (squared L2) of a subset of indices, as a share of the whole.
fn energy_share(state: &[f32], keep_out: &[usize]) -> f64 {
    let total: f64 = state.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let removed: f64 = keep_out
        .iter()
        .map(|&i| (state[i] as f64) * (state[i] as f64))
        .sum();
    removed / total
}

/// Deterministic LCG. Uses HIGH bits only — the low bits of an LCG are weak,
/// and `% n` on them would bias the sampled subsets.
struct Lcg(u64);
impl Lcg {
    fn next_below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n.max(1)
    }
}

fn sample_distinct(rng: &mut Lcg, n: usize, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    // Partial Fisher-Yates: first k entries become a uniform sample without replacement.
    for i in 0..k.min(n) {
        let j = i + rng.next_below(n - i);
        idx.swap(i, j);
    }
    idx.truncate(k.min(n));
    idx
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ═══════════════════════════════════════════════════════════════════════════
// A. Reachability
// ═══════════════════════════════════════════════════════════════════════════

fn section_a_reachability() {
    println!("\n═══ A. DEAD-PATH REACHABILITY ═══");
    println!(
        "{:<44} {:>9} {:>18} {:>8}",
        "config", "frac", "source", "masked?"
    );

    struct Case(&'static str, CognitiveLoopConfig);
    let mut cases = Vec::new();

    cases.push(Case("default (nothing enabled)", base_config()));

    let mut c = base_config();
    c.enable_substrate_encoding_noise = true;
    cases.push(Case("encoding noise only", c));

    let mut c = base_config();
    c.enable_recurrent_dim_masking = true;
    cases.push(Case("masking on, silicon, no override", c));

    let mut c = base_config();
    c.enable_recurrent_dim_masking = true;
    c.enable_substrate_speed_modulation = true;
    c.substrate_type = SubstrateType::QuantumComputer;
    cases.push(Case("masking on, quantum, speed-mod on", c));

    let mut c = base_config();
    c.enable_recurrent_dim_masking = true;
    c.substrate_type = SubstrateType::QuantumComputer;
    cases.push(Case("masking on, quantum, speed-mod OFF", c));

    for f in [1.0_f32, 0.5, 0.0] {
        cases.push(Case(
            Box::leak(format!("masking on, override {f:.2}").into_boxed_str()),
            masked_config(f),
        ));
    }

    for Case(label, config) in cases {
        let mut svc = CognitiveLoopService::new(config).expect("service");
        let frac = svc.substrate_effective_dim_fraction();
        let source = svc.substrate_effective_dim_source();
        let mut masked = false;
        for _ in 0..3 {
            let r = svc.cycle(PROBE);
            if r.metadata
                .substrate
                .recurrent_mask
                .as_ref()
                .is_some_and(|e| e.executed)
            {
                masked = true;
            }
        }
        println!("{label:<44} {frac:>9.3} {source:>18?} {masked:>8}");
    }
    println!(
        "\n  Note: the DEFAULT config reaches frac=1.0 by two independent routes \
         (SiliconDigital\n  has positive scale pressure; speed-mod off zeroes scale \
         pressure) and is gated off\n  by a third (the flag).\n\n  \
         CORRECTION (measured 2026-07-30): the override is NOT the only entry point, as\n  \
         an earlier draft of this note claimed. A scale-constrained substrate with speed\n  \
         modulation enabled reaches a sub-unit fraction on its own -- QuantumComputer\n  \
         lands at 0.300 and masks for real, with source=SubstratePressure. Any later\n  \
         ladder must therefore hold substrate_type AND\n  \
         enable_substrate_speed_modulation fixed, or a run that varies substrate for\n  \
         unrelated reasons will silently vary the lesion too. This is exactly the class\n  \
         of hidden coupling the provenance field exists to expose."
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// B + D. Suffix dependence and stability
// ═══════════════════════════════════════════════════════════════════════════

/// Collect CfC state snapshots from an unmasked run, so the analysis sees the
/// natural state rather than one already shaped by masking.
fn collect_states(seed_run: usize, cycles: usize) -> Vec<Vec<f32>> {
    let mut svc = CognitiveLoopService::new(base_config()).expect("service");
    let mut out = Vec::new();
    for i in 0..cycles {
        svc.cycle(PROBE);
        // Skip a warmup so the state is not dominated by initialization.
        if i >= cycles / 2 {
            if let Some(s) = svc.cfc_state_snapshot() {
                out.push(s);
            }
        }
    }
    let _ = seed_run;
    out
}

fn section_b_suffix_dependence(states: &[Vec<f32>], label: &str) -> Vec<(f32, f64, f64, f64)> {
    println!("\n═══ B. SUFFIX DEPENDENCE ({label}) ═══");
    if states.is_empty() {
        println!("  no state snapshots captured");
        return Vec::new();
    }
    let n = states[0].len();
    println!("  state dim = {n}, snapshots = {}", states.len());
    println!(
        "{:>6} {:>8} {:>14} {:>14} {:>14} {:>12}",
        "frac", "dims_cut", "trailing(prod)", "leading", "random(med)", "flat_expect"
    );

    let mut rows = Vec::new();
    for frac in [0.75_f32, 0.5, 0.25] {
        let mask_start = (frac * n as f32) as usize;
        let k = n - mask_start;

        let trailing: Vec<usize> = (mask_start..n).collect();
        let leading: Vec<usize> = (0..k).collect();

        let tr = median(states.iter().map(|s| energy_share(s, &trailing)).collect());
        let ld = median(states.iter().map(|s| energy_share(s, &leading)).collect());

        let mut rnd = Vec::new();
        for r in 0..RANDOM_SUBSETS {
            let mut rng = Lcg(0x5EED_0000 ^ (r as u64).wrapping_mul(0x9E37_79B9));
            for s in states.iter().take(8) {
                let subset = sample_distinct(&mut rng, n, k);
                rnd.push(energy_share(s, &subset));
            }
        }
        let rd = median(rnd);
        let flat = k as f64 / n as f64;

        println!("{frac:>6.2} {k:>8} {tr:>14.4} {ld:>14.4} {rd:>14.4} {flat:>12.4}");
        rows.push((frac, tr, ld, rd));
    }

    println!(
        "\n  If trailing << flat_expect, the production mask removes far less than its\n  \
         nominal fraction suggests, and severity is set by dimension ORDERING rather\n  \
         than by the fraction.\n\n  \
         MEASURED 2026-07-30: it does NOT. trailing, leading, random and flat all agree,\n  \
         and trailing/leading swap rank between runs -- systematic concentration cannot\n  \
         flip sign, noise can. Energy is essentially uniform across dimensions and the\n  \
         fraction maps roughly linearly to energy removed. This RETRACTS the ordering\n  \
         claim in 132b919be1's commit message, which read an in-situ measurement taken\n  \
         inside an already-masked run as evidence about the state's dimension ordering.\n  \
         It was evidence about HISTORY -- see section C. Full record:\n  \
         docs/RECURRENT_MASK_CHARACTERIZATION_2026-07-30.md"
    );
    rows
}

// ═══════════════════════════════════════════════════════════════════════════
// C. Redistribution
// ═══════════════════════════════════════════════════════════════════════════

fn section_c_redistribution() {
    println!("\n═══ C. STATE REDISTRIBUTION UNDER SUSTAINED MASKING ═══");
    println!(
        "{:>10} {:>14} {:>14} {:>14}",
        "cycle", "masked_pre_l2", "unmasked_l2", "ratio"
    );

    let mut masked = CognitiveLoopService::new(masked_config(0.5)).expect("service");
    let mut plain = CognitiveLoopService::new(base_config()).expect("service");

    for i in 0..CYCLES {
        let rm = masked.cycle(PROBE);
        plain.cycle(PROBE);
        let pre = rm
            .metadata
            .substrate
            .recurrent_mask
            .as_ref()
            .map(|e| e.pre_mask_norm as f64)
            .unwrap_or(f64::NAN);
        let un = plain
            .cfc_state_snapshot()
            .map(|s| l2(&s))
            .unwrap_or(f64::NAN);
        if i % 40 == 0 || i == CYCLES - 1 {
            println!("{:>10} {pre:>14.5} {un:>14.5} {:>14.4}", i, pre / un);
        }
    }
    println!(
        "\n  `masked_pre_l2` is the norm BEFORE each cycle's mask, i.e. what the\n  \
         full-width step produced from an already-lesioned prior state. A ratio that\n  \
         recovers toward 1.0 means the surviving prefix compensates; one that decays\n  \
         means the lesion compounds.\n\n  \
         MEASURED 2026-07-30: it compounds, hard -- the ratio decays monotonically to\n  \
         ~6-8% by cycle 160+. The surviving prefix does not compensate. Holding a fixed\n  \
         fraction is therefore NOT a steady state; the effective lesion deepens with\n  \
         duration, so duration is a controlled variable, not an incidental one.\n\n  \
         CAVEAT: the unmasked control is not stationary either -- its norm grows ~5\n  \
         orders of magnitude over 200 cycles and is non-monotonic at the tail. This\n  \
         ratio compares two moving quantities; read the direction and magnitude of the\n  \
         gap, not the precise decay rate."
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// E. Overhead
// ═══════════════════════════════════════════════════════════════════════════

struct CostRow {
    label: String,
    min_step_us: u64,
    med_step_us: f64,
    sum_step_us: u64,
    sum_overhead_us: u64,
    executed: usize,
    /// Median of per-cycle paired differences against the no-op condition,
    /// where both members of a pair ran adjacently under the same contention.
    med_paired_delta_us: f64,
}

/// Run all conditions ROUND-ROBIN in one process.
///
/// Sequential runs would let slow load drift masquerade as a condition effect —
/// which is exactly how the Step 0 gate produced a masked step that appeared
/// *faster* than the unmasked one. Cycling the conditions adjacently makes
/// contention a shared term, so the per-cycle paired difference measures the
/// condition and not the host.
fn section_e_overhead() {
    println!("\n═══ E. OVERHEAD (interleaved, load-robust) ═══");

    let labels = ["frac 1.00 (no-op)", "frac 0.50", "frac 0.25"];
    let fracs = [1.0_f32, 0.5, 0.25];
    let mut svcs: Vec<CognitiveLoopService> = fracs
        .iter()
        .map(|&f| CognitiveLoopService::new(masked_config(f)).expect("service"))
        .collect();

    // Warm up every condition before any measurement, so first-cycle allocation
    // cannot land in a minimum.
    for svc in svcs.iter_mut() {
        for _ in 0..10 {
            svc.cycle(PROBE);
        }
    }

    let n = svcs.len();
    let mut steps: Vec<Vec<u64>> = vec![Vec::with_capacity(CYCLES); n];
    let mut overhead = vec![0u64; n];
    let mut executed = vec![0usize; n];

    for _ in 0..CYCLES {
        // One round = one cycle of each condition, adjacent in time.
        for (i, svc) in svcs.iter_mut().enumerate() {
            let r = svc.cycle(PROBE);
            if let Some(e) = r.metadata.substrate.recurrent_mask.as_ref() {
                steps[i].push(e.step_duration_us);
                overhead[i] += e.mask_overhead_us;
                if e.executed {
                    executed[i] += 1;
                }
            } else {
                steps[i].push(0);
            }
        }
    }

    let rows: Vec<CostRow> = (0..n)
        .map(|i| {
            let paired: Vec<f64> = steps[i]
                .iter()
                .zip(steps[0].iter())
                .map(|(&a, &b)| a as f64 - b as f64)
                .collect();
            CostRow {
                label: labels[i].to_string(),
                min_step_us: steps[i]
                    .iter()
                    .copied()
                    .filter(|&x| x > 0)
                    .min()
                    .unwrap_or(0),
                med_step_us: median(steps[i].iter().map(|&x| x as f64).collect()),
                sum_step_us: steps[i].iter().sum(),
                sum_overhead_us: overhead[i],
                executed: executed[i],
                med_paired_delta_us: median(paired),
            }
        })
        .collect();

    println!(
        "{:<20} {:>10} {:>10} {:>12} {:>12} {:>9} {:>16}",
        "condition", "min_step", "med_step", "sum_step", "sum_ovhd", "masked", "med_paired_delta"
    );
    for r in &rows {
        println!(
            "{:<20} {:>10} {:>10.1} {:>12} {:>12} {:>9} {:>16.1}",
            r.label,
            r.min_step_us,
            r.med_step_us,
            r.sum_step_us,
            r.sum_overhead_us,
            r.executed,
            r.med_paired_delta_us
        );
    }

    let base = &rows[0];
    println!(
        "\n  min_step is the uncontended cost (contention only adds). med_paired_delta is\n  \
         the median per-cycle difference against the no-op condition measured in the same\n  \
         round, so host load cancels."
    );
    for r in &rows[1..] {
        let d = r.min_step_us as i64 - base.min_step_us as i64;
        let pct = if base.sum_step_us > 0 {
            100.0 * r.sum_overhead_us as f64 / base.sum_step_us as f64
        } else {
            f64::NAN
        };
        println!(
            "  {:<18} min_step delta {:+} us | paired delta {:+.1} us | mask overhead = {:.3}% of step time",
            r.label, d, r.med_paired_delta_us, pct
        );
    }
    println!(
        "\n  The mask runs AFTER the step on its output, so a negative min_step delta is\n  \
         residual host noise, not a saving — the step is called identically in every\n  \
         condition. Overhead is strictly additive by construction; the percentage above\n  \
         is what it costs."
    );
}

fn main() {
    println!("Recurrent-dimension masking — Step 1 characterization");
    println!("cycles/condition = {CYCLES}, stability runs = {STABILITY_RUNS}");

    section_a_reachability();

    let mut all_rows = Vec::new();
    for run in 0..STABILITY_RUNS {
        let states = collect_states(run, 80);
        let rows = section_b_suffix_dependence(&states, &format!("run {run}"));
        all_rows.push(rows);
    }

    println!("\n═══ D. STABILITY ACROSS RUNS ═══");
    println!(
        "{:>6} {:>16} {:>16} {:>12}",
        "frac", "trailing_min", "trailing_max", "spread"
    );
    for i in 0..all_rows.first().map(|r| r.len()).unwrap_or(0) {
        let frac = all_rows[0][i].0;
        let vals: Vec<f64> = all_rows
            .iter()
            .filter_map(|r| r.get(i))
            .map(|t| t.1)
            .collect();
        let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{frac:>6.2} {lo:>16.4} {hi:>16.4} {:>12.4}", hi - lo);
    }
    println!(
        "\n  A wide spread would mean the energy profile is run-dependent, not a fixed\n  \
         property — which matters because loop behavior on this host was shown to be\n  \
         wall-clock-time dependent (2026-07-28)."
    );

    section_c_redistribution();
    section_e_overhead();

    println!("\nDone.");
}
