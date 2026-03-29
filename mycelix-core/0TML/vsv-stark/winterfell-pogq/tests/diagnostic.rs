// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic test to debug constraint validation
//! Emits per-row residuals for each constraint

use winterfell_pogq::{AirPublicInputs as PublicInputs, TraceBuilder, SCALE, AIR_SCHEMA_REV};
use math::StarkField; // For as_int() method

fn q(val: f32) -> u64 {
    (val * SCALE as f32) as u64
}

#[test]
fn test_ema_constraint_diagnostic() {
    let public = PublicInputs {
        beta: q(0.85),
        w: 3,
        k: 2,
        m: 3,
        threshold: q(0.90),
        ema_init: q(0.85),
        viol_init: 0,
        clear_init: 0,
        quar_init: 0,
        round_init: 0,
        quar_out: 0,
        trace_length: 8,
            prov_hash: [0, 0, 0, 0],
            profile_id: 128,
            air_rev: AIR_SCHEMA_REV,
    };

    let witness_scores = vec![q(0.92); 8];

    println!("\n=== PoGQ Trace Diagnostic ===");
    println!("SCALE = {}", SCALE);
    println!("beta = {} (0.85 * {})", public.beta, SCALE);
    println!("threshold = {} (0.90 * {})", public.threshold, SCALE);
    println!();

    let builder = TraceBuilder::new(public.clone(), witness_scores.clone());
    let trace = builder.build();

    println!("Row | ema_t      | x_t        | ema_next   | rem        | LHS (ema*S+rem) | RHS (sum)      | Residual | Pass?");
    println!("----|------------|------------|------------|------------|-----------------|----------------|----------|-------");

    for row in 0..8 {
        let ema_t = trace.get(0, row).as_int() as u64;
        let x_t = trace.get(4, row).as_int() as u64;
        let beta = public.beta;

        // What the trace computed for next row
        let ema_next_trace = if row < 7 {
            trace.get(0, row + 1).as_int() as u64
        } else {
            0 // No next row
        };

        // Get remainder from trace (column 11) - from NEXT row since it corresponds to ema_next
        let rem_trace = if row < 7 {
            trace.get(11, row + 1).as_int() as u64
        } else {
            0
        };

        if row < 7 {
            // NEW AIR constraint check with remainder column:
            // ema_next * SCALE + rem_next = beta * ema + (SCALE - beta) * x
            let lhs = (ema_next_trace as u128) * (SCALE as u128) + (rem_trace as u128);
            let rhs = (beta as u128) * (ema_t as u128) + ((SCALE as u128) - (beta as u128)) * (x_t as u128);

            let residual = if lhs >= rhs {
                lhs - rhs
            } else {
                rhs - lhs
            };

            let passes = residual == 0;

            println!(
                "{:3} | {:10} | {:10} | {:10} | {:10} | {:15} | {:14} | {:8} | {}",
                row, ema_t, x_t, ema_next_trace, rem_trace, lhs, rhs, residual,
                if passes { "✓" } else { "✗" }
            );

            if !passes {
                println!("    CONSTRAINT VIOLATION at row {}", row);
                println!("    LHS = ema_next * SCALE + rem = {} * {} + {} = {}", ema_next_trace, SCALE, rem_trace, lhs);
                println!("    RHS = beta * ema + (S-beta) * x = {}", rhs);
                println!("    residual = {} (should be 0!)", residual);
            }
        } else {
            println!("{:3} | {:10} | {:10} | (final)    | -          | -               | -              | -        | -", row, ema_t, x_t);
        }
    }

    println!();
    println!("Detailed computation for row 0 -> 1:");
    let ema_0 = trace.get(0, 0).as_int() as u64;
    let x_0 = trace.get(4, 0).as_int() as u64;
    let ema_1 = trace.get(0, 1).as_int() as u64;
    let beta = public.beta;

    println!("  ema[0] = {}", ema_0);
    println!("  x[0] = {}", x_0);
    println!("  beta = {}", beta);
    println!();
    println!("  Trace computation (with truncating division):");
    let beta_times_ema0 = (beta as u128) * (ema_0 as u128);
    let one_minus_beta_times_x0 = ((SCALE as u128) - (beta as u128)) * (x_0 as u128);
    let sum = beta_times_ema0 + one_minus_beta_times_x0;
    println!("    beta * ema[0] = {} * {} = {}", beta, ema_0, beta_times_ema0);
    println!("    (S - beta) * x[0] = {} * {} = {}", (SCALE as u128) - (beta as u128), x_0, one_minus_beta_times_x0);
    println!("    sum = {}", sum);
    println!("    ema[1] = sum / SCALE = {} / {} = {}", sum, SCALE, ema_1);
    println!();
    println!("  AIR constraint check (scaled version to avoid division):");
    let lhs = (ema_1 as u128) * (SCALE as u128);
    let rhs = sum;
    println!("    LHS = ema[1] * SCALE = {} * {} = {}", ema_1, SCALE, lhs);
    println!("    RHS = beta * ema[0] + (S-beta) * x[0] = {}", rhs);
    let residual = if lhs >= rhs { lhs - rhs } else { rhs - lhs };
    println!("    residual = |LHS - RHS| = {}", residual);
    println!("    bounded? {} (residual < SCALE)", residual < SCALE);
}
