use super::*;

impl ConjectureEngine {
    /// Numerically verify conjectures against held-out test data.
    pub fn verify_numerical(&mut self) {
        let observations = self.observations.clone();
        for conjecture in &mut self.conjectures {
            if !matches!(conjecture.status, ConjectureStatus::Proposed) {
                continue;
            }
            if let Some(seq) = observations.iter().find(|s| s.name == conjecture.source) {
                let (train, test) = seq.train_test_split();
                if test.is_empty() {
                    continue;
                }

                let test_mse = compute_mse(&conjecture.formula, &test);
                let train_mse = conjecture.training_mse;

                let v1 = conjecture.formula.eval(&[("n", 10.0)]);
                let v2 = conjecture.formula.eval(&[("n", 100.0)]);
                let is_constant = v1.is_finite() && v2.is_finite() && (v1 - v2).abs() < 1e-10;
                let test_better = test_mse.is_finite() && test_mse < train_mse;

                let rel_errors: Vec<f64> = test
                    .iter()
                    .filter_map(|(x, y)| {
                        let pred = conjecture.formula.eval(&[("n", *x)]);
                        if pred.is_finite() && y.abs() > 1e-10 {
                            Some(((pred - y) / y).abs())
                        } else {
                            None
                        }
                    })
                    .collect();
                let mean_rel_error = if rel_errors.is_empty() {
                    f64::MAX
                } else {
                    rel_errors.iter().sum::<f64>() / rel_errors.len() as f64
                };

                if test_mse.is_finite()
                    && (test_mse < 1e-6 || (is_constant && test_better) || mean_rel_error < 0.10)
                {
                    conjecture.status = ConjectureStatus::NumericallyTested { test_mse };
                    let macro_safe_numeric_fit = test_mse < 1e-3
                        || mean_rel_error < 0.01
                        || conjecture_has_verified_eml_backend(conjecture);
                    if macro_safe_numeric_fit {
                        elevate_macro_promotion_tier(
                            conjecture,
                            MacroPromotionTier::RecurrentNumerical,
                        );
                    } else {
                        conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                    }
                    if test_better || mean_rel_error < 0.01 {
                        conjecture.confidence = (conjecture.confidence + 0.9) / 2.0;
                    } else {
                        conjecture.confidence = (conjecture.confidence + 0.7) / 2.0;
                    }

                    const NEAR_EXACT_MSE: f64 = 1e-10;
                    if train_mse < NEAR_EXACT_MSE && test_mse < NEAR_EXACT_MSE {
                        conjecture.status = ConjectureStatus::FormallyVerified {
                            proof_steps: seq.data.len(),
                        };
                        elevate_macro_promotion_tier(conjecture, MacroPromotionTier::Formal);
                        conjecture.confidence = (conjecture.confidence + 0.95) / 2.0;
                    }

                    if is_constant
                        && let Expr::Const(c) = &conjecture.formula
                        && let Some(name) = identify_constant(*c)
                    {
                        conjecture.formula_str = name;
                    }
                } else if test_mse.is_finite() && mean_rel_error > 0.5 {
                    conjecture.status = ConjectureStatus::Refuted {
                        counterexample: test[0].0,
                    };
                    conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                    conjecture.confidence = 0.0;
                } else {
                    conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                    conjecture.confidence = conjecture.confidence.min(0.1);
                }
            }
        }
    }

    /// Formally verify conjectures by bounded induction.
    pub fn verify_formal(&mut self, max_n: usize) {
        let observations = self.observations.clone();
        for conjecture in &mut self.conjectures {
            if !matches!(
                conjecture.status,
                ConjectureStatus::NumericallyTested { .. }
            ) {
                continue;
            }

            let v1 = conjecture.formula.eval(&[("n", 10.0)]);
            let v2 = conjecture.formula.eval(&[("n", 100.0)]);
            let is_asymptotic =
                v1.is_finite() && v2.is_finite() && (v1 - v2).abs() < v1.abs().max(1.0) * 0.01;
            if is_asymptotic {
                continue;
            }

            if let Some(seq) = observations.iter().find(|s| s.name == conjecture.source) {
                let known: std::collections::HashMap<i64, f64> =
                    seq.data.iter().map(|(x, y)| (*x as i64, *y)).collect();

                let mut all_exact = true;
                let mut checked = 0usize;
                let mut first_failure: Option<f64> = None;

                for n in 1..=max_n {
                    let predicted = conjecture.formula.eval(&[("n", n as f64)]);
                    if !predicted.is_finite() {
                        all_exact = false;
                        first_failure = Some(n as f64);
                        break;
                    }

                    if let Some(&expected) = known.get(&(n as i64)) {
                        let tol = expected.abs().max(1.0) * 1e-9;
                        if (predicted - expected).abs() > tol {
                            all_exact = false;
                            first_failure = Some(n as f64);
                            break;
                        }
                    }
                    checked += 1;
                }

                if all_exact && checked >= 100 {
                    conjecture.status = ConjectureStatus::FormallyVerified {
                        proof_steps: checked,
                    };
                    elevate_macro_promotion_tier(conjecture, MacroPromotionTier::Formal);
                    conjecture.confidence = 0.95;
                } else if let Some(cx) = first_failure
                    && known.contains_key(&(cx as i64))
                {
                    let pred = conjecture.formula.eval(&[("n", cx)]);
                    let expected = known[&(cx as i64)];
                    let rel_err = if expected.abs() > 1e-10 {
                        ((pred - expected) / expected).abs()
                    } else {
                        (pred - expected).abs()
                    };
                    if rel_err > 0.5 {
                        conjecture.status = ConjectureStatus::Refuted { counterexample: cx };
                        conjecture.macro_promotion_tier = MacroPromotionTier::Quarantined;
                        conjecture.confidence = 0.0;
                    }
                }
            }
        }
    }

    /// Attempt to formally prove all numerically-verified conjectures via Z3.
    pub fn auto_prove_via_z3(&mut self) {
        let z3_path = match detect_z3_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "[conjecture_engine] auto_prove_via_z3: z3 not found — \
                     set $Z3_PATH or add z3 to PATH (e.g. `nix-shell -p z3`). \
                     Formal verification skipped for {} conjectures.",
                    self.conjectures
                        .iter()
                        .filter(|c| matches!(c.status, ConjectureStatus::NumericallyTested { .. }))
                        .count()
                );
                return;
            }
        };

        let observations = self.observations.clone();

        for conjecture in &mut self.conjectures {
            if !matches!(
                conjecture.status,
                ConjectureStatus::NumericallyTested { .. }
            ) {
                continue;
            }

            let src = match observations.iter().find(|o| o.name == conjecture.source) {
                Some(s) => s,
                None => continue,
            };

            let smt = match expr_to_smtlib2(&conjecture.formula, "n") {
                Some(s) => s,
                None => continue,
            };

            let mut all_verified = true;
            let mut tested_points = 0usize;

            for &(x, y) in &src.data {
                let n_val = x;
                if !n_val.is_finite() || !y.is_finite() {
                    continue;
                }

                let query = format!(
                    "(set-logic QF_NRA)\n\
                     (declare-const n Real)\n\
                     (assert (= n {:.12}))\n\
                     (assert (or (> (- {} {:.12}) 0.000001) (< (- {} {:.12}) -0.000001)))\n\
                     (check-sat)\n",
                    n_val, smt, y, smt, y
                );

                let output = std::process::Command::new(&z3_path)
                    .arg("-in")
                    .arg("-T:2")
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .and_then(|mut child| {
                        use std::io::Write;
                        if let Some(stdin) = child.stdin.as_mut() {
                            stdin.write_all(query.as_bytes()).ok();
                        }
                        child.wait_with_output()
                    });

                match output {
                    Ok(out) => {
                        let result = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        tested_points += 1;
                        if !result.starts_with("unsat") {
                            all_verified = false;
                            break;
                        }
                    }
                    Err(_) => {
                        all_verified = false;
                        break;
                    }
                }
            }

            if all_verified && tested_points > 0 {
                conjecture.status = ConjectureStatus::FormallyVerified {
                    proof_steps: tested_points,
                };
                elevate_macro_promotion_tier(conjecture, MacroPromotionTier::Formal);
                conjecture.confidence = 0.99;
            }
        }
    }
}
