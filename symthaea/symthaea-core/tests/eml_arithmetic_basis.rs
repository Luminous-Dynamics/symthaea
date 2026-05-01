// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_core::hdc::autodiff::{ad_begin_f64, ad_end, GenericVar, Scalar};
use symthaea_core::hdc::eml_regressor::{EmlMasterNode, EmlRegressor};

#[test]
fn test_recovery_addition() {
    // Target: y = x1 + x2
    let mut dataset = Vec::new();
    for i in 1..5 {
        for j in 1..5 {
            let x1 = i as f64;
            let x2 = j as f64;
            dataset.push((vec![x1, x2], x1 + x2));
        }
    }

    let mut regressor = EmlRegressor::<f64>::new(0, 2);
    // Addition is complex for EML. Parallel train with curriculum.
    regressor.parallel_train(&dataset, 1000, 4, 3);
    regressor.root.snap();

    let mse = regressor.calculate_mse(&dataset);
    assert!(mse < 0.1, "Addition recovery failed, MSE: {}", mse);
}

#[test]
fn test_recovery_multiplication() {
    // Target: y = x * 2.0
    let mut dataset = Vec::new();
    for i in 1..10 {
        let x = i as f64;
        dataset.push((vec![x], x * 2.0));
    }

    let mut regressor = EmlRegressor::<f64>::new(0, 1);
    regressor.parallel_train(&dataset, 1000, 5, 2);
    regressor.root.snap();

    let mse = regressor.calculate_mse(&dataset);
    assert!(mse < 0.1, "Multiplication recovery failed, MSE: {}", mse);
}
