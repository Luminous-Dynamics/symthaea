// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![cfg(feature = "symthaea-backend")]

use symthaea_psych_bench::harness::live_runner::CognitiveLoopBenchmarkRunner;
use symthaea_psych_bench::live_runner_contract::CognitiveLoopBenchmarkRunnerContractExt;

#[test]
fn cognitive_loop_runner_implements_explicit_contract_adapter() {
    fn assert_impl<T: CognitiveLoopBenchmarkRunnerContractExt>() {}
    assert_impl::<CognitiveLoopBenchmarkRunner>();
}
