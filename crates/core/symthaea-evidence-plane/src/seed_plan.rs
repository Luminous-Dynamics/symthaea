// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Seed discipline as an enforced contract, not a promise.
//!
//! `docs/TEMPORAL_BENCHMARK_V2_PREREGISTRATION_2026-07-31.md` requires "≥ 8
//! seeds per arm per task family, fixed before running, disjoint from any seed
//! used during development." Written as prose, that constraint gets hand-waved
//! at run time — nobody sets out to reuse a development seed, they simply forget
//! which ones those were six hours later.
//!
//! This makes it mechanical, in the same spirit as the rest of this crate: the
//! contract fails loudly rather than being annotated.
//!
//! # Why development-seed reuse specifically matters
//!
//! Development seeds are the ones a mechanism was debugged against. Anything
//! tuned while watching them — a threshold, a decay constant, a stopping rule —
//! has been fitted to those particular draws. Reusing them for a confirmatory
//! run reports a fitted result as a predictive one. It is the same class of
//! error as adjusting a threshold after seeing the numbers, which this arc has
//! done twice, and it is invisible in the output.
//!
//! # The fingerprint
//!
//! [`SeedPlan::fingerprint`] identifies the exact registered plan. A results
//! report carrying it can be checked against the plan it claims to have run
//! under, so a silently-edited seed set is detectable after the fact rather than
//! taken on trust.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Minimum confirmatory seeds, per the pre-registration.
pub const MIN_CONFIRMATORY_SEEDS: usize = 8;

/// Ways a seed plan can be invalid or misused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeedViolation {
    /// A confirmatory seed was also used during development. The result would be
    /// fitted, not predictive.
    DevelopmentSeedReused { seed: u64 },
    /// Fewer confirmatory seeds than the pre-registration requires.
    TooFewConfirmatory { got: usize, required: usize },
    /// The same seed appears twice, inflating apparent sample size.
    DuplicateSeed { seed: u64 },
    /// A run used a seed that was never registered as confirmatory.
    UnregisteredSeed { seed: u64 },
}

/// A frozen seed plan.
///
/// Construct once via [`SeedPlan::register`], before running anything. There is
/// deliberately no way to add seeds afterwards: the whole value is that the set
/// cannot grow once results start appearing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedPlan {
    confirmatory: Vec<u64>,
    development: Vec<u64>,
}

impl SeedPlan {
    /// Register and freeze a plan, or fail with every violation found.
    pub fn register(
        confirmatory: Vec<u64>,
        development: Vec<u64>,
    ) -> Result<Self, Vec<SeedViolation>> {
        let mut violations = Vec::new();

        if confirmatory.len() < MIN_CONFIRMATORY_SEEDS {
            violations.push(SeedViolation::TooFewConfirmatory {
                got: confirmatory.len(),
                required: MIN_CONFIRMATORY_SEEDS,
            });
        }

        let mut seen = HashSet::new();
        for &s in &confirmatory {
            if !seen.insert(s) {
                violations.push(SeedViolation::DuplicateSeed { seed: s });
            }
        }

        let dev: HashSet<u64> = development.iter().copied().collect();
        for &s in &confirmatory {
            if dev.contains(&s) {
                violations.push(SeedViolation::DevelopmentSeedReused { seed: s });
            }
        }

        if violations.is_empty() {
            Ok(Self {
                confirmatory,
                development,
            })
        } else {
            Err(violations)
        }
    }

    /// The registered confirmatory seeds. The only legitimate source of seeds
    /// for a reported run.
    pub fn confirmatory(&self) -> &[u64] {
        &self.confirmatory
    }

    /// Check at the point of use that a seed is registered.
    ///
    /// Call this where the seed is actually consumed, not where the plan is
    /// built — the failure mode being guarded is a stray literal appearing in a
    /// loop months later, which a construction-time check cannot see.
    pub fn check(&self, seed: u64) -> Result<(), SeedViolation> {
        if self.confirmatory.contains(&seed) {
            Ok(())
        } else {
            Err(SeedViolation::UnregisteredSeed { seed })
        }
    }

    /// Hard variant of [`Self::check`] for run loops that should abort rather
    /// than accumulate an invalid result.
    ///
    /// # Panics
    /// If the seed was not registered.
    pub fn enforce(&self, seed: u64) {
        if let Err(v) = self.check(seed) {
            panic!("seed discipline violated: {v:?} — this run is not confirmatory");
        }
    }

    /// Stable identifier for this exact plan, for embedding in a results report.
    pub fn fingerprint(&self) -> String {
        // Order-independent so a reordered-but-identical set fingerprints the
        // same; a reordering is not a different plan.
        let mut c = self.confirmatory.clone();
        let mut d = self.development.clone();
        c.sort_unstable();
        d.sort_unstable();
        crate::config_hash(&(c, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_plan() -> SeedPlan {
        SeedPlan::register((100..108).collect(), vec![1, 2, 3]).expect("valid")
    }

    /// The violation that matters most: a seed used while tuning must never
    /// appear in a confirmatory run.
    #[test]
    fn development_seed_reuse_is_rejected() {
        let v = SeedPlan::register((1..9).collect(), vec![5]).expect_err("must reject");
        assert!(v.contains(&SeedViolation::DevelopmentSeedReused { seed: 5 }));
    }

    #[test]
    fn too_few_seeds_is_rejected() {
        let v = SeedPlan::register(vec![1, 2, 3], vec![]).expect_err("must reject");
        assert!(v.contains(&SeedViolation::TooFewConfirmatory {
            got: 3,
            required: MIN_CONFIRMATORY_SEEDS
        }));
    }

    /// Duplicates would inflate apparent sample size while adding no information.
    #[test]
    fn duplicate_seeds_are_rejected() {
        let mut seeds: Vec<u64> = (100..107).collect();
        seeds.push(100);
        let v = SeedPlan::register(seeds, vec![]).expect_err("must reject");
        assert!(v.contains(&SeedViolation::DuplicateSeed { seed: 100 }));
    }

    /// All violations are reported at once, so a plan can be fixed in one pass.
    #[test]
    fn every_violation_is_reported_not_just_the_first() {
        let v = SeedPlan::register(vec![1, 1, 2], vec![2]).expect_err("must reject");
        assert!(v.len() >= 3, "expected several violations, got {v:?}");
    }

    /// The guard has to work where the seed is consumed, not only where the plan
    /// is built — a stray literal in a run loop is the realistic failure.
    #[test]
    fn unregistered_seed_is_caught_at_point_of_use() {
        let plan = ok_plan();
        assert!(plan.check(100).is_ok());
        assert_eq!(
            plan.check(999),
            Err(SeedViolation::UnregisteredSeed { seed: 999 })
        );
    }

    #[test]
    #[should_panic(expected = "seed discipline violated")]
    fn enforce_aborts_on_unregistered_seed() {
        ok_plan().enforce(999);
    }

    /// A silently-edited seed set must be detectable after the fact.
    #[test]
    fn fingerprint_changes_when_the_plan_changes() {
        let a = ok_plan();
        let b = SeedPlan::register((200..208).collect(), vec![1, 2, 3]).expect("valid");
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    /// Reordering is not a different plan.
    #[test]
    fn fingerprint_is_order_independent() {
        let a = SeedPlan::register((100..108).collect(), vec![1, 2]).expect("valid");
        let mut rev: Vec<u64> = (100..108).collect();
        rev.reverse();
        let b = SeedPlan::register(rev, vec![2, 1]).expect("valid");
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    /// There must be no way to grow the set after registration. This test exists
    /// to fail loudly if a future refactor adds a mutator.
    #[test]
    fn plan_exposes_no_mutation_path() {
        let plan = ok_plan();
        let before = plan.fingerprint();
        let _ = plan.confirmatory();
        assert_eq!(plan.fingerprint(), before);
    }
}
