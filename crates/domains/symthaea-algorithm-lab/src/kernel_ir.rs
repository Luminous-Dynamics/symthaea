// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, deterministic source-generation grammar for exact BinaryHV Hamming kernels.
//!
//! This module emits Rust source but does not compile or execute it. Generated source must
//! be evaluated by the separately permissioned runner planned in AR-6B (#12787).

use std::fmt::Write as _;

use symthaea_algorithms::ContentId;
use thiserror::Error;

const GRAMMAR_DOMAIN: &str = "symthaea.hdc-kernel-grammar.v1";
const SOURCE_DOMAIN: &str = "symthaea.hdc-kernel-source.v1";
const GRAMMAR_SCHEMA: &[u8] = b"binaryhv-hamming-word-width-unroll-accumulators-v1";

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum KernelPlanError {
    #[error("unroll factor must be one of 1, 2, 4, or 8")]
    InvalidUnrollFactor,
    #[error("accumulator count must be one of 1, 2, 4, or 8")]
    InvalidAccumulatorCount,
    #[error("unroll factor must divide the fixed word count exactly")]
    UnrollDoesNotDivideWordCount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WordWidth {
    U8,
    U32,
    U64,
    U128,
}

impl WordWidth {
    pub const ALL: [Self; 4] = [Self::U8, Self::U32, Self::U64, Self::U128];

    pub const fn bits(self) -> usize {
        match self {
            Self::U8 => 8,
            Self::U32 => 32,
            Self::U64 => 64,
            Self::U128 => 128,
        }
    }

    pub const fn bytes(self) -> usize {
        self.bits() / 8
    }

    pub const fn rust_type(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::U128 => "u128",
        }
    }
}

/// One point in the first source-generation grammar.
///
/// The grammar explores word width, static unrolling, and the number of independent
/// accumulation lanes. All plans preserve exact XOR + population-count semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelPlan {
    pub word_width: WordWidth,
    pub unroll_factor: u8,
    pub accumulator_count: u8,
}

impl KernelPlan {
    pub const UNROLL_FACTORS: [u8; 4] = [1, 2, 4, 8];
    pub const ACCUMULATOR_COUNTS: [u8; 4] = [1, 2, 4, 8];

    pub fn new(
        word_width: WordWidth,
        unroll_factor: u8,
        accumulator_count: u8,
    ) -> Result<Self, KernelPlanError> {
        let plan = Self {
            word_width,
            unroll_factor,
            accumulator_count,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(self) -> Result<(), KernelPlanError> {
        if !Self::UNROLL_FACTORS.contains(&self.unroll_factor) {
            return Err(KernelPlanError::InvalidUnrollFactor);
        }
        if !Self::ACCUMULATOR_COUNTS.contains(&self.accumulator_count) {
            return Err(KernelPlanError::InvalidAccumulatorCount);
        }
        let word_count = 2048 / self.word_width.bytes();
        if word_count % usize::from(self.unroll_factor) != 0 {
            return Err(KernelPlanError::UnrollDoesNotDivideWordCount);
        }
        Ok(())
    }

    /// Enumerate the complete finite v1 grammar in a stable canonical order.
    pub fn all_v1() -> Vec<Self> {
        let mut plans = Vec::with_capacity(
            WordWidth::ALL.len()
                * Self::UNROLL_FACTORS.len()
                * Self::ACCUMULATOR_COUNTS.len(),
        );
        for word_width in WordWidth::ALL {
            for unroll_factor in Self::UNROLL_FACTORS {
                for accumulator_count in Self::ACCUMULATOR_COUNTS {
                    // Every v1 combination divides 2048 bytes exactly. Keep validation here so
                    // a future grammar edit cannot silently introduce an invalid plan.
                    if let Ok(plan) = Self::new(word_width, unroll_factor, accumulator_count) {
                        plans.push(plan);
                    }
                }
            }
        }
        plans
    }

    pub fn id(self) -> ContentId {
        let width = (self.word_width.bits() as u64).to_be_bytes();
        let unroll = [self.unroll_factor];
        let accumulators = [self.accumulator_count];
        ContentId::derive(
            GRAMMAR_DOMAIN,
            [
                GRAMMAR_SCHEMA,
                width.as_slice(),
                unroll.as_slice(),
                accumulators.as_slice(),
            ],
        )
    }

    /// Deterministically render one safe, portable Rust implementation.
    ///
    /// This is a source artifact only. The function does not invoke rustc/Cargo and does not
    /// authorize execution. The output contains no unsafe blocks, target intrinsics, or
    /// architecture-specific assumptions.
    pub fn render_rust_source(self) -> Result<String, KernelPlanError> {
        self.validate()?;

        let bytes_per_word = self.word_width.bytes();
        let word_count = 2048 / bytes_per_word;
        let unroll = usize::from(self.unroll_factor);
        let accumulators = usize::from(self.accumulator_count);
        let rust_type = self.word_width.rust_type();

        let mut source = String::with_capacity(2048 + unroll * 420);
        writeln!(
            source,
            "// generated by symthaea HDC kernel grammar v1; plan={}",
            self.id()
        )
        .expect("writing to String cannot fail");
        source.push_str("use symthaea_core::hdc::binary_hv::BinaryHV;\n\n");
        source.push_str("#[inline(never)]\n");
        source.push_str(
            "pub fn generated_hamming(a: &BinaryHV, b: &BinaryHV) -> u32 {\n",
        );
        writeln!(source, "    const WORD_BYTES: usize = {bytes_per_word};")
            .expect("writing to String cannot fail");
        writeln!(source, "    const WORD_COUNT: usize = {word_count};")
            .expect("writing to String cannot fail");
        writeln!(source, "    const UNROLL: usize = {unroll};")
            .expect("writing to String cannot fail");
        writeln!(
            source,
            "    let mut accumulators = [0u32; {accumulators}];"
        )
        .expect("writing to String cannot fail");
        source.push_str("    for base_word in (0..WORD_COUNT).step_by(UNROLL) {\n");

        for offset in 0..unroll {
            writeln!(
                source,
                "        let word_index_{offset} = base_word + {offset};"
            )
            .expect("writing to String cannot fail");
            writeln!(
                source,
                "        let byte_index_{offset} = word_index_{offset} * WORD_BYTES;"
            )
            .expect("writing to String cannot fail");
            writeln!(
                source,
                "        let left_{offset} = {rust_type}::from_ne_bytes(\n            a.0[byte_index_{offset}..byte_index_{offset} + WORD_BYTES].try_into().expect(\"fixed-width word\"),\n        );"
            )
            .expect("writing to String cannot fail");
            writeln!(
                source,
                "        let right_{offset} = {rust_type}::from_ne_bytes(\n            b.0[byte_index_{offset}..byte_index_{offset} + WORD_BYTES].try_into().expect(\"fixed-width word\"),\n        );"
            )
            .expect("writing to String cannot fail");
            writeln!(
                source,
                "        accumulators[word_index_{offset} % {accumulators}] += (left_{offset} ^ right_{offset}).count_ones();"
            )
            .expect("writing to String cannot fail");
        }

        source.push_str("    }\n");
        source.push_str("    accumulators.into_iter().sum()\n");
        source.push_str("}\n");
        Ok(source)
    }

    pub fn source_id(self) -> Result<ContentId, KernelPlanError> {
        let source = self.render_rust_source()?;
        Ok(ContentId::derive(SOURCE_DOMAIN, [source.as_bytes()]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_grammar_enumerates_sixty_four_valid_plans() {
        let plans = KernelPlan::all_v1();
        assert_eq!(plans.len(), 64);
        assert!(plans.iter().all(|plan| plan.validate().is_ok()));
        assert_eq!(
            plans.iter().map(|plan| plan.id()).collect::<std::collections::BTreeSet<_>>().len(),
            plans.len()
        );
    }

    #[test]
    fn identical_plan_renders_identical_source_and_identity() {
        let plan = KernelPlan::new(WordWidth::U64, 4, 4).unwrap();
        assert_eq!(plan.render_rust_source().unwrap(), plan.render_rust_source().unwrap());
        assert_eq!(plan.source_id().unwrap(), plan.source_id().unwrap());
        assert_eq!(plan.id(), plan.id());
    }

    #[test]
    fn changing_any_search_dimension_changes_plan_and_source_identity() {
        let baseline = KernelPlan::new(WordWidth::U64, 4, 4).unwrap();
        let variants = [
            KernelPlan::new(WordWidth::U32, 4, 4).unwrap(),
            KernelPlan::new(WordWidth::U64, 2, 4).unwrap(),
            KernelPlan::new(WordWidth::U64, 4, 8).unwrap(),
        ];
        for variant in variants {
            assert_ne!(baseline.id(), variant.id());
            assert_ne!(baseline.source_id().unwrap(), variant.source_id().unwrap());
        }
    }

    #[test]
    fn renderer_is_portable_and_contains_no_unsafe_or_arch_specific_intrinsics() {
        for plan in KernelPlan::all_v1() {
            let source = plan.render_rust_source().unwrap();
            assert!(source.contains("pub fn generated_hamming"));
            assert!(source.contains("count_ones()"));
            assert!(!source.contains("unsafe"));
            assert!(!source.contains("_mm256"));
            assert!(!source.contains("_mm512"));
            assert!(!source.contains("std::arch"));
        }
    }

    #[test]
    fn invalid_plan_parameters_fail_closed() {
        assert_eq!(
            KernelPlan::new(WordWidth::U64, 3, 4),
            Err(KernelPlanError::InvalidUnrollFactor)
        );
        assert_eq!(
            KernelPlan::new(WordWidth::U64, 4, 3),
            Err(KernelPlanError::InvalidAccumulatorCount)
        );
    }

    #[test]
    fn generated_source_is_bound_to_exact_plan_identity() {
        let plan = KernelPlan::new(WordWidth::U128, 8, 2).unwrap();
        let source = plan.render_rust_source().unwrap();
        assert!(source.contains(&plan.id().to_string()));
        assert_eq!(
            plan.source_id().unwrap(),
            ContentId::derive(SOURCE_DOMAIN, [source.as_bytes()])
        );
    }
}
