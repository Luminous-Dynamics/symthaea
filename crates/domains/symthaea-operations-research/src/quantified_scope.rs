// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Quantifier-prefix and Skolem-dependency semantics for quantified strategies.
//!
//! This module validates the ordered variable scope of a prenex QBF contract.
//! It does not solve QBF, #QBF, or MaxQBF. Its purpose is to prevent future
//! strategy plans from representing quantified formulas as unordered
//! visible/hidden sets and silently losing dependency order.

/// A quantifier in a prenex Boolean formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quantifier {
    ForAll,
    Exists,
}

/// One non-empty block in a prenex quantifier prefix.
///
/// Variables in a block share the same quantifier and do not depend on one
/// another. Block order is semantically significant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantifierBlock {
    pub quantifier: Quantifier,
    pub variables: Vec<usize>,
}

/// Resource limits for validating a quantified variable universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantifierPrefixBudget {
    pub max_variables: usize,
    pub max_blocks: usize,
    pub max_variable_occurrences: usize,
}

impl Default for QuantifierPrefixBudget {
    fn default() -> Self {
        Self {
            max_variables: 1 << 20,
            max_blocks: 1 << 20,
            max_variable_occurrences: 1 << 20,
        }
    }
}

/// A validated prenex prefix with an exact, duplicate-free variable universe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedQuantifierPrefix {
    variable_count: usize,
    blocks: Vec<QuantifierBlock>,
    locations: Vec<(usize, Quantifier)>,
}

/// Errors in quantifier-prefix structure or Skolem dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantifierScopeError {
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    BlockBudgetExceeded {
        required: usize,
        limit: usize,
    },
    VariableOccurrenceBudgetExceeded {
        required: usize,
        limit: usize,
    },
    VariableOccurrenceCountMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyQuantifierBlock {
        block: usize,
    },
    QuantifiedVariableOutOfRange {
        block: usize,
        variable: usize,
        variable_count: usize,
    },
    VariableRepeated {
        variable: usize,
    },
    VariableMissing {
        variable: usize,
    },
    NotExistential {
        variable: usize,
    },
    NotUniversal {
        variable: usize,
    },
    DuplicateHerbrandDependency {
        universal: usize,
        dependency: usize,
    },
    HerbrandDependencyCountExceedsVariableCount {
        universal: usize,
        required: usize,
        variable_count: usize,
    },
    HerbrandDependencyNotAvailable {
        universal: usize,
        dependency: usize,
    },
    StrategyDependencyOrderMismatch {
        variable: usize,
        dependency: usize,
    },
    DuplicateSkolemDependency {
        existential: usize,
        dependency: usize,
    },
    SkolemDependencyOrderMismatch {
        existential: usize,
        dependency: usize,
    },
    SkolemDependencyCountExceedsVariableCount {
        existential: usize,
        required: usize,
        variable_count: usize,
    },
    SkolemDependencyNotAvailable {
        existential: usize,
        dependency: usize,
    },
    ResourceAccountingOverflow,
}

impl ValidatedQuantifierPrefix {
    /// Validate that the blocks quantify every declared variable exactly once.
    ///
    /// An empty prefix is valid only for a zero-variable formula. The input
    /// block order and each block's variable order are preserved.
    pub fn new(
        variable_count: usize,
        blocks: Vec<QuantifierBlock>,
        budget: QuantifierPrefixBudget,
    ) -> Result<Self, QuantifierScopeError> {
        if variable_count > budget.max_variables {
            return Err(QuantifierScopeError::VariableBudgetExceeded {
                required: variable_count,
                limit: budget.max_variables,
            });
        }
        if blocks.len() > budget.max_blocks {
            return Err(QuantifierScopeError::BlockBudgetExceeded {
                required: blocks.len(),
                limit: budget.max_blocks,
            });
        }

        let occurrence_count =
            blocks
                .iter()
                .enumerate()
                .try_fold(0usize, |total, (block_index, block)| {
                    if block.variables.is_empty() {
                        return Err(QuantifierScopeError::EmptyQuantifierBlock {
                            block: block_index,
                        });
                    }
                    total
                        .checked_add(block.variables.len())
                        .ok_or(QuantifierScopeError::ResourceAccountingOverflow)
                })?;

        if occurrence_count > budget.max_variable_occurrences {
            return Err(QuantifierScopeError::VariableOccurrenceBudgetExceeded {
                required: occurrence_count,
                limit: budget.max_variable_occurrences,
            });
        }
        if occurrence_count != variable_count {
            return Err(QuantifierScopeError::VariableOccurrenceCountMismatch {
                expected: variable_count,
                actual: occurrence_count,
            });
        }

        let mut locations = vec![(usize::MAX, Quantifier::Exists); variable_count];
        for (block_index, block) in blocks.iter().enumerate() {
            if block.variables.is_empty() {
                return Err(QuantifierScopeError::EmptyQuantifierBlock { block: block_index });
            }
            for &variable in &block.variables {
                if variable >= variable_count {
                    return Err(QuantifierScopeError::QuantifiedVariableOutOfRange {
                        block: block_index,
                        variable,
                        variable_count,
                    });
                }
                if locations[variable].0 != usize::MAX {
                    return Err(QuantifierScopeError::VariableRepeated { variable });
                }
                locations[variable] = (block_index, block.quantifier);
            }
        }

        for (variable, (block, _)) in locations.iter().enumerate() {
            if *block == usize::MAX {
                return Err(QuantifierScopeError::VariableMissing { variable });
            }
        }

        Ok(Self {
            variable_count,
            blocks,
            locations,
        })
    }

    /// Number of variables quantified by this prefix.
    pub const fn variable_count(&self) -> usize {
        self.variable_count
    }

    /// Ordered quantifier blocks.
    pub fn blocks(&self) -> &[QuantifierBlock] {
        &self.blocks
    }

    /// Return all universal variables preceding an existential variable.
    ///
    /// This is the standard prenex-QBF Skolem dependency scope. Variables in
    /// the same existential block do not depend on one another. An existential
    /// may use any subset of the returned scope.
    pub fn existential_dependencies(
        &self,
        variable: usize,
    ) -> Result<Vec<usize>, QuantifierScopeError> {
        let Some(&(target_block, quantifier)) = self.locations.get(variable) else {
            return Err(QuantifierScopeError::QuantifiedVariableOutOfRange {
                block: self.blocks.len(),
                variable,
                variable_count: self.variable_count,
            });
        };
        if quantifier != Quantifier::Exists {
            return Err(QuantifierScopeError::NotExistential { variable });
        }

        let mut dependencies = Vec::new();
        for block in self.blocks.iter().take(target_block) {
            if block.quantifier == Quantifier::ForAll {
                dependencies.extend(block.variables.iter().copied());
            }
        }
        Ok(dependencies)
    }

    /// Return all existential variables preceding a universal variable.
    ///
    /// This is the dual Herbrand dependency scope. Variables in the same
    /// universal block do not depend on one another.
    pub fn universal_dependencies(
        &self,
        variable: usize,
    ) -> Result<Vec<usize>, QuantifierScopeError> {
        let Some(&(target_block, quantifier)) = self.locations.get(variable) else {
            return Err(QuantifierScopeError::QuantifiedVariableOutOfRange {
                block: self.blocks.len(),
                variable,
                variable_count: self.variable_count,
            });
        };
        if quantifier != Quantifier::ForAll {
            return Err(QuantifierScopeError::NotUniversal { variable });
        }

        let mut dependencies = Vec::new();
        for block in self.blocks.iter().take(target_block) {
            if block.quantifier == Quantifier::Exists {
                dependencies.extend(block.variables.iter().copied());
            }
        }
        Ok(dependencies)
    }

    /// Validate a declared Herbrand dependency set against the prefix.
    ///
    /// A universal strategy may ignore available earlier existential
    /// variables. Dependencies are a duplicate-free subset, listed in prefix
    /// order so that truth-table keys have a canonical meaning.
    pub fn validate_herbrand_dependencies(
        &self,
        universal: usize,
        declared: &[usize],
    ) -> Result<(), QuantifierScopeError> {
        let Some(&(target_block, quantifier)) = self.locations.get(universal) else {
            return Err(QuantifierScopeError::QuantifiedVariableOutOfRange {
                block: self.blocks.len(),
                variable: universal,
                variable_count: self.variable_count,
            });
        };
        if quantifier != Quantifier::ForAll {
            return Err(QuantifierScopeError::NotUniversal {
                variable: universal,
            });
        }
        if declared.len() > self.variable_count {
            return Err(
                QuantifierScopeError::HerbrandDependencyCountExceedsVariableCount {
                    universal,
                    required: declared.len(),
                    variable_count: self.variable_count,
                },
            );
        }

        let mut seen = vec![false; self.variable_count];
        for &dependency in declared {
            if dependency >= self.variable_count {
                return Err(QuantifierScopeError::HerbrandDependencyNotAvailable {
                    universal,
                    dependency,
                });
            }
            if seen[dependency] {
                return Err(QuantifierScopeError::DuplicateHerbrandDependency {
                    universal,
                    dependency,
                });
            }
            seen[dependency] = true;
            let (dependency_block, dependency_quantifier) = self.locations[dependency];
            if dependency_quantifier != Quantifier::Exists || dependency_block >= target_block {
                return Err(QuantifierScopeError::HerbrandDependencyNotAvailable {
                    universal,
                    dependency,
                });
            }
        }

        let allowed = self.universal_dependencies(universal)?;
        let mut cursor = 0usize;
        for &dependency in declared {
            let Some(offset) = allowed[cursor..]
                .iter()
                .position(|candidate| *candidate == dependency)
            else {
                return Err(QuantifierScopeError::StrategyDependencyOrderMismatch {
                    variable: universal,
                    dependency,
                });
            };
            cursor = cursor.saturating_add(offset).saturating_add(1);
        }
        Ok(())
    }

    /// Validate a declared Skolem dependency set against the prefix.
    ///
    /// A Skolem function may ignore available earlier universal variables, so
    /// declared dependencies need only be a duplicate-free subset of scope.
    pub fn validate_skolem_dependencies(
        &self,
        existential: usize,
        declared: &[usize],
    ) -> Result<(), QuantifierScopeError> {
        let Some(&(target_block, quantifier)) = self.locations.get(existential) else {
            return Err(QuantifierScopeError::QuantifiedVariableOutOfRange {
                block: self.blocks.len(),
                variable: existential,
                variable_count: self.variable_count,
            });
        };
        if quantifier != Quantifier::Exists {
            return Err(QuantifierScopeError::NotExistential {
                variable: existential,
            });
        }
        if declared.len() > self.variable_count {
            return Err(
                QuantifierScopeError::SkolemDependencyCountExceedsVariableCount {
                    existential,
                    required: declared.len(),
                    variable_count: self.variable_count,
                },
            );
        }

        let mut seen = vec![false; self.variable_count];
        for &dependency in declared {
            if dependency >= self.variable_count {
                return Err(QuantifierScopeError::SkolemDependencyNotAvailable {
                    existential,
                    dependency,
                });
            }
            if seen[dependency] {
                return Err(QuantifierScopeError::DuplicateSkolemDependency {
                    existential,
                    dependency,
                });
            }
            seen[dependency] = true;
            let (dependency_block, dependency_quantifier) = self.locations[dependency];
            if dependency_quantifier != Quantifier::ForAll || dependency_block >= target_block {
                return Err(QuantifierScopeError::SkolemDependencyNotAvailable {
                    existential,
                    dependency,
                });
            }
        }

        let allowed = self.existential_dependencies(existential)?;
        let mut cursor = 0usize;
        for &dependency in declared {
            let Some(offset) = allowed[cursor..]
                .iter()
                .position(|candidate| *candidate == dependency)
            else {
                return Err(QuantifierScopeError::SkolemDependencyOrderMismatch {
                    existential,
                    dependency,
                });
            };
            cursor = cursor.saturating_add(offset).saturating_add(1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn standard_prefix() -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(
            4,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
                block(Quantifier::Exists, &[3]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap()
    }

    #[test]
    fn existential_dependencies_follow_prefix_order() {
        let prefix = standard_prefix();
        assert_eq!(prefix.existential_dependencies(1).unwrap(), vec![0]);
        assert_eq!(prefix.existential_dependencies(3).unwrap(), vec![0, 2]);
    }

    #[test]
    fn existential_may_ignore_available_universal_dependencies() {
        let prefix = standard_prefix();
        assert!(prefix.validate_skolem_dependencies(3, &[2]).is_ok());
        assert!(prefix.validate_skolem_dependencies(3, &[]).is_ok());
    }

    #[test]
    fn existential_cannot_depend_on_future_universal() {
        let prefix = standard_prefix();
        assert_eq!(
            prefix.validate_skolem_dependencies(1, &[2]).unwrap_err(),
            QuantifierScopeError::SkolemDependencyNotAvailable {
                existential: 1,
                dependency: 2,
            }
        );
    }

    #[test]
    fn skolem_dependencies_must_use_canonical_prefix_order() {
        let prefix = ValidatedQuantifierPrefix::new(
            4,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
                block(Quantifier::Exists, &[3]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap();
        assert_eq!(
            prefix.validate_skolem_dependencies(3, &[2, 0]).unwrap_err(),
            QuantifierScopeError::SkolemDependencyOrderMismatch {
                existential: 3,
                dependency: 0,
            }
        );
    }

    #[test]
    fn existential_rejects_duplicate_skolem_dependencies() {
        let prefix = standard_prefix();
        assert_eq!(
            prefix.validate_skolem_dependencies(3, &[0, 0]).unwrap_err(),
            QuantifierScopeError::DuplicateSkolemDependency {
                existential: 3,
                dependency: 0,
            }
        );
    }

    #[test]
    fn existential_cannot_depend_on_another_variable_in_its_block() {
        let prefix = ValidatedQuantifierPrefix::new(
            3,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1, 2]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap();
        assert_eq!(prefix.existential_dependencies(1).unwrap(), vec![0]);
        assert_eq!(prefix.existential_dependencies(2).unwrap(), vec![0]);
    }

    #[test]
    fn universal_dependencies_follow_prefix_order() {
        let prefix = standard_prefix();
        assert_eq!(prefix.universal_dependencies(0).unwrap(), vec![]);
        assert_eq!(prefix.universal_dependencies(2).unwrap(), vec![1]);
    }

    #[test]
    fn universal_may_ignore_available_existential_dependencies() {
        let prefix = ValidatedQuantifierPrefix::new(
            4,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
                block(Quantifier::ForAll, &[3]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap();
        assert!(prefix.validate_herbrand_dependencies(3, &[0]).is_ok());
        assert!(prefix.validate_herbrand_dependencies(3, &[]).is_ok());
    }

    #[test]
    fn universal_cannot_depend_on_future_existential() {
        let prefix = standard_prefix();
        assert_eq!(
            prefix.validate_herbrand_dependencies(2, &[3]).unwrap_err(),
            QuantifierScopeError::HerbrandDependencyNotAvailable {
                universal: 2,
                dependency: 3,
            }
        );
    }

    #[test]
    fn universal_rejects_duplicate_herbrand_dependencies() {
        let prefix = ValidatedQuantifierPrefix::new(
            3,
            vec![
                block(Quantifier::Exists, &[0, 1]),
                block(Quantifier::ForAll, &[2]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap();
        assert_eq!(
            prefix
                .validate_herbrand_dependencies(2, &[0, 0])
                .unwrap_err(),
            QuantifierScopeError::DuplicateHerbrandDependency {
                universal: 2,
                dependency: 0,
            }
        );
    }

    #[test]
    fn strategy_dependencies_must_use_canonical_prefix_order() {
        let prefix = ValidatedQuantifierPrefix::new(
            4,
            vec![
                block(Quantifier::Exists, &[0]),
                block(Quantifier::ForAll, &[1]),
                block(Quantifier::Exists, &[2]),
                block(Quantifier::ForAll, &[3]),
            ],
            QuantifierPrefixBudget::default(),
        )
        .unwrap();
        assert_eq!(
            prefix
                .validate_herbrand_dependencies(3, &[2, 0])
                .unwrap_err(),
            QuantifierScopeError::StrategyDependencyOrderMismatch {
                variable: 3,
                dependency: 0,
            }
        );
    }

    #[test]
    fn universal_variable_has_no_skolem_function_scope() {
        let prefix = standard_prefix();
        assert_eq!(
            prefix.existential_dependencies(2).unwrap_err(),
            QuantifierScopeError::NotExistential { variable: 2 }
        );
    }

    #[test]
    fn prefix_rejects_duplicate_variable() {
        assert_eq!(
            ValidatedQuantifierPrefix::new(
                2,
                vec![
                    block(Quantifier::ForAll, &[0]),
                    block(Quantifier::Exists, &[0]),
                ],
                QuantifierPrefixBudget::default(),
            )
            .unwrap_err(),
            QuantifierScopeError::VariableRepeated { variable: 0 }
        );
    }

    #[test]
    fn prefix_rejects_missing_variable() {
        assert_eq!(
            ValidatedQuantifierPrefix::new(
                2,
                vec![block(Quantifier::ForAll, &[0])],
                QuantifierPrefixBudget::default(),
            )
            .unwrap_err(),
            QuantifierScopeError::VariableOccurrenceCountMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn prefix_rejects_out_of_range_variable() {
        assert_eq!(
            ValidatedQuantifierPrefix::new(
                1,
                vec![block(Quantifier::Exists, &[1])],
                QuantifierPrefixBudget::default(),
            )
            .unwrap_err(),
            QuantifierScopeError::QuantifiedVariableOutOfRange {
                block: 0,
                variable: 1,
                variable_count: 1,
            }
        );
    }

    #[test]
    fn prefix_rejects_empty_blocks() {
        assert_eq!(
            ValidatedQuantifierPrefix::new(
                0,
                vec![block(Quantifier::Exists, &[])],
                QuantifierPrefixBudget::default(),
            )
            .unwrap_err(),
            QuantifierScopeError::EmptyQuantifierBlock { block: 0 }
        );
    }

    #[test]
    fn prefix_resource_budgets_are_fail_closed() {
        let budget = QuantifierPrefixBudget {
            max_variables: 2,
            max_blocks: 1,
            max_variable_occurrences: 2,
        };
        assert_eq!(
            ValidatedQuantifierPrefix::new(3, vec![block(Quantifier::Exists, &[0, 1, 2])], budget,)
                .unwrap_err(),
            QuantifierScopeError::VariableBudgetExceeded {
                required: 3,
                limit: 2,
            }
        );
    }

    #[test]
    fn exhaustive_three_variable_prefixes_preserve_skolem_dependencies() {
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        for order in orders {
            for quantifier_mask in 0..8usize {
                let blocks: Vec<QuantifierBlock> = order
                    .iter()
                    .enumerate()
                    .map(|(index, &variable)| QuantifierBlock {
                        quantifier: if ((quantifier_mask >> index) & 1) == 0 {
                            Quantifier::ForAll
                        } else {
                            Quantifier::Exists
                        },
                        variables: vec![variable],
                    })
                    .collect();
                let prefix = ValidatedQuantifierPrefix::new(
                    3,
                    blocks.clone(),
                    QuantifierPrefixBudget::default(),
                )
                .unwrap();

                for (position, block) in blocks.iter().enumerate() {
                    let variable = block.variables[0];
                    if block.quantifier == Quantifier::Exists {
                        let expected: Vec<usize> = blocks[..position]
                            .iter()
                            .filter(|previous| previous.quantifier == Quantifier::ForAll)
                            .flat_map(|previous| previous.variables.iter().copied())
                            .collect();
                        assert_eq!(
                            prefix.existential_dependencies(variable).unwrap(),
                            expected,
                            "quantifier mask {quantifier_mask}, order {order:?}, variable {variable}"
                        );
                    } else {
                        assert_eq!(
                            prefix.existential_dependencies(variable).unwrap_err(),
                            QuantifierScopeError::NotExistential { variable }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_prefix_is_valid_for_empty_universe() {
        let prefix =
            ValidatedQuantifierPrefix::new(0, vec![], QuantifierPrefixBudget::default()).unwrap();
        assert_eq!(prefix.variable_count(), 0);
        assert!(prefix.blocks().is_empty());
    }
}
