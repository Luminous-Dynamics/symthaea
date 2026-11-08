// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict, bounded QDIMACS PCNF ingestion.
//!
//! Accepts one p cnf header, ordered a/e prefix lines and zero-terminated
//! clauses. Adjacent same-quantifier lines are merged. Every declared
//! variable must occur exactly once in the prefix. DQDIMACS, QCIR, non-CNF
//! matrices and implicit/free variables are rejected rather than guessed.

use super::decomposition::{CnfProblem, Literal};
use super::quantified_scope::{
    Quantifier, QuantifierBlock, QuantifierPrefixBudget, QuantifierScopeError,
    ValidatedQuantifierPrefix,
};

/// Deterministic parser limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QdimacsParseBudget {
    pub max_input_bytes: usize,
    pub max_lines: usize,
    pub max_variables: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub prefix: QuantifierPrefixBudget,
}

impl Default for QdimacsParseBudget {
    fn default() -> Self {
        Self {
            max_input_bytes: 64 * 1024 * 1024,
            max_lines: 1 << 20,
            max_variables: 1 << 20,
            max_clauses: 1 << 20,
            max_literals: 1 << 24,
            prefix: QuantifierPrefixBudget::default(),
        }
    }
}

/// Source counters retained beside the parsed structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QdimacsSourceStats {
    pub source_bytes: usize,
    pub source_lines: usize,
    pub declared_variables: usize,
    pub declared_clauses: usize,
    pub parsed_clauses: usize,
    pub parsed_literals: usize,
    pub quantifier_blocks: usize,
    /// Variables omitted from the explicit prefix and existentially closed
    /// at the innermost position, as required by QDIMACS semantics.
    pub implicitly_existential_variables: usize,
    /// Largest 1-based variable ID in a matrix literal; zero if none.
    pub max_literal_variable_id: usize,
}

/// A parsed QDIMACS PCNF instance with its exact ordered quantifier prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QdimacsInstance {
    pub problem: CnfProblem,
    pub prefix: ValidatedQuantifierPrefix,
    pub stats: QdimacsSourceStats,
}

/// Typed, fail-closed parser errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QdimacsError {
    InputByteBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LineBudgetExceeded {
        required: usize,
        limit: usize,
    },
    MissingHeader,
    HeaderMustBeFirst {
        line: usize,
    },
    DuplicateHeader {
        line: usize,
    },
    MalformedHeader {
        line: usize,
    },
    UnsupportedFormat {
        line: usize,
        format: String,
    },
    InvalidHeaderVariableCount {
        line: usize,
        token: String,
    },
    InvalidHeaderClauseCount {
        line: usize,
        token: String,
    },
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ClauseBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralBudgetExceeded {
        required: usize,
        limit: usize,
    },
    PrefixBlockBudgetExceeded {
        required: usize,
        limit: usize,
    },
    PrefixVariableOccurrenceBudgetExceeded {
        required: usize,
        limit: usize,
    },
    MalformedQuantifierLine {
        line: usize,
    },
    EmptyQuantifierBlock {
        line: usize,
    },
    InvalidQuantifierVariable {
        line: usize,
        token: String,
    },
    QuantifierVariableOutOfRange {
        line: usize,
        variable: usize,
        variable_count: usize,
    },
    DuplicateQuantifiedVariable {
        line: usize,
        variable: usize,
    },
    QuantifierAfterMatrix {
        line: usize,
    },
    UnsupportedDependencyLine {
        line: usize,
        marker: String,
    },
    InvalidLiteral {
        line: usize,
        token: String,
    },
    LiteralVariableOutOfRange {
        line: usize,
        variable: usize,
        variable_count: usize,
    },
    UnterminatedClause {
        line: usize,
        literals_in_clause: usize,
    },
    TooManyClauses {
        line: usize,
        declared: usize,
        observed_at_least: usize,
    },
    ClauseCountMismatch {
        declared: usize,
        actual: usize,
    },
    ResourceAccountingOverflow,
    InvalidQuantifierPrefix(QuantifierScopeError),
}

#[derive(Debug, Clone, Copy)]
struct Header {
    variables: usize,
    clauses: usize,
}

fn parse_header(line_number: usize, tokens: &[&str]) -> Result<Header, QdimacsError> {
    if tokens.len() != 4 || tokens[0] != "p" {
        return Err(QdimacsError::MalformedHeader { line: line_number });
    }
    if tokens[1] != "cnf" {
        return Err(QdimacsError::UnsupportedFormat {
            line: line_number,
            format: tokens[1].to_owned(),
        });
    }
    let variables =
        tokens[2]
            .parse::<i32>()
            .map_err(|_| QdimacsError::InvalidHeaderVariableCount {
                line: line_number,
                token: tokens[2].to_owned(),
            })?;
    let clauses = tokens[3]
        .parse::<i32>()
        .map_err(|_| QdimacsError::InvalidHeaderClauseCount {
            line: line_number,
            token: tokens[3].to_owned(),
        })?;
    let variables =
        usize::try_from(variables).map_err(|_| QdimacsError::InvalidHeaderVariableCount {
            line: line_number,
            token: tokens[2].to_owned(),
        })?;
    let clauses = usize::try_from(clauses).map_err(|_| QdimacsError::InvalidHeaderClauseCount {
        line: line_number,
        token: tokens[3].to_owned(),
    })?;
    Ok(Header { variables, clauses })
}

fn parse_quantifier_line(
    line_number: usize,
    tokens: &[&str],
    header: Header,
    seen_variables: &mut [bool],
    blocks: &mut Vec<QuantifierBlock>,
    occurrences: &mut usize,
    budget: QdimacsParseBudget,
) -> Result<(), QdimacsError> {
    let quantifier = match tokens.first().copied() {
        Some("a") => Quantifier::ForAll,
        Some("e") => Quantifier::Exists,
        _ => return Err(QdimacsError::MalformedQuantifierLine { line: line_number }),
    };

    if tokens.len() < 2 || tokens.last() != Some(&"0") {
        return Err(QdimacsError::MalformedQuantifierLine { line: line_number });
    }
    if tokens[1..tokens.len() - 1].contains(&"0") {
        return Err(QdimacsError::MalformedQuantifierLine { line: line_number });
    }
    if tokens.len() == 2 {
        return Err(QdimacsError::EmptyQuantifierBlock { line: line_number });
    }

    for token in &tokens[1..tokens.len() - 1] {
        let variable =
            token
                .parse::<i32>()
                .map_err(|_| QdimacsError::InvalidQuantifierVariable {
                    line: line_number,
                    token: (*token).to_owned(),
                })?;
        if variable <= 0 || usize::try_from(variable).unwrap_or(usize::MAX) > header.variables {
            return Err(QdimacsError::QuantifierVariableOutOfRange {
                line: line_number,
                variable: usize::try_from(variable).unwrap_or(0),
                variable_count: header.variables,
            });
        }
        let index =
            usize::try_from(variable - 1).map_err(|_| QdimacsError::InvalidQuantifierVariable {
                line: line_number,
                token: (*token).to_owned(),
            })?;
        if seen_variables[index] {
            return Err(QdimacsError::DuplicateQuantifiedVariable {
                line: line_number,
                variable: index + 1,
            });
        }

        let next_occurrences = occurrences
            .checked_add(1)
            .ok_or(QdimacsError::ResourceAccountingOverflow)?;
        if next_occurrences > budget.prefix.max_variable_occurrences {
            return Err(QdimacsError::PrefixVariableOccurrenceBudgetExceeded {
                required: next_occurrences,
                limit: budget.prefix.max_variable_occurrences,
            });
        }

        let append_to_last = blocks
            .last()
            .is_some_and(|last| last.quantifier == quantifier);
        if !append_to_last {
            let next_blocks = blocks
                .len()
                .checked_add(1)
                .ok_or(QdimacsError::ResourceAccountingOverflow)?;
            if next_blocks > budget.prefix.max_blocks {
                return Err(QdimacsError::PrefixBlockBudgetExceeded {
                    required: next_blocks,
                    limit: budget.prefix.max_blocks,
                });
            }
            blocks.push(QuantifierBlock {
                quantifier,
                variables: Vec::new(),
            });
        }
        blocks
            .last_mut()
            .expect("block was created or already exists")
            .variables
            .push(index);
        seen_variables[index] = true;
        *occurrences = next_occurrences;
    }
    Ok(())
}

fn parse_clause_token(
    line_number: usize,
    token: &str,
    variable_count: usize,
) -> Result<Literal, QdimacsError> {
    let signed = token
        .parse::<i32>()
        .map_err(|_| QdimacsError::InvalidLiteral {
            line: line_number,
            token: token.to_owned(),
        })?;
    if signed == 0 {
        return Err(QdimacsError::InvalidLiteral {
            line: line_number,
            token: token.to_owned(),
        });
    }
    let absolute = signed
        .checked_abs()
        .ok_or_else(|| QdimacsError::InvalidLiteral {
            line: line_number,
            token: token.to_owned(),
        })?;
    let variable =
        usize::try_from(absolute).map_err(|_| QdimacsError::LiteralVariableOutOfRange {
            line: line_number,
            variable: usize::MAX,
            variable_count,
        })?;
    if variable == 0 || variable > variable_count {
        return Err(QdimacsError::LiteralVariableOutOfRange {
            line: line_number,
            variable,
            variable_count,
        });
    }
    Ok(Literal {
        variable: variable - 1,
        positive: signed > 0,
    })
}

fn is_comment(line: &str) -> bool {
    line == "c" || line.starts_with("c ") || line.starts_with("c\t")
}

/// Parse a strict QDIMACS PCNF instance.
///
/// Comments and blank lines are ignored. Clause data may span lines and may
/// contain multiple zero-terminated clauses on one line. Quantifier records
/// are line-oriented. The source is not rewritten or normalized.
pub fn parse_qdimacs(
    source: &str,
    budget: QdimacsParseBudget,
) -> Result<QdimacsInstance, QdimacsError> {
    if source.len() > budget.max_input_bytes {
        return Err(QdimacsError::InputByteBudgetExceeded {
            required: source.len(),
            limit: budget.max_input_bytes,
        });
    }
    let source_lines = source.lines().count();
    if source_lines > budget.max_lines {
        return Err(QdimacsError::LineBudgetExceeded {
            required: source_lines,
            limit: budget.max_lines,
        });
    }

    let mut header = None;
    let mut seen_variables = Vec::new();
    let mut blocks = Vec::new();
    let mut prefix_occurrences = 0usize;
    let mut matrix_started = false;
    let mut clauses = Vec::new();
    let mut current_clause = Vec::new();
    let mut literal_count = 0usize;
    let mut max_literal_variable_id = 0usize;
    let mut last_data_line = 0usize;

    for (zero_line, raw_line) in source.lines().enumerate() {
        let line_number = zero_line
            .checked_add(1)
            .ok_or(QdimacsError::ResourceAccountingOverflow)?;
        let line = if zero_line == 0 {
            raw_line.trim_start_matches('\u{feff}').trim()
        } else {
            raw_line.trim()
        };
        if line.is_empty() || is_comment(line) {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let first = tokens.first().copied().unwrap_or_default();

        if header.is_none() && first != "p" {
            return Err(QdimacsError::HeaderMustBeFirst { line: line_number });
        }
        if first == "p" {
            if header.is_some() {
                return Err(QdimacsError::DuplicateHeader { line: line_number });
            }
            let parsed = parse_header(line_number, &tokens)?;
            let variable_limit = budget.max_variables.min(budget.prefix.max_variables);
            if parsed.variables > variable_limit {
                return Err(QdimacsError::VariableBudgetExceeded {
                    required: parsed.variables,
                    limit: variable_limit,
                });
            }
            if parsed.clauses > budget.max_clauses {
                return Err(QdimacsError::ClauseBudgetExceeded {
                    required: parsed.clauses,
                    limit: budget.max_clauses,
                });
            }
            seen_variables = vec![false; parsed.variables];
            clauses = Vec::with_capacity(parsed.clauses);
            header = Some(parsed);
            continue;
        }

        let parsed_header = header.ok_or(QdimacsError::MissingHeader)?;
        last_data_line = line_number;

        match first {
            "a" | "e" => {
                if matrix_started {
                    return Err(QdimacsError::QuantifierAfterMatrix { line: line_number });
                }
                parse_quantifier_line(
                    line_number,
                    &tokens,
                    parsed_header,
                    &mut seen_variables,
                    &mut blocks,
                    &mut prefix_occurrences,
                    budget,
                )?;
            }
            "d" | "r" => {
                return Err(QdimacsError::UnsupportedDependencyLine {
                    line: line_number,
                    marker: first.to_owned(),
                });
            }
            _ => {
                matrix_started = true;
                for token in tokens {
                    if token == "a" || token == "e" {
                        return Err(QdimacsError::QuantifierAfterMatrix { line: line_number });
                    }
                    if token == "d" || token == "r" {
                        return Err(QdimacsError::UnsupportedDependencyLine {
                            line: line_number,
                            marker: token.to_owned(),
                        });
                    }
                    if token == "0" {
                        let next_clause_count = clauses
                            .len()
                            .checked_add(1)
                            .ok_or(QdimacsError::ResourceAccountingOverflow)?;
                        if next_clause_count > parsed_header.clauses {
                            return Err(QdimacsError::TooManyClauses {
                                line: line_number,
                                declared: parsed_header.clauses,
                                observed_at_least: next_clause_count,
                            });
                        }
                        clauses.push(std::mem::take(&mut current_clause));
                        continue;
                    }

                    let literal = parse_clause_token(line_number, token, parsed_header.variables)?;
                    literal_count = literal_count
                        .checked_add(1)
                        .ok_or(QdimacsError::ResourceAccountingOverflow)?;
                    if literal_count > budget.max_literals {
                        return Err(QdimacsError::LiteralBudgetExceeded {
                            required: literal_count,
                            limit: budget.max_literals,
                        });
                    }
                    max_literal_variable_id =
                        max_literal_variable_id.max(literal.variable.saturating_add(1));
                    current_clause.push(literal);
                }
            }
        }
    }

    let Some(parsed_header) = header else {
        return Err(QdimacsError::MissingHeader);
    };
    if !current_clause.is_empty() {
        return Err(QdimacsError::UnterminatedClause {
            line: last_data_line,
            literals_in_clause: current_clause.len(),
        });
    }
    if clauses.len() != parsed_header.clauses {
        return Err(QdimacsError::ClauseCountMismatch {
            declared: parsed_header.clauses,
            actual: clauses.len(),
        });
    }
    let implicit_variables: Vec<usize> = seen_variables
        .iter()
        .enumerate()
        .filter_map(|(index, seen)| (!seen).then_some(index))
        .collect();
    let total_prefix_occurrences = prefix_occurrences
        .checked_add(implicit_variables.len())
        .ok_or(QdimacsError::ResourceAccountingOverflow)?;
    if total_prefix_occurrences > budget.prefix.max_variable_occurrences {
        return Err(QdimacsError::PrefixVariableOccurrenceBudgetExceeded {
            required: total_prefix_occurrences,
            limit: budget.prefix.max_variable_occurrences,
        });
    }

    // QDIMACS assigns existential semantics to variables omitted from the
    // explicit prefix. Existential closure places them after the stated prefix.
    if !implicit_variables.is_empty() {
        if blocks
            .last()
            .is_some_and(|last| last.quantifier == Quantifier::Exists)
        {
            blocks
                .last_mut()
                .expect("last block was checked")
                .variables
                .extend(implicit_variables.iter().copied());
        } else {
            let required = blocks
                .len()
                .checked_add(1)
                .ok_or(QdimacsError::ResourceAccountingOverflow)?;
            if required > budget.prefix.max_blocks {
                return Err(QdimacsError::PrefixBlockBudgetExceeded {
                    required,
                    limit: budget.prefix.max_blocks,
                });
            }
            blocks.push(QuantifierBlock {
                quantifier: Quantifier::Exists,
                variables: implicit_variables.clone(),
            });
        }
    }

    let prefix = ValidatedQuantifierPrefix::new(parsed_header.variables, blocks, budget.prefix)
        .map_err(QdimacsError::InvalidQuantifierPrefix)?;

    let variables = (1..=parsed_header.variables)
        .map(|id| id.to_string())
        .collect();
    let problem = CnfProblem { variables, clauses };
    let stats = QdimacsSourceStats {
        source_bytes: source.len(),
        source_lines,
        declared_variables: parsed_header.variables,
        declared_clauses: parsed_header.clauses,
        parsed_clauses: clauses_len(&problem),
        parsed_literals: literal_count,
        quantifier_blocks: prefix.blocks().len(),
        implicitly_existential_variables: implicit_variables.len(),
        max_literal_variable_id,
    };

    Ok(QdimacsInstance {
        problem,
        prefix,
        stats,
    })
}

fn clauses_len(problem: &CnfProblem) -> usize {
    problem.clauses.len()
}

#[cfg(test)]
mod tests {
    use super::super::quantified_proof::{
        QuantifiedProofBudget, build_qbf_game_proof, verify_qbf_game_proof,
    };
    use super::super::quantified_strategy::{
        QuantifiedEvaluationBudget, StrategyOwner, evaluate_qbf_with_strategy,
    };
    use super::*;

    fn budget() -> QdimacsParseBudget {
        QdimacsParseBudget {
            max_input_bytes: 4096,
            max_lines: 128,
            max_variables: 16,
            max_clauses: 32,
            max_literals: 128,
            prefix: QuantifierPrefixBudget {
                max_variables: 16,
                max_blocks: 16,
                max_variable_occurrences: 16,
            },
        }
    }

    #[test]
    fn parses_forall_exists_equality_and_preserves_order() {
        let source = "c equality\np cnf 2 2\na 1 0\ne 2 0\n-1 2 0\n1 -2 0\n";
        let parsed = parse_qdimacs(source, budget()).unwrap();
        assert_eq!(parsed.problem.variables, vec!["1", "2"]);
        assert_eq!(
            parsed.problem.clauses,
            vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ]
        );
        assert_eq!(
            parsed.prefix.blocks(),
            &[
                QuantifierBlock {
                    quantifier: Quantifier::ForAll,
                    variables: vec![0],
                },
                QuantifierBlock {
                    quantifier: Quantifier::Exists,
                    variables: vec![1],
                },
            ]
        );
        assert_eq!(parsed.stats.parsed_literals, 4);
        assert_eq!(parsed.stats.max_literal_variable_id, 2);
    }

    #[test]
    fn parsed_qdimacs_flows_through_independent_qbf_strategy_and_proof_verifiers() {
        let source = "p cnf 2 2\na 1 0\ne 2 0\n-1 2 0\n1 -2 0\n";
        let parsed = parse_qdimacs(source, budget()).unwrap();

        let result = evaluate_qbf_with_strategy(
            &parsed.problem,
            &parsed.prefix,
            QuantifiedEvaluationBudget::default(),
        )
        .unwrap();
        assert!(result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Existential);

        let (proof, _) = build_qbf_game_proof(
            &parsed.problem,
            &parsed.prefix,
            QuantifiedProofBudget::default(),
        )
        .unwrap();
        let proof_report = verify_qbf_game_proof(
            &parsed.problem,
            &parsed.prefix,
            &proof,
            QuantifiedProofBudget::default(),
        )
        .unwrap();
        assert!(proof_report.root_value);
        assert_eq!(proof_report.root_value, result.value);
    }

    #[test]
    fn implicit_existential_variables_are_used_as_innermost_skolem_dependencies() {
        // The omitted variable y is existentially closed after forall x.
        // forall x exists y: (x OR y) is true, with y(x) = !x.
        let parsed = parse_qdimacs("p cnf 2 1\na 1 0\n1 2 0\n", budget()).unwrap();
        let result = evaluate_qbf_with_strategy(
            &parsed.problem,
            &parsed.prefix,
            QuantifiedEvaluationBudget::default(),
        )
        .unwrap();
        assert!(result.value);
        assert_eq!(result.strategy.owner, StrategyOwner::Existential);
        let y = result
            .strategy
            .functions
            .iter()
            .find(|function| function.variable == 1)
            .unwrap();
        assert_eq!(y.dependencies, vec![0]);
        assert_eq!(y.table.get(&vec![false]), Some(&true));
        assert_eq!(y.table.get(&vec![true]), Some(&false));
    }

    #[test]
    fn merges_adjacent_quantifier_lines_with_same_quantifier() {
        let parsed = parse_qdimacs("p cnf 3 0\na 1 0\na 2 0\ne 3 0\n", budget()).unwrap();
        assert_eq!(parsed.prefix.blocks().len(), 2);
        assert_eq!(parsed.prefix.blocks()[0].variables, vec![0, 1]);
    }

    #[test]
    fn parses_clauses_spanning_lines_and_multiple_clauses_per_line() {
        let parsed = parse_qdimacs("p cnf 2 2\ne 1 2 0\n1\n-2 0 2 0\n", budget()).unwrap();
        assert_eq!(
            parsed.problem.clauses,
            vec![
                vec![Literal::positive(0), Literal::negative(1)],
                vec![Literal::positive(1)],
            ]
        );
    }

    #[test]
    fn preserves_explicit_empty_clause() {
        let parsed = parse_qdimacs("p cnf 1 1\na 1 0\n0\n", budget()).unwrap();
        assert_eq!(parsed.problem.clauses, vec![vec![]]);
    }

    #[test]
    fn accepts_empty_zero_variable_formula() {
        let parsed = parse_qdimacs("p cnf 0 0\n", budget()).unwrap();
        assert!(parsed.problem.variables.is_empty());
        assert!(parsed.problem.clauses.is_empty());
        assert!(parsed.prefix.blocks().is_empty());
    }

    #[test]
    fn ignores_full_line_comments_and_blank_lines() {
        let parsed = parse_qdimacs(
            "\n c preliminary\np cnf 1 1\nc prefix\ne 1 0\n\n1 0\n",
            budget(),
        )
        .unwrap();
        assert_eq!(parsed.problem.clauses.len(), 1);
    }

    #[test]
    fn rejects_missing_header() {
        assert_eq!(
            parse_qdimacs("c no header\n", budget()).unwrap_err(),
            QdimacsError::MissingHeader
        );
    }

    #[test]
    fn rejects_duplicate_header() {
        assert_eq!(
            parse_qdimacs("p cnf 0 0\np cnf 0 0\n", budget()).unwrap_err(),
            QdimacsError::DuplicateHeader { line: 2 }
        );
    }

    #[test]
    fn rejects_malformed_and_unsupported_headers() {
        assert_eq!(
            parse_qdimacs("p cnf 1\n", budget()).unwrap_err(),
            QdimacsError::MalformedHeader { line: 1 }
        );
        assert!(matches!(
            parse_qdimacs("p qbf 1 0\n", budget()),
            Err(QdimacsError::UnsupportedFormat { .. })
        ));
    }

    #[test]
    fn rejects_header_after_data() {
        assert_eq!(
            parse_qdimacs("c x\ne 1 0\np cnf 1 0\n", budget()).unwrap_err(),
            QdimacsError::HeaderMustBeFirst { line: 2 }
        );
    }

    #[test]
    fn rejects_invalid_header_counts() {
        assert!(matches!(
            parse_qdimacs("p cnf x 0\n", budget()),
            Err(QdimacsError::InvalidHeaderVariableCount { .. })
        ));
        assert!(matches!(
            parse_qdimacs("p cnf 1 y\n", budget()),
            Err(QdimacsError::InvalidHeaderClauseCount { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_quantifier_variables() {
        assert!(matches!(
            parse_qdimacs("p cnf 1 0\na 2 0\n", budget()),
            Err(QdimacsError::QuantifierVariableOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_quantifier_variables() {
        assert_eq!(
            parse_qdimacs("p cnf 2 0\na 1 0\ne 1 2 0\n", budget()).unwrap_err(),
            QdimacsError::DuplicateQuantifiedVariable {
                line: 3,
                variable: 1
            }
        );
    }

    #[test]
    fn unquantified_variables_receive_innermost_existential_semantics() {
        let parsed = parse_qdimacs("p cnf 3 0\na 1 0\n", budget()).unwrap();
        assert_eq!(parsed.stats.implicitly_existential_variables, 2);
        assert_eq!(
            parsed.prefix.blocks(),
            &[
                QuantifierBlock {
                    quantifier: Quantifier::ForAll,
                    variables: vec![0],
                },
                QuantifierBlock {
                    quantifier: Quantifier::Exists,
                    variables: vec![1, 2],
                },
            ]
        );
    }

    #[test]
    fn rejects_empty_quantifier_blocks_and_early_terminators() {
        assert_eq!(
            parse_qdimacs("p cnf 1 0\na 0\n", budget()).unwrap_err(),
            QdimacsError::EmptyQuantifierBlock { line: 2 }
        );
        assert_eq!(
            parse_qdimacs("p cnf 1 0\na 0 1 0\n", budget()).unwrap_err(),
            QdimacsError::MalformedQuantifierLine { line: 2 }
        );
    }

    #[test]
    fn rejects_quantifier_lines_after_matrix_start() {
        assert_eq!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n1 0\na 2 0\n", budget()).unwrap_err(),
            QdimacsError::QuantifierAfterMatrix { line: 4 }
        );
    }

    #[test]
    fn rejects_qbf_dependency_lines_instead_of_misparsing_them_as_clauses() {
        assert_eq!(
            parse_qdimacs("p cnf 1 0\ne 1 0\nd 1 0\n", budget()).unwrap_err(),
            QdimacsError::UnsupportedDependencyLine {
                line: 3,
                marker: "d".into()
            }
        );
    }

    #[test]
    fn rejects_invalid_and_out_of_range_literals() {
        assert!(matches!(
            parse_qdimacs("p cnf 1 1\ne 1 0\nx 0\n", budget()),
            Err(QdimacsError::InvalidLiteral { .. })
        ));
        assert_eq!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n-2 0\n", budget()).unwrap_err(),
            QdimacsError::LiteralVariableOutOfRange {
                line: 3,
                variable: 2,
                variable_count: 1
            }
        );
        assert!(matches!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n-2147483648 0\n", budget()),
            Err(QdimacsError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn rejects_unterminated_clause_and_clause_count_mismatch() {
        assert_eq!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n1\n", budget()).unwrap_err(),
            QdimacsError::UnterminatedClause {
                line: 3,
                literals_in_clause: 1
            }
        );
        assert_eq!(
            parse_qdimacs("p cnf 1 2\ne 1 0\n1 0\n", budget()).unwrap_err(),
            QdimacsError::ClauseCountMismatch {
                declared: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn rejects_extra_clauses() {
        assert!(matches!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n1 0 -1 0\n", budget()),
            Err(QdimacsError::TooManyClauses { .. })
        ));
    }

    #[test]
    fn rejects_resource_budgets_before_unbounded_growth() {
        let tight = QdimacsParseBudget {
            max_input_bytes: 8,
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 1 0\ne 1 0\n", tight),
            Err(QdimacsError::InputByteBudgetExceeded { .. })
        ));

        let tight = QdimacsParseBudget {
            max_lines: 1,
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 1 0\ne 1 0\n", tight),
            Err(QdimacsError::LineBudgetExceeded { .. })
        ));

        let tight = QdimacsParseBudget {
            max_variables: 1,
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 2 0\na 1 2 0\n", tight),
            Err(QdimacsError::VariableBudgetExceeded { .. })
        ));

        let tight = QdimacsParseBudget {
            max_clauses: 0,
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n1 0\n", tight),
            Err(QdimacsError::ClauseBudgetExceeded { .. })
        ));

        let tight = QdimacsParseBudget {
            max_literals: 1,
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 1 1\ne 1 0\n1 -1 0\n", tight),
            Err(QdimacsError::LiteralBudgetExceeded { .. })
        ));
    }

    #[test]
    fn rejects_prefix_block_and_occurrence_budgets() {
        let tight = QdimacsParseBudget {
            prefix: QuantifierPrefixBudget {
                max_variables: 4,
                max_blocks: 1,
                max_variable_occurrences: 4,
            },
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 2 0\na 1 0\ne 2 0\n", tight),
            Err(QdimacsError::PrefixBlockBudgetExceeded { .. })
        ));

        let tight = QdimacsParseBudget {
            prefix: QuantifierPrefixBudget {
                max_variables: 4,
                max_blocks: 4,
                max_variable_occurrences: 1,
            },
            ..budget()
        };
        assert!(matches!(
            parse_qdimacs("p cnf 2 0\ne 1 2 0\n", tight),
            Err(QdimacsError::PrefixVariableOccurrenceBudgetExceeded { .. })
        ));
    }
}
