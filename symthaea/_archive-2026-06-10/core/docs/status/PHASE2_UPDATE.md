# Phase 2 Documentation Update: Conjecture Engine

Date: 2026-06-16

## 1. Instrumentation: SMT Witness Persistence
The `conjecture_engine` has been instrumented to persist SMT-LIB2 queries as witness files in the workspace root. 
- **Purpose:** Enables forensic analysis and debugging of formal proof queries generated for complex symbolic regression conjectures.
- **Implementation:** Added file system write logic in `crates/core/symthaea-core/src/hdc/conjecture_engine/verification.rs`. Queries are saved as `witness_{source}_{tested_points}.smt2`.

## 2. Grammar Extension: Nested Radicals
Extended the symbolic expression grammar to support nested radical operations.
- **New Operator:** `UnaryFn::SqrtNested` (computes `sqrt(sqrt(x))`).
- **AST:** Updated `Expr` enum and supporting logic.
- **Evaluation:** Added evaluation logic to `Expr::eval`.
- **Rendering:** Updated `Display` (CLI) and `expr_to_latex` (Paper-ready LaTeX).
- **Generation:** Included in `random_expr` (univariate) and `random_expr_multivar` (autonomous multivariate) to increase the diversity of generated conjectures.
- **Mutation:** Integrated into `mutate_multivar` to allow the evolution of nested radical invariants.
- **Verification:** Added `test_nested_sqrt_evaluation` in `crates/core/symthaea-core/src/hdc/conjecture_engine/expressions.rs` for validation.
