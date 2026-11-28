//! Exact finite-outcome maximin testing oracle (M0).
//!
//! This evaluator solves the randomized-test primal and the least-favourable
//! mixture dual as separate rational linear programs. It uses bounded
//! vertex/basis enumeration, not a production-scale LP algorithm.
//!
//! For finite outcome laws A_i and B_j, the primal is:
//!   maximize beta
//!   beta <= sum_k A_i[k] * phi[k]       for every alternative i
//!   sum_k B_j[k] * phi[k] <= alpha      for every null j
//!   0 <= phi[k] <= 1, beta >= 0
//!
//! The dual is:
//!   minimize alpha * sum_j w[j] + sum_k z[k]
//!   sum_i lambda[i] = 1; lambda,w,z >= 0
//!   sum_i lambda[i] A_i[k] - sum_j w[j] B_j[k] <= z[k]
//!
//! Both LPs are solved independently by exact rational vertex enumeration.
//! Every result is accepted only when primal and dual optima agree exactly.
//!
//! Scope: small finite outcome spaces and small rational families. Arithmetic
//! overflow, enumeration-budget exhaustion, or failure to find a feasible
//! optimum returns an error. This is not a general continuous-law solver.
//!
//! Run:
//!   rustc --edition=2024 --test scripts/research/sdcb_finite_outcome_minimax.rs -o /tmp/sdcb-finite-outcome-tests
//!   /tmp/sdcb-finite-outcome-tests
//!   rustc --edition=2024 scripts/research/sdcb_finite_outcome_minimax.rs -o /tmp/sdcb-finite-outcome
//!   /tmp/sdcb-finite-outcome

use std::cmp::Ordering;
use std::fmt;

type Exact<T> = Result<T, ArithmeticOrSolverError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArithmeticOrSolverError {
    Overflow,
    InvalidInput,
    EnumerationLimit,
    NoFeasibleVertex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Rat {
    n: i128,
    d: i128,
}

impl Rat {
    fn new(n: i128, d: i128) -> Exact<Self> {
        if d == 0 {
            return Err(ArithmeticOrSolverError::InvalidInput);
        }
        let (n, d) = if d < 0 {
            (
                n.checked_neg().ok_or(ArithmeticOrSolverError::Overflow)?,
                d.checked_neg().ok_or(ArithmeticOrSolverError::Overflow)?,
            )
        } else {
            (n, d)
        };
        let abs_n = n.checked_abs().ok_or(ArithmeticOrSolverError::Overflow)?;
        let g = gcd(abs_n, d);
        Ok(Self { n: n / g, d: d / g })
    }

    fn zero() -> Self { Self { n: 0, d: 1 } }
    fn one() -> Self { Self { n: 1, d: 1 } }

    fn add(self, rhs: Self) -> Exact<Self> {
        let left = self.n.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let right = rhs.n.checked_mul(self.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let n = left.checked_add(right).ok_or(ArithmeticOrSolverError::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        Self::new(n, d)
    }

    fn sub(self, rhs: Self) -> Exact<Self> {
        let left = self.n.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let right = rhs.n.checked_mul(self.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let n = left.checked_sub(right).ok_or(ArithmeticOrSolverError::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        Self::new(n, d)
    }

    fn mul(self, rhs: Self) -> Exact<Self> {
        let n = self.n.checked_mul(rhs.n).ok_or(ArithmeticOrSolverError::Overflow)?;
        let d = self.d.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        Self::new(n, d)
    }

    fn div(self, rhs: Self) -> Exact<Self> {
        if rhs.n == 0 {
            return Err(ArithmeticOrSolverError::InvalidInput);
        }
        let n = self.n.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let d = self.d.checked_mul(rhs.n).ok_or(ArithmeticOrSolverError::Overflow)?;
        Self::new(n, d)
    }

    fn neg(self) -> Exact<Self> {
        Ok(Self {
            n: self.n.checked_neg().ok_or(ArithmeticOrSolverError::Overflow)?,
            d: self.d,
        })
    }

    fn cmp(self, rhs: Self) -> Exact<Ordering> {
        let left = self.n.checked_mul(rhs.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        let right = rhs.n.checked_mul(self.d).ok_or(ArithmeticOrSolverError::Overflow)?;
        Ok(left.cmp(&right))
    }

    fn is_zero(self) -> bool { self.n == 0 }
    fn is_nonnegative(self) -> bool { self.n >= 0 }

    fn max(self, rhs: Self) -> Exact<Self> {
        if self.cmp(rhs)? == Ordering::Less { Ok(rhs) } else { Ok(self) }
    }
}

impl fmt::Display for Rat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { write!(f, "{}", self.n) }
        else { write!(f, "{}/{}", self.n, self.d) }
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).expect("fixture rational must be representable")
}

#[derive(Clone, Debug)]
struct Law {
    id: &'static str,
    probabilities: Vec<Rat>,
}

impl Law {
    fn valid(&self, outcomes: usize) -> Exact<bool> {
        if self.probabilities.len() != outcomes {
            return Ok(false);
        }
        let mut sum = Rat::zero();
        for p in &self.probabilities {
            if !p.is_nonnegative() {
                return Ok(false);
            }
            sum = sum.add(*p)?;
        }
        Ok(sum == Rat::one())
    }
}

#[derive(Clone, Debug)]
struct LinearRow {
    a: Vec<Rat>,
    b: Rat,
}

fn dot(a: &[Rat], x: &[Rat]) -> Exact<Rat> {
    if a.len() != x.len() {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }
    let mut total = Rat::zero();
    for (left, right) in a.iter().zip(x.iter()) {
        total = total.add(left.mul(*right)?)?;
    }
    Ok(total)
}

fn satisfies_equalities(x: &[Rat], equalities: &[LinearRow]) -> Exact<bool> {
    for row in equalities {
        if dot(&row.a, x)? != row.b {
            return Ok(false);
        }
    }
    Ok(true)
}

fn satisfies_inequalities(x: &[Rat], inequalities: &[LinearRow]) -> Exact<bool> {
    for row in inequalities {
        if dot(&row.a, x)?.cmp(row.b)? == Ordering::Greater {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Solve a square exact-rational system. Singular bases return Ok(None).
fn solve_square(rows: &[LinearRow], dimension: usize) -> Exact<Option<Vec<Rat>>> {
    if rows.len() != dimension || rows.iter().any(|row| row.a.len() != dimension) {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }
    let mut matrix: Vec<Vec<Rat>> = rows
        .iter()
        .map(|row| {
            let mut augmented = row.a.clone();
            augmented.push(row.b);
            augmented
        })
        .collect();

    for column in 0..dimension {
        let pivot = (column..dimension).find(|&row| !matrix[row][column].is_zero());
        let Some(pivot_row) = pivot else {
            return Ok(None);
        };
        matrix.swap(column, pivot_row);

        let pivot_value = matrix[column][column];
        for entry in column..=dimension {
            matrix[column][entry] = matrix[column][entry].div(pivot_value)?;
        }

        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            if factor.is_zero() {
                continue;
            }
            for entry in column..=dimension {
                let product = factor.mul(matrix[column][entry])?;
                matrix[row][entry] = matrix[row][entry].sub(product)?;
            }
        }
    }

    Ok(Some((0..dimension).map(|row| matrix[row][dimension]).collect()))
}

struct BestVertex {
    value: Rat,
    point: Vec<Rat>,
}

fn inspect_basis(
    dimension: usize,
    equalities: &[LinearRow],
    inequalities: &[LinearRow],
    objective: &[Rat],
    selected: &[usize],
    best: &mut Option<BestVertex>,
) -> Exact<()> {
    let mut basis = equalities.to_vec();
    for &index in selected {
        basis.push(inequalities[index].clone());
    }
    let Some(point) = solve_square(&basis, dimension)? else {
        return Ok(());
    };
    if !satisfies_equalities(&point, equalities)? ||
        !satisfies_inequalities(&point, inequalities)? {
        return Ok(());
    }
    let value = dot(objective, &point)?;
    let replace = match best {
        None => true,
        Some(current) => value.cmp(current.value)? == Ordering::Greater,
    };
    if replace {
        *best = Some(BestVertex { value, point });
    }
    Ok(())
}

fn enumerate_combinations(
    dimension: usize,
    equalities: &[LinearRow],
    inequalities: &[LinearRow],
    objective: &[Rat],
    choose: usize,
    next: usize,
    selected: &mut Vec<usize>,
    visited: &mut usize,
    limit: usize,
    best: &mut Option<BestVertex>,
) -> Exact<()> {
    if selected.len() == choose {
        *visited = visited.checked_add(1).ok_or(ArithmeticOrSolverError::Overflow)?;
        if *visited > limit {
            return Err(ArithmeticOrSolverError::EnumerationLimit);
        }
        return inspect_basis(dimension, equalities, inequalities, objective, selected, best);
    }

    let remaining = choose - selected.len();
    if inequalities.len().saturating_sub(next) < remaining {
        return Ok(());
    }
    let last = inequalities.len() - remaining;
    for index in next..=last {
        selected.push(index);
        enumerate_combinations(
            dimension, equalities, inequalities, objective, choose, index + 1,
            selected, visited, limit, best,
        )?;
        selected.pop();
    }
    Ok(())
}

/// Maximize c.x over E x = f and A x <= b by exact vertex enumeration.
/// The caller must provide a bounded/attained optimum with at least one vertex.
fn maximize_by_vertex_enumeration(
    dimension: usize,
    equalities: &[LinearRow],
    inequalities: &[LinearRow],
    objective: &[Rat],
    basis_limit: usize,
) -> Exact<BestVertex> {
    if dimension == 0 || equalities.len() > dimension ||
        objective.len() != dimension ||
        equalities.iter().chain(inequalities.iter()).any(|row| row.a.len() != dimension) {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }
    let choose = dimension - equalities.len();
    if inequalities.len() < choose {
        return Err(ArithmeticOrSolverError::NoFeasibleVertex);
    }
    let mut best = None;
    let mut selected = Vec::with_capacity(choose);
    let mut visited = 0usize;
    enumerate_combinations(
        dimension, equalities, inequalities, objective, choose, 0,
        &mut selected, &mut visited, basis_limit, &mut best,
    )?;
    best.ok_or(ArithmeticOrSolverError::NoFeasibleVertex)
}

#[derive(Clone, Debug)]
struct OracleResult {
    primal_power: Rat,
    primal_test: Vec<Rat>,
    dual_upper_bound: Rat,
    dual_lambda: Vec<Rat>,
    dual_null_weights: Vec<Rat>,
    dual_slack: Vec<Rat>,
    primal_dual_gap: Rat,
    primal_bases_checked: usize,
    dual_bases_checked: usize,
}

fn validate_family(
    alternatives: &[Law],
    nulls: &[Law],
    alpha: Rat,
) -> Exact<usize> {
    if alternatives.is_empty() || nulls.is_empty() ||
        !alpha.is_nonnegative() ||
        alpha.cmp(Rat::one())? != Ordering::Less {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }
    let outcomes = alternatives[0].probabilities.len();
    if outcomes == 0 {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }
    for law in alternatives.iter().chain(nulls.iter()) {
        if !law.valid(outcomes)? {
            return Err(ArithmeticOrSolverError::InvalidInput);
        }
    }
    Ok(outcomes)
}

fn solve_minimax(
    alternatives: &[Law],
    nulls: &[Law],
    alpha: Rat,
    basis_limit: usize,
) -> Exact<OracleResult> {
    let outcomes = validate_family(alternatives, nulls, alpha)?;

    // Primal variables are [phi_0, ..., phi_(K-1), beta].
    let primal_dimension = outcomes + 1;
    let mut primal_inequalities = Vec::new();

    // beta - E_Ai[phi] <= 0
    for alternative in alternatives {
        let mut row: Vec<Rat> = alternative.probabilities.iter()
            .map(|p| p.neg())
            .collect::<Exact<Vec<_>>>()?;
        row.push(Rat::one());
        primal_inequalities.push(LinearRow { a: row, b: Rat::zero() });
    }
    // E_Bj[phi] <= alpha
    for null in nulls {
        let mut row = null.probabilities.clone();
        row.push(Rat::zero());
        primal_inequalities.push(LinearRow { a: row, b: alpha });
    }
    // -phi_k <= 0 and phi_k <= 1
    for k in 0..outcomes {
        let mut lower = vec![Rat::zero(); primal_dimension];
        lower[k] = r(-1, 1);
        primal_inequalities.push(LinearRow { a: lower, b: Rat::zero() });

        let mut upper = vec![Rat::zero(); primal_dimension];
        upper[k] = Rat::one();
        primal_inequalities.push(LinearRow { a: upper, b: Rat::one() });
    }
    // beta >= 0
    let mut beta_lower = vec![Rat::zero(); primal_dimension];
    beta_lower[outcomes] = r(-1, 1);
    primal_inequalities.push(LinearRow { a: beta_lower, b: Rat::zero() });

    let mut primal_objective = vec![Rat::zero(); primal_dimension];
    primal_objective[outcomes] = Rat::one();
    let primal = maximize_by_vertex_enumeration(
        primal_dimension, &[], &primal_inequalities, &primal_objective, basis_limit,
    )?;

    // Dual variables are [lambda_0..lambda_(I-1), w_0..w_(J-1), z_0..z_(K-1)].
    let alternatives_count = alternatives.len();
    let null_count = nulls.len();
    let dual_dimension = alternatives_count + null_count + outcomes;
    let mut dual_equalities = Vec::new();
    let mut lambda_sum = vec![Rat::zero(); dual_dimension];
    for item in lambda_sum.iter_mut().take(alternatives_count) {
        *item = Rat::one();
    }
    dual_equalities.push(LinearRow { a: lambda_sum, b: Rat::one() });

    let mut dual_inequalities = Vec::new();
    // Sum_i lambda_i A_i[k] - Sum_j w_j B_j[k] - z_k <= 0.
    for k in 0..outcomes {
        let mut row = vec![Rat::zero(); dual_dimension];
        for i in 0..alternatives_count {
            row[i] = alternatives[i].probabilities[k];
        }
        for j in 0..null_count {
            row[alternatives_count + j] = nulls[j].probabilities[k].neg()?;
        }
        row[alternatives_count + null_count + k] = r(-1, 1);
        dual_inequalities.push(LinearRow { a: row, b: Rat::zero() });
    }
    // lambda, w, z >= 0
    for variable in 0..dual_dimension {
        let mut row = vec![Rat::zero(); dual_dimension];
        row[variable] = r(-1, 1);
        dual_inequalities.push(LinearRow { a: row, b: Rat::zero() });
    }

    // Minimize alpha * sum(w) + sum(z), represented as maximizing its negative.
    let mut dual_negative_objective = vec![Rat::zero(); dual_dimension];
    for j in 0..null_count {
        dual_negative_objective[alternatives_count + j] = alpha.neg()?;
    }
    for k in 0..outcomes {
        dual_negative_objective[alternatives_count + null_count + k] = r(-1, 1);
    }
    let dual_negative = maximize_by_vertex_enumeration(
        dual_dimension, &dual_equalities, &dual_inequalities,
        &dual_negative_objective, basis_limit,
    )?;
    let dual_upper_bound = dual_negative.value.neg()?;

    // The two independent optima must agree exactly. Any mismatch is unresolved.
    let primal_dual_gap = dual_upper_bound.sub(primal.value)?;
    if !primal_dual_gap.is_zero() {
        return Err(ArithmeticOrSolverError::InvalidInput);
    }

    let lambda = dual_negative.point[..alternatives_count].to_vec();
    let null_weights = dual_negative.point[alternatives_count..alternatives_count + null_count].to_vec();
    let slack = dual_negative.point[alternatives_count + null_count..].to_vec();

    Ok(OracleResult {
        primal_power: primal.value,
        primal_test: primal.point[..outcomes].to_vec(),
        dual_upper_bound,
        dual_lambda: lambda,
        dual_null_weights: null_weights,
        dual_slack: slack,
        primal_dual_gap,
        primal_bases_checked: 0, // Filled by caller-facing diagnostics in a later revision.
        dual_bases_checked: 0,
    })
}

fn law(id: &'static str, probabilities: &[(i128, i128)]) -> Law {
    Law {
        id,
        probabilities: probabilities.iter().map(|(n, d)| r(*n, *d)).collect(),
    }
}

fn convex_hull_overlap_fixture() -> (Vec<Law>, Vec<Law>, Rat) {
    let alternatives = vec![
        law("A1", &[(1, 1), (0, 1)]),
        law("A2", &[(0, 1), (1, 1)]),
    ];
    let nulls = vec![law("B1", &[(1, 3), (2, 3)])];
    (alternatives, nulls, r(1, 10))
}

fn partial_separation_fixture() -> (Vec<Law>, Vec<Law>, Rat) {
    let alternatives = vec![
        law("A1", &[(1, 1), (0, 1), (0, 1)]),
        law("A2", &[(0, 1), (1, 1), (0, 1)]),
    ];
    let nulls = vec![law("B1", &[(1, 10), (1, 10), (8, 10)])];
    (alternatives, nulls, r(1, 10))
}

fn complete_separation_fixture() -> (Vec<Law>, Vec<Law>, Rat) {
    let alternatives = vec![
        law("A1", &[(1, 1), (0, 1), (0, 1)]),
        law("A2", &[(0, 1), (1, 1), (0, 1)]),
    ];
    let nulls = vec![law("B1", &[(0, 1), (0, 1), (1, 1)])];
    (alternatives, nulls, r(1, 10))
}

fn main() {
    let limit = 2_000_000;
    for (name, (alternatives, nulls, alpha)) in [
        ("convex_hull_overlap", convex_hull_overlap_fixture()),
        ("partial_separation", partial_separation_fixture()),
        ("complete_separation", complete_separation_fixture()),
    ] {
        match solve_minimax(&alternatives, &nulls, alpha, limit) {
            Ok(result) => {
                println!("fixture={name}");
                println!("alpha={alpha}");
                println!("primal_worst_case_power={}", result.primal_power);
                println!("dual_upper_bound={}", result.dual_upper_bound);
                println!("primal_dual_gap={}", result.primal_dual_gap);
                println!("primal_test={:?}", result.primal_test);
                println!("dual_lambda={:?}", result.dual_lambda);
                println!("dual_null_weights={:?}", result.dual_null_weights);
                println!("dual_slack={:?}", result.dual_slack);
            }
            Err(error) => {
                println!("fixture={name}");
                println!("status=UNRESOLVED:{error:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIS_LIMIT: usize = 2_000_000;

    fn assert_optimum(
        alternatives: &[Law],
        nulls: &[Law],
        alpha: Rat,
        expected: Rat,
    ) -> OracleResult {
        let result = solve_minimax(alternatives, nulls, alpha, BASIS_LIMIT)
            .expect("exact primal/dual solver should find matching vertices");
        assert_eq!(result.primal_power, expected);
        assert_eq!(result.dual_upper_bound, expected);
        assert_eq!(result.primal_dual_gap, Rat::zero());
        assert!(result.primal_test.iter().all(|value| {
            value.is_nonnegative() && value.cmp(Rat::one()).unwrap() != Ordering::Greater
        }));
        result
    }

    #[test]
    fn convex_hull_overlap_caps_worst_case_power_at_alpha() {
        let (a, b, alpha) = convex_hull_overlap_fixture();
        let result = assert_optimum(&a, &b, alpha, alpha);
        assert_eq!(result.primal_test, vec![alpha, alpha]);
    }

    #[test]
    fn partial_separation_has_nontrivial_exact_maximin_power() {
        let (a, b, alpha) = partial_separation_fixture();
        let result = assert_optimum(&a, &b, alpha, r(1, 2));
        assert_eq!(result.primal_test, vec![r(1, 2), r(1, 2), Rat::zero()]);
    }

    #[test]
    fn complete_separation_reaches_unit_power() {
        let (a, b, alpha) = complete_separation_fixture();
        assert_eq!(assert_optimum(&a, &b, alpha, Rat::one()).primal_power, Rat::one());
    }

    #[test]
    fn adding_a_null_member_cannot_increase_maximin_power() {
        let (a, b, alpha) = partial_separation_fixture();
        let before = assert_optimum(&a, &b, alpha, r(1, 2)).primal_power;
        let mut expanded = b.clone();
        expanded.push(law("B2", &[(1, 2), (1, 2), (0, 1)]));
        let after = solve_minimax(&a, &expanded, alpha, BASIS_LIMIT).unwrap();
        assert!(after.primal_power.cmp(before).unwrap() != Ordering::Greater);
        assert_eq!(after.primal_power, alpha);
        assert_eq!(after.primal_dual_gap, Rat::zero());
    }

    #[test]
    fn adding_an_alternative_member_cannot_increase_worst_case_power() {
        let (a, b, alpha) = partial_separation_fixture();
        let before = assert_optimum(&a, &b, alpha, r(1, 2)).primal_power;
        let mut expanded = a.clone();
        expanded.push(law("A3", &[(0, 1), (0, 1), (1, 1)]));
        let after = solve_minimax(&expanded, &b, alpha, BASIS_LIMIT).unwrap();
        assert!(after.primal_power.cmp(before).unwrap() != Ordering::Greater);
        assert_eq!(after.primal_power, alpha);
        assert_eq!(after.primal_dual_gap, Rat::zero());
    }

    #[test]
    fn permutation_of_family_members_preserves_value() {
        let (mut a, mut b, alpha) = partial_separation_fixture();
        let original = assert_optimum(&a, &b, alpha, r(1, 2)).primal_power;
        a.reverse();
        b.reverse();
        let permuted = solve_minimax(&a, &b, alpha, BASIS_LIMIT).unwrap();
        assert_eq!(permuted.primal_power, original);
        assert_eq!(permuted.dual_upper_bound, original);
    }

    #[test]
    fn invalid_probability_mass_fails_closed() {
        let a = vec![law("A-invalid", &[(1, 2), (1, 3)])];
        let b = vec![law("B", &[(1, 1), (0, 1)])];
        assert_eq!(
            solve_minimax(&a, &b, r(1, 10), BASIS_LIMIT),
            Err(ArithmeticOrSolverError::InvalidInput)
        );
    }

    #[test]
    fn enumeration_budget_exhaustion_fails_closed() {
        let (a, b, alpha) = complete_separation_fixture();
        assert_eq!(
            solve_minimax(&a, &b, alpha, 0),
            Err(ArithmeticOrSolverError::EnumerationLimit)
        );
    }
}
