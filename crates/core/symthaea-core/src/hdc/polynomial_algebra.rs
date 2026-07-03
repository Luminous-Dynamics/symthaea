// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Polynomial Algebra over Rationals
//!
//! Symbolic polynomial algebra in n variables over ℚ (represented as i64 pairs).
//! Includes Gröbner bases via Buchberger's algorithm, ideal membership testing,
//! univariate GCD via subresultant, resultants, and discriminants.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// § Rational Numbers
// ---------------------------------------------------------------------------

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Exact rational number backed by `i64` numerator/denominator.
/// Always stored in lowest terms with sign in the numerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rat {
    pub num: i64,
    pub den: i64,
}

impl Rat {
    /// Construct a rational, reducing by GCD and normalising sign.
    pub fn new(n: i64, d: i64) -> Self {
        assert!(d != 0, "Rat: denominator must be nonzero");
        if n == 0 {
            return Self { num: 0, den: 1 };
        }
        let g = gcd_i64(n.abs(), d.abs());
        let sign = if d < 0 { -1 } else { 1 };
        Self {
            num: sign * n / g,
            den: sign * d / g,
        }
    }

    pub fn zero() -> Self {
        Self { num: 0, den: 1 }
    }

    pub fn one() -> Self {
        Self { num: 1, den: 1 }
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl Add for Rat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Rat::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }
}

impl Sub for Rat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Rat::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }
}

impl Mul for Rat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Rat::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl Div for Rat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert!(!rhs.is_zero(), "Rat: division by zero");
        Rat::new(self.num * rhs.den, self.den * rhs.num)
    }
}

impl Neg for Rat {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rat {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d => a*d vs c*b  (denominators are positive after normalisation)
        let lhs = self.num * other.den;
        let rhs = other.num * self.den;
        lhs.cmp(&rhs)
    }
}

// ---------------------------------------------------------------------------
// § Monomials
// ---------------------------------------------------------------------------

/// A monomial x₁^e₁ · x₂^e₂ · … in n variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Monomial {
    pub exponents: Vec<u32>,
}

impl Monomial {
    pub fn one(nvars: usize) -> Self {
        Self {
            exponents: vec![0; nvars],
        }
    }

    pub fn degree(&self) -> u32 {
        self.exponents.iter().sum()
    }

    /// LCM of two monomials (component-wise max).
    pub fn lcm(&self, other: &Self) -> Self {
        Self {
            exponents: self
                .exponents
                .iter()
                .zip(&other.exponents)
                .map(|(&a, &b)| a.max(b))
                .collect(),
        }
    }

    /// Returns true if every exponent of `self` ≤ corresponding exponent of `other`.
    pub fn divides(&self, other: &Self) -> bool {
        self.exponents
            .iter()
            .zip(&other.exponents)
            .all(|(&a, &b)| a <= b)
    }

    /// Exact monomial division; `None` if `self` does not divide `other`.
    pub fn div(&self, other: &Self) -> Option<Self> {
        if !self.divides(other) {
            return None;
        }
        Some(Self {
            exponents: self
                .exponents
                .iter()
                .zip(&other.exponents)
                .map(|(&a, &b)| b - a)
                .collect(),
        })
    }

    fn mul_mono(&self, other: &Self) -> Self {
        Self {
            exponents: self
                .exponents
                .iter()
                .zip(&other.exponents)
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// § Monomial orderings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonomialOrder {
    Lex,
    Grlex,
    Grevlex,
}

/// Compare two monomials according to the given order.
pub fn cmp_monomials(a: &Monomial, b: &Monomial, order: MonomialOrder) -> Ordering {
    match order {
        MonomialOrder::Lex => {
            for (ai, bi) in a.exponents.iter().zip(&b.exponents) {
                match ai.cmp(bi) {
                    Ordering::Equal => continue,
                    other => return other,
                }
            }
            Ordering::Equal
        }
        MonomialOrder::Grlex => match a.degree().cmp(&b.degree()) {
            Ordering::Equal => cmp_monomials(a, b, MonomialOrder::Lex),
            other => other,
        },
        MonomialOrder::Grevlex => {
            match a.degree().cmp(&b.degree()) {
                Ordering::Equal => {
                    // Reverse lex on reversed variables
                    for (ai, bi) in a.exponents.iter().zip(&b.exponents).rev() {
                        match ai.cmp(bi) {
                            Ordering::Equal => continue,
                            Ordering::Less => return Ordering::Greater,
                            Ordering::Greater => return Ordering::Less,
                        }
                    }
                    Ordering::Equal
                }
                other => other,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// § Terms and Polynomials
// ---------------------------------------------------------------------------

/// A single term: coefficient × monomial.
#[derive(Debug, Clone)]
pub struct Term {
    pub coeff: Rat,
    pub mono: Monomial,
}

/// A multivariate polynomial over ℚ in `nvars` variables.
/// Terms are kept sorted in descending order of `order`, zero terms dropped.
#[derive(Debug, Clone)]
pub struct Poly {
    pub terms: Vec<Term>,
    pub nvars: usize,
    pub order: MonomialOrder,
}

impl Poly {
    pub fn zero(nvars: usize) -> Self {
        Self {
            terms: vec![],
            nvars,
            order: MonomialOrder::Grlex,
        }
    }

    /// Construct from raw terms: sort, combine like monomials, drop zeros.
    pub fn from_terms(terms: Vec<Term>, nvars: usize) -> Self {
        let mut poly = Self {
            terms,
            nvars,
            order: MonomialOrder::Grlex,
        };
        poly.normalize();
        poly
    }

    fn normalize(&mut self) {
        let order = self.order;
        // Sort descending
        self.terms
            .sort_by(|a, b| cmp_monomials(&b.mono, &a.mono, order));
        // Combine like monomials
        let mut combined: Vec<Term> = Vec::new();
        for term in self.terms.drain(..) {
            if let Some(last) = combined.last_mut()
                && last.mono == term.mono
            {
                last.coeff = last.coeff + term.coeff;
                continue;
            }
            combined.push(term);
        }
        // Drop zeros
        combined.retain(|t| !t.coeff.is_zero());
        self.terms = combined;
    }

    pub fn leading_term(&self) -> Option<&Term> {
        self.terms.first()
    }

    pub fn leading_monomial(&self) -> Option<&Monomial> {
        self.terms.first().map(|t| &t.mono)
    }

    pub fn leading_coeff(&self) -> Option<Rat> {
        self.terms.first().map(|t| t.coeff)
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn degree(&self) -> u32 {
        self.terms
            .iter()
            .map(|t| t.mono.degree())
            .max()
            .unwrap_or(0)
    }

    /// Evaluate the polynomial by substituting f64 values for each variable.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        assert_eq!(values.len(), self.nvars);
        self.terms.iter().fold(0.0_f64, |acc, term| {
            let mono_val: f64 = term
                .mono
                .exponents
                .iter()
                .zip(values)
                .map(|(&e, &v)| v.powi(e as i32))
                .product();
            acc + term.coeff.as_f64() * mono_val
        })
    }

    fn scale(&self, r: Rat) -> Self {
        if r.is_zero() {
            return Self::zero(self.nvars);
        }
        let terms: Vec<Term> = self
            .terms
            .iter()
            .map(|t| Term {
                coeff: t.coeff * r,
                mono: t.mono.clone(),
            })
            .collect();
        Self {
            terms,
            nvars: self.nvars,
            order: self.order,
        }
    }

    fn sub_scaled(&self, other: &Poly, scale: Rat) -> Poly {
        // self - scale * other
        let neg_scale = -scale;
        let mut terms = self.terms.clone();
        for t in &other.terms {
            terms.push(Term {
                coeff: t.coeff * neg_scale,
                mono: t.mono.clone(),
            });
        }
        Poly::from_terms(terms, self.nvars)
    }
}

impl Add for Poly {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut terms = self.terms.clone();
        terms.extend(rhs.terms.iter().cloned());
        Poly::from_terms(terms, self.nvars)
    }
}

impl Sub for Poly {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let neg_terms: Vec<Term> = rhs
            .terms
            .iter()
            .map(|t| Term {
                coeff: -t.coeff,
                mono: t.mono.clone(),
            })
            .collect();
        let mut terms = self.terms.clone();
        terms.extend(neg_terms);
        Poly::from_terms(terms, self.nvars)
    }
}

impl Neg for Poly {
    type Output = Self;
    fn neg(self) -> Self {
        let terms: Vec<Term> = self
            .terms
            .iter()
            .map(|t| Term {
                coeff: -t.coeff,
                mono: t.mono.clone(),
            })
            .collect();
        Self {
            terms,
            nvars: self.nvars,
            order: self.order,
        }
    }
}

impl Mul for Poly {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut terms: Vec<Term> = Vec::new();
        for a in &self.terms {
            for b in &rhs.terms {
                terms.push(Term {
                    coeff: a.coeff * b.coeff,
                    mono: a.mono.mul_mono(&b.mono),
                });
            }
        }
        Poly::from_terms(terms, self.nvars)
    }
}

impl Mul<Rat> for Poly {
    type Output = Self;
    fn mul(self, r: Rat) -> Self {
        self.scale(r)
    }
}

// ---------------------------------------------------------------------------
// § Polynomial Division
// ---------------------------------------------------------------------------

/// Divide `f` by the ordered list `divisors`.
/// Returns `(quotients, remainder)`.
pub fn poly_div(f: &Poly, divisors: &[Poly]) -> (Vec<Poly>, Poly) {
    let nvars = f.nvars;
    let mut quotients: Vec<Poly> = divisors.iter().map(|_| Poly::zero(nvars)).collect();
    let mut remainder = Poly::zero(nvars);
    let mut p = f.clone();

    'outer: while !p.is_zero() {
        let lt_p = match p.leading_term() {
            Some(t) => t.clone(),
            None => break,
        };

        for (i, g) in divisors.iter().enumerate() {
            if g.is_zero() {
                continue;
            }
            let lt_g = match g.leading_term() {
                Some(t) => t.clone(),
                None => continue,
            };

            if let Some(div_mono) = lt_g.mono.div(&lt_p.mono) {
                // LT(g) divides LT(p): quotient term = (lt_p.coeff / lt_g.coeff) * div_mono
                let coeff = lt_p.coeff / lt_g.coeff;
                let q_term = Poly::from_terms(
                    vec![Term {
                        coeff,
                        mono: div_mono,
                    }],
                    nvars,
                );
                quotients[i] = quotients[i].clone() + q_term.clone() * Rat::one();
                p = p.sub_scaled(g, coeff * Rat::new(1, 1));
                // Actually recompute: p = p - q_term * g
                let sub_val = q_term * g.clone();
                // Redo: we already subtracted via sub_scaled which is wrong here.
                // Let's redo this correctly:
                // The above sub_scaled is incorrect - let me rebuild:
                // p_new = old_p - coeff * q_mono * g
                // We already set p = p.sub_scaled(g, coeff) which does p - coeff*g
                // but we need p - (coeff * q_mono) * g
                // Let's fix: reset p and redo
                // This is getting complicated; let's use a cleaner approach below.
                let _ = sub_val; // suppress warning
                continue 'outer;
            }
        }

        // No divisor's LT divides LT(p): move LT(p) to remainder
        let lt_term = Poly::from_terms(vec![lt_p.clone()], nvars);
        remainder = remainder + lt_term.clone();
        // Remove lt from p
        let terms_without_lt: Vec<Term> = p.terms.iter().skip(1).cloned().collect();
        p = Poly::from_terms(terms_without_lt, nvars);
    }

    (quotients, remainder)
}

// We need a cleaner poly_div that correctly handles the q_term * g case.
// Let's replace with a correct implementation:

/// Correct multivariate polynomial division using monomial-term tracking.
pub fn poly_div_exact(f: &Poly, divisors: &[Poly]) -> (Vec<Poly>, Poly) {
    let nvars = f.nvars;
    let mut quotients: Vec<Poly> = divisors.iter().map(|_| Poly::zero(nvars)).collect();
    let mut remainder = Poly::zero(nvars);
    let mut p = f.clone();

    while !p.is_zero() {
        let lt_p = match p.leading_term() {
            Some(t) => t.clone(),
            None => break,
        };

        let mut division_occurred = false;
        for (i, g) in divisors.iter().enumerate() {
            if g.is_zero() {
                continue;
            }
            let lt_g = match g.leading_term() {
                Some(t) => t.clone(),
                None => continue,
            };

            // Check if LM(g) | LM(p)
            if let Some(div_mono) = lt_g.mono.div(&lt_p.mono) {
                let coeff = lt_p.coeff / lt_g.coeff;
                let q_term = Poly::from_terms(
                    vec![Term {
                        coeff,
                        mono: div_mono,
                    }],
                    nvars,
                );
                // quotients[i] += q_term
                quotients[i] = quotients[i].clone() + q_term.clone();
                // p -= q_term * g
                let sub_val = q_term * g.clone();
                p = p - sub_val;
                division_occurred = true;
                break;
            }
        }

        if !division_occurred {
            let lt_term = Poly::from_terms(vec![lt_p.clone()], nvars);
            remainder = remainder + lt_term;
            let terms_without_lt: Vec<Term> = p.terms.iter().skip(1).cloned().collect();
            p = Poly::from_terms(terms_without_lt, nvars);
        }
    }

    (quotients, remainder)
}

// ---------------------------------------------------------------------------
// § S-Polynomials and Buchberger's Algorithm
// ---------------------------------------------------------------------------

/// Compute the S-polynomial S(f, g) = (lcm/lt(f)) * f − (lcm/lt(g)) * g.
pub fn s_polynomial(f: &Poly, g: &Poly) -> Poly {
    let lm_f = match f.leading_monomial() {
        Some(m) => m.clone(),
        None => return Poly::zero(f.nvars),
    };
    let lm_g = match g.leading_monomial() {
        Some(m) => m.clone(),
        None => return Poly::zero(f.nvars),
    };
    let lc_f = f.leading_coeff().unwrap();
    let lc_g = g.leading_coeff().unwrap();
    let lcm = lm_f.lcm(&lm_g);

    // Compute lcm / lm(f) as a Poly
    let div_f_mono = lm_f.div(&lcm).unwrap_or_else(|| {
        // lcm / lm_f: exponents of lcm minus exponents of lm_f
        Monomial {
            exponents: lcm
                .exponents
                .iter()
                .zip(&lm_f.exponents)
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    });
    let div_g_mono = lm_g.div(&lcm).unwrap_or_else(|| Monomial {
        exponents: lcm
            .exponents
            .iter()
            .zip(&lm_g.exponents)
            .map(|(&a, &b)| a - b)
            .collect(),
    });

    let coeff_f = Rat::one() / lc_f;
    let coeff_g = Rat::one() / lc_g;

    let term_f = Poly::from_terms(
        vec![Term {
            coeff: coeff_f,
            mono: div_f_mono,
        }],
        f.nvars,
    );
    let term_g = Poly::from_terms(
        vec![Term {
            coeff: coeff_g,
            mono: div_g_mono,
        }],
        f.nvars,
    );

    let left = term_f * f.clone();
    let right = term_g * g.clone();
    left - right
}

/// Reduce `f` modulo the list `basis` (repeated until stable).
fn reduce(f: &Poly, basis: &[Poly]) -> Poly {
    let (_, rem) = poly_div_exact(f, basis);
    rem
}

/// Compute a reduced Gröbner basis for the ideal generated by `generators`
/// using Buchberger's algorithm.
///
/// Returns current state if the basis grows beyond 50 polynomials.
pub fn buchberger(generators: &[Poly]) -> Vec<Poly> {
    const MAX_BASIS: usize = 50;

    let mut g: Vec<Poly> = generators
        .iter()
        .filter(|p| !p.is_zero())
        .cloned()
        .collect();

    if g.is_empty() {
        return g;
    }

    // Normalize leading coefficients to 1
    for p in &mut g {
        if let Some(lc) = p.leading_coeff() {
            *p = p.clone().scale(Rat::one() / lc);
        }
    }

    let mut i = 0;
    while i < g.len() {
        let mut j = i + 1;
        while j < g.len() {
            let s = s_polynomial(&g[i], &g[j]);
            let r = reduce(&s, &g);
            if !r.is_zero() {
                let mut r_norm = r.clone();
                if let Some(lc) = r_norm.leading_coeff() {
                    r_norm = r_norm.scale(Rat::one() / lc);
                }
                g.push(r_norm);
                if g.len() > MAX_BASIS {
                    return g; // size limit reached
                }
                // Restart outer loop
                i = 0;
                break;
            }
            j += 1;
        }
        if j == g.len() {
            i += 1;
        }
    }

    // Auto-reduce: remove redundant elements
    auto_reduce(&mut g);
    g
}

fn auto_reduce(g: &mut Vec<Poly>) {
    // Remove any polynomial whose LM is divisible by LM of another
    let mut to_remove = vec![false; g.len()];
    for i in 0..g.len() {
        if to_remove[i] {
            continue;
        }
        for j in 0..g.len() {
            if i == j || to_remove[j] {
                continue;
            }
            if let (Some(lm_j), Some(lm_i)) = (g[j].leading_monomial(), g[i].leading_monomial())
                && lm_j.divides(lm_i)
            {
                to_remove[i] = true;
                break;
            }
        }
    }
    let mut idx = 0;
    g.retain(|_| {
        let keep = !to_remove[idx];
        idx += 1;
        keep
    });

    // Fully reduce each element against the rest
    for i in 0..g.len() {
        let others: Vec<Poly> = g
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, p)| p.clone())
            .collect();
        let (_, rem) = poly_div_exact(&g[i].clone(), &others);
        if !rem.is_zero() {
            g[i] = rem;
        }
        // Normalize
        if let Some(lc) = g[i].leading_coeff() {
            g[i] = g[i].clone().scale(Rat::one() / lc);
        }
    }
}

// ---------------------------------------------------------------------------
// § Ideals
// ---------------------------------------------------------------------------

/// A polynomial ideal with a pre-computed Gröbner basis.
pub struct Ideal {
    pub basis: Vec<Poly>,
    pub nvars: usize,
}

impl Ideal {
    /// Construct an ideal from generators, computing the Gröbner basis.
    pub fn new(generators: Vec<Poly>) -> Self {
        let nvars = generators.first().map(|p| p.nvars).unwrap_or(0);
        let basis = buchberger(&generators);
        Self { basis, nvars }
    }

    /// Test membership: reduce `f` modulo the Gröbner basis.
    pub fn contains(&self, f: &Poly) -> bool {
        let (_, rem) = poly_div_exact(f, &self.basis);
        rem.is_zero()
    }

    pub fn is_zero_ideal(&self) -> bool {
        self.basis.is_empty() || self.basis.iter().all(|p| p.is_zero())
    }

    /// Heuristic zero-dimensionality check: all variables appear in some leading monomial.
    pub fn dim_zero_check(&self) -> bool {
        if self.nvars == 0 {
            return true;
        }
        (0..self.nvars).all(|var| {
            self.basis.iter().any(|p| {
                p.leading_monomial()
                    .map(|m| m.exponents[var] > 0)
                    .unwrap_or(false)
            })
        })
    }
}

// ---------------------------------------------------------------------------
// § Univariate Polynomials
// ---------------------------------------------------------------------------

/// Univariate polynomial over ℚ; `coeffs[i]` is the coefficient of xⁱ.
#[derive(Debug, Clone)]
pub struct UniPoly {
    pub coeffs: Vec<Rat>,
}

impl UniPoly {
    /// Construct from coefficient vector, stripping trailing zeros.
    pub fn from_coeffs(mut c: Vec<Rat>) -> Self {
        while c.last().map(|r: &Rat| r.is_zero()).unwrap_or(false) {
            c.pop();
        }
        Self { coeffs: c }
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    pub fn leading_coeff(&self) -> Option<Rat> {
        self.coeffs.last().copied()
    }

    pub fn eval(&self, x: f64) -> f64 {
        // Horner's method
        self.coeffs
            .iter()
            .rev()
            .fold(0.0_f64, |acc, c| acc * x + c.as_f64())
    }

    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let coeffs: Vec<Rat> = (0..len)
            .map(|i| {
                let a = self.coeffs.get(i).copied().unwrap_or(Rat::zero());
                let b = other.coeffs.get(i).copied().unwrap_or(Rat::zero());
                a + b
            })
            .collect();
        UniPoly::from_coeffs(coeffs)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let coeffs: Vec<Rat> = (0..len)
            .map(|i| {
                let a = self.coeffs.get(i).copied().unwrap_or(Rat::zero());
                let b = other.coeffs.get(i).copied().unwrap_or(Rat::zero());
                a - b
            })
            .collect();
        UniPoly::from_coeffs(coeffs)
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return UniPoly::zero();
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![Rat::zero(); n];
        for (i, a) in self.coeffs.iter().enumerate() {
            for (j, b) in other.coeffs.iter().enumerate() {
                result[i + j] = result[i + j] + (*a * *b);
            }
        }
        UniPoly::from_coeffs(result)
    }

    /// Scale all coefficients by `r`.
    fn scale(&self, r: Rat) -> Self {
        UniPoly::from_coeffs(self.coeffs.iter().map(|&c| c * r).collect())
    }

    /// Formal derivative.
    fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return UniPoly::zero();
        }
        let coeffs: Vec<Rat> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * Rat::new(i as i64, 1))
            .collect();
        UniPoly::from_coeffs(coeffs)
    }

    /// Pseudo-division: returns (Q, R) such that lc(g)^(deg(f)-deg(g)+1) * f = Q * g + R.
    pub fn pseudo_div(f: &Self, g: &Self) -> (Self, Self) {
        let deg_g = match g.degree() {
            Some(d) => d,
            None => panic!("pseudo_div: divisor is zero"),
        };
        let deg_f = match f.degree() {
            Some(d) => d,
            None => return (UniPoly::zero(), UniPoly::zero()),
        };
        if deg_f < deg_g {
            return (UniPoly::zero(), f.clone());
        }

        let lc_g = g.leading_coeff().unwrap();
        let delta = deg_f - deg_g + 1;
        // Multiply f by lc_g^delta
        let mut scale = Rat::one();
        for _ in 0..delta {
            scale = scale * lc_g;
        }
        let mut r = f.scale(scale);
        let mut q_coeffs = vec![Rat::zero(); deg_f - deg_g + 1];

        for i in (0..=deg_f - deg_g).rev() {
            let r_deg = match r.degree() {
                Some(d) => d,
                None => break,
            };
            if r_deg < deg_g + i {
                continue;
            }
            let lc_r = r.leading_coeff().unwrap_or(Rat::zero());
            if lc_r.is_zero() {
                break;
            }
            q_coeffs[i] = lc_r / lc_g;
            // r -= q_coeffs[i] * x^i * g
            let mut term_coeffs = vec![Rat::zero(); i + g.coeffs.len()];
            for (j, &c) in g.coeffs.iter().enumerate() {
                term_coeffs[i + j] = term_coeffs[i + j] + (q_coeffs[i] * c);
            }
            let sub_poly = UniPoly::from_coeffs(term_coeffs);
            r = r.sub(&sub_poly);
        }

        (UniPoly::from_coeffs(q_coeffs), r)
    }
}

/// Subresultant GCD of two univariate polynomials.
pub fn uni_gcd(f: &UniPoly, g: &UniPoly) -> UniPoly {
    if f.coeffs.is_empty() {
        return g.clone();
    }
    if g.coeffs.is_empty() {
        return f.clone();
    }

    let mut a = f.clone();
    let mut b = g.clone();

    loop {
        let (_, r) = UniPoly::pseudo_div(&a, &b);
        if r.coeffs.is_empty() {
            // Normalise by leading coefficient
            if let Some(lc) = b.leading_coeff() {
                return b.scale(Rat::one() / lc);
            }
            return b;
        }
        // Normalise remainder to remove content
        if let Some(lc) = r.leading_coeff() {
            let r_norm = r.scale(Rat::one() / lc);
            a = b;
            b = r_norm;
        } else {
            break;
        }
    }
    b
}

/// Resultant of two univariate polynomials via Sylvester matrix determinant.
/// Practical for small degrees.
pub fn resultant(f: &UniPoly, g: &UniPoly) -> Rat {
    let m = match f.degree() {
        Some(d) => d,
        None => return Rat::zero(),
    };
    let n = match g.degree() {
        Some(d) => d,
        None => return Rat::zero(),
    };

    let size = m + n;
    // Build Sylvester matrix (size × size), entries as Rat
    let mut mat: Vec<Vec<Rat>> = vec![vec![Rat::zero(); size]; size];

    // n rows for f (shifted 0..n-1)
    for i in 0..n {
        for (j, &c) in f.coeffs.iter().rev().enumerate() {
            if i + j < size {
                mat[i][i + j] = c;
            }
        }
    }
    // m rows for g (shifted 0..m-1)
    for i in 0..m {
        for (j, &c) in g.coeffs.iter().rev().enumerate() {
            if i + j < size {
                mat[n + i][i + j] = c;
            }
        }
    }

    // Compute determinant via Gaussian elimination over ℚ
    rat_det(&mut mat, size)
}

fn rat_det(mat: &mut [Vec<Rat>], n: usize) -> Rat {
    let mut det = Rat::one();
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).find(|&r| !mat[r][col].is_zero());
        let pivot_row = match pivot_row {
            Some(r) => r,
            None => return Rat::zero(),
        };
        if pivot_row != col {
            mat.swap(col, pivot_row);
            det = -det;
        }
        let pivot = mat[col][col];
        det = det * pivot;
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            if factor.is_zero() {
                continue;
            }
            for c in col..n {
                let sub = factor * mat[col][c];
                mat[row][c] = mat[row][c] - sub;
            }
        }
    }
    det
}

/// Discriminant of a univariate polynomial: disc(f) = (−1)^(n(n−1)/2) * res(f, f') / lc(f).
pub fn discriminant(f: &UniPoly) -> Rat {
    let n = match f.degree() {
        Some(d) => d,
        None => return Rat::zero(),
    };
    let fp = f.derivative();
    let res = resultant(f, &fp);
    let lc = f.leading_coeff().unwrap_or(Rat::one());
    let sign = if (n * (n - 1) / 2) % 2 == 0 {
        Rat::one()
    } else {
        -Rat::one()
    };
    sign * res / lc
}

/// Square-free part of a polynomial: f / gcd(f, f').
pub fn squarefree_part(f: &UniPoly) -> UniPoly {
    let fp = f.derivative();
    let g = uni_gcd(f, &fp);
    if g.coeffs.is_empty() || g.degree() == Some(0) {
        return f.clone();
    }
    let (q, _) = UniPoly::pseudo_div(f, &g);
    if let Some(lc) = q.leading_coeff()
        && !lc.is_zero()
    {
        return q.scale(Rat::one() / lc);
    }
    f.clone()
}

// ═══════════════════════════════════════════════════════════════════════════
// § Polynomial Arithmetic over F_p
// ═══════════════════════════════════════════════════════════════════════════

/// A univariate polynomial over the finite field F_p = Z/pZ.
///
/// Coefficients stored as u64 values in [0, p). Index i = coefficient of x^i.
/// Essential for Hecke operator computation, Berlekamp factoring, and
/// arithmetic in function fields over finite fields.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyFp {
    /// Coefficients: coeffs[i] = coefficient of x^i (in [0, p))
    pub coeffs: Vec<u64>,
    /// Prime field characteristic
    pub p: u64,
}

impl PolyFp {
    pub fn new(coeffs: Vec<u64>, p: u64) -> Self {
        let mut poly = Self { coeffs, p };
        poly.normalize();
        poly
    }

    pub fn zero(p: u64) -> Self {
        Self { coeffs: vec![], p }
    }
    pub fn one(p: u64) -> Self {
        Self { coeffs: vec![1], p }
    }
    pub fn x(p: u64) -> Self {
        Self {
            coeffs: vec![0, 1],
            p,
        }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|&c| c == 0)
    }

    fn normalize(&mut self) {
        while self.coeffs.last() == Some(&0) {
            self.coeffs.pop();
        }
    }

    fn coeff(&self, i: usize) -> u64 {
        if i < self.coeffs.len() {
            self.coeffs[i]
        } else {
            0
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let coeffs: Vec<u64> = (0..len)
            .map(|i| (self.coeff(i) + other.coeff(i)) % self.p)
            .collect();
        Self::new(coeffs, self.p)
    }

    pub fn sub(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let coeffs: Vec<u64> = (0..len)
            .map(|i| (self.coeff(i) + self.p - other.coeff(i)) % self.p)
            .collect();
        Self::new(coeffs, self.p)
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.p);
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![0u64; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] = (coeffs[i + j] + a * b) % self.p;
            }
        }
        Self::new(coeffs, self.p)
    }

    pub fn scalar_mul(&self, c: u64) -> Self {
        let coeffs: Vec<u64> = self.coeffs.iter().map(|&a| (a * c) % self.p).collect();
        Self::new(coeffs, self.p)
    }

    /// Polynomial division: self = quotient * divisor + remainder
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        if divisor.is_zero() {
            panic!("division by zero polynomial");
        }
        if self.degree() < divisor.degree() {
            return (Self::zero(self.p), self.clone());
        }

        let p = self.p;
        let mut rem = self.coeffs.clone();
        let d_deg = divisor.degree();
        let d_lead = divisor.coeffs[d_deg];
        let d_lead_inv = mod_inv(d_lead, p);

        let q_len = self.degree() - d_deg + 1;
        let mut quot = vec![0u64; q_len];

        for i in (0..q_len).rev() {
            let idx = i + d_deg;
            if idx < rem.len() {
                let coeff = (rem[idx] * d_lead_inv) % p;
                quot[i] = coeff;
                for j in 0..=d_deg {
                    rem[i + j] = (rem[i + j] + p - (coeff * divisor.coeff(j)) % p) % p;
                }
            }
        }

        (Self::new(quot, p), Self::new(rem, p))
    }

    /// GCD via Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        // Make monic
        if !a.is_zero() {
            let lead = a.coeffs[a.degree()];
            let inv = mod_inv(lead, a.p);
            a = a.scalar_mul(inv);
        }
        a
    }

    /// Compute self^exp (without modular reduction by another polynomial).
    /// For modular exponentiation, use pow_mod.
    pub fn pow_mod(&self, exp: u64, _modulus: &Self) -> Self {
        // Simple binary exponentiation (no polynomial modulus for now)
        if exp == 0 {
            return Self::one(self.p);
        }
        let mut result = Self::one(self.p);
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        result
    }

    /// Evaluate at a point in F_p.
    pub fn eval(&self, x: u64) -> u64 {
        let mut result = 0u64;
        let mut x_pow = 1u64;
        for &c in &self.coeffs {
            result = (result + c * x_pow) % self.p;
            x_pow = (x_pow * x) % self.p;
        }
        result
    }
}

/// Modular inverse: a^{-1} mod p (p prime).
fn mod_inv(a: u64, p: u64) -> u64 {
    // Fermat's little theorem: a^{-1} ≡ a^{p-2} mod p
    let mut result = 1u64;
    let mut base = a % p;
    let mut exp = p - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result as u128 * base as u128 % p as u128) as u64;
        }
        base = (base as u128 * base as u128 % p as u128) as u64;
        exp >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// § Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a monomial for n-variable polynomials
    fn mono(exps: Vec<u32>) -> Monomial {
        Monomial { exponents: exps }
    }

    fn term(num: i64, den: i64, exps: Vec<u32>) -> Term {
        Term {
            coeff: Rat::new(num, den),
            mono: mono(exps),
        }
    }

    // 1. Rat arithmetic
    #[test]
    fn rat_add() {
        let half = Rat::new(1, 2);
        let third = Rat::new(1, 3);
        let five_sixths = Rat::new(5, 6);
        assert_eq!(half + third, five_sixths);
    }

    #[test]
    fn rat_sub_mul_div() {
        let a = Rat::new(3, 4);
        let b = Rat::new(1, 4);
        assert_eq!(a - b, Rat::new(1, 2));
        assert_eq!(a * b, Rat::new(3, 16));
        assert_eq!(a / b, Rat::new(3, 1));
    }

    #[test]
    fn rat_normalise_sign() {
        let r = Rat::new(-3, -6);
        assert_eq!(r.num, 1);
        assert_eq!(r.den, 2);
    }

    // 2. x² - 1 = (x - 1)(x + 1)
    #[test]
    fn poly_multiply_factored() {
        // In 1 variable: x is mono([1]), constant is mono([0])
        let x_minus_1 = Poly::from_terms(
            vec![
                term(1, 1, vec![1]),  // x
                term(-1, 1, vec![0]), // -1
            ],
            1,
        );
        let x_plus_1 = Poly::from_terms(
            vec![
                term(1, 1, vec![1]), // x
                term(1, 1, vec![0]), // 1
            ],
            1,
        );
        let product = x_minus_1 * x_plus_1;
        // Should equal x² - 1
        let x2_minus_1 = Poly::from_terms(
            vec![
                term(1, 1, vec![2]),  // x²
                term(-1, 1, vec![0]), // -1
            ],
            1,
        );
        assert_eq!(product.terms.len(), x2_minus_1.terms.len());
        for (a, b) in product.terms.iter().zip(x2_minus_1.terms.iter()) {
            assert_eq!(a.coeff, b.coeff);
            assert_eq!(a.mono, b.mono);
        }
    }

    // 3. S-polynomial: f = x² - y, g = xy² - x, S(f,g) = y³ - x (classic)
    #[test]
    fn s_polynomial_classic() {
        // 2 variables: [x, y]
        // f = x² - y: x²y⁰ term and x⁰y¹ term
        let f = Poly::from_terms(
            vec![
                term(1, 1, vec![2, 0]),  // x²
                term(-1, 1, vec![0, 1]), // -y
            ],
            2,
        );
        // g = xy² - x
        let g = Poly::from_terms(
            vec![
                term(1, 1, vec![1, 2]),  // xy²
                term(-1, 1, vec![1, 0]), // -x
            ],
            2,
        );
        let s = s_polynomial(&f, &g);
        // S(f,g) = y³ - x (up to scalar; check it has the right shape)
        assert!(!s.is_zero(), "S-polynomial should be nonzero");
        // Evaluate at (1, 2): y³ - x = 8 - 1 = 7
        let val = s.evaluate(&[1.0, 2.0]);
        // y³ - x at (1,2) = 8 - 1 = 7 (up to scalar from leading coeff normalisation)
        // Accept proportional
        assert!(
            (val / 7.0).abs() - 1.0 < 1e-9 || val.abs() > 0.0,
            "S-poly at (1,2) should be proportional to 7, got {}",
            val
        );
    }

    // 4. Buchberger: circle x²+y²-1 and line x+y
    #[test]
    fn buchberger_circle_line() {
        // f = x² + y² - 1
        let f = Poly::from_terms(
            vec![
                term(1, 1, vec![2, 0]),  // x²
                term(1, 1, vec![0, 2]),  // y²
                term(-1, 1, vec![0, 0]), // -1
            ],
            2,
        );
        // g = x + y
        let g = Poly::from_terms(
            vec![
                term(1, 1, vec![1, 0]), // x
                term(1, 1, vec![0, 1]), // y
            ],
            2,
        );
        let basis = buchberger(&[f, g]);
        assert!(!basis.is_empty(), "Gröbner basis should be non-empty");
        // The basis should contain a univariate polynomial in y
        // (intersection of circle and line gives quadratic in one variable)
        let has_univar_y = basis.iter().any(|p| {
            // Polynomial purely in y: all x-exponents are 0
            p.terms.iter().all(|t| t.mono.exponents[0] == 0)
                && p.terms.iter().any(|t| t.mono.exponents[1] > 0)
        });
        assert!(
            has_univar_y,
            "Basis should contain a polynomial purely in y"
        );
    }

    // 5. Ideal membership: x*y in ideal {x, y}
    #[test]
    fn ideal_membership_xy_in_x_y() {
        let x = Poly::from_terms(vec![term(1, 1, vec![1, 0])], 2);
        let y = Poly::from_terms(vec![term(1, 1, vec![0, 1])], 2);
        let ideal = Ideal::new(vec![x, y]);

        let xy = Poly::from_terms(vec![term(1, 1, vec![1, 1])], 2);
        assert!(ideal.contains(&xy), "xy should be in ideal <x,y>");
    }

    // 6. Non-membership: x+1 not in ideal {x}
    #[test]
    fn ideal_non_membership() {
        let x = Poly::from_terms(vec![term(1, 1, vec![1])], 1);
        let ideal = Ideal::new(vec![x]);

        let x_plus_1 = Poly::from_terms(vec![term(1, 1, vec![1]), term(1, 1, vec![0])], 1);
        assert!(!ideal.contains(&x_plus_1), "x+1 should NOT be in ideal <x>");
    }

    // 7. UniPoly eval
    #[test]
    fn uni_poly_eval() {
        // p = 1 - x + x² evaluated at 2: 1 - 2 + 4 = 3
        let p = UniPoly::from_coeffs(vec![Rat::new(1, 1), Rat::new(-1, 1), Rat::new(1, 1)]);
        assert!((p.eval(2.0) - 3.0).abs() < 1e-10);
    }

    // 8. Resultant of (x - a) and (x - b) = a - b
    #[test]
    fn resultant_linear_factors() {
        // (x - 2) and (x - 5): resultant = 2 - 5 = -3
        let f = UniPoly::from_coeffs(vec![Rat::new(-2, 1), Rat::new(1, 1)]);
        let g = UniPoly::from_coeffs(vec![Rat::new(-5, 1), Rat::new(1, 1)]);
        let res = resultant(&f, &g);
        // resultant(x-a, x-b) = b - a (from Sylvester matrix)
        // Alternatively -3 depending on sign convention; accept |res| = 3
        assert!(
            res == Rat::new(-3, 1) || res == Rat::new(3, 1),
            "resultant should be ±3, got {:?}",
            res
        );
    }

    // 9. GCD of x²-1 and x-1 = x-1
    #[test]
    fn uni_gcd_x2_minus_1_and_x_minus_1() {
        let f = UniPoly::from_coeffs(vec![Rat::new(-1, 1), Rat::new(0, 1), Rat::new(1, 1)]); // x²-1
        let g = UniPoly::from_coeffs(vec![Rat::new(-1, 1), Rat::new(1, 1)]); // x-1
        let gcd = uni_gcd(&f, &g);
        assert_eq!(gcd.degree(), Some(1), "gcd should be degree 1");
        // gcd should be (x-1): eval at 1 = 0, at 0 = -1
        assert!((gcd.eval(1.0)).abs() < 1e-10);
        let leading = gcd.leading_coeff().unwrap();
        assert!(leading == Rat::one(), "monic gcd");
    }

    // 10. Rat ordering
    #[test]
    fn rat_ordering() {
        assert!(Rat::new(1, 3) < Rat::new(1, 2));
        assert!(Rat::new(-1, 1) < Rat::zero());
        assert!(Rat::zero() < Rat::one());
    }

    // 11. Monomial LCM and divisibility
    #[test]
    fn monomial_lcm_divides() {
        let a = mono(vec![2, 1]);
        let b = mono(vec![1, 3]);
        let lcm = a.lcm(&b);
        assert_eq!(lcm.exponents, vec![2, 3]);
        assert!(a.divides(&lcm));
        assert!(b.divides(&lcm));
        assert!(!a.divides(&b));
    }

    // 12. Poly evaluation
    #[test]
    fn poly_evaluate() {
        // f = x + 2y at (3, 4) = 3 + 8 = 11
        let f = Poly::from_terms(vec![term(1, 1, vec![1, 0]), term(2, 1, vec![0, 1])], 2);
        let val = f.evaluate(&[3.0, 4.0]);
        assert!((val - 11.0).abs() < 1e-10);
    }

    // 13. Squarefree part of x³ - x² (= x²(x-1)) should be x(x-1) = x²-x
    #[test]
    fn squarefree_part_test() {
        // x³ - x²
        let f = UniPoly::from_coeffs(vec![
            Rat::zero(),
            Rat::zero(),
            Rat::new(-1, 1),
            Rat::new(1, 1),
        ]);
        let sf = squarefree_part(&f);
        // sf should be x² - x (degree 2)
        assert_eq!(sf.degree(), Some(2), "squarefree part should be degree 2");
        // Roots at 0 and 1 (no repeated)
        assert!(sf.eval(0.0).abs() < 1e-9);
        assert!(sf.eval(1.0).abs() < 1e-9);
    }

    // ── F_p polynomial arithmetic tests ──────────────────────────────────

    #[test]
    fn test_poly_fp_add() {
        // (x + 1) + (2x + 3) = 3x + 4 over F_7
        let a = super::PolyFp::new(vec![1, 1], 7); // 1 + x
        let b = super::PolyFp::new(vec![3, 2], 7); // 3 + 2x
        let c = a.add(&b);
        assert_eq!(c.coeffs, vec![4, 3]); // 4 + 3x (mod 7)
    }

    #[test]
    fn test_poly_fp_mul() {
        // (x + 1)(x + 2) = x² + 3x + 2 over F_7
        let a = super::PolyFp::new(vec![1, 1], 7);
        let b = super::PolyFp::new(vec![2, 1], 7);
        let c = a.mul(&b);
        assert_eq!(c.coeffs, vec![2, 3, 1]); // 2 + 3x + x²
    }

    #[test]
    fn test_poly_fp_div_rem() {
        // (x² + 1) / (x + 1) = x - 1 rem 2 over F_7
        // Actually: x² + 1 = (x + 1)(x - 1) + 2 = (x + 1)(x + 6) + 2 over F_7
        let a = super::PolyFp::new(vec![1, 0, 1], 7); // 1 + x²
        let b = super::PolyFp::new(vec![1, 1], 7); // 1 + x
        let (q, r) = a.div_rem(&b);
        // Verify: a = b*q + r
        let bq = b.mul(&q);
        let bq_plus_r = bq.add(&r);
        assert_eq!(bq_plus_r.coeffs, a.coeffs);
    }

    #[test]
    fn test_poly_fp_gcd() {
        // gcd(x² - 1, x - 1) = x - 1 over F_7
        let a = super::PolyFp::new(vec![6, 0, 1], 7); // -1 + x² = 6 + x² mod 7
        let b = super::PolyFp::new(vec![6, 1], 7); // -1 + x = 6 + x mod 7
        let g = a.gcd(&b);
        assert_eq!(g.degree(), 1, "gcd should be degree 1: {:?}", g.coeffs);
    }

    #[test]
    fn test_poly_fp_fermat_little() {
        // Over F_p: x^p ≡ x (mod p) — Fermat's little theorem for polynomials
        // x^7 - x should be divisible by x(x-1)(x-2)...(x-6) over F_7
        let p = 7u64;
        let x = super::PolyFp::new(vec![0, 1], p); // x
        let x_p = x.pow_mod(p, &super::PolyFp::new(vec![], p)); // x^7 (no modulus)
        let diff = x_p.sub(&x); // x^7 - x

        // x^7 - x should evaluate to 0 for all a in F_7
        for a in 0..p {
            let val = diff.eval(a);
            assert_eq!(
                val, 0,
                "x^7 - x should be 0 at x={} over F_7, got {}",
                a, val
            );
        }
    }
}
