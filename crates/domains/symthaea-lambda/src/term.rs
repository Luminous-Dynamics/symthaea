// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Untyped lambda terms in **de Bruijn** notation, with capture-free
//! substitution and normal-order (leftmost-outermost) reduction.
//!
//! Using de Bruijn indices makes α-equivalence *structural equality*, so there
//! is no variable-capture problem to get wrong.

/// A lambda term. `Var` is a de Bruijn index (0 = the nearest enclosing `Abs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Var(usize),
    Abs(Box<Term>),
    App(Box<Term>, Box<Term>),
}

/// `λ. body`.
pub fn abs(body: Term) -> Term {
    Term::Abs(Box::new(body))
}

/// Application `f a`.
pub fn app(f: Term, a: Term) -> Term {
    Term::App(Box::new(f), Box::new(a))
}

/// Shift the free variables of `t` (those with index ≥ `cutoff`) by `d`.
fn shift(t: &Term, d: i64, cutoff: usize) -> Term {
    match t {
        Term::Var(k) => {
            if *k >= cutoff {
                Term::Var((*k as i64 + d) as usize)
            } else {
                Term::Var(*k)
            }
        }
        Term::Abs(b) => Term::Abs(Box::new(shift(b, d, cutoff + 1))),
        Term::App(f, a) => Term::App(Box::new(shift(f, d, cutoff)), Box::new(shift(a, d, cutoff))),
    }
}

/// Substitute `s` for the variable with index `j` in `t`.
fn subst(t: &Term, j: usize, s: &Term) -> Term {
    match t {
        Term::Var(k) => {
            if *k == j {
                s.clone()
            } else {
                Term::Var(*k)
            }
        }
        Term::Abs(b) => Term::Abs(Box::new(subst(b, j + 1, &shift(s, 1, 0)))),
        Term::App(f, a) => Term::App(Box::new(subst(f, j, s)), Box::new(subst(a, j, s))),
    }
}

/// β-reduce the redex `(λ. body) arg`.
fn beta(body: &Term, arg: &Term) -> Term {
    // Substitute the shifted argument for index 0, then shift the result down.
    shift(&subst(body, 0, &shift(arg, 1, 0)), -1, 0)
}

/// One normal-order (leftmost-outermost) reduction step, reducing under
/// abstractions too so that full normal forms are reachable. `None` if `t` is
/// already in normal form.
pub fn step(t: &Term) -> Option<Term> {
    match t {
        Term::App(f, a) => {
            if let Term::Abs(body) = f.as_ref() {
                Some(beta(body, a)) // top-level redex
            } else if let Some(f2) = step(f) {
                Some(Term::App(Box::new(f2), a.clone()))
            } else {
                step(a).map(|a2| Term::App(f.clone(), Box::new(a2)))
            }
        }
        Term::Abs(b) => step(b).map(|b2| Term::Abs(Box::new(b2))),
        Term::Var(_) => None,
    }
}

/// Reduce to normal form, up to `max_steps` reductions. `None` if it did not
/// terminate within the budget (β-reduction is not guaranteed to terminate).
pub fn normalize(t: &Term, max_steps: usize) -> Option<Term> {
    let mut cur = t.clone();
    for _ in 0..max_steps {
        match step(&cur) {
            Some(next) => cur = next,
            None => return Some(cur),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_application() {
        // (λ. 0) applied to a free variable → that variable.
        let id = abs(Term::Var(0));
        let t = app(id, Term::Var(5));
        assert_eq!(normalize(&t, 100).unwrap(), Term::Var(5));
    }

    #[test]
    fn k_combinator_discards_second_arg() {
        // K = λ.λ.1 ;  K a b → a.
        let k = abs(abs(Term::Var(1)));
        let t = app(app(k, Term::Var(7)), Term::Var(9));
        assert_eq!(normalize(&t, 100).unwrap(), Term::Var(7));
    }

    #[test]
    fn no_variable_capture() {
        // (λ. λ. 1) 0  — the free 0 must NOT be captured by the inner λ.
        // Result is λ. 1 (the outer free variable, shifted under one λ).
        let t = app(abs(abs(Term::Var(1))), Term::Var(0));
        assert_eq!(normalize(&t, 100).unwrap(), abs(Term::Var(1)));
    }
}
