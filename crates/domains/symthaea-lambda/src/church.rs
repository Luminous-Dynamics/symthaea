// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Church encodings (numerals, booleans, arithmetic) and the S/K/I combinators,
//! all as de Bruijn [`Term`]s — the classic demonstration that the untyped
//! lambda calculus is a full model of computation.

use crate::term::{Term, abs, app};

/// The Church numeral `n = λf. λx. fⁿ x`.
pub fn numeral(n: usize) -> Term {
    // Under two abstractions: f = Var(1), x = Var(0).
    let mut body = Term::Var(0);
    for _ in 0..n {
        body = app(Term::Var(1), body);
    }
    abs(abs(body))
}

/// `add = λm. λn. λf. λx. m f (n f x)`.
pub fn add() -> Term {
    // m=3, n=2, f=1, x=0.
    abs(abs(abs(abs(app(
        app(Term::Var(3), Term::Var(1)),
        app(app(Term::Var(2), Term::Var(1)), Term::Var(0)),
    )))))
}

/// `mul = λm. λn. λf. m (n f)`.
pub fn mul() -> Term {
    // m=2, n=1, f=0.
    abs(abs(abs(app(Term::Var(2), app(Term::Var(1), Term::Var(0))))))
}

/// `succ = λn. λf. λx. f (n f x)`.
pub fn succ() -> Term {
    // n=2, f=1, x=0.
    abs(abs(abs(app(
        Term::Var(1),
        app(app(Term::Var(2), Term::Var(1)), Term::Var(0)),
    ))))
}

/// Church `true = λx. λy. x`.
pub fn tru() -> Term {
    abs(abs(Term::Var(1)))
}

/// Church `false = λx. λy. y`.
pub fn fls() -> Term {
    abs(abs(Term::Var(0)))
}

/// The identity combinator `I = λx. x`.
pub fn i() -> Term {
    abs(Term::Var(0))
}

/// The `K = λx. λy. x` combinator (also Church `true`).
pub fn k() -> Term {
    abs(abs(Term::Var(1)))
}

/// The `S = λx. λy. λz. x z (y z)` combinator.
pub fn s() -> Term {
    // x=2, y=1, z=0.
    abs(abs(abs(app(
        app(Term::Var(2), Term::Var(0)),
        app(Term::Var(1), Term::Var(0)),
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::{app, normalize};

    #[test]
    fn addition() {
        // add 2 3 → 5.
        let t = app(app(add(), numeral(2)), numeral(3));
        assert_eq!(normalize(&t, 1000).unwrap(), numeral(5));
    }

    #[test]
    fn multiplication() {
        // mul 2 3 → 6.
        let t = app(app(mul(), numeral(2)), numeral(3));
        assert_eq!(normalize(&t, 1000).unwrap(), numeral(6));
    }

    #[test]
    fn successor() {
        let t = app(succ(), numeral(4));
        assert_eq!(normalize(&t, 1000).unwrap(), numeral(5));
    }

    #[test]
    fn skk_equals_identity() {
        // S K K reduces to the identity combinator.
        let t = app(app(s(), k()), k());
        assert_eq!(normalize(&t, 1000).unwrap(), i());
    }

    #[test]
    fn booleans_select() {
        // true a b → a ; false a b → b.
        let t_true = app(app(tru(), Term::Var(3)), Term::Var(4));
        assert_eq!(normalize(&t_true, 100).unwrap(), Term::Var(3));
        let t_false = app(app(fls(), Term::Var(3)), Term::Var(4));
        assert_eq!(normalize(&t_false, 100).unwrap(), Term::Var(4));
    }
}
