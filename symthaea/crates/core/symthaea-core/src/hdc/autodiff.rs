// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Generic Reverse-Mode Automatic Differentiation

use num_complex::Complex64;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Default
    + Send
    + Sync
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(f: f64) -> Self;
    fn to_f64(self) -> f64; // Real part or value
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
    fn norm_sq(self) -> f64;

    fn safe_exp(self) -> Self;
    fn safe_ln(self) -> Self;

    fn push_node(value: Self, parents: Vec<(usize, Self)>) -> usize;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(f: f64) -> Self {
        f
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn norm_sq(self) -> f64 {
        self * self
    }

    fn safe_exp(self) -> Self {
        self.clamp(-100.0, 80.0).exp()
    }
    fn safe_ln(self) -> Self {
        if self > 1e-15 {
            self.ln()
        } else {
            -34.538776394910684
        }
    }

    fn push_node(value: Self, parents: Vec<(usize, Self)>) -> usize {
        F64_TAPE.with(|t| {
            let mut opt = t.borrow_mut();
            let tape = opt.as_mut().expect("f64 AD tape not initialized");
            let index = tape.nodes.len();
            tape.nodes.push(TapeNode {
                index,
                parents,
                value,
            });
            index
        })
    }
}

impl Scalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn from_f64(f: f64) -> Self {
        Complex64::new(f, 0.0)
    }
    fn to_f64(self) -> f64 {
        self.re
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn norm_sq(self) -> f64 {
        self.norm_sqr()
    }

    fn safe_exp(self) -> Self {
        let re = self.re.clamp(-100.0, 80.0);
        Complex64::new(re, self.im).exp()
    }
    fn safe_ln(self) -> Self {
        if self.norm_sqr() > 1e-30 {
            self.ln()
        } else {
            Complex64::new(-34.538776394910684, 0.0)
        }
    }

    fn push_node(value: Self, parents: Vec<(usize, Self)>) -> usize {
        C64_TAPE.with(|t| {
            let mut opt = t.borrow_mut();
            let tape = opt.as_mut().expect("C64 AD tape not initialized");
            let index = tape.nodes.len();
            tape.nodes.push(TapeNode {
                index,
                parents,
                value,
            });
            index
        })
    }
}

#[derive(Debug, Clone)]
struct TapeNode<T: Scalar> {
    index: usize,
    parents: Vec<(usize, T)>,
    value: T,
}

#[derive(Debug)]
pub struct Tape<T: Scalar> {
    nodes: Vec<TapeNode<T>>,
}

thread_local! {
    static F64_TAPE: RefCell<Option<Tape<f64>>> = const { RefCell::new(None) };
    static C64_TAPE: RefCell<Option<Tape<Complex64>>> = const { RefCell::new(None) };
}

impl<T: Scalar> Tape<T> {
    fn new() -> Self {
        Tape {
            nodes: Vec::with_capacity(128),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenericVar<T: Scalar> {
    pub index: usize,
    pub value: T,
}

pub type Var = GenericVar<f64>;

impl<T: Scalar> GenericVar<T> {
    pub fn new(value: T) -> Self {
        let index = T::push_node(value, vec![]);
        GenericVar { index, value }
    }

    pub fn constant(value: T) -> Self {
        let index = T::push_node(value, vec![]);
        GenericVar { index, value }
    }

    fn push_node(value: T, parents: Vec<(usize, T)>) -> Self {
        let index = T::push_node(value, parents);
        GenericVar { index, value }
    }

    pub fn add(self, other: Self) -> Self {
        Self::push_node(
            self.value + other.value,
            vec![(self.index, T::one()), (other.index, T::one())],
        )
    }

    pub fn sub(self, other: Self) -> Self {
        Self::push_node(
            self.value - other.value,
            vec![(self.index, T::one()), (other.index, -T::one())],
        )
    }

    pub fn mul(self, other: Self) -> Self {
        Self::push_node(
            self.value * other.value,
            vec![(self.index, other.value), (other.index, self.value)],
        )
    }

    pub fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        let d_self = T::one() / other.value;
        let d_other = -self.value / (other.value * other.value);
        Self::push_node(value, vec![(self.index, d_self), (other.index, d_other)])
    }

    pub fn exp(self) -> Self {
        let val = self.value.safe_exp();
        Self::push_node(val, vec![(self.index, val)])
    }

    pub fn ln(self) -> Self {
        let val = self.value.safe_ln();
        Self::push_node(val, vec![(self.index, T::one() / self.value)])
    }

    pub fn eml(self, other: Self) -> Self {
        let val_x = self.value.safe_exp();
        let val_y = other.value.safe_ln();
        let value = val_x - val_y;
        Self::push_node(
            value,
            vec![(self.index, val_x), (other.index, -T::one() / other.value)],
        )
    }

    pub fn neg(self) -> Self {
        Self::push_node(-self.value, vec![(self.index, -T::one())])
    }
}

pub fn ad_begin_f64() {
    F64_TAPE.with(|t| *t.borrow_mut() = Some(Tape::new()));
}
pub fn ad_begin_c64() {
    C64_TAPE.with(|t| *t.borrow_mut() = Some(Tape::new()));
}
pub fn ad_begin() {
    ad_begin_f64();
}

pub fn ad_gradient_f64(output: GenericVar<f64>) -> Vec<f64> {
    F64_TAPE.with(|t| {
        let t = t.borrow();
        let tape = t.as_ref().expect("f64 AD tape not initialized");
        let n = tape.nodes.len();
        let mut adjoints = vec![0.0; n];
        adjoints[output.index] = 1.0;
        for i in (0..n).rev() {
            let adj = adjoints[i];
            if adj == 0.0 {
                continue;
            }
            for &(p_idx, local_grad) in &tape.nodes[i].parents {
                adjoints[p_idx] += adj * local_grad;
            }
        }
        adjoints
    })
}

pub fn ad_gradient_c64(output: GenericVar<Complex64>) -> Vec<Complex64> {
    C64_TAPE.with(|t| {
        let t = t.borrow();
        let tape = t.as_ref().expect("C64 AD tape not initialized");
        let n = tape.nodes.len();
        let mut adjoints = vec![Complex64::zero(); n];
        adjoints[output.index] = Complex64::one();
        for i in (0..n).rev() {
            let adj = adjoints[i];
            if adj == Complex64::zero() {
                continue;
            }
            for &(p_idx, local_grad) in &tape.nodes[i].parents {
                adjoints[p_idx] += adj * local_grad;
            }
        }
        adjoints
    })
}

pub fn ad_gradient(output: Var) -> Vec<f64> {
    ad_gradient_f64(output)
}

pub fn ad_end() {
    F64_TAPE.with(|t| *t.borrow_mut() = None);
    C64_TAPE.with(|t| *t.borrow_mut() = None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ad_basic() {
        ad_begin_f64();
        let x = Var::new(2.0);
        let y = Var::new(3.0);
        let z = x.mul(y).add(x);
        assert_eq!(z.value, 8.0);
        let grad = ad_gradient_f64(z);
        assert_eq!(grad[x.index], 4.0);
        assert_eq!(grad[y.index], 2.0);
        ad_end();
    }
}
