//! Gradual physical typing shared by discovery and engineering.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use crate::dimensional_inference::UnitMap;
use crate::types::DimensionalSignature;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantityKind {
    Unknown, Dimensionless, Angle, Length, Mass, Time, Velocity, Acceleration,
    Momentum, Force, Energy, Power, Charge, Temperature, Pressure, Stress,
    Strain, Frequency, Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ScalarDomain { Unknown, Integer, Rational, Real, Complex, ApproximateReal }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Refinement { Positive, NonNegative, NonZero, Bounded, Periodic }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalType {
    pub kind: QuantityKind,
    /// None means genuinely unknown, never dimensionless.
    pub dimension: Option<DimensionalSignature>,
    pub scalar: ScalarDomain,
    pub refinements: Vec<Refinement>,
}

impl PhysicalType {
    pub fn unknown() -> Self {
        Self { kind: QuantityKind::Unknown, dimension: None, scalar: ScalarDomain::Unknown, refinements: Vec::new() }
    }
    pub fn dimensionless() -> Self {
        Self { kind: QuantityKind::Dimensionless, dimension: Some(DimensionalSignature::DIMENSIONLESS), scalar: ScalarDomain::Real, refinements: Vec::new() }
    }
    pub fn with_kind(kind: QuantityKind, dimension: DimensionalSignature) -> Self {
        Self { kind, dimension: Some(dimension), scalar: ScalarDomain::Real, refinements: Vec::new() }
    }
    pub fn has_refinement(&self, refinement: Refinement) -> bool { self.refinements.contains(&refinement) }
    fn compatible_additive(&self, other: &Self) -> Option<bool> {
        match (self.dimension, other.dimension) {
            (Some(a), Some(b)) if a == b => Some(self.kind == QuantityKind::Unknown || other.kind == QuantityKind::Unknown || self.kind == other.kind),
            (Some(_), Some(_)) => Some(false),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeJudgement<T> { Valid(T), Invalid(PhysicalTypeError), Unknown(String) }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalTypeError { pub operation: String, pub reason: String }

fn derived_kind_mul(a: QuantityKind, b: QuantityKind) -> QuantityKind {
    use QuantityKind::*;
    match (a, b) {
        (Mass, Acceleration) | (Acceleration, Mass) => Force,
        (Force, Length) | (Length, Force) => Energy,
        (Velocity, Time) | (Time, Velocity) => Length,
        _ => Unknown,
    }
}

fn derived_kind_div(a: QuantityKind, b: QuantityKind) -> QuantityKind {
    use QuantityKind::*;
    match (a, b) {
        (Energy, Time) => Power,
        (Length, Time) => Velocity,
        (Velocity, Time) => Acceleration,
        _ => Unknown,
    }
}

fn kind_from_dimension(d: DimensionalSignature) -> QuantityKind {
    use QuantityKind::*;
    if d.is_dimensionless() { Dimensionless }
    else if d == DimensionalSignature::LENGTH { Length }
    else if d == DimensionalSignature::MASS { Mass }
    else if d == DimensionalSignature::TIME { Time }
    else if d == DimensionalSignature::VELOCITY { Velocity }
    else if d == DimensionalSignature::ACCELERATION { Acceleration }
    else if d == DimensionalSignature::MOMENTUM { Momentum }
    else if d == DimensionalSignature::FORCE { Force }
    else if d == DimensionalSignature::ENERGY { Energy }
    else if d == DimensionalSignature::PRESSURE { Pressure }
    else { Custom }
}

pub fn variable_type(name: &str, units: &UnitMap) -> PhysicalType {
    match units.get(name).copied() {
        Some(d) => PhysicalType::with_kind(kind_from_dimension(d), d),
        None => PhysicalType::unknown(),
    }
}

fn valid(t: PhysicalType) -> TypeJudgement<PhysicalType> { TypeJudgement::Valid(t) }

pub fn infer_expr_type(expr: &Expr, units: &UnitMap) -> TypeJudgement<PhysicalType> {
    use BinOp::*;
    use QuantityKind::*;
    match expr {
        Expr::Var(name) => valid(variable_type(name, units)),
        Expr::Const(_) => valid(PhysicalType::dimensionless()),
        Expr::Sum(body, _) => infer_expr_type(body, units),
        Expr::Func(function, arg) => {
            let arg_ty = match infer_expr_type(arg, units) { TypeJudgement::Valid(t) => t, other => return other };
            let Some(dimension) = arg_ty.dimension else { return TypeJudgement::Unknown("function input physical type is unknown".into()) };
            match function {
                UnaryFn::Sin | UnaryFn::Cos => if arg_ty.kind == Angle || dimension.is_dimensionless() { valid(PhysicalType::dimensionless()) } else { TypeJudgement::Invalid(PhysicalTypeError { operation: format!("{:?}", function), reason: "trigonometric input must be angle or dimensionless".into() }) },
                UnaryFn::Exp => if dimension.is_dimensionless() { valid(PhysicalType::dimensionless()) } else { TypeJudgement::Invalid(PhysicalTypeError { operation: "exp".into(), reason: "exponential input must be dimensionless".into() }) },
                UnaryFn::Log => if !dimension.is_dimensionless() { TypeJudgement::Invalid(PhysicalTypeError { operation: "log".into(), reason: "log input must be dimensionless; normalize by a reference quantity first".into() }) } else if arg_ty.has_refinement(Refinement::Positive) { valid(PhysicalType::dimensionless()) } else { TypeJudgement::Unknown("log domain positivity is not established".into()) },
                UnaryFn::Sqrt => {
                    let a = dimension.as_array();
                    if a.iter().all(|e| e % 2 == 0) { valid(PhysicalType::with_kind(Unknown, DimensionalSignature::from_array(a.map(|e| e / 2)))) }
                    else { TypeJudgement::Invalid(PhysicalTypeError { operation: "sqrt".into(), reason: "dimension exponents must be even".into() }) }
                }
                UnaryFn::Abs | UnaryFn::Floor => valid(arg_ty),
            }
        }
        Expr::BinOp(Add, left, right) | Expr::BinOp(Sub, left, right) => {
            let l = match infer_expr_type(left, units) { TypeJudgement::Valid(t) => t, other => return other };
            let r = match infer_expr_type(right, units) { TypeJudgement::Valid(t) => t, other => return other };
            match l.compatible_additive(&r) {
                Some(true) => valid(l),
                Some(false) => TypeJudgement::Invalid(PhysicalTypeError { operation: "add/sub".into(), reason: "operands require compatible physical kind and dimension".into() }),
                None => TypeJudgement::Unknown("one or both additive operands have unknown physical type".into()),
            }
        }
        Expr::BinOp(Mul, left, right) => {
            let l = match infer_expr_type(left, units) { TypeJudgement::Valid(t) => t, other => return other };
            let r = match infer_expr_type(right, units) { TypeJudgement::Valid(t) => t, other => return other };
            let (Some(ld), Some(rd)) = (l.dimension, r.dimension) else { return TypeJudgement::Unknown("multiplicative operand dimension is unknown".into()) };
            valid(PhysicalType::with_kind(if l.kind == Unknown || r.kind == Unknown { Unknown } else { derived_kind_mul(l.kind, r.kind) }, ld.add(&rd)))
        }
        Expr::BinOp(Div, left, right) => {
            let l = match infer_expr_type(left, units) { TypeJudgement::Valid(t) => t, other => return other };
            let r = match infer_expr_type(right, units) { TypeJudgement::Valid(t) => t, other => return other };
            let (Some(ld), Some(rd)) = (l.dimension, r.dimension) else { return TypeJudgement::Unknown("division operand dimension is unknown".into()) };
            if r.kind == Unknown || r.has_refinement(Refinement::NonZero) { valid(PhysicalType::with_kind(derived_kind_div(l.kind, r.kind), ld.sub(&rd))) }
            else { TypeJudgement::Unknown("division denominator is not established nonzero".into()) }
        }
        Expr::BinOp(Pow, base, exponent) => {
            let b = match infer_expr_type(base, units) { TypeJudgement::Valid(t) => t, other => return other };
            let Some(dimension) = b.dimension else { return TypeJudgement::Unknown("power base physical type is unknown".into()) };
            match exponent.as_ref() {
                Expr::Const(k) if (k - k.round()).abs() < 1e-9 => match dimension.scale(*k as i8) {
                    Some(d) => valid(PhysicalType::with_kind(Unknown, d)),
                    None => TypeJudgement::Invalid(PhysicalTypeError { operation: "pow".into(), reason: "dimension exponent overflow".into() }),
                },
                Expr::Const(k) if (*k - 0.5).abs() < 1e-9 => {
                    let a=dimension.as_array();
                    if a.iter().all(|e| e % 2 == 0) { valid(PhysicalType::with_kind(Unknown, DimensionalSignature::from_array(a.map(|e| e/2)))) }
                    else { TypeJudgement::Invalid(PhysicalTypeError { operation: "pow".into(), reason: "square-root exponent requires even dimensions".into() }) }
                }
                Expr::Const(k) => if dimension.is_dimensionless() { valid(PhysicalType::dimensionless()) } else { TypeJudgement::Invalid(PhysicalTypeError { operation: "pow".into(), reason: format!("non-integer exponent {k} requires dimensionless base") }) },
                _ => if dimension.is_dimensionless() { valid(PhysicalType::dimensionless()) } else { TypeJudgement::Invalid(PhysicalTypeError { operation: "pow".into(), reason: "variable exponent requires dimensionless base".into() }) },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    fn units() -> UnitMap { HashMap::from([
        ("m".into(), DimensionalSignature::MASS),
        ("v".into(), DimensionalSignature::VELOCITY),
        ("t".into(), DimensionalSignature::TIME),
        ("E".into(), DimensionalSignature::ENERGY),
        ("x".into(), DimensionalSignature::LENGTH),
    ]) }
    #[test] fn rejects_energy_plus_length() {
        let e=Expr::BinOp(BinOp::Add,Box::new(Expr::Var("E".into())),Box::new(Expr::Var("x".into())));
        assert!(matches!(infer_expr_type(&e,&units()),TypeJudgement::Invalid(_)));
    }
    #[test] fn unknown_is_not_dimensionless() {
        match infer_expr_type(&Expr::Var("mystery".into()),&units()) { TypeJudgement::Valid(t)=>assert!(t.dimension.is_none()), _=>panic!() }
    }
    #[test] fn unknown_function_input_stays_unknown() {
        let e=Expr::Func(UnaryFn::Sin,Box::new(Expr::Var("mystery".into())));
        assert!(matches!(infer_expr_type(&e,&units()),TypeJudgement::Unknown(_)));
    }
    #[test] fn mass_acceleration_is_force() {
        let expr=Expr::BinOp(BinOp::Mul,Box::new(Expr::Var("m".into())),Box::new(Expr::BinOp(BinOp::Div,Box::new(Expr::Var("v".into())),Box::new(Expr::Var("t".into())))));
        match infer_expr_type(&expr,&units()) { TypeJudgement::Valid(t)=>assert_eq!(t.kind,QuantityKind::Force), other=>panic!("{other:?}") }
    }
    #[test] fn log_of_energy_is_invalid() {
        let e=Expr::Func(UnaryFn::Log,Box::new(Expr::Var("E".into())));
        assert!(matches!(infer_expr_type(&e,&units()),TypeJudgement::Invalid(_)));
    }
}
