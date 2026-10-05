//! Expression-level inference over the canonical shared physical types.

use std::collections::HashMap;
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_types::{
    PhysicalDimension, PhysicalType, PhysicalTypeError, QuantityKind, Refinement, ScalarDomain,
    TypeJudgement,
};
use crate::dimensional_inference::UnitMap;
use crate::types::DimensionalSignature;

pub use symthaea_types::{
    ModelMaturity, UnitRef, UnitScale,
};

fn to_shared_dimension(d: DimensionalSignature) -> PhysicalDimension {
    PhysicalDimension::from_array(d.as_array())
}

fn from_shared_dimension(d: PhysicalDimension) -> DimensionalSignature {
    DimensionalSignature::from_array(d.as_array())
}

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
    else if d == DimensionalSignature::ENERGY { Custom }
    else if d == DimensionalSignature::PRESSURE { Custom }
    else { Custom }
}

/// Construct an explicitly typed quantity when dimensions alone are ambiguous.
pub fn explicit_type(kind: QuantityKind, dimension: DimensionalSignature) -> PhysicalType {
    PhysicalType::with_kind(kind, to_shared_dimension(dimension))
}

pub fn variable_type(name: &str, units: &UnitMap) -> PhysicalType {
    match units.get(name).copied() {
        Some(d) => PhysicalType::with_kind(kind_from_dimension(d), to_shared_dimension(d)),
        None => PhysicalType::unknown(),
    }
}

fn valid(t: PhysicalType) -> TypeJudgement<PhysicalType> {
    TypeJudgement::Valid(t)
}

fn constant_type(value: f64) -> TypeJudgement<PhysicalType> {
    if !value.is_finite() {
        return TypeJudgement::Invalid(PhysicalTypeError {
            operation: "constant".into(),
            reason: "non-finite constant".into(),
        });
    }
    let mut ty = PhysicalType::dimensionless();
    if value > 0.0 {
        ty.refinements.push(Refinement::Positive);
        ty.refinements.push(Refinement::NonZero);
    } else if value < 0.0 {
        ty.refinements.push(Refinement::NonZero);
    } else {
        ty.refinements.push(Refinement::NonNegative);
    }
    valid(ty)
}

pub fn infer_expr_type(expr: &Expr, units: &UnitMap) -> TypeJudgement<PhysicalType> {
    infer_expr_type_with_lookup(expr, &|name| variable_type(name, units))
}

/// Infer using explicit physical type annotations. Prefer this when semantic
/// quantity kind matters (for example Energy versus Torque).
pub fn infer_expr_type_with_variables(
    expr: &Expr,
    variables: &HashMap<String, PhysicalType>,
) -> TypeJudgement<PhysicalType> {
    infer_expr_type_with_lookup(expr, &|name| {
        variables.get(name).cloned().unwrap_or_else(PhysicalType::unknown)
    })
}

fn infer_expr_type_with_lookup<F>(expr: &Expr, lookup: &F) -> TypeJudgement<PhysicalType>
where
    F: Fn(&str) -> PhysicalType,
{
    use BinOp::*;
    use QuantityKind::*;

    match expr {
        Expr::Var(name) => valid(lookup(name)),
        Expr::Const(value) => constant_type(*value),
        Expr::Sum(body, _) => infer_expr_type_with_lookup(body, lookup),

        Expr::Func(function, arg) => {
            let arg_ty = match infer_expr_type_with_lookup(arg, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let Some(dimension) = arg_ty.dimension else {
                return TypeJudgement::Unknown("function input physical type is unknown".into());
            };

            match function {
                UnaryFn::Sin | UnaryFn::Cos => {
                    if arg_ty.kind == Angle || dimension.is_dimensionless() {
                        valid(PhysicalType::dimensionless())
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: format!("{function:?}"),
                            reason: "trigonometric input must be angle or dimensionless".into(),
                        })
                    }
                }
                UnaryFn::Exp => {
                    if dimension.is_dimensionless() {
                        valid(PhysicalType::dimensionless())
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "exp".into(),
                            reason: "exponential input must be dimensionless".into(),
                        })
                    }
                }
                UnaryFn::Log => {
                    if !dimension.is_dimensionless() {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "log".into(),
                            reason: "log input must be dimensionless; normalize by a reference quantity first".into(),
                        })
                    } else if arg_ty.has_refinement(Refinement::Positive) {
                        valid(PhysicalType::dimensionless())
                    } else {
                        TypeJudgement::Unknown("log domain positivity is not established".into())
                    }
                }
                UnaryFn::Sqrt => {
                    let a = dimension.as_array();
                    if !a.iter().all(|e| e % 2 == 0) {
                        return TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "sqrt".into(),
                            reason: "dimension exponents must be even".into(),
                        });
                    }
                    if arg_ty.scalar == ScalarDomain::Complex
                        || arg_ty.has_refinement(Refinement::NonNegative)
                    {
                        valid(PhysicalType::with_kind(
                            Unknown,
                            PhysicalDimension::from_array(a.map(|e| e / 2)),
                        ))
                    } else {
                        TypeJudgement::Unknown(
                            "real square-root requires a nonnegative domain refinement".into(),
                        )
                    }
                }
                UnaryFn::Abs | UnaryFn::Floor => valid(arg_ty),
            }
        }

        Expr::BinOp(Add, left, right) | Expr::BinOp(Sub, left, right) => {
            let l = match infer_expr_type_with_lookup(left, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let r = match infer_expr_type_with_lookup(right, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            match (l.dimension, r.dimension) {
                (Some(ld), Some(rd)) if ld == rd
                    && (l.kind == Unknown || r.kind == Unknown || l.kind == r.kind) =>
                {
                    valid(l)
                }
                (Some(_), Some(_)) => TypeJudgement::Invalid(PhysicalTypeError {
                    operation: "add/sub".into(),
                    reason: "operands require compatible physical kind and dimension".into(),
                }),
                _ => TypeJudgement::Unknown(
                    "one or both additive operands have unknown physical type".into(),
                ),
            }
        }

        Expr::BinOp(Mul, left, right) => {
            let l = match infer_expr_type_with_lookup(left, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let r = match infer_expr_type_with_lookup(right, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let (Some(ld), Some(rd)) = (l.dimension, r.dimension) else {
                return TypeJudgement::Unknown(
                    "multiplicative operand dimension is unknown".into(),
                );
            };
            match ld.checked_add(rd) {
                Some(dimension) => valid(PhysicalType::with_kind(
                    if l.kind == Unknown || r.kind == Unknown {
                        Unknown
                    } else {
                        derived_kind_mul(l.kind, r.kind)
                    },
                    dimension,
                )),
                None => TypeJudgement::Invalid(PhysicalTypeError {
                    operation: "mul".into(),
                    reason: "physical dimension exponent overflow".into(),
                }),
            }
        }

        Expr::BinOp(Div, left, right) => {
            let l = match infer_expr_type_with_lookup(left, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let r = match infer_expr_type_with_lookup(right, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let (Some(ld), Some(rd)) = (l.dimension, r.dimension) else {
                return TypeJudgement::Unknown("division operand dimension is unknown".into());
            };
            if r.has_refinement(Refinement::NonZero) {
                match ld.checked_sub(rd) {
                    Some(dimension) => valid(PhysicalType::with_kind(
                        derived_kind_div(l.kind, r.kind),
                        dimension,
                    )),
                    None => TypeJudgement::Invalid(PhysicalTypeError {
                        operation: "div".into(),
                        reason: "physical dimension exponent overflow".into(),
                    }),
                }
            } else {
                TypeJudgement::Unknown(
                    "division denominator is not established nonzero".into(),
                )
            }
        }

        Expr::BinOp(Pow, base, exponent) => {
            let b = match infer_expr_type_with_lookup(base, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let Some(dimension) = b.dimension else {
                return TypeJudgement::Unknown("power base physical type is unknown".into());
            };

            match exponent.as_ref() {
                Expr::Const(k) if (k - k.round()).abs() < 1e-9 => {
                    match dimension.scale(*k as i8) {
                        Some(d) => valid(PhysicalType::with_kind(Unknown, d)),
                        None => TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "dimension exponent overflow".into(),
                        }),
                    }
                }
                Expr::Const(k) if (*k - 0.5).abs() < 1e-9 => {
                    let a = dimension.as_array();
                    if a.iter().all(|e| e % 2 == 0) {
                        valid(PhysicalType::with_kind(
                            Unknown,
                            PhysicalDimension::from_array(a.map(|e| e / 2)),
                        ))
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "square-root exponent requires even dimensions".into(),
                        })
                    }
                }
                Expr::Const(k) => {
                    if dimension.is_dimensionless() {
                        valid(PhysicalType::dimensionless())
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: format!(
                                "non-integer exponent {k} requires dimensionless base"
                            ),
                        })
                    }
                }
                _ => {
                    if dimension.is_dimensionless() {
                        valid(PhysicalType::dimensionless())
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "variable exponent requires dimensionless base".into(),
                        })
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units() -> UnitMap {
        std::collections::HashMap::from([
            ("m".into(), DimensionalSignature::MASS),
            ("v".into(), DimensionalSignature::VELOCITY),
            ("t".into(), DimensionalSignature::TIME),
            ("E".into(), DimensionalSignature::ENERGY),
            ("x".into(), DimensionalSignature::LENGTH),
        ])
    }

    #[test]
    fn rejects_energy_plus_length() {
        let e = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("E".into())),
            Box::new(Expr::Var("x".into())),
        );
        assert!(matches!(
            infer_expr_type(&e, &units()),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn unknown_is_not_dimensionless() {
        match infer_expr_type(&Expr::Var("mystery".into()), &units()) {
            TypeJudgement::Valid(t) => assert!(t.dimension.is_none()),
            _ => panic!("unknown variables should produce an explicit unknown type"),
        }
    }

    #[test]
    fn unknown_function_input_stays_unknown() {
        let e = Expr::Func(UnaryFn::Sin, Box::new(Expr::Var("mystery".into())));
        assert!(matches!(
            infer_expr_type(&e, &units()),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn mass_acceleration_is_force() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Var("v".into())),
                Box::new(Expr::Var("t".into())),
            )),
        );
        match infer_expr_type(&expr, &units()) {
            TypeJudgement::Valid(t) => assert_eq!(t.kind, QuantityKind::Force),
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn explicit_variables_are_used_through_nested_arithmetic() {
        let variables = HashMap::from([
            (
                String::from("m"),
                PhysicalType::with_kind(QuantityKind::Mass, PhysicalDimension::MASS),
            ),
            (
                String::from("v"),
                PhysicalType::with_kind(QuantityKind::Velocity, PhysicalDimension::VELOCITY),
            ),
            (
                String::from("t"),
                PhysicalType::with_kind(QuantityKind::Time, PhysicalDimension::TIME),
            ),
        ]);
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Var("v".into())),
                Box::new(Expr::Var("t".into())),
            )),
        );
        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(t) => assert_eq!(t.kind, QuantityKind::Force),
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn explicit_energy_and_torque_are_not_additively_compatible() {
        let energy = explicit_type(QuantityKind::Energy, DimensionalSignature::ENERGY);
        let torque = explicit_type(QuantityKind::Torque, DimensionalSignature::ENERGY);
        let variables = HashMap::from([
            (String::from("E"), energy),
            (String::from("tau"), torque),
        ]);
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("E".into())),
            Box::new(Expr::Var("tau".into())),
        );
        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn nonzero_constant_can_be_a_division_denominator() {
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        );
        assert!(matches!(infer_expr_type(&expr, &units()), TypeJudgement::Valid(_)));
    }

    #[test]
    fn zero_constant_is_not_a_valid_division_denominator() {
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(0.0)),
        );
        assert!(matches!(
            infer_expr_type(&expr, &units()),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn square_root_requires_real_domain_refinement() {
        let expr = Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Var("E".into())));
        assert!(matches!(
            infer_expr_type(&expr, &units()),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn log_of_energy_is_invalid() {
        let e = Expr::Func(UnaryFn::Log, Box::new(Expr::Var("E".into())));
        assert!(matches!(
            infer_expr_type(&e, &units()),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn shared_roundtrip_preserves_dimensions() {
        let d = PhysicalDimension::from_array(DimensionalSignature::ENERGY.as_array());
        assert_eq!(from_shared_dimension(d), DimensionalSignature::ENERGY);
    }
}
