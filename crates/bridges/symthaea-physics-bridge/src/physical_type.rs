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

fn derived_kind_add(a: QuantityKind, b: QuantityKind) -> QuantityKind {
    use QuantityKind::*;
    match (a, b) {
        (Temperature, TemperatureDifference)
        | (TemperatureDifference, Temperature) => Temperature,
        (TemperatureDifference, TemperatureDifference) => TemperatureDifference,
        (left, right) if left == right => left,
        _ => Unknown,
    }
}

fn derived_kind_sub(a: QuantityKind, b: QuantityKind) -> QuantityKind {
    use QuantityKind::*;
    match (a, b) {
        (Temperature, Temperature) => TemperatureDifference,
        (Temperature, TemperatureDifference) => Temperature,
        (TemperatureDifference, TemperatureDifference) => TemperatureDifference,
        (left, right) if left == right => left,
        _ => Unknown,
    }
}

fn combine_scalar(a: ScalarDomain, b: ScalarDomain, division: bool) -> ScalarDomain {
    use ScalarDomain::*;
    match (a, b) {
        (Unknown, _) | (_, Unknown) => Unknown,
        (Complex, _) | (_, Complex) => Complex,
        (ApproximateReal, _) | (_, ApproximateReal) => ApproximateReal,
        (Real, _) | (_, Real) => Real,
        (Integer, Integer) if division => Rational,
        (Integer, Integer) => Integer,
        (Integer, Rational) | (Rational, Integer) | (Rational, Rational) => Rational,
    }
}

/// Scalar domain after an analytic real-valued function. Integer/rational
/// inputs can yield irrational values, while complex and approximate-real
/// domains must remain explicitly represented.
fn analytic_scalar(input: ScalarDomain) -> ScalarDomain {
    match input {
        ScalarDomain::Unknown => ScalarDomain::Unknown,
        ScalarDomain::Complex => ScalarDomain::Complex,
        ScalarDomain::ApproximateReal => ScalarDomain::ApproximateReal,
        ScalarDomain::Integer | ScalarDomain::Rational | ScalarDomain::Real => {
            ScalarDomain::Real
        }
    }
}

/// A variable exponent makes even a dimensionless power conservative:
/// unknown/complex/approximate scalar information must not collapse to Real.
fn power_scalar(base: ScalarDomain, exponent: ScalarDomain) -> ScalarDomain {
    match (base, exponent) {
        (ScalarDomain::Unknown, _) | (_, ScalarDomain::Unknown) => ScalarDomain::Unknown,
        (ScalarDomain::Complex, _) | (_, ScalarDomain::Complex) => ScalarDomain::Complex,
        (ScalarDomain::ApproximateReal, _) | (_, ScalarDomain::ApproximateReal) => {
            ScalarDomain::ApproximateReal
        }
        _ => ScalarDomain::Real,
    }
}

fn additive_result(
    left: &PhysicalType,
    right: &PhysicalType,
    operation: BinOp,
) -> TypeJudgement<PhysicalType> {
    let (Some(left_dimension), Some(right_dimension)) = (left.dimension, right.dimension) else {
        return TypeJudgement::Unknown(
            "one or both additive operands have unknown physical type".into(),
        );
    };
    if left_dimension != right_dimension {
        return TypeJudgement::Invalid(PhysicalTypeError {
            operation: "add/sub".into(),
            reason: "operands require compatible physical kind and dimension".into(),
        });
    }

    if left.kind == QuantityKind::Unknown || right.kind == QuantityKind::Unknown {
        return TypeJudgement::Unknown(
            "add/sub requires known quantity kinds when physical dimension is shared".into(),
        );
    }

    let result_kind = match operation {
        BinOp::Add => derived_kind_add(left.kind, right.kind),
        BinOp::Sub => derived_kind_sub(left.kind, right.kind),
        _ => unreachable!("additive_result only accepts add/sub"),
    };

    if result_kind == QuantityKind::Unknown {
        return TypeJudgement::Invalid(PhysicalTypeError {
            operation: "add/sub".into(),
            reason: "operands require compatible physical kind and dimension".into(),
        });
    }

    let mut result = if result_kind == left.kind && operation == BinOp::Add {
        left.clone()
    } else if result_kind == right.kind && operation == BinOp::Add {
        right.clone()
    } else {
        PhysicalType::with_kind(result_kind, left_dimension)
    };
    if !(result_kind == left.kind && operation == BinOp::Add)
        && !(result_kind == right.kind && operation == BinOp::Add)
    {
        result.scalar = combine_scalar(left.scalar, right.scalar, false);
    }

    // Subtracting two absolute temperatures yields a temperature difference.
    // Do not carry a Celsius/Fahrenheit affine offset into the delta result.
    if result_kind == QuantityKind::TemperatureDifference
        && (left.kind == QuantityKind::Temperature || right.kind == QuantityKind::Temperature)
    {
        result.unit = None;
    }

    TypeJudgement::Valid(result)
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
        Expr::Var(name) => {
            let physical_type = lookup(name);
            match physical_type.validate() {
                Ok(()) => valid(physical_type),
                Err(error) => TypeJudgement::Invalid(error),
            }
        },
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
                        let mut result = PhysicalType::dimensionless();
                        result.scalar = analytic_scalar(arg_ty.scalar);
                        valid(result)
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: format!("{function:?}"),
                            reason: "trigonometric input must be angle or dimensionless".into(),
                        })
                    }
                }
                UnaryFn::Exp => {
                    if dimension.is_dimensionless() {
                        let mut result = PhysicalType::dimensionless();
                        result.scalar = analytic_scalar(arg_ty.scalar);
                        valid(result)
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
                        let mut result = PhysicalType::dimensionless();
                        result.scalar = analytic_scalar(arg_ty.scalar);
                        valid(result)
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
                        let mut result = PhysicalType::with_kind(
                            Unknown,
                            PhysicalDimension::from_array(a.map(|e| e / 2)),
                        );
                        result.scalar = analytic_scalar(arg_ty.scalar);
                        valid(result)
                    } else {
                        TypeJudgement::Unknown(
                            "real square-root requires a nonnegative domain refinement".into(),
                        )
                    }
                }
                UnaryFn::Abs | UnaryFn::Floor => valid(arg_ty),
            }
        }

        Expr::BinOp(Add, left, right) => {
            let l = match infer_expr_type_with_lookup(left, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let r = match infer_expr_type_with_lookup(right, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            additive_result(&l, &r, Add)
        }

        Expr::BinOp(Sub, left, right) => {
            let l = match infer_expr_type_with_lookup(left, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            let r = match infer_expr_type_with_lookup(right, lookup) {
                TypeJudgement::Valid(t) => t,
                other => return other,
            };
            additive_result(&l, &r, Sub)
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
                Some(dimension) => {
                    let mut result = PhysicalType::with_kind(
                        if l.kind == Unknown || r.kind == Unknown {
                            Unknown
                        } else {
                            derived_kind_mul(l.kind, r.kind)
                        },
                        dimension,
                    );
                    result.scalar = combine_scalar(l.scalar, r.scalar, false);
                    valid(result)
                },
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
                    Some(dimension) => {
                        let mut result = PhysicalType::with_kind(
                            derived_kind_div(l.kind, r.kind),
                            dimension,
                        );
                        result.scalar = combine_scalar(l.scalar, r.scalar, true);
                        valid(result)
                    },
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
                Expr::Const(k) => {
                    if !k.is_finite() {
                        return TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "power exponent must be finite".into(),
                        });
                    }

                    if (k - k.round()).abs() < 1e-9 {
                        let rounded = k.round();
                        if rounded < i8::MIN as f64 || rounded > i8::MAX as f64 {
                            return TypeJudgement::Invalid(PhysicalTypeError {
                                operation: "pow".into(),
                                reason:
                                    "integer power exponent is outside the supported i8 range"
                                        .into(),
                            });
                        }
                        match dimension.scale(rounded as i8) {
                            Some(d) => {
                                let mut result = PhysicalType::with_kind(Unknown, d);
                                result.scalar = if rounded < 0.0 {
                                    match b.scalar {
                                        ScalarDomain::Integer | ScalarDomain::Rational => {
                                            ScalarDomain::Rational
                                        }
                                        other => other,
                                    }
                                } else {
                                    b.scalar
                                };
                                valid(result)
                            }
                            None => TypeJudgement::Invalid(PhysicalTypeError {
                                operation: "pow".into(),
                                reason: "dimension exponent overflow".into(),
                            }),
                        }
                    } else if (*k - 0.5).abs() < 1e-9 {
                        let a = dimension.as_array();
                        if a.iter().all(|e| e % 2 == 0) {
                            let mut result = PhysicalType::with_kind(
                                Unknown,
                                PhysicalDimension::from_array(a.map(|e| e / 2)),
                            );
                            result.scalar = analytic_scalar(b.scalar);
                            valid(result)
                        } else {
                            TypeJudgement::Invalid(PhysicalTypeError {
                                operation: "pow".into(),
                                reason: "square-root exponent requires even dimensions".into(),
                            })
                        }
                    } else if dimension.is_dimensionless() {
                        let mut result = PhysicalType::dimensionless();
                        result.scalar = analytic_scalar(b.scalar);
                        valid(result)
                    } else {
                        TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: format!(
                                "non-integer exponent {k} requires dimensionless base"
                            ),
                        })
                    }
                }
                exponent_expr => {
                    let exponent_ty = match infer_expr_type_with_lookup(exponent_expr, lookup) {
                        TypeJudgement::Valid(t) => t,
                        other => return other,
                    };

                    if !exponent_ty
                        .dimension
                        .is_some_and(PhysicalDimension::is_dimensionless)
                    {
                        return TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "exponent must be dimensionless".into(),
                        });
                    }

                    if !dimension.is_dimensionless() {
                        return TypeJudgement::Invalid(PhysicalTypeError {
                            operation: "pow".into(),
                            reason: "variable exponent requires dimensionless base".into(),
                        });
                    }

                    let mut result = PhysicalType::dimensionless();
                    result.scalar = power_scalar(b.scalar, exponent_ty.scalar);
                    if result.scalar == ScalarDomain::Unknown {
                        TypeJudgement::Unknown(
                            "variable power requires known scalar domains for base and exponent"
                                .into(),
                        )
                    } else {
                        valid(result)
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
    fn temperature_subtraction_yields_temperature_difference() {
        let absolute = PhysicalType::with_kind(
            QuantityKind::Temperature,
            PhysicalDimension::TEMPERATURE,
        );
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Var("t1".into())),
            Box::new(Expr::Var("t2".into())),
        );
        let variables = HashMap::from([
            ("t1".into(), absolute.clone()),
            ("t2".into(), absolute),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.kind, QuantityKind::TemperatureDifference);
                assert_eq!(result.dimension, Some(PhysicalDimension::TEMPERATURE));
                assert!(result.unit.is_none());
            }
            other => panic!("unexpected temperature subtraction judgment: {other:?}"),
        }
    }

    #[test]
    fn temperature_plus_difference_yields_absolute_temperature() {
        let absolute = PhysicalType::with_kind(
            QuantityKind::Temperature,
            PhysicalDimension::TEMPERATURE,
        );
        let difference = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        );
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("t".into())),
            Box::new(Expr::Var("dt".into())),
        );
        let variables = HashMap::from([
            ("t".into(), absolute.clone()),
            ("dt".into(), difference),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => assert_eq!(result.kind, QuantityKind::Temperature),
            other => panic!("unexpected temperature addition judgment: {other:?}"),
        }
    }

    #[test]
    fn temperature_difference_minus_absolute_temperature_is_invalid() {
        let difference = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        );
        let absolute = PhysicalType::with_kind(
            QuantityKind::Temperature,
            PhysicalDimension::TEMPERATURE,
        );
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Var("dt".into())),
            Box::new(Expr::Var("t".into())),
        );
        let variables = HashMap::from([
            ("dt".into(), difference),
            ("t".into(), absolute),
        ]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Invalid(_)
        ));
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
    fn explicit_variable_with_incoherent_physical_type_is_rejected() {
        let malformed = PhysicalType::with_kind(
            QuantityKind::Energy,
            PhysicalDimension::LENGTH,
        );
        let expr = Expr::Var("bad".into());
        let variables = HashMap::from([("bad".into(), malformed)]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
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
    fn unknown_quantity_kind_is_not_promoted_to_valid_addition() {
        let known =
            PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
        let unknown_same_dimension =
            PhysicalType::with_kind(QuantityKind::Unknown, PhysicalDimension::ENERGY);
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("known".into())),
            Box::new(Expr::Var("unknown".into())),
        );
        let variables = HashMap::from([
            ("known".into(), known),
            ("unknown".into(), unknown_same_dimension),
        ]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn scalar_domain_is_preserved_through_multiplication() {
        let mut mass =
            PhysicalType::with_kind(QuantityKind::Mass, PhysicalDimension::MASS);
        mass.scalar = ScalarDomain::Complex;
        let acceleration =
            PhysicalType::with_kind(QuantityKind::Acceleration, PhysicalDimension::ACCELERATION);
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::Var("a".into())),
        );
        let variables = HashMap::from([
            ("m".into(), mass),
            ("a".into(), acceleration),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.kind, QuantityKind::Force);
                assert_eq!(result.scalar, ScalarDomain::Complex);
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn integer_division_does_not_claim_an_integer_result() {
        let mut length =
            PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        length.scalar = ScalarDomain::Integer;
        let mut time =
            PhysicalType::with_kind(QuantityKind::Time, PhysicalDimension::TIME);
        time.scalar = ScalarDomain::Integer;
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("t".into())),
        );
        let variables = HashMap::from([
            ("x".into(), length),
            ("t".into(), time),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.kind, QuantityKind::Velocity);
                assert_eq!(result.scalar, ScalarDomain::Rational);
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn approximate_scalar_domain_does_not_collapse_to_exact_real() {
        let mut energy =
            PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
        energy.scalar = ScalarDomain::ApproximateReal;
        let mut time =
            PhysicalType::with_kind(QuantityKind::Time, PhysicalDimension::TIME);
        time.scalar = ScalarDomain::Real;
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("e".into())),
            Box::new(Expr::Var("t".into())),
        );
        let variables = HashMap::from([
            ("e".into(), energy),
            ("t".into(), time),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.kind, QuantityKind::Power);
                assert_eq!(result.scalar, ScalarDomain::ApproximateReal);
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn variable_exponent_requires_dimensionless_exponent() {
        let base = PhysicalType::dimensionless();
        let exponent = PhysicalType::with_kind(
            QuantityKind::Length,
            PhysicalDimension::LENGTH,
        );
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("base".into())),
            Box::new(Expr::Var("exponent".into())),
        );
        let variables = HashMap::from([
            ("base".into(), base),
            ("exponent".into(), exponent),
        ]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn variable_exponent_with_unknown_scalar_stays_unknown() {
        let base = PhysicalType::dimensionless();
        let mut exponent = PhysicalType::dimensionless();
        exponent.scalar = ScalarDomain::Unknown;
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("base".into())),
            Box::new(Expr::Var("exponent".into())),
        );
        let variables = HashMap::from([
            ("base".into(), base),
            ("exponent".into(), exponent),
        ]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn complex_variable_exponent_preserves_complex_power_domain() {
        let mut base = PhysicalType::dimensionless();
        base.scalar = ScalarDomain::Real;
        let mut exponent = PhysicalType::dimensionless();
        exponent.scalar = ScalarDomain::Complex;
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("base".into())),
            Box::new(Expr::Var("exponent".into())),
        );
        let variables = HashMap::from([
            ("base".into(), base),
            ("exponent".into(), exponent),
        ]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.scalar, ScalarDomain::Complex);
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn oversized_integer_power_exponent_is_rejected() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(128.0)),
        );
        let variables = HashMap::from([(
            "x".into(),
            PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH),
        )]);

        assert!(matches!(
            infer_expr_type_with_variables(&expr, &variables),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn non_finite_power_exponent_is_rejected_even_for_dimensionless_base() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Const(f64::NAN)),
        );

        assert!(matches!(
            infer_expr_type(&expr, &units()),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn complex_sine_does_not_collapse_to_real() {
        let mut input = PhysicalType::dimensionless();
        input.scalar = ScalarDomain::Complex;
        let expr = Expr::Func(UnaryFn::Sin, Box::new(Expr::Var("z".into())));
        let variables = HashMap::from([("z".into(), input)]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.scalar, ScalarDomain::Complex);
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn rational_log_has_real_result_domain() {
        let mut input = PhysicalType::dimensionless();
        input.scalar = ScalarDomain::Rational;
        input.refinements.push(Refinement::Positive);
        let expr = Expr::Func(UnaryFn::Log, Box::new(Expr::Var("x".into())));
        let variables = HashMap::from([("x".into(), input)]);

        match infer_expr_type_with_variables(&expr, &variables) {
            TypeJudgement::Valid(result) => {
                assert_eq!(result.scalar, ScalarDomain::Real);
            }
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
