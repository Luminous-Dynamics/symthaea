// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_regenerative_resilience::{
    BasisId, CanonicalId, CanonicalQuantity, EventTiming, ExactRatio, IdentifierError,
    NonNegativeQuantity, QuantityError, Tick, TimeError, TimeInterval, UnitId, MAX_DECIMAL_SCALE,
    MAX_ID_BYTES,
};

fn unit(name: &str) -> UnitId {
    UnitId::new(name).unwrap()
}

fn basis(name: &str) -> BasisId {
    BasisId::new(name).unwrap()
}

#[test]
fn canonical_id_rejects_empty_non_ascii_invalid_and_overlong_values() {
    assert_eq!(CanonicalId::new(""), Err(IdentifierError::Empty));
    assert!(matches!(
        CanonicalId::new("snowman-☃"),
        Err(IdentifierError::NonAscii { .. })
    ));
    assert!(matches!(
        CanonicalId::new("contains space"),
        Err(IdentifierError::InvalidByte { .. })
    ));
    let overlong = "a".repeat(MAX_ID_BYTES + 1);
    assert_eq!(
        CanonicalId::new(overlong),
        Err(IdentifierError::TooLong {
            actual: MAX_ID_BYTES + 1,
            maximum: MAX_ID_BYTES,
        })
    );
}

#[test]
fn canonical_id_preserves_exact_case_and_path_punctuation() {
    let id = CanonicalId::new("Mycelix/Service:Water.v1").unwrap();
    assert_eq!(id.as_str(), "Mycelix/Service:Water.v1");
    assert_ne!(
        id,
        CanonicalId::new("mycelix/service:water.v1").unwrap()
    );
}

#[test]
fn fixed_decimal_quantity_has_one_trailing_zero_normal_form() {
    let quantity = CanonicalQuantity::new(1200, 2, unit("kg"), basis("dry")).unwrap();
    assert_eq!(quantity.mantissa(), 12);
    assert_eq!(quantity.scale(), 0);

    let zero = CanonicalQuantity::new(0, 9, unit("kg"), basis("dry")).unwrap();
    assert_eq!(zero.mantissa(), 0);
    assert_eq!(zero.scale(), 0);
}

#[test]
fn excessive_decimal_scale_is_rejected() {
    assert!(matches!(
        CanonicalQuantity::new(
            1,
            MAX_DECIMAL_SCALE + 1,
            unit("kg"),
            basis("dry")
        ),
        Err(QuantityError::ScaleTooLarge { .. })
    ));
}

#[test]
fn compatible_cross_scale_addition_and_subtraction_are_exact() {
    let a = CanonicalQuantity::new(12, 1, unit("kg"), basis("dry")).unwrap(); // 1.2
    let b = CanonicalQuantity::new(3, 2, unit("kg"), basis("dry")).unwrap(); // 0.03
    let sum = a.checked_add(&b).unwrap();
    assert_eq!(sum.mantissa(), 123);
    assert_eq!(sum.scale(), 2);

    let restored = sum.checked_sub(&b).unwrap();
    assert_eq!(restored, a);
}

#[test]
fn incompatible_units_and_bases_never_mix() {
    let dry_kg = CanonicalQuantity::new(10, 0, unit("kg"), basis("dry")).unwrap();
    let dry_mg = CanonicalQuantity::new(10, 0, unit("mg"), basis("dry")).unwrap();
    let wet_kg = CanonicalQuantity::new(10, 0, unit("kg"), basis("as_received")).unwrap();

    assert!(matches!(
        dry_kg.checked_add(&dry_mg),
        Err(QuantityError::UnitMismatch { .. })
    ));
    assert!(matches!(
        dry_kg.checked_add(&wet_kg),
        Err(QuantityError::BasisMismatch { .. })
    ));
}

#[test]
fn exact_arithmetic_reports_overflow_instead_of_wrapping() {
    let max = CanonicalQuantity::new(i128::MAX, 0, unit("u"), basis("b")).unwrap();
    let one = CanonicalQuantity::new(1, 0, unit("u"), basis("b")).unwrap();
    assert_eq!(max.checked_add(&one), Err(QuantityError::Overflow));

    let tiny_scale = CanonicalQuantity::new(i128::MAX, 0, unit("u"), basis("b")).unwrap();
    let high_scale = CanonicalQuantity::new(1, 18, unit("u"), basis("b")).unwrap();
    assert_eq!(
        tiny_scale.checked_add(&high_scale),
        Err(QuantityError::Overflow)
    );
}

#[test]
fn nonnegative_quantity_rejects_negative_inputs_and_underflowing_subtraction() {
    assert_eq!(
        NonNegativeQuantity::new(-1, 0, unit("kg"), basis("dry")),
        Err(QuantityError::NegativeQuantity)
    );

    let one = NonNegativeQuantity::new(1, 0, unit("kg"), basis("dry")).unwrap();
    let two = NonNegativeQuantity::new(2, 0, unit("kg"), basis("dry")).unwrap();
    assert_eq!(one.checked_sub(&two), Err(QuantityError::NegativeQuantity));
}

#[test]
fn exact_ratio_is_reduced_and_zero_denominator_fails_closed() {
    let half = ExactRatio::new(2, 4).unwrap();
    assert_eq!(half.numerator(), 1);
    assert_eq!(half.denominator(), 2);
    assert_eq!(
        ExactRatio::new(1, 0),
        Err(QuantityError::ZeroRatioDenominator)
    );
}

#[test]
fn exact_ratio_floor_application_preserves_unit_basis_and_nonnegativity() {
    let source = NonNegativeQuantity::new(101, 0, unit("kg"), basis("dry")).unwrap();
    let half = ExactRatio::new(1, 2).unwrap();
    let result = half.apply_floor(&source).unwrap();
    assert_eq!(result.quantity().mantissa(), 50);
    assert_eq!(result.quantity().scale(), 0);
    assert_eq!(result.quantity().unit().as_str(), "kg");
    assert_eq!(result.quantity().basis().as_str(), "dry");
}

#[test]
fn time_interval_is_nonempty_half_open_and_exact() {
    let interval = TimeInterval::new(Tick(10), Tick(15)).unwrap();
    assert_eq!(interval.duration_ticks(), 5);
    assert!(interval.contains(Tick(10)));
    assert!(interval.contains(Tick(14)));
    assert!(!interval.contains(Tick(15)));
    assert!(!interval.contains(Tick(9)));
}

#[test]
fn empty_or_reversed_intervals_are_rejected() {
    assert_eq!(
        TimeInterval::new(Tick(5), Tick(5)),
        Err(TimeError::EmptyOrReversedInterval {
            start: Tick(5),
            end_exclusive: Tick(5),
        })
    );
    assert!(matches!(
        TimeInterval::new(Tick(6), Tick(5)),
        Err(TimeError::EmptyOrReversedInterval { .. })
    ));
}

#[test]
fn instantaneous_timing_is_distinct_from_interval_timing() {
    let instant = EventTiming::Instant(Tick(5));
    assert!(instant.active_at(Tick(5)));
    assert!(!instant.active_at(Tick(6)));

    let interval = EventTiming::Interval(TimeInterval::new(Tick(5), Tick(6)).unwrap());
    assert!(interval.active_at(Tick(5)));
    assert!(!interval.active_at(Tick(6)));
    assert_ne!(instant, interval);
}
