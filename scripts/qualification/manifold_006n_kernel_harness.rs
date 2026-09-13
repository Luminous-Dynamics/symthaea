// Standalone qualification harness for MANIFOLD-006N.
// Intentionally std-only so it can be compiled directly with rustc.

#[path = "../../crates/core/symthaea-core/src/certified_interval.rs"]
mod certified_interval;

use certified_interval::{
    CertifiedIntervalError, OutwardInterval, canonical_zero, next_down, next_up, outward_add,
    outward_div, outward_mul, outward_sub,
};
use std::{env, fs::File, io::Write, path::PathBuf};

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn emit(
    out: &mut File,
    op: &str,
    a: OutwardInterval,
    b: OutwardInterval,
    result: OutwardInterval,
) {
    writeln!(
        out,
        "vector\t{op}\t{}\t{}\t{}\t{}\t{}\t{}",
        bits(a.lower),
        bits(a.upper),
        bits(b.lower),
        bits(b.upper),
        bits(result.lower),
        bits(result.upper),
    )
    .unwrap();
}

fn emit_rejection(out: &mut File, op: &str, left: f64, right: f64, reason: &str) {
    writeln!(
        out,
        "reject\t{op}\t{}\t{}\t{reason}",
        bits(left),
        bits(right)
    )
    .unwrap();
}

fn point(value: f64) -> OutwardInterval {
    OutwardInterval::point(value).unwrap()
}

fn emit_scalar_result(
    out: &mut File,
    op: &str,
    left: f64,
    right: f64,
    result: Result<OutwardInterval, CertifiedIntervalError>,
) {
    match result {
        Ok(interval) => emit(out, op, point(left), point(right), interval),
        Err(error) => {
            let reason = if op.starts_with("div") && right == 0.0 {
                assert_eq!(
                    error.reason(),
                    "division by zero in outward robust arithmetic"
                );
                "division-by-zero"
            } else {
                "finite-domain"
            };
            emit_rejection(out, op, left, right, reason);
        }
    }
}

fn main() {
    let output = PathBuf::from(env::args().nth(1).expect("output path argument"));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut out = File::create(output).unwrap();
    writeln!(out, "schema\tsymthaea.manifold-006n.kernel-vectors.v3").unwrap();
    writeln!(out, "encoding\tieee754-binary64-u64-hex").unwrap();

    for (op, left, right) in [
        ("add-decimal", 0.1, 0.2),
        ("add-cancel", 1.0, -1.0),
        ("sub-decimal", 0.3, 0.1),
        ("mul-decimal", 0.1, 0.3),
        ("mul-underflow", f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        ("mul-power2", 0.5, 2.0),
        ("mul-negative-one", 0.125, -1.0),
        ("div-decimal", 1.0, 10.0),
        ("div-power2", 1.0, 2.0),
    ] {
        let result = match op {
            name if name.starts_with("add") => outward_add(left, right),
            name if name.starts_with("sub") => outward_sub(left, right),
            name if name.starts_with("mul") => outward_mul(left, right),
            name if name.starts_with("div") => outward_div(left, right),
            _ => unreachable!(),
        };
        emit_scalar_result(&mut out, op, left, right, result);
    }

    let a = OutwardInterval::ordered(-0.1, 0.2, "harness a").unwrap();
    let b = OutwardInterval::ordered(0.3, 0.4, "harness b").unwrap();
    emit(&mut out, "interval-add", a, b, a.add(b).unwrap());
    emit(&mut out, "interval-sub", a, b, a.sub(b).unwrap());
    emit(
        &mut out,
        "interval-mul-positive",
        a,
        point(0.5),
        a.mul_positive(0.5).unwrap(),
    );
    emit(
        &mut out,
        "interval-div-positive",
        a,
        point(2.0),
        a.div_positive(2.0).unwrap(),
    );

    // Deterministic adversarial scalar corpus. It spans signed zero, both sides of
    // the subnormal/normal boundary, mantissa neighbors around 1, exact powers of
    // two, non-dyadic decimal inputs, very large/small normal magnitudes, and the
    // finite binary64 extrema. Every ordered pair is exercised for +, -, *, and /.
    let min_subnormal = f64::from_bits(1);
    let max_subnormal = f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1);
    let next_normal = f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1);
    let one_down = f64::from_bits(1.0_f64.to_bits() - 1);
    let one_up = f64::from_bits(1.0_f64.to_bits() + 1);
    let tiny_normal = 2.0_f64.powi(-500);
    let huge_normal = 2.0_f64.powi(500);
    let adversarial = [
        0.0,
        -0.0,
        min_subnormal,
        -min_subnormal,
        max_subnormal,
        -max_subnormal,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        next_normal,
        -next_normal,
        tiny_normal,
        -tiny_normal,
        0.1,
        -0.1,
        0.3,
        -0.3,
        0.5,
        -0.5,
        one_down,
        -one_down,
        1.0,
        -1.0,
        one_up,
        -one_up,
        2.0,
        -2.0,
        huge_normal,
        -huge_normal,
        f64::MAX,
        -f64::MAX,
    ];
    writeln!(out, "meta\tadversarial-value-count\t{}", adversarial.len()).unwrap();

    for (left_index, left) in adversarial.iter().copied().enumerate() {
        for (right_index, right) in adversarial.iter().copied().enumerate() {
            for (kind, result) in [
                ("add", outward_add(left, right)),
                ("sub", outward_sub(left, right)),
                ("mul", outward_mul(left, right)),
                ("div", outward_div(left, right)),
            ] {
                let op = format!("{kind}-sweep-{left_index:02}-{right_index:02}");
                emit_scalar_result(&mut out, &op, left, right, result);
            }
        }
    }

    // Qualified 006C distinguishes point canonicalization from ordered endpoint bits.
    assert_eq!(canonical_zero(-0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(point(-0.0).lower.to_bits(), 0.0_f64.to_bits());
    let ordered_zero = OutwardInterval::ordered(-0.0, 0.0, "signed-zero ordered").unwrap();
    assert_eq!(ordered_zero.lower.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(ordered_zero.upper.to_bits(), 0.0_f64.to_bits());

    assert_eq!(next_up(0.0).unwrap().to_bits(), 1);
    assert_eq!(next_down(0.0).unwrap().to_bits(), (1_u64 << 63) | 1);
    assert_eq!(next_up(-f64::from_bits(1)).unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(next_down(f64::from_bits(1)).unwrap().to_bits(), 0.0_f64.to_bits());
    assert!(next_up(f64::MAX).is_err());
    assert!(next_down(-f64::MAX).is_err());
    assert!(OutwardInterval::ordered(1.0, -1.0, "reversed").is_err());
    assert!(a.mul_positive(0.0).is_err());
    assert!(a.div_positive(0.0).is_err());

    let divide_by_zero = outward_div(1.0, 0.0).unwrap_err();
    assert_eq!(
        divide_by_zero.reason(),
        "division by zero in outward robust arithmetic"
    );

    // Non-finite results must fail closed rather than fabricate finite enclosure endpoints.
    assert!(outward_add(f64::MAX, f64::MAX).is_err());
    assert!(outward_mul(f64::MAX, 2.0).is_err());
    assert!(outward_div(f64::MAX, f64::MIN_POSITIVE).is_err());

    // Critical fast-path regression from qualified 006C.
    let underflow = outward_mul(f64::MIN_POSITIVE, f64::MIN_POSITIVE).unwrap();
    assert!(underflow.lower <= 0.0);
    assert!(underflow.upper > 0.0);

    writeln!(out, "gate\tsigned-zero-canonical\tPASS").unwrap();
    writeln!(out, "gate\tordered-signed-zero-preserved\tPASS").unwrap();
    writeln!(out, "gate\tneighbor-zero-subnormal\tPASS").unwrap();
    writeln!(out, "gate\tfinite-edge-rejection\tPASS").unwrap();
    writeln!(out, "gate\tnonfinite-result-rejection\tPASS").unwrap();
    writeln!(out, "gate\tprecondition-rejection\tPASS").unwrap();
    writeln!(out, "gate\tqualified-error-text-preserved\tPASS").unwrap();
    writeln!(out, "gate\tnonzero-underflow-enclosure\tPASS").unwrap();
}
