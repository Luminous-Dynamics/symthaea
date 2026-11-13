use ieee754_enclosure_qualifier::{BinaryOp, EnclosureError, enclose_binary};

fn push_pair(pairs: &mut Vec<(f64, f64)>, a: f64, b: f64) {
    if a.is_finite() && b.is_finite() {
        pairs.push((a, b));
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn emit_case(case_id: u64, op: BinaryOp, a: f64, b: f64) {
    let a_bits = format!("{:016x}", a.to_bits());
    let b_bits = format!("{:016x}", b.to_bits());
    match enclose_binary(op, a, b) {
        Ok(enclosure) => println!(
            "{case_id}|{}|{a_bits}|{b_bits}|{:016x}|{:016x}|OK",
            op.as_str(),
            enclosure.lower.to_bits(),
            enclosure.upper.to_bits()
        ),
        Err(EnclosureError::DivisionByZero) => println!(
            "{case_id}|{}|{a_bits}|{b_bits}|||REJECTED_ZERO_DIVISOR",
            op.as_str()
        ),
        Err(error) => println!(
            "{case_id}|{}|{a_bits}|{b_bits}|||REJECTED_{error:?}",
            op.as_str()
        ),
    }
}

fn main() {
    let special = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        0.3,
        0.03,
        0.7,
        1.0 / 3.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(2),
        f64::from_bits((1_u64 << 52) - 1),
        -f64::MIN_POSITIVE,
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        2.0_f64.powi(-500),
        2.0_f64.powi(500),
        f64::from_bits(1.0_f64.to_bits() - 1),
        f64::from_bits(1.0_f64.to_bits() + 1),
    ];

    let mut pairs = Vec::new();
    for &a in &special {
        for &b in &special {
            push_pair(&mut pairs, a, b);
        }
    }

    // Boundary-oriented pairs: adjacent representable values and values that
    // exercise positive/negative cancellation and overflow-adjacent products.
    for &x in &special {
        push_pair(&mut pairs, x, f64::from_bits(x.to_bits().wrapping_add(1)));
        push_pair(&mut pairs, x, -x);
        push_pair(&mut pairs, x, 0.5);
        push_pair(&mut pairs, x, 2.0);
    }

    // Deterministic pseudo-random finite bit-pattern pairs. Repeated values
    // and signs are retained; NaN/infinity patterns are filtered.
    let mut state = 0x4d595df4d0f33173_u64;
    for _ in 0..8192 {
        let mut a_bits = xorshift64(&mut state);
        let mut b_bits = xorshift64(&mut state);
        if ((a_bits >> 52) & 0x7ff) == 0x7ff {
            a_bits &= !(0x7ff_u64 << 52);
        }
        if ((b_bits >> 52) & 0x7ff) == 0x7ff {
            b_bits &= !(0x7ff_u64 << 52);
        }
        let a = f64::from_bits(a_bits);
        let b = f64::from_bits(b_bits);
        push_pair(&mut pairs, a, b);
    }

    let ops = [
        BinaryOp::Add,
        BinaryOp::Subtract,
        BinaryOp::Multiply,
        BinaryOp::Divide,
    ];
    let mut case_id = 0_u64;
    for (a, b) in pairs {
        for op in ops {
            emit_case(case_id, op, a, b);
            case_id += 1;
        }
    }
}
