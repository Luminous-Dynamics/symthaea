//! Prime density estimation functions.
//! Uses the logarithmic integral Li(x) to approximate the prime-counting function π(x).

/// Approximates the number of primes less than or equal to x using the Logarithmic Integral Li(x).
/// Note: This implementation uses a simple numerical integration method (Trapezoidal Rule).
pub fn logarithmic_integral(x: f64) -> f64 {
    if x < 2.0 {
        return 0.0;
    }

    // Li(x) = ∫[2, x] dt / ln(t)
    // We approximate this using a simple trapezoidal rule.
    let steps = 10000;
    let h = (x - 2.0) / steps as f64;
    let mut sum = 0.0;

    for i in 0..steps {
        let t0 = 2.0 + i as f64 * h;
        let t1 = 2.0 + (i + 1) as f64 * h;

        let f0 = 1.0 / t0.ln();
        let f1 = 1.0 / t1.ln();

        sum += (f0 + f1) * h / 2.0;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_li_approximation() {
        // Known rough values for π(x)
        // π(10) = 4
        // Li(10) ≈ 6.16
        let val = logarithmic_integral(10.0);
        assert!(val > 5.0 && val < 7.0);
    }
}
