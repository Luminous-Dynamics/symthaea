/// Calculates the singular series constant C(H) for a k-tuple H = {h_1, ..., h_k}.
pub fn calculate_singular_series(tuple: &[u64], prime_limit: u64) -> f64 {
    let k = tuple.len() as f64;
    let mut singular_series = 1.0;

    let primes = crate::data::generate_primes(prime_limit);

    for p in primes {
        let mut residue_classes = std::collections::HashSet::new();
        for &h in tuple {
            residue_classes.insert(h % p);
        }
        let w_p = residue_classes.len() as f64;

        if w_p == p as f64 {
            return 0.0;
        }

        let p_f = p as f64;
        let factor = (1.0 - w_p / p_f) / (1.0 - 1.0 / p_f).powf(k);
        singular_series *= factor;
    }

    singular_series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singular_series_inadmissible() {
        assert_eq!(calculate_singular_series(&[0, 1], 100), 0.0);
    }
}
