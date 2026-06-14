pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=((n as f64).sqrt() as u64) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

pub fn generate_primes(limit: u64) -> Vec<u64> {
    (2..limit).filter(|&n| is_prime(n)).collect()
}

pub fn get_prime_gaps(limit: u64) -> Vec<u64> {
    let primes = generate_primes(limit);
    primes.windows(2).map(|w| w[1] - w[0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(9));
        assert!(!is_prime(21));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(97));
    }

    #[test]
    fn test_generate_primes() {
        assert_eq!(generate_primes(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
    }

    #[test]
    fn test_get_prime_gaps() {
        assert_eq!(get_prime_gaps(20), vec![1, 2, 2, 4, 2, 4, 2]);
    }
}
