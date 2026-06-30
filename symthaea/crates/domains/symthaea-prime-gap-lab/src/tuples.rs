pub struct PrimeTuple {
    pub elements: Vec<u64>,
}

impl PrimeTuple {
    pub fn new(elements: Vec<u64>) -> Self {
        Self { elements }
    }

    pub fn is_admissible(&self) -> bool {
        let k = self.elements.len() as u64;
        let primes = crate::data::generate_primes(k + 1);

        for &p in &primes {
            let mut covered = std::collections::HashSet::new();
            for &x in &self.elements {
                covered.insert(x % p);
            }
            if covered.len() as u64 == p {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admissibility() {
        assert!(PrimeTuple::new(vec![0, 2]).is_admissible());
        assert!(!PrimeTuple::new(vec![0, 1]).is_admissible());
        assert!(PrimeTuple::new(vec![0, 2, 6]).is_admissible());
        assert!(!PrimeTuple::new(vec![0, 2, 4]).is_admissible());
    }
}
