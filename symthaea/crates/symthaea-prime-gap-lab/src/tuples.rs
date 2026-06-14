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

pub fn enumerate_tuples(k: usize, max_width: u64) -> Vec<PrimeTuple> {
    let mut results = Vec::new();
    fn backtrack(current: &mut Vec<u64>, k: usize, max_width: u64, results: &mut Vec<PrimeTuple>) {
        if current.len() == k {
            results.push(PrimeTuple::new(current.clone()));
            return;
        }
        let start = *current.last().unwrap_or(&0);
        for next in (start + 1)..=max_width {
            current.push(next);
            backtrack(current, k, max_width, results);
            current.pop();
        }
    }
    backtrack(&mut vec![0], k, max_width, &mut results);
    results
}

pub fn enumerate_admissible_tuples(k: usize, max_width: u64) -> Vec<PrimeTuple> {
    enumerate_tuples(k, max_width)
        .into_iter()
        .filter(|t| t.is_admissible())
        .collect()
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

    #[test]
    fn test_enumeration() {
        let admissible = enumerate_admissible_tuples(2, 6);
        assert!(admissible.iter().any(|t| t.elements == vec![0, 2]));
        assert!(admissible.iter().any(|t| t.elements == vec![0, 6]));
    }
}
