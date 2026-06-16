pub fn search_for_counterexample(k: usize, limit: u64) -> Option<Vec<u64>> {
    // Search for an admissible tuple that violates Hardy-Littlewood heuristics for small ranges
    let tuples = crate::tuples::enumerate_admissible_tuples(k, limit);
    for t in tuples {
        // Simple counterexample check: if singular series is unexpectedly 0
        // where it should be positive (heuristic failure)
        let ss = crate::hardy_littlewood::calculate_singular_series(&t.elements, 100);
        if ss < 0.0 {
            return Some(t.elements);
        }
    }
    None
}
