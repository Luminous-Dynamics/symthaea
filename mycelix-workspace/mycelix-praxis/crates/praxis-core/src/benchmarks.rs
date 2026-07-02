// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Performance Benchmarks
//!
//! Timing tests for hot-path computations. These don't use criterion
//! (to avoid the dependency) but provide actionable performance data.

#[cfg(test)]
mod tests {
    use std::time::Instant;

    // ---- BKT Replay Performance ----

    #[test]
    fn bench_bkt_validation_100_attempts() {
        use crate::validation::validate_bkt_integrity;

        let start = Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = validate_bkt_integrity(0.7, 100, 70, 0.1);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iterations;
        eprintln!(
            "BKT validation (100 attempts): {:.1}µs/op ({} iterations in {:?})",
            per_op.as_nanos() as f64 / 1000.0,
            iterations,
            elapsed
        );
        // Should complete in < 100µs per validation
        assert!(per_op.as_micros() < 100, "BKT validation too slow: {:?}", per_op);
    }

    #[test]
    fn bench_bkt_validation_1000_attempts() {
        use crate::validation::validate_bkt_integrity;

        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = validate_bkt_integrity(0.85, 1000, 800, 0.1);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iterations;
        eprintln!(
            "BKT validation (1000 attempts): {:.1}µs/op ({} iterations in {:?})",
            per_op.as_nanos() as f64 / 1000.0,
            iterations,
            elapsed
        );
        // 1000 attempts should still complete in < 1ms
        assert!(per_op.as_micros() < 1000, "BKT 1000 validation too slow: {:?}", per_op);
    }

    // ---- TEND Computation Performance ----

    #[test]
    fn bench_tend_computation() {
        use crate::validation::compute_tend_pure;

        let start = Instant::now();
        let iterations = 100_000;
        let mut total = 0.0f32;
        for i in 0..iterations {
            let q = (i % 1000) as u16;
            let d = ((i % 10) * 900 + 900) as u32;
            let r = (i % 1000) as u16;
            total += compute_tend_pure(q, d, r, Some(500));
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iterations;
        eprintln!(
            "TEND computation: {:.0}ns/op ({} iterations in {:?}, total={})",
            per_op.as_nanos(),
            iterations,
            elapsed,
            total
        );
        // TEND computation is trivial arithmetic — should be < 1µs
        assert!(per_op.as_nanos() < 1000, "TEND computation too slow: {:?}", per_op);
    }

    // ---- Ebbinghaus Vitality Performance ----

    #[test]
    fn bench_ebbinghaus_vitality() {
        // Test the pure math that runs in the professional-graph coordinator.
        // Inline the same computation here for benchmarking.
        fn ebbinghaus_stability(mastery_permille: u16, reviews: u32) -> f64 {
            let mastery = mastery_permille as f64 / 1000.0;
            let base = 1440.0;
            let mastery_factor = 0.5 + mastery * 2.0;
            let review_factor = 1.5_f64.powi(reviews.min(10) as i32);
            base * mastery_factor * review_factor
        }

        fn predict_vitality(stability: f64, elapsed: f64) -> u16 {
            if stability <= 0.0 { return 1000; }
            let retention = (-elapsed / stability).exp();
            (retention * 1000.0).round().clamp(0.0, 1000.0) as u16
        }

        let start = Instant::now();
        let iterations = 100_000;
        let mut total = 0u64;
        for i in 0..iterations {
            let mastery = (i % 1000) as u16;
            let reviews = (i % 15) as u32;
            let elapsed = (i % 10000) as f64;
            let s = ebbinghaus_stability(mastery, reviews);
            total += predict_vitality(s, elapsed) as u64;
        }
        let elapsed_time = start.elapsed();
        let per_op = elapsed_time / iterations;
        eprintln!(
            "Ebbinghaus vitality: {:.0}ns/op ({} iterations in {:?}, checksum={})",
            per_op.as_nanos(),
            iterations,
            elapsed_time,
            total
        );
        // Should be < 500ns (just exp() + basic arithmetic)
        assert!(per_op.as_nanos() < 500, "Ebbinghaus too slow: {:?}", per_op);
    }

    // ---- Cosine Similarity Performance (SemDeDup hot path) ----

    #[test]
    fn bench_cosine_similarity_256d() {
        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
            dot / (norm_a * norm_b)
        }

        // 256D vectors (embeddinggemma:300m output dimension)
        let dim = 256;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01 + 0.5).sin()).collect();

        let start = Instant::now();
        let iterations = 100_000;
        let mut total = 0.0f32;
        for _ in 0..iterations {
            total += cosine_similarity(&a, &b);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iterations;
        eprintln!(
            "Cosine similarity (256D): {:.0}ns/op ({} iterations in {:?})",
            per_op.as_nanos(),
            iterations,
            elapsed
        );
        // 256D cosine should be < 50µs (debug mode is slower)
        assert!(per_op.as_micros() < 50, "Cosine 256D too slow: {:?}", per_op);
        assert!(total != 0.0); // Prevent dead code elimination
    }

    #[test]
    fn bench_cosine_pairwise_100_items() {
        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
            dot / (norm_a * norm_b)
        }

        let dim = 256;
        let n = 100;
        let embeddings: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect())
            .collect();

        let start = Instant::now();
        let mut comparisons = 0u64;
        let mut max_sim = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
                if sim > max_sim { max_sim = sim; }
                comparisons += 1;
            }
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Pairwise cosine (100 items, 256D): {:?} for {} comparisons ({:.1}µs/comparison, max_sim={:.4})",
            elapsed,
            comparisons,
            elapsed.as_nanos() as f64 / comparisons as f64 / 1000.0,
            max_sim
        );
        assert_eq!(comparisons, 4950); // n*(n-1)/2
        // 100 items pairwise should complete in < 1s (debug mode)
        assert!(elapsed.as_secs() < 1, "Pairwise 100 items too slow: {:?}", elapsed);
    }

    #[test]
    fn bench_cosine_pairwise_500_items() {
        fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
            dot / (norm_a * norm_b)
        }

        let dim = 256;
        let n = 500;
        let embeddings: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect())
            .collect();

        let start = Instant::now();
        let mut comparisons = 0u64;
        for i in 0..n {
            for j in (i + 1)..n {
                let _ = cosine_similarity(&embeddings[i], &embeddings[j]);
                comparisons += 1;
            }
        }
        let elapsed = start.elapsed();
        eprintln!(
            "Pairwise cosine (500 items, 256D): {:?} for {} comparisons",
            elapsed, comparisons
        );
        assert_eq!(comparisons, 124750);
        // 500 items pairwise should complete in < 15s (debug mode; ~1s in release)
        assert!(elapsed.as_secs() < 15, "Pairwise 500 items too slow: {:?}", elapsed);
    }
}
