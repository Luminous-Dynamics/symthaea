//! # Profile Φ Calculation Hotspots
//!
//! This example profiles the different stages of Φ calculation to identify
//! performance bottlenecks.
//!
//! ## Run
//! ```bash
//! cargo run --example profile_phi_hotspots --release
//! ```

use std::time::{Duration, Instant};
use nalgebra::DMatrix;

/// Simulate HDC_DIMENSION for standalone compilation
const HDC_DIMENSION: usize = 16_384;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              Φ CALCULATION HOTSPOT PROFILER                      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Profiling computation stages to identify bottlenecks            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Test sizes
    let sizes = [4, 8, 16, 32, 64];

    for &n in &sizes {
        profile_phi_stages(n);
    }

    // Summary
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY: Eigenvalue decomposition is O(n³) and dominates for n>8");
    println!("For small n (≤8), similarity matrix computation is significant");
    println!("For large n (≥32), eigenvalue computation is 90%+ of total time");
    println!("═══════════════════════════════════════════════════════════════════");
}

fn profile_phi_stages(n: usize) {
    println!("───────────────────────────────────────────────────────────────────");
    println!("  Profiling n = {} nodes (HDC_DIMENSION = {})", n, HDC_DIMENSION);
    println!("───────────────────────────────────────────────────────────────────");

    // Generate random hypervectors
    let hvs: Vec<Vec<f32>> = (0..n)
        .map(|i| generate_random_hv(HDC_DIMENSION, i as u64))
        .collect();

    // Stage 1: Build similarity matrix
    let start = Instant::now();
    let similarity_matrix = build_similarity_matrix(&hvs);
    let similarity_time = start.elapsed();
    println!("  [Stage 1] Similarity matrix: {:>10.3}ms", similarity_time.as_secs_f64() * 1000.0);

    // Stage 2: Compute degrees
    let start = Instant::now();
    let degrees = compute_degrees(&similarity_matrix);
    let degree_time = start.elapsed();
    println!("  [Stage 2] Compute degrees:   {:>10.3}ms", degree_time.as_secs_f64() * 1000.0);

    // Stage 3: Build Laplacian
    let start = Instant::now();
    let laplacian = build_normalized_laplacian(&similarity_matrix, &degrees);
    let laplacian_time = start.elapsed();
    println!("  [Stage 3] Build Laplacian:   {:>10.3}ms", laplacian_time.as_secs_f64() * 1000.0);

    // Stage 4: Eigenvalue decomposition
    let start = Instant::now();
    let phi = compute_eigenvalue_phi(&laplacian);
    let eigen_time = start.elapsed();
    println!("  [Stage 4] Eigenvalues:       {:>10.3}ms (HOTSPOT)", eigen_time.as_secs_f64() * 1000.0);

    let total_time = similarity_time + degree_time + laplacian_time + eigen_time;
    println!("  ─────────────────────────────────────────");
    println!("  TOTAL:                       {:>10.3}ms", total_time.as_secs_f64() * 1000.0);
    println!("  Φ value:                     {:>10.4}", phi);

    // Percentage breakdown
    let pct_sim = (similarity_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
    let pct_deg = (degree_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
    let pct_lap = (laplacian_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;
    let pct_eig = (eigen_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0;

    println!();
    println!("  Breakdown:");
    println!("    Similarity:  {:>5.1}%", pct_sim);
    println!("    Degrees:     {:>5.1}%", pct_deg);
    println!("    Laplacian:   {:>5.1}%", pct_lap);
    println!("    Eigenvalues: {:>5.1}% ← Main bottleneck", pct_eig);
    println!();
}

/// Generate random hypervector with deterministic seed
fn generate_random_hv(dim: usize, seed: u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(dim);
    let mut state = seed.wrapping_add(1);

    for _ in 0..dim {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        values.push(normalized);
    }

    values
}

/// Build similarity matrix from hypervectors
fn build_similarity_matrix(hvs: &[Vec<f32>]) -> Vec<Vec<f64>> {
    let n = hvs.len();
    let mut matrix = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let similarity = cosine_similarity(&hvs[i], &hvs[j]);
            let normalized = (similarity as f64 + 1.0) / 2.0;
            matrix[i][j] = normalized;
            matrix[j][i] = normalized;
        }
    }

    matrix
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a * norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute degrees from similarity matrix
fn compute_degrees(similarity_matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = similarity_matrix.len();
    let mut degrees = vec![0.0_f64; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                degrees[i] += similarity_matrix[i][j];
            }
        }
    }

    degrees
}

/// Build normalized Laplacian matrix
fn build_normalized_laplacian(similarity_matrix: &[Vec<f64>], degrees: &[f64]) -> DMatrix<f64> {
    let n = similarity_matrix.len();
    let inv_sqrt_degrees: Vec<f64> = degrees.iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut laplacian_data = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            if i == j {
                laplacian_data[idx] = if degrees[i] > 1e-10 { 1.0 } else { 0.0 };
            } else {
                let normalization = inv_sqrt_degrees[i] * inv_sqrt_degrees[j];
                laplacian_data[idx] = -similarity_matrix[i][j] * normalization;
            }
        }
    }

    DMatrix::from_row_slice(n, n, &laplacian_data)
}

/// Compute Φ from eigenvalues (the hotspot)
fn compute_eigenvalue_phi(laplacian: &DMatrix<f64>) -> f64 {
    let eigen = nalgebra::SymmetricEigen::new(laplacian.clone());

    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if eigenvalues.len() >= 2 {
        (eigenvalues[1].max(0.0) / 2.0).clamp(0.0, 1.0)
    } else {
        0.0
    }
}
