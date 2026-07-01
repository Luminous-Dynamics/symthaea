// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active learning for nuclear measurement prioritization.
//!
//! Uses RF uncertainty + information-theoretic acquisition scoring to
//! rank unmeasured nuclei in the FRIB-accessible region for experimental
//! measurement campaigns.
//!
//! Key idea: prioritize nuclei where the model is *uncertain* AND where
//! a measurement would maximally constrain predictions for *neighboring*
//! unmeasured nuclei (information gain).

#[cfg(test)]
mod tests {
    use crate::ame2020::ame2020_reference_nuclei;
    use crate::ml_mass::MlMassPredictor;
    use std::collections::HashSet;

    /// Magic numbers for shell proximity.
    const MAGIC_Z: &[u16] = &[2, 8, 20, 28, 50, 82, 114, 126];
    const MAGIC_N: &[u16] = &[2, 8, 20, 28, 50, 82, 126, 184];

    /// Count unmeasured neighbors within delta_z <= 2, delta_n <= 2.
    fn count_unmeasured_neighbors(z: u16, n: u16, measured: &HashSet<(u16, u16)>) -> usize {
        let mut count = 0;
        let z_lo = z.saturating_sub(2);
        let n_lo = n.saturating_sub(2);
        for zz in z_lo..=(z + 2) {
            for nn in n_lo..=(n + 2) {
                if zz == z && nn == n {
                    continue;
                }
                if !measured.contains(&(zz, nn)) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Shell proximity bonus: 1 / (1 + min_distance_to_any_magic_number).
    /// Considers both Z and N magic numbers.
    fn shell_proximity(z: u16, n: u16) -> f64 {
        let min_z_dist = MAGIC_Z
            .iter()
            .map(|&m| (z as i32 - m as i32).unsigned_abs())
            .min()
            .unwrap_or(999);
        let min_n_dist = MAGIC_N
            .iter()
            .map(|&m| (n as i32 - m as i32).unsigned_abs())
            .min()
            .unwrap_or(999);
        let min_dist = min_z_dist.min(min_n_dist);
        1.0 / (1.0 + min_dist as f64)
    }

    /// Select a diverse batch using greedy farthest-point selection.
    /// min_distance: minimum Manhattan distance |dZ| + |dN| between any two selected nuclei.
    fn select_diverse_batch(
        candidates: &[(u16, u16, f64, f64, f64, usize)],
        batch_size: usize,
        min_distance: u16,
    ) -> Vec<(u16, u16, f64, f64, f64, usize)> {
        let mut selected: Vec<(u16, u16, f64, f64, f64, usize)> = Vec::new();

        for &(z, n, be, sigma, score, neigh) in candidates {
            if selected.len() >= batch_size {
                break;
            }
            // Check diversity constraint against all already-selected
            let too_close = selected.iter().any(|&(sz, sn, _, _, _, _)| {
                let dz = (z as i32 - sz as i32).unsigned_abs() as u16;
                let dn = (n as i32 - sn as i32).unsigned_abs() as u16;
                dz + dn < min_distance
            });
            if !too_close {
                selected.push((z, n, be, sigma, score, neigh));
            }
        }

        selected
    }

    #[test]
    fn rank_frib_priorities() {
        let pred = MlMassPredictor::new();
        let measured: HashSet<(u16, u16)> = ame2020_reference_nuclei()
            .iter()
            .filter(|n| n.is_measured)
            .map(|n| (n.z, n.n))
            .collect();

        eprintln!("\nMeasured nuclei in dataset: {}", measured.len());

        // Scan unmeasured nuclei in FRIB-accessible range
        // Z=20-60, N=30-100 (neutron-rich, reachable by FRIB)
        let mut candidates = Vec::new();
        let mut scanned = 0u32;
        let mut skipped_bound = 0u32;
        let mut skipped_measured = 0u32;

        for z in 20..=60u16 {
            for n in 30..=100u16 {
                scanned += 1;
                if measured.contains(&(z, n)) {
                    skipped_measured += 1;
                    continue;
                }
                let p = pred.predict(z, n);
                if p.binding_energy <= 0.0 || p.ba < 5.0 {
                    skipped_bound += 1;
                    continue;
                }

                let uncertainty = p.uncertainty;
                let neighbors = count_unmeasured_neighbors(z, n, &measured);
                let shell_bonus = shell_proximity(z, n);
                let score =
                    uncertainty * (1.0 + (neighbors as f64 + 1.0).ln()) * (1.0 + 0.2 * shell_bonus);

                candidates.push((z, n, p.binding_energy, uncertainty, score, neighbors));
            }
        }

        // Sort by acquisition score descending
        candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!("\n=== FRIB ACTIVE LEARNING SCAN ===");
        eprintln!("Region: Z=20-60, N=30-100 (neutron-rich FRIB range)");
        eprintln!("Scanned:  {}", scanned);
        eprintln!("Measured: {} (skipped)", skipped_measured);
        eprintln!("Unbound:  {} (skipped, BE<=0 or B/A<5)", skipped_bound);
        eprintln!("Candidates: {}", candidates.len());

        // Statistics on acquisition scores
        if !candidates.is_empty() {
            let scores: Vec<f64> = candidates.iter().map(|c| c.4).collect();
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "Score range: {:.4} - {:.4} (mean: {:.4})",
                min_score, max_score, mean_score
            );

            let uncertainties: Vec<f64> = candidates.iter().map(|c| c.3).collect();
            let mean_unc = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
            eprintln!("Mean uncertainty: {:.4} MeV", mean_unc);
        }

        // Top 30 raw (before diversity filter)
        eprintln!("\n--- TOP 30 BY ACQUISITION SCORE (raw) ---");
        eprintln!(
            "{:>4} {:>4} {:>4} {:>10} {:>8} {:>8} {:>6} {:>8}",
            "Rank", "Z", "N", "BE(MeV)", "s(MeV)", "Score", "Neigh", "Shell"
        );
        eprintln!("{}", "-".repeat(66));
        for (rank, &(z, n, be, sigma, score, neigh)) in candidates.iter().take(30).enumerate() {
            eprintln!(
                "{:>4} {:>4} {:>4} {:>10.2} {:>8.3} {:>8.3} {:>6} {:>8.3}",
                rank + 1,
                z,
                n,
                be,
                sigma,
                score,
                neigh,
                shell_proximity(z, n)
            );
        }

        // Batch selection with diversity (min dZ+dN >= 5)
        let batch = select_diverse_batch(&candidates, 20, 5);

        eprintln!("\n=== TOP 20 FRIB MEASUREMENT PRIORITIES (diverse batch, min dist=5) ===");
        eprintln!(
            "{:>4} {:>4} {:>4} {:>10} {:>8} {:>8} {:>6}",
            "Rank", "Z", "N", "BE(MeV)", "s(MeV)", "Score", "Neigh"
        );
        eprintln!("{}", "-".repeat(56));
        for (rank, &(z, n, be, sigma, score, neigh)) in batch.iter().enumerate() {
            eprintln!(
                "{:>4} {:>4} {:>4} {:>10.2} {:>8.3} {:>8.3} {:>6}",
                rank + 1,
                z,
                n,
                be,
                sigma,
                score,
                neigh
            );
        }

        // Group by element
        eprintln!("\n--- HIGHEST-PRIORITY ISOTOPE PER ELEMENT ---");
        eprintln!(
            "{:>4} {:>4} {:>10} {:>8} {:>8}",
            "Z", "N", "BE(MeV)", "s(MeV)", "Score"
        );
        eprintln!("{}", "-".repeat(40));
        let mut seen_z = HashSet::new();
        for &(z, n, be, sigma, score, _) in &candidates {
            if seen_z.insert(z) {
                eprintln!(
                    "{:>4} {:>4} {:>10.2} {:>8.3} {:>8.3}",
                    z, n, be, sigma, score
                );
            }
        }

        // Region analysis: which Z range has highest mean uncertainty?
        eprintln!("\n--- MEAN UNCERTAINTY BY Z (in candidate set) ---");
        for z_bin_start in (20..=55).step_by(5) {
            let z_bin_end = z_bin_start + 4;
            let bin_uncs: Vec<f64> = candidates
                .iter()
                .filter(|c| c.0 >= z_bin_start && c.0 <= z_bin_end)
                .map(|c| c.3)
                .collect();
            if !bin_uncs.is_empty() {
                let mean = bin_uncs.iter().sum::<f64>() / bin_uncs.len() as f64;
                let max = bin_uncs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "  Z={}-{}: mean s = {:.3} MeV, max s = {:.3} MeV ({} candidates)",
                    z_bin_start,
                    z_bin_end,
                    mean,
                    max,
                    bin_uncs.len()
                );
            }
        }

        // Assertions
        assert!(
            !candidates.is_empty(),
            "Should find unmeasured candidates in FRIB range"
        );
        assert!(
            batch.len() >= 5,
            "Should select at least 5 diverse candidates, got {}",
            batch.len()
        );
        // Verify diversity constraint
        for i in 0..batch.len() {
            for j in (i + 1)..batch.len() {
                let dz = (batch[i].0 as i32 - batch[j].0 as i32).unsigned_abs() as u16;
                let dn = (batch[i].1 as i32 - batch[j].1 as i32).unsigned_abs() as u16;
                assert!(
                    dz + dn >= 5,
                    "Diversity violated: ({},{}) and ({},{}) have dist {}",
                    batch[i].0,
                    batch[i].1,
                    batch[j].0,
                    batch[j].1,
                    dz + dn
                );
            }
        }
    }
}
