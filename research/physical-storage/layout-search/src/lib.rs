//! Parameterized, deterministic physical shard-layout search.
//!
//! This crate scores abstract physical tile geometry only. It is not an optical,
//! magnetic, thermal, manufacturing, or retention model. Deterministic fault masks
//! are coverage probes, never estimated physical probabilities.

use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutError(pub String);

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LayoutError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Grid3 {
    pub fn new(x: usize, y: usize, z: usize) -> Result<Self, LayoutError> {
        if x == 0 || y == 0 || z == 0 {
            return Err(LayoutError("grid dimensions must be non-zero".into()));
        }
        x.checked_mul(y)
            .and_then(|xy| xy.checked_mul(z))
            .ok_or_else(|| LayoutError("grid volume overflows usize".into()))?;
        Ok(Self { x, y, z })
    }

    pub fn volume(self) -> usize {
        self.x * self.y * self.z
    }

    pub fn tile(self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x >= self.x || y >= self.y || z >= self.z {
            return None;
        }
        Some(z * self.x * self.y + y * self.x + x)
    }

    pub fn coordinates(self, tile: usize) -> Option<(usize, usize, usize)> {
        if tile >= self.volume() {
            return None;
        }
        let plane = self.x * self.y;
        let z = tile / plane;
        let within = tile % plane;
        Some((within % self.x, within / self.x, z))
    }
}

#[derive(Debug, Clone)]
pub struct LayoutProblem {
    pub grid: Grid3,
    pub reserved_tiles: Vec<usize>,
    pub stripe_count: usize,
    pub shards_per_stripe: usize,
    pub erasure_budget: usize,
}

impl LayoutProblem {
    pub fn new(
        grid: Grid3,
        reserved_tiles: Vec<usize>,
        stripe_count: usize,
        shards_per_stripe: usize,
        erasure_budget: usize,
    ) -> Result<Self, LayoutError> {
        if stripe_count == 0 || shards_per_stripe == 0 {
            return Err(LayoutError("stripe_count and shards_per_stripe must be non-zero".into()));
        }
        if erasure_budget >= shards_per_stripe {
            return Err(LayoutError("erasure budget must be smaller than shards per stripe".into()));
        }

        let mut seen = vec![false; grid.volume()];
        for &tile in &reserved_tiles {
            if tile >= grid.volume() {
                return Err(LayoutError(format!("reserved tile {tile} is outside the grid")));
            }
            if std::mem::replace(&mut seen[tile], true) {
                return Err(LayoutError(format!("reserved tile {tile} is duplicated")));
            }
        }

        let active_count = seen.iter().filter(|reserved| !**reserved).count();
        let required = stripe_count
            .checked_mul(shards_per_stripe)
            .ok_or_else(|| LayoutError("required shard count overflows usize".into()))?;
        if required > active_count {
            return Err(LayoutError(format!(
                "layout requires {required} shard tiles but only {active_count} are available"
            )));
        }

        Ok(Self {
            grid,
            reserved_tiles,
            stripe_count,
            shards_per_stripe,
            erasure_budget,
        })
    }

    pub fn required_shards(&self) -> usize {
        self.stripe_count * self.shards_per_stripe
    }

    pub fn active_tiles(&self) -> Vec<usize> {
        let mut reserved = vec![false; self.grid.volume()];
        for &tile in &self.reserved_tiles {
            reserved[tile] = true;
        }
        (0..self.grid.volume())
            .filter(|&tile| !reserved[tile])
            .collect()
    }

    pub fn validate_mapping(&self, mapping: &Mapping) -> Result<(), LayoutError> {
        if mapping.stripe_count != self.stripe_count
            || mapping.shards_per_stripe != self.shards_per_stripe
        {
            return Err(LayoutError("mapping code geometry does not match problem".into()));
        }
        if mapping.tile_for_slot.len() != self.required_shards() {
            return Err(LayoutError("mapping does not assign exactly one tile per shard".into()));
        }

        let mut seen = vec![false; self.grid.volume()];
        for &tile in &self.reserved_tiles {
            seen[tile] = true;
        }
        for &tile in &mapping.tile_for_slot {
            if tile >= self.grid.volume() {
                return Err(LayoutError(format!("mapped tile {tile} is outside the grid")));
            }
            if std::mem::replace(&mut seen[tile], true) {
                return Err(LayoutError(format!(
                    "mapped tile {tile} is reserved or assigned more than once"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultMask {
    pub id: String,
    pub erased_tiles: Vec<usize>,
}

impl FaultMask {
    pub fn new(id: impl Into<String>, erased_tiles: Vec<usize>) -> Self {
        Self { id: id.into(), erased_tiles }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assignment {
    Contiguous,
    Interleaved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping {
    pub name: String,
    /// Index is stripe * shards_per_stripe + shard.
    pub tile_for_slot: Vec<usize>,
    pub stripe_count: usize,
    pub shards_per_stripe: usize,
}

impl Mapping {
    pub fn from_order(
        problem: &LayoutProblem,
        name: impl Into<String>,
        physical_order: &[usize],
        assignment: Assignment,
    ) -> Result<Self, LayoutError> {
        if physical_order.len() < problem.required_shards() {
            return Err(LayoutError("physical ordering is shorter than the shard count".into()));
        }
        let mut unique = vec![false; problem.grid.volume()];
        for &tile in physical_order {
            if tile >= problem.grid.volume() {
                return Err(LayoutError(format!("physical order contains out-of-range tile {tile}")));
            }
            if unique[tile] {
                return Err(LayoutError(format!("physical order repeats tile {tile}")));
            }
            unique[tile] = true;
        }

        let stripes = problem.stripe_count;
        let shards = problem.shards_per_stripe;
        let mut tile_for_slot = vec![0usize; problem.required_shards()];
        for stripe in 0..stripes {
            for shard in 0..shards {
                let slot = stripe * shards + shard;
                let physical_index = match assignment {
                    Assignment::Contiguous => slot,
                    Assignment::Interleaved => shard * stripes + stripe,
                };
                tile_for_slot[slot] = physical_order[physical_index];
            }
        }

        let mapping = Self {
            name: name.into(),
            tile_for_slot,
            stripe_count: stripes,
            shards_per_stripe: shards,
        };
        problem.validate_mapping(&mapping)?;
        Ok(mapping)
    }

    pub fn fingerprint_fnv1a64(&self) -> String {
        // Stable, non-cryptographic identity for comparison receipts only.
        let mut hash = 0xcbf29ce484222325u64;
        for value in [self.stripe_count as u64, self.shards_per_stripe as u64]
            .into_iter()
            .chain(self.tile_for_slot.iter().map(|tile| *tile as u64))
        {
            for byte in value.to_le_bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        format!("{hash:016x}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Score {
    pub fault_sets: usize,
    pub unrecoverable_fault_sets: usize,
    pub worst_erased_shards_in_one_stripe: usize,
    pub total_erased_shards: usize,
    pub total_excess_erasures: usize,
}

impl Score {
    pub fn better_than(self, other: Self) -> bool {
        (
            self.unrecoverable_fault_sets,
            self.total_excess_erasures,
            self.worst_erased_shards_in_one_stripe,
            self.total_erased_shards,
        ) < (
            other.unrecoverable_fault_sets,
            other.total_excess_erasures,
            other.worst_erased_shards_in_one_stripe,
            other.total_erased_shards,
        )
    }
}

pub fn score_mapping(
    problem: &LayoutProblem,
    mapping: &Mapping,
    fault_masks: &[FaultMask],
) -> Result<Score, LayoutError> {
    problem.validate_mapping(mapping)?;
    let mut unrecoverable = 0usize;
    let mut worst = 0usize;
    let mut total_erased = 0usize;
    let mut total_excess = 0usize;
    let mut mask_seen = vec![false; problem.grid.volume()];

    for fault in fault_masks {
        mask_seen.fill(false);
        for &tile in &fault.erased_tiles {
            if tile >= problem.grid.volume() {
                return Err(LayoutError(format!(
                    "fault mask {} contains out-of-range tile {tile}", fault.id
                )));
            }
            if std::mem::replace(&mut mask_seen[tile], true) {
                return Err(LayoutError(format!(
                    "fault mask {} repeats tile {tile}", fault.id
                )));
            }
        }

        let mut case_worst = 0usize;
        let mut case_total = 0usize;
        for stripe in 0..problem.stripe_count {
            let erased = (0..problem.shards_per_stripe)
                .filter(|&shard| {
                    let tile = mapping.tile_for_slot[stripe * problem.shards_per_stripe + shard];
                    mask_seen[tile]
                })
                .count();
            case_worst = case_worst.max(erased);
            case_total += erased;
            total_excess += erased.saturating_sub(problem.erasure_budget);
        }

        if case_worst > problem.erasure_budget {
            unrecoverable += 1;
        }
        worst = worst.max(case_worst);
        total_erased += case_total;
    }

    Ok(Score {
        fault_sets: fault_masks.len(),
        unrecoverable_fault_sets: unrecoverable,
        worst_erased_shards_in_one_stripe: worst,
        total_erased_shards: total_erased,
        total_excess_erasures: total_excess,
    })
}

pub fn cuboid_fault_masks(
    grid: Grid3,
    shapes_xyz: &[(usize, usize, usize)],
) -> Result<Vec<FaultMask>, LayoutError> {
    let mut output = Vec::new();
    for &(dx, dy, dz) in shapes_xyz {
        if dx == 0 || dy == 0 || dz == 0 || dx > grid.x || dy > grid.y || dz > grid.z {
            return Err(LayoutError(format!("invalid cuboid shape {dx}x{dy}x{dz}")));
        }
        for z in 0..=grid.z - dz {
            for y in 0..=grid.y - dy {
                for x in 0..=grid.x - dx {
                    let mut tiles = Vec::with_capacity(dx * dy * dz);
                    for zz in z..z + dz {
                        for yy in y..y + dy {
                            for xx in x..x + dx {
                                tiles.push(grid.tile(xx, yy, zz).expect("bounded coordinate"));
                            }
                        }
                    }
                    output.push(FaultMask::new(
                        format!("cuboid_{dx}x{dy}x{dz}_at_{x}_{y}_{z}"),
                        tiles,
                    ));
                }
            }
        }
    }
    Ok(output)
}

pub fn row_major_order(problem: &LayoutProblem) -> Vec<usize> {
    problem.active_tiles()
}

pub fn serpentine_order(problem: &LayoutProblem) -> Vec<usize> {
    let mut output = Vec::with_capacity(problem.grid.volume());
    for z in 0..problem.grid.z {
        for y in 0..problem.grid.y {
            if (y + z) % 2 == 0 {
                for x in 0..problem.grid.x {
                    let tile = problem.grid.tile(x, y, z).unwrap();
                    if !problem.reserved_tiles.contains(&tile) {
                        output.push(tile);
                    }
                }
            } else {
                for x in (0..problem.grid.x).rev() {
                    let tile = problem.grid.tile(x, y, z).unwrap();
                    if !problem.reserved_tiles.contains(&tile) {
                        output.push(tile);
                    }
                }
            }
        }
    }
    output
}

fn morton_cmp(grid: Grid3, left: usize, right: usize) -> Ordering {
    let a = grid.coordinates(left).expect("tile is in range");
    let b = grid.coordinates(right).expect("tile is in range");
    for bit in (0..usize::BITS).rev() {
        // Morton interleaving compares z, then y, then x at each bit level.
        for axis in [2usize, 1, 0] {
            let av = (match axis { 0 => a.0, 1 => a.1, _ => a.2 } >> bit) & 1;
            let bv = (match axis { 0 => b.0, 1 => b.1, _ => b.2 } >> bit) & 1;
            if av != bv {
                return av.cmp(&bv);
            }
        }
    }
    left.cmp(&right)
}

pub fn morton_order(problem: &LayoutProblem) -> Vec<usize> {
    let mut tiles = problem.active_tiles();
    tiles.sort_by(|left, right| morton_cmp(problem.grid, *left, *right));
    tiles
}

pub fn seeded_random_order(problem: &LayoutProblem, seed: u64) -> Vec<usize> {
    let mut tiles = problem.active_tiles();
    let mut state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
    for index in (1..tiles.len()).rev() {
        // xorshift64*: deterministic, reproducible permutation generator.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let random = state.wrapping_mul(0x2545f4914f6cdd1d);
        let swap_index = (random % (index as u64 + 1)) as usize;
        tiles.swap(index, swap_index);
    }
    tiles
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub mapping: Mapping,
    pub score: Score,
    pub candidate_evaluations: usize,
    pub termination: String,
}

/// Deterministic best-improvement pair-swap search.
///
/// The evaluation budget includes the initial score. Search only changes tile
/// assignment; code parameters and physical fault masks are frozen.
pub fn greedy_pair_swap_search(
    problem: &LayoutProblem,
    fault_masks: &[FaultMask],
    initial: &Mapping,
    evaluation_budget: usize,
) -> Result<SearchResult, LayoutError> {
    problem.validate_mapping(initial)?;
    if evaluation_budget == 0 {
        return Err(LayoutError("evaluation budget must be at least one".into()));
    }

    let mut current = initial.clone();
    let mut current_score = score_mapping(problem, &current, fault_masks)?;
    let mut evaluations = 1usize;
    let mut termination = "local_optimum".to_string();

    while evaluations < evaluation_budget {
        let mut best_swap: Option<(usize, usize, Score)> = None;
        let mut budget_exhausted = false;

        'pairs: for left in 0..current.tile_for_slot.len() {
            for right in left + 1..current.tile_for_slot.len() {
                if evaluations >= evaluation_budget {
                    budget_exhausted = true;
                    break 'pairs;
                }
                current.tile_for_slot.swap(left, right);
                let trial_score = score_mapping(problem, &current, fault_masks)?;
                current.tile_for_slot.swap(left, right);
                evaluations += 1;

                let improves_current = trial_score.better_than(current_score);
                let improves_best = best_swap
                    .as_ref()
                    .map(|(_, _, best_score)| trial_score.better_than(*best_score))
                    .unwrap_or(true);
                if improves_current && improves_best {
                    best_swap = Some((left, right, trial_score));
                }
            }
        }

        if let Some((left, right, score)) = best_swap {
            current.tile_for_slot.swap(left, right);
            current_score = score;
        } else {
            termination = if budget_exhausted {
                "budget_exhausted_without_improvement".to_string()
            } else {
                "local_optimum".to_string()
            };
            break;
        }

        if budget_exhausted {
            termination = "budget_exhausted".to_string();
            break;
        }
    }

    if evaluations >= evaluation_budget && termination == "local_optimum" {
        termination = "budget_exhausted".to_string();
    }

    current.name = format!("greedy-pair-swap-from-{}", initial.name);
    Ok(SearchResult {
        mapping: current,
        score: current_score,
        candidate_evaluations: evaluations,
        termination,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_problem() -> LayoutProblem {
        LayoutProblem::new(
            Grid3::new(4, 4, 3).unwrap(),
            vec![0, 15, 16, 31, 32, 47],
            7,
            6,
            2,
        )
        .unwrap()
    }

    fn mappings(problem: &LayoutProblem) -> Vec<Mapping> {
        let row = row_major_order(problem);
        let snake = serpentine_order(problem);
        let morton = morton_order(problem);
        let random = seeded_random_order(problem, 2026);
        vec![
            Mapping::from_order(problem, "row-major-contiguous", &row, Assignment::Contiguous).unwrap(),
            Mapping::from_order(problem, "row-major-interleaved", &row, Assignment::Interleaved).unwrap(),
            Mapping::from_order(problem, "serpentine-contiguous", &snake, Assignment::Contiguous).unwrap(),
            Mapping::from_order(problem, "serpentine-interleaved", &snake, Assignment::Interleaved).unwrap(),
            Mapping::from_order(problem, "morton-contiguous", &morton, Assignment::Contiguous).unwrap(),
            Mapping::from_order(problem, "morton-interleaved", &morton, Assignment::Interleaved).unwrap(),
            Mapping::from_order(problem, "random-2026-contiguous", &random, Assignment::Contiguous).unwrap(),
            Mapping::from_order(problem, "random-2026-interleaved", &random, Assignment::Interleaved).unwrap(),
        ]
    }

    #[test]
    fn corrected_cuboid_denominators_match_reference_fixture() {
        let problem = fixture_problem();
        let masks = cuboid_fault_masks(
            problem.grid,
            &[(2, 2, 1), (3, 3, 1), (2, 2, 2), (3, 3, 2), (4, 4, 1)],
        ).unwrap();
        let counts = [
            masks.iter().filter(|m| m.id.starts_with("cuboid_2x2x1_")).count(),
            masks.iter().filter(|m| m.id.starts_with("cuboid_3x3x1_")).count(),
            masks.iter().filter(|m| m.id.starts_with("cuboid_2x2x2_")).count(),
            masks.iter().filter(|m| m.id.starts_with("cuboid_3x3x2_")).count(),
            masks.iter().filter(|m| m.id.starts_with("cuboid_4x4x1_")).count(),
        ];
        assert_eq!(counts, [27, 12, 18, 8, 3]);
    }

    #[test]
    fn reference_spatial_stride_and_row_major_scores_match_existing_fixture() {
        let problem = fixture_problem();
        let masks = cuboid_fault_masks(
            problem.grid,
            &[(2, 2, 1), (3, 3, 1), (2, 2, 2), (3, 3, 2), (4, 4, 1)],
        ).unwrap();
        let order = row_major_order(&problem);
        let row = Mapping::from_order(&problem, "row-major", &order, Assignment::Contiguous).unwrap();
        let spatial = Mapping::from_order(&problem, "spatial-stride", &order, Assignment::Interleaved).unwrap();
        let row_score = score_mapping(&problem, &row, &masks).unwrap();
        let spatial_score = score_mapping(&problem, &spatial, &masks).unwrap();
        assert_eq!(row_score.unrecoverable_fault_sets, 50);
        assert_eq!(spatial_score.unrecoverable_fault_sets, 8);
    }

    #[test]
    fn mapping_generators_are_bijective_and_deterministic() {
        let problem = fixture_problem();
        let all = mappings(&problem);
        for mapping in &all {
            problem.validate_mapping(mapping).unwrap();
            let unique: std::collections::HashSet<usize> =
                mapping.tile_for_slot.iter().copied().collect();
            assert_eq!(unique.len(), problem.required_shards());
        }
        assert_eq!(seeded_random_order(&problem, 2026), seeded_random_order(&problem, 2026));
    }

    #[test]
    fn greedy_search_never_worsens_the_starting_mapping() {
        let problem = fixture_problem();
        let masks = cuboid_fault_masks(problem.grid, &[(2, 2, 1), (3, 3, 1), (2, 2, 2)]).unwrap();
        let start = Mapping::from_order(
            &problem, "row-major", &row_major_order(&problem), Assignment::Contiguous
        ).unwrap();
        let before = score_mapping(&problem, &start, &masks).unwrap();
        let result = greedy_pair_swap_search(&problem, &masks, &start, 500).unwrap();
        assert!(!before.better_than(result.score));
        assert!(result.candidate_evaluations <= 500);
        problem.validate_mapping(&result.mapping).unwrap();
    }

    #[test]
    fn invalid_or_aliased_layouts_fail_closed() {
        let problem = fixture_problem();
        let mut mapping = Mapping::from_order(
            &problem, "valid", &row_major_order(&problem), Assignment::Contiguous
        ).unwrap();
        mapping.tile_for_slot[1] = mapping.tile_for_slot[0];
        assert!(problem.validate_mapping(&mapping).is_err());
    }
}
