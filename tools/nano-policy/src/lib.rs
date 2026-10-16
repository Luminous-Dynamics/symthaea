use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub const MAX_INPUT_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_CANDIDATES: usize = 100_000;
pub const MAX_FEATURES: usize = 512;
pub const MAX_FEATURE_CELLS: usize = 20_000_000;
pub const MAX_FEATURE_ABS: f64 = 1.0e12;
pub const MAX_DISTANCE_OPERATIONS: usize = 100_000_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub features: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateTable {
    pub feature_names: Vec<String>,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    Random,
    MaxMin,
}

impl Policy {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "random" => Ok(Self::Random),
            "maxmin" => Ok(Self::MaxMin),
            _ => Err(format!(
                "unknown policy {value:?}; expected random or maxmin"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::MaxMin => "maxmin",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    pub candidate_index: usize,
    pub rank: usize,
    pub distance_to_prior_selected: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunReport {
    pub policy: Policy,
    pub seed: u64,
    pub budget: usize,
    pub duplicate_of: Vec<Option<usize>>,
    pub selected: Vec<Selection>,
}

impl RunReport {
    pub fn eligible_count(&self, candidate_count: usize) -> usize {
        candidate_count - self.duplicate_of.iter().filter(|item| item.is_some()).count()
    }
}

fn valid_feature_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

pub fn parse_tsv(input: &str) -> Result<CandidateTable, String> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(format!("input exceeds the {MAX_INPUT_BYTES}-byte limit"));
    }

    let mut lines = input
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty());

    let (header_line_zero, header_text) = lines
        .next()
        .ok_or_else(|| "input is empty".to_string())?;
    let header: Vec<&str> = header_text.split('\t').map(str::trim).collect();

    if header.len() < 2 || header[0] != "candidate_id" {
        return Err(format!(
            "line {} must start with candidate_id and include at least one feature column",
            header_line_zero + 1
        ));
    }
    let feature_names: Vec<String> =
        header[1..].iter().map(|name| (*name).to_string()).collect();
    if feature_names.len() > MAX_FEATURES {
        return Err(format!(
            "feature count {} exceeds the limit of {MAX_FEATURES}",
            feature_names.len()
        ));
    }

    let mut seen_feature_names = HashSet::with_capacity(feature_names.len());
    for name in &feature_names {
        if !valid_feature_name(name) {
            return Err(format!(
                "invalid feature name {name:?}; use ASCII letters, digits, underscore, dot or hyphen"
            ));
        }
        if !seen_feature_names.insert(name.as_str()) {
            return Err(format!("duplicate feature name {name:?}"));
        }
    }

    let feature_count = feature_names.len();
    let mut candidates = Vec::new();
    let mut seen_ids = HashSet::new();

    for (line_zero, row_text) in lines {
        let fields: Vec<&str> = row_text.split('\t').collect();
        if fields.len() != header.len() {
            return Err(format!(
                "line {} has {} columns; expected {}",
                line_zero + 1,
                fields.len(),
                header.len()
            ));
        }
        if candidates.len() >= MAX_CANDIDATES {
            return Err(format!(
                "candidate count exceeds the limit of {MAX_CANDIDATES}"
            ));
        }
        let next_count = candidates.len() + 1;
        let cells = next_count
            .checked_mul(feature_count)
            .ok_or_else(|| "feature-cell count overflow".to_string())?;
        if cells > MAX_FEATURE_CELLS {
            return Err(format!(
                "candidate-feature cell count exceeds the limit of {MAX_FEATURE_CELLS}"
            ));
        }

        let id = fields[0].trim();
        if id.is_empty()
            || id.starts_with('#')
            || id.chars().any(|character| character.is_control())
        {
            return Err(format!(
                "line {} has an empty, comment-like, or control-character candidate ID",
                line_zero + 1
            ));
        }
        if !seen_ids.insert(id.to_string()) {
            return Err(format!("line {} repeats candidate ID {id:?}", line_zero + 1));
        }

        let mut features = Vec::with_capacity(feature_count);
        for (column, raw_value) in fields[1..].iter().enumerate() {
            let value = raw_value.trim().parse::<f64>().map_err(|_| {
                format!(
                    "line {}, feature {} is not a number",
                    line_zero + 1,
                    feature_names[column]
                )
            })?;
            if !value.is_finite() {
                return Err(format!(
                    "line {}, feature {} must be finite",
                    line_zero + 1,
                    feature_names[column]
                ));
            }
            if value.abs() > MAX_FEATURE_ABS {
                return Err(format!(
                    "line {}, feature {} exceeds the absolute-value limit of {MAX_FEATURE_ABS}",
                    line_zero + 1,
                    feature_names[column]
                ));
            }
            features.push(value);
        }
        candidates.push(Candidate {
            id: id.to_string(),
            features,
        });
    }

    if candidates.is_empty() {
        return Err("input has a header but no candidate rows".to_string());
    }

    Ok(CandidateTable {
        feature_names,
        candidates,
    })
}

fn validate_table(table: &CandidateTable) -> Result<(), String> {
    if table.candidates.is_empty() {
        return Err("candidate table is empty".to_string());
    }
    if table.candidates.len() > MAX_CANDIDATES {
        return Err(format!(
            "candidate count exceeds the limit of {MAX_CANDIDATES}"
        ));
    }

    let feature_count = table.feature_names.len();
    if feature_count == 0 || feature_count > MAX_FEATURES {
        return Err(format!(
            "feature count must be between 1 and {MAX_FEATURES}"
        ));
    }
    let cells = table
        .candidates
        .len()
        .checked_mul(feature_count)
        .ok_or_else(|| "feature-cell count overflow".to_string())?;
    if cells > MAX_FEATURE_CELLS {
        return Err(format!(
            "candidate-feature cell count exceeds the limit of {MAX_FEATURE_CELLS}"
        ));
    }

    let mut feature_names = HashSet::with_capacity(feature_count);
    for name in &table.feature_names {
        if !valid_feature_name(name) {
            return Err(format!("invalid feature name {name:?}"));
        }
        if !feature_names.insert(name.as_str()) {
            return Err(format!("duplicate feature name {name:?}"));
        }
    }

    let mut candidate_ids = HashSet::with_capacity(table.candidates.len());
    for candidate in &table.candidates {
        if candidate.features.len() != feature_count {
            return Err(format!(
                "candidate {:?} has {} features; expected {feature_count}",
                candidate.id,
                candidate.features.len()
            ));
        }
        if candidate.id.is_empty()
            || candidate.id.starts_with('#')
            || candidate.id.chars().any(|character| character.is_control())
        {
            return Err("candidate ID is empty, comment-like, or contains control characters"
                .to_string());
        }
        if !candidate_ids.insert(candidate.id.as_str()) {
            return Err(format!("duplicate candidate ID {:?}", candidate.id));
        }
        for (feature_index, value) in candidate.features.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "candidate {:?}, feature {} must be finite",
                    candidate.id, table.feature_names[feature_index]
                ));
            }
            if value.abs() > MAX_FEATURE_ABS {
                return Err(format!(
                    "candidate {:?}, feature {} exceeds the absolute-value limit of {MAX_FEATURE_ABS}",
                    candidate.id, table.feature_names[feature_index]
                ));
            }
        }
    }

    Ok(())
}

fn exact_duplicate_map(candidates: &[Candidate]) -> Vec<Option<usize>> {
    let mut first_by_features: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut duplicate_of = vec![None; candidates.len()];

    for (index, candidate) in candidates.iter().enumerate() {
        let key: Vec<u64> = candidate
            .features
            .iter()
            .map(|value| if *value == 0.0 { 0 } else { value.to_bits() })
            .collect();
        if let Some(first_index) = first_by_features.get(&key) {
            duplicate_of[index] = Some(*first_index);
        } else {
            first_by_features.insert(key, index);
        }
    }

    duplicate_of
}

fn normalize_features(candidates: &[Candidate], eligible: &[usize]) -> Vec<Vec<f64>> {
    let feature_count = candidates[0].features.len();
    let mut means = vec![0.0; feature_count];
    let mut m2 = vec![0.0; feature_count];
    let mut count = 0.0_f64;

    // Welford's algorithm avoids summing raw feature values.
    for &candidate_index in eligible {
        count += 1.0;
        for (feature_index, value) in candidates[candidate_index].features.iter().enumerate() {
            let delta = *value - means[feature_index];
            means[feature_index] += delta / count;
            let delta_after = *value - means[feature_index];
            m2[feature_index] = (m2[feature_index] + delta * delta_after).max(0.0);
        }
    }

    let scales: Vec<f64> = m2
        .iter()
        .map(|sum_squares| {
            let standard_deviation = (sum_squares / count).sqrt();
            if standard_deviation.is_finite() && standard_deviation > 0.0 {
                standard_deviation
            } else {
                0.0
            }
        })
        .collect();

    candidates
        .iter()
        .map(|candidate| {
            candidate
                .features
                .iter()
                .enumerate()
                .map(|(feature_index, value)| {
                    let scale = scales[feature_index];
                    if scale == 0.0 {
                        0.0
                    } else {
                        (*value - means[feature_index]) / scale
                    }
                })
                .collect()
        })
        .collect()
}

fn squared_euclidean(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn euclidean(left: &[f64], right: &[f64]) -> f64 {
    squared_euclidean(left, right).sqrt()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        let bound = upper_bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

pub fn select(
    table: &CandidateTable,
    policy: Policy,
    budget: usize,
    seed: u64,
) -> Result<RunReport, String> {
    validate_table(table)?;
    if budget == 0 {
        return Err("budget must be at least 1".to_string());
    }
    let feature_count = table.feature_names.len();

    let duplicate_of = exact_duplicate_map(&table.candidates);
    let mut remaining: Vec<usize> = duplicate_of
        .iter()
        .enumerate()
        .filter(|(_, duplicate)| duplicate.is_none())
        .map(|(index, _)| index)
        .collect();
    if budget > remaining.len() {
        return Err(format!(
            "budget {budget} exceeds {} unique feature vectors",
            remaining.len()
        ));
    }

    let estimated_operations = remaining
        .len()
        .checked_mul(budget)
        .and_then(|value| value.checked_mul(feature_count))
        .ok_or_else(|| "distance-operation estimate overflow".to_string())?;
    if estimated_operations > MAX_DISTANCE_OPERATIONS {
        return Err(format!(
            "estimated feature-distance work {estimated_operations} exceeds limit {MAX_DISTANCE_OPERATIONS}; reduce candidates, features, or budget"
        ));
    }

    let normalized = normalize_features(&table.candidates, &remaining);
    let mut rng = SplitMix64::new(seed);

    // Both policies share the same seeded first candidate for a paired comparison.
    let first_position = rng.index(remaining.len());
    let first_index = remaining.swap_remove(first_position);
    let mut nearest_distance = vec![f64::INFINITY; table.candidates.len()];
    for &index in &remaining {
        nearest_distance[index] = euclidean(&normalized[index], &normalized[first_index]);
    }

    let mut selected = Vec::with_capacity(budget);
    selected.push(Selection {
        candidate_index: first_index,
        rank: 1,
        distance_to_prior_selected: None,
    });

    while selected.len() < budget {
        let chosen_position = match policy {
            Policy::Random => rng.index(remaining.len()),
            Policy::MaxMin => {
                let mut best_position = 0;
                for position in 1..remaining.len() {
                    let candidate_index = remaining[position];
                    let best_index = remaining[best_position];
                    match nearest_distance[candidate_index]
                        .total_cmp(&nearest_distance[best_index])
                    {
                        Ordering::Greater => best_position = position,
                        Ordering::Equal if candidate_index < best_index => {
                            best_position = position;
                        }
                        _ => {}
                    }
                }
                best_position
            }
        };

        let chosen_index = remaining[chosen_position];
        let distance_to_prior_selected = nearest_distance[chosen_index];
        let rank = selected.len() + 1;
        selected.push(Selection {
            candidate_index: chosen_index,
            rank,
            distance_to_prior_selected: Some(distance_to_prior_selected),
        });
        remaining.swap_remove(chosen_position);

        for &candidate_index in &remaining {
            let distance = euclidean(&normalized[candidate_index], &normalized[chosen_index]);
            if distance < nearest_distance[candidate_index] {
                nearest_distance[candidate_index] = distance;
            }
        }
    }

    Ok(RunReport {
        policy,
        seed,
        budget,
        duplicate_of,
        selected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> CandidateTable {
        parse_tsv(text).expect("valid test table")
    }

    const SAMPLE: &str =
        "candidate_id\tx\ty\nA\t0\t0\nB\t1\t0\nC\t2\t1\nD\t10\t8\nE\t-2\t4\n";

    #[test]
    fn parses_valid_tsv() {
        let parsed = table(SAMPLE);
        assert_eq!(parsed.feature_names, vec!["x", "y"]);
        assert_eq!(parsed.candidates.len(), 5);
        assert_eq!(parsed.candidates[0].id, "A");
    }

    #[test]
    fn rejects_comment_like_candidate_ids() {
        let input = "candidate_id\\tx\\n#design\\t1\\n";
        assert!(parse_tsv(input).unwrap_err().contains("comment-like"));
    }

    #[test]
    fn select_rejects_invalid_programmatic_candidates() {
        let invalid = CandidateTable {
            feature_names: vec!["x".to_string()],
            candidates: vec![Candidate {
                id: "A".to_string(),
                features: vec![f64::NAN],
            }],
        };
        assert!(select(&invalid, Policy::Random, 1, 0)
            .unwrap_err()
            .contains("must be finite"));
    }

    #[test]
    fn rejects_work_budget_before_distance_search() {
        let feature_names: Vec<String> = (0..512).map(|index| format!("f{index}")).collect();
        let candidates: Vec<Candidate> = (0..500)
            .map(|index| {
                let mut features = vec![0.0; 512];
                features[0] = index as f64;
                Candidate {
                    id: format!("c{index}"),
                    features,
                }
            })
            .collect();
        let oversized = CandidateTable {
            feature_names,
            candidates,
        };
        let error = select(&oversized, Policy::MaxMin, 500, 0).unwrap_err();
        assert!(error.contains("exceeds limit"));
    }

    #[test]
    fn rejects_duplicate_candidate_ids() {
        let input = "candidate_id\tx\nA\t1\nA\t2\n";
        assert!(parse_tsv(input)
            .unwrap_err()
            .contains("repeats candidate ID"));
    }

    #[test]
    fn rejects_duplicate_feature_names() {
        let input = "candidate_id\tx\tx\nA\t1\t2\n";
        assert!(parse_tsv(input)
            .unwrap_err()
            .contains("duplicate feature name"));
    }

    #[test]
    fn rejects_non_finite_features() {
        let input = "candidate_id\tx\nA\tNaN\n";
        assert!(parse_tsv(input).unwrap_err().contains("finite"));
    }

    #[test]
    fn rejects_feature_values_over_limit() {
        let input = "candidate_id\tx\nA\t1000000000001\n";
        assert!(parse_tsv(input)
            .unwrap_err()
            .contains("absolute-value limit"));
    }

    #[test]
    fn rejects_malformed_row_width() {
        let input = "candidate_id\tx\ty\nA\t1\n";
        assert!(parse_tsv(input).unwrap_err().contains("columns"));
    }

    #[test]
    fn rejects_header_without_rows() {
        let input = "candidate_id\tx\n";
        assert!(parse_tsv(input).unwrap_err().contains("no candidate rows"));
    }

    #[test]
    fn exact_feature_duplicates_are_visible_and_not_selected_twice() {
        let parsed = table("candidate_id\tx\ty\nA\t1\t2\nB\t1\t2\nC\t3\t4\n");
        let report = select(&parsed, Policy::MaxMin, 2, 7).expect("selects two unique rows");
        assert_eq!(report.duplicate_of, vec![None, Some(0), None]);
        assert_eq!(report.selected.len(), 2);
        assert!(report.selected.iter().all(|item| item.candidate_index != 1));
        assert_eq!(report.eligible_count(parsed.candidates.len()), 2);
    }

    #[test]
    fn rejects_budget_larger_than_unique_candidate_count() {
        let parsed = table("candidate_id\tx\nA\t1\nB\t1\nC\t2\n");
        let error = select(&parsed, Policy::Random, 3, 0).unwrap_err();
        assert!(error.contains("2 unique feature vectors"));
    }

    #[test]
    fn random_and_maxmin_share_the_same_seeded_first_pick() {
        let parsed = table(SAMPLE);
        let random = select(&parsed, Policy::Random, 3, 1234).expect("random selection");
        let maxmin = select(&parsed, Policy::MaxMin, 3, 1234).expect("maxmin selection");
        assert_eq!(
            random.selected[0].candidate_index,
            maxmin.selected[0].candidate_index
        );
    }

    #[test]
    fn seeded_runs_are_deterministic() {
        let parsed = table(SAMPLE);
        let first = select(&parsed, Policy::MaxMin, 4, 99).expect("first run");
        let second = select(&parsed, Policy::MaxMin, 4, 99).expect("second run");
        assert_eq!(first, second);
    }

    #[test]
    fn maxmin_selects_the_farthest_remaining_candidate() {
        let parsed = table(SAMPLE);
        let report = select(&parsed, Policy::MaxMin, 2, 19).expect("maxmin selection");
        let selected_first = report.selected[0].candidate_index;
        let all_indices: Vec<usize> = (0..parsed.candidates.len()).collect();
        let normalized = normalize_features(&parsed.candidates, &all_indices);
        let best_distance = (0..parsed.candidates.len())
            .filter(|index| *index != selected_first)
            .map(|index| euclidean(&normalized[index], &normalized[selected_first]))
            .fold(f64::NEG_INFINITY, f64::max);
        let actual = report.selected[1]
            .distance_to_prior_selected
            .expect("second selection has a prior candidate");
        assert!((actual - best_distance).abs() < 1.0e-10);
    }

    #[test]
    fn constant_features_do_not_create_non_finite_distances() {
        let parsed = table("candidate_id\tconstant\tx\nA\t7\t0\nB\t7\t1\nC\t7\t2\n");
        let report = select(&parsed, Policy::MaxMin, 3, 4).expect("constant feature handled");
        assert!(report
            .selected
            .iter()
            .filter_map(|selection| selection.distance_to_prior_selected)
            .all(f64::is_finite));
    }

    #[test]
    fn rejects_unknown_policy_names() {
        assert!(Policy::parse("active-inference").is_err());
    }
}
