//! Provider-neutral known-answer scoring for firmware extraction reports.
//!
//! A region's offsets are meaningful only relative to its coordinate space.
//! Nested decompression produces a new coordinate space with its own digest;
//! child offsets must never be silently reinterpreted as root-file offsets.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ORACLE_SCHEMA: &str = "symthaea.rea-differential-oracle.v1";
pub const OBSERVATION_SCHEMA: &str = "symthaea.rea-differential-observation.v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SourceIdentity {
    pub sha256: String,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct CoordinateSpace {
    pub id: String,
    pub sha256: String,
    pub byte_length: u64,
    pub parent_coordinate_space_id: Option<String>,
    pub parent_start: Option<u64>,
    pub parent_end: Option<u64>,
    /// "source" for root bytes; otherwise a typed transform such as "decompress:gzip".
    pub transform_kind: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Region {
    pub coordinate_space_id: String,
    pub kind: String,
    pub start: u64,
    pub end: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpectedFile {
    pub path: String,
    pub sha256: String,
    pub byte_length: u64,
    pub logical_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct OracleCase {
    pub schema_version: String,
    pub corpus_id: String,
    pub case_id: String,
    pub source: SourceIdentity,
    pub coordinate_spaces: Vec<CoordinateSpace>,
    pub expected_regions: Vec<Region>,
    pub expected_files: Vec<ExpectedFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct AccountingSpan {
    pub coordinate_space_id: String,
    pub start: u64,
    pub end: u64,
    /// Allowed values: identified, padding, unknown, unsupported, ambiguous.
    pub disposition: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Observation {
    pub schema_version: String,
    pub case_id: String,
    pub provider_id: String,
    pub provider_version: String,
    /// Allowed values: supported, unsupported, unknown.
    pub capability: String,
    pub source: SourceIdentity,
    pub coordinate_spaces: Vec<CoordinateSpace>,
    pub accounting: Vec<AccountingSpan>,
    pub regions: Vec<Region>,
    pub files: Vec<ExpectedFile>,
    pub complete_extraction_claim: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Metrics {
    pub source_identity_match: bool,
    pub coordinate_spaces_expected: usize,
    pub coordinate_spaces_exact: usize,
    pub coordinate_spaces_missing_or_wrong: usize,
    pub coordinate_spaces_extra: usize,
    pub known_files_expected: usize,
    pub known_files_exact: usize,
    pub known_file_recall_percent: u8,
    pub output_files_observed: usize,
    pub extra_or_wrong_files: usize,
    pub duplicate_paths: usize,
    pub expected_regions: usize,
    pub exact_regions: usize,
    pub missed_or_false_regions: usize,
    pub source_bytes: u64,
    pub coordinate_space_bytes: u64,
    pub accounted_unique_bytes: u64,
    pub unexplained_bytes: u64,
    pub overlapping_accounting_bytes: u64,
    pub unknown_bytes: u64,
    pub unsupported_bytes: u64,
    pub ambiguous_bytes: u64,
    pub padding_bytes: u64,
    pub identified_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Scorecard {
    pub schema_version: String,
    pub corpus_id: String,
    pub case_id: String,
    pub provider_id: String,
    pub provider_version: String,
    /// pass, partial, fail, or abstain.
    pub status: String,
    pub complete_extraction_claim_accepted: bool,
    pub metrics: Metrics,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreError(pub String);

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ScoreError {}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalized_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && !path.contains(':')
        && path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}

fn validate_file(file: &ExpectedFile, side: &str, reasons: &mut Vec<String>) {
    if !normalized_relative_path(&file.path) {
        reasons.push(format!("{side} file path is not normalized and relative: {}", file.path));
    }
    if !valid_digest(&file.sha256) {
        reasons.push(format!("{side} file SHA-256 is malformed: {}", file.path));
    }
    if file.logical_id.is_empty() {
        reasons.push(format!("{side} file has an empty logical ID: {}", file.path));
    }
}

fn coordinate_map(spaces: &[CoordinateSpace], side: &str, reasons: &mut Vec<String>) -> BTreeMap<String, CoordinateSpace> {
    let mut result = BTreeMap::new();
    for space in spaces {
        if space.id.is_empty() {
            reasons.push(format!("{side} coordinate space has an empty ID"));
        }
        if !valid_digest(&space.sha256) {
            reasons.push(format!("{side} coordinate space {} has malformed SHA-256", space.id));
        }
        if result.insert(space.id.clone(), space.clone()).is_some() {
            reasons.push(format!("{side} coordinate space ID is duplicated: {}", space.id));
        }
    }
    result
}

fn validate_coordinate_graph(spaces: &BTreeMap<String, CoordinateSpace>, side: &str, reasons: &mut Vec<String>) {
    if !spaces.contains_key("root") {
        reasons.push(format!("{side} coordinate graph lacks root space"));
        return;
    }
    for space in spaces.values() {
        if space.id == "root" {
            if space.parent_coordinate_space_id.is_some()
                || space.parent_start.is_some()
                || space.parent_end.is_some()
                || space.transform_kind != "source"
            {
                reasons.push(format!("{side} root coordinate space must be parentless and use source transform kind"));
            }
            continue;
        }
        let parent_id = match space.parent_coordinate_space_id.as_deref() {
            Some(id) if !id.is_empty() => id,
            _ => {
                reasons.push(format!("{side} coordinate space {} has no parent identity", space.id));
                continue;
            }
        };
        let parent = match spaces.get(parent_id) {
            Some(parent) => parent,
            None => {
                reasons.push(format!("{side} coordinate space {} references missing parent {parent_id}", space.id));
                continue;
            }
        };
        match (space.parent_start, space.parent_end) {
            (Some(start), Some(end)) if start < end && end <= parent.byte_length => {}
            _ => reasons.push(format!("{side} coordinate space {} has invalid parent byte range", space.id)),
        }
        if space.transform_kind.is_empty() || space.transform_kind == "source" {
            reasons.push(format!("{side} coordinate space {} has invalid transform kind", space.id));
        }

        // Every derived coordinate space must reach the root without a cycle.
        let mut seen = BTreeSet::new();
        let mut cursor = space.id.as_str();
        loop {
            if !seen.insert(cursor.to_owned()) {
                reasons.push(format!("{side} coordinate graph contains a cycle involving {}", space.id));
                break;
            }
            if cursor == "root" {
                break;
            }
            match spaces.get(cursor).and_then(|node| node.parent_coordinate_space_id.as_deref()) {
                Some(parent) if spaces.contains_key(parent) => cursor = parent,
                _ => {
                    reasons.push(format!("{side} coordinate space {} is not rooted", space.id));
                    break;
                }
            }
        }
    }
}

fn same_root_space(source: &SourceIdentity, spaces: &BTreeMap<String, CoordinateSpace>, side: &str, reasons: &mut Vec<String>) -> bool {
    match spaces.get("root") {
        Some(root) if root.sha256 == source.sha256 && root.byte_length == source.byte_length => true,
        _ => {
            reasons.push(format!("{side} root coordinate space does not match the selected source identity"));
            false
        }
    }
}

fn sum_dispositions(
    accounting: &[AccountingSpan],
    spaces: &BTreeMap<String, CoordinateSpace>,
    reasons: &mut Vec<String>,
) -> (BTreeMap<String, u64>, u64, u64, u64) {
    let allowed: BTreeSet<&str> =
        ["identified", "padding", "unknown", "unsupported", "ambiguous"].into_iter().collect();
    let mut grouped: BTreeMap<String, Vec<&AccountingSpan>> = BTreeMap::new();
    for span in accounting {
        if !allowed.contains(span.disposition.as_str()) {
            reasons.push(format!("unknown accounting disposition: {}", span.disposition));
            continue;
        }
        if !spaces.contains_key(&span.coordinate_space_id) {
            reasons.push(format!("accounting span references unknown coordinate space: {}", span.coordinate_space_id));
            continue;
        }
        grouped.entry(span.coordinate_space_id.clone()).or_default().push(span);
    }

    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut unique_covered = 0_u64;
    let mut unexplained = 0_u64;
    let mut overlapping = 0_u64;

    // Every oracle-declared space is evaluated, even if a provider omitted the
    // corresponding child space; omission cannot silently reduce the denominator.
    for (space_id, space) in spaces {
        let mut spans = grouped.remove(space_id).unwrap_or_default();
        spans.sort_by_key(|span| (span.start, span.end, span.disposition.clone()));
        let mut cursor = 0_u64;
        for span in spans {
            if span.start >= span.end {
                reasons.push(format!(
                    "accounting span is empty or reversed in {space_id}: {}..{}",
                    span.start, span.end
                ));
                continue;
            }
            if span.end > space.byte_length {
                reasons.push(format!(
                    "accounting span exceeds {space_id} length: {}..{}",
                    span.start, span.end
                ));
                continue;
            }
            if span.start > cursor {
                unexplained = unexplained.saturating_add(span.start - cursor);
            }
            if span.start < cursor {
                let duplicate = span.end.min(cursor).saturating_sub(span.start);
                overlapping = overlapping.saturating_add(duplicate);
                reasons.push(format!("accounting spans overlap by {duplicate} bytes in {space_id}"));
            }
            let fresh_start = span.start.max(cursor);
            if span.end > fresh_start {
                let fresh_length = span.end - fresh_start;
                unique_covered = unique_covered.saturating_add(fresh_length);
                *counts.entry(span.disposition.clone()).or_default() += fresh_length;
            }
            cursor = cursor.max(span.end);
        }
        if cursor < space.byte_length {
            unexplained = unexplained.saturating_add(space.byte_length - cursor);
        }
    }
    (counts, unique_covered, unexplained, overlapping)
}

fn file_multiset(files: &[ExpectedFile]) -> BTreeMap<ExpectedFile, usize> {
    let mut result = BTreeMap::new();
    for file in files {
        *result.entry(file.clone()).or_default() += 1;
    }
    result
}

fn duplicate_path_count(files: &[ExpectedFile]) -> usize {
    let mut seen = BTreeSet::new();
    let mut duplicate_paths = BTreeSet::new();
    for file in files {
        if !seen.insert(file.path.as_str()) {
            duplicate_paths.insert(file.path.as_str());
        }
    }
    duplicate_paths.len()
}

fn exact_file_overlap(expected: &[ExpectedFile], observed: &[ExpectedFile]) -> usize {
    let mut expected_counts = file_multiset(expected);
    let mut matches = 0;
    for file in observed {
        if let Some(count) = expected_counts.get_mut(file) {
            if *count > 0 {
                *count -= 1;
                matches += 1;
            }
        }
    }
    matches
}

fn exact_region_overlap(expected: &[Region], observed: &[Region]) -> usize {
    let mut expected_counts: BTreeMap<Region, usize> = BTreeMap::new();
    for region in expected {
        *expected_counts.entry(region.clone()).or_default() += 1;
    }
    let mut matches = 0;
    for region in observed {
        if let Some(count) = expected_counts.get_mut(region) {
            if *count > 0 {
                *count -= 1;
                matches += 1;
            }
        }
    }
    matches
}

/// Scores one observed provider report against an independently authored oracle.
///
/// A child coordinate space is a separately digested byte sequence, such as a
/// decompressed tar stream. Its offsets are never projected onto the root input.
/// Unsupported/unknown capability abstains without recall credit. A complete
/// claim fails if any declared coordinate space has unresolved or double-counted
/// bytes.
pub fn score_case(oracle: &OracleCase, observed: &Observation) -> Result<Scorecard, ScoreError> {
    if oracle.schema_version != ORACLE_SCHEMA {
        return Err(ScoreError(format!("unsupported oracle schema: {}", oracle.schema_version)));
    }
    if observed.schema_version != OBSERVATION_SCHEMA {
        return Err(ScoreError(format!("unsupported observation schema: {}", observed.schema_version)));
    }
    if oracle.case_id != observed.case_id {
        return Err(ScoreError("oracle and observation case IDs do not match".into()));
    }
    if oracle.corpus_id.is_empty() || observed.provider_id.is_empty() || observed.provider_version.is_empty() {
        return Err(ScoreError("corpus, provider ID, and provider version must be non-empty".into()));
    }
    if !valid_digest(&oracle.source.sha256) || !valid_digest(&observed.source.sha256) {
        return Err(ScoreError("source SHA-256 must be 64 lowercase hexadecimal characters".into()));
    }
    let capability_supported = observed.capability == "supported";
    if !["supported", "unsupported", "unknown"].contains(&observed.capability.as_str()) {
        return Err(ScoreError(format!("invalid capability value: {}", observed.capability)));
    }

    let mut reasons = Vec::new();
    let source_identity_match = oracle.source == observed.source;
    if !source_identity_match {
        reasons.push("observed source length or SHA-256 differs from independent oracle".into());
    }

    let oracle_spaces = coordinate_map(&oracle.coordinate_spaces, "oracle", &mut reasons);
    let observed_spaces = coordinate_map(&observed.coordinate_spaces, "observed", &mut reasons);
    validate_coordinate_graph(&oracle_spaces, "oracle", &mut reasons);
    validate_coordinate_graph(&observed_spaces, "observed", &mut reasons);
    same_root_space(&oracle.source, &oracle_spaces, "oracle", &mut reasons);
    if capability_supported {
        same_root_space(&observed.source, &observed_spaces, "observed", &mut reasons);
    }

    let mut coordinate_spaces_exact = 0_usize;
    let mut coordinate_spaces_missing_or_wrong = 0_usize;
    let mut coordinate_spaces_extra = 0_usize;
    if capability_supported {
        for (id, expected) in &oracle_spaces {
            if observed_spaces.get(id) == Some(expected) {
                coordinate_spaces_exact += 1;
            } else {
                coordinate_spaces_missing_or_wrong += 1;
                reasons.push(format!("coordinate space differs from oracle: {id}"));
            }
        }
        for id in observed_spaces.keys() {
            if !oracle_spaces.contains_key(id) {
                coordinate_spaces_extra += 1;
                reasons.push(format!("extra coordinate space not present in oracle: {id}"));
            }
        }
    }

    for file in &oracle.expected_files {
        validate_file(file, "oracle", &mut reasons);
    }
    for file in &observed.files {
        validate_file(file, "observed", &mut reasons);
    }
    for region in &oracle.expected_regions {
        if !oracle_spaces.contains_key(&region.coordinate_space_id) {
            reasons.push(format!("oracle region references unknown coordinate space: {}", region.coordinate_space_id));
        } else if region.start >= region.end || region.end > oracle_spaces[&region.coordinate_space_id].byte_length {
            reasons.push(format!("oracle region is out of bounds or empty: {} {}..{}", region.kind, region.start, region.end));
        }
    }
    if duplicate_path_count(&oracle.expected_files) > 0 {
        reasons.push("oracle contains duplicate output paths; oracle must define collision semantics explicitly".into());
    }
    let duplicate_paths = duplicate_path_count(&observed.files);
    if duplicate_paths > 0 {
        reasons.push(format!("{duplicate_paths} output path(s) collide in observed tree"));
    }

    let (dispositions, accounted_unique_bytes, unexplained_bytes, overlapping_accounting_bytes) =
        sum_dispositions(&observed.accounting, &oracle_spaces, &mut reasons);
    let get = |name: &str| dispositions.get(name).copied().unwrap_or(0);
    let unknown_bytes = get("unknown");
    let unsupported_bytes = get("unsupported");
    let ambiguous_bytes = get("ambiguous");
    let padding_bytes = get("padding");
    let identified_bytes = get("identified");

    for region in &observed.regions {
        match oracle_spaces.get(&region.coordinate_space_id) {
            None => reasons.push(format!("detected region references unknown coordinate space: {}", region.coordinate_space_id)),
            Some(space) if region.start >= region.end || region.end > space.byte_length => {
                reasons.push(format!("detected region is out of bounds or empty: {} {} {}..{}", region.coordinate_space_id, region.kind, region.start, region.end));
            }
            _ => {}
        }
    }
    let exact_regions = if capability_supported {
        exact_region_overlap(&oracle.expected_regions, &observed.regions)
    } else {
        0
    };
    let missed_or_false_regions = if capability_supported {
        oracle.expected_regions.len() + observed.regions.len() - 2 * exact_regions
    } else {
        0
    };
    if capability_supported && missed_or_false_regions > 0 {
        reasons.push(format!("{missed_or_false_regions} missed or false-positive region(s)"));
    }

    let known_files_exact = if capability_supported {
        exact_file_overlap(&oracle.expected_files, &observed.files)
    } else {
        0
    };
    let extra_or_wrong_files = if capability_supported {
        oracle.expected_files.len() + observed.files.len() - 2 * known_files_exact
    } else {
        0
    };
    if capability_supported && (known_files_exact != oracle.expected_files.len() || observed.files.len() != oracle.expected_files.len()) {
        reasons.push(format!(
            "known-answer output tree differs: expected {}, exact matches {}, observed {}",
            oracle.expected_files.len(),
            known_files_exact,
            observed.files.len()
        ));
    }
    if !capability_supported && observed.complete_extraction_claim {
        reasons.push("unsupported/unknown provider capability cannot claim completeness".into());
    }
    if !capability_supported && !observed.files.is_empty() {
        reasons.push("unsupported/unknown capability emitted files; claim requires review".into());
    }

    let known_file_recall_percent = if !capability_supported {
        0
    } else if oracle.expected_files.is_empty() {
        if observed.files.is_empty() { 100 } else { 0 }
    } else {
        ((known_files_exact * 100) / oracle.expected_files.len()).min(100) as u8
    };

    let unresolved_bytes = unknown_bytes
        .saturating_add(unsupported_bytes)
        .saturating_add(ambiguous_bytes)
        .saturating_add(unexplained_bytes)
        .saturating_add(overlapping_accounting_bytes);
    if unexplained_bytes > 0 {
        reasons.push(format!("{unexplained_bytes} bytes across declared coordinate spaces are not accounted for by any span"));
    }
    if unknown_bytes > 0 || unsupported_bytes > 0 || ambiguous_bytes > 0 {
        reasons.push(format!(
            "unresolved byte classes: unknown={unknown_bytes}, unsupported={unsupported_bytes}, ambiguous={ambiguous_bytes}"
        ));
    }
    if observed.complete_extraction_claim && unresolved_bytes > 0 {
        reasons.push(format!(
            "complete-extraction claim rejected with {unresolved_bytes} unresolved/double-accounted bytes"
        ));
    }

    let content_or_structure_failed = reasons.iter().any(|reason| {
        reason.contains("differs")
            || reason.contains("duplicate output paths")
            || reason.contains("collide")
            || reason.contains("out of bounds")
            || reason.contains("unknown coordinate space")
            || reason.contains("extra coordinate space")
            || reason.contains("duplicated")
            || reason.contains("does not match the selected source identity")
            || reason.contains("parent identity")
            || reason.contains("parent byte range")
            || reason.contains("transform kind")
            || reason.contains("coordinate graph")
            || reason.contains("is not rooted")
            || reason.contains("missed or false-positive")
            || reason.contains("known-answer output tree differs")
            || reason.contains("overlap")
            || reason.contains("span exceeds")
            || reason.contains("span is empty")
            || reason.contains("unknown accounting disposition")
            || reason.contains("not normalized and relative")
            || reason.contains("SHA-256 is malformed")
            || reason.contains("empty logical ID")
    });

    let status = if !capability_supported {
        if observed.complete_extraction_claim || !observed.files.is_empty() || content_or_structure_failed {
            "fail"
        } else {
            "abstain"
        }
    } else if content_or_structure_failed || (observed.complete_extraction_claim && unresolved_bytes > 0) {
        "fail"
    } else if unexplained_bytes > 0
        || overlapping_accounting_bytes > 0
        || unknown_bytes > 0
        || unsupported_bytes > 0
        || ambiguous_bytes > 0
        || !observed.complete_extraction_claim
    {
        "partial"
    } else {
        "pass"
    };

    let coordinate_space_bytes = oracle_spaces.values().fold(0_u64, |sum, space| sum.saturating_add(space.byte_length));
    Ok(Scorecard {
        schema_version: "symthaea.rea-differential-scorecard.v1".into(),
        corpus_id: oracle.corpus_id.clone(),
        case_id: oracle.case_id.clone(),
        provider_id: observed.provider_id.clone(),
        provider_version: observed.provider_version.clone(),
        status: status.into(),
        complete_extraction_claim_accepted: status == "pass" && observed.complete_extraction_claim,
        metrics: Metrics {
            source_identity_match,
            coordinate_spaces_expected: oracle_spaces.len(),
            coordinate_spaces_exact,
            coordinate_spaces_missing_or_wrong,
            coordinate_spaces_extra,
            known_files_expected: oracle.expected_files.len(),
            known_files_exact,
            known_file_recall_percent,
            output_files_observed: observed.files.len(),
            extra_or_wrong_files,
            duplicate_paths,
            expected_regions: oracle.expected_regions.len(),
            exact_regions,
            missed_or_false_regions,
            source_bytes: oracle.source.byte_length,
            coordinate_space_bytes,
            accounted_unique_bytes,
            unexplained_bytes,
            overlapping_accounting_bytes,
            unknown_bytes,
            unsupported_bytes,
            ambiguous_bytes,
            padding_bytes,
            identified_bytes,
        },
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle() -> OracleCase {
        OracleCase {
            schema_version: ORACLE_SCHEMA.into(),
            corpus_id: "rea-differential-v1".into(),
            case_id: "nested-inert-001".into(),
            source: SourceIdentity { sha256: "a".repeat(64), byte_length: 10 },
            coordinate_spaces: vec![
                CoordinateSpace { id: "root".into(), sha256: "a".repeat(64), byte_length: 10, parent_coordinate_space_id: None, parent_start: None, parent_end: None, transform_kind: "source".into() },
                CoordinateSpace { id: "tar-stream".into(), sha256: "b".repeat(64), byte_length: 7, parent_coordinate_space_id: Some("root".into()), parent_start: Some(0), parent_end: Some(10), transform_kind: "decompress:gzip".into() },
            ],
            expected_regions: vec![
                Region { coordinate_space_id: "root".into(), kind: "gzip-container".into(), start: 0, end: 10 },
                Region { coordinate_space_id: "tar-stream".into(), kind: "tar-container".into(), start: 0, end: 7 },
            ],
            expected_files: vec![
                ExpectedFile { path: "payload.txt".into(), sha256: "c".repeat(64), byte_length: 4, logical_id: "payload-001".into() },
            ],
        }
    }

    fn observation() -> Observation {
        Observation {
            schema_version: OBSERVATION_SCHEMA.into(),
            case_id: "nested-inert-001".into(),
            provider_id: "fixture-adapter".into(),
            provider_version: "0.0-test".into(),
            capability: "supported".into(),
            source: SourceIdentity { sha256: "a".repeat(64), byte_length: 10 },
            coordinate_spaces: vec![
                CoordinateSpace { id: "root".into(), sha256: "a".repeat(64), byte_length: 10, parent_coordinate_space_id: None, parent_start: None, parent_end: None, transform_kind: "source".into() },
                CoordinateSpace { id: "tar-stream".into(), sha256: "b".repeat(64), byte_length: 7, parent_coordinate_space_id: Some("root".into()), parent_start: Some(0), parent_end: Some(10), transform_kind: "decompress:gzip".into() },
            ],
            accounting: vec![
                AccountingSpan { coordinate_space_id: "root".into(), start: 0, end: 10, disposition: "identified".into() },
                AccountingSpan { coordinate_space_id: "tar-stream".into(), start: 0, end: 7, disposition: "identified".into() },
            ],
            regions: vec![
                Region { coordinate_space_id: "root".into(), kind: "gzip-container".into(), start: 0, end: 10 },
                Region { coordinate_space_id: "tar-stream".into(), kind: "tar-container".into(), start: 0, end: 7 },
            ],
            files: vec![
                ExpectedFile { path: "payload.txt".into(), sha256: "c".repeat(64), byte_length: 4, logical_id: "payload-001".into() },
            ],
            complete_extraction_claim: true,
        }
    }

    #[test]
    fn scores_exact_nested_known_answer_as_pass() {
        let score = score_case(&oracle(), &observation()).unwrap();
        assert_eq!(score.status, "pass");
        assert!(score.complete_extraction_claim_accepted);
        assert_eq!(score.metrics.known_file_recall_percent, 100);
        assert_eq!(score.metrics.coordinate_spaces_exact, 2);
        assert_eq!(score.metrics.unexplained_bytes, 0);
        assert_eq!(score.metrics.coordinate_space_bytes, 17);
    }

    #[test]
    fn child_offsets_are_checked_in_child_space_not_root() {
        let mut obs = observation();
        obs.regions[1].start = 8;
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.metrics.missed_or_false_regions > 0);
    }

    #[test]
    fn rejects_complete_claim_when_child_bytes_are_unknown() {
        let mut obs = observation();
        obs.accounting[1].disposition = "unknown".into();
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(!score.complete_extraction_claim_accepted);
        assert_eq!(score.metrics.unknown_bytes, 7);
    }

    #[test]
    fn reports_unexplained_gaps_in_a_child_coordinate_space() {
        let mut obs = observation();
        obs.accounting[1] = AccountingSpan { coordinate_space_id: "tar-stream".into(), start: 0, end: 5, disposition: "identified".into() };
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "partial");
        assert_eq!(score.metrics.unexplained_bytes, 2);
        assert!(!score.complete_extraction_claim_accepted);
    }

    #[test]
    fn fails_overlapping_byte_accounting_in_one_coordinate_space() {
        let mut obs = observation();
        obs.accounting[1] = AccountingSpan { coordinate_space_id: "tar-stream".into(), start: 5, end: 7, disposition: "padding".into() };
        obs.accounting.push(AccountingSpan { coordinate_space_id: "tar-stream".into(), start: 4, end: 6, disposition: "identified".into() });
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.metrics.overlapping_accounting_bytes > 0);
    }

    #[test]
    fn wrong_source_hash_is_not_a_valid_reproduction() {
        let mut obs = observation();
        obs.source.sha256 = "d".repeat(64);
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(!score.metrics.source_identity_match);
    }

    #[test]
    fn wrong_child_space_digest_is_not_a_valid_lineage() {
        let mut obs = observation();
        obs.coordinate_spaces[1].sha256 = "d".repeat(64);
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert_eq!(score.metrics.coordinate_spaces_missing_or_wrong, 1);
    }

    #[test]
    fn wrong_content_hash_is_a_missing_known_answer_file() {
        let mut obs = observation();
        obs.files[0].sha256 = "e".repeat(64);
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert_eq!(score.metrics.known_files_exact, 0);
    }

    #[test]
    fn duplicate_paths_are_rejected_even_if_one_copy_is_correct() {
        let mut obs = observation();
        obs.files.push(obs.files[0].clone());
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert_eq!(score.metrics.duplicate_paths, 1);
    }

    #[test]
    fn unknown_provider_capability_abstains_without_credit() {
        let mut obs = observation();
        obs.capability = "unsupported".into();
        obs.files.clear();
        obs.coordinate_spaces.clear();
        obs.accounting.clear();
        obs.regions.clear();
        obs.complete_extraction_claim = false;
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "abstain");
        assert!(!score.complete_extraction_claim_accepted);
        assert_eq!(score.metrics.known_files_exact, 0);
        assert_eq!(score.metrics.known_file_recall_percent, 0);
    }

    #[test]
    fn unknown_capability_cannot_claim_complete_extraction() {
        let mut obs = observation();
        obs.capability = "unknown".into();
        obs.complete_extraction_claim = true;
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(!score.complete_extraction_claim_accepted);
    }

    #[test]
    fn rejects_non_normalized_paths() {
        let mut obs = observation();
        obs.files[0].path = "../escape.bin".into();
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
    }

    #[test]
    fn rejects_extra_unexpected_output_files() {
        let mut obs = observation();
        obs.files.push(ExpectedFile {
            path: "unexpected.bin".into(),
            sha256: "f".repeat(64),
            byte_length: 1,
            logical_id: "unexpected-001".into(),
        });
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert_eq!(score.metrics.extra_or_wrong_files, 1);
    }

    #[test]
    fn unsupported_spans_are_accounted_but_never_complete() {
        let mut obs = observation();
        obs.accounting[1].disposition = "unsupported".into();
        obs.complete_extraction_claim = false;
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "partial");
        assert_eq!(score.metrics.unsupported_bytes, 7);
        assert!(!score.complete_extraction_claim_accepted);
    }

    #[test]
    fn rejects_out_of_bounds_region_even_when_other_results_match() {
        let mut obs = observation();
        obs.regions.push(Region { coordinate_space_id: "tar-stream".into(), kind: "fake".into(), start: 6, end: 9 });
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.reasons.iter().any(|reason| reason.contains("out of bounds")));
    }

    #[test]
    fn refuses_wrong_schema_and_wrong_case_identity() {
        let mut obs = observation();
        obs.schema_version = "unknown.schema".into();
        assert!(score_case(&oracle(), &obs).unwrap_err().to_string().contains("unsupported observation schema"));
        obs.schema_version = OBSERVATION_SCHEMA.into();
        obs.case_id = "different-case".into();
        assert!(score_case(&oracle(), &obs).unwrap_err().to_string().contains("case IDs"));
    }

    #[test]
    fn rejects_missed_and_false_positive_regions() {
        let mut obs = observation();
        obs.regions[1].kind = "unknown".into();
        obs.regions[1].start = 1;
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.metrics.missed_or_false_regions > 0);
    }

    #[test]
    fn rejects_false_coordinate_parent_edge() {
        let mut obs = observation();
        obs.coordinate_spaces[1].parent_coordinate_space_id = Some("missing-parent".into());
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.reasons.iter().any(|reason| reason.contains("missing parent")));
    }

    #[test]
    fn rejects_wrong_transform_kind_for_child_space() {
        let mut obs = observation();
        obs.coordinate_spaces[1].transform_kind = "source".into();
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.reasons.iter().any(|reason| reason.contains("transform kind")));
    }

    #[test]
    fn rejects_duplicate_coordinate_space_ids() {
        let mut obs = observation();
        obs.coordinate_spaces.push(obs.coordinate_spaces[1].clone());
        let score = score_case(&oracle(), &obs).unwrap();
        assert_eq!(score.status, "fail");
        assert!(score.reasons.iter().any(|reason| reason.contains("duplicated")));
    }
}
