//! Discover candidate cross-family relationships from linked artifact paths.
//!
//! Paths are bibliographic/source-provenance metadata. Shared path vocabulary
//! is a review lead only; it is never sufficient to classify a mathematical
//! dependency, benchmark leakage, or a split exclusion.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fs,
};

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const UPSTREAM_COMMIT: &str = "1581d4cbdd69a0faf34b2089614fe2a814316a6a";
const SPLIT_SEED: &str = "pitb-v0-openai-math-2026-10-19";
const SPLIT_ALGORITHM: &str = concat!(
    "sha256(seed || NUL || family_id), sort by digest then family ID, ",
    "floor 70% train and floor 85% train+validation"
);
const EXPECTED_PARTITION_DIGEST: &str =
    "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970";
const EXPECTED_COUNTS: (usize, usize, usize) = (260, 56, 56);
const MAX_DOCUMENT_FREQUENCY: usize = 15;
const MIN_SHARED_RARE_TOKENS: usize = 2;
const MIN_PATH_JACCARD: f64 = 0.10;
const MIN_RARITY_SCORE: f64 = 4.0;

#[derive(Debug, Clone)]
struct Family {
    id: String,
    title: String,
    split: String,
    paths: Vec<String>,
    path_tokens: BTreeSet<String>,
}

#[derive(Debug, Serialize)]
struct Candidate {
    source_family_id: String,
    target_family_id: String,
    source_split: String,
    target_split: String,
    source_title: String,
    target_title: String,
    source_matching_paths: Vec<String>,
    target_matching_paths: Vec<String>,
    shared_path_tokens: Vec<String>,
    rare_shared_path_tokens: Vec<String>,
    path_token_jaccard: f64,
    rare_token_rarity_score: f64,
    candidate_relation: &'static str,
    candidate_status: &'static str,
    evidence_sha256: String,
}

#[derive(Debug, Serialize)]
struct SplitCounts {
    train: usize,
    validation: usize,
    test: usize,
}

#[derive(Debug, Serialize)]
struct CandidateCounts {
    candidate_pairs: usize,
    cross_split_pairs: usize,
    train_test_pairs: usize,
}

#[derive(Debug, Serialize)]
struct Config {
    tokenizer: &'static str,
    stopword_policy: &'static str,
    max_document_frequency: usize,
    min_shared_rare_tokens: usize,
    min_path_jaccard: f64,
    min_rarity_score: f64,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    upstream_repository: &'static str,
    upstream_commit: &'static str,
    source_partition_digest_sha256: String,
    total_family_count: usize,
    original_family_counts: SplitCounts,
    config: Config,
    counts: CandidateCounts,
    candidates: Vec<Candidate>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn stopwords() -> BTreeSet<&'static str> {
    concat!(
        "the a an and or nor but of on in to for from with by at as is are was were be been being ",
        "not does do did every each any all some over under into beyond ",
        "across between within without against after before through during ",
        "result results family families conjecture conjectures theorem theorems ",
        "lemma lemmas proof proofs bound bounds formula formulas problem problems ",
        "new related general using its their this that preprints preprint lean docs paper pdf main ",
        "final manuscript version draft supplement supplemental appendix appendices ",
        "october september august july june may april march january february december november"
    )
    .split_whitespace()
    .collect()
}

fn path_token_set(path: &str, token_re: &Regex, stop: &BTreeSet<&str>) -> BTreeSet<String> {
    let normalized: String = path
        .chars()
        .map(|ch| {
            if matches!(ch, '/' | '\\' | '-' | '_' | '.' | '(' | ')') {
                ' '
            } else {
                ch
            }
        })
        .collect();
    token_re
        .find_iter(&normalized.to_lowercase())
        .map(|m| m.as_str().to_owned())
        .filter(|token| !stop.contains(token.as_str()))
        .collect()
}

fn deterministic_splits(ids: &[String]) -> Result<BTreeMap<String, String>> {
    if ids.len() < 3 {
        bail!("at least three families are required");
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.trim().is_empty() || !unique.insert(id.clone()) {
            bail!("family IDs must be nonblank and unique");
        }
    }
    let mut ranked: Vec<(String, String)> = ids
        .iter()
        .map(|id| (sha256(format!("{SPLIT_SEED}\0{id}").as_bytes()), id.clone()))
        .collect();
    ranked.sort_unstable();
    let train_end = ranked.len() * 70 / 100;
    let validation_end = ranked.len() * 85 / 100;

    Ok(ranked
        .into_iter()
        .enumerate()
        .map(|(index, (_, id))| {
            let split = if index < train_end {
                "train"
            } else if index < validation_end {
                "validation"
            } else {
                "test"
            };
            (id, split.to_owned())
        })
        .collect())
}

fn partition_digest(splits: &BTreeMap<String, String>) -> String {
    let mut material = String::new();
    for (id, split) in splits {
        material.push_str(id);
        material.push('\t');
        material.push_str(split);
        material.push('\n');
    }
    sha256(material.as_bytes())
}

fn split_counts(splits: &BTreeMap<String, String>) -> SplitCounts {
    let mut result = SplitCounts {
        train: 0,
        validation: 0,
        test: 0,
    };
    for split in splits.values() {
        match split.as_str() {
            "train" => result.train += 1,
            "validation" => result.validation += 1,
            "test" => result.test += 1,
            _ => unreachable!("split was validated"),
        }
    }
    result
}

fn load_catalogue(doc: &Value) -> Result<(String, Vec<Family>, SplitCounts)> {
    if doc["schema_version"].as_str()
        != Some("symthaea.proof-insight-transfer.splits.v0")
        || doc["upstream"]["commit"].as_str() != Some(UPSTREAM_COMMIT)
    {
        bail!("catalogue does not identify the pinned PITB v0 source snapshot");
    }
    let plan = &doc["split_plan"];
    if plan["seed"].as_str() != Some(SPLIT_SEED)
        || plan["algorithm"].as_str() != Some(SPLIT_ALGORITHM)
        || plan["train_percent"].as_u64() != Some(70)
        || plan["validation_percent"].as_u64() != Some(15)
        || plan["test_percent"].as_u64() != Some(15)
    {
        bail!("split plan differs from registered v0 settings");
    }

    let rows = doc["families"].as_array().context("families must be an array")?;
    let mut splits = BTreeMap::new();
    for row in rows {
        let id = row["family_id"]
            .as_str()
            .context("family_id missing")?
            .to_owned();
        let split = row["split"].as_str().context("split missing")?.to_owned();
        if !matches!(split.as_str(), "train" | "validation" | "test")
            || id.trim().is_empty()
            || splits.insert(id, split).is_some()
        {
            bail!("invalid or duplicate family record");
        }
    }
    let ids: Vec<String> = splits.keys().cloned().collect();
    if splits != deterministic_splits(&ids)? {
        bail!("family assignments differ from the registered deterministic split");
    }
    let digest = partition_digest(&splits);
    if digest != EXPECTED_PARTITION_DIGEST
        || plan["partition_digest_sha256"].as_str() != Some(digest.as_str())
        || splits.len() != 372
    {
        bail!("catalogue does not match the exact frozen partition digest and size");
    }
    let counts = split_counts(&splits);
    if (counts.train, counts.validation, counts.test) != EXPECTED_COUNTS {
        bail!("catalogue counts differ from the pinned PITB v0 partition");
    }
    for (name, value) in [
        ("train", counts.train),
        ("validation", counts.validation),
        ("test", counts.test),
    ] {
        if plan["counts"][name].as_u64() != Some(value as u64) {
            bail!("declared {name} count does not match frozen assignments");
        }
    }

    let token_re = Regex::new(r"(?u)[\p{L}]{3,}")
        .context("compile Unicode path-token regex")?;
    let stop = stopwords();
    let mut families = Vec::with_capacity(rows.len());
    for row in rows {
        let id = row["family_id"].as_str().unwrap().to_owned();
        let title = row["title"].as_str().context("family title missing")?.to_owned();
        let split = row["split"].as_str().unwrap().to_owned();
        let paths: Vec<String> = row["linked_paths"]
            .as_array()
            .context("linked_paths must be an array")?
            .iter()
            .map(|value| {
                let path = value.as_str().context("linked path must be a string")?;
                if path.trim().is_empty() || path.starts_with('/') || path.contains("..") {
                    bail!("family {id} contains a blank, absolute, or traversing linked path");
                }
                Ok(path.to_owned())
            })
            .collect::<Result<_>>()?;
        let mut path_tokens = BTreeSet::new();
        for path in &paths {
            path_tokens.extend(path_token_set(path, &token_re, &stop));
        }
        families.push(Family { id, title, split, paths, path_tokens });
    }
    families.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((digest, families, counts))
}

fn matching_paths(
    family: &Family,
    shared_tokens: &BTreeSet<String>,
    token_re: &Regex,
) -> Vec<String> {
    family
        .paths
        .iter()
        .filter(|path| {
            token_re
                .find_iter(&path.to_lowercase())
                .any(|token| shared_tokens.contains(token.as_str()))
        })
        .cloned()
        .collect()
}

fn discover(families: &[Family]) -> Vec<Candidate> {
    let mut df = BTreeMap::<String, usize>::new();
    for family in families {
        for token in &family.path_tokens {
            *df.entry(token.clone()).or_default() += 1;
        }
    }

    let token_re = Regex::new(r"(?u)[\p{L}]{3,}").expect("static path-token regex");
    let mut candidates = Vec::new();
    for left in 0..families.len() {
        for right in (left + 1)..families.len() {
            let a = &families[left];
            let b = &families[right];
            let shared: BTreeSet<String> =
                a.path_tokens.intersection(&b.path_tokens).cloned().collect();
            let rare: BTreeSet<String> = shared
                .iter()
                .filter(|token| {
                    df.get(*token).copied().unwrap_or(usize::MAX)
                        <= MAX_DOCUMENT_FREQUENCY
                })
                .cloned()
                .collect();
            if rare.len() < MIN_SHARED_RARE_TOKENS {
                continue;
            }
            let union = a.path_tokens.union(&b.path_tokens).count();
            if union == 0 {
                continue;
            }
            let jaccard = shared.len() as f64 / union as f64;
            let rarity = rare
                .iter()
                .map(|token| {
                    let frequency = df.get(token).copied().unwrap_or(usize::MAX);
                    ((families.len() + 1) as f64 / (frequency + 1) as f64).ln()
                })
                .sum::<f64>();
            if jaccard < MIN_PATH_JACCARD && rarity < MIN_RARITY_SCORE {
                continue;
            }

            let source_matching_paths = matching_paths(a, &rare, &token_re);
            let target_matching_paths = matching_paths(b, &rare, &token_re);
            if source_matching_paths.is_empty() || target_matching_paths.is_empty() {
                continue;
            }
            let evidence = format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{:.6}\n{:.6}",
                a.id,
                b.id,
                source_matching_paths.join("\n"),
                target_matching_paths.join("\n"),
                shared.iter().cloned().collect::<Vec<_>>().join("\t"),
                rare.iter().cloned().collect::<Vec<_>>().join("\t"),
                jaccard,
                rarity
            );
            candidates.push(Candidate {
                source_family_id: a.id.clone(),
                target_family_id: b.id.clone(),
                source_split: a.split.clone(),
                target_split: b.split.clone(),
                source_title: a.title.clone(),
                target_title: b.title.clone(),
                source_matching_paths,
                target_matching_paths,
                shared_path_tokens: shared.into_iter().collect(),
                rare_shared_path_tokens: rare.into_iter().collect(),
                path_token_jaccard: (jaccard * 1000.0).round() / 1000.0,
                rare_token_rarity_score: (rarity * 100.0).round() / 100.0,
                candidate_relation: "artifact_path_similarity_only",
                candidate_status: "unreviewed",
                evidence_sha256: sha256(evidence.as_bytes()),
            });
        }
    }
    candidates.sort_by(|a, b| {
        b.rare_token_rarity_score
            .total_cmp(&a.rare_token_rarity_score)
            .then_with(|| b.path_token_jaccard.total_cmp(&a.path_token_jaccard))
            .then_with(|| a.source_family_id.cmp(&b.source_family_id))
            .then_with(|| a.target_family_id.cmp(&b.target_family_id))
    });
    candidates
}

fn report_counts(candidates: &[Candidate]) -> CandidateCounts {
    CandidateCounts {
        candidate_pairs: candidates.len(),
        cross_split_pairs: candidates
            .iter()
            .filter(|row| row.source_split != row.target_split)
            .count(),
        train_test_pairs: candidates
            .iter()
            .filter(|row| {
                (row.source_split == "train" && row.target_split == "test")
                    || (row.source_split == "test" && row.target_split == "train")
            })
            .count(),
    }
}

fn make_report(
    digest: String,
    families: &[Family],
    counts: SplitCounts,
    candidates: Vec<Candidate>,
) -> Report {
    Report {
        schema_version: "pitb.family-artifact-path-candidates.v0",
        upstream_repository: "https://github.com/openai/math",
        upstream_commit: UPSTREAM_COMMIT,
        source_partition_digest_sha256: digest,
        total_family_count: families.len(),
        original_family_counts: counts,
        config: Config {
            tokenizer: "lowercase Unicode letter tokens over manifest linked_paths",
            stopword_policy: "fixed explicit stopword list compiled into this tool",
            max_document_frequency: MAX_DOCUMENT_FREQUENCY,
            min_shared_rare_tokens: MIN_SHARED_RARE_TOKENS,
            min_path_jaccard: MIN_PATH_JACCARD,
            min_rarity_score: MIN_RARITY_SCORE,
            note: concat!(
                "artifact-path lexical similarity only; not proof of shared artifact bytes ",
                "or mathematical dependency",
            ),
        },
        counts: report_counts(&candidates),
        candidates,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        bail!(concat!(
            "usage: discover_artifact_path_candidates <frozen-splits.json> ",
            "<candidate-report.json>"
        ));
    }
    let doc: Value = serde_json::from_slice(
        &fs::read(&args[0]).with_context(|| format!("cannot read {}", args[0]))?,
    )
    .context("frozen split manifest is not valid JSON")?;
    let (digest, families, counts) = load_catalogue(&doc)?;
    let candidates = discover(&families);
    let report = make_report(digest, &families, counts, candidates);
    fs::write(&args[1], serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("cannot write {}", args[1]))?;
    println!(
        concat!(
            "PITB_ARTIFACT_PATH_CANDIDATES: pairs={} cross_split={} ",
            "train_test={} relation=artifact_path_similarity_only"
        ),
        report.counts.candidate_pairs,
        report.counts.cross_split_pairs,
        report.counts.train_test_pairs
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_release_metadata_does_not_create_path_candidates() {
        let regex = Regex::new(r"(?u)[\p{L}]{3,}").unwrap();
        let stop = stopwords();
        let a = path_token_set(
            "preprints/Some-Theorem-October-7-2026/paper.pdf",
            &regex,
            &stop,
        );
        let b = path_token_set(
            "preprints/Another-Theorem-October-7-2026/main.pdf",
            &regex,
            &stop,
        );
        assert!(!a.contains("october"));
        assert!(!a.contains("preprints"));
        assert!(!a.contains("paper"));
        assert!(!b.contains("main"));
    }

    #[test]
    fn overlapping_path_terms_do_not_get_a_confirmed_label() {
        let a = Family {
            id: "001".to_owned(),
            title: "Torus packets".to_owned(),
            split: "train".to_owned(),
            paths: vec!["preprints/Equidistribution-of-Prime-Degree-Torus-Packets.pdf".to_owned()],
            path_tokens: BTreeSet::from([
                "prime".to_owned(),
                "degree".to_owned(),
                "torus".to_owned(),
                "packets".to_owned(),
            ]),
        };
        let b = Family {
            id: "002".to_owned(),
            title: "Polynomial differences".to_owned(),
            split: "test".to_owned(),
            paths: vec!["preprints/Power-Saving-for-Prime-Degree-Arguments.pdf".to_owned()],
            path_tokens: BTreeSet::from([
                "prime".to_owned(),
                "degree".to_owned(),
                "power".to_owned(),
                "saving".to_owned(),
                "arguments".to_owned(),
            ]),
        };
        let candidates = discover(&[a, b]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].candidate_relation, "artifact_path_similarity_only");
        assert_eq!(candidates[0].candidate_status, "unreviewed");
        assert_eq!(candidates[0].source_split, "train");
        assert_eq!(candidates[0].target_split, "test");
    }

    #[test]
    fn path_candidate_order_is_deterministic() {
        let one = Candidate {
            source_family_id: "001".to_owned(),
            target_family_id: "002".to_owned(),
            source_split: "train".to_owned(),
            target_split: "test".to_owned(),
            source_title: "A".to_owned(),
            target_title: "B".to_owned(),
            source_matching_paths: vec!["a.pdf".to_owned()],
            target_matching_paths: vec!["b.pdf".to_owned()],
            shared_path_tokens: vec!["alpha".to_owned(), "beta".to_owned()],
            rare_shared_path_tokens: vec!["alpha".to_owned(), "beta".to_owned()],
            path_token_jaccard: 0.5,
            rare_token_rarity_score: 3.0,
            candidate_relation: "artifact_path_similarity_only",
            candidate_status: "unreviewed",
            evidence_sha256: "a".repeat(64),
        };
        let mut rows = vec![one];
        rows.sort_by(|a, b| {
            b.rare_token_rarity_score
                .total_cmp(&a.rare_token_rarity_score)
                .then_with(|| b.path_token_jaccard.total_cmp(&a.path_token_jaccard))
                .then_with(|| a.source_family_id.cmp(&b.source_family_id))
                .then_with(|| a.target_family_id.cmp(&b.target_family_id))
        });
        assert_eq!(rows[0].source_family_id, "001");
    }
}
