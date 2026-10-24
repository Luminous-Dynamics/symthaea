//! Deterministically discover *candidate* cross-family relationships from titles.
//!
//! Candidates are deliberately NOT hard-leakage judgments. They must be
//! independently reviewed before they can affect the frozen split or be added
//! as confirmed dependency edges to the family-leakage inventory.

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
const EXPECTED_FAMILY_COUNTS: (usize, usize, usize) = (260, 56, 56);
const RARE_TOKEN_MAX_DOCUMENT_FREQUENCY: usize = 15;
const MIN_RARE_SHARED_TOKENS: usize = 2;
const MIN_JACCARD: f64 = 0.16;
const MIN_RARITY_SCORE: f64 = 4.0;

#[derive(Debug, Clone)]
struct Family {
    id: String,
    title: String,
    split: String,
    tokens: BTreeSet<String>,
}

#[derive(Debug, Serialize)]
struct Candidate {
    source_family_id: String,
    target_family_id: String,
    source_split: String,
    target_split: String,
    source_title: String,
    target_title: String,
    shared_tokens: Vec<String>,
    rare_shared_tokens: Vec<String>,
    title_token_jaccard: f64,
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
struct DiscoveryConfig {
    tokenizer: &'static str,
    stopword_policy: &'static str,
    rare_token_max_document_frequency: usize,
    min_rare_shared_tokens: usize,
    min_jaccard: f64,
    min_rarity_score: f64,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct DiscoveryReport {
    schema_version: &'static str,
    upstream_repository: &'static str,
    upstream_commit: &'static str,
    source_partition_digest_sha256: String,
    total_family_count: usize,
    original_family_counts: SplitCounts,
    config: DiscoveryConfig,
    counts: CandidateCounts,
    candidates: Vec<Candidate>,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn token_set(
    title: &str,
    tag_re: &Regex,
    entity_re: &Regex,
    token_re: &Regex,
    stop: &BTreeSet<&str>,
) -> BTreeSet<String> {
    let no_tags = tag_re.replace_all(title, " ");
    let no_entities = entity_re.replace_all(&no_tags, " ");
    token_re
        .find_iter(&no_entities.to_lowercase())
        .map(|m| m.as_str().to_owned())
        .filter(|token| !stop.contains(token.as_str()))
        .collect()
}

fn stopwords() -> BTreeSet<&'static str> {
    concat!(
        "the a an and or nor but of on in to for from with by at as is are was were be been being ",
        "not does do did every each any all some over under into beyond ",
        "across between within without against after before through during ",
        "result results family families conjecture conjectures theorem theorems ",
        "lemma lemmas proof proofs bound bounds formula formulas problem problems ",
        "new related general using its their this that"
    )
    .split_whitespace()
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
        .map(|id| {
            (
                sha256(format!("{SPLIT_SEED}\0{id}").as_bytes()),
                id.clone(),
            )
        })
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

fn validate_catalogue(doc: &Value) -> Result<(String, Vec<Family>, SplitCounts)> {
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
        let id = row["family_id"].as_str().context("family_id missing")?.to_owned();
        let split = row["split"].as_str().context("split missing")?.to_owned();
        let _title = row["title"].as_str().context("title missing")?;
        if !matches!(split.as_str(), "train" | "validation" | "test")
            || id.trim().is_empty()
            || splits.insert(id, split).is_some()
        {
            bail!("invalid or duplicate family record");
        }
    }

    if splits != deterministic_splits(&splits.keys().cloned().collect::<Vec<_>>())? {
        bail!("family assignments differ from the registered deterministic split");
    }
    let digest = partition_digest(&splits);
    if digest != EXPECTED_PARTITION_DIGEST
        || plan["partition_digest_sha256"].as_str() != Some(digest.as_str())
        || splits.len() != 372
    {
        bail!("catalogue does not match the exact frozen partition digest and size");
    }

    let mut counts = SplitCounts {
        train: 0,
        validation: 0,
        test: 0,
    };
    for split in splits.values() {
        match split.as_str() {
            "train" => counts.train += 1,
            "validation" => counts.validation += 1,
            "test" => counts.test += 1,
            _ => unreachable!("split was validated"),
        }
    }
    if (counts.train, counts.validation, counts.test) != EXPECTED_FAMILY_COUNTS {
        bail!("catalogue split counts differ from the frozen PITB v0 counts");
    }
    for (name, count) in [
        ("train", counts.train),
        ("validation", counts.validation),
        ("test", counts.test),
    ] {
        if plan["counts"][name].as_u64() != Some(count as u64) {
            bail!("declared {name} count differs from actual family rows");
        }
    }

    let tag_re = Regex::new(r"<[^>]*>").context("compile HTML-tag tokenizer")?;
    let entity_re = Regex::new(r"&[A-Za-z]+;").context("compile entity tokenizer")?;
    let token_re = Regex::new(r"(?u)[\p{L}\p{N}]{3,}").context("compile Unicode tokenizer")?;
    let stop = stopwords();
    let mut families = Vec::with_capacity(rows.len());
    for row in rows {
        let id = row["family_id"].as_str().unwrap().to_owned();
        let title = row["title"].as_str().unwrap().to_owned();
        let split = row["split"].as_str().unwrap().to_owned();
        let tokens = token_set(&title, &tag_re, &entity_re, &token_re, &stop);
        if tokens.is_empty() {
            bail!("family {id} has no meaningful title tokens");
        }
        families.push(Family {
            id,
            title,
            split,
            tokens,
        });
    }
    families.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((digest, families, counts))
}

fn discover(families: &[Family]) -> Vec<Candidate> {
    let mut document_frequency: BTreeMap<String, usize> = BTreeMap::new();
    for family in families {
        for token in &family.tokens {
            *document_frequency.entry(token.clone()).or_default() += 1;
        }
    }

    let mut candidates = Vec::new();
    for left in 0..families.len() {
        for right in (left + 1)..families.len() {
            let a = &families[left];
            let b = &families[right];
            let shared: Vec<String> = a
                .tokens
                .intersection(&b.tokens)
                .cloned()
                .collect();
            let rare: Vec<String> = shared
                .iter()
                .filter(|token| {
                    document_frequency.get(*token).copied().unwrap_or(usize::MAX)
                        <= RARE_TOKEN_MAX_DOCUMENT_FREQUENCY
                })
                .cloned()
                .collect();
            if rare.len() < MIN_RARE_SHARED_TOKENS {
                continue;
            }
            let union_count = a.tokens.union(&b.tokens).count();
            if union_count == 0 {
                continue;
            }
            let jaccard = shared.len() as f64 / union_count as f64;
            let rarity = rare
                .iter()
                .map(|token| {
                    let frequency = document_frequency
                        .get(token)
                        .copied()
                        .unwrap_or(usize::MAX);
                    ((families.len() + 1) as f64 / (frequency + 1) as f64).ln()
                })
                .sum::<f64>();
            if jaccard < MIN_JACCARD && rarity < MIN_RARITY_SCORE {
                continue;
            }

            let canonical_evidence = format!(
                "{}\n{}\n{}\n{}\n{}\n{:.6}\n{:.6}",
                a.id,
                b.id,
                a.title,
                b.title,
                rare.join("\t"),
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
                shared_tokens: shared,
                rare_shared_tokens: rare,
                title_token_jaccard: (jaccard * 1000.0).round() / 1000.0,
                rare_token_rarity_score: (rarity * 100.0).round() / 100.0,
                candidate_relation: "title_similarity_only",
                candidate_status: "unreviewed",
                evidence_sha256: sha256(canonical_evidence.as_bytes()),
            });
        }
    }
    candidates.sort_by(|a, b| {
        b.rare_token_rarity_score
            .total_cmp(&a.rare_token_rarity_score)
            .then_with(|| {
                b.title_token_jaccard
                    .total_cmp(&a.title_token_jaccard)
            })
            .then_with(|| a.source_family_id.cmp(&b.source_family_id))
            .then_with(|| a.target_family_id.cmp(&b.target_family_id))
    });
    candidates
}

fn counts(candidates: &[Candidate]) -> CandidateCounts {
    CandidateCounts {
        candidate_pairs: candidates.len(),
        cross_split_pairs: candidates
            .iter()
            .filter(|c| c.source_split != c.target_split)
            .count(),
        train_test_pairs: candidates.iter().filter(|c| {
            (c.source_split == "train" && c.target_split == "test")
                || (c.source_split == "test" && c.target_split == "train")
        }).count(),
    }
}

fn make_report(
    digest: String,
    families: &[Family],
    split_counts: SplitCounts,
    candidates: Vec<Candidate>,
) -> DiscoveryReport {
    let counts = counts(&candidates);
    DiscoveryReport {
        schema_version: "pitb.family-leakage-candidate-discovery.v0",
        upstream_repository: "https://github.com/openai/math",
        upstream_commit: UPSTREAM_COMMIT,
        source_partition_digest_sha256: digest,
        total_family_count: families.len(),
        original_family_counts: split_counts,
        config: DiscoveryConfig {
            tokenizer: "lowercase Unicode letter/number tokens; strip HTML tags and HTML entities",
            stopword_policy: "fixed explicit stopword list compiled into this tool",
            rare_token_max_document_frequency: RARE_TOKEN_MAX_DOCUMENT_FREQUENCY,
            min_rare_shared_tokens: MIN_RARE_SHARED_TOKENS,
            min_jaccard: MIN_JACCARD,
            min_rarity_score: MIN_RARITY_SCORE,
            note: concat!(
                "title-only lexical candidate generation; never a hard-leakage or ",
                "mathematical-dependency decision",
            ),
        },
        counts,
        candidates,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        bail!(concat!(
            "usage: discover_family_leakage_candidates <frozen-splits.json> ",
            "<candidate-report.json>"
        ));
    }
    let doc: Value = serde_json::from_slice(
        &fs::read(&args[0]).with_context(|| format!("cannot read {}", args[0]))?,
    )
    .context("frozen split manifest is not valid JSON")?;
    let (digest, families, split_counts) = validate_catalogue(&doc)?;
    let candidates = discover(&families);
    let report = make_report(digest, &families, split_counts, candidates);
    fs::write(&args[1], serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("cannot write {}", args[1]))?;
    println!(
        concat!(
            "PITB_FAMILY_LEAKAGE_CANDIDATES: pairs={} cross_split={} ",
            "train_test={} relation=title_similarity_only"
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

    fn family(id: &str, title: &str, split: &str) -> Family {
        Family {
            id: id.to_owned(),
            title: title.to_owned(),
            split: split.to_owned(),
            tokens: BTreeSet::new(),
        }
    }

    #[test]
    fn discover_marks_similarity_as_unreviewed_and_never_as_hard_dependency() {
        let tag = Regex::new(r"<[^>]*>").unwrap();
        let entity = Regex::new(r"&[A-Za-z]+;").unwrap();
        let token = Regex::new(r"(?u)[\p{L}\p{N}]{3,}").unwrap();
        let stop = stopwords();
        let mut a = family(
            "001",
            "Iitaka subadditivity and logarithmic additivity",
            "train",
        );
        let mut b = family(
            "002",
            "Log abundance under logarithmic Iitaka subadditivity",
            "test",
        );
        a.tokens = token_set(&a.title, &tag, &entity, &token, &stop);
        b.tokens = token_set(&b.title, &tag, &entity, &token, &stop);
        let candidates = discover(&[a, b]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].candidate_relation, "title_similarity_only");
        assert_eq!(candidates[0].candidate_status, "unreviewed");
        assert_eq!(candidates[0].source_split, "train");
        assert_eq!(candidates[0].target_split, "test");
        assert!(valid_digest_for_test(&candidates[0].evidence_sha256));
    }

    fn valid_digest_for_test(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
    }

    #[test]
    fn title_tokenization_removes_markup_and_common_stopwords() {
        let tag = Regex::new(r"<[^>]*>").unwrap();
        let entity = Regex::new(r"&[A-Za-z]+;").unwrap();
        let token = Regex::new(r"(?u)[\p{L}\p{N}]{3,}").unwrap();
        let stop = stopwords();
        let tokens = token_set(
            "The <i>log</i> abundance &amp; theorem",
            &tag,
            &entity,
            &token,
            &stop,
        );
        assert!(tokens.contains("log"));
        assert!(tokens.contains("abundance"));
        assert!(!tokens.contains("the"));
        assert!(!tokens.contains("amp"));
    }

    #[test]
    fn candidate_order_is_deterministic() {
        let tag = Regex::new(r"<[^>]*>").unwrap();
        let entity = Regex::new(r"&[A-Za-z]+;").unwrap();
        let token = Regex::new(r"(?u)[\p{L}\p{N}]{3,}").unwrap();
        let stop = stopwords();
        let mut a = family("001", "Iitaka subadditivity and logarithmic additivity", "train");
        let mut b = family("002", "Log abundance under logarithmic Iitaka subadditivity", "test");
        a.tokens = token_set(&a.title, &tag, &entity, &token, &stop);
        b.tokens = token_set(&b.title, &tag, &entity, &token, &stop);
        let first = discover(&[a.clone(), b.clone()]);
        let second = discover(&[a, b]);
        assert_eq!(
            serde_json::to_value(first).unwrap(),
            serde_json::to_value(second).unwrap()
        );
    }
}
