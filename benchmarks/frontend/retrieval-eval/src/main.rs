use anyhow::{Context, Result, bail};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::time::Instant;

const DEFAULT_QUERIES: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/retrieval-queries-v0.json";
const DEFAULT_CORPUS: &str =
    "benchmarks/frontend/knowledge/leptos-0.8/patterns-v0.json";
const DEFAULT_DIMENSIONS: usize = 16_384;
const RRF_K: f64 = 60.0;

#[derive(Clone)]
struct PatternDoc {
    id: String,
    tokens: Vec<String>,
    term_frequency: BTreeMap<String, usize>,
    hdc: Vec<i8>,
}

#[derive(Clone)]
struct Query {
    id: String,
    text: String,
    expected: HashSet<String>,
    is_positive: bool,
}

#[derive(Clone)]
struct Ranked {
    id: String,
    score: f64,
}

struct CorpusIndex {
    docs: Vec<PatternDoc>,
    document_frequency: BTreeMap<String, usize>,
    inverse_document_frequency: BTreeMap<String, f64>,
    tfidf_vectors: Vec<BTreeMap<String, f64>>,
    tfidf_norms: Vec<f64>,
    average_document_length: f64,
}

fn read_json(path: &str) -> Result<Value> {
    let input = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    serde_json::from_str(&input).with_context(|| format!("parse JSON in {path}"))
}

fn tokenize(input: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
        "how", "i", "in", "into", "is", "it", "me", "my", "of", "on", "or",
        "should", "so", "that", "the", "their", "there", "this", "to", "up",
        "use", "using", "what", "when", "where", "which", "with", "without",
        "you", "your", "leptos", "symthaea",
    ];
    let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();

    input
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .filter(|token| token.len() > 1 && !stopwords.contains(token.as_str()))
        .collect()
}

fn text_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn pattern_text(pattern: &Value) -> (String, String) {
    let title = text_field(pattern, "title");
    let mut fields = vec![
        title.clone(),
        text_field(pattern, "concept"),
        text_field(pattern, "code"),
    ];
    fields.extend(string_array(pattern, "use_when"));
    fields.extend(string_array(pattern, "avoid"));
    (title, fields.join("\n"))
}

fn term_frequency(tokens: &[String]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for token in tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }
    counts
}

/// Add word, adjacent-word and character-trigram features for the HDC encoder.
/// Character trigrams are only used by the HDC arm; lexical baselines remain
/// directly interpretable token-based implementations.
fn hdc_features(tokens: &[String]) -> Vec<String> {
    let mut features = Vec::new();

    for token in tokens {
        features.push(format!("w:{token}"));

        let padded = format!("^{token}$");
        let chars: Vec<char> = padded.chars().collect();
        for tri in chars.windows(3) {
            features.push(format!("c:{}{}{}", tri[0], tri[1], tri[2]));
        }
    }

    for pair in tokens.windows(2) {
        features.push(format!("b:{}:{}", pair[0], pair[1]));
    }

    features
}

/// Deterministic 16,384D bipolar hypervector bundling.
///
/// Each feature gets a pseudorandom ±1 vector generated with a domain-separated
/// BLAKE3 stream. Features are superposed in an integer accumulator and
/// thresholded to a bipolar vector. This is a simple, reproducible HDC baseline,
/// not a claim that this encoder is equivalent to Symthaea's full HDC memory.
fn encode_hdc(input: &str, dimensions: usize) -> Vec<i8> {
    let tokens = tokenize(input);
    let features = hdc_features(&tokens);
    let mut accumulator = vec![0_i32; dimensions];

    for feature in features {
        let mut seed_hasher = blake3::Hasher::new();
        seed_hasher.update(b"SFEB-HDC-FEATURE-v1\0");
        seed_hasher.update(feature.as_bytes());
        let seed = seed_hasher.finalize();

        for (block, chunk) in accumulator.chunks_mut(256).enumerate() {
            let mut block_hasher = blake3::Hasher::new();
            block_hasher.update(seed.as_bytes());
            block_hasher.update(&(block as u64).to_le_bytes());
            let digest = block_hasher.finalize();

            for (byte_index, byte) in digest.as_bytes().iter().enumerate() {
                for bit in 0..8 {
                    let index = byte_index * 8 + bit;
                    if index >= chunk.len() {
                        break;
                    }
                    let sign = if (byte >> bit) & 1 == 1 { 1 } else { -1 };
                    chunk[index] += sign;
                }
            }
        }
    }

    accumulator
        .into_iter()
        .map(|value| if value >= 0 { 1 } else { -1 })
        .collect()
}

fn hdc_similarity(left: &[i8], right: &[i8]) -> f64 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let dot: i64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| i64::from(*a) * i64::from(*b))
        .sum();
    dot as f64 / left.len() as f64
}

impl CorpusIndex {
    fn from_json(corpus: &Value, dimensions: usize) -> Result<Self> {
        if !(256..=65_536).contains(&dimensions) || dimensions % 256 != 0 {
            bail!("HDC dimensions must be a multiple of 256 in the supported range 256..=65536");
        }

        let patterns = corpus
            .get("patterns")
            .and_then(Value::as_array)
            .context("corpus.patterns must be an array")?;

        if patterns.len() > 10_000 {
            bail!("pattern corpus exceeds the 10,000-document safety limit");
        }

        let mut docs = Vec::new();
        let mut pattern_ids = HashSet::new();
        for pattern in patterns {
            let id = text_field(pattern, "id");
            let (title, content) = pattern_text(pattern);
            if id.trim().is_empty() || title.trim().is_empty() {
                bail!("every pattern must have a non-empty id and title");
            }
            if !pattern_ids.insert(id.clone()) {
                bail!("duplicate pattern id in corpus: {id}");
            }
            let tokens = tokenize(&content);
            if tokens.is_empty() {
                bail!("pattern {id} has no indexable terms after tokenization");
            }
            let hdc = encode_hdc(&content, dimensions);
            let term_frequency = term_frequency(&tokens);
            docs.push(PatternDoc {
                id,
                tokens,
                term_frequency,
                hdc,
            });
        }
        if docs.is_empty() {
            bail!("pattern corpus is empty");
        }

        let mut document_frequency = BTreeMap::new();
        for doc in &docs {
            for term in doc.term_frequency.keys() {
                *document_frequency.entry(term.clone()).or_insert(0) += 1;
            }
        }

        let document_count = docs.len() as f64;
        let inverse_document_frequency: BTreeMap<String, f64> = document_frequency
            .iter()
            .map(|(term, frequency)| {
                let idf = (1.0 + document_count / (1.0 + *frequency as f64)).ln() + 1.0;
                (term.clone(), idf)
            })
            .collect();

        let tfidf_vectors: Vec<BTreeMap<String, f64>> = docs
            .iter()
            .map(|doc| {
                doc.term_frequency
                    .iter()
                    .map(|(term, count)| {
                        let tf = 1.0 + (*count as f64).ln();
                        let idf = inverse_document_frequency.get(term).copied().unwrap_or(0.0);
                        (term.clone(), tf * idf)
                    })
                    .collect()
            })
            .collect();

        let tfidf_norms = tfidf_vectors
            .iter()
            .map(|vector| vector.values().map(|weight| weight * weight).sum::<f64>().sqrt())
            .collect();

        let average_document_length =
            docs.iter().map(|doc| doc.tokens.len() as f64).sum::<f64>() / document_count;

        Ok(Self {
            docs,
            document_frequency,
            inverse_document_frequency,
            tfidf_vectors,
            tfidf_norms,
            average_document_length,
        })
    }

    fn score_keyword(&self, query_tokens: &[String]) -> Vec<Ranked> {
        let query_terms: HashSet<&str> = query_tokens.iter().map(String::as_str).collect();
        let denominator = query_terms.len().max(1) as f64;

        self.docs
            .iter()
            .map(|doc| {
                let matched = query_terms
                    .iter()
                    .filter(|term| doc.term_frequency.contains_key(**term))
                    .count();
                Ranked {
                    id: doc.id.clone(),
                    score: matched as f64 / denominator,
                }
            })
            .collect()
    }

    fn score_bm25(&self, query_tokens: &[String]) -> Vec<Ranked> {
        const K1: f64 = 1.2;
        const B: f64 = 0.75;

        let query_frequency = term_frequency(query_tokens);
        self.docs
            .iter()
            .map(|doc| {
                let document_length = doc.tokens.len() as f64;
                let normalization = K1
                    * (1.0 - B
                        + B * document_length / self.average_document_length.max(1.0));
                let mut score = 0.0;

                for (term, query_count) in &query_frequency {
                    let Some(term_frequency) = doc.term_frequency.get(term) else {
                        continue;
                    };
                    let document_frequency =
                        self.document_frequency.get(term).copied().unwrap_or(0) as f64;
                    let n = self.docs.len() as f64;
                    let idf = (1.0
                        + (n - document_frequency + 0.5) / (document_frequency + 0.5))
                        .ln();
                    let tf = *term_frequency as f64;
                    let term_score = idf * (tf * (K1 + 1.0)) / (tf + normalization);
                    score += term_score * (1.0 + (*query_count as f64).ln());
                }

                Ranked {
                    id: doc.id.clone(),
                    score,
                }
            })
            .collect()
    }

    fn score_tfidf(&self, query_tokens: &[String]) -> Vec<Ranked> {
        let query_frequency = term_frequency(query_tokens);
        let query_vector: BTreeMap<String, f64> = query_frequency
            .iter()
            .map(|(term, count)| {
                let tf = 1.0 + (*count as f64).ln();
                let idf = self
                    .inverse_document_frequency
                    .get(term)
                    .copied()
                    .unwrap_or(0.0);
                (term.clone(), tf * idf)
            })
            .collect();
        let query_norm = query_vector
            .values()
            .map(|weight| weight * weight)
            .sum::<f64>()
            .sqrt();

        self.docs
            .iter()
            .enumerate()
            .map(|(index, doc)| {
                let dot = query_vector
                    .iter()
                    .map(|(term, query_weight)| {
                        query_weight
                            * self.tfidf_vectors[index]
                                .get(term)
                                .copied()
                                .unwrap_or(0.0)
                    })
                    .sum::<f64>();
                let denominator = query_norm * self.tfidf_norms[index];
                Ranked {
                    id: doc.id.clone(),
                    score: if denominator > 0.0 { dot / denominator } else { 0.0 },
                }
            })
            .collect()
    }

    fn score_hdc(&self, query: &str, dimensions: usize) -> Vec<Ranked> {
        let query_vector = encode_hdc(query, dimensions);
        self.docs
            .iter()
            .map(|doc| Ranked {
                id: doc.id.clone(),
                score: hdc_similarity(&query_vector, &doc.hdc),
            })
            .collect()
    }
}

fn sort_ranked(mut ranked: Vec<Ranked>) -> Vec<Ranked> {
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    ranked
}

fn rank_positions(ranked: &[Ranked]) -> HashMap<String, usize> {
    ranked
        .iter()
        .enumerate()
        .map(|(index, item)| (item.id.clone(), index + 1))
        .collect()
}

fn reciprocal_rank_fusion(rankings: &[Vec<Ranked>]) -> Vec<Ranked> {
    let mut fused = HashMap::<String, f64>::new();
    for ranking in rankings {
        for (index, item) in ranking.iter().enumerate() {
            *fused.entry(item.id.clone()).or_insert(0.0) += 1.0 / (RRF_K + (index + 1) as f64);
        }
    }
    sort_ranked(
        fused
            .into_iter()
            .map(|(id, score)| Ranked { id, score })
            .collect(),
    )
}

fn load_queries(data: &Value) -> Result<Vec<Query>> {
    let entries = data
        .get("queries")
        .and_then(Value::as_array)
        .context("retrieval.queries must be an array")?;
    let mut queries = Vec::new();
    let mut ids = HashSet::new();

    for entry in entries {
        let id = text_field(entry, "id");
        let text = text_field(entry, "query");
        if id.is_empty() || text.is_empty() || !ids.insert(id.clone()) {
            bail!("retrieval query ids and text must be non-empty and IDs unique");
        }
        let kind = text_field(entry, "kind");
        let expected_values = entry
            .get("expected_pattern_ids")
            .and_then(Value::as_array)
            .context("each retrieval query requires expected_pattern_ids")?;
        let expected: HashSet<String> = expected_values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        let is_positive = kind == "positive";
        if is_positive == expected.is_empty() {
            bail!("query {id} has inconsistent kind and expected pattern set");
        }
        if !is_positive && kind != "hard_negative" {
            bail!("query {id} has unsupported kind {kind}");
        }
        queries.push(Query {
            id,
            text,
            expected,
            is_positive,
        });
    }
    if queries.is_empty() {
        bail!("retrieval query set is empty");
    }
    Ok(queries)
}

fn summarize(
    method: &str,
    queries: &[Query],
    rankings: &[Vec<Ranked>],
    latency_ms: &[f64],
) -> Value {
    let mut positive_count = 0usize;
    let mut hit_at_1 = 0usize;
    let mut hit_at_3 = 0usize;
    let mut recall_at_1_sum = 0.0;
    let mut recall_at_3_sum = 0.0;
    let mut reciprocal_rank_sum = 0.0;
    let mut ndcg_at_3_sum = 0.0;
    let mut precision_at_3_sum = 0.0;
    let mut negative_top_scores = Vec::new();
    let mut positive_top_scores = Vec::new();
    let mut details = Vec::new();

    for (query, ranking) in queries.iter().zip(rankings) {
        if query.is_positive {
            positive_count += 1;
            let positions = rank_positions(ranking);
            let mut relevant_positions: Vec<usize> = query
                .expected
                .iter()
                .filter_map(|pattern_id| positions.get(pattern_id).copied())
                .collect();
            relevant_positions.sort_unstable();

            let relevant_at_1 = relevant_positions.iter().filter(|rank| **rank <= 1).count();
            let relevant_at_3 = relevant_positions.iter().filter(|rank| **rank <= 3).count();
            let relevant_total = query.expected.len().max(1) as f64;

            recall_at_1_sum += relevant_at_1 as f64 / relevant_total;
            recall_at_3_sum += relevant_at_3 as f64 / relevant_total;
            if relevant_at_1 > 0 {
                hit_at_1 += 1;
            }
            if relevant_at_3 > 0 {
                hit_at_3 += 1;
            }
            if let Some(rank) = relevant_positions.first().copied() {
                reciprocal_rank_sum += 1.0 / rank as f64;
            }

            precision_at_3_sum += relevant_at_3 as f64 / 3.0;

            let dcg = relevant_positions
                .iter()
                .filter(|rank| **rank <= 3)
                .map(|rank| 1.0 / ((*rank as f64 + 1.0).log2()))
                .sum::<f64>();
            let ideal_relevant = query.expected.len().min(3);
            let ideal_dcg = (1..=ideal_relevant)
                .map(|rank| 1.0 / ((rank as f64 + 1.0).log2()))
                .sum::<f64>();
            if ideal_dcg > 0.0 {
                ndcg_at_3_sum += dcg / ideal_dcg;
            }

            if let Some(top) = ranking.first() {
                positive_top_scores.push(top.score);
            }
        } else if let Some(top) = ranking.first() {
            negative_top_scores.push(top.score);
        }

        let mut expected_pattern_ids: Vec<&str> =
            query.expected.iter().map(String::as_str).collect();
        expected_pattern_ids.sort_unstable();
        details.push(json!({
            "query_id": query.id,
            "kind": if query.is_positive { "positive" } else { "hard_negative" },
            "expected_pattern_ids": expected_pattern_ids,
            "top_5": ranking.iter().take(5).map(|item| json!({
                "pattern_id": item.id,
                "score": item.score
            })).collect::<Vec<_>>()
        }));
    }

    let n = positive_count.max(1) as f64;
    let mut negative_sorted = negative_top_scores.clone();
    negative_sorted.sort_by(f64::total_cmp);
    let mut positive_sorted = positive_top_scores.clone();
    positive_sorted.sort_by(f64::total_cmp);

    json!({
        "method": method,
        "positive_queries": positive_count,
        "hard_negative_queries": negative_top_scores.len(),
        "recall_at_1": recall_at_1_sum / n,
        "recall_at_3": recall_at_3_sum / n,
        "hit_rate_at_1": hit_at_1 as f64 / n,
        "hit_rate_at_3": hit_at_3 as f64 / n,
        "mean_reciprocal_rank": reciprocal_rank_sum / n,
        "ndcg_at_3": ndcg_at_3_sum / n,
        "precision_at_3": precision_at_3_sum / n,
        "abstention_threshold": null,
        "hard_negative_top_score_distribution": score_distribution(&negative_sorted),
        "positive_top_score_distribution": score_distribution(&positive_sorted),
        "latency_ms_per_query": latency_distribution(latency_ms),
        "per_query": details
    })
}

fn score_distribution(sorted: &[f64]) -> Value {
    if sorted.is_empty() {
        return Value::Null;
    }
    let percentile = |p: f64| -> f64 {
        let index = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[index]
    };
    json!({
        "min": sorted[0],
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "max": sorted[sorted.len() - 1]
    })
}

fn latency_distribution(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| -> f64 {
        let index = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[index]
    };
    json!({
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64
    })
}

fn evaluate_method(
    name: &str,
    queries: &[Query],
    index: &CorpusIndex,
    dimensions: usize,
) -> Value {
    let mut rankings = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());

    for query in queries {
        let started = Instant::now();
        let tokens = tokenize(&query.text);
        let ranked = match name {
            "keyword_overlap" => index.score_keyword(&tokens),
            "bm25" => index.score_bm25(&tokens),
            "tfidf" => index.score_tfidf(&tokens),
            "hdc_bipolar" => index.score_hdc(&query.text, dimensions),
            _ => unreachable!("method is selected by the evaluator"),
        };
        let ranking = sort_ranked(ranked);
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        rankings.push(ranking);
    }

    summarize(name, queries, &rankings, &latencies)
}

fn evaluate_hybrid(queries: &[Query], index: &CorpusIndex, dimensions: usize) -> Value {
    let mut rankings = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());

    for query in queries {
        let started = Instant::now();
        let tokens = tokenize(&query.text);
        let bm25 = sort_ranked(index.score_bm25(&tokens));
        let tfidf = sort_ranked(index.score_tfidf(&tokens));
        let hdc = sort_ranked(index.score_hdc(&query.text, dimensions));
        let ranking = reciprocal_rank_fusion(&[bm25, tfidf, hdc]);
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
        rankings.push(ranking);
    }

    summarize("hybrid_rrf_bm25_tfidf_hdc", queries, &rankings, &latencies)
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage: cargo run --manifest-path benchmarks/frontend/retrieval-eval/Cargo.toml -- [QUERY_SET.json [PATTERN_CORPUS.json [HDC_DIMENSIONS]]]\n\nDefaults run keyword overlap, BM25, TF-IDF, deterministic bipolar HDC and reciprocal-rank fusion on the checked-in Leptos pilot set."
        );
        return Ok(());
    }
    if args.len() > 3 {
        bail!("expected at most three arguments; use --help");
    }

    let queries_path = args.first().map(String::as_str).unwrap_or(DEFAULT_QUERIES);
    let corpus_path = args.get(1).map(String::as_str).unwrap_or(DEFAULT_CORPUS);
    let dimensions = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("HDC dimensions must be an integer")?
        .unwrap_or(DEFAULT_DIMENSIONS);

    let corpus_json = read_json(corpus_path)?;
    let query_json = read_json(queries_path)?;
    let corpus = CorpusIndex::from_json(&corpus_json, dimensions)?;
    let queries = load_queries(&query_json)?;

    let run_started = Instant::now();
    let mut methods = Vec::new();
    for method in ["keyword_overlap", "bm25", "tfidf", "hdc_bipolar"] {
        methods.push(evaluate_method(method, &queries, &corpus, dimensions));
    }
    methods.push(evaluate_hybrid(&queries, &corpus, dimensions));

    let result = json!({
        "schema": "sfeb-pattern-retrieval-result-v1",
        "status": "pilot_measurement",
        "corpus_version": corpus_json.get("corpus_version"),
        "corpus_source_commit": corpus_json["framework"]["source_commit"],
        "queries_source": queries_path,
        "query_count": queries.len(),
        "positive_queries": queries.iter().filter(|q| q.is_positive).count(),
        "hard_negative_queries": queries.iter().filter(|q| !q.is_positive).count(),
        "hdc_dimensions": dimensions,
        "hdc_encoding": "deterministic BLAKE3-seeded bipolar word/bigram/character-trigram bundling",
        "threshold_calibration": "not performed; no abstention threshold or accuracy is reported",
        "methods": methods,
        "total_runtime_ms": run_started.elapsed().as_secs_f64() * 1000.0,
        "important_limitations": [
            "The HDC encoder is a simple deterministic associative baseline, not Symthaea's full learned or cognitive HDC memory.",
            "The query set is public pilot data; results cannot support hidden-holdout or contamination-resistant claims.",
            "Dense embeddings are not implemented in this offline runner and require an explicitly pinned encoder adapter.",
            "No benchmark metrics should be treated as SOTA comparisons without matched task, corpus, hardware and evaluator protocols."
        ]
    });

    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sfeb-retrieval-eval: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenization_is_case_insensitive_and_discards_stopwords() {
        assert_eq!(
            tokenize("How do I use Leptos Signals with Symthaea?"),
            vec!["signals"]
        );
    }

    #[test]
    fn hdc_encoding_is_deterministic_and_dimension_bounded() {
        let first = encode_hdc("reactive signal state transition", 256);
        let second = encode_hdc("reactive signal state transition", 256);
        assert_eq!(first, second);
        assert_eq!(first.len(), 256);
        assert!(first.iter().all(|value| *value == -1 || *value == 1));
    }

    #[test]
    fn hdc_similarity_is_one_for_identical_vectors() {
        let vector = encode_hdc("typed component property", 512);
        assert_eq!(hdc_similarity(&vector, &vector), 1.0);
    }

    #[test]
    fn ranking_has_stable_tie_breaking() {
        let ranking = sort_ranked(vec![
            Ranked { id: "b".into(), score: 0.5 },
            Ranked { id: "a".into(), score: 0.5 },
        ]);
        assert_eq!(ranking[0].id, "a");
        assert_eq!(ranking[1].id, "b");
    }

    #[test]
    fn reciprocal_rank_fusion_uses_rank_not_raw_score_scales() {
        let a = vec![
            Ranked { id: "p1".into(), score: 9.0 },
            Ranked { id: "p2".into(), score: 1.0 },
        ];
        let b = vec![
            Ranked { id: "p2".into(), score: 0.99 },
            Ranked { id: "p1".into(), score: 0.10 },
        ];
        let fused = reciprocal_rank_fusion(&[a, b]);
        assert!(fused.iter().any(|item| item.id == "p1"));
        assert!(fused.iter().any(|item| item.id == "p2"));
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn precision_and_ndcg_support_multiple_relevant_patterns() {
        let queries = vec![Query {
            id: "q-multi".into(),
            text: "related patterns".into(),
            expected: HashSet::from(["p1".to_string(), "p2".to_string()]),
            is_positive: true,
        }];
        let rankings = vec![vec![
            Ranked { id: "p1".into(), score: 1.0 },
            Ranked { id: "p2".into(), score: 0.9 },
            Ranked { id: "p3".into(), score: 0.8 },
        ]];
        let summary = summarize("test", &queries, &rankings, &[0.1]);

        assert_eq!(summary["recall_at_1"], 1.0);
        assert_eq!(summary["recall_at_3"], 1.0);
        assert!((summary["precision_at_3"].as_f64().unwrap() - (2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(summary["ndcg_at_3"], 1.0);
    }

    #[test]
    fn recall_and_hit_rate_are_distinct_for_multi_relevant_queries() {
        let queries = vec![Query {
            id: "q-partial".into(),
            text: "related patterns".into(),
            expected: HashSet::from(["p1".to_string(), "p2".to_string()]),
            is_positive: true,
        }];
        let rankings = vec![vec![
            Ranked { id: "p1".into(), score: 1.0 },
            Ranked { id: "p3".into(), score: 0.9 },
            Ranked { id: "p4".into(), score: 0.8 },
        ]];
        let summary = summarize("test", &queries, &rankings, &[0.1]);

        assert_eq!(summary["recall_at_1"], 0.5);
        assert_eq!(summary["recall_at_3"], 0.5);
        assert_eq!(summary["hit_rate_at_1"], 1.0);
        assert_eq!(summary["hit_rate_at_3"], 1.0);
    }

    #[test]
    fn bm25_and_tfidf_return_finite_scores() {
        let corpus = json!({
            "patterns": [
                {"id":"p1","title":"Reactive signals","concept":"Reactive signal state","code":"RwSignal::new(1)","use_when":["shared component state"],"avoid":[]},
                {"id":"p2","title":"Router navigation","concept":"Route based application navigation","code":"<Router/>","use_when":["multi page applications"],"avoid":[]}
            ]
        });
        let index = CorpusIndex::from_json(&corpus, 256).unwrap();
        let tokens = tokenize("reactive component state");
        for ranking in [
            index.score_bm25(&tokens),
            index.score_tfidf(&tokens),
            index.score_keyword(&tokens),
        ] {
            assert_eq!(ranking.len(), 2);
            assert!(ranking.iter().all(|item| item.score.is_finite()));
        }
    }

    #[test]
    fn corpus_rejects_duplicate_pattern_ids() {
        let corpus = json!({
            "patterns": [
                {"id":"duplicate","title":"First pattern","concept":"reactive state"},
                {"id":"duplicate","title":"Second pattern","concept":"router navigation"}
            ]
        });
        assert!(CorpusIndex::from_json(&corpus, 256).is_err());
    }

    #[test]
    fn corpus_rejects_empty_indexable_pattern_text() {
        let corpus = json!({
            "patterns": [
                {"id":"empty","title":"The and a","concept":"is it"}
            ]
        });
        assert!(CorpusIndex::from_json(&corpus, 256).is_err());
    }

    #[test]
    fn hdc_dimensions_are_bounded_to_supported_ablation_range() {
        let corpus = json!({
            "patterns": [
                {"id":"p1","title":"Reactive signals","concept":"reactive state"}
            ]
        });
        assert!(CorpusIndex::from_json(&corpus, 0).is_err());
        assert!(CorpusIndex::from_json(&corpus, 65_792).is_err());
        assert!(CorpusIndex::from_json(&corpus, 16_384).is_ok());
    }

    #[test]
    fn hybrid_ranking_is_deterministic() {
        let corpus = json!({
            "patterns": [
                {"id":"p1","title":"Reactive signals","concept":"Reactive signal state","code":"RwSignal::new(1)","use_when":["shared component state"],"avoid":[]},
                {"id":"p2","title":"Router navigation","concept":"Route based application navigation","code":"<Router/>","use_when":["multi page applications"],"avoid":[]}
            ]
        });
        let queries = vec![Query {
            id: "q1".into(),
            text: "shared reactive state".into(),
            expected: HashSet::from(["p1".to_string()]),
            is_positive: true,
        }];
        let index = CorpusIndex::from_json(&corpus, 256).unwrap();
        let first = evaluate_hybrid(&queries, &index, 256);
        let second = evaluate_hybrid(&queries, &index, 256);
        assert_eq!(first["per_query"][0]["top_5"], second["per_query"][0]["top_5"]);
    }
}
