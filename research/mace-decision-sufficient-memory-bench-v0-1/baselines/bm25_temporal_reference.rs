//! Deterministic temporal/scope-aware BM25 reference baseline for MACE DSM.
//!
//! Standalone validation:
//!   rustc --edition=2024 --test research/mace-decision-sufficient-memory-bench-v0-1/baselines/bm25_temporal_reference.rs -o /tmp/mace-bm25
//!   /tmp/mace-bm25
//!
//! This is a transparent lexical comparator, not a reproduction of Resurface,
//! Khipu, InsightMem, a dense retriever, or a production Add/Search service.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25ConfigV1 {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25ConfigV1 {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDocumentV1 {
    pub document_id: String,
    pub scope_id: String,
    /// Time the candidate learned/stored this source event (transaction time).
    pub observed_at_ms: u64,
    /// Beginning of the interval during which the claim applies (valid time).
    pub valid_from_ms: u64,
    /// Exclusive upper valid-time boundary. None means no declared end.
    pub valid_until_ms: Option<u64>,
    /// Verbatim candidate-visible source text; never evaluator truth.
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetrievalQueryV1 {
    pub query_text: String,
    pub scope_id: String,
    /// Query-time cut. Both knowledge-time and valid-time must pass.
    pub as_of_ms: u64,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetrievalHitV1 {
    pub document_id: String,
    pub score: f64,
    /// One-based deterministic rank.
    pub rank: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetrievalWorkV1 {
    pub eligible_document_count: u64,
    pub postings_visited: u64,
    pub scored_document_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bm25ResultV1 {
    pub hits: Vec<RetrievalHitV1>,
    pub work: RetrievalWorkV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaselineErrorV1 {
    EmptyDocumentId,
    DuplicateDocumentId,
    EmptyDocumentScope,
    EmptyQueryScope,
    InvalidValidityInterval,
    InvalidBm25Parameters,
    EmptyChannelId,
    DuplicateChannelId,
    InvalidChannelWeight,
    InvalidRrfConstant,
    DuplicateChannelDocument,
    UnknownChannelDocument,
    IneligibleChannelDocument,
}

#[derive(Clone, Debug)]
struct PreparedDocument {
    document: SourceDocumentV1,
    term_frequencies: BTreeMap<String, u32>,
    token_count: u64,
}

#[derive(Clone, Debug)]
pub struct Bm25IndexV1 {
    documents: Vec<PreparedDocument>,
    by_id: BTreeMap<String, usize>,
    postings: BTreeMap<String, Vec<(usize, u32)>>,
    config: Bm25ConfigV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedChannelV1 {
    pub channel_id: String,
    pub weight: f64,
    /// Document IDs ordered best-first; duplicate IDs in one channel are invalid.
    pub document_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusedHitV1 {
    pub document_id: String,
    pub score: f64,
    pub rank: usize,
}

/// Whitespace/punctuation-tolerant Unicode tokenization, intentionally with no
/// stemming, stopword list, query expansion or learned vocabulary.
pub fn tokenize_v1(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() || character == '_' || character == '-' {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

impl Bm25IndexV1 {
    pub fn build(
        documents: Vec<SourceDocumentV1>,
        config: Bm25ConfigV1,
    ) -> Result<Self, BaselineErrorV1> {
        if !config.k1.is_finite()
            || config.k1 <= 0.0
            || !config.b.is_finite()
            || !(0.0..=1.0).contains(&config.b)
        {
            return Err(BaselineErrorV1::InvalidBm25Parameters);
        }

        let mut prepared = Vec::with_capacity(documents.len());
        let mut by_id = BTreeMap::new();
        let mut postings = BTreeMap::<String, Vec<(usize, u32)>>::new();

        for document in documents {
            if document.document_id.trim().is_empty() {
                return Err(BaselineErrorV1::EmptyDocumentId);
            }
            if document.scope_id.trim().is_empty() {
                return Err(BaselineErrorV1::EmptyDocumentScope);
            }
            if document
                .valid_until_ms
                .is_some_and(|end| end <= document.valid_from_ms)
            {
                return Err(BaselineErrorV1::InvalidValidityInterval);
            }
            if by_id.contains_key(&document.document_id) {
                return Err(BaselineErrorV1::DuplicateDocumentId);
            }

            let mut term_frequencies = BTreeMap::<String, u32>::new();
            for token in tokenize_v1(&document.text) {
                let count = term_frequencies.entry(token).or_default();
                *count = count.saturating_add(1);
            }
            let token_count = term_frequencies
                .values()
                .map(|count| u64::from(*count))
                .sum();
            let index = prepared.len();
            by_id.insert(document.document_id.clone(), index);
            for (term, frequency) in &term_frequencies {
                postings
                    .entry(term.clone())
                    .or_default()
                    .push((index, *frequency));
            }
            prepared.push(PreparedDocument {
                document,
                term_frequencies,
                token_count,
            });
        }

        Ok(Self {
            documents: prepared,
            by_id,
            postings,
            config,
        })
    }

    fn is_visible(&self, document_id: &str, scope_id: &str, as_of_ms: u64) -> bool {
        self.by_id
            .get(document_id)
            .and_then(|index| self.documents.get(*index))
            .is_some_and(|prepared| {
                let document = &prepared.document;
                document.scope_id == scope_id
                    && document.observed_at_ms <= as_of_ms
                    && document.valid_from_ms <= as_of_ms
                    && document.valid_until_ms.is_none_or(|end| as_of_ms < end)
            })
    }

    /// Search over documents visible in both valid-time and knowledge-time at
    /// the requested cut. Corpus statistics are recomputed over the eligible
    /// scope/time cohort, preventing excluded tenants or future facts from
    /// changing IDF or average document length for this query.
    pub fn search(
        &self,
        query: &RetrievalQueryV1,
    ) -> Result<Bm25ResultV1, BaselineErrorV1> {
        if query.scope_id.trim().is_empty() {
            return Err(BaselineErrorV1::EmptyQueryScope);
        }

        let mut query_terms = BTreeMap::<String, u32>::new();
        for token in tokenize_v1(&query.query_text) {
            let count = query_terms.entry(token).or_default();
            *count = count.saturating_add(1);
        }
        if query_terms.is_empty() || query.top_k == 0 {
            return Ok(Bm25ResultV1 {
                hits: Vec::new(),
                work: RetrievalWorkV1 {
                    eligible_document_count: self
                        .documents
                        .iter()
                        .filter(|prepared| {
                            let d = &prepared.document;
                            d.scope_id == query.scope_id
                                && d.observed_at_ms <= query.as_of_ms
                                && d.valid_from_ms <= query.as_of_ms
                                && d.valid_until_ms.is_none_or(|end| query.as_of_ms < end)
                        })
                        .count() as u64,
                    postings_visited: 0,
                    scored_document_count: 0,
                },
            });
        }

        let eligible: Vec<bool> = self
            .documents
            .iter()
            .map(|prepared| {
                let document = &prepared.document;
                document.scope_id == query.scope_id
                    && document.observed_at_ms <= query.as_of_ms
                    && document.valid_from_ms <= query.as_of_ms
                    && document.valid_until_ms.is_none_or(|end| query.as_of_ms < end)
            })
            .collect();
        let eligible_count = eligible.iter().filter(|value| **value).count();
        if eligible_count == 0 {
            return Ok(Bm25ResultV1 {
                hits: Vec::new(),
                work: RetrievalWorkV1 {
                    eligible_document_count: 0,
                    postings_visited: 0,
                    scored_document_count: 0,
                },
            });
        }

        let total_tokens: u128 = self
            .documents
            .iter()
            .zip(&eligible)
            .filter(|(_, is_eligible)| **is_eligible)
            .map(|(prepared, _)| u128::from(prepared.token_count))
            .sum();
        let average_length = total_tokens as f64 / eligible_count as f64;
        let mut scores = vec![0.0_f64; self.documents.len()];
        let mut postings_visited = 0_u64;

        if average_length > 0.0 {
            for (term, query_frequency) in query_terms {
                let Some(postings) = self.postings.get(&term) else {
                    continue;
                };
                let document_frequency = postings
                    .iter()
                    .filter(|(index, _)| eligible[*index])
                    .count();
                if document_frequency == 0 {
                    continue;
                }
                let n = eligible_count as f64;
                let df = document_frequency as f64;
                let inverse_document_frequency = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                for (index, frequency) in postings {
                    postings_visited = postings_visited.saturating_add(1);
                    if !eligible[*index] {
                        continue;
                    }
                    let tf = f64::from(*frequency);
                    let length = self.documents[*index].token_count as f64;
                    let normalization = tf
                        + self.config.k1
                            * (1.0 - self.config.b + self.config.b * length / average_length);
                    let term_score = inverse_document_frequency
                        * (tf * (self.config.k1 + 1.0) / normalization)
                        * f64::from(query_frequency);
                    scores[*index] += term_score;
                }
            }
        }

        let mut ranked: Vec<(String, f64)> = self
            .documents
            .iter()
            .enumerate()
            .filter(|(index, _)| eligible[*index] && scores[*index] > 0.0)
            .map(|(_, prepared)| {
                (
                    prepared.document.document_id.clone(),
                    scores[self.by_id[&prepared.document.document_id]],
                )
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let scored_document_count = ranked.len() as u64;
        ranked.truncate(query.top_k);
        let hits = ranked
            .into_iter()
            .enumerate()
            .map(|(index, (document_id, score))| RetrievalHitV1 {
                document_id,
                score,
                rank: index + 1,
            })
            .collect();

        Ok(Bm25ResultV1 {
            hits,
            work: RetrievalWorkV1 {
                eligible_document_count: eligible_count as u64,
                postings_visited,
                scored_document_count,
            },
        })
    }
}

/// Weighted reciprocal-rank fusion over independently produced candidate
/// channels (e.g. lexical, entity, date or graph). It does not implement those
/// channels; it validates their identities and admissibility before fusion.
pub fn weighted_rrf_v1(
    index: &Bm25IndexV1,
    scope_id: &str,
    as_of_ms: u64,
    channels: &[RankedChannelV1],
    rrf_k: u32,
    top_k: usize,
) -> Result<Vec<FusedHitV1>, BaselineErrorV1> {
    if scope_id.trim().is_empty() {
        return Err(BaselineErrorV1::EmptyQueryScope);
    }
    if rrf_k == 0 {
        return Err(BaselineErrorV1::InvalidRrfConstant);
    }
    let mut channel_ids = std::collections::BTreeSet::new();
    let mut fused = BTreeMap::<String, f64>::new();

    for channel in channels {
        if channel.channel_id.trim().is_empty() {
            return Err(BaselineErrorV1::EmptyChannelId);
        }
        if !channel_ids.insert(channel.channel_id.as_str()) {
            return Err(BaselineErrorV1::DuplicateChannelId);
        }
        if !channel.weight.is_finite() || channel.weight <= 0.0 {
            return Err(BaselineErrorV1::InvalidChannelWeight);
        }
        let mut seen_documents = std::collections::BTreeSet::new();
        for (zero_based_rank, document_id) in channel.document_ids.iter().enumerate() {
            if !seen_documents.insert(document_id.as_str()) {
                return Err(BaselineErrorV1::DuplicateChannelDocument);
            }
            let Some(index_position) = index.by_id.get(document_id) else {
                return Err(BaselineErrorV1::UnknownChannelDocument);
            };
            let document = &index.documents[*index_position].document;
            if !index.is_visible(&document.document_id, scope_id, as_of_ms) {
                return Err(BaselineErrorV1::IneligibleChannelDocument);
            }
            let one_based_rank = zero_based_rank as f64 + 1.0;
            *fused.entry(document_id.clone()).or_default() +=
                channel.weight / (f64::from(rrf_k) + one_based_rank);
        }
    }

    let mut ranked: Vec<(String, f64)> = fused.into_iter().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k);
    Ok(ranked
        .into_iter()
        .enumerate()
        .map(|(index, (document_id, score))| FusedHitV1 {
            document_id,
            score,
            rank: index + 1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(
        id: &str,
        scope: &str,
        text: &str,
        observed_at_ms: u64,
        valid_from_ms: u64,
        valid_until_ms: Option<u64>,
    ) -> SourceDocumentV1 {
        SourceDocumentV1 {
            document_id: id.to_owned(),
            scope_id: scope.to_owned(),
            observed_at_ms,
            valid_from_ms,
            valid_until_ms,
            text: text.to_owned(),
        }
    }

    fn query(text: &str, scope: &str, as_of_ms: u64, top_k: usize) -> RetrievalQueryV1 {
        RetrievalQueryV1 {
            query_text: text.to_owned(),
            scope_id: scope.to_owned(),
            as_of_ms,
            top_k,
        }
    }

    fn index(documents: Vec<SourceDocumentV1>) -> Bm25IndexV1 {
        Bm25IndexV1::build(documents, Bm25ConfigV1::default()).expect("valid BM25 index")
    }

    #[test]
    fn lexical_bm25_ranks_multi_term_evidence_above_partial_matches() {
        let index = index(vec![
            document("partial", "tenant-a", "motor inspection record", 1, 1, None),
            document(
                "complete",
                "tenant-a",
                "rocket motor inspection checklist and torque result",
                1,
                1,
                None,
            ),
            document("noise", "tenant-a", "garden irrigation schedule", 1, 1, None),
        ]);
        let result = index.search(&query("rocket motor inspection", "tenant-a", 10, 10))
            .unwrap();
        assert_eq!(result.hits[0].document_id, "complete");
        assert_eq!(result.hits[1].document_id, "partial");
        assert_eq!(result.work.scored_document_count, 2);
    }

    #[test]
    fn future_knowledge_is_not_retrieved_even_for_backdated_validity() {
        let index = index(vec![
            document("known", "tenant-a", "valve replaced", 10, 5, None),
            document("learned-later", "tenant-a", "valve replaced", 30, 5, None),
        ]);
        let result = index.search(&query("valve replaced", "tenant-a", 20, 10)).unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].document_id, "known");
    }

    #[test]
    fn valid_time_upper_bound_is_exclusive() {
        let index = index(vec![
            document("expired", "tenant-a", "calibration certificate", 1, 1, Some(20)),
            document("current", "tenant-a", "calibration certificate", 1, 20, None),
        ]);
        let result = index.search(&query("calibration certificate", "tenant-a", 20, 10)).unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].document_id, "current");
    }

    #[test]
    fn cross_scope_documents_do_not_change_in_scope_scores() {
        let base = index(vec![
            document("a", "tenant-a", "pump pressure inspection", 1, 1, None),
            document("b", "tenant-a", "pump repair log", 1, 1, None),
        ]);
        let with_other_scope = index(vec![
            document("a", "tenant-a", "pump pressure inspection", 1, 1, None),
            document("b", "tenant-a", "pump repair log", 1, 1, None),
            document("secret", "tenant-b", "pump pump pump pump pressure", 1, 1, None),
        ]);
        let q = query("pump pressure", "tenant-a", 10, 10);
        assert_eq!(base.search(&q).unwrap(), with_other_scope.search(&q).unwrap());
    }

    #[test]
    fn query_before_first_valid_or_observed_time_returns_no_hits() {
        let index = index(vec![document("future", "tenant-a", "release approved", 20, 15, None)]);
        assert!(index.search(&query("release approved", "tenant-a", 10, 10)).unwrap().hits.is_empty());
    }

    #[test]
    fn empty_query_and_zero_top_k_are_explicit_empty_results() {
        let index = index(vec![document("a", "tenant-a", "alpha beta", 1, 1, None)]);
        assert!(index.search(&query("!!!", "tenant-a", 10, 10)).unwrap().hits.is_empty());
        assert!(index.search(&query("alpha", "tenant-a", 10, 0)).unwrap().hits.is_empty());
    }

    #[test]
    fn ranking_ties_are_stable_by_document_id_not_ingest_order() {
        let docs_a = vec![
            document("z", "tenant-a", "same token", 1, 1, None),
            document("a", "tenant-a", "same token", 1, 1, None),
        ];
        let docs_b = vec![
            document("a", "tenant-a", "same token", 1, 1, None),
            document("z", "tenant-a", "same token", 1, 1, None),
        ];
        let q = query("same token", "tenant-a", 10, 10);
        assert_eq!(
            index(docs_a).search(&q).unwrap().hits,
            index(docs_b).search(&q).unwrap().hits
        );
        assert_eq!(index(vec![
            document("z", "tenant-a", "same token", 1, 1, None),
            document("a", "tenant-a", "same token", 1, 1, None),
        ]).search(&q).unwrap().hits[0].document_id, "a");
    }

    #[test]
    fn unicode_case_folding_preserves_non_ascii_terms() {
        let index = index(vec![
            document("af", "tenant-a", "Afrikaanse veiligheidsinspeksie", 1, 1, None),
            document("zu", "tenant-a", "ukuhlolwa kokuphepha", 1, 1, None),
            document("other", "tenant-a", "unrelated data", 1, 1, None),
        ]);
        assert_eq!(
            index.search(&query("AFRIKAANSE veiligheidsinspeksie", "tenant-a", 10, 10))
                .unwrap().hits[0].document_id,
            "af"
        );
        assert_eq!(
            index.search(&query("UKUHLOLWA kokuphepha", "tenant-a", 10, 10))
                .unwrap().hits[0].document_id,
            "zu"
        );
    }

    #[test]
    fn duplicate_document_ids_and_invalid_intervals_fail_closed() {
        assert_eq!(
            Bm25IndexV1::build(
                vec![
                    document("a", "tenant-a", "one", 1, 1, None),
                    document("a", "tenant-a", "two", 2, 2, None),
                ],
                Bm25ConfigV1::default(),
            ),
            Err(BaselineErrorV1::DuplicateDocumentId)
        );
        assert_eq!(
            Bm25IndexV1::build(
                vec![document("a", "tenant-a", "one", 1, 10, Some(10))],
                Bm25ConfigV1::default(),
            ),
            Err(BaselineErrorV1::InvalidValidityInterval)
        );
    }

    #[test]
    fn invalid_bm25_parameters_are_rejected() {
        assert_eq!(
            Bm25IndexV1::build(
                vec![],
                Bm25ConfigV1 { k1: 0.0, b: 0.75 },
            ),
            Err(BaselineErrorV1::InvalidBm25Parameters)
        );
        assert_eq!(
            Bm25IndexV1::build(
                vec![],
                Bm25ConfigV1 { k1: 1.2, b: 1.5 },
            ),
            Err(BaselineErrorV1::InvalidBm25Parameters)
        );
    }

    #[test]
    fn weighted_rrf_rewards_complementary_channels() {
        let index = index(vec![
            document("a", "tenant-a", "alpha", 1, 1, None),
            document("b", "tenant-a", "beta", 1, 1, None),
            document("c", "tenant-a", "gamma", 1, 1, None),
        ]);
        let fused = weighted_rrf_v1(
            &index,
            "tenant-a",
            10,
            &[
                RankedChannelV1 {
                    channel_id: "lexical".into(),
                    weight: 1.0,
                    document_ids: vec!["a".into(), "b".into()],
                },
                RankedChannelV1 {
                    channel_id: "entity".into(),
                    weight: 1.0,
                    document_ids: vec!["c".into(), "b".into()],
                },
            ],
            1,
            10,
        )
        .unwrap();
        assert_eq!(fused[0].document_id, "b");
        assert!(fused[0].score > fused[1].score);
    }

    #[test]
    fn rrf_rejects_unknown_duplicate_and_ineligible_documents() {
        let index = index(vec![
            document("a", "tenant-a", "alpha", 1, 1, None),
            document("private", "tenant-b", "alpha", 1, 1, None),
        ]);
        let bad_duplicate = vec![RankedChannelV1 {
            channel_id: "lexical".into(),
            weight: 1.0,
            document_ids: vec!["a".into(), "a".into()],
        }];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &bad_duplicate, 60, 10),
            Err(BaselineErrorV1::DuplicateChannelDocument)
        );
        let bad_scope = vec![RankedChannelV1 {
            channel_id: "entity".into(),
            weight: 1.0,
            document_ids: vec!["private".into()],
        }];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &bad_scope, 60, 10),
            Err(BaselineErrorV1::IneligibleChannelDocument)
        );
        let bad_unknown = vec![RankedChannelV1 {
            channel_id: "date".into(),
            weight: 1.0,
            document_ids: vec!["absent".into()],
        }];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &bad_unknown, 60, 10),
            Err(BaselineErrorV1::UnknownChannelDocument)
        );
    }

    #[test]
    fn rrf_rejects_duplicate_channels_invalid_weights_and_zero_constant() {
        let index = index(vec![document("a", "tenant-a", "alpha", 1, 1, None)]);
        let repeated_channel = vec![
            RankedChannelV1 { channel_id: "lexical".into(), weight: 1.0, document_ids: vec!["a".into()] },
            RankedChannelV1 { channel_id: "lexical".into(), weight: 1.0, document_ids: vec!["a".into()] },
        ];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &repeated_channel, 60, 10),
            Err(BaselineErrorV1::DuplicateChannelId)
        );
        let invalid_weight = vec![RankedChannelV1 {
            channel_id: "lexical".into(),
            weight: f64::NAN,
            document_ids: vec!["a".into()],
        }];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &invalid_weight, 60, 10),
            Err(BaselineErrorV1::InvalidChannelWeight)
        );
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 10, &[], 0, 10),
            Err(BaselineErrorV1::InvalidRrfConstant)
        );
    }

    #[test]
    fn rrf_cannot_resurrect_a_document_after_its_valid_time_ends() {
        let index = index(vec![document("expired", "tenant-a", "old approval", 1, 1, Some(20))]);
        let channels = vec![RankedChannelV1 {
            channel_id: "lexical".into(),
            weight: 1.0,
            document_ids: vec!["expired".into()],
        }];
        assert_eq!(
            weighted_rrf_v1(&index, "tenant-a", 20, &channels, 60, 10),
            Err(BaselineErrorV1::IneligibleChannelDocument)
        );
    }
}
