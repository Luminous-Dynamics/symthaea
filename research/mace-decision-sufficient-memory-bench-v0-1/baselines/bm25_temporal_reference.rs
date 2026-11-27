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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityTierV1 {
    Verified,
    Authenticated,
    UserAsserted,
    Untrusted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventRelationV1 {
    Asserts,
    Supersedes,
    Retracts,
    Contradicts,
    DerivedFrom,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDocumentV1 {
    /// Exact candidate-visible event ID; this is the join key used by MACE decision receipts.
    pub document_id: String,
    pub source_id: String,
    pub scope_id: String,
    pub authority_tier: AuthorityTierV1,
    pub uncertainty_ppm: u32,
    pub relation_to_parent: EventRelationV1,
    pub parent_event_ids: Vec<String>,
    pub source_family: Option<String>,
    pub source_root_digest_sha256: Option<String>,
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
    /// Posting rows visited inside the requested scope and knowledge-time cut.
    pub scope_knowledge_postings_visited: u64,
    /// Posting rows that also pass valid-time filtering and contribute to scores.
    pub eligible_postings_scored: u64,
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
    EmptySourceId,
    EmptyDocumentScope,
    InvalidUncertainty,
    InvalidSourceFamily,
    InvalidSourceRootDigest,
    InvalidParentEvent,
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

#[derive(Clone, Debug, PartialEq)]
struct PreparedDocument {
    document: SourceDocumentV1,
    token_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bm25IndexV1 {
    documents: Vec<PreparedDocument>,
    by_id: BTreeMap<String, usize>,
    /// Document indexes partitioned by scope and sorted by knowledge time.
    scope_documents: BTreeMap<String, Vec<usize>>,
    /// Term postings are partitioned by scope and sorted by knowledge time.
    postings: BTreeMap<(String, String), Vec<(usize, u32)>>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineageEdgeStatusV1 {
    Resolved,
    MissingParent,
    CrossScopeParent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageEdgeV1 {
    pub child_event_id: String,
    pub parent_event_id: String,
    pub status: LineageEdgeStatusV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageAuditReportV1 {
    pub edges: Vec<LineageEdgeV1>,
    /// Strongly connected components containing a directed lineage cycle.
    /// Each component is sorted by event ID; components are sorted lexicographically.
    pub cycle_components: Vec<Vec<String>>,
    pub resolved_edge_count: u64,
    pub missing_parent_edge_count: u64,
    pub cross_scope_edge_count: u64,
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
        let mut scope_documents = BTreeMap::<String, Vec<usize>>::new();
        let mut postings = BTreeMap::<(String, String), Vec<(usize, u32)>>::new();

        for document in documents {
            if document.document_id.trim().is_empty() {
                return Err(BaselineErrorV1::EmptyDocumentId);
            }
            if document.source_id.trim().is_empty() {
                return Err(BaselineErrorV1::EmptySourceId);
            }
            if document.scope_id.trim().is_empty() {
                return Err(BaselineErrorV1::EmptyDocumentScope);
            }
            if document.uncertainty_ppm > 1_000_000 {
                return Err(BaselineErrorV1::InvalidUncertainty);
            }
            if document
                .source_family
                .as_ref()
                .is_some_and(|family| family.trim().is_empty())
            {
                return Err(BaselineErrorV1::InvalidSourceFamily);
            }
            if document
                .source_root_digest_sha256
                .as_ref()
                .is_some_and(|root| {
                    root.len() != 64
                        || !root
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                || (document.source_root_digest_sha256.is_some()
                    && document.source_family.is_none())
            {
                return Err(BaselineErrorV1::InvalidSourceRootDigest);
            }
            let mut parent_ids = std::collections::BTreeSet::new();
            for parent_id in &document.parent_event_ids {
                if parent_id.trim().is_empty()
                    || parent_id == &document.document_id
                    || !parent_ids.insert(parent_id.as_str())
                {
                    return Err(BaselineErrorV1::InvalidParentEvent);
                }
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
            scope_documents
                .entry(document.scope_id.clone())
                .or_default()
                .push(index);
            for (term, frequency) in &term_frequencies {
                postings
                    .entry((document.scope_id.clone(), term.clone()))
                    .or_default()
                    .push((index, *frequency));
            }
            prepared.push(PreparedDocument {
                document,
                token_count,
            });
        }

        for indexes in scope_documents.values_mut() {
            indexes.sort_by_key(|index| prepared[*index].document.observed_at_ms);
        }
        for rows in postings.values_mut() {
            rows.sort_by_key(|(index, _)| prepared[*index].document.observed_at_ms);
        }

        Ok(Self {
            documents: prepared,
            by_id,
            scope_documents,
            postings,
            config,
        })
    }

    /// Resolve a retrieved event ID back to the exact canonical candidate-visible source record.
    /// This lets the caller join BM25 hits to MACE provenance without text-similarity matching.
    pub fn resolve_source_event(&self, document_id: &str) -> Option<&SourceDocumentV1> {
        self.by_id
            .get(document_id)
            .and_then(|index| self.documents.get(*index))
            .map(|prepared| &prepared.document)
    }

    /// Audit candidate-archive lineage by exact event ID.
    ///
    /// This is diagnostic metadata, not a trust decision: a resolved edge does not
    /// grant the child its parent's authority, freshness, validity or admissibility.
    /// Cross-scope references are reported rather than silently traversed.
    pub fn audit_lineage(&self) -> LineageAuditReportV1 {
        let mut edges = Vec::new();
        for prepared in &self.documents {
            let child = &prepared.document;
            for parent_event_id in &child.parent_event_ids {
                let status = match self.by_id.get(parent_event_id) {
                    None => LineageEdgeStatusV1::MissingParent,
                    Some(parent_index) => {
                        let parent = &self.documents[*parent_index].document;
                        if parent.scope_id == child.scope_id {
                            LineageEdgeStatusV1::Resolved
                        } else {
                            LineageEdgeStatusV1::CrossScopeParent
                        }
                    }
                };
                edges.push(LineageEdgeV1 {
                    child_event_id: child.document_id.clone(),
                    parent_event_id: parent_event_id.clone(),
                    status,
                });
            }
        }
        edges.sort_by(|left, right| {
            left.child_event_id
                .cmp(&right.child_event_id)
                .then_with(|| left.parent_event_id.cmp(&right.parent_event_id))
        });

        let resolved_edge_count = edges
            .iter()
            .filter(|edge| edge.status == LineageEdgeStatusV1::Resolved)
            .count() as u64;
        let missing_parent_edge_count = edges
            .iter()
            .filter(|edge| edge.status == LineageEdgeStatusV1::MissingParent)
            .count() as u64;
        let cross_scope_edge_count = edges
            .iter()
            .filter(|edge| edge.status == LineageEdgeStatusV1::CrossScopeParent)
            .count() as u64;

        LineageAuditReportV1 {
            edges,
            cycle_components: lineage_cycle_components(&self.documents, &self.by_id),
            resolved_edge_count,
            missing_parent_edge_count,
            cross_scope_edge_count,
        }
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
    pub fn search(&self, query: &RetrievalQueryV1) -> Result<Bm25ResultV1, BaselineErrorV1> {
        if query.scope_id.trim().is_empty() {
            return Err(BaselineErrorV1::EmptyQueryScope);
        }

        let mut query_terms = BTreeMap::<String, u32>::new();
        for token in tokenize_v1(&query.query_text) {
            let count = query_terms.entry(token).or_default();
            *count = count.saturating_add(1);
        }

        // Scope partitioning happens before corpus statistics, document vectors,
        // term postings or scoring are traversed. Knowledge-time ordering lets us
        // cut future-observed records from the candidate workset with one binary search.
        let Some(scope_indexes) = self.scope_documents.get(&query.scope_id) else {
            return Ok(Bm25ResultV1 {
                hits: Vec::new(),
                work: RetrievalWorkV1 {
                    eligible_document_count: 0,
                    scope_knowledge_postings_visited: 0,
                    eligible_postings_scored: 0,
                    scored_document_count: 0,
                },
            });
        };
        let knowledge_visible_count = scope_indexes.partition_point(|index| {
            self.documents[*index].document.observed_at_ms <= query.as_of_ms
        });
        let knowledge_visible_indexes = &scope_indexes[..knowledge_visible_count];
        let eligible_indexes: Vec<usize> = knowledge_visible_indexes
            .iter()
            .copied()
            .filter(|index| {
                let document = &self.documents[*index].document;
                document.valid_from_ms <= query.as_of_ms
                    && document
                        .valid_until_ms
                        .is_none_or(|end| query.as_of_ms < end)
            })
            .collect();
        let eligible_count = eligible_indexes.len();

        if query_terms.is_empty() || query.top_k == 0 || eligible_count == 0 {
            return Ok(Bm25ResultV1 {
                hits: Vec::new(),
                work: RetrievalWorkV1 {
                    eligible_document_count: eligible_count as u64,
                    scope_knowledge_postings_visited: 0,
                    eligible_postings_scored: 0,
                    scored_document_count: 0,
                },
            });
        }

        let eligible_set: std::collections::BTreeSet<usize> =
            eligible_indexes.iter().copied().collect();
        let total_tokens: u128 = eligible_indexes
            .iter()
            .map(|index| u128::from(self.documents[*index].token_count))
            .sum();
        let average_length = total_tokens as f64 / eligible_count as f64;
        let mut scores = BTreeMap::<usize, f64>::new();
        let mut scope_knowledge_postings_visited = 0_u64;
        let mut eligible_postings_scored = 0_u64;

        if average_length > 0.0 {
            for (term, query_frequency) in query_terms {
                let key = (query.scope_id.clone(), term);
                let Some(postings) = self.postings.get(&key) else {
                    continue;
                };
                let knowledge_cutoff = postings.partition_point(|(index, _)| {
                    self.documents[*index].document.observed_at_ms <= query.as_of_ms
                });
                let visible_postings = &postings[..knowledge_cutoff];
                scope_knowledge_postings_visited =
                    scope_knowledge_postings_visited.saturating_add(visible_postings.len() as u64);
                let document_frequency = visible_postings
                    .iter()
                    .filter(|(index, _)| eligible_set.contains(index))
                    .count();
                if document_frequency == 0 {
                    continue;
                }
                let n = eligible_count as f64;
                let df = document_frequency as f64;
                let inverse_document_frequency = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                for (index, frequency) in visible_postings {
                    if !eligible_set.contains(index) {
                        continue;
                    }
                    eligible_postings_scored = eligible_postings_scored.saturating_add(1);
                    let tf = f64::from(*frequency);
                    let length = self.documents[*index].token_count as f64;
                    let normalization = tf
                        + self.config.k1
                            * (1.0 - self.config.b + self.config.b * length / average_length);
                    let term_score = inverse_document_frequency
                        * (tf * (self.config.k1 + 1.0) / normalization)
                        * f64::from(query_frequency);
                    *scores.entry(*index).or_default() += term_score;
                }
            }
        }

        let mut ranked: Vec<(String, f64)> = scores
            .iter()
            .filter(|(_, score)| **score > 0.0)
            .map(|(index, score)| (self.documents[*index].document.document_id.clone(), *score))
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
                scope_knowledge_postings_visited,
                eligible_postings_scored,
                scored_document_count,
            },
        })
    }
}

/// Iterative Kosaraju pass: avoid recursion depth depending on untrusted archive size.
fn lineage_cycle_components(
    documents: &[PreparedDocument],
    by_id: &BTreeMap<String, usize>,
) -> Vec<Vec<String>> {
    let count = documents.len();
    let mut adjacency = vec![Vec::<usize>::new(); count];
    let mut reverse = vec![Vec::<usize>::new(); count];

    for (child_index, prepared) in documents.iter().enumerate() {
        for parent_id in &prepared.document.parent_event_ids {
            let Some(parent_index) = by_id.get(parent_id).copied() else {
                continue;
            };
            if documents[parent_index].document.scope_id != prepared.document.scope_id {
                continue;
            }
            adjacency[child_index].push(parent_index);
            reverse[parent_index].push(child_index);
        }
    }
    for neighbors in adjacency.iter_mut().chain(reverse.iter_mut()) {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut visited = vec![false; count];
    let mut finish_order = Vec::with_capacity(count);
    for start in 0..count {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0_usize)];
        while let Some(&(node, next_neighbor)) = stack.last() {
            if next_neighbor < adjacency[node].len() {
                let neighbor = adjacency[node][next_neighbor];
                stack.last_mut().expect("DFS frame exists").1 += 1;
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push((neighbor, 0));
                }
            } else {
                stack.pop();
                finish_order.push(node);
            }
        }
    }

    let mut assigned = vec![false; count];
    let mut components = Vec::new();
    for start in finish_order.into_iter().rev() {
        if assigned[start] {
            continue;
        }
        assigned[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(node) = stack.pop() {
            component.push(documents[node].document.document_id.clone());
            for &neighbor in &reverse[node] {
                if !assigned[neighbor] {
                    assigned[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        let has_self_loop =
            component.len() == 1 && adjacency[start].contains(&start);
        if component.len() > 1 || has_self_loop {
            component.sort();
            components.push(component);
        }
    }
    components.sort();
    components
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
            source_id: format!("source-{id}"),
            scope_id: scope.to_owned(),
            authority_tier: AuthorityTierV1::Authenticated,
            uncertainty_ppm: 0,
            relation_to_parent: EventRelationV1::Asserts,
            parent_event_ids: Vec::new(),
            source_family: Some(scope.to_owned()),
            source_root_digest_sha256: None,
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
            document(
                "noise",
                "tenant-a",
                "garden irrigation schedule",
                1,
                1,
                None,
            ),
        ]);
        let result = index
            .search(&query("rocket motor inspection", "tenant-a", 10, 10))
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
        let result = index
            .search(&query("valve replaced", "tenant-a", 20, 10))
            .unwrap();
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].document_id, "known");
    }

    #[test]
    fn valid_time_upper_bound_is_exclusive() {
        let index = index(vec![
            document(
                "expired",
                "tenant-a",
                "calibration certificate",
                1,
                1,
                Some(20),
            ),
            document(
                "current",
                "tenant-a",
                "calibration certificate",
                1,
                20,
                None,
            ),
        ]);
        let result = index
            .search(&query("calibration certificate", "tenant-a", 20, 10))
            .unwrap();
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
            document(
                "secret",
                "tenant-b",
                "pump pump pump pump pressure",
                1,
                1,
                None,
            ),
        ]);
        let q = query("pump pressure", "tenant-a", 10, 10);
        assert_eq!(
            base.search(&q).unwrap(),
            with_other_scope.search(&q).unwrap()
        );
    }

    #[test]
    fn other_scopes_and_future_knowledge_do_not_enter_the_query_workset() {
        let base = index(vec![document(
            "known",
            "tenant-a",
            "pump pressure inspection",
            1,
            1,
            None,
        )]);
        let mut extended_documents = vec![document(
            "known",
            "tenant-a",
            "pump pressure inspection",
            1,
            1,
            None,
        )];
        for i in 0..256 {
            extended_documents.push(document(
                &format!("other-{i:03}"),
                "tenant-b",
                "pump pressure pump pressure pump pressure",
                1,
                1,
                None,
            ));
            extended_documents.push(document(
                &format!("future-{i:03}"),
                "tenant-a",
                "pump pressure pump pressure pump pressure",
                100,
                1,
                None,
            ));
        }
        let extended = index(extended_documents);
        let q = query("pump pressure inspection", "tenant-a", 10, 10);
        assert_eq!(base.search(&q).unwrap(), extended.search(&q).unwrap());
        assert_eq!(
            extended
                .search(&q)
                .unwrap()
                .work
                .scope_knowledge_postings_visited,
            base.search(&q)
                .unwrap()
                .work
                .scope_knowledge_postings_visited
        );
    }

    #[test]
    fn query_before_first_valid_or_observed_time_returns_no_hits() {
        let index = index(vec![document(
            "future",
            "tenant-a",
            "release approved",
            20,
            15,
            None,
        )]);
        assert!(
            index
                .search(&query("release approved", "tenant-a", 10, 10))
                .unwrap()
                .hits
                .is_empty()
        );
    }

    #[test]
    fn empty_query_and_zero_top_k_are_explicit_empty_results() {
        let index = index(vec![document("a", "tenant-a", "alpha beta", 1, 1, None)]);
        assert!(
            index
                .search(&query("!!!", "tenant-a", 10, 10))
                .unwrap()
                .hits
                .is_empty()
        );
        assert!(
            index
                .search(&query("alpha", "tenant-a", 10, 0))
                .unwrap()
                .hits
                .is_empty()
        );
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
        assert_eq!(
            index(vec![
                document("z", "tenant-a", "same token", 1, 1, None),
                document("a", "tenant-a", "same token", 1, 1, None),
            ])
            .search(&q)
            .unwrap()
            .hits[0]
                .document_id,
            "a"
        );
    }

    #[test]
    fn unicode_case_folding_preserves_non_ascii_terms() {
        let index = index(vec![
            document(
                "af",
                "tenant-a",
                "Afrikaanse veiligheidsinspeksie",
                1,
                1,
                None,
            ),
            document("zu", "tenant-a", "ukuhlolwa kokuphepha", 1, 1, None),
            document("other", "tenant-a", "unrelated data", 1, 1, None),
        ]);
        assert_eq!(
            index
                .search(&query(
                    "AFRIKAANSE veiligheidsinspeksie",
                    "tenant-a",
                    10,
                    10
                ))
                .unwrap()
                .hits[0]
                .document_id,
            "af"
        );
        assert_eq!(
            index
                .search(&query("UKUHLOLWA kokuphepha", "tenant-a", 10, 10))
                .unwrap()
                .hits[0]
                .document_id,
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
    fn retrieved_event_ids_resolve_to_exact_source_lineage_and_authority() {
        let mut source = document("e-current", "lab-a", "calibration certificate", 10, 5, None);
        source.source_id = "cert-41-v2".into();
        source.authority_tier = AuthorityTierV1::Verified;
        source.uncertainty_ppm = 500;
        source.source_family = Some("supplier_registry".into());
        source.source_root_digest_sha256 =
            Some("ad74977d718be3043ed9ae57f75765a53724767753b8933e6c81bd94f7068019".into());
        source.relation_to_parent = EventRelationV1::Supersedes;
        source.parent_event_ids = vec!["e-old".into()];
        let index = index(vec![source]);
        let result = index
            .search(&query("calibration certificate", "lab-a", 20, 5))
            .unwrap();
        assert_eq!(result.hits[0].document_id, "e-current");
        let resolved = index
            .resolve_source_event(&result.hits[0].document_id)
            .expect("retrieved ID resolves to canonical event");
        assert_eq!(resolved.source_id, "cert-41-v2");
        assert_eq!(resolved.authority_tier, AuthorityTierV1::Verified);
        assert_eq!(resolved.uncertainty_ppm, 500);
        assert_eq!(resolved.source_family.as_deref(), Some("supplier_registry"));
        assert_eq!(
            resolved.source_root_digest_sha256.as_deref(),
            Some("ad74977d718be3043ed9ae57f75765a53724767753b8933e6c81bd94f7068019")
        );
        assert_eq!(resolved.relation_to_parent, EventRelationV1::Supersedes);
        assert_eq!(resolved.parent_event_ids, vec!["e-old"]);
    }

    #[test]
    fn malformed_source_lineage_fails_closed_at_index_construction() {
        let mut bad_root = document("e1", "lab-a", "approved", 1, 1, None);
        bad_root.source_root_digest_sha256 = Some("not-a-digest".into());
        assert_eq!(
            Bm25IndexV1::build(vec![bad_root], Bm25ConfigV1::default()),
            Err(BaselineErrorV1::InvalidSourceRootDigest)
        );

        let mut bad_parent = document("e1", "lab-a", "approved", 1, 1, None);
        bad_parent.parent_event_ids = vec!["e1".into()];
        assert_eq!(
            Bm25IndexV1::build(vec![bad_parent], Bm25ConfigV1::default()),
            Err(BaselineErrorV1::InvalidParentEvent)
        );

        let mut bad_uncertainty = document("e1", "lab-a", "approved", 1, 1, None);
        bad_uncertainty.uncertainty_ppm = 1_000_001;
        assert_eq!(
            Bm25IndexV1::build(vec![bad_uncertainty], Bm25ConfigV1::default()),
            Err(BaselineErrorV1::InvalidUncertainty)
        );
    }

    #[test]
    fn invalid_bm25_parameters_are_rejected() {
        assert_eq!(
            Bm25IndexV1::build(vec![], Bm25ConfigV1 { k1: 0.0, b: 0.75 },),
            Err(BaselineErrorV1::InvalidBm25Parameters)
        );
        assert_eq!(
            Bm25IndexV1::build(vec![], Bm25ConfigV1 { k1: 1.2, b: 1.5 },),
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
            RankedChannelV1 {
                channel_id: "lexical".into(),
                weight: 1.0,
                document_ids: vec!["a".into()],
            },
            RankedChannelV1 {
                channel_id: "lexical".into(),
                weight: 1.0,
                document_ids: vec!["a".into()],
            },
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
    fn lineage_audit_distinguishes_resolved_missing_and_cross_scope_edges() {
        let mut child = document("child", "tenant-a", "correction", 2, 1, None);
        child.parent_event_ids = vec!["parent".into(), "missing".into(), "private-parent".into()];
        child.relation_to_parent = EventRelationV1::Retracts;
        let index = index(vec![
            document("parent", "tenant-a", "original claim", 1, 1, None),
            document("private-parent", "tenant-b", "separate scope", 1, 1, None),
            child,
        ]);

        let audit = index.audit_lineage();
        assert_eq!(audit.resolved_edge_count, 1);
        assert_eq!(audit.missing_parent_edge_count, 1);
        assert_eq!(audit.cross_scope_edge_count, 1);
        assert_eq!(audit.edges.len(), 3);
        assert!(audit.cycle_components.is_empty());
        assert_eq!(
            audit
                .edges
                .iter()
                .find(|edge| edge.parent_event_id == "private-parent")
                .unwrap()
                .status,
            LineageEdgeStatusV1::CrossScopeParent
        );
    }

    #[test]
    fn lineage_audit_finds_exact_cycle_components_without_marking_downstream_nodes() {
        let mut a = document("a", "tenant-a", "event a", 1, 1, None);
        a.parent_event_ids = vec!["b".into()];
        let mut b = document("b", "tenant-a", "event b", 1, 1, None);
        b.parent_event_ids = vec!["c".into()];
        let mut c = document("c", "tenant-a", "event c", 1, 1, None);
        c.parent_event_ids = vec!["a".into()];
        let mut downstream = document("downstream", "tenant-a", "derived event", 1, 1, None);
        downstream.parent_event_ids = vec!["a".into()];

        let audit = index(vec![downstream, c, a, b]).audit_lineage();
        assert_eq!(
            audit.cycle_components,
            vec![vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]]
        );
        assert_eq!(audit.resolved_edge_count, 4);
        assert_eq!(audit.missing_parent_edge_count, 0);
        assert_eq!(audit.cross_scope_edge_count, 0);
    }

    #[test]
    fn rrf_cannot_resurrect_a_document_after_its_valid_time_ends() {
        let index = index(vec![document(
            "expired",
            "tenant-a",
            "old approval",
            1,
            1,
            Some(20),
        )]);
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
