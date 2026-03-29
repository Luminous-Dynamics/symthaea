// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG Query System
//!
//! Provides filtering and querying capabilities for verifiable triples.
//! Supports queries by subject, predicate, object, or combinations thereof.
//!
//! # Query Patterns
//!
//! - `query_by_subject("sky")` - All claims about the sky
//! - `query_by_predicate("color")` - All color claims
//! - Combined filters for complex queries

use super::{EpistemicType, TripleValue, VerifiableTriple};
use serde::{Deserialize, Serialize};

/// Filter criteria for querying triples
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryFilter {
    /// Filter by subject (exact match)
    pub subject: Option<String>,
    /// Filter by predicate (exact match or prefix)
    pub predicate: Option<String>,
    /// Filter by object value
    pub object: Option<ObjectFilter>,
    /// Filter by epistemic type
    pub epistemic_type: Option<EpistemicType>,
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by creator
    pub creator: Option<String>,
    /// Minimum confidence threshold
    pub min_confidence: Option<f64>,
    /// Maximum age in seconds
    pub max_age_secs: Option<u64>,
    /// Limit number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Filter for object values
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObjectFilter {
    /// Exact string match
    ExactString(String),
    /// String contains substring
    ContainsString(String),
    /// Numeric comparison
    NumericRange {
        /// Minimum value (inclusive)
        min: Option<f64>,
        /// Maximum value (inclusive)
        max: Option<f64>,
    },
    /// Boolean value match
    Boolean(bool),
}

impl QueryFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: filter by subject
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Builder: filter by predicate
    pub fn with_predicate(mut self, predicate: impl Into<String>) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    /// Builder: filter by object
    pub fn with_object(mut self, filter: ObjectFilter) -> Self {
        self.object = Some(filter);
        self
    }

    /// Builder: filter by epistemic type
    pub fn with_epistemic_type(mut self, t: EpistemicType) -> Self {
        self.epistemic_type = Some(t);
        self
    }

    /// Builder: filter by domain
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Builder: filter by creator
    pub fn with_creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }

    /// Builder: set minimum confidence
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = Some(confidence);
        self
    }

    /// Builder: set maximum age
    pub fn with_max_age(mut self, age_secs: u64) -> Self {
        self.max_age_secs = Some(age_secs);
        self
    }

    /// Builder: set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Builder: set offset for pagination
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Check if a triple matches this filter
    pub fn matches(&self, triple: &VerifiableTriple, current_time: u64) -> bool {
        // Subject filter
        if let Some(ref subject) = self.subject {
            if &triple.subject != subject {
                return false;
            }
        }

        // Predicate filter
        if let Some(ref predicate) = self.predicate {
            if &triple.predicate.url != predicate {
                return false;
            }
        }

        // Object filter
        if let Some(ref obj_filter) = self.object {
            if !matches_object(&triple.object, obj_filter) {
                return false;
            }
        }

        // Epistemic type filter
        if let Some(ref ep_type) = self.epistemic_type {
            if &triple.epistemic_type != ep_type {
                return false;
            }
        }

        // Domain filter
        if let Some(ref domain) = self.domain {
            match &triple.domain {
                Some(d) if d == domain => {}
                _ => return false,
            }
        }

        // Creator filter
        if let Some(ref creator) = self.creator {
            if &triple.creator != creator {
                return false;
            }
        }

        // Age filter
        if let Some(max_age) = self.max_age_secs {
            let age = current_time.saturating_sub(triple.created_at);
            if age > max_age {
                return false;
            }
        }

        true
    }
}

/// Check if a triple value matches an object filter
fn matches_object(value: &TripleValue, filter: &ObjectFilter) -> bool {
    match filter {
        ObjectFilter::ExactString(s) => {
            matches!(value, TripleValue::String(v) if v == s)
        }
        ObjectFilter::ContainsString(substring) => {
            matches!(value, TripleValue::String(v) if v.contains(substring))
        }
        ObjectFilter::NumericRange { min, max } => {
            if let Some(num) = value.as_float() {
                let above_min = min.is_none_or(|m| num >= m);
                let below_max = max.is_none_or(|m| num <= m);
                above_min && below_max
            } else {
                false
            }
        }
        ObjectFilter::Boolean(b) => {
            matches!(value, TripleValue::Boolean(v) if v == b)
        }
    }
}

/// Query triples by subject (convenience function)
pub fn query_by_subject<'a>(
    triples: &'a [VerifiableTriple],
    subject: &str,
    current_time: u64,
) -> Vec<&'a VerifiableTriple> {
    let filter = QueryFilter::new().with_subject(subject);
    triples
        .iter()
        .filter(|t| filter.matches(t, current_time))
        .collect()
}

/// Query triples by predicate (convenience function)
pub fn query_by_predicate<'a>(
    triples: &'a [VerifiableTriple],
    predicate: &str,
    current_time: u64,
) -> Vec<&'a VerifiableTriple> {
    let filter = QueryFilter::new().with_predicate(predicate);
    triples
        .iter()
        .filter(|t| filter.matches(t, current_time))
        .collect()
}

/// Query triples with a full filter
pub fn query_triples<'a>(
    triples: &'a [VerifiableTriple],
    filter: &QueryFilter,
    current_time: u64,
) -> Vec<&'a VerifiableTriple> {
    let mut results: Vec<&VerifiableTriple> = triples
        .iter()
        .filter(|t| filter.matches(t, current_time))
        .collect();

    // Apply pagination
    if let Some(offset) = filter.offset {
        if offset < results.len() {
            results = results[offset..].to_vec();
        } else {
            results.clear();
        }
    }

    if let Some(limit) = filter.limit {
        results.truncate(limit);
    }

    results
}

/// Result of a confidence-weighted query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedQueryResult {
    /// The matching triple
    pub triple: VerifiableTriple,
    /// Computed confidence score
    pub confidence: f64,
    /// Number of attestations
    pub attestation_count: usize,
    /// Number of endorsements
    pub endorsements: usize,
    /// Number of challenges
    pub challenges: usize,
}

// =============================================================================
// KREDIT-INTEGRATED QUERY SYSTEM
// =============================================================================

use super::phi_integration::ConsciousnessMetrics;
use super::phi_query_router::{
    deduct_query_cost, PhiQuery, PhiQueryResult, PhiQueryRouter, QueryCostDeduction,
    QueryRouterConfig, QueryType,
};
use super::StoredTriple;
use std::collections::HashMap;

/// Agent KREDIT balance tracker for query costs
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentKreditBalance {
    /// Current balance
    pub balance: u64,
    /// Minimum allowed balance
    pub min_balance: u64,
    /// Total spent on queries
    pub total_spent: u64,
    /// Number of queries executed
    pub query_count: u64,
    /// Timestamp of last query
    pub last_query_time: u64,
    /// History of recent deductions (for audit)
    pub recent_deductions: Vec<QueryCostDeduction>,
}

impl AgentKreditBalance {
    /// Create a new balance tracker with initial balance
    pub fn new(initial_balance: u64, min_balance: u64) -> Self {
        Self {
            balance: initial_balance,
            min_balance,
            total_spent: 0,
            query_count: 0,
            last_query_time: 0,
            recent_deductions: Vec::new(),
        }
    }

    /// Deduct cost from balance
    pub fn deduct(&mut self, cost: u64) -> Result<(), String> {
        let new_balance = deduct_query_cost(self.balance, cost, self.min_balance)?;
        self.balance = new_balance;
        self.total_spent += cost;
        self.query_count += 1;
        Ok(())
    }

    /// Record a deduction for audit
    pub fn record_deduction(&mut self, deduction: QueryCostDeduction) {
        self.last_query_time = deduction.timestamp;
        self.recent_deductions.push(deduction);

        // Keep only last 100 deductions
        if self.recent_deductions.len() > 100 {
            self.recent_deductions.remove(0);
        }
    }

    /// Check if agent can afford a query
    pub fn can_afford(&self, cost: u64) -> bool {
        self.balance >= cost && (self.balance - cost) >= self.min_balance
    }

    /// Add KREDIT to balance (e.g., from rewards)
    pub fn credit(&mut self, amount: u64) {
        self.balance += amount;
    }
}

/// KREDIT-aware query executor
///
/// Executes queries with automatic cost deduction based on agent coherence.
pub struct KreditQueryExecutor {
    router: PhiQueryRouter,
    /// Agent balances
    balances: HashMap<String, AgentKreditBalance>,
    /// Source coherence cache
    source_coherence: HashMap<String, f64>,
}

impl KreditQueryExecutor {
    /// Create a new executor with default config
    pub fn new() -> Self {
        Self {
            router: PhiQueryRouter::new(),
            balances: HashMap::new(),
            source_coherence: HashMap::new(),
        }
    }

    /// Create with custom router config
    pub fn with_config(config: QueryRouterConfig) -> Self {
        Self {
            router: PhiQueryRouter::with_config(config),
            balances: HashMap::new(),
            source_coherence: HashMap::new(),
        }
    }

    /// Register an agent with initial balance
    pub fn register_agent(&mut self, agent_id: String, initial_balance: u64, min_balance: u64) {
        self.balances.insert(
            agent_id,
            AgentKreditBalance::new(initial_balance, min_balance),
        );
    }

    /// Update source coherence cache
    pub fn set_source_coherence(&mut self, creator: String, coherence: f64) {
        self.source_coherence.insert(creator, coherence);
    }

    /// Get agent's current balance
    pub fn get_balance(&self, agent_id: &str) -> Option<&AgentKreditBalance> {
        self.balances.get(agent_id)
    }

    /// Get mutable balance for crediting
    pub fn get_balance_mut(&mut self, agent_id: &str) -> Option<&mut AgentKreditBalance> {
        self.balances.get_mut(agent_id)
    }

    /// Execute a query with automatic KREDIT deduction
    ///
    /// Returns the query results and records the cost deduction.
    pub fn execute_query(
        &mut self,
        agent_id: &str,
        filter: QueryFilter,
        agent_metrics: ConsciousnessMetrics,
        query_type: QueryType,
        triples: &[StoredTriple],
        current_time: u64,
    ) -> Result<KreditQueryResult, String> {
        // Check agent is registered
        let balance = self
            .balances
            .get(agent_id)
            .ok_or_else(|| format!("Agent {} not registered", agent_id))?;

        // Build the query
        let query = PhiQuery {
            filter,
            agent_metrics: agent_metrics.clone(),
            query_type: query_type.clone(),
            is_batch: false,
            max_cost: Some(balance.balance),
        };

        // Route the query to get cost
        let routing = self.router.route_query(&query);

        if !routing.allowed {
            return Err(routing.denial_reason.unwrap_or("Query denied".to_string()));
        }

        // Check affordability
        if !balance.can_afford(routing.computed_cost) {
            return Err(format!(
                "Insufficient KREDIT: need {}, have {} (min {})",
                routing.computed_cost, balance.balance, balance.min_balance
            ));
        }

        // Execute the query
        let result =
            self.router
                .execute_query(&query, triples, &self.source_coherence, current_time)?;

        // Deduct the cost
        let balance = self
            .balances
            .get_mut(agent_id)
            .ok_or_else(|| format!("Agent {} balance disappeared", agent_id))?;
        balance.deduct(result.cost_deducted)?;

        // Record the deduction
        let deduction = QueryCostDeduction {
            agent_id: agent_id.to_string(),
            amount: result.cost_deducted,
            query_type,
            coherence_state: agent_metrics.coherence_state,
            timestamp: current_time,
            query_hash: format!("{:x}", calculate_query_hash(&query)),
        };
        balance.record_deduction(deduction.clone());

        Ok(KreditQueryResult {
            query_result: result,
            balance_after: balance.balance,
            deduction,
        })
    }

    /// Execute a batch of queries with combined cost
    pub fn execute_batch(
        &mut self,
        agent_id: &str,
        queries: Vec<(QueryFilter, QueryType)>,
        agent_metrics: ConsciousnessMetrics,
        triples: &[StoredTriple],
        current_time: u64,
    ) -> Result<Vec<KreditQueryResult>, String> {
        let mut results = Vec::new();

        for (filter, query_type) in queries {
            let result = self.execute_query(
                agent_id,
                filter,
                agent_metrics.clone(),
                query_type,
                triples,
                current_time,
            )?;
            results.push(result);
        }

        Ok(results)
    }

    /// Preview query cost without executing
    pub fn preview_cost(
        &self,
        filter: &QueryFilter,
        agent_metrics: &ConsciousnessMetrics,
        query_type: &QueryType,
    ) -> QueryCostPreview {
        let query = PhiQuery {
            filter: filter.clone(),
            agent_metrics: agent_metrics.clone(),
            query_type: query_type.clone(),
            is_batch: false,
            max_cost: None,
        };

        let routing = self.router.route_query(&query);

        QueryCostPreview {
            estimated_cost: routing.computed_cost,
            allowed: routing.allowed,
            denial_reason: routing.denial_reason,
            restrictions: routing.restrictions.len(),
            requires_approval: routing.requires_approval,
        }
    }
}

impl Default for KreditQueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a KREDIT-aware query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KreditQueryResult {
    /// The underlying query result
    pub query_result: PhiQueryResult,
    /// Agent's balance after deduction
    pub balance_after: u64,
    /// The deduction record
    pub deduction: QueryCostDeduction,
}

/// Preview of query cost without execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryCostPreview {
    /// Estimated cost
    pub estimated_cost: u64,
    /// Whether the query would be allowed
    pub allowed: bool,
    /// Reason if not allowed
    pub denial_reason: Option<String>,
    /// Number of restrictions that would apply
    pub restrictions: usize,
    /// Whether approval would be required
    pub requires_approval: bool,
}

/// Calculate a hash of the query for audit purposes
fn calculate_query_hash(query: &PhiQuery) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash filter components
    if let Some(ref s) = query.filter.subject {
        s.hash(&mut hasher);
    }
    if let Some(ref p) = query.filter.predicate {
        p.hash(&mut hasher);
    }
    if let Some(ref d) = query.filter.domain {
        d.hash(&mut hasher);
    }

    // Hash query type
    format!("{:?}", query.query_type).hash(&mut hasher);

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<VerifiableTriple> {
        vec![
            VerifiableTriple::new("sky", "color", TripleValue::String("blue".into()))
                .with_epistemic_type(EpistemicType::Empirical)
                .with_timestamp(1700000000),
            VerifiableTriple::new("sky", "color", TripleValue::String("green".into()))
                .with_epistemic_type(EpistemicType::Empirical)
                .with_timestamp(1700000000),
            VerifiableTriple::new("grass", "color", TripleValue::String("green".into()))
                .with_epistemic_type(EpistemicType::Empirical)
                .with_timestamp(1700000000),
            VerifiableTriple::new("pi", "value", TripleValue::Float(3.14159))
                .with_epistemic_type(EpistemicType::Empirical)
                .with_domain("mathematics")
                .with_timestamp(1700000000),
        ]
    }

    #[test]
    fn test_query_by_subject() {
        let triples = sample_triples();
        let results = query_by_subject(&triples, "sky", 1700000000);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_predicate() {
        let triples = sample_triples();
        let results = query_by_predicate(&triples, "color", 1700000000);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_filter_combined() {
        let triples = sample_triples();
        let filter = QueryFilter::new()
            .with_subject("sky")
            .with_predicate("color");

        let results = query_triples(&triples, &filter, 1700000000);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_filter_object_exact() {
        let triples = sample_triples();
        let filter = QueryFilter::new().with_object(ObjectFilter::ExactString("blue".into()));

        let results = query_triples(&triples, &filter, 1700000000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "sky");
    }

    #[test]
    fn test_query_filter_object_numeric() {
        let triples = sample_triples();
        let filter = QueryFilter::new().with_object(ObjectFilter::NumericRange {
            min: Some(3.0),
            max: Some(4.0),
        });

        let results = query_triples(&triples, &filter, 1700000000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "pi");
    }

    #[test]
    fn test_query_filter_domain() {
        let triples = sample_triples();
        let filter = QueryFilter::new().with_domain("mathematics");

        let results = query_triples(&triples, &filter, 1700000000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "pi");
    }

    #[test]
    fn test_query_filter_pagination() {
        let triples = sample_triples();

        // First page
        let filter1 = QueryFilter::new().with_predicate("color").with_limit(2);
        let page1 = query_triples(&triples, &filter1, 1700000000);
        assert_eq!(page1.len(), 2);

        // Second page
        let filter2 = QueryFilter::new()
            .with_predicate("color")
            .with_offset(2)
            .with_limit(2);
        let page2 = query_triples(&triples, &filter2, 1700000000);
        assert_eq!(page2.len(), 1);
    }

    #[test]
    fn test_query_filter_age() {
        let triples = sample_triples();
        let current_time = 1700000000 + 3600; // 1 hour later

        let filter = QueryFilter::new().with_max_age(1800); // 30 minutes max age

        let results = query_triples(&triples, &filter, current_time);
        assert_eq!(results.len(), 0); // All triples are too old
    }
}
