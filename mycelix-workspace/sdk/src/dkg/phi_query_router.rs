//! Phi Query Router - Coherence-Based Query Routing
//!
//! Routes DKG queries based on agent coherence levels.
//!
//! # DEPRECATION NOTICE
//!
//! This module uses Phi (coherence) for query routing, which is over-engineered.
//! Consider using trust score instead:
//! - Trust score already captures agent reliability
//! - Phi adds complexity without clear benefit for query routing
//! - Cost multipliers based on "consciousness" are arbitrary
//!
//! **Recommended alternative**: Use `agent.k_vector.trust_score()` for access control
//! and KREDIT pricing, not Phi coherence.
//!
//! # Current Routing Strategy (DEPRECATED)
//!
//! 1. **Coherent agents** (Phi >= 0.7): Full access, priority results, lower costs
//! 2. **Stable agents** (Phi 0.5-0.7): Standard access, normal costs
//! 3. **Unstable agents** (Phi 0.3-0.5): Limited access, higher costs
//! 4. **Degraded/Critical** (Phi < 0.3): Restricted access, maximum costs

use super::{
    phi_integration::{coherence_allows_operation, CoherenceState, ConsciousnessMetrics},
    query::QueryFilter,
    StoredTriple, VerifiableTriple,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Query routing configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct QueryRouterConfig {
    /// Base KREDIT cost per query
    pub base_query_cost: u64,
    /// Cost per result returned
    pub per_result_cost: u64,
    /// Maximum results for unstable agents
    pub unstable_max_results: usize,
    /// Whether to require approval for sensitive queries from unstable agents
    pub require_approval_for_sensitive: bool,
    /// Domains considered sensitive
    pub sensitive_domains: Vec<String>,
    /// Minimum coherence for complex queries
    pub min_coherence_complex: f64,
}

impl Default for QueryRouterConfig {
    fn default() -> Self {
        Self {
            base_query_cost: 10,
            per_result_cost: 1,
            unstable_max_results: 50,
            require_approval_for_sensitive: true,
            sensitive_domains: vec![
                "medical".to_string(),
                "financial".to_string(),
                "legal".to_string(),
                "defense".to_string(),
            ],
            min_coherence_complex: 0.5,
        }
    }
}

/// Query request with consciousness context
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhiQuery {
    /// The query filter
    pub filter: QueryFilter,
    /// Agent's consciousness metrics
    pub agent_metrics: ConsciousnessMetrics,
    /// Query type for cost calculation
    pub query_type: QueryType,
    /// Whether this is a batch query
    pub is_batch: bool,
    /// Maximum KREDIT budget for this query
    pub max_cost: Option<u64>,
}

/// Types of queries with different cost profiles
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum QueryType {
    /// Simple lookup by subject/predicate
    Simple,
    /// Complex filter with multiple conditions
    Complex,
    /// Aggregation query (count, sum, etc.)
    Aggregation,
    /// Graph traversal query
    GraphTraversal,
    /// Full-text search
    Search,
}

impl QueryType {
    /// Get cost multiplier for query type
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            QueryType::Simple => 1.0,
            QueryType::Complex => 2.0,
            QueryType::Aggregation => 1.5,
            QueryType::GraphTraversal => 3.0,
            QueryType::Search => 2.5,
        }
    }
}

/// Result of query routing decision
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryRoutingDecision {
    /// Whether the query is allowed
    pub allowed: bool,
    /// Reason if not allowed
    pub denial_reason: Option<String>,
    /// Computed cost for this query
    pub computed_cost: u64,
    /// Any restrictions applied
    pub restrictions: Vec<QueryRestriction>,
    /// Priority level for result ordering
    pub priority_level: u8,
    /// Whether human approval is required
    pub requires_approval: bool,
}

/// Restrictions that may be applied to queries
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QueryRestriction {
    /// Maximum number of results
    MaxResults(usize),
    /// Excluded domains
    ExcludedDomains(Vec<String>),
    /// Minimum confidence threshold
    MinConfidence(f64),
    /// Maximum query complexity
    MaxComplexity(u8),
    /// Rate limiting (queries per minute)
    RateLimit(u32),
}

/// Query result with routing metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhiQueryResult {
    /// The query results, sorted by relevance
    pub results: Vec<CoherenceWeightedResult>,
    /// Total KREDIT cost deducted
    pub cost_deducted: u64,
    /// Number of results before filtering
    pub total_matches: usize,
    /// Restrictions that were applied
    pub restrictions_applied: Vec<QueryRestriction>,
    /// Query execution time in microseconds
    pub execution_time_us: u64,
    /// Agent's coherence state during query
    pub agent_coherence: CoherenceState,
}

/// Query result weighted by source coherence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoherenceWeightedResult {
    /// The matching triple
    pub triple: VerifiableTriple,
    /// Confidence score
    pub confidence: f64,
    /// Source coherence factor (if known)
    pub source_coherence: Option<f64>,
    /// Combined relevance score
    pub relevance_score: f64,
    /// Number of attestations
    pub attestation_count: usize,
}

/// The Phi Query Router
pub struct PhiQueryRouter {
    config: QueryRouterConfig,
}

impl PhiQueryRouter {
    /// Create a new router with default config
    pub fn new() -> Self {
        Self {
            config: QueryRouterConfig::default(),
        }
    }

    /// Create a new router with custom config
    pub fn with_config(config: QueryRouterConfig) -> Self {
        Self { config }
    }

    /// Evaluate whether a query should be allowed and compute its cost
    pub fn route_query(&self, query: &PhiQuery) -> QueryRoutingDecision {
        let coherence = query.agent_metrics.coherence_state;

        // Check if agent can perform this operation type
        let operation_type = self.query_operation_type(query);
        if let Err(reason) = coherence_allows_operation(&query.agent_metrics, &operation_type) {
            return QueryRoutingDecision {
                allowed: false,
                denial_reason: Some(reason),
                computed_cost: 0,
                restrictions: vec![],
                priority_level: 0,
                requires_approval: false,
            };
        }

        // Compute base cost
        let base_cost = self.compute_query_cost(query);

        // Apply coherence-based cost adjustment
        let coherence_multiplier = self.coherence_cost_multiplier(coherence);
        let final_cost = (base_cost as f64 * coherence_multiplier) as u64;

        // Check budget
        if let Some(max_cost) = query.max_cost {
            if final_cost > max_cost {
                return QueryRoutingDecision {
                    allowed: false,
                    denial_reason: Some(format!(
                        "Query cost {} exceeds budget {}",
                        final_cost, max_cost
                    )),
                    computed_cost: final_cost,
                    restrictions: vec![],
                    priority_level: 0,
                    requires_approval: false,
                };
            }
        }

        // Determine restrictions based on coherence
        let restrictions = self.compute_restrictions(query);

        // Check if approval is required
        let requires_approval = self.requires_human_approval(query);

        // Compute priority level (higher = better)
        let priority_level = match coherence {
            CoherenceState::Coherent => 4,
            CoherenceState::Stable => 3,
            CoherenceState::Unstable => 2,
            CoherenceState::Degraded => 1,
            CoherenceState::Critical => 0,
        };

        QueryRoutingDecision {
            allowed: true,
            denial_reason: None,
            computed_cost: final_cost,
            restrictions,
            priority_level,
            requires_approval,
        }
    }

    /// Execute a query with consciousness-aware routing
    pub fn execute_query(
        &self,
        query: &PhiQuery,
        triples: &[StoredTriple],
        source_coherence_map: &std::collections::HashMap<String, f64>,
        current_time: u64,
    ) -> Result<PhiQueryResult, String> {
        let start_time = std::time::Instant::now();

        // Route the query first
        let routing = self.route_query(query);

        if !routing.allowed {
            return Err(routing.denial_reason.unwrap_or("Query denied".to_string()));
        }

        if routing.requires_approval {
            return Err("Query requires human approval".to_string());
        }

        // Apply filter to triples
        let mut matches: Vec<CoherenceWeightedResult> = triples
            .iter()
            .filter(|st| query.filter.matches(&st.triple, current_time))
            .map(|st| {
                let source_coherence = source_coherence_map.get(&st.triple.creator).copied();

                // Compute confidence with source coherence boost
                let coherence_boost = source_coherence.unwrap_or(0.5);

                // Relevance combines confidence and source coherence
                let relevance = st.cached_confidence * (0.7 + 0.3 * coherence_boost);

                CoherenceWeightedResult {
                    triple: st.triple.clone(),
                    confidence: st.cached_confidence,
                    source_coherence,
                    relevance_score: relevance,
                    attestation_count: st.attestation_count as usize,
                }
            })
            .collect();

        let total_matches = matches.len();

        // Apply restrictions
        for restriction in &routing.restrictions {
            match restriction {
                QueryRestriction::MaxResults(max) => {
                    matches.truncate(*max);
                }
                QueryRestriction::MinConfidence(min) => {
                    matches.retain(|r| r.confidence >= *min);
                }
                QueryRestriction::ExcludedDomains(domains) => {
                    matches.retain(|r| {
                        r.triple
                            .domain
                            .as_ref()
                            .map(|d| !domains.contains(d))
                            .unwrap_or(true)
                    });
                }
                _ => {}
            }
        }

        // Sort by relevance (coherent sources first)
        matches.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply pagination from filter
        if let Some(offset) = query.filter.offset {
            if offset < matches.len() {
                matches = matches[offset..].to_vec();
            } else {
                matches.clear();
            }
        }

        if let Some(limit) = query.filter.limit {
            matches.truncate(limit);
        }

        let execution_time = start_time.elapsed().as_micros() as u64;

        Ok(PhiQueryResult {
            results: matches,
            cost_deducted: routing.computed_cost,
            total_matches,
            restrictions_applied: routing.restrictions,
            execution_time_us: execution_time,
            agent_coherence: query.agent_metrics.coherence_state,
        })
    }

    /// Compute the cost for a query
    fn compute_query_cost(&self, query: &PhiQuery) -> u64 {
        let base = self.config.base_query_cost;
        let type_mult = query.query_type.cost_multiplier();

        // Estimate result count for per-result cost
        let estimated_results = query.filter.limit.unwrap_or(100) as u64;
        let result_cost = estimated_results * self.config.per_result_cost;

        // Batch queries get a discount
        let batch_mult = if query.is_batch { 0.8 } else { 1.0 };

        ((base as f64 * type_mult + result_cost as f64) * batch_mult) as u64
    }

    /// Get cost multiplier based on coherence state
    fn coherence_cost_multiplier(&self, coherence: CoherenceState) -> f64 {
        match coherence {
            CoherenceState::Coherent => 0.8, // 20% discount
            CoherenceState::Stable => 1.0,   // Normal
            CoherenceState::Unstable => 1.5, // 50% premium
            CoherenceState::Degraded => 2.0, // 100% premium
            CoherenceState::Critical => 3.0, // 200% premium
        }
    }

    /// Determine operation type for permission check
    fn query_operation_type(&self, query: &PhiQuery) -> String {
        // Check if query targets sensitive domains
        if let Some(ref domain) = query.filter.domain {
            if self.config.sensitive_domains.contains(domain) {
                return "high_stakes".to_string();
            }
        }

        // Complex queries require higher coherence
        if query.query_type == QueryType::GraphTraversal || query.query_type == QueryType::Complex {
            return "query_complex".to_string();
        }

        "simple".to_string()
    }

    /// Compute restrictions based on agent coherence
    fn compute_restrictions(&self, query: &PhiQuery) -> Vec<QueryRestriction> {
        let mut restrictions = Vec::new();

        match query.agent_metrics.coherence_state {
            CoherenceState::Coherent => {
                // No restrictions for coherent agents
            }
            CoherenceState::Stable => {
                // Minor restrictions
                restrictions.push(QueryRestriction::RateLimit(100));
            }
            CoherenceState::Unstable => {
                restrictions.push(QueryRestriction::MaxResults(
                    self.config.unstable_max_results,
                ));
                restrictions.push(QueryRestriction::MinConfidence(0.5));
                restrictions.push(QueryRestriction::RateLimit(30));
            }
            CoherenceState::Degraded => {
                restrictions.push(QueryRestriction::MaxResults(20));
                restrictions.push(QueryRestriction::MinConfidence(0.7));
                restrictions.push(QueryRestriction::ExcludedDomains(
                    self.config.sensitive_domains.clone(),
                ));
                restrictions.push(QueryRestriction::RateLimit(10));
            }
            CoherenceState::Critical => {
                restrictions.push(QueryRestriction::MaxResults(5));
                restrictions.push(QueryRestriction::MinConfidence(0.9));
                restrictions.push(QueryRestriction::ExcludedDomains(
                    self.config.sensitive_domains.clone(),
                ));
                restrictions.push(QueryRestriction::RateLimit(1));
            }
        }

        restrictions
    }

    /// Check if human approval is required
    fn requires_human_approval(&self, query: &PhiQuery) -> bool {
        if !self.config.require_approval_for_sensitive {
            return false;
        }

        // Unstable or worse agents need approval for sensitive queries
        let needs_approval = matches!(
            query.agent_metrics.coherence_state,
            CoherenceState::Unstable | CoherenceState::Degraded | CoherenceState::Critical
        );

        let is_sensitive = query
            .filter
            .domain
            .as_ref()
            .map(|d| self.config.sensitive_domains.contains(d))
            .unwrap_or(false);

        needs_approval && is_sensitive
    }
}

impl Default for PhiQueryRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// KREDIT deduction for query execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryCostDeduction {
    /// Agent being charged
    pub agent_id: String,
    /// Amount deducted
    pub amount: u64,
    /// Query type
    pub query_type: QueryType,
    /// Coherence state at time of query
    pub coherence_state: CoherenceState,
    /// Timestamp
    pub timestamp: u64,
    /// Query hash for audit
    pub query_hash: String,
}

/// Deduct KREDIT for a query (returns updated balance)
pub fn deduct_query_cost(current_balance: u64, cost: u64, min_balance: u64) -> Result<u64, String> {
    if current_balance < cost {
        return Err(format!(
            "Insufficient KREDIT: have {}, need {}",
            current_balance, cost
        ));
    }

    let new_balance = current_balance - cost;
    if new_balance < min_balance {
        return Err(format!(
            "Would leave balance {} below minimum {}",
            new_balance, min_balance
        ));
    }

    Ok(new_balance)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metrics(phi: f64) -> ConsciousnessMetrics {
        ConsciousnessMetrics::new(phi, 2, 1700000000)
    }

    fn test_query(phi: f64, query_type: QueryType) -> PhiQuery {
        PhiQuery {
            filter: QueryFilter::new().with_limit(100),
            agent_metrics: test_metrics(phi),
            query_type,
            is_batch: false,
            max_cost: None,
        }
    }

    #[test]
    fn test_coherent_agent_routing() {
        let router = PhiQueryRouter::new();
        let query = test_query(0.85, QueryType::Simple);

        let decision = router.route_query(&query);

        assert!(decision.allowed);
        assert_eq!(decision.priority_level, 4);
        assert!(!decision.requires_approval);
        assert!(decision.restrictions.is_empty());
    }

    #[test]
    fn test_unstable_agent_restrictions() {
        let router = PhiQueryRouter::new();
        let query = test_query(0.4, QueryType::Simple);

        let decision = router.route_query(&query);

        assert!(decision.allowed);
        assert_eq!(decision.priority_level, 2);
        assert!(!decision.restrictions.is_empty());

        // Should have MaxResults restriction
        let has_max_results = decision
            .restrictions
            .iter()
            .any(|r| matches!(r, QueryRestriction::MaxResults(_)));
        assert!(has_max_results);
    }

    #[test]
    fn test_critical_agent_denied_complex() {
        let router = PhiQueryRouter::new();
        let mut query = test_query(0.1, QueryType::Complex);
        query.filter = query.filter.with_domain("medical");

        let decision = router.route_query(&query);

        // Critical agents should be denied complex queries on sensitive domains
        // or require approval
        assert!(decision.requires_approval || !decision.allowed);
    }

    #[test]
    fn test_cost_calculation() {
        let router = PhiQueryRouter::new();

        // Coherent agent gets discount
        let coherent_query = test_query(0.85, QueryType::Simple);
        let coherent_decision = router.route_query(&coherent_query);

        // Degraded agent pays premium
        let degraded_query = test_query(0.2, QueryType::Simple);
        let degraded_decision = router.route_query(&degraded_query);

        assert!(degraded_decision.computed_cost > coherent_decision.computed_cost);
    }

    #[test]
    fn test_sensitive_domain_denied_for_unstable() {
        let router = PhiQueryRouter::new();
        let mut query = test_query(0.4, QueryType::Simple); // Unstable
        query.filter = query.filter.with_domain("financial");

        let decision = router.route_query(&query);

        // Unstable agents are denied high_stakes operations on sensitive domains
        // (not just requiring approval - the operation is blocked)
        assert!(!decision.allowed);
        assert!(decision.denial_reason.is_some());
    }

    #[test]
    fn test_stable_agent_sensitive_domain_allowed() {
        let router = PhiQueryRouter::new();
        let mut query = test_query(0.6, QueryType::Simple); // Stable (phi 0.5-0.7)
        query.filter = query.filter.with_domain("financial");

        let decision = router.route_query(&query);

        // Stable agents can access sensitive domains
        assert!(decision.allowed);
        assert!(!decision.requires_approval); // Stable doesn't require approval
    }

    #[test]
    fn test_budget_check() {
        let router = PhiQueryRouter::new();
        let mut query = test_query(0.85, QueryType::GraphTraversal);
        query.max_cost = Some(5); // Very low budget

        let decision = router.route_query(&query);

        assert!(!decision.allowed);
        assert!(decision.denial_reason.unwrap().contains("exceeds budget"));
    }

    #[test]
    fn test_kredit_deduction() {
        let balance = 1000;
        let cost = 50;
        let min_balance = 100;

        let result = deduct_query_cost(balance, cost, min_balance);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 950);

        // Insufficient balance
        let result2 = deduct_query_cost(30, 50, 100);
        assert!(result2.is_err());

        // Would go below minimum
        let result3 = deduct_query_cost(150, 100, 100);
        assert!(result3.is_err());
    }
}
