// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query Coordinator Zome
//! Business logic for knowledge graph queries
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use query_integrity::*;

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Execute a query on the knowledge graph
#[hdk_extern]
pub fn execute_query(input: ExecuteQueryInput) -> ExternResult<QueryExecutionResult> {
    let start = sys_time()?;

    // Parse and validate query
    let plan = parse_query(&input.query)?;

    // Execute query plan
    let results = execute_plan(&plan, &input.parameters)?;

    let end = sys_time()?;
    let execution_time_ms = (end.as_micros() - start.as_micros()) / 1000;

    Ok(QueryExecutionResult {
        results,
        count: 0, // Will be set based on actual results
        execution_time_ms: execution_time_ms as u64,
        plan: Some(plan),
    })
}

/// Input for executing a query
#[derive(Serialize, Deserialize, Debug)]
pub struct ExecuteQueryInput {
    pub query: String,
    pub parameters: Option<String>,
    pub use_cache: bool,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

/// Query execution result
#[derive(Serialize, Deserialize, Debug)]
pub struct QueryExecutionResult {
    pub results: Vec<String>, // JSON-encoded claim IDs
    pub count: u64,
    pub execution_time_ms: u64,
    pub plan: Option<QueryPlan>,
}

/// Parse a query string into a query plan
fn parse_query(query: &str) -> ExternResult<QueryPlan> {
    // Simple query parser
    // Supports basic syntax like:
    // SELECT * FROM claims WHERE tag = 'energy'
    // SELECT * FROM claims WHERE e > 0.7 AND n < 0.5

    let query_lower = query.to_lowercase();

    let mut steps = Vec::new();

    if query_lower.contains("where") {
        // Extract conditions
        let parts: Vec<&str> = query_lower.split("where").collect();
        if parts.len() > 1 {
            let conditions = parts[1].trim();

            // Parse simple conditions
            if conditions.contains("tag") {
                steps.push(QueryStep {
                    step_type: QueryStepType::TagLookup,
                    target: extract_value(conditions, "tag"),
                    filters: Vec::new(),
                });
            } else if conditions.contains("author") {
                steps.push(QueryStep {
                    step_type: QueryStepType::AuthorLookup,
                    target: extract_value(conditions, "author"),
                    filters: Vec::new(),
                });
            } else if conditions.contains(" e ") || conditions.contains(" n ") || conditions.contains(" m ") {
                steps.push(QueryStep {
                    step_type: QueryStepType::EpistemicFilter,
                    target: String::new(),
                    filters: parse_epistemic_filters(conditions),
                });
            } else {
                steps.push(QueryStep {
                    step_type: QueryStepType::FullScan,
                    target: String::new(),
                    filters: Vec::new(),
                });
            }
        }
    } else {
        // No WHERE clause - full scan
        steps.push(QueryStep {
            step_type: QueryStepType::FullScan,
            target: String::new(),
            filters: Vec::new(),
        });
    }

    Ok(QueryPlan {
        steps,
        estimated_cost: 1.0,
        use_cache: true,
    })
}

/// Extract a value from a condition string
fn extract_value(conditions: &str, field: &str) -> String {
    // Very basic extraction - would be more robust in production
    let parts: Vec<&str> = conditions.split('=').collect();
    if parts.len() > 1 {
        parts[1].trim().trim_matches('\'').trim_matches('"').to_string()
    } else {
        String::new()
    }
}

/// Parse epistemic filter conditions
fn parse_epistemic_filters(conditions: &str) -> Vec<query_integrity::QueryFilter> {
    let mut filters = Vec::new();

    // Parse conditions like "e > 0.7 AND n < 0.5"
    let parts: Vec<&str> = conditions.split("and").collect();

    for part in parts {
        let part = part.trim();

        for (field, prefix) in [("e", " e "), ("n", " n "), ("m", " m ")].iter() {
            if part.contains(prefix) || part.starts_with(*field) {
                // Extract operator and value
                if let Some(filter) = parse_comparison(part, field) {
                    filters.push(filter);
                }
            }
        }
    }

    filters
}

/// Parse a comparison into a filter
fn parse_comparison(part: &str, field: &str) -> Option<query_integrity::QueryFilter> {
    let operators = [
        (">=", FilterOperator::GreaterOrEqual),
        ("<=", FilterOperator::LessOrEqual),
        (">", FilterOperator::GreaterThan),
        ("<", FilterOperator::LessThan),
        ("=", FilterOperator::Equals),
    ];

    for (op_str, op) in operators.iter() {
        if part.contains(op_str) {
            let parts: Vec<&str> = part.split(op_str).collect();
            if parts.len() > 1 {
                return Some(query_integrity::QueryFilter {
                    field: field.to_string(),
                    operator: op.clone(),
                    value: parts[1].trim().to_string(),
                });
            }
        }
    }

    None
}

/// Execute a query plan using cross-zome calls to the claims coordinator
fn execute_plan(plan: &QueryPlan, _parameters: &Option<String>) -> ExternResult<Vec<String>> {
    let mut results = Vec::new();

    for step in &plan.steps {
        match step.step_type {
            QueryStepType::TagLookup => {
                let response = call(
                    CallTargetCell::Local,
                    ZomeName::from("claims"),
                    FunctionName::from("get_claims_by_tag"),
                    None,
                    step.target.clone(),
                )?;

                match response {
                    ZomeCallResponse::Ok(bytes) => {
                        let records: Vec<Record> = bytes.decode()
                            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                                format!("Failed to decode claims by tag '{}': {}", step.target, e)
                            )))?;
                        for record in records {
                            results.push(record.action_address().to_string());
                        }
                    }
                    ZomeCallResponse::NetworkError(e) => {
                        debug!("Cross-zome call to claims::get_claims_by_tag failed (network): {:?}", e);
                    }
                    ZomeCallResponse::Unauthorized(_, _, _, _) => {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Unauthorized to query claims zome - check zome call capabilities".into()
                        )));
                    }
                    other => {
                        debug!("Cross-zome call to claims::get_claims_by_tag returned unexpected response: {:?}", other);
                    }
                }
            }
            QueryStepType::AuthorLookup => {
                let response = call(
                    CallTargetCell::Local,
                    ZomeName::from("claims"),
                    FunctionName::from("get_claims_by_author"),
                    None,
                    step.target.clone(),
                )?;

                match response {
                    ZomeCallResponse::Ok(bytes) => {
                        let records: Vec<Record> = bytes.decode()
                            .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                                format!("Failed to decode claims by author '{}': {}", step.target, e)
                            )))?;
                        for record in records {
                            results.push(record.action_address().to_string());
                        }
                    }
                    ZomeCallResponse::NetworkError(e) => {
                        debug!("Cross-zome call to claims::get_claims_by_author failed (network): {:?}", e);
                    }
                    ZomeCallResponse::Unauthorized(_, _, _, _) => {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Unauthorized to query claims zome - check zome call capabilities".into()
                        )));
                    }
                    other => {
                        debug!("Cross-zome call to claims::get_claims_by_author returned unexpected response: {:?}", other);
                    }
                }
            }
            _ => {
                // Full scan and epistemic filter not yet implemented via cross-zome
                debug!("Query step type {:?} not yet implemented for cross-zome execution", step.step_type);
            }
        }
    }

    Ok(results)
}

/// Save a query for reuse
#[hdk_extern]
pub fn save_query(query: SavedQuery) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::SavedQuery(query.clone()))?;

    // Create anchor and link creator to query
    let creator_anchor = format!("creator:{}", query.creator);
    create_entry(&EntryTypes::Anchor(Anchor(creator_anchor.clone())))?;
    create_link(
        anchor_hash(&creator_anchor)?,
        action_hash.clone(),
        LinkTypes::CreatorToQuery,
        (),
    )?;

    // Link to public index if public
    if query.public {
        let public_anchor = "public_queries".to_string();
        create_entry(&EntryTypes::Anchor(Anchor(public_anchor.clone())))?;
        create_link(
            anchor_hash(&public_anchor)?,
            action_hash.clone(),
            LinkTypes::PublicQueries,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find saved query".into()
        )))
}

/// Get saved queries by creator
#[hdk_extern]
pub fn get_my_queries(creator_did: String) -> ExternResult<Vec<Record>> {
    let creator_anchor = format!("creator:{}", creator_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&creator_anchor)?, LinkTypes::CreatorToQuery)?,
        GetStrategy::default(),
    )?;

    let mut queries = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            queries.push(record);
        }
    }

    Ok(queries)
}

/// Get public queries
#[hdk_extern]
pub fn get_public_queries(_: ()) -> ExternResult<Vec<Record>> {
    let public_anchor = "public_queries".to_string();
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&public_anchor)?, LinkTypes::PublicQueries)?,
        GetStrategy::default(),
    )?;

    let mut queries = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            queries.push(record);
        }
    }

    Ok(queries)
}
