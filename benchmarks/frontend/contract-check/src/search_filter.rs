//! Deterministic, browser-independent reference evaluator for SFEB-L2-005.
//! No network, wall clock, locale, floating-point money or nondeterministic ordering.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Asset {
    pub id: String, pub name: String, pub sku: String, pub category: String,
    pub description: String, pub price_cents: u64, pub stock_count: u64,
}
#[derive(Debug, Deserialize)]
struct DatasetDocument {
    schema: String, dataset_id: String, currency: String, price_unit: String,
    assets: Vec<Asset>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct QueryState {
    pub q: String, pub category: Option<String>, pub available_only: bool,
    pub min_price_cents: Option<u64>, pub max_price_cents: Option<u64>,
    pub sort: String, pub page: u32, pub page_size: u32,
}
impl Default for QueryState {
    fn default() -> Self {
        Self { q: String::new(), category: None, available_only: false,
            min_price_cents: None, max_price_cents: None,
            sort: "name_asc".to_string(), page: 1, page_size: 5 }
    }
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PageResult {
    pub asset_ids: Vec<String>, pub total_matches: usize,
    pub page: u32, pub page_size: u32, pub page_count: usize,
}
pub fn parse_dataset(source: &str) -> Result<(String, Vec<Asset>), String> {
    let document: DatasetDocument = serde_json::from_str(source)
        .map_err(|error| format!("dataset JSON is invalid: {error}"))?;
    if document.schema != "sfeb-search-filter-dataset-v1"
        || document.currency != "USD" || document.price_unit != "integer cents" {
        return Err("dataset schema, currency or price_unit is unsupported".to_string());
    }
    if document.dataset_id.is_empty() || document.assets.is_empty() {
        return Err("dataset_id and assets must be non-empty".to_string());
    }
    let mut ids = BTreeSet::new();
    let mut skus = BTreeSet::new();
    for asset in &document.assets {
        if asset.id.is_empty() || asset.name.is_empty() || asset.sku.is_empty()
            || asset.category.is_empty() || asset.description.is_empty() {
            return Err("asset identity, category and text fields must be non-empty".to_string());
        }
        if !ids.insert(asset.id.clone()) { return Err(format!("duplicate asset ID: {}", asset.id)); }
        if !skus.insert(asset.sku.clone()) { return Err(format!("duplicate asset SKU: {}", asset.sku)); }
    }
    Ok((document.dataset_id, document.assets))
}
fn validate_query(query: &QueryState) -> Result<(), String> {
    if query.q.chars().count() > 80 { return Err("query_too_long".to_string()); }
    if query.q.chars().any(char::is_control) { return Err("query_contains_control_character".to_string()); }
    if query.category.as_deref() == Some("") { return Err("empty_category".to_string()); }
    if query.min_price_cents.zip(query.max_price_cents)
        .is_some_and(|(minimum, maximum)| minimum > maximum) {
        return Err("min_price_cents_exceeds_max_price_cents".to_string());
    }
    if !(1..=50).contains(&query.page_size) { return Err("invalid_page_size".to_string()); }
    if query.page == 0 || query.page > 100_000 { return Err("invalid_page".to_string()); }
    if !matches!(query.sort.as_str(), "name_asc" | "price_asc" | "price_desc" | "stock_asc") {
        return Err("unsupported_sort".to_string());
    }
    Ok(())
}
pub fn evaluate(assets: &[Asset], query: &QueryState) -> Result<PageResult, String> {
    validate_query(query)?;
    let needle = query.q.trim().to_ascii_lowercase();
    let mut matches: Vec<&Asset> = assets.iter().filter(|asset| {
        if !needle.is_empty()
            && !asset.name.to_ascii_lowercase().contains(&needle)
            && !asset.sku.to_ascii_lowercase().contains(&needle)
            && !asset.description.to_ascii_lowercase().contains(&needle) { return false; }
        if query.category.as_deref().is_some_and(|category| asset.category != category) { return false; }
        if query.available_only && asset.stock_count == 0 { return false; }
        if query.min_price_cents.is_some_and(|minimum| asset.price_cents < minimum) { return false; }
        if query.max_price_cents.is_some_and(|maximum| asset.price_cents > maximum) { return false; }
        true
    }).collect();
    matches.sort_by(|left, right| {
        let primary = match query.sort.as_str() {
            "name_asc" => left.name.to_ascii_lowercase().cmp(&right.name.to_ascii_lowercase()),
            "price_asc" => left.price_cents.cmp(&right.price_cents),
            "price_desc" => right.price_cents.cmp(&left.price_cents),
            "stock_asc" => left.stock_count.cmp(&right.stock_count),
            _ => unreachable!("sort mode validated above"),
        };
        primary.then_with(|| left.id.cmp(&right.id))
    });
    let total_matches = matches.len();
    let page_size = query.page_size as usize;
    let page_count = total_matches.div_ceil(page_size);
    let start = (query.page as usize - 1).checked_mul(page_size)
        .ok_or_else(|| "pagination_offset_overflow".to_string())?;
    let asset_ids = matches.iter().skip(start).take(page_size).map(|asset| asset.id.clone()).collect();
    Ok(PageResult { asset_ids, total_matches, page: query.page, page_size: query.page_size, page_count })
}
fn percent_encode_component(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len());
    for byte in bytes {
        if byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(*byte as char);
        } else {
            encoded.push('%');
            encoded.push(char::from_digit((byte >> 4) as u32, 16).unwrap().to_ascii_uppercase());
            encoded.push(char::from_digit((byte & 0x0f) as u32, 16).unwrap().to_ascii_uppercase());
        }
    }
    encoded
}
fn percent_decode_component(encoded: &str) -> Result<Vec<u8>, String> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() { return Err("invalid_state_url_percent_escape".to_string()); }
            let high = (bytes[index + 1] as char).to_digit(16)
                .ok_or_else(|| "invalid_state_url_percent_escape".to_string())?;
            let low = (bytes[index + 2] as char).to_digit(16)
                .ok_or_else(|| "invalid_state_url_percent_escape".to_string())?;
            decoded.push(((high << 4) | low) as u8);
            index += 3;
        } else { decoded.push(bytes[index]); index += 1; }
    }
    Ok(decoded)
}
pub fn encode_state_url(query: &QueryState) -> Result<String, String> {
    validate_query(query)?;
    let json = serde_json::to_vec(query).map_err(|error| format!("query serialization failed: {error}"))?;
    Ok(format!("?state={}", percent_encode_component(&json)))
}
pub fn decode_state_url(url: &str) -> Result<QueryState, String> {
    let encoded = url.strip_prefix("?state=")
        .ok_or_else(|| "state_url_must_contain_only_state_parameter".to_string())?;
    if encoded.is_empty() || encoded.contains('&') {
        return Err("state_url_must_contain_only_state_parameter".to_string());
    }
    let decoded = percent_decode_component(encoded)?;
    let json = std::str::from_utf8(&decoded).map_err(|error| format!("state URL is not UTF-8: {error}"))?;
    let query: QueryState = serde_json::from_str(json).map_err(|error| format!("state URL JSON is invalid: {error}"))?;
    validate_query(&query)?;
    Ok(query)
}
pub fn validate_golden_cases(dataset_source: &str, cases_source: &str) -> Result<usize, String> {
    let (dataset_id, assets) = parse_dataset(dataset_source)?;
    let cases: Value = serde_json::from_str(cases_source).map_err(|error| format!("oracle cases JSON invalid: {error}"))?;
    if cases.get("schema").and_then(Value::as_str) != Some("sfeb-search-filter-oracle-cases-v1")
        || cases.get("dataset_id").and_then(Value::as_str) != Some(dataset_id.as_str()) {
        return Err("oracle cases are not bound to this dataset schema and ID".to_string());
    }
    let definitions = cases.get("cases").and_then(Value::as_array)
        .ok_or_else(|| "oracle cases must contain a cases array".to_string())?;
    if definitions.is_empty() { return Err("oracle case set must not be empty".to_string()); }
    let mut seen = BTreeSet::new();
    for definition in definitions {
        let case_id = definition.get("case_id").and_then(Value::as_str)
            .ok_or_else(|| "every oracle case needs case_id".to_string())?;
        if !seen.insert(case_id.to_string()) { return Err(format!("duplicate oracle case ID: {case_id}")); }
        let input: QueryState = serde_json::from_value(definition.get("input").cloned()
            .ok_or_else(|| format!("{case_id}: input is required"))?)
            .map_err(|error| format!("{case_id}: invalid input: {error}"))?;
        let result = evaluate(&assets, &input);
        if let Some(expected_error) = definition.get("expected_error").and_then(Value::as_str) {
            match result {
                Err(actual) if actual == expected_error => {},
                Err(actual) => return Err(format!("{case_id}: expected {expected_error}, received {actual}")),
                Ok(_) => return Err(format!("{case_id}: expected an error but query succeeded")),
            }
            continue;
        }
        let actual = result.map_err(|error| format!("{case_id}: unexpected error: {error}"))?;
        let expected = definition.get("expected").ok_or_else(|| format!("{case_id}: expected missing"))?;
        let expected_ids: Vec<String> = serde_json::from_value(expected.get("asset_ids").cloned()
            .ok_or_else(|| format!("{case_id}: expected asset_ids missing"))?)
            .map_err(|error| format!("{case_id}: invalid expected IDs: {error}"))?;
        if actual.asset_ids != expected_ids
            || expected.get("total_matches").and_then(Value::as_u64) != Some(actual.total_matches as u64)
            || expected.get("page").and_then(Value::as_u64) != Some(actual.page as u64)
            || expected.get("page_size").and_then(Value::as_u64) != Some(actual.page_size as u64)
            || expected.get("page_count").and_then(Value::as_u64) != Some(actual.page_count as u64) {
            return Err(format!("{case_id}: result mismatch; actual={:?}, expected={expected}", actual));
        }
    }
    Ok(definitions.len())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn assets() -> Vec<Asset> {
        let data = include_str!("../../fixtures/search-filter-explorer/dataset.json");
        parse_dataset(data).expect("dataset parses").1
    }
    #[test]
    fn all_eleven_golden_cases_pass() {
        let data = include_str!("../../fixtures/search-filter-explorer/dataset.json");
        let cases = include_str!("../../fixtures/search-filter-explorer/oracle-cases.json");
        assert_eq!(validate_golden_cases(data, cases).expect("golden cases"), 11);
    }
    #[test]
    fn query_case_change_is_invariant() {
        let a = QueryState { q:"sensor".into(), page_size:50, ..QueryState::default() };
        let b = QueryState { q:"SENSOR".into(), ..a.clone() };
        assert_eq!(evaluate(&assets(), &a).unwrap(), evaluate(&assets(), &b).unwrap());
    }
    #[test]
    fn adding_availability_filter_cannot_add_results() {
        let base = QueryState { category:Some("Sensors".into()), page_size:50, ..QueryState::default() };
        let filtered = QueryState { available_only:true, ..base.clone() };
        let base_ids: BTreeSet<_> = evaluate(&assets(), &base).unwrap().asset_ids.into_iter().collect();
        let kept = evaluate(&assets(), &filtered).unwrap().asset_ids;
        assert!(kept.iter().all(|id| base_ids.contains(id)));
    }
    #[test]
    fn stable_price_ties_use_ascending_asset_id() {
        let base = QueryState { category:Some("Devices".into()), sort:"price_asc".into(), page_size:50, ..QueryState::default() };
        assert_eq!(evaluate(&assets(), &base).unwrap().asset_ids, vec!["AST-011","AST-013","AST-006"]);
        let descending = QueryState { sort:"price_desc".into(), ..base };
        assert_eq!(evaluate(&assets(), &descending).unwrap().asset_ids, vec!["AST-006","AST-011","AST-013"]);
    }
    #[test]
    fn concatenating_pages_reproduces_full_result_once() {
        let first = QueryState::default();
        let mut ids = evaluate(&assets(), &first).unwrap().asset_ids;
        ids.extend(evaluate(&assets(), &QueryState { page:2, ..first.clone() }).unwrap().asset_ids);
        ids.extend(evaluate(&assets(), &QueryState { page:3, ..first.clone() }).unwrap().asset_ids);
        let all = QueryState { page_size:50, ..first };
        assert_eq!(ids, evaluate(&assets(), &all).unwrap().asset_ids);
        assert_eq!(ids.iter().collect::<BTreeSet<_>>().len(), ids.len());
    }
    #[test]
    fn state_url_round_trip_preserves_all_query_fields() {
        let q = QueryState { q:"pump & sensor + tool".into(), category:Some("Power".into()),
            available_only:true, min_price_cents:Some(5000), max_price_cents:Some(250000),
            sort:"price_desc".into(), page:3, page_size:10 };
        assert_eq!(decode_state_url(&encode_state_url(&q).unwrap()).unwrap(), q);
    }
    #[test]
    fn reversed_price_range_fails_closed() {
        let q = QueryState { min_price_cents:Some(200000), max_price_cents:Some(100000), ..QueryState::default() };
        assert_eq!(evaluate(&assets(), &q).unwrap_err(), "min_price_cents_exceeds_max_price_cents");
    }
    #[test]
    fn extra_state_url_parameters_are_rejected() {
        assert_eq!(decode_state_url("?state=%7B%7D&admin=true").unwrap_err(),
            "state_url_must_contain_only_state_parameter");
    }

    #[test]
    fn seeded_generated_queries_preserve_pagination_filter_and_url_properties() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        let assets = assets();
        let terms = ["", "sensor", "pump", "DEV", "battery", "pressure", "thermal", "no-match", "FIELD", "SEN-"];
        let categories = [None, Some("Devices"), Some("Inspection"), Some("Power"), Some("Safety"), Some("Sensors")];
        let sorts = ["name_asc", "price_asc", "price_desc", "stock_asc"];
        let mut seed = 20_261_122_u64;

        for _ in 0..2_048 {
            let a = next(&mut seed) % 350_001;
            let b = next(&mut seed) % 350_001;
            let (minimum, maximum) = if a <= b { (a, b) } else { (b, a) };
            let term = terms[(next(&mut seed) as usize) % terms.len()];
            let category = categories[(next(&mut seed) as usize) % categories.len()].map(str::to_string);
            let available_only = next(&mut seed) % 2 == 1;
            let min_price_cents = if next(&mut seed) % 4 == 0 { Some(minimum) } else { None };
            let max_price_cents = if next(&mut seed) % 4 == 0 { Some(maximum) } else { None };
            let sort = sorts[(next(&mut seed) as usize) % sorts.len()].to_string();
            let page_size = (next(&mut seed) % 7 + 1) as u32;
            let page = (next(&mut seed) % 6 + 1) as u32;
            let query = QueryState {
                q: term.to_string(), category, available_only,
                min_price_cents, max_price_cents, sort, page, page_size,
            };

            let page_result = evaluate(&assets, &query).expect("generated query must be valid");
            assert_eq!(page_result.asset_ids.iter().collect::<BTreeSet<_>>().len(), page_result.asset_ids.len());
            assert_eq!(page_result.page_count,
                (page_result.total_matches + page_result.page_size as usize - 1) / page_result.page_size as usize);

            let mut concatenated = Vec::new();
            for current_page in 1..=page_result.page_count {
                let page = evaluate(&assets, &QueryState { page: current_page as u32, ..query.clone() })
                    .expect("page query must be valid");
                concatenated.extend(page.asset_ids);
            }
            let full = evaluate(&assets, &QueryState { page: 1, page_size: 50, ..query.clone() })
                .expect("full result query must be valid");
            assert_eq!(concatenated, full.asset_ids);
            assert_eq!(concatenated.iter().collect::<BTreeSet<_>>().len(), concatenated.len());

            let encoded = encode_state_url(&query).expect("query state URL encodes");
            assert_eq!(decode_state_url(&encoded).expect("query state URL decodes"), query);

            if query.category.is_some() {
                let base = evaluate(&assets, &QueryState {
                    page: 1, page_size: 50, available_only: false, ..query.clone()
                }).expect("unfiltered category state");
                let filtered = evaluate(&assets, &QueryState {
                    page: 1, page_size: 50, available_only: true, ..query.clone()
                }).expect("available-only category state");
                let base_ids: BTreeSet<_> = base.asset_ids.into_iter().collect();
                assert!(filtered.asset_ids.iter().all(|id| base_ids.contains(id)));
            }
        }
    }

}