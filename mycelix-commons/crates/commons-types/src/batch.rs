//! Batch operations module — solves N+1 query patterns
//!
//! Extracted from mycelix-property/zomes/shared for reuse across all domains.

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

/// Options for batch record fetching
#[derive(Clone, Debug, Default)]
pub struct BatchGetOptions {
    pub limit: usize,
    pub skip_deleted: bool,
}

/// Result of a batch get operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchGetResult {
    pub records: Vec<Record>,
    pub not_found: Vec<ActionHash>,
    pub errors: Vec<(ActionHash, String)>,
    pub total_requested: usize,
    pub success_count: usize,
}

impl BatchGetResult {
    pub fn new(total_requested: usize) -> Self {
        Self {
            records: Vec::new(),
            not_found: Vec::new(),
            errors: Vec::new(),
            total_requested,
            success_count: 0,
        }
    }
}

/// Batch get records from multiple action hashes
pub fn batch_get_records(
    hashes: Vec<ActionHash>,
    options: BatchGetOptions,
) -> ExternResult<BatchGetResult> {
    let total = hashes.len();
    let mut result = BatchGetResult::new(total);

    let limit = if options.limit == 0 {
        total
    } else {
        options.limit.min(total)
    };

    for hash in hashes.into_iter().take(limit) {
        match get(hash.clone(), GetOptions::default()) {
            Ok(Some(record)) => {
                if options.skip_deleted {
                    if let Action::Delete(_) = record.action() {
                        continue;
                    }
                }
                result.records.push(record);
                result.success_count += 1;
            }
            Ok(None) => {
                result.not_found.push(hash);
            }
            Err(e) => {
                result.errors.push((hash, format!("{:?}", e)));
            }
        }
    }

    Ok(result)
}

/// Get records from links — solves N+1 pattern
pub fn links_to_records(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let hashes: Vec<ActionHash> = links
        .into_iter()
        .filter_map(|link| link.target.into_action_hash())
        .collect();

    let batch_result = batch_get_records(hashes, BatchGetOptions::default())?;
    Ok(batch_result.records)
}

/// Extract and decode entries from records
pub fn extract_entries<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    records: &[Record],
) -> Vec<T> {
    records
        .iter()
        .filter_map(|r| r.entry().to_app_option::<T>().ok().flatten())
        .collect()
}

/// Filter records by a predicate on their entry content
pub fn filter_records_by<T, F>(records: &[Record], predicate: F) -> Vec<Record>
where
    T: TryFrom<SerializedBytes, Error = SerializedBytesError>,
    F: Fn(&T) -> bool,
{
    records
        .iter()
        .filter(|r| {
            r.entry()
                .to_app_option::<T>()
                .ok()
                .flatten()
                .map(|entry| predicate(&entry))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}
