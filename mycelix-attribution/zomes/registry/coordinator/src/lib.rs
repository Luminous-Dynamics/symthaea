// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use registry_integrity::*;
use serde::{Deserialize, Serialize};

// ── Signals ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum RegistrySignal {
    DependencyRegistered {
        dependency_id: String,
        name: String,
        ecosystem: String,
    },
    DependencyUpdated {
        dependency_id: String,
    },
    DependencyVerified {
        dependency_id: String,
    },
}

// ── Input Types ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateDependencyInput {
    pub original_action_hash: ActionHash,
    pub dependency: DependencyIdentity,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginationInput {
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedDependencies {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
    pub has_more: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BulkRegisterResult {
    pub registered: Vec<Record>,
    pub skipped: Vec<String>,
}

// ── Init ─────────────────────────────────────────────────────────────

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    let anchor = Anchor("all_deps".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

// ── Constants ───────────────────────────────────────────────────────

const MAX_PAGE_SIZE: u64 = 1000;

// ── Helpers ──────────────────────────────────────────────────────────
// NOTE: anchor_hash/resolve_links are intentionally duplicated across
// registry, usage, and reciprocity coordinators. Each zome uses its own
// Anchor/LinkTypes from its integrity crate, preventing shared extraction.

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

fn validate_pagination(input: &PaginationInput) -> ExternResult<()> {
    if input.limit > MAX_PAGE_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Pagination limit {} exceeds maximum {}",
            input.limit, MAX_PAGE_SIZE
        ))));
    }
    Ok(())
}

fn require_author(maintainer_did: &str) -> ExternResult<()> {
    let info = agent_info()?;
    let expected_did = format!("did:mycelix:{}", info.agent_initial_pubkey);
    if maintainer_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the maintainer can perform this action".into()
        )));
    }
    Ok(())
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::default())?;
    let mut records = Vec::new();
    for link in links {
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ── Externs ──────────────────────────────────────────────────────────

#[hdk_extern]
pub fn register_dependency(dep: DependencyIdentity) -> ExternResult<Record> {
    // Duplicate check: reject if ID already registered
    let id_tag = format!("dep:{}", dep.id);
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Dependency '{}' is already registered",
            dep.id
        ))));
    }

    let action_hash = create_entry(&EntryTypes::DependencyIdentity(dep.clone()))?;
    let entry_hash = hash_entry(&dep)?;

    // Link: all_deps → dependency
    create_link(
        anchor_hash("all_deps")?,
        entry_hash.clone(),
        LinkTypes::AllDependencies,
        (),
    )?;

    // Link: eco:{ecosystem} → dependency
    let eco_tag = format!("eco:{}", dep.ecosystem);
    create_link(
        anchor_hash(&eco_tag)?,
        entry_hash.clone(),
        LinkTypes::EcosystemToDependency,
        (),
    )?;

    // Link: maint:{did} → dependency
    let maint_tag = format!("maint:{}", dep.maintainer_did);
    create_link(
        anchor_hash(&maint_tag)?,
        entry_hash.clone(),
        LinkTypes::MaintainerToDependency,
        (),
    )?;

    // Link: dep:{id} → dependency (O(1) lookup)
    let id_tag = format!("dep:{}", dep.id);
    create_link(
        anchor_hash(&id_tag)?,
        entry_hash,
        LinkTypes::DependencyById,
        (),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyRegistered {
        dependency_id: dep.id.clone(),
        name: dep.name.clone(),
        ecosystem: dep.ecosystem.to_string(),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created dependency".into()
    )))
}

#[hdk_extern]
pub fn update_dependency(input: UpdateDependencyInput) -> ExternResult<Record> {
    // Author-only: verify caller is the maintainer
    require_author(&input.dependency.maintainer_did)?;

    let action_hash = update_entry(
        input.original_action_hash,
        &EntryTypes::DependencyIdentity(input.dependency.clone()),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyUpdated {
        dependency_id: input.dependency.id.clone(),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch updated dependency".into()
    )))
}

#[hdk_extern]
pub fn get_dependency(id: String) -> ExternResult<Option<Record>> {
    let id_tag = format!("dep:{}", id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
        GetStrategy::default(),
    )?;

    match links.first() {
        Some(link) => {
            let entry_hash = EntryHash::try_from(link.target.clone())
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            get(entry_hash, GetOptions::default())
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn get_all_dependencies(_: ()) -> ExternResult<Vec<Record>> {
    resolve_links(anchor_hash("all_deps")?, LinkTypes::AllDependencies)
}

#[hdk_extern]
pub fn get_dependencies_by_ecosystem(ecosystem: String) -> ExternResult<Vec<Record>> {
    let eco_tag = format!("eco:{}", ecosystem);
    resolve_links(anchor_hash(&eco_tag)?, LinkTypes::EcosystemToDependency)
}

#[hdk_extern]
pub fn get_maintainer_dependencies(did: String) -> ExternResult<Vec<Record>> {
    let maint_tag = format!("maint:{}", did);
    resolve_links(anchor_hash(&maint_tag)?, LinkTypes::MaintainerToDependency)
}

#[hdk_extern]
pub fn verify_dependency(id: String) -> ExternResult<Record> {
    let record = get_dependency(id.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Dependency '{}' not found",
        id
    ))))?;

    let mut dep: DependencyIdentity = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize dependency: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Record has no entry".into()
        )))?;

    // Author-only: verify caller is the maintainer
    require_author(&dep.maintainer_did)?;

    dep.verified = true;

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::DependencyIdentity(dep),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyVerified { dependency_id: id });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch verified dependency".into()
    )))
}

// ── Batch Registration ──────────────────────────────────────────────

#[hdk_extern]
pub fn bulk_register_dependencies(
    deps: Vec<DependencyIdentity>,
) -> ExternResult<BulkRegisterResult> {
    let mut registered = Vec::new();
    let mut skipped = Vec::new();

    for dep in deps {
        let id_tag = format!("dep:{}", dep.id);
        let existing = get_links(
            LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
            GetStrategy::default(),
        )?;
        if !existing.is_empty() {
            skipped.push(dep.id.clone());
            continue;
        }

        let action_hash = create_entry(&EntryTypes::DependencyIdentity(dep.clone()))?;
        let entry_hash = hash_entry(&dep)?;

        create_link(
            anchor_hash("all_deps")?,
            entry_hash.clone(),
            LinkTypes::AllDependencies,
            (),
        )?;

        let eco_tag = format!("eco:{}", dep.ecosystem);
        create_link(
            anchor_hash(&eco_tag)?,
            entry_hash.clone(),
            LinkTypes::EcosystemToDependency,
            (),
        )?;

        let maint_tag = format!("maint:{}", dep.maintainer_did);
        create_link(
            anchor_hash(&maint_tag)?,
            entry_hash.clone(),
            LinkTypes::MaintainerToDependency,
            (),
        )?;

        create_link(
            anchor_hash(&id_tag)?,
            entry_hash,
            LinkTypes::DependencyById,
            (),
        )?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            registered.push(record);
        }
    }

    if !registered.is_empty() {
        let _ = emit_signal(&RegistrySignal::DependencyRegistered {
            dependency_id: format!("batch:{}", registered.len()),
            name: format!("{} dependencies", registered.len()),
            ecosystem: "Mixed".to_string(),
        });
    }

    Ok(BulkRegisterResult {
        registered,
        skipped,
    })
}

// ── Paginated Queries ───────────────────────────────────────────────

#[hdk_extern]
pub fn get_all_dependencies_paginated(
    input: PaginationInput,
) -> ExternResult<PaginatedDependencies> {
    validate_pagination(&input)?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_deps")?, LinkTypes::AllDependencies)?,
        GetStrategy::default(),
    )?;
    let total = links.len() as u64;

    let page_links: Vec<_> = links
        .into_iter()
        .skip(input.offset as usize)
        .take(input.limit as usize)
        .collect();

    let mut items = Vec::new();
    for link in page_links {
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            items.push(record);
        }
    }

    let has_more = input.offset + input.limit < total;
    Ok(PaginatedDependencies {
        items,
        total,
        offset: input.offset,
        limit: input.limit,
        has_more,
    })
}

/// Pure computation of has_more flag (testable without HDK).
#[cfg(test)]
fn compute_has_more(offset: u64, limit: u64, total: u64) -> bool {
    offset + limit < total
}

// ── Ecosystem Statistics ────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EcosystemStat {
    pub ecosystem: String,
    pub count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EcosystemStatistics {
    pub total_dependencies: u64,
    pub ecosystems: Vec<EcosystemStat>,
    pub verified_count: u64,
}

#[hdk_extern]
pub fn get_ecosystem_statistics(_: ()) -> ExternResult<EcosystemStatistics> {
    let all_records = resolve_links(anchor_hash("all_deps")?, LinkTypes::AllDependencies)?;
    let total = all_records.len() as u64;

    let mut eco_counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut verified = 0u64;

    for record in &all_records {
        let dep: Option<DependencyIdentity> = record.entry().to_app_option().ok().flatten();
        if let Some(d) = dep {
            *eco_counts.entry(d.ecosystem.to_string()).or_insert(0) += 1;
            if d.verified {
                verified += 1;
            }
        }
    }

    let ecosystems: Vec<EcosystemStat> = eco_counts
        .into_iter()
        .map(|(ecosystem, count)| EcosystemStat { ecosystem, count })
        .collect();

    Ok(EcosystemStatistics {
        total_dependencies: total,
        ecosystems,
        verified_count: verified,
    })
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pagination validation ───────────────────────────────────────

    #[test]
    fn test_validate_pagination_within_limit() {
        let input = PaginationInput {
            offset: 0,
            limit: 100,
        };
        assert!(validate_pagination(&input).is_ok());
    }

    #[test]
    fn test_validate_pagination_at_max() {
        let input = PaginationInput {
            offset: 0,
            limit: MAX_PAGE_SIZE,
        };
        assert!(validate_pagination(&input).is_ok());
    }

    #[test]
    fn test_validate_pagination_exceeds_max() {
        let input = PaginationInput {
            offset: 0,
            limit: MAX_PAGE_SIZE + 1,
        };
        let err = validate_pagination(&input).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("exceeds maximum"), "error: {}", msg);
    }

    #[test]
    fn test_validate_pagination_zero_limit() {
        let input = PaginationInput {
            offset: 0,
            limit: 0,
        };
        assert!(validate_pagination(&input).is_ok());
    }

    #[test]
    fn test_validate_pagination_large_offset_ok() {
        let input = PaginationInput {
            offset: 999_999,
            limit: 50,
        };
        assert!(validate_pagination(&input).is_ok());
    }

    #[test]
    fn test_max_page_size_is_1000() {
        assert_eq!(MAX_PAGE_SIZE, 1000);
    }

    // ── has_more computation ────────────────────────────────────────

    #[test]
    fn test_has_more_true_when_remaining() {
        assert!(compute_has_more(0, 10, 20));
    }

    #[test]
    fn test_has_more_false_at_exact_end() {
        assert!(!compute_has_more(10, 10, 20));
    }

    #[test]
    fn test_has_more_false_past_end() {
        assert!(!compute_has_more(15, 10, 20));
    }

    #[test]
    fn test_has_more_false_empty_total() {
        assert!(!compute_has_more(0, 10, 0));
    }

    #[test]
    fn test_has_more_single_item_first_page() {
        assert!(!compute_has_more(0, 10, 1));
    }

    #[test]
    fn test_has_more_boundary_one_remaining() {
        // offset=0, limit=9, total=10 → 0+9 < 10 → true
        assert!(compute_has_more(0, 9, 10));
    }

    // ── Serde roundtrip tests ───────────────────────────────────────

    #[test]
    fn test_pagination_input_serde_roundtrip() {
        let input = PaginationInput {
            offset: 42,
            limit: 100,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PaginationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.offset, 42);
        assert_eq!(back.limit, 100);
    }

    #[test]
    fn test_ecosystem_stat_serde_roundtrip() {
        let stat = EcosystemStat {
            ecosystem: "rust_crate".into(),
            count: 42,
        };
        let json = serde_json::to_string(&stat).unwrap();
        let back: EcosystemStat = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ecosystem, "rust_crate");
        assert_eq!(back.count, 42);
    }

    #[test]
    fn test_ecosystem_statistics_serde_roundtrip() {
        let stats = EcosystemStatistics {
            total_dependencies: 100,
            ecosystems: vec![
                EcosystemStat {
                    ecosystem: "rust_crate".into(),
                    count: 60,
                },
                EcosystemStat {
                    ecosystem: "npm_package".into(),
                    count: 40,
                },
            ],
            verified_count: 25,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: EcosystemStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_dependencies, 100);
        assert_eq!(back.ecosystems.len(), 2);
        assert_eq!(back.verified_count, 25);
    }

    #[test]
    fn test_bulk_register_result_serde_roundtrip() {
        let result = BulkRegisterResult {
            registered: vec![],
            skipped: vec!["crate:foo".into(), "crate:bar".into()],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: BulkRegisterResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.skipped.len(), 2);
        assert_eq!(back.skipped[0], "crate:foo");
    }

    // ── Signal variant serde ────────────────────────────────────────

    #[test]
    fn test_registry_signal_serde_roundtrip() {
        let signals = vec![
            RegistrySignal::DependencyRegistered {
                dependency_id: "dep1".into(),
                name: "serde".into(),
                ecosystem: "rust_crate".into(),
            },
            RegistrySignal::DependencyUpdated {
                dependency_id: "dep1".into(),
            },
            RegistrySignal::DependencyVerified {
                dependency_id: "dep1".into(),
            },
        ];
        for sig in signals {
            let json = serde_json::to_string(&sig).unwrap();
            let back: RegistrySignal = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2);
        }
    }
}
