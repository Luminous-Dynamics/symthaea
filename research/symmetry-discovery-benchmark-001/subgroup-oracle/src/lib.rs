// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent exact oracle for the active finite-subgroup benchmark.
//! Recomputes group axioms, the complete subgroup lattice, greedy selection,
//! exact static basis cost and exact adaptive minimax cost from committed JSON.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

const FIXTURE: &str = include_str!("../../fixtures/v0.1/active_subgroup_identification.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: String,
    benchmark_id: String,
    groups: Vec<GroupFixture>,
}

#[derive(Debug, Clone, Deserialize)]
struct GroupFixture {
    group_id: String,
    notation: String,
    element_ids: Vec<String>,
    multiplication: Vec<Vec<usize>>,
    identity: usize,
    subgroups: Vec<Vec<usize>>,
    probes: Vec<ProbeFixture>,
    expected: ExpectedMetrics,
}

#[derive(Debug, Clone, Deserialize)]
struct ProbeFixture {
    element_index: usize,
    cost: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct ExpectedMetrics {
    subgroup_count: usize,
    full_scan_cost: u64,
    adaptive_worst_case_cost: u64,
    exact_static_basis_cost: u64,
    greedy_worst_case_cost: u64,
    adaptive_reduction_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Subgroup {
    members: Vec<usize>,
    mask: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Metrics {
    subgroup_count: usize,
    full_scan_cost: u64,
    adaptive_worst_case_cost: u64,
    exact_static_basis_cost: u64,
    greedy_worst_case_cost: u64,
}

fn validate_group(group: &GroupFixture) -> Result<Vec<usize>, String> {
    let n = group.element_ids.len();
    if n == 0 || n > 16 {
        return Err(format!("{}: unsupported group order {n}", group.group_id));
    }
    if group.group_id.trim().is_empty() || group.notation.trim().is_empty() {
        return Err(format!("{}: empty group identity", group.group_id));
    }
    if group.multiplication.len() != n
        || group.multiplication.iter().any(|row| row.len() != n)
    {
        return Err(format!("{}: multiplication table is not square", group.group_id));
    }
    if group.identity >= n {
        return Err(format!("{}: identity index is out of range", group.group_id));
    }
    let mut ids = BTreeSet::new();
    if group.element_ids.iter().any(|id| id.trim().is_empty() || !ids.insert(id)) {
        return Err(format!("{}: empty or duplicate element ID", group.group_id));
    }

    for a in 0..n {
        let mut row_seen = vec![false; n];
        let mut column_seen = vec![false; n];
        for b in 0..n {
            let product = group.multiplication[a][b];
            let column_product = group.multiplication[b][a];
            if product >= n || column_product >= n {
                return Err(format!("{}: product index out of range", group.group_id));
            }
            if row_seen[product] || column_seen[column_product] {
                return Err(format!("{}: multiplication table is not Latin", group.group_id));
            }
            row_seen[product] = true;
            column_seen[column_product] = true;
        }
        if group.multiplication[group.identity][a] != a
            || group.multiplication[a][group.identity] != a
        {
            return Err(format!("{}: two-sided identity check failed", group.group_id));
        }
    }
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                if group.multiplication[group.multiplication[a][b]][c]
                    != group.multiplication[a][group.multiplication[b][c]]
                {
                    return Err(format!("{}: associativity failed", group.group_id));
                }
            }
        }
    }

    let mut inverses = vec![usize::MAX; n];
    for a in 0..n {
        inverses[a] = (0..n)
            .find(|b| {
                group.multiplication[a][*b] == group.identity
                    && group.multiplication[*b][a] == group.identity
            })
            .ok_or_else(|| format!("{}: inverse missing", group.group_id))?;
    }
    Ok(inverses)
}

fn enumerate_subgroups(group: &GroupFixture, inverses: &[usize]) -> Vec<Subgroup> {
    let n = group.element_ids.len();
    let mut result = Vec::new();
    for mask in 0_u64..(1_u64 << n) {
        if mask & (1_u64 << group.identity) == 0 {
            continue;
        }
        let members: Vec<usize> = (0..n)
            .filter(|i| mask & (1_u64 << i) != 0)
            .collect();
        let closed = members.iter().all(|a| {
            mask & (1_u64 << inverses[*a]) != 0
                && members.iter().all(|b| {
                    mask & (1_u64 << group.multiplication[*a][*b]) != 0
                })
        });
        if closed {
            result.push(Subgroup { members, mask });
        }
    }
    result.sort_by(|a, b| (a.members.len(), &a.members).cmp(&(b.members.len(), &b.members)));
    result
}

fn partitions(
    mask: u64,
    outcomes: &[Vec<u32>],
    probe_index: usize,
) -> BTreeMap<u32, u64> {
    let mut buckets = BTreeMap::new();
    for (hypothesis, row) in outcomes.iter().enumerate() {
        if mask & (1_u64 << hypothesis) != 0 {
            *buckets.entry(row[probe_index]).or_insert(0_u64) |= 1_u64 << hypothesis;
        }
    }
    buckets
}

fn adaptive_cost(
    mask: u64,
    outcomes: &[Vec<u32>],
    probes: &[ProbeFixture],
    memo: &mut BTreeMap<u64, Option<u64>>,
) -> Option<u64> {
    if mask == 0 || mask.count_ones() == 1 {
        return Some(0);
    }
    if let Some(cached) = memo.get(&mask) {
        return *cached;
    }
    let mut best: Option<u64> = None;
    for (probe_index, probe) in probes.iter().enumerate() {
        let buckets = partitions(mask, outcomes, probe_index);
        if buckets.len() < 2 {
            continue;
        }
        let mut worst_child = 0_u64;
        let mut solvable = true;
        for child_mask in buckets.values() {
            let Some(child_cost) = adaptive_cost(*child_mask, outcomes, probes, memo) else {
                solvable = false;
                break;
            };
            worst_child = worst_child.max(child_cost);
        }
        if !solvable {
            continue;
        }
        let total = probe.cost.checked_add(worst_child)?;
        best = Some(best.map_or(total, |old| old.min(total)));
    }
    memo.insert(mask, best);
    best
}

fn greedy_cost(mask: u64, outcomes: &[Vec<u32>], probes: &[ProbeFixture]) -> Option<u64> {
    if mask == 0 || mask.count_ones() == 1 {
        return Some(0);
    }
    let mut best: Option<(usize, u64, usize, BTreeMap<u32, u64>)> = None;
    for (probe_index, probe) in probes.iter().enumerate() {
        let buckets = partitions(mask, outcomes, probe_index);
        if buckets.len() < 2 {
            continue;
        }
        let largest = buckets.values().map(|m| m.count_ones() as usize).max().unwrap_or(0);
        let candidate = (largest, probe.cost, probe_index, buckets);
        if best.as_ref().is_none_or(|current| {
            (candidate.0, candidate.1, candidate.2) < (current.0, current.1, current.2)
        }) {
            best = Some(candidate);
        }
    }
    let (_, probe_cost, _, buckets) = best?;
    let mut worst_child = 0_u64;
    for child_mask in buckets.values() {
        worst_child = worst_child.max(greedy_cost(*child_mask, outcomes, probes)?);
    }
    probe_cost.checked_add(worst_child)
}

fn exact_static_cost(outcomes: &[Vec<u32>], probes: &[ProbeFixture]) -> Option<u64> {
    if probes.len() > 20 {
        return None;
    }
    let mut best: Option<u64> = None;
    for subset in 0_u64..(1_u64 << probes.len()) {
        let mut covers_pairs = true;
        'pairs: for a in 0..outcomes.len() {
            for b in (a + 1)..outcomes.len() {
                let differs = probes.iter().enumerate().any(|(j, _)| {
                    subset & (1_u64 << j) != 0 && outcomes[a][j] != outcomes[b][j]
                });
                if !differs {
                    covers_pairs = false;
                    break 'pairs;
                }
            }
        }
        if !covers_pairs {
            continue;
        }
        let mut cost = 0_u64;
        for (j, probe) in probes.iter().enumerate() {
            if subset & (1_u64 << j) != 0 {
                cost = cost.checked_add(probe.cost)?;
            }
        }
        best = Some(best.map_or(cost, |old| old.min(cost)));
    }
    best
}

fn compute_metrics(group: &GroupFixture, subgroups: &[Subgroup]) -> Result<Metrics, String> {
    let n = group.element_ids.len();
    let mut seen = BTreeSet::new();
    for probe in &group.probes {
        if probe.element_index >= n || probe.element_index == group.identity {
            return Err(format!("{}: invalid or identity probe", group.group_id));
        }
        if probe.cost == 0 || !seen.insert(probe.element_index) {
            return Err(format!("{}: duplicate or zero-cost probe", group.group_id));
        }
    }
    let outcomes: Vec<Vec<u32>> = subgroups.iter().map(|subgroup| {
        group.probes.iter().map(|probe| {
            u32::from(subgroup.mask & (1_u64 << probe.element_index) != 0)
        }).collect()
    }).collect();
    let root_mask = (1_u64 << subgroups.len()) - 1;
    let adaptive = adaptive_cost(root_mask, &outcomes, &group.probes, &mut BTreeMap::new())
        .ok_or_else(|| format!("{}: no complete adaptive policy", group.group_id))?;
    let greedy = greedy_cost(root_mask, &outcomes, &group.probes)
        .ok_or_else(|| format!("{}: greedy policy could not identify all subgroups", group.group_id))?;
    let static_cost = exact_static_cost(&outcomes, &group.probes)
        .ok_or_else(|| format!("{}: no exact static probe cover", group.group_id))?;
    let full_scan = group.probes.iter().try_fold(0_u64, |sum, probe| {
        sum.checked_add(probe.cost)
    }).ok_or_else(|| format!("{}: full scan cost overflow", group.group_id))?;
    Ok(Metrics {
        subgroup_count: subgroups.len(),
        full_scan_cost: full_scan,
        adaptive_worst_case_cost: adaptive,
        exact_static_basis_cost: static_cost,
        greedy_worst_case_cost: greedy,
    })
}

fn verify_group(group: &GroupFixture) -> Result<(), String> {
    let inverses = validate_group(group)?;
    let subgroups = enumerate_subgroups(group, &inverses);
    let actual: Vec<Vec<usize>> = subgroups.iter().map(|subgroup| subgroup.members.clone()).collect();
    if actual != group.subgroups {
        return Err(format!("{}: canonical subgroup catalogue mismatch", group.group_id));
    }
    let metrics = compute_metrics(group, &subgroups)?;
    let expected = &group.expected;
    if metrics.subgroup_count != expected.subgroup_count
        || metrics.full_scan_cost != expected.full_scan_cost
        || metrics.adaptive_worst_case_cost != expected.adaptive_worst_case_cost
        || metrics.exact_static_basis_cost != expected.exact_static_basis_cost
        || metrics.greedy_worst_case_cost != expected.greedy_worst_case_cost
    {
        return Err(format!("{}: frozen cost metrics mismatch", group.group_id));
    }
    let ratio = metrics.full_scan_cost as f64 / metrics.adaptive_worst_case_cost as f64;
    if (ratio - expected.adaptive_reduction_ratio).abs() > 1.0e-12 {
        return Err(format!("{}: frozen adaptive ratio mismatch", group.group_id));
    }
    Ok(())
}

pub fn verify_fixture() -> Result<(), String> {
    let fixture: Fixture = serde_json::from_str(FIXTURE)
        .map_err(|error| format!("invalid subgroup fixture JSON: {error}"))?;
    if fixture.schema_version != "0.1.0"
        || fixture.benchmark_id != "LUMINOUS-SYMM-ACTIVE-SUBGROUP-001"
    {
        return Err("unexpected subgroup fixture identity".to_owned());
    }
    if fixture.groups.len() != 4 {
        return Err("expected C4, V4, S3 and D4 fixtures".to_owned());
    }
    for (group, expected_id) in fixture.groups.iter().zip(["C4", "V4", "S3", "D4"]) {
        if group.group_id != expected_id {
            return Err(format!("expected {expected_id}, got {}", group.group_id));
        }
        verify_group(group)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_group_lattices_and_all_cost_baselines_replay_from_json() {
        verify_fixture().unwrap();
    }

    #[test]
    fn malformed_non_latin_or_nonassociative_table_is_rejected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
        let mut group = fixture.groups[3].clone();
        group.multiplication[1][1] = usize::MAX;
        assert!(validate_group(&group).is_err());
    }

    #[test]
    fn omitted_subgroup_is_detected() {
        let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
        let group = fixture.groups[3].clone();
        let inverses = validate_group(&group).unwrap();
        let mut catalogued = enumerate_subgroups(&group, &inverses);
        catalogued.pop();
        let actual: Vec<Vec<usize>> = catalogued.iter().map(|subgroup| subgroup.members.clone()).collect();
        assert_ne!(actual, group.subgroups);
    }
}
