#![forbid(unsafe_code)]

//! Conservative common-cause closure for evidence units used in calibration.
//!
//! Shared declared causes create connected components. Each component is one
//! effective statistical unit and carries the maximum dangerous score among
//! its member episodes. This prevents declared pseudo-replication; it does not
//! prove independence or infer undeclared physical causes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceSplit {
    Training,
    ScaleValidation,
    Calibration,
    HeldOutEvaluation,
    Deployment,
}

impl EvidenceSplit {
    pub const ALL: [Self; 5] = [
        Self::Training,
        Self::ScaleValidation,
        Self::Calibration,
        Self::HeldOutEvaluation,
        Self::Deployment,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DependenceKind {
    DeclaredCluster,
    PhysicalAsset,
    CollectionSession,
    ScenarioSeed,
    ExternalShock,
    RawSourceTrace,
    TemporalBlock,
}

/// A canonical typed common-cause key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DependenceKey {
    pub kind: DependenceKind,
    pub value: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceUnit {
    pub unit_id: String,
    pub independent_cluster_id: String,
    pub cluster_identity_receipt_sha256: String,
    pub split: EvidenceSplit,
    /// Higher values must mean more dangerous / more optimistic residual.
    pub dangerous_score_units: u64,
    pub score_receipt_sha256: String,
    pub dependence_keys: Vec<DependenceKey>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyReceiptBinding {
    pub kind: DependenceKind,
    pub value: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndependentBlock {
    /// Deterministic identity: lexicographically smallest member unit ID.
    pub block_id: String,
    pub split: EvidenceSplit,
    pub member_unit_ids: Vec<String>,
    /// Conservative aggregation for dangerous upper-tail calibration.
    pub maximum_dangerous_score_units: u64,
    /// Includes every member score receipt, not only the max-scoring episode.
    pub score_receipts_sha256: Vec<String>,
    pub dependence_key_receipts: Vec<KeyReceiptBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitCount {
    pub split: EvidenceSplit,
    pub raw_unit_count: usize,
    pub independent_block_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureManifest {
    pub source_manifest_sha256: String,
    pub closure_algorithm: String,
    pub raw_unit_count: usize,
    pub total_independent_block_count: usize,
    pub calibration_raw_unit_count: usize,
    pub calibration_effective_block_count: usize,
    pub calibration_pseudoreplication_factor_numerator: usize,
    pub calibration_pseudoreplication_factor_denominator: usize,
    pub split_counts: Vec<SplitCount>,
    pub blocks: Vec<IndependentBlock>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureError {
    InvalidSha256(&'static str),
    EmptyManifest,
    EmptyIdentity(&'static str),
    DuplicateUnitId,
    ConflictingDependenceReceipt {
        kind: DependenceKind,
        value: String,
    },
    CrossSplitDependency {
        block_id: String,
        unit_ids: Vec<String>,
        splits: Vec<EvidenceSplit>,
    },
    MissingCalibrationBlocks,
}

#[derive(Clone, Debug)]
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize, unit_ids: &[String]) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        // Always retain the lexicographically smaller unit ID as root so the
        // closure is stable under permutations of the input inventory.
        if unit_ids[left_root] <= unit_ids[right_root] {
            self.parent[right_root] = left_root;
        } else {
            self.parent[left_root] = right_root;
        }
    }
}

/// Build connected common-cause blocks and reject any dependency crossing splits.
///
/// A mandatory DeclaredCluster key is added from each unit's declared cluster
/// identity. Additional asset/session/seed/shock/trace/time keys extend closure
/// transitively. Calibration effective sample count is the resulting number of
/// calibration blocks, not the raw number of rows or episodes.
pub fn build_common_cause_closure(
    source_manifest_sha256: &str,
    units: &[EvidenceUnit],
) -> Result<ClosureManifest, ClosureError> {
    validate_sha256(source_manifest_sha256, "source_manifest_sha256")?;
    if units.is_empty() {
        return Err(ClosureError::EmptyManifest);
    }

    let mut unit_ids = Vec::with_capacity(units.len());
    let mut seen_unit_ids = BTreeSet::new();
    for unit in units {
        if unit.unit_id.trim().is_empty() {
            return Err(ClosureError::EmptyIdentity("unit_id"));
        }
        if unit.independent_cluster_id.trim().is_empty() {
            return Err(ClosureError::EmptyIdentity("independent_cluster_id"));
        }
        if !seen_unit_ids.insert(unit.unit_id.as_str()) {
            return Err(ClosureError::DuplicateUnitId);
        }
        validate_sha256(
            &unit.cluster_identity_receipt_sha256,
            "cluster_identity_receipt_sha256",
        )?;
        validate_sha256(&unit.score_receipt_sha256, "score_receipt_sha256")?;
        unit_ids.push(unit.unit_id.clone());
    }

    let mut union_find = UnionFind::new(units.len());
    let mut first_by_key: HashMap<(DependenceKind, String), (usize, String)> = HashMap::new();

    for (index, unit) in units.iter().enumerate() {
        let mut local_keys = BTreeMap::<(DependenceKind, String), String>::new();
        local_keys.insert(
            (
                DependenceKind::DeclaredCluster,
                unit.independent_cluster_id.clone(),
            ),
            unit.cluster_identity_receipt_sha256.clone(),
        );

        for key in &unit.dependence_keys {
            if key.value.trim().is_empty() {
                return Err(ClosureError::EmptyIdentity("dependence_key.value"));
            }
            validate_sha256(&key.receipt_sha256, "dependence_key.receipt_sha256")?;
            let identity = (key.kind, key.value.clone());
            if let Some(previous_receipt) =
                local_keys.insert(identity.clone(), key.receipt_sha256.clone())
                && previous_receipt != key.receipt_sha256
            {
                return Err(ClosureError::ConflictingDependenceReceipt {
                    kind: key.kind,
                    value: key.value.clone(),
                });
            }
        }

        for (identity, receipt) in local_keys {
            if let Some((first_index, first_receipt)) = first_by_key.get(&identity) {
                if first_receipt != &receipt {
                    return Err(ClosureError::ConflictingDependenceReceipt {
                        kind: identity.0,
                        value: identity.1,
                    });
                }
                union_find.union(index, *first_index, &unit_ids);
            } else {
                first_by_key.insert(identity, (index, receipt));
            }
        }
    }

    let mut components = BTreeMap::<String, Vec<usize>>::new();
    for index in 0..units.len() {
        let root = union_find.find(index);
        components
            .entry(unit_ids[root].clone())
            .or_default()
            .push(index);
    }

    let mut blocks = Vec::with_capacity(components.len());
    for (block_id, mut member_indexes) in components {
        member_indexes.sort_by(|left, right| unit_ids[*left].cmp(&unit_ids[*right]));
        let mut split_set = BTreeSet::new();
        let mut member_unit_ids = Vec::with_capacity(member_indexes.len());
        let mut score_receipts = BTreeSet::new();
        let mut key_receipts = BTreeSet::new();
        let mut maximum_dangerous_score_units = 0_u64;

        for index in &member_indexes {
            let unit = &units[*index];
            split_set.insert(unit.split);
            member_unit_ids.push(unit.unit_id.clone());
            score_receipts.insert(unit.score_receipt_sha256.clone());
            maximum_dangerous_score_units =
                maximum_dangerous_score_units.max(unit.dangerous_score_units);
            key_receipts.insert(KeyReceiptBinding {
                kind: DependenceKind::DeclaredCluster,
                value: unit.independent_cluster_id.clone(),
                receipt_sha256: unit.cluster_identity_receipt_sha256.clone(),
            });
            for key in &unit.dependence_keys {
                key_receipts.insert(KeyReceiptBinding {
                    kind: key.kind,
                    value: key.value.clone(),
                    receipt_sha256: key.receipt_sha256.clone(),
                });
            }
        }

        if split_set.len() != 1 {
            return Err(ClosureError::CrossSplitDependency {
                block_id,
                unit_ids: member_unit_ids,
                splits: split_set.into_iter().collect(),
            });
        }
        let split = *split_set.first().ok_or(ClosureError::EmptyManifest)?;

        blocks.push(IndependentBlock {
            block_id,
            split,
            member_unit_ids,
            maximum_dangerous_score_units,
            score_receipts_sha256: score_receipts.into_iter().collect(),
            dependence_key_receipts: key_receipts.into_iter().collect(),
        });
    }

    blocks.sort_by(|left, right| (left.split, &left.block_id).cmp(&(right.split, &right.block_id)));

    let mut split_counts = Vec::new();
    for split in EvidenceSplit::ALL {
        let raw_unit_count = units.iter().filter(|unit| unit.split == split).count();
        let independent_block_count = blocks.iter().filter(|block| block.split == split).count();
        if raw_unit_count > 0 {
            split_counts.push(SplitCount {
                split,
                raw_unit_count,
                independent_block_count,
            });
        }
    }

    let calibration_raw_unit_count = units
        .iter()
        .filter(|unit| unit.split == EvidenceSplit::Calibration)
        .count();
    let calibration_effective_block_count = blocks
        .iter()
        .filter(|block| block.split == EvidenceSplit::Calibration)
        .count();
    if calibration_effective_block_count == 0 {
        return Err(ClosureError::MissingCalibrationBlocks);
    }

    Ok(ClosureManifest {
        source_manifest_sha256: source_manifest_sha256.to_string(),
        closure_algorithm: "typed-common-cause-union-find-max-score-v0.1".to_string(),
        raw_unit_count: units.len(),
        total_independent_block_count: blocks.len(),
        calibration_raw_unit_count,
        calibration_effective_block_count,
        calibration_pseudoreplication_factor_numerator: calibration_raw_unit_count,
        calibration_pseudoreplication_factor_denominator: calibration_effective_block_count,
        split_counts,
        blocks,
    })
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), ClosureError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ClosureError::InvalidSha256(field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn key(kind: DependenceKind, value: &str, receipt: char) -> DependenceKey {
        DependenceKey {
            kind,
            value: value.into(),
            receipt_sha256: digest(receipt),
        }
    }

    fn unit(
        id: &str,
        cluster: &str,
        split: EvidenceSplit,
        score: u64,
        cluster_receipt: char,
        score_receipt: char,
        keys: Vec<DependenceKey>,
    ) -> EvidenceUnit {
        EvidenceUnit {
            unit_id: id.into(),
            independent_cluster_id: cluster.into(),
            cluster_identity_receipt_sha256: digest(cluster_receipt),
            split,
            dangerous_score_units: score,
            score_receipt_sha256: digest(score_receipt),
            dependence_keys: keys,
        }
    }

    fn calibration_units() -> Vec<EvidenceUnit> {
        vec![
            unit(
                "episode-a",
                "cluster-a",
                EvidenceSplit::Calibration,
                10,
                '1',
                'a',
                vec![key(DependenceKind::PhysicalAsset, "asset-1", '2')],
            ),
            unit(
                "episode-b",
                "cluster-b",
                EvidenceSplit::Calibration,
                90,
                '3',
                'b',
                vec![
                    key(DependenceKind::PhysicalAsset, "asset-1", '2'),
                    key(DependenceKind::CollectionSession, "session-7", '4'),
                ],
            ),
            unit(
                "episode-c",
                "cluster-c",
                EvidenceSplit::Calibration,
                30,
                '5',
                'c',
                vec![key(DependenceKind::CollectionSession, "session-7", '4')],
            ),
            unit(
                "episode-d",
                "cluster-d",
                EvidenceSplit::Calibration,
                20,
                '6',
                'd',
                vec![key(DependenceKind::ScenarioSeed, "seed-11", '7')],
            ),
        ]
    }

    #[test]
    fn transitive_common_causes_collapse_to_one_max_score_block() {
        let result = build_common_cause_closure(&digest('f'), &calibration_units()).unwrap();
        assert_eq!(result.calibration_raw_unit_count, 4);
        assert_eq!(result.calibration_effective_block_count, 2);
        let shared = result
            .blocks
            .iter()
            .find(|block| block.member_unit_ids.len() == 3)
            .unwrap();
        assert_eq!(
            shared.member_unit_ids,
            vec![
                "episode-a".to_string(),
                "episode-b".to_string(),
                "episode-c".to_string()
            ]
        );
        assert_eq!(shared.maximum_dangerous_score_units, 90);
        assert_eq!(shared.score_receipts_sha256.len(), 3);
    }

    #[test]
    fn cross_split_asset_dependency_is_rejected() {
        let units = vec![
            unit(
                "train-a",
                "cluster-a",
                EvidenceSplit::Training,
                1,
                '1',
                'a',
                vec![key(DependenceKind::PhysicalAsset, "asset-x", '2')],
            ),
            unit(
                "cal-a",
                "cluster-b",
                EvidenceSplit::Calibration,
                5,
                '3',
                'b',
                vec![key(DependenceKind::PhysicalAsset, "asset-x", '2')],
            ),
        ];
        assert!(matches!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::CrossSplitDependency { .. })
        ));
    }

    #[test]
    fn transitive_seed_and_trace_path_cannot_leak_splits() {
        let units = vec![
            unit(
                "cal-a",
                "cluster-a",
                EvidenceSplit::Calibration,
                1,
                '1',
                'a',
                vec![key(DependenceKind::ScenarioSeed, "seed", '2')],
            ),
            unit(
                "middle",
                "cluster-b",
                EvidenceSplit::Calibration,
                2,
                '3',
                'b',
                vec![
                    key(DependenceKind::ScenarioSeed, "seed", '2'),
                    key(DependenceKind::RawSourceTrace, "trace", '4'),
                ],
            ),
            unit(
                "heldout",
                "cluster-c",
                EvidenceSplit::HeldOutEvaluation,
                3,
                '5',
                'c',
                vec![key(DependenceKind::RawSourceTrace, "trace", '4')],
            ),
        ];
        assert!(matches!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::CrossSplitDependency { .. })
        ));
    }

    #[test]
    fn output_is_deterministic_under_input_permutation() {
        let units = calibration_units();
        let forward = build_common_cause_closure(&digest('f'), &units).unwrap();
        let mut reversed = units;
        reversed.reverse();
        let backward = build_common_cause_closure(&digest('f'), &reversed).unwrap();
        assert_eq!(forward, backward);
    }

    #[test]
    fn inconsistent_receipts_for_shared_key_fail_closed() {
        let units = vec![
            unit(
                "one",
                "cluster-one",
                EvidenceSplit::Calibration,
                1,
                '1',
                'a',
                vec![key(DependenceKind::ExternalShock, "shock", '2')],
            ),
            unit(
                "two",
                "cluster-two",
                EvidenceSplit::Calibration,
                2,
                '3',
                'b',
                vec![key(DependenceKind::ExternalShock, "shock", '4')],
            ),
        ];
        assert_eq!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::ConflictingDependenceReceipt {
                kind: DependenceKind::ExternalShock,
                value: "shock".into(),
            })
        );
    }

    #[test]
    fn duplicate_unit_identity_is_rejected() {
        let mut units = calibration_units();
        units[1].unit_id = units[0].unit_id.clone();
        assert_eq!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::DuplicateUnitId)
        );
    }

    #[test]
    fn distinct_units_remain_distinct_when_no_causes_overlap() {
        let units = vec![
            unit("a", "ca", EvidenceSplit::Calibration, 1, '1', 'a', vec![]),
            unit("b", "cb", EvidenceSplit::Calibration, 2, '2', 'b', vec![]),
        ];
        let result = build_common_cause_closure(&digest('f'), &units).unwrap();
        assert_eq!(result.calibration_effective_block_count, 2);
        assert_eq!(result.calibration_raw_unit_count, 2);
    }

    #[test]
    fn missing_calibration_split_is_rejected() {
        let units = vec![unit(
            "training",
            "train-cluster",
            EvidenceSplit::Training,
            1,
            '1',
            'a',
            vec![],
        )];
        assert_eq!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::MissingCalibrationBlocks)
        );
    }

    #[test]
    fn malformed_receipt_is_rejected() {
        let mut units = calibration_units();
        units[0].score_receipt_sha256 = "not-a-sha".into();
        assert_eq!(
            build_common_cause_closure(&digest('f'), &units),
            Err(ClosureError::InvalidSha256("score_receipt_sha256"))
        );
    }
}
