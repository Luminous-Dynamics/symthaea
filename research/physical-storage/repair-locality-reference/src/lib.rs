//! Deterministic read/repair-locality reference for physical storage.
//!
//! This crate scores an explicit helper-read plan; it does not make the plan
//! mathematically valid. A separate algebra verifier must set
//! independent_algebra_check = Some(true) before a repair can be accepted.
//! Physical seek/energy semantics remain medium-specific and are not inferred
//! from abstract tile regions.

use std::collections::HashSet;

pub const LRC_7_4_GENERATOR: [[u8; 4]; 7] = [
    [1, 0, 0, 0], // D0
    [0, 1, 0, 0], // D1
    [0, 0, 1, 0], // D2
    [0, 0, 0, 1], // D3
    [1, 1, 0, 0], // local parity for D0,D1
    [0, 0, 1, 1], // local parity for D2,D3
    [1, 2, 4, 8], // global parity
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    Available,
    Missing,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairSource {
    pub source_id: String,
    pub region_id: String,
    pub layer: u16,
    pub bytes_read: u64,
    pub availability: Availability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairPlan {
    pub target_id: String,
    pub requested_bytes: u64,
    /// Ordered physical read plan; order is preserved for access-transition scoring.
    pub helpers: Vec<RepairSource>,
    /// Must be supplied by an independent algebra verifier, not the planner.
    pub independent_algebra_check: Option<bool>,
    pub expected_sha256: Option<String>,
    pub observed_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairDisposition {
    Accepted,
    Failed,
    Incomplete,
    Unverified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairReceipt {
    pub target_id: String,
    pub disposition: RepairDisposition,
    pub requested_bytes: u64,
    pub bytes_read: u64,
    pub read_operations: usize,
    pub unique_regions: usize,
    pub layer_transitions: usize,
    /// Ceiling of (bytes_read / requested_bytes) * 1000.
    pub read_amplification_milli: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairError {
    EmptyTargetId,
    ZeroRequestedBytes,
    NoHelpers,
    EmptySourceId,
    DuplicateSourceId,
    ReadByteOverflow,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Score an explicit repair plan. This calculates access cost but does not
/// independently prove that the helpers can reconstruct the target.
pub fn score_repair(plan: &RepairPlan) -> Result<RepairReceipt, RepairError> {
    if plan.target_id.trim().is_empty() {
        return Err(RepairError::EmptyTargetId);
    }
    if plan.requested_bytes == 0 {
        return Err(RepairError::ZeroRequestedBytes);
    }
    if plan.helpers.is_empty() {
        return Err(RepairError::NoHelpers);
    }

    let mut seen_ids = HashSet::with_capacity(plan.helpers.len());
    let mut regions = HashSet::with_capacity(plan.helpers.len());
    let mut bytes_read = 0u64;
    let mut layer_transitions = 0usize;
    let mut previous_layer: Option<u16> = None;
    let mut missing = false;
    let mut unknown = false;

    for helper in &plan.helpers {
        if helper.source_id.trim().is_empty() {
            return Err(RepairError::EmptySourceId);
        }
        if !seen_ids.insert(helper.source_id.as_str()) {
            return Err(RepairError::DuplicateSourceId);
        }
        regions.insert(helper.region_id.as_str());
        bytes_read = bytes_read
            .checked_add(helper.bytes_read)
            .ok_or(RepairError::ReadByteOverflow)?;
        if previous_layer.is_some_and(|layer| layer != helper.layer) {
            layer_transitions += 1;
        }
        previous_layer = Some(helper.layer);

        match helper.availability {
            Availability::Available => {}
            Availability::Missing => missing = true,
            Availability::Unknown => unknown = true,
        }
    }

    let disposition = if missing {
        RepairDisposition::Failed
    } else if unknown {
        RepairDisposition::Incomplete
    } else {
        match plan.independent_algebra_check {
            Some(false) => RepairDisposition::Failed,
            None => RepairDisposition::Unverified,
            Some(true) => match (
                plan.expected_sha256.as_deref(),
                plan.observed_sha256.as_deref(),
            ) {
                (Some(expected), Some(observed))
                    if is_sha256_hex(expected)
                        && is_sha256_hex(observed)
                        && expected.eq_ignore_ascii_case(observed) =>
                {
                    RepairDisposition::Accepted
                }
                (Some(expected), Some(observed))
                    if is_sha256_hex(expected) && is_sha256_hex(observed) =>
                {
                    RepairDisposition::Failed
                }
                _ => RepairDisposition::Incomplete,
            },
        }
    };

    let amplification = (u128::from(bytes_read) * 1000)
        .div_ceil(u128::from(plan.requested_bytes))
        .min(u128::from(u64::MAX)) as u64;

    Ok(RepairReceipt {
        target_id: plan.target_id.clone(),
        disposition,
        requested_bytes: plan.requested_bytes,
        bytes_read,
        read_operations: plan.helpers.len(),
        unique_regions: regions.len(),
        layer_transitions,
        read_amplification_milli: amplification,
    })
}

fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1d; // reduction byte for full polynomial 0x11d
        }
        b >>= 1;
    }
    product
}

fn gf_pow(mut value: u8, mut exponent: u16) -> u8 {
    let mut result = 1u8;
    while exponent > 0 {
        if exponent & 1 != 0 {
            result = gf_mul(result, value);
        }
        value = gf_mul(value, value);
        exponent >>= 1;
    }
    result
}

fn gf_inv(value: u8) -> Option<u8> {
    if value == 0 {
        None
    } else {
        Some(gf_pow(value, 254))
    }
}

fn matrix_rank(rows: &[[u8; 4]]) -> usize {
    let mut matrix = rows.to_vec();
    let mut rank = 0usize;

    for column in 0..4 {
        let Some(pivot) = (rank..matrix.len()).find(|&row| matrix[row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let Some(inverse) = gf_inv(matrix[rank][column]) else {
            continue;
        };
        for entry in column..4 {
            matrix[rank][entry] = gf_mul(matrix[rank][entry], inverse);
        }
        for row in 0..matrix.len() {
            if row == rank {
                continue;
            }
            let factor = matrix[row][column];
            if factor == 0 {
                continue;
            }
            for entry in column..4 {
                matrix[row][entry] ^= gf_mul(factor, matrix[rank][entry]);
            }
        }
        rank += 1;
        if rank == 4 {
            break;
        }
    }
    rank
}

/// Proves that any two shard erasures leave four independent generator rows.
/// This is a finite algebra check of the declared matrix, not a physical result.
pub fn verifies_any_two_erasures() -> bool {
    for first in 0..7 {
        for second in first + 1..7 {
            let survivors: Vec<[u8; 4]> = (0..7)
                .filter(|&index| index != first && index != second)
                .map(|index| LRC_7_4_GENERATOR[index])
                .collect();
            if matrix_rank(&survivors) != 4 {
                return false;
            }
        }
    }
    true
}

/// The declared code is not MDS(7,4): at least one three-erasure pattern fails.
/// This negative property prevents advertising a stronger distance than proved.
pub fn has_three_erasure_limit() -> bool {
    for first in 0..7 {
        for second in first + 1..7 {
            for third in second + 1..7 {
                let survivors: Vec<[u8; 4]> = (0..7)
                    .filter(|&index| index != first && index != second && index != third)
                    .map(|index| LRC_7_4_GENERATOR[index])
                    .collect();
                if matrix_rank(&survivors) < 4 {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: &str, region: &str, layer: u16, bytes: u64) -> RepairSource {
        RepairSource {
            source_id: id.into(),
            region_id: region.into(),
            layer,
            bytes_read: bytes,
            availability: Availability::Available,
        }
    }

    fn verified_plan(helpers: Vec<RepairSource>) -> RepairPlan {
        RepairPlan {
            target_id: "object-001".into(),
            requested_bytes: 4096,
            helpers,
            independent_algebra_check: Some(true),
            expected_sha256: Some("a".repeat(64)),
            observed_sha256: Some("a".repeat(64)),
        }
    }

    #[test]
    fn local_repair_reads_two_helpers_and_has_half_the_rs_read_amplification() {
        let local = score_repair(&verified_plan(vec![
            source("D1", "region-1", 0, 4096),
            source("P0", "region-1", 0, 4096),
        ]))
        .unwrap();
        let rs = score_repair(&verified_plan(vec![
            source("S0", "region-1", 0, 4096),
            source("S1", "region-2", 0, 4096),
            source("S2", "region-3", 1, 4096),
            source("S3", "region-4", 1, 4096),
        ]))
        .unwrap();

        assert_eq!(local.disposition, RepairDisposition::Accepted);
        assert_eq!(local.read_operations, 2);
        assert_eq!(local.read_amplification_milli, 2000);
        assert_eq!(rs.read_operations, 4);
        assert_eq!(rs.read_amplification_milli, 4000);
    }

    #[test]
    fn access_regions_and_layer_transitions_are_counted_separately() {
        let receipt = score_repair(&verified_plan(vec![
            source("A", "r1", 0, 1024),
            source("B", "r1", 0, 1024),
            source("C", "r2", 1, 2048),
        ]))
        .unwrap();
        assert_eq!(receipt.unique_regions, 2);
        assert_eq!(receipt.layer_transitions, 1);
        assert_eq!(receipt.bytes_read, 4096);
    }

    #[test]
    fn unavailable_unknown_and_unverified_sources_cannot_pass() {
        let mut missing = source("A", "r1", 0, 4096);
        missing.availability = Availability::Missing;
        assert_eq!(
            score_repair(&verified_plan(vec![missing])).unwrap().disposition,
            RepairDisposition::Failed
        );

        let mut unknown = source("A", "r1", 0, 4096);
        unknown.availability = Availability::Unknown;
        assert_eq!(
            score_repair(&verified_plan(vec![unknown])).unwrap().disposition,
            RepairDisposition::Incomplete
        );

        let mut unverified = verified_plan(vec![source("A", "r1", 0, 4096)]);
        unverified.independent_algebra_check = None;
        assert_eq!(
            score_repair(&unverified).unwrap().disposition,
            RepairDisposition::Unverified
        );
    }

    #[test]
    fn digest_mismatch_blocks_repair_acceptance() {
        let mut plan = verified_plan(vec![source("A", "r1", 0, 4096)]);
        plan.observed_sha256 = Some("b".repeat(64));
        assert_eq!(
            score_repair(&plan).unwrap().disposition,
            RepairDisposition::Failed
        );
    }

    #[test]
    fn duplicate_helpers_and_zero_requests_are_rejected() {
        let plan = verified_plan(vec![
            source("A", "r1", 0, 1024),
            source("A", "r2", 0, 1024),
        ]);
        assert_eq!(score_repair(&plan), Err(RepairError::DuplicateSourceId));

        let mut zero = verified_plan(vec![source("A", "r1", 0, 1024)]);
        zero.requested_bytes = 0;
        assert_eq!(score_repair(&zero), Err(RepairError::ZeroRequestedBytes));
    }

    #[test]
    fn synthetic_local_code_recovers_any_two_erasures() {
        assert!(verifies_any_two_erasures());
        assert!(has_three_erasure_limit());
    }
}
