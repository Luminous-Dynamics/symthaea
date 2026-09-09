// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//!
//! Internal error bridge for revalidating non-Serde wrappers inside distributed
//! evidence admission. These lower-layer failures indicate that a supposedly
//! validated in-process context/policy no longer matches its own invariants, so the
//! evidence layer fails closed at the corresponding context boundary.

use crate::distributed_evidence::DistributedEvidenceError;
use crate::distributed_state::DistributedStateError;
use crate::failure_domain::FailureDomainPolicyError;

impl From<DistributedStateError> for DistributedEvidenceError {
    fn from(_: DistributedStateError) -> Self {
        Self::ClaimContextMismatch
    }
}

impl From<FailureDomainPolicyError> for DistributedEvidenceError {
    fn from(_: FailureDomainPolicyError) -> Self {
        Self::FailureDomainPolicyContextMismatch
    }
}
