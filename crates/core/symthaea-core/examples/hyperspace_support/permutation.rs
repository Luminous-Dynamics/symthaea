// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Experiment-only coordinate-permutation adapter shared by HYPERSPACE-002 controls.
//!
//! This module is intentionally outside the production library API. It preserves
//! the exact `hyperspace-coordinate-permutation-v1` identity and mapping semantics
//! originally preregistered by HYPERSPACE-002 while allowing independent controls
//! to exercise the very same adapter implementation.

use std::error::Error;

use symthaea_core::continuous_reachability::{
    ContinuousReachabilityError, ContinuousValidityOracle, ContinuousValidityOracleProfile,
    OracleVerdict,
};
use symthaea_core::hyperspace_benchmark::{FiniteWShellOracle, HyperspaceDimension};
use symthaea_core::sampling_reachability::BoundedEuclideanSamplingOracle;
use symthaea_core::state_space::{EuclideanSpace, StateSpace};

/// Exact experiment-only bijection between canonical `(x,y,z,w)` and an external
/// four-coordinate representation.
#[derive(Clone, Debug)]
pub struct PermutationOracle {
    canonical: FiniteWShellOracle,
    canonical_to_external: [usize; 4],
    external_to_canonical: [usize; 4],
    sampling_min: Vec<f64>,
    sampling_max: Vec<f64>,
    profile: ContinuousValidityOracleProfile,
}

impl PermutationOracle {
    /// Construct the exact HYPERSPACE-002 placement where canonical `w` appears
    /// at `external_w_axis` and canonical xyz occupy the remaining external axes
    /// in ascending order.
    pub fn from_external_w_axis(
        canonical: FiniteWShellOracle,
        external_w_axis: usize,
    ) -> Result<Self, Box<dyn Error>> {
        if canonical.benchmark_dimension() != HyperspaceDimension::R4 {
            return Err("permutation wrapper requires canonical R4 oracle".into());
        }
        if external_w_axis >= 4 {
            return Err(format!("external w axis must be in 0..4, got {external_w_axis}").into());
        }
        let spatial_external: Vec<usize> =
            (0..4).filter(|axis| *axis != external_w_axis).collect();
        Self::new(
            canonical,
            [
                spatial_external[0],
                spatial_external[1],
                spatial_external[2],
                external_w_axis,
            ],
        )
    }

    /// Construct any exact permutation, expressed as canonical-axis -> external-axis.
    pub fn new(
        canonical: FiniteWShellOracle,
        canonical_to_external: [usize; 4],
    ) -> Result<Self, Box<dyn Error>> {
        if canonical.benchmark_dimension() != HyperspaceDimension::R4 {
            return Err("permutation wrapper requires canonical R4 oracle".into());
        }

        let mut seen = [false; 4];
        for axis in canonical_to_external {
            if axis >= 4 || seen[axis] {
                return Err(format!("not a permutation of 0..4: {canonical_to_external:?}").into());
            }
            seen[axis] = true;
        }

        let mut external_to_canonical = [0_usize; 4];
        for (canonical_axis, external_axis) in canonical_to_external.iter().copied().enumerate() {
            external_to_canonical[external_axis] = canonical_axis;
        }

        let mut sampling_min = vec![0.0; 4];
        let mut sampling_max = vec![0.0; 4];
        for external_axis in 0..4 {
            let canonical_axis = external_to_canonical[external_axis];
            sampling_min[external_axis] = canonical.sampling_min()[canonical_axis];
            sampling_max[external_axis] = canonical.sampling_max()[canonical_axis];
        }

        let state_space = EuclideanSpace::new(4);
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&canonical.profile().identity());
        parameters.extend(canonical_to_external.iter().map(|axis| *axis as u8));
        let profile = ContinuousValidityOracleProfile::new(
            state_space.profile().identity(),
            "hyperspace-coordinate-permutation-v1",
            parameters,
        )?;

        Ok(Self {
            canonical,
            canonical_to_external,
            external_to_canonical,
            sampling_min,
            sampling_max,
            profile,
        })
    }

    /// Map one external representation back into canonical `(x,y,z,w)`.
    pub fn to_canonical(
        &self,
        external: &[f64],
    ) -> Result<Vec<f64>, ContinuousReachabilityError> {
        require_four_finite(external, "external")?;
        let mut canonical = vec![0.0; 4];
        for external_axis in 0..4 {
            canonical[self.external_to_canonical[external_axis]] = external[external_axis];
        }
        Ok(canonical)
    }

    /// Map one canonical `(x,y,z,w)` state into the external representation.
    pub fn to_external(
        &self,
        canonical: &[f64],
    ) -> Result<Vec<f64>, ContinuousReachabilityError> {
        require_four_finite(canonical, "canonical")?;
        let mut external = vec![0.0; 4];
        for canonical_axis in 0..4 {
            external[self.canonical_to_external[canonical_axis]] = canonical[canonical_axis];
        }
        Ok(external)
    }

    /// Exact canonical-axis -> external-axis permutation.
    pub fn permutation(&self) -> [usize; 4] {
        self.canonical_to_external
    }
}

impl ContinuousValidityOracle for PermutationOracle {
    fn profile(&self) -> &ContinuousValidityOracleProfile {
        &self.profile
    }

    fn state_verdict(&self, state: &[f64]) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical.state_verdict(&self.to_canonical(state)?)
    }

    fn segment_verdict(
        &self,
        from: &[f64],
        to: &[f64],
    ) -> Result<OracleVerdict, ContinuousReachabilityError> {
        self.canonical
            .segment_verdict(&self.to_canonical(from)?, &self.to_canonical(to)?)
    }
}

impl BoundedEuclideanSamplingOracle for PermutationOracle {
    fn sampling_min(&self) -> &[f64] {
        &self.sampling_min
    }

    fn sampling_max(&self) -> &[f64] {
        &self.sampling_max
    }
}

fn require_four_finite(
    state: &[f64],
    role: &str,
) -> Result<(), ContinuousReachabilityError> {
    if state.len() != 4 || state.iter().any(|value| !value.is_finite()) {
        return Err(ContinuousReachabilityError::OracleEvaluation {
            reason: format!("{role} shell state must contain exactly four finite coordinates"),
        });
    }
    Ok(())
}
