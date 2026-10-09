// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Explicit, validated per-load service allocation and restoration gating.
//!
//! This module is a deterministic simulation contract. It does not command
//! circuits, authenticate classification records, or establish electrical
//! protection or field safety. Provenance values are caller-supplied metadata,
//! not cryptographic attestations. A production deployment would need an
//! external trust anchor, review workflow, certified protection, and plant data.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadClass {
    /// Must be allocated ahead of deferrable and auxiliary loads.
    Critical,
    /// May be shed, but is prioritized ahead of auxiliary loads.
    Deferrable,
    /// Lowest default service tier; eligible deficits are intentional shedding.
    Auxiliary,
}

impl LoadClass {
    fn allocation_tier(self) -> u8 {
        match self {
            Self::Critical => 0,
            Self::Deferrable => 1,
            Self::Auxiliary => 2,
        }
    }

    fn is_critical(self) -> bool {
        matches!(self, Self::Critical)
    }
}

/// Explicit source label for a load classification.
///
/// These labels make assumptions visible in receipts. They are not themselves
/// trusted: in particular, a caller can construct a ReviewedRecord value, so
/// no result from this module should be interpreted as field authorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassificationProvenance {
    /// A named deterministic test/simulation scenario, not a field claim.
    SyntheticScenario { scenario_id: String },
    /// A reference to an external review record. The record and reviewer
    /// identities are descriptive only and are not authenticated here.
    ReviewedRecord {
        record_id: String,
        record_revision: String,
        reviewer_id: String,
    },
}

impl ClassificationProvenance {
    fn is_well_formed(&self) -> bool {
        match self {
            Self::SyntheticScenario { scenario_id } => valid_label(scenario_id),
            Self::ReviewedRecord {
                record_id,
                record_revision,
                reviewer_id,
            } => {
                valid_label(record_id)
                    && valid_label(record_revision)
                    && valid_label(reviewer_id)
            }
        }
    }

    pub fn is_synthetic_assumption(&self) -> bool {
        matches!(self, Self::SyntheticScenario { .. })
    }
}

/// Per-load configuration. A lower shed_priority means the load is more
/// eligible to be shed within its class; higher values are protected longer.
/// A lower restore_priority is considered earlier for restoration within its
/// class. Classes restore in Critical, Deferrable, Auxiliary order. Ties are
/// resolved deterministically by stable load_id, not by insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadRegistryEntry {
    pub load_id: String,
    pub class: LoadClass,
    pub rated_power_kw: f64,
    pub shed_priority: u16,
    pub restore_priority: u16,
    pub provenance: ClassificationProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadRegistryError {
    EmptyVersion,
    EmptyRegistry,
    MissingCriticalLoad,
    EmptyLoadId,
    DuplicateLoadId(String),
    InvalidRatedPower(String),
    InvalidProvenance(String),
    InvalidStepDuration,
    InvalidAvailableEnergy,
    MissingDemand(String),
    DuplicateDemand(String),
    UnknownDemandLoad(String),
    InvalidRequestedPower(String),
    RequestedPowerExceedsRating(String),
    InvalidDerivedLedger,
    InvalidRestorationPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadRegistry {
    version: String,
    entries: Vec<LoadRegistryEntry>,
    registry_digest: String,
}

impl LoadRegistry {
    pub fn new(
        version: impl Into<String>,
        entries: Vec<LoadRegistryEntry>,
    ) -> Result<Self, LoadRegistryError> {
        let version = version.into();
        if !valid_label(&version) {
            return Err(LoadRegistryError::EmptyVersion);
        }
        if entries.is_empty() {
            return Err(LoadRegistryError::EmptyRegistry);
        }

        let mut seen = BTreeMap::<String, ()>::new();
        for entry in &entries {
            if !valid_label(&entry.load_id) {
                return Err(LoadRegistryError::EmptyLoadId);
            }
            if seen.insert(entry.load_id.clone(), ()).is_some() {
                return Err(LoadRegistryError::DuplicateLoadId(entry.load_id.clone()));
            }
            if !entry.rated_power_kw.is_finite() || entry.rated_power_kw <= 0.0 {
                return Err(LoadRegistryError::InvalidRatedPower(entry.load_id.clone()));
            }
            if !entry.provenance.is_well_formed() {
                return Err(LoadRegistryError::InvalidProvenance(entry.load_id.clone()));
            }
        }

        if !entries.iter().any(|entry| entry.class.is_critical()) {
            return Err(LoadRegistryError::MissingCriticalLoad);
        }

        let registry_digest = canonical_registry_digest(&version, &entries);
        Ok(Self {
            version,
            entries,
            registry_digest,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    /// BLAKE3 digest of a domain-separated, length-prefixed canonical
    /// encoding of the version and all registry entries sorted by load ID.
    /// This establishes content identity, not authorship or authorization.
    pub fn registry_digest(&self) -> &str {
        &self.registry_digest
    }

    pub fn entries(&self) -> &[LoadRegistryEntry] {
        &self.entries
    }

    /// True when at least one critical classification is explicitly backed
    /// only by a synthetic scenario label. False does not imply that any
    /// referenced review record was authenticated or accepted.
    pub fn has_synthetic_criticality_assumptions(&self) -> bool {
        self.entries.iter().any(|entry| {
            entry.class.is_critical() && entry.provenance.is_synthetic_assumption()
        })
    }

    /// Stable restoration order is class-first, then lowest restore_priority,
    /// then ID. Actual restoration still requires the separate gate below.
    pub fn restoration_order(&self) -> Vec<&LoadRegistryEntry> {
        let mut ordered: Vec<&LoadRegistryEntry> = self.entries.iter().collect();
        ordered.sort_by(|left, right| {
            left.class
                .allocation_tier()
                .cmp(&right.class.allocation_tier())
                .then_with(|| left.restore_priority.cmp(&right.restore_priority))
                .then_with(|| left.load_id.cmp(&right.load_id))
        });
        ordered
    }

    /// Verify a ledger against this exact registry with the separate checker.
    /// The checker does not call this method or reuse the allocator, avoiding
    /// a circular "verification" that would simply trust producer logic.
    /// Success establishes internal consistency only; it does not authenticate
    /// the registry, provenance claims, demand inputs, or physical plant state.
    pub fn verify_ledger(&self, ledger: &LoadServiceLedger) -> bool {
        crate::load_registry_verifier::verify_load_service_ledger(self, ledger).is_ok()
    }

    /// Allocate a finite energy budget to a complete snapshot of load demands.
    ///
    /// Every registered load must appear exactly once, even if its requested
    /// power is zero. Demand is continuous/fractional in this model: a load
    /// may receive partial energy. Circuit-level all-or-nothing behavior must
    /// be modeled separately when the physical load requires it.
    ///
    /// The allocation order is Critical, Deferrable, Auxiliary; within a
    /// class, higher shed_priority is served first, then stable load ID. A
    /// deficit in a Critical load is reported as unserved; deficits in the
    /// other classes are reported as intentional shedding.
    pub fn allocate(
        &self,
        demands: &[LoadDemand],
        step_duration_hours: f64,
        available_energy_kwh: f64,
    ) -> Result<LoadServiceLedger, LoadRegistryError> {
        if !step_duration_hours.is_finite() || step_duration_hours <= 0.0 {
            return Err(LoadRegistryError::InvalidStepDuration);
        }
        if !available_energy_kwh.is_finite() || available_energy_kwh < 0.0 {
            return Err(LoadRegistryError::InvalidAvailableEnergy);
        }

        let mut demand_by_id = BTreeMap::<String, f64>::new();
        for demand in demands {
            if !valid_label(&demand.load_id) {
                return Err(LoadRegistryError::EmptyLoadId);
            }
            let Some(entry) = self
                .entries
                .iter()
                .find(|entry| entry.load_id == demand.load_id)
            else {
                return Err(LoadRegistryError::UnknownDemandLoad(demand.load_id.clone()));
            };
            if demand_by_id
                .insert(demand.load_id.clone(), demand.requested_power_kw)
                .is_some()
            {
                return Err(LoadRegistryError::DuplicateDemand(demand.load_id.clone()));
            }
            if !demand.requested_power_kw.is_finite() || demand.requested_power_kw < 0.0 {
                return Err(LoadRegistryError::InvalidRequestedPower(demand.load_id.clone()));
            }
            if demand.requested_power_kw > entry.rated_power_kw {
                return Err(LoadRegistryError::RequestedPowerExceedsRating(
                    demand.load_id.clone(),
                ));
            }
        }

        for entry in &self.entries {
            if !demand_by_id.contains_key(&entry.load_id) {
                return Err(LoadRegistryError::MissingDemand(entry.load_id.clone()));
            }
        }

        let mut ordered: Vec<&LoadRegistryEntry> = self.entries.iter().collect();
        ordered.sort_by(|left, right| {
            left.class
                .allocation_tier()
                .cmp(&right.class.allocation_tier())
                .then_with(|| right.shed_priority.cmp(&left.shed_priority))
                .then_with(|| left.load_id.cmp(&right.load_id))
        });

        let mut remaining_kwh = available_energy_kwh;
        let mut records = Vec::with_capacity(ordered.len());
        for entry in ordered {
            let Some(requested_power_kw) = demand_by_id.get(&entry.load_id).copied() else {
                return Err(LoadRegistryError::MissingDemand(entry.load_id.clone()));
            };
            let requested_kwh = requested_power_kw * step_duration_hours;
            if !requested_kwh.is_finite() {
                return Err(LoadRegistryError::InvalidRequestedPower(entry.load_id.clone()));
            }

            let served_kwh = requested_kwh.min(remaining_kwh).max(0.0);
            remaining_kwh = (remaining_kwh - served_kwh).max(0.0);
            let deficit_kwh = (requested_kwh - served_kwh).max(0.0);
            let (intentional_shed_kwh, unserved_kwh) = if entry.class.is_critical() {
                (0.0, deficit_kwh)
            } else {
                (deficit_kwh, 0.0)
            };

            records.push(LoadServiceRecord {
                load_id: entry.load_id.clone(),
                class: entry.class,
                provenance: entry.provenance.clone(),
                rated_power_kw: entry.rated_power_kw,
                shed_priority: entry.shed_priority,
                restore_priority: entry.restore_priority,
                requested_power_kw,
                requested_kwh,
                served_kwh,
                intentional_shed_kwh,
                unserved_kwh,
            });
        }

        let total_demand_kwh = records.iter().map(|record| record.requested_kwh).sum();
        let total_served_kwh = records.iter().map(|record| record.served_kwh).sum();
        let intentional_shed_kwh = records
            .iter()
            .map(|record| record.intentional_shed_kwh)
            .sum();
        let total_unserved_kwh = records.iter().map(|record| record.unserved_kwh).sum();
        let unused_energy_kwh = (available_energy_kwh - total_served_kwh).max(0.0);

        let ledger = LoadServiceLedger {
            registry_version: self.version.clone(),
            registry_digest: self.registry_digest.clone(),
            step_duration_hours,
            available_energy_kwh,
            total_demand_kwh,
            total_served_kwh,
            intentional_shed_kwh,
            total_unserved_kwh,
            unused_energy_kwh,
            records,
        };
        if !ledger.is_valid() {
            return Err(LoadRegistryError::InvalidDerivedLedger);
        }
        Ok(ledger)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadDemand {
    pub load_id: String,
    pub requested_power_kw: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadServiceRecord {
    pub load_id: String,
    pub class: LoadClass,
    pub provenance: ClassificationProvenance,
    pub rated_power_kw: f64,
    pub shed_priority: u16,
    pub restore_priority: u16,
    pub requested_power_kw: f64,
    pub requested_kwh: f64,
    pub served_kwh: f64,
    pub intentional_shed_kwh: f64,
    pub unserved_kwh: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadServiceLedger {
    pub registry_version: String,
    /// Digest of the exact registry configuration used for this allocation.
    pub registry_digest: String,
    pub step_duration_hours: f64,
    pub available_energy_kwh: f64,
    pub total_demand_kwh: f64,
    pub total_served_kwh: f64,
    pub intentional_shed_kwh: f64,
    pub total_unserved_kwh: f64,
    pub unused_energy_kwh: f64,
    pub records: Vec<LoadServiceRecord>,
}

impl LoadServiceLedger {
    pub fn energy_balance_residual_kwh(&self) -> f64 {
        self.total_demand_kwh
            - self.total_served_kwh
            - self.intentional_shed_kwh
            - self.total_unserved_kwh
    }

    /// Recompute the ledger invariants without trusting its summary fields.
    /// This is useful to an independent checker and deliberately public.
    pub fn is_valid(&self) -> bool {
        if !valid_label(&self.registry_version)
            || !valid_digest(&self.registry_digest)
            || !self.step_duration_hours.is_finite()
            || self.step_duration_hours <= 0.0
        {
            return false;
        }
        let summary_values = [
            self.available_energy_kwh,
            self.total_demand_kwh,
            self.total_served_kwh,
            self.intentional_shed_kwh,
            self.total_unserved_kwh,
            self.unused_energy_kwh,
        ];
        if summary_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return false;
        }
        if self.records.is_empty() {
            return false;
        }

        let mut seen = BTreeMap::<String, ()>::new();
        for record in &self.records {
            if !valid_label(&record.load_id)
                || seen.insert(record.load_id.clone(), ()).is_some()
                || !record.provenance.is_well_formed()
            {
                return false;
            }
            let values = [
                record.rated_power_kw,
                record.requested_power_kw,
                record.requested_kwh,
                record.served_kwh,
                record.intentional_shed_kwh,
                record.unserved_kwh,
            ];
            if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return false;
            }
            if record.rated_power_kw <= 0.0
                || record.requested_power_kw > record.rated_power_kw
                || !close_enough(
                    record.requested_power_kw * self.step_duration_hours,
                    record.requested_kwh,
                )
                || !close_enough(
                    record.requested_kwh,
                    record.served_kwh + record.intentional_shed_kwh + record.unserved_kwh,
                )
            {
                return false;
            }
            if record.served_kwh > record.requested_kwh + tolerance(record.requested_kwh) {
                return false;
            }
            if record.class.is_critical() {
                if !close_enough(record.intentional_shed_kwh, 0.0) {
                    return false;
                }
            } else if !close_enough(record.unserved_kwh, 0.0) {
                return false;
            }
        }

        let demand = self.records.iter().map(|record| record.requested_kwh).sum::<f64>();
        let served = self.records.iter().map(|record| record.served_kwh).sum::<f64>();
        let shed = self.records.iter().map(|record| record.intentional_shed_kwh).sum::<f64>();
        let unserved = self.records.iter().map(|record| record.unserved_kwh).sum::<f64>();
        close_enough(demand, self.total_demand_kwh)
            && close_enough(served, self.total_served_kwh)
            && close_enough(shed, self.intentional_shed_kwh)
            && close_enough(unserved, self.total_unserved_kwh)
            && close_enough(
                self.total_served_kwh + self.unused_energy_kwh,
                self.available_energy_kwh,
            )
            && self.total_served_kwh
                <= self.available_energy_kwh + tolerance(self.available_energy_kwh)
            && close_enough(self.energy_balance_residual_kwh(), 0.0)
    }
}

/// Fail-closed configuration for deciding when a previously shed load may be
/// considered for restoration. This is a gate/report only; it does not switch
/// a load. Stable observations, available reserve after restoration, and the
/// time since shedding are separate inputs because none can safely substitute
/// for the others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadRestorationPolicy {
    pub minimum_stable_steps: u32,
    pub minimum_reserve_after_restore_kwh: f64,
    pub minimum_shed_dwell_steps: u64,
}

impl LoadRestorationPolicy {
    pub fn validate(self) -> Result<Self, LoadRegistryError> {
        if self.minimum_stable_steps == 0
            || !self.minimum_reserve_after_restore_kwh.is_finite()
            || self.minimum_reserve_after_restore_kwh < 0.0
        {
            return Err(LoadRegistryError::InvalidRestorationPolicy);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadRestorationDecision {
    Eligible,
    ConditionsNotStable {
        consecutive_stable_steps: u32,
        required_stable_steps: u32,
    },
    InsufficientReserve,
    MinimumShedDwellNotMet {
        observed_steps: u64,
        required_steps: u64,
    },
    InvalidEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadRestorationGate {
    policy: LoadRestorationPolicy,
    consecutive_stable_steps: u32,
}

impl LoadRestorationGate {
    pub fn new(policy: LoadRestorationPolicy) -> Result<Self, LoadRegistryError> {
        Ok(Self {
            policy: policy.validate()?,
            consecutive_stable_steps: 0,
        })
    }

    pub fn consecutive_stable_steps(&self) -> u32 {
        self.consecutive_stable_steps
    }

    pub fn reset(&mut self) {
        self.consecutive_stable_steps = 0;
    }

    /// Assess a single observation. Any unstable or invalid reserve reading
    /// resets the stability streak. The reserve value must already account for
    /// the proposed restored load; this gate does not invent generation or
    /// battery capacity.
    pub fn assess(
        &mut self,
        stable_operating_conditions: bool,
        reserve_after_restore_kwh: f64,
        steps_since_shed: u64,
    ) -> LoadRestorationDecision {
        if !reserve_after_restore_kwh.is_finite() || reserve_after_restore_kwh < 0.0 {
            self.reset();
            return LoadRestorationDecision::InvalidEvidence;
        }
        if !stable_operating_conditions {
            self.reset();
            return LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 0,
                required_stable_steps: self.policy.minimum_stable_steps,
            };
        }

        self.consecutive_stable_steps = self.consecutive_stable_steps.saturating_add(1);
        if self.consecutive_stable_steps < self.policy.minimum_stable_steps {
            return LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: self.consecutive_stable_steps,
                required_stable_steps: self.policy.minimum_stable_steps,
            };
        }
        if reserve_after_restore_kwh < self.policy.minimum_reserve_after_restore_kwh {
            return LoadRestorationDecision::InsufficientReserve;
        }
        if steps_since_shed < self.policy.minimum_shed_dwell_steps {
            return LoadRestorationDecision::MinimumShedDwellNotMet {
                observed_steps: steps_since_shed,
                required_steps: self.policy.minimum_shed_dwell_steps,
            };
        }
        LoadRestorationDecision::Eligible
    }
}

fn valid_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn canonical_registry_digest(version: &str, entries: &[LoadRegistryEntry]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, b"symthaea-load-registry-digest-v1");
    hash_field(&mut hasher, version.as_bytes());

    let mut canonical_entries: Vec<&LoadRegistryEntry> = entries.iter().collect();
    canonical_entries.sort_by(|left, right| left.load_id.cmp(&right.load_id));
    hasher.update(&(canonical_entries.len() as u64).to_le_bytes());

    for entry in canonical_entries {
        hash_field(&mut hasher, entry.load_id.as_bytes());
        let class = match entry.class {
            LoadClass::Critical => "critical",
            LoadClass::Deferrable => "deferrable",
            LoadClass::Auxiliary => "auxiliary",
        };
        hash_field(&mut hasher, class.as_bytes());
        hasher.update(&entry.rated_power_kw.to_bits().to_le_bytes());
        hasher.update(&entry.shed_priority.to_le_bytes());
        hasher.update(&entry.restore_priority.to_le_bytes());
        match &entry.provenance {
            ClassificationProvenance::SyntheticScenario { scenario_id } => {
                hash_field(&mut hasher, b"synthetic-scenario");
                hash_field(&mut hasher, scenario_id.as_bytes());
            }
            ClassificationProvenance::ReviewedRecord {
                record_id,
                record_revision,
                reviewer_id,
            } => {
                hash_field(&mut hasher, b"reviewed-record-label");
                hash_field(&mut hasher, record_id.as_bytes());
                hash_field(&mut hasher, record_revision.as_bytes());
                hash_field(&mut hasher, reviewer_id.as_bytes());
            }
        }
    }

    hasher.finalize().to_hex().to_string()
}

fn tolerance(value: f64) -> f64 {
    1e-9 * value.abs().max(1.0)
}

fn close_enough(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= tolerance(left.abs().max(right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> ClassificationProvenance {
        ClassificationProvenance::SyntheticScenario {
            scenario_id: "unit-test-fixture-v1".to_string(),
        }
    }

    fn entry(
        load_id: &str,
        class: LoadClass,
        rated_power_kw: f64,
        shed_priority: u16,
        restore_priority: u16,
    ) -> LoadRegistryEntry {
        LoadRegistryEntry {
            load_id: load_id.to_string(),
            class,
            rated_power_kw,
            shed_priority,
            restore_priority,
            provenance: provenance(),
        }
    }

    fn registry() -> LoadRegistry {
        LoadRegistry::new(
            "test-registry-v1",
            vec![
                entry("critical-water", LoadClass::Critical, 2.0, 10, 2),
                entry("deferrable-pump", LoadClass::Deferrable, 3.0, 1, 3),
                entry("deferrable-comms", LoadClass::Deferrable, 1.0, 10, 1),
                entry("aux-lighting", LoadClass::Auxiliary, 2.0, 0, 4),
            ],
        )
        .unwrap()
    }

    fn demands() -> Vec<LoadDemand> {
        vec![
            LoadDemand { load_id: "critical-water".into(), requested_power_kw: 2.0 },
            LoadDemand { load_id: "deferrable-pump".into(), requested_power_kw: 3.0 },
            LoadDemand { load_id: "deferrable-comms".into(), requested_power_kw: 1.0 },
            LoadDemand { load_id: "aux-lighting".into(), requested_power_kw: 2.0 },
        ]
    }

    #[test]
    fn registry_rejects_empty_registry_and_duplicate_or_blank_ids() {
        assert_eq!(
            LoadRegistry::new("v1", vec![]).unwrap_err(),
            LoadRegistryError::EmptyRegistry
        );
        assert_eq!(
            LoadRegistry::new(
                "v1",
                vec![
                    entry("duplicate", LoadClass::Critical, 1.0, 1, 1),
                    entry("duplicate", LoadClass::Auxiliary, 1.0, 1, 1),
                ],
            ).unwrap_err(),
            LoadRegistryError::DuplicateLoadId("duplicate".into())
        );
        assert_eq!(
            LoadRegistry::new(
                "v1",
                vec![entry("aux", LoadClass::Auxiliary, 1.0, 1, 1)],
            )
            .unwrap_err(),
            LoadRegistryError::MissingCriticalLoad
        );
        assert_eq!(
            LoadRegistry::new("v1", vec![entry(" ", LoadClass::Critical, 1.0, 1, 1)])
                .unwrap_err(),
            LoadRegistryError::EmptyLoadId
        );
    }

    #[test]
    fn registry_rejects_nonfinite_or_nonpositive_ratings_and_missing_provenance() {
        for rating in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                LoadRegistry::new("v1", vec![entry("load", LoadClass::Critical, rating, 1, 1)])
                    .unwrap_err(),
                LoadRegistryError::InvalidRatedPower("load".into())
            );
        }
        let mut ungrounded = entry("critical", LoadClass::Critical, 1.0, 1, 1);
        ungrounded.provenance = ClassificationProvenance::ReviewedRecord {
            record_id: "record-1".into(),
            record_revision: "".into(),
            reviewer_id: "reviewer-1".into(),
        };
        assert_eq!(
            LoadRegistry::new("v1", vec![ungrounded]).unwrap_err(),
            LoadRegistryError::InvalidProvenance("critical".into())
        );
    }

    #[test]
    fn registry_digest_is_stable_across_input_order_but_binds_configuration() {
        let original = registry();
        let mut reversed_entries = original.entries().to_vec();
        reversed_entries.reverse();
        let reordered = LoadRegistry::new(original.version(), reversed_entries).unwrap();
        assert_eq!(original.registry_digest(), reordered.registry_digest());

        let mut changed_priority = original.entries().to_vec();
        changed_priority[0].restore_priority += 1;
        let changed_policy = LoadRegistry::new(original.version(), changed_priority).unwrap();
        assert_ne!(original.registry_digest(), changed_policy.registry_digest());

        let mut changed_rating = original.entries().to_vec();
        changed_rating[0].rated_power_kw += 0.125;
        let changed_rating = LoadRegistry::new(original.version(), changed_rating).unwrap();
        assert_ne!(original.registry_digest(), changed_rating.registry_digest());

        let mut changed_class = original.entries().to_vec();
        changed_class[1].class = LoadClass::Auxiliary;
        let changed_class = LoadRegistry::new(original.version(), changed_class).unwrap();
        assert_ne!(original.registry_digest(), changed_class.registry_digest());

        let mut changed_provenance = original.entries().to_vec();
        changed_provenance[1].provenance = ClassificationProvenance::SyntheticScenario {
            scenario_id: "different-scenario".into(),
        };
        let changed_provenance =
            LoadRegistry::new(original.version(), changed_provenance).unwrap();
        assert_ne!(original.registry_digest(), changed_provenance.registry_digest());

        let changed_version =
            LoadRegistry::new("test-registry-v2", original.entries().to_vec()).unwrap();
        assert_ne!(original.registry_digest(), changed_version.registry_digest());
    }

    #[test]
    fn ledger_rejects_malformed_registry_digest() {
        let mut ledger = registry().allocate(&demands(), 1.0, 3.0).unwrap();
        ledger.registry_digest = "not-a-digest".into();
        assert!(!ledger.is_valid());
        assert!(!registry().verify_ledger(&ledger));
    }

    #[test]
    fn registry_explicitly_reports_synthetic_criticality_as_unverified_metadata() {
        assert!(registry().has_synthetic_criticality_assumptions());
        let reviewed = LoadRegistry::new(
            "review-label-test-v1",
            vec![LoadRegistryEntry {
                provenance: ClassificationProvenance::ReviewedRecord {
                    record_id: "external-record-17".into(),
                    record_revision: "rev-4".into(),
                    reviewer_id: "reviewer-identity-3".into(),
                },
                ..entry("critical-water", LoadClass::Critical, 2.0, 10, 2)
            }],
        ).unwrap();
        assert!(!reviewed.has_synthetic_criticality_assumptions());
        // The ReviewedRecord variant is only a caller-supplied label; this
        // module performs no authentication or authorization.
    }

    #[test]
    fn allocator_prioritizes_class_then_shed_priority_deterministically() {
        let ledger = registry().allocate(&demands(), 1.0, 2.5).unwrap();
        assert!(ledger.is_valid(), "{ledger:?}");
        assert_eq!(ledger.registry_digest, registry().registry_digest());
        assert_eq!(ledger.total_demand_kwh, 8.0);
        assert_eq!(ledger.total_served_kwh, 2.5);
        assert_eq!(ledger.total_unserved_kwh, 0.0);
        assert_eq!(ledger.intentional_shed_kwh, 5.5);

        let lookup = |id: &str| {
            ledger
                .records
                .iter()
                .find(|record| record.load_id == id)
                .unwrap()
        };
        assert_eq!(lookup("critical-water").served_kwh, 2.0);
        assert_eq!(lookup("deferrable-comms").served_kwh, 0.5);
        assert_eq!(lookup("deferrable-pump").served_kwh, 0.0);
        assert_eq!(lookup("aux-lighting").served_kwh, 0.0);
        assert!(ledger.energy_balance_residual_kwh().abs() < 1e-12);
    }

    #[test]
    fn critical_shortfall_is_unserved_not_relabelled_as_intentional_shedding() {
        let ledger = registry().allocate(&demands(), 1.0, 0.5).unwrap();
        assert!(ledger.is_valid());
        assert_eq!(ledger.total_served_kwh, 0.5);
        assert_eq!(ledger.total_unserved_kwh, 1.5);
        assert_eq!(ledger.intentional_shed_kwh, 6.0);
        let critical = ledger
            .records
            .iter()
            .find(|record| record.class == LoadClass::Critical)
            .unwrap();
        assert_eq!(critical.unserved_kwh, 1.5);
        assert_eq!(critical.intentional_shed_kwh, 0.0);
    }

    #[test]
    fn demand_snapshot_must_be_complete_unique_known_and_within_ratings() {
        let registry = registry();
        let mut input = demands();
        input.pop();
        assert_eq!(
            registry.allocate(&input, 1.0, 10.0).unwrap_err(),
            LoadRegistryError::MissingDemand("aux-lighting".into())
        );

        let mut input = demands();
        input.push(input[0].clone());
        assert_eq!(
            registry.allocate(&input, 1.0, 10.0).unwrap_err(),
            LoadRegistryError::DuplicateDemand("critical-water".into())
        );

        let mut input = demands();
        input[0].load_id = "unregistered".into();
        assert_eq!(
            registry.allocate(&input, 1.0, 10.0).unwrap_err(),
            LoadRegistryError::UnknownDemandLoad("unregistered".into())
        );

        let mut input = demands();
        input[0].requested_power_kw = 2.01;
        assert_eq!(
            registry.allocate(&input, 1.0, 10.0).unwrap_err(),
            LoadRegistryError::RequestedPowerExceedsRating("critical-water".into())
        );
    }

    #[test]
    fn allocator_rejects_invalid_time_energy_and_requested_power() {
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                registry().allocate(&demands(), dt, 1.0).unwrap_err(),
                LoadRegistryError::InvalidStepDuration
            );
        }
        for energy in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                registry().allocate(&demands(), 1.0, energy).unwrap_err(),
                LoadRegistryError::InvalidAvailableEnergy
            );
        }
        let mut input = demands();
        input[0].requested_power_kw = f64::NAN;
        assert_eq!(
            registry().allocate(&input, 1.0, 1.0).unwrap_err(),
            LoadRegistryError::InvalidRequestedPower("critical-water".into())
        );
    }

    #[test]
    fn restoration_order_uses_separate_restore_priority_and_stable_id_ties() {
        let reg = LoadRegistry::new(
            "restore-test-v1",
            vec![
                entry("zeta", LoadClass::Auxiliary, 1.0, 1, 1),
                entry("alpha", LoadClass::Deferrable, 1.0, 1, 1),
                entry("later", LoadClass::Critical, 1.0, 1, 3),
                entry("middle", LoadClass::Deferrable, 1.0, 1, 2),
            ],
        ).unwrap();
        let ids: Vec<&str> = reg
            .restoration_order()
            .iter()
            .map(|entry| entry.load_id.as_str())
            .collect();
        assert_eq!(ids, vec!["later", "alpha", "middle", "zeta"]);
    }

    fn restoration_gate() -> LoadRestorationGate {
        LoadRestorationGate::new(LoadRestorationPolicy {
            minimum_stable_steps: 3,
            minimum_reserve_after_restore_kwh: 5.0,
            minimum_shed_dwell_steps: 4,
        }).unwrap()
    }

    #[test]
    fn restoration_requires_stability_reserve_and_shed_dwell_independently() {
        let mut gate = restoration_gate();
        assert_eq!(
            gate.assess(true, 10.0, 10),
            LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 1,
                required_stable_steps: 3,
            }
        );
        assert!(matches!(
            gate.assess(true, 10.0, 10),
            LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 2,
                ..
            }
        ));
        assert_eq!(
            gate.assess(true, 10.0, 3),
            LoadRestorationDecision::MinimumShedDwellNotMet {
                observed_steps: 3,
                required_steps: 4,
            }
        );
        assert_eq!(
            gate.assess(true, 4.99, 4),
            LoadRestorationDecision::InsufficientReserve
        );
        assert_eq!(gate.assess(true, 5.0, 4), LoadRestorationDecision::Eligible);
    }

    #[test]
    fn instability_or_invalid_reserve_resets_stability_streak_fail_closed() {
        let mut gate = restoration_gate();
        assert!(matches!(
            gate.assess(true, 10.0, 10),
            LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 1,
                ..
            }
        ));
        assert!(matches!(
            gate.assess(false, 10.0, 10),
            LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 0,
                ..
            }
        ));
        assert_eq!(gate.consecutive_stable_steps(), 0);
        assert_eq!(
            gate.assess(true, f64::NAN, 10),
            LoadRestorationDecision::InvalidEvidence
        );
        assert_eq!(gate.consecutive_stable_steps(), 0);
        assert!(matches!(
            gate.assess(true, 10.0, 10),
            LoadRestorationDecision::ConditionsNotStable {
                consecutive_stable_steps: 1,
                ..
            }
        ));
    }

    #[test]
    fn restoration_policy_rejects_zero_stability_and_invalid_reserve_floor() {
        for policy in [
            LoadRestorationPolicy {
                minimum_stable_steps: 0,
                minimum_reserve_after_restore_kwh: 0.0,
                minimum_shed_dwell_steps: 0,
            },
            LoadRestorationPolicy {
                minimum_stable_steps: 1,
                minimum_reserve_after_restore_kwh: f64::NAN,
                minimum_shed_dwell_steps: 0,
            },
            LoadRestorationPolicy {
                minimum_stable_steps: 1,
                minimum_reserve_after_restore_kwh: -0.1,
                minimum_shed_dwell_steps: 0,
            },
        ] {
            assert_eq!(
                LoadRestorationGate::new(policy).unwrap_err(),
                LoadRegistryError::InvalidRestorationPolicy
            );
        }
    }

    #[test]
    fn ledger_validator_recomputes_and_rejects_forged_summary_and_metadata() {
        let reg = registry();
        let mut ledger = reg.allocate(&demands(), 1.0, 3.0).unwrap();
        assert!(ledger.is_valid());
        assert!(reg.verify_ledger(&ledger));

        ledger.total_served_kwh += 1.0;
        assert!(!ledger.is_valid());
        assert!(!reg.verify_ledger(&ledger));

        let mut relabeled = reg.allocate(&demands(), 1.0, 3.0).unwrap();
        let record = relabeled.records.iter_mut()
            .find(|record| record.load_id == "deferrable-comms")
            .unwrap();
        record.class = LoadClass::Auxiliary;
        assert!(
            relabeled.is_valid(),
            "the internal ledger may still balance after a classification claim changes"
        );
        assert!(
            !reg.verify_ledger(&relabeled),
            "the exact registry must reject the changed classification"
        );
    }
}
