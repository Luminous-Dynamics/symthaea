//! Candidate-side memory topology and logical state placement validation.
//!
//! Workload-side state demand belongs to QARCH-021. This module describes
//! candidate memory regions, directed transfers, explicit routes, and the
//! placement of logical state classes. A valid topology is not a measurement;
//! comparison requires an exact join to QARCH-020 evidence and QARCH-021 regime.
use std::collections::{HashMap, HashSet};

use crate::architecture_evidence::{
    ArchitectureEvidenceError, ArchitectureEvidenceOrigin, ArchitectureEvidenceRecord,
    validate_architecture_evidence_record, validate_architecture_evidence_record_for_comparison,
};
use crate::architecture_portfolio::{MemoryInterval, MetricInterval};
use crate::architecture_regime::{
    ArchitectureWorkloadRegime, WorkloadRegimeError, validate_workload_regime_profile,
};

/// Origin of the candidate-side memory-topology description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTopologySourceOrigin {
    /// Declared from the architecture design by its owner.
    DeclaredArchitecture,
    /// Transcribed from a published hardware specification.
    PublishedSpecification,
    /// Directly observed on an identified physical target.
    DirectMeasurement,
    /// Produced by an executed software, cycle, or hardware simulation.
    Simulation,
    /// Produced by a synthesis or implementation tool.
    Synthesis,
    /// Produced by an analytical cost or capacity model.
    AnalyticalModel,
    /// Synthetic fixture data for validator tests only.
    SyntheticFixture,
}

/// Provenance for the topology itself, independent of the workload result receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTopologySource {
    /// Source maturity and origin.
    pub origin: MemoryTopologySourceOrigin,
    /// DOI, URL, repository path, report ID, or synthetic fixture locator.
    pub source_reference: String,
    /// SHA-256 digest of the source artifact.
    pub source_artifact_sha256: String,
    /// SHA-256 digest of the extraction or measurement procedure when applicable.
    pub extraction_procedure_sha256: Option<String>,
}

/// Physical or logical memory tier represented as a topology node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Processor register file.
    Registers,
    /// Explicitly managed local scratchpad.
    Scratchpad,
    /// Level-one cache.
    L1Cache,
    /// Level-two cache.
    L2Cache,
    /// Level-three or last-level cache.
    L3Cache,
    /// Static RAM or embedded SRAM.
    Sram,
    /// High-bandwidth memory.
    Hbm,
    /// DDR memory.
    Ddr,
    /// Low-power DDR memory.
    Lpddr,
    /// CXL-attached memory.
    CxlMemory,
    /// Persistent storage such as NVMe.
    PersistentStorage,
    /// Remote memory accessed across a network or fabric.
    RemoteMemory,
    /// A separately identified tier described by the source.
    Other,
}

/// Physical or logical scope that owns a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Thread-local region.
    Thread,
    /// Core-local region.
    Core,
    /// Cluster-local region.
    Cluster,
    /// Package-local region.
    Package,
    /// Accelerator or device-local region.
    Device,
    /// Host-local region.
    Host,
    /// Rack-level region.
    Rack,
    /// Region outside the local host or rack.
    Remote,
}

/// Persistence semantics of a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPersistence {
    /// Contents are lost when power is removed.
    Volatile,
    /// Contents survive power loss but are not necessarily durable.
    NonVolatile,
    /// Contents have an explicit persistence/durability contract.
    Persistent,
}

/// Coherence or consistency mechanism associated with a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCoherenceModel {
    /// No hardware coherence guarantee.
    None,
    /// Single-writer discipline.
    SingleWriter,
    /// Directory-based coherence.
    Directory,
    /// MESI-family cache coherence.
    Mesi,
    /// MOESI-family cache coherence.
    Moesi,
    /// Message-passing consistency.
    MessagePassing,
    /// A separately specified coherence model.
    Other,
}

/// Communication fabric joining a source and destination region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTransferFabric {
    /// On-chip network or local interconnect.
    OnChipNetwork,
    /// DDR/LPDDR memory channel.
    DdrMemoryChannel,
    /// HBM stack or memory channel.
    HbmMemoryChannel,
    /// PCI Express.
    Pcie,
    /// Compute Express Link.
    Cxl,
    /// Ethernet-family network.
    Ethernet,
    /// InfiniBand-family network.
    InfiniBand,
    /// Optical or co-packaged chip-to-chip link.
    OpticalChipToChip,
    /// Event-address or asynchronous routing network.
    EventAddressNetwork,
    /// A separately named fabric.
    Other,
}

/// Transfer operation semantics. The source/destination IDs define data direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTransferDirection {
    /// Data is read from the source region and delivered to the destination.
    Read,
    /// Data is written from the source region to the destination region.
    Write,
    /// A bidirectional link is summarized by this edge.
    Bidirectional,
}

/// Placement strategy for one logical state class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPlacementMode {
    /// State is explicitly pinned to this region.
    Pinned,
    /// Residency is managed by a cache or transparent runtime.
    CacheManaged,
    /// State is consumed as a stream rather than retained as a resident set.
    Streamed,
    /// A copy of the logical state is maintained in this region.
    Replicated,
    /// State may be evicted or moved to a lower tier.
    Spillable,
    /// State can be reconstructed rather than kept resident.
    Reconstructed,
}

/// Strength of the residency guarantee for a state placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryResidencyMode {
    /// The declared live set is guaranteed resident in this region.
    Guaranteed,
    /// Residency is best-effort and may be evicted.
    BestEffort,
    /// Data is loaded only when requested.
    OnDemand,
    /// Data passes through without a resident working set guarantee.
    Streamed,
    /// A replica is kept in this region.
    Replicated,
}

/// Consistency semantics required by a logical state placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryConsistencyModel {
    /// A single writer owns updates.
    SingleWriter,
    /// Multiple writers may update the state.
    MultiWriter,
    /// Updates converge eventually.
    Eventual,
    /// Sequentially consistent accesses.
    SequentiallyConsistent,
    /// Updates follow a transactional contract.
    Transactional,
    /// The state does not require a consistency model.
    NotApplicable,
}

/// Scenario under which the topology or its performance was characterized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTopologyScenario {
    /// State and indexes must be loaded from a cold state.
    ColdStart,
    /// Repeated operation with warm/resident state.
    WarmSteadyState,
    /// Concurrent requests or competing workloads apply pressure.
    ConcurrentPressure,
    /// The live set exceeds an explicitly declared memory capacity.
    CapacityOverflow,
    /// State moves between memory regions or tiers.
    Migration,
    /// Checkpoint, restore, or state recovery is exercised.
    Recovery,
}

/// One memory region in the candidate topology graph.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRegion {
    /// Stable region identifier.
    pub region_id: String,
    /// Memory tier or medium.
    pub tier: MemoryTier,
    /// Scope of the region.
    pub scope: MemoryScope,
    /// Usable capacity interval, if known.
    pub capacity_bytes: Option<MemoryInterval>,
    /// Existing reserved/background occupancy, excluding the logical state placements
    /// listed in this topology record. Required to prove guaranteed placement capacity.
    pub occupancy_bytes: Option<MemoryInterval>,
    /// Read access latency in seconds, if measured.
    pub read_latency_seconds: Option<MetricInterval>,
    /// Write access latency in seconds, if measured.
    pub write_latency_seconds: Option<MetricInterval>,
    /// Sustained read bandwidth in bytes/second, if measured.
    pub read_bandwidth_bytes_per_second: Option<MetricInterval>,
    /// Sustained write bandwidth in bytes/second, if measured.
    pub write_bandwidth_bytes_per_second: Option<MetricInterval>,
    /// Directly measured read energy in joules/byte, if available.
    pub read_energy_joules_per_byte: Option<MetricInterval>,
    /// Directly measured write energy in joules/byte, if available.
    pub write_energy_joules_per_byte: Option<MetricInterval>,
    /// Access granularity in bytes, if declared.
    pub access_granularity_bytes: Option<u64>,
    /// Volatility/persistence semantics, if known.
    pub persistence: Option<MemoryPersistence>,
    /// Coherence or consistency mechanism, if applicable.
    pub coherence_model: Option<MemoryCoherenceModel>,
}

/// One directed transfer path between two memory regions.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryTransferEdge {
    /// Stable transfer-edge identifier.
    pub edge_id: String,
    /// Source memory region.
    pub source_region_id: String,
    /// Destination memory region.
    pub destination_region_id: String,
    /// Operation represented by this edge.
    pub direction: MemoryTransferDirection,
    /// Physical or logical communication fabric.
    pub fabric: MemoryTransferFabric,
    /// Unique QARCH-021 boundary identity used to account for these bytes.
    pub communication_boundary_id: String,
    /// Conversion/serialization latency in seconds, if measured separately.
    pub conversion_latency_seconds: Option<MetricInterval>,
    /// Conversion energy in joules/bit, if directly measured.
    pub conversion_energy_joules_per_bit: Option<MetricInterval>,
}

/// An explicit ordered route through transfer edges.
///
/// The whole topology may contain cycles (e.g. coherent fabrics). A declared
/// route can independently require an acyclic path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRoute {
    /// Stable route identifier.
    pub route_id: String,
    /// Expected source region.
    pub source_region_id: String,
    /// Expected destination region.
    pub destination_region_id: String,
    /// Ordered transfer edge IDs.
    pub edge_ids: Vec<String>,
    /// Reject a route that revisits any region when true.
    pub require_acyclic: bool,
}

/// Logical state placement in one memory region.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStatePlacement {
    /// Stable semantic state-class identifier.
    pub state_class_id: String,
    /// Region where the state resides or is processed.
    pub region_id: String,
    /// Placement strategy.
    pub placement_mode: MemoryPlacementMode,
    /// Residency guarantee.
    pub residency_mode: MemoryResidencyMode,
    /// Live-state bytes for this placement, if known.
    pub live_bytes: Option<MemoryInterval>,
    /// Reads per second, if measured.
    pub read_rate_per_second: Option<MetricInterval>,
    /// Writes per second, if measured.
    pub write_rate_per_second: Option<MetricInterval>,
    /// State updates per second, if measured.
    pub update_rate_hz: Option<MetricInterval>,
    /// Bytes moved per migration event, if measured.
    pub migration_bytes_per_event: Option<MemoryInterval>,
    /// Migration latency in seconds, if measured.
    pub migration_latency_seconds: Option<MetricInterval>,
    /// Versioned precision contract for this state class.
    pub precision_contract_id: String,
    /// Consistency contract for this state class.
    pub consistency_model: MemoryConsistencyModel,
}

/// Source-bound candidate memory topology, joined to one QARCH-020/QARCH-021 record.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureMemoryTopology {
    /// Stable topology-record identifier.
    pub topology_record_id: String,
    /// Exact architecture identity from the bound observation.
    pub architecture_id: String,
    /// Exact observation identity; also joins the QARCH-020 evidence receipt.
    pub observation_record_id: String,
    /// Exact workload fingerprint from the observation and regime.
    pub workload_fingerprint: String,
    /// Exact QARCH-021 regime profile ID.
    pub workload_regime_profile_id: String,
    /// Exact comparison-contract fingerprint.
    pub comparison_contract_fingerprint: String,
    /// Provenance for the topology source itself.
    pub source: MemoryTopologySource,
    /// Declared memory regions.
    pub memory_regions: Vec<MemoryRegion>,
    /// Directed transfer edges.
    pub transfer_edges: Vec<MemoryTransferEdge>,
    /// Explicit end-to-end routes through transfer edges.
    pub routes: Vec<MemoryRoute>,
    /// Logical state placements.
    pub state_placements: Vec<MemoryStatePlacement>,
    /// Scenarios covered by this topology description or measurement.
    pub scenarios: Vec<MemoryTopologyScenario>,
    /// Optional human-readable caveat.
    pub notes: Option<String>,
}

/// Validation and evidence-binding failures for memory topology records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTopologyError {
    /// A required identity or source locator is blank.
    EmptyField,
    /// A required source SHA-256 digest is malformed.
    InvalidDigest,
    /// Direct/simulated/analytical origin has no extraction procedure digest.
    MissingExtractionProcedure,
    /// Required region, placement, or scenario collection is empty.
    EmptyCollection,
    /// A region identifier occurs more than once.
    DuplicateMemoryRegionId,
    /// A transfer-edge identifier occurs more than once.
    DuplicateTransferEdgeId,
    /// A route identifier occurs more than once.
    DuplicateMemoryRouteId,
    /// A state class is placed more than once in the same region.
    DuplicateStatePlacement,
    /// An edge or placement refers to an unknown memory region.
    UnknownMemoryRegion,
    /// A route refers to an edge that does not exist.
    UnknownTransferEdge,
    /// Two edges claim the same accounting boundary in one topology record.
    DuplicateBoundaryAccounting,
    /// A numeric metric interval is non-finite, negative, or inverted.
    InvalidMetricInterval,
    /// A byte interval is inverted.
    InvalidByteInterval,
    /// A count or granularity is zero where a positive value is required.
    InvalidPositiveCount,
    /// A measured occupancy definitely exceeds declared capacity.
    MemoryCapacityExceeded,
    /// Guaranteed residency has no known existing/background occupancy bound.
    UnknownOccupancyForGuaranteedPlacement,
    /// Capacity intervals overlap so a guaranteed fit cannot be proven.
    MemoryCapacityNotProven,
    /// Guaranteed residency has no known capacity.
    UnknownCapacityForGuaranteedPlacement,
    /// Guaranteed residency has no known live-set size.
    MissingLiveSetForGuaranteedPlacement,
    /// The aggregate placement upper bound exceeds the capacity upper bound.
    PlacementCapacityExceeded,
    /// The aggregate placement upper bound exceeds the conservative capacity lower bound.
    PlacementCapacityNotProven,
    /// A route contains no transfer edges.
    EmptyMemoryRoute,
    /// Route edges do not connect in their declared order.
    DisconnectedMemoryRoute,
    /// Route's declared source or destination disagrees with its edges.
    RouteEndpointMismatch,
    /// A route requiring an acyclic path revisits a region.
    MemoryRouteCycle,
    /// A scenario identifier is duplicated.
    DuplicateMemoryScenario,
    /// Topology does not bind to the exact observation.
    ObservationMismatch,
    /// Topology does not bind to the exact architecture.
    ArchitectureMismatch,
    /// Topology does not bind to the exact workload fingerprint.
    WorkloadMismatch,
    /// Topology does not bind to the exact workload-regime profile.
    WorkloadRegimeMismatch,
    /// Topology does not bind to the exact comparison contract.
    ComparisonContractMismatch,
    /// The topology precision contract differs from the workload contract.
    PrecisionContractMismatch,
    /// Provenance origin contradicts the QARCH-020 evidence origin.
    EvidenceOriginMismatch,
    /// Synthetic fixtures cannot be admitted to evidence comparison.
    SyntheticFixtureNotComparable,
    /// Underlying QARCH-020 evidence validation failed.
    Evidence(ArchitectureEvidenceError),
    /// Underlying QARCH-021 regime validation failed.
    Regime(WorkloadRegimeError),
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_metric(interval: Option<MetricInterval>, allow_zero: bool) -> bool {
    interval.is_none_or(|value| {
        value.lower.is_finite()
            && value.upper.is_finite()
            && (if allow_zero { value.lower >= 0.0 } else { value.lower > 0.0 })
            && value.lower <= value.upper
    })
}

fn valid_bytes(interval: Option<MemoryInterval>) -> bool {
    interval.is_none_or(|value| value.lower_bytes <= value.upper_bytes)
}

/// Validate the local structure and internal invariants of a memory-topology record.
///
/// This does not establish that any region, bandwidth, latency, or energy value
/// describes a real device. Source-bound comparison requires the binding validator.
pub fn validate_memory_topology(
    topology: &ArchitectureMemoryTopology,
) -> Result<(), MemoryTopologyError> {
    for value in [
        topology.topology_record_id.as_str(),
        topology.architecture_id.as_str(),
        topology.observation_record_id.as_str(),
        topology.workload_fingerprint.as_str(),
        topology.workload_regime_profile_id.as_str(),
        topology.comparison_contract_fingerprint.as_str(),
        topology.source.source_reference.as_str(),
    ] {
        if !non_empty(value) {
            return Err(MemoryTopologyError::EmptyField);
        }
    }

    if !is_sha256(&topology.source.source_artifact_sha256) {
        return Err(MemoryTopologyError::InvalidDigest);
    }
    match topology.source.origin {
        MemoryTopologySourceOrigin::DirectMeasurement
        | MemoryTopologySourceOrigin::Simulation
        | MemoryTopologySourceOrigin::Synthesis
        | MemoryTopologySourceOrigin::AnalyticalModel => {
            let Some(digest) = topology.source.extraction_procedure_sha256.as_deref() else {
                return Err(MemoryTopologyError::MissingExtractionProcedure);
            };
            if !is_sha256(digest) {
                return Err(MemoryTopologyError::InvalidDigest);
            }
        }
        MemoryTopologySourceOrigin::DeclaredArchitecture
        | MemoryTopologySourceOrigin::PublishedSpecification
        | MemoryTopologySourceOrigin::SyntheticFixture => {
            if topology
                .source
                .extraction_procedure_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
            {
                return Err(MemoryTopologyError::InvalidDigest);
            }
        }
    }

    if topology.memory_regions.is_empty()
        || topology.state_placements.is_empty()
        || topology.scenarios.is_empty()
    {
        return Err(MemoryTopologyError::EmptyCollection);
    }
    let mut scenarios = HashSet::new();
    if topology.scenarios.iter().any(|scenario| !scenarios.insert(*scenario)) {
        return Err(MemoryTopologyError::DuplicateMemoryScenario);
    }

    let mut region_ids = HashSet::new();
    let mut region_by_id = HashMap::new();
    for region in &topology.memory_regions {
        if !non_empty(&region.region_id) {
            return Err(MemoryTopologyError::EmptyField);
        }
        if !region_ids.insert(region.region_id.as_str()) {
            return Err(MemoryTopologyError::DuplicateMemoryRegionId);
        }
        if region
            .capacity_bytes
            .is_some_and(|value| value.lower_bytes == 0 || value.lower_bytes > value.upper_bytes)
            || !valid_bytes(region.occupancy_bytes)
        {
            return Err(MemoryTopologyError::InvalidByteInterval);
        }
        if let (Some(capacity), Some(occupancy)) =
            (region.capacity_bytes, region.occupancy_bytes)
        {
            if occupancy.lower_bytes > capacity.upper_bytes {
                return Err(MemoryTopologyError::MemoryCapacityExceeded);
            }
            if occupancy.upper_bytes > capacity.lower_bytes {
                return Err(MemoryTopologyError::MemoryCapacityNotProven);
            }
        }
        for metric in [
            region.read_latency_seconds,
            region.write_latency_seconds,
            region.read_bandwidth_bytes_per_second,
            region.write_bandwidth_bytes_per_second,
        ] {
            if !valid_metric(metric, false) {
                return Err(MemoryTopologyError::InvalidMetricInterval);
            }
        }
        for metric in [
            region.read_energy_joules_per_byte,
            region.write_energy_joules_per_byte,
        ] {
            if !valid_metric(metric, true) {
                return Err(MemoryTopologyError::InvalidMetricInterval);
            }
        }
        if region.access_granularity_bytes == Some(0) {
            return Err(MemoryTopologyError::InvalidPositiveCount);
        }
        region_by_id.insert(region.region_id.as_str(), region);
    }

    let mut edge_ids = HashSet::new();
    let mut boundary_edges: HashMap<&str, &MemoryTransferEdge> = HashMap::new();
    let mut edge_by_id = HashMap::new();
    for edge in &topology.transfer_edges {
        for value in [
            edge.edge_id.as_str(),
            edge.source_region_id.as_str(),
            edge.destination_region_id.as_str(),
            edge.communication_boundary_id.as_str(),
        ] {
            if !non_empty(value) {
                return Err(MemoryTopologyError::EmptyField);
            }
        }
        if !edge_ids.insert(edge.edge_id.as_str()) {
            return Err(MemoryTopologyError::DuplicateTransferEdgeId);
        }
        if let Some(existing) = boundary_edges.get(edge.communication_boundary_id.as_str()) {
            let opposite_full_duplex_pair =
                existing.source_region_id == edge.destination_region_id
                    && existing.destination_region_id == edge.source_region_id
                    && matches!(
                        (existing.direction, edge.direction),
                        (MemoryTransferDirection::Read, MemoryTransferDirection::Write)
                            | (MemoryTransferDirection::Write, MemoryTransferDirection::Read)
                    );
            if !opposite_full_duplex_pair {
                return Err(MemoryTopologyError::DuplicateBoundaryAccounting);
            }
        } else {
            boundary_edges.insert(edge.communication_boundary_id.as_str(), edge);
        }
        if !region_by_id.contains_key(edge.source_region_id.as_str())
            || !region_by_id.contains_key(edge.destination_region_id.as_str())
        {
            return Err(MemoryTopologyError::UnknownMemoryRegion);
        }
        if !valid_metric(edge.conversion_latency_seconds, false)
            || !valid_metric(edge.conversion_energy_joules_per_bit, true)
        {
            return Err(MemoryTopologyError::InvalidMetricInterval);
        }
        edge_by_id.insert(edge.edge_id.as_str(), edge);
    }

    let mut route_ids = HashSet::new();
    for route in &topology.routes {
        for value in [
            route.route_id.as_str(),
            route.source_region_id.as_str(),
            route.destination_region_id.as_str(),
        ] {
            if !non_empty(value) {
                return Err(MemoryTopologyError::EmptyField);
            }
        }
        if !route_ids.insert(route.route_id.as_str()) {
            return Err(MemoryTopologyError::DuplicateMemoryRouteId);
        }
        if route.edge_ids.is_empty() {
            return Err(MemoryTopologyError::EmptyMemoryRoute);
        }
        let mut visited_regions = HashSet::new();
        visited_regions.insert(route.source_region_id.as_str());
        let mut current_region = route.source_region_id.as_str();
        for edge_id in &route.edge_ids {
            let Some(edge) = edge_by_id.get(edge_id.as_str()) else {
                return Err(MemoryTopologyError::UnknownTransferEdge);
            };
            if edge.source_region_id.as_str() != current_region {
                return Err(MemoryTopologyError::DisconnectedMemoryRoute);
            }
            current_region = edge.destination_region_id.as_str();
            if route.require_acyclic && !visited_regions.insert(current_region) {
                return Err(MemoryTopologyError::MemoryRouteCycle);
            }
        }
        if current_region != route.destination_region_id.as_str() {
            return Err(MemoryTopologyError::RouteEndpointMismatch);
        }
    }

    let mut placement_pairs = HashSet::new();
    let mut placement_lower_by_region: HashMap<&str, u64> = HashMap::new();
    let mut placement_upper_by_region: HashMap<&str, u64> = HashMap::new();
    let mut guaranteed_regions = HashSet::new();
    for placement in &topology.state_placements {
        if !non_empty(&placement.state_class_id)
            || !non_empty(&placement.region_id)
            || !non_empty(&placement.precision_contract_id)
        {
            return Err(MemoryTopologyError::EmptyField);
        }
        if !region_by_id.contains_key(placement.region_id.as_str()) {
            return Err(MemoryTopologyError::UnknownMemoryRegion);
        }
        if !placement_pairs.insert((
            placement.state_class_id.as_str(),
            placement.region_id.as_str(),
        )) {
            return Err(MemoryTopologyError::DuplicateStatePlacement);
        }
        if !valid_bytes(placement.live_bytes)
            || !valid_bytes(placement.migration_bytes_per_event)
        {
            return Err(MemoryTopologyError::InvalidByteInterval);
        }
        for metric in [
            placement.read_rate_per_second,
            placement.write_rate_per_second,
            placement.update_rate_hz,
        ] {
            if !valid_metric(metric, true) {
                return Err(MemoryTopologyError::InvalidMetricInterval);
            }
        }
        if !valid_metric(placement.migration_latency_seconds, false) {
            return Err(MemoryTopologyError::InvalidMetricInterval);
        }
        if placement.residency_mode == MemoryResidencyMode::Guaranteed {
            let region = region_by_id[placement.region_id.as_str()];
            if region.capacity_bytes.is_none() {
                return Err(MemoryTopologyError::UnknownCapacityForGuaranteedPlacement);
            }
            if region.occupancy_bytes.is_none() {
                return Err(MemoryTopologyError::UnknownOccupancyForGuaranteedPlacement);
            }
            guaranteed_regions.insert(placement.region_id.as_str());
            if placement.live_bytes.is_none() {
                return Err(MemoryTopologyError::MissingLiveSetForGuaranteedPlacement);
            }
        }
        if let Some(live_bytes) = placement.live_bytes {
            let lower_current = placement_lower_by_region
                .get(placement.region_id.as_str())
                .copied()
                .unwrap_or(0);
            let lower_total = lower_current
                .checked_add(live_bytes.lower_bytes)
                .ok_or(MemoryTopologyError::PlacementCapacityExceeded)?;
            placement_lower_by_region.insert(placement.region_id.as_str(), lower_total);

            let upper_current = placement_upper_by_region
                .get(placement.region_id.as_str())
                .copied()
                .unwrap_or(0);
            let upper_total = upper_current
                .checked_add(live_bytes.upper_bytes)
                .ok_or(MemoryTopologyError::PlacementCapacityExceeded)?;
            placement_upper_by_region.insert(placement.region_id.as_str(), upper_total);
        }
    }

    for (region_id, placement_upper) in placement_upper_by_region {
        let region = region_by_id[region_id];
        if let (Some(capacity), Some(occupancy)) = (region.capacity_bytes, region.occupancy_bytes) {
            let placement_lower = placement_lower_by_region[region_id];
            let minimum_total = occupancy
                .lower_bytes
                .checked_add(placement_lower)
                .ok_or(MemoryTopologyError::PlacementCapacityExceeded)?;
            let maximum_total = occupancy
                .upper_bytes
                .checked_add(placement_upper)
                .ok_or(MemoryTopologyError::PlacementCapacityExceeded)?;

            if minimum_total > capacity.upper_bytes {
                return Err(MemoryTopologyError::PlacementCapacityExceeded);
            }
            if guaranteed_regions.contains(region_id) && maximum_total > capacity.lower_bytes {
                return Err(MemoryTopologyError::PlacementCapacityNotProven);
            }
        }
    }

    Ok(())
}

/// Validate a topology against the exact architecture observation and workload regime.
///
/// The observation identity is the join key to the QARCH-020 evidence receipt:
/// that receipt is already bound to the same observation, workload, and contract.
pub fn validate_memory_topology_binding(
    topology: &ArchitectureMemoryTopology,
    evidence: &ArchitectureEvidenceRecord,
    regime: &ArchitectureWorkloadRegime,
) -> Result<(), MemoryTopologyError> {
    validate_memory_topology(topology)?;
    validate_architecture_evidence_record(evidence)
        .map_err(MemoryTopologyError::Evidence)?;
    validate_workload_regime_profile(regime).map_err(MemoryTopologyError::Regime)?;

    if topology.observation_record_id != evidence.observation.record_id
        || topology.observation_record_id != evidence.receipt.observation_record_id
        || topology.observation_record_id != regime.observation_record_id
    {
        return Err(MemoryTopologyError::ObservationMismatch);
    }
    if topology.architecture_id != evidence.observation.architecture_id {
        return Err(MemoryTopologyError::ArchitectureMismatch);
    }
    if topology.workload_fingerprint != evidence.observation.workload_fingerprint
        || topology.workload_fingerprint != evidence.receipt.workload_fingerprint
        || topology.workload_fingerprint != regime.workload_fingerprint
    {
        return Err(MemoryTopologyError::WorkloadMismatch);
    }
    if topology.workload_regime_profile_id != regime.regime_profile_id {
        return Err(MemoryTopologyError::WorkloadRegimeMismatch);
    }
    if topology.comparison_contract_fingerprint
        != evidence.observation.comparison_contract_fingerprint
        || topology.comparison_contract_fingerprint
            != evidence.receipt.comparison_contract_fingerprint
        || topology.comparison_contract_fingerprint
            != regime.comparison_contract_fingerprint
    {
        return Err(MemoryTopologyError::ComparisonContractMismatch);
    }
    if topology
        .state_placements
        .iter()
        .any(|placement| placement.precision_contract_id != regime.precision_contract_id)
    {
        return Err(MemoryTopologyError::PrecisionContractMismatch);
    }

    match topology.source.origin {
        MemoryTopologySourceOrigin::DirectMeasurement
            if evidence.receipt.origin != ArchitectureEvidenceOrigin::DirectMeasurement =>
        {
            return Err(MemoryTopologyError::EvidenceOriginMismatch);
        }
        MemoryTopologySourceOrigin::Simulation | MemoryTopologySourceOrigin::Synthesis
            if evidence.receipt.origin != ArchitectureEvidenceOrigin::SimulationExecution =>
        {
            return Err(MemoryTopologyError::EvidenceOriginMismatch);
        }
        MemoryTopologySourceOrigin::AnalyticalModel
            if evidence.receipt.origin != ArchitectureEvidenceOrigin::AnalyticalReference =>
        {
            return Err(MemoryTopologyError::EvidenceOriginMismatch);
        }
        _ => {}
    }

    Ok(())
}

/// Validate a topology and all source-bound evidence preconditions for comparison.
///
/// Synthetic fixtures are useful for testing the contract but cannot enter a
/// measured architecture ranking.
pub fn validate_memory_topology_for_comparison(
    topology: &ArchitectureMemoryTopology,
    evidence: &ArchitectureEvidenceRecord,
    regime: &ArchitectureWorkloadRegime,
) -> Result<(), MemoryTopologyError> {
    validate_memory_topology_binding(topology, evidence, regime)?;
    validate_architecture_evidence_record_for_comparison(evidence)
        .map_err(MemoryTopologyError::Evidence)?;
    if topology.source.origin == MemoryTopologySourceOrigin::SyntheticFixture {
        return Err(MemoryTopologyError::SyntheticFixtureNotComparable);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_evidence::{
        ArchitectureBenchmarkSuite, ArchitectureEvidenceReceipt,
    };
    use crate::architecture_portfolio::{
        ArchitectureEvidenceClass, ArchitectureMeasurementBoundary,
        ArchitectureResourceEnvelope, ArchitectureWorkloadKind, ComputeArchitectureFamily,
    };
    use crate::architecture_regime::{
        RegimeCharacterizationOrigin, WorkloadExecutionMode, WorkloadPrecisionProfile,
        WorkloadTemporalPattern,
    };

    fn metric(lower: f64, upper: f64) -> MetricInterval {
        MetricInterval { lower, upper }
    }

    fn bytes(lower: u64, upper: u64) -> MemoryInterval {
        MemoryInterval { lower_bytes: lower, upper_bytes: upper }
    }

    fn base_topology() -> ArchitectureMemoryTopology {
        ArchitectureMemoryTopology {
            topology_record_id: "topology-test".to_owned(),
            architecture_id: "device-observation-1".to_owned(),
            observation_record_id: "observation-1".to_owned(),
            workload_fingerprint: "workload-fingerprint-1".to_owned(),
            workload_regime_profile_id: "regime-1".to_owned(),
            comparison_contract_fingerprint: "contract-1".to_owned(),
            source: MemoryTopologySource {
                origin: MemoryTopologySourceOrigin::SyntheticFixture,
                source_reference: "synthetic://memory-topology/test".to_owned(),
                source_artifact_sha256: "a".repeat(64),
                extraction_procedure_sha256: None,
            },
            memory_regions: vec![
                MemoryRegion {
                    region_id: "dram".to_owned(),
                    tier: MemoryTier::Ddr,
                    scope: MemoryScope::Host,
                    capacity_bytes: Some(bytes(4096, 4096)),
                    occupancy_bytes: Some(bytes(128, 256)),
                    read_latency_seconds: Some(metric(1.0e-8, 2.0e-8)),
                    write_latency_seconds: Some(metric(1.0e-8, 3.0e-8)),
                    read_bandwidth_bytes_per_second: Some(metric(1.0e9, 2.0e9)),
                    write_bandwidth_bytes_per_second: Some(metric(1.0e9, 2.0e9)),
                    read_energy_joules_per_byte: None,
                    write_energy_joules_per_byte: None,
                    access_granularity_bytes: Some(64),
                    persistence: Some(MemoryPersistence::Volatile),
                    coherence_model: Some(MemoryCoherenceModel::Mesi),
                },
                MemoryRegion {
                    region_id: "scratch".to_owned(),
                    tier: MemoryTier::Scratchpad,
                    scope: MemoryScope::Device,
                    capacity_bytes: Some(bytes(1024, 1024)),
                    occupancy_bytes: Some(bytes(64, 128)),
                    read_latency_seconds: Some(metric(1.0e-9, 2.0e-9)),
                    write_latency_seconds: Some(metric(1.0e-9, 2.0e-9)),
                    read_bandwidth_bytes_per_second: Some(metric(1.0e10, 2.0e10)),
                    write_bandwidth_bytes_per_second: Some(metric(1.0e10, 2.0e10)),
                    read_energy_joules_per_byte: None,
                    write_energy_joules_per_byte: None,
                    access_granularity_bytes: Some(16),
                    persistence: Some(MemoryPersistence::Volatile),
                    coherence_model: Some(MemoryCoherenceModel::None),
                },
            ],
            transfer_edges: vec![MemoryTransferEdge {
                edge_id: "dram-to-scratch".to_owned(),
                source_region_id: "dram".to_owned(),
                destination_region_id: "scratch".to_owned(),
                direction: MemoryTransferDirection::Read,
                fabric: MemoryTransferFabric::Pcie,
                communication_boundary_id: "boundary-dram-scratch".to_owned(),
                conversion_latency_seconds: None,
                conversion_energy_joules_per_bit: None,
            }],
            routes: vec![MemoryRoute {
                route_id: "dram-to-scratch-route".to_owned(),
                source_region_id: "dram".to_owned(),
                destination_region_id: "scratch".to_owned(),
                edge_ids: vec!["dram-to-scratch".to_owned()],
                require_acyclic: true,
            }],
            state_placements: vec![MemoryStatePlacement {
                state_class_id: "query-state".to_owned(),
                region_id: "scratch".to_owned(),
                placement_mode: MemoryPlacementMode::Pinned,
                residency_mode: MemoryResidencyMode::Guaranteed,
                live_bytes: Some(bytes(128, 256)),
                read_rate_per_second: Some(metric(1.0, 100.0)),
                write_rate_per_second: Some(metric(1.0, 10.0)),
                update_rate_hz: Some(metric(1.0, 10.0)),
                migration_bytes_per_event: None,
                migration_latency_seconds: None,
                precision_contract_id: "f32-contract".to_owned(),
                consistency_model: MemoryConsistencyModel::SingleWriter,
            }],
            scenarios: vec![
                MemoryTopologyScenario::ColdStart,
                MemoryTopologyScenario::WarmSteadyState,
            ],
            notes: Some("Synthetic test fixture only.".to_owned()),
        }
    }

    fn test_regime() -> ArchitectureWorkloadRegime {
        ArchitectureWorkloadRegime {
            regime_profile_id: "regime-1".to_owned(),
            observation_record_id: "observation-1".to_owned(),
            workload_fingerprint: "workload-fingerprint-1".to_owned(),
            workload_family_id: "family-1".to_owned(),
            holdout_group_id: "holdout-1".to_owned(),
            comparison_contract_fingerprint: "contract-1".to_owned(),
            origin: RegimeCharacterizationOrigin::DeclaredWorkload,
            source_reference: "synthetic://regime/test".to_owned(),
            source_artifact_sha256: "d".repeat(64),
            extraction_procedure_sha256: None,
            execution_mode: WorkloadExecutionMode::Streaming,
            temporal_pattern: WorkloadTemporalPattern::Continuous,
            unit_of_work: "state-update".to_owned(),
            event_rate_hz: None,
            burst_peak_to_mean_ratio: None,
            burst_window_seconds: None,
            active_fraction: None,
            temporal_horizon_seconds: None,
            working_set_bytes: Some(bytes(128, 256)),
            state_update_rate_hz: Some(metric(1.0, 10.0)),
            deadline_seconds: None,
            batch_size: None,
            precision_profile: WorkloadPrecisionProfile::Float32,
            precision_contract_id: "f32-contract".to_owned(),
            communication_boundaries: Vec::new(),
        }
    }

    fn test_evidence() -> ArchitectureEvidenceRecord {
        let observation = crate::architecture_portfolio::ArchitectureBenchmarkObservation {
            record_id: "observation-1".to_owned(),
            architecture_id: "device-observation-1".to_owned(),
            architecture_family: ComputeArchitectureFamily::ClassicalGpu,
            workload_kind: ArchitectureWorkloadKind::HdcAlgebra,
            workload_fingerprint: "workload-fingerprint-1".to_owned(),
            comparison_contract_fingerprint: "contract-1".to_owned(),
            evidence_class: ArchitectureEvidenceClass::PhysicalPrototype,
            measurement_boundary: ArchitectureMeasurementBoundary::ApplicationEndToEnd,
            quality: metric(0.95, 0.97),
            resources: Some(ArchitectureResourceEnvelope {
                wall_time_seconds: metric(0.001, 0.002),
                energy_joules: metric(0.1, 0.2),
                monetary_cost_usd: metric(0.001, 0.002),
                peak_memory: bytes(1024, 2048),
            }),
        };
        let receipt = ArchitectureEvidenceReceipt {
            observation_record_id: observation.record_id.clone(),
            workload_fingerprint: observation.workload_fingerprint.clone(),
            comparison_contract_fingerprint: observation.comparison_contract_fingerprint.clone(),
            benchmark_suite: ArchitectureBenchmarkSuite::SymthaeaCanonical,
            source_reference: "https://example.invalid/measurement".to_owned(),
            benchmark_version: "v1".to_owned(),
            workload_case_id: "case-1".to_owned(),
            origin: ArchitectureEvidenceOrigin::DirectMeasurement,
            source_revision: Some("b".repeat(40)),
            source_artifact_sha256: "a".repeat(64),
            runner_artifact_sha256: Some("b".repeat(64)),
            fixture_sha256: "c".repeat(64),
            run_manifest_sha256: "d".repeat(64),
            environment_sha256: "e".repeat(64),
            raw_output_sha256: "f".repeat(64),
            interval_analysis_sha256: "1".repeat(64),
            summary_sha256: "2".repeat(64),
            executor_identity: "test-lab".to_owned(),
            device_identity: Some("device-observation-1".to_owned()),
            toolchain_identity: Some("rustc-test".to_owned()),
            repeat_count: Some(30),
            failed_run_count: Some(0),
            interval_method_id: Some("bootstrap-percentile-v1".to_owned()),
            aggregation_method_id: Some("median-and-quantiles-v1".to_owned()),
        };
        ArchitectureEvidenceRecord { observation, receipt }
    }

    #[test]
    fn validates_well_formed_memory_topology() {
        assert_eq!(validate_memory_topology(&base_topology()), Ok(()));
    }

    #[test]
    fn rejects_duplicate_scenarios() {
        let mut topology = base_topology();
        topology.scenarios.push(MemoryTopologyScenario::ColdStart);
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::DuplicateMemoryScenario)
        );
    }

    #[test]
    fn rejects_duplicate_memory_region_ids() {
        let mut topology = base_topology();
        topology.memory_regions.push(topology.memory_regions[0].clone());
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::DuplicateMemoryRegionId)
        );
    }

    #[test]
    fn rejects_unknown_edge_endpoint() {
        let mut topology = base_topology();
        topology.transfer_edges[0].destination_region_id = "missing".to_owned();
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::UnknownMemoryRegion)
        );
    }

    #[test]
    fn rejects_duplicate_boundary_accounting() {
        let mut topology = base_topology();
        let mut edge = topology.transfer_edges[0].clone();
        edge.edge_id = "second-edge".to_owned();
        edge.communication_boundary_id =
            topology.transfer_edges[0].communication_boundary_id.clone();
        topology.transfer_edges.push(edge);
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::DuplicateBoundaryAccounting)
        );
    }

    #[test]
    fn rejects_disconnected_route_edges() {
        let mut topology = base_topology();
        topology.transfer_edges.push(MemoryTransferEdge {
            edge_id: "scratch-to-dram".to_owned(),
            source_region_id: "scratch".to_owned(),
            destination_region_id: "dram".to_owned(),
            direction: MemoryTransferDirection::Write,
            fabric: MemoryTransferFabric::Pcie,
            communication_boundary_id: "boundary-scratch-dram".to_owned(),
            conversion_latency_seconds: None,
            conversion_energy_joules_per_bit: None,
        });
        topology.routes[0].edge_ids = vec![
            "dram-to-scratch".to_owned(),
            "dram-to-scratch".to_owned(),
        ];
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::DisconnectedMemoryRoute)
        );
    }

    #[test]
    fn rejects_route_cycles_only_when_route_requires_acyclicity() {
        let mut topology = base_topology();
        topology.transfer_edges.push(MemoryTransferEdge {
            edge_id: "scratch-to-dram".to_owned(),
            source_region_id: "scratch".to_owned(),
            destination_region_id: "dram".to_owned(),
            direction: MemoryTransferDirection::Write,
            fabric: MemoryTransferFabric::Pcie,
            communication_boundary_id: "boundary-scratch-dram".to_owned(),
            conversion_latency_seconds: None,
            conversion_energy_joules_per_bit: None,
        });
        topology.routes[0] = MemoryRoute {
            route_id: "round-trip".to_owned(),
            source_region_id: "dram".to_owned(),
            destination_region_id: "dram".to_owned(),
            edge_ids: vec!["dram-to-scratch".to_owned(), "scratch-to-dram".to_owned()],
            require_acyclic: true,
        };
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::MemoryRouteCycle)
        );
        topology.routes[0].require_acyclic = false;
        assert_eq!(validate_memory_topology(&topology), Ok(()));
    }

    #[test]
    fn rejects_guaranteed_placement_when_capacity_is_unknown() {
        let mut topology = base_topology();
        topology.memory_regions[1].capacity_bytes = None;
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::UnknownCapacityForGuaranteedPlacement)
        );
    }

    #[test]
    fn rejects_placement_set_that_exceeds_capacity() {
        let mut topology = base_topology();
        topology.memory_regions[1].capacity_bytes = Some(bytes(300, 300));
        topology.memory_regions[1].occupancy_bytes = Some(bytes(64, 128));
        topology.state_placements[0].live_bytes = Some(bytes(400, 400));
        assert_eq!(
            validate_memory_topology(&topology),
            Err(MemoryTopologyError::PlacementCapacityExceeded)
        );
    }

    #[test]
    fn rejects_precision_contract_mismatch() {
        let mut topology = base_topology();
        topology.state_placements[0].precision_contract_id = "bf16-contract".to_owned();
        assert_eq!(
            validate_memory_topology_binding(&topology, &test_evidence(), &test_regime()),
            Err(MemoryTopologyError::PrecisionContractMismatch)
        );
    }

    #[test]
    fn rejects_synthetic_records_for_comparison_after_exact_binding() {
        let topology = base_topology();
        assert_eq!(
            validate_memory_topology_binding(&topology, &test_evidence(), &test_regime()),
            Ok(())
        );
        assert_eq!(
            validate_memory_topology_for_comparison(&topology, &test_evidence(), &test_regime()),
            Err(MemoryTopologyError::SyntheticFixtureNotComparable)
        );
    }

    #[test]
    fn rejects_topology_bound_to_another_workload() {
        let mut topology = base_topology();
        topology.workload_fingerprint = "different".to_owned();
        assert_eq!(
            validate_memory_topology_binding(&topology, &test_evidence(), &test_regime()),
            Err(MemoryTopologyError::WorkloadMismatch)
        );
    }
}
