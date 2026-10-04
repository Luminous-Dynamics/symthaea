// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Solver-neutral contracts for binding typed passive-device interfaces to a
//! concrete solver boundary.
//!
//! The adapter layer deliberately owns the vendor/solver-specific handle.
//! Symthaea owns the invariant that the handle was bound to the exact
//! PortInterface identity and exact candidate mesh supplied to the adapter.
//!
//! This crate does not run a solver and does not certify physical transport.

use blake3::Hasher;
use symthaea_fabrication_kernel::mesh::TriangleMesh;
use symthaea_passive_void_compiler::{
    BoundaryConditionDomain, PortInterface, SolverBoundaryIdentity,
};
use symthaea_passive_void_graph::PortId;

/// Identity of the realized boundary patch selected from a candidate mesh.
///
/// Construction is sealed behind the checked constructor so a verified binding
/// cannot be assembled without computing the exact candidate-mesh identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RealizedBoundaryIdentity {
    candidate_geometry_digest: [u8; 32],
    candidate_mesh_digest: [u8; 32],
    boundary_patch_digest: [u8; 32],
    boundary_edge_count: usize,
    boundary_perimeter_micrometers: u64,
    max_plane_residual_micrometers: u64,
    max_radial_residual_micrometers: u64,
    boundary_matching_tolerance_micrometers: u64,
}

impl RealizedBoundaryIdentity {
    fn new(
        candidate_geometry_digest: [u8; 32],
        candidate_mesh: &TriangleMesh,
        interface: &PortInterface,
        selection: &BoundaryPatchSelection,
        tolerance_mm: f64,
    ) -> Result<Self, SolverBindingError> {
        if candidate_geometry_digest == [0; 32] {
            return Err(SolverBindingError::EmptyCandidateGeometryDigest);
        }

        let certificate =
            certify_boundary_patch(interface, candidate_mesh, selection, tolerance_mm)?;

        Ok(Self {
            candidate_geometry_digest,
            candidate_mesh_digest: digest_triangle_mesh(candidate_mesh),
            boundary_patch_digest: certificate.patch_digest,
            boundary_edge_count: certificate.boundary_edge_count,
            boundary_perimeter_micrometers: certificate.boundary_perimeter_micrometers,
            max_plane_residual_micrometers: certificate.max_plane_residual_micrometers,
            max_radial_residual_micrometers: certificate.max_radial_residual_micrometers,
            boundary_matching_tolerance_micrometers: micrometers_from_mm(tolerance_mm)?,
        })
    }

    pub fn candidate_geometry_digest(&self) -> [u8; 32] {
        self.candidate_geometry_digest
    }

    pub fn candidate_mesh_digest(&self) -> [u8; 32] {
        self.candidate_mesh_digest
    }

    pub fn boundary_patch_digest(&self) -> [u8; 32] {
        self.boundary_patch_digest
    }

    pub fn boundary_edge_count(&self) -> usize {
        self.boundary_edge_count
    }

    pub fn boundary_perimeter_micrometers(&self) -> u64 {
        self.boundary_perimeter_micrometers
    }

    pub fn max_plane_residual_micrometers(&self) -> u64 {
        self.max_plane_residual_micrometers
    }

    pub fn max_radial_residual_micrometers(&self) -> u64 {
        self.max_radial_residual_micrometers
    }

    pub fn boundary_matching_tolerance_micrometers(&self) -> u64 {
        self.boundary_matching_tolerance_micrometers
    }
}

/// Solver-specific adapter binding to an external boundary entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverBoundaryBinding {
    pub port: PortId,
    pub interface_digest: [u8; 32],
    pub solver_boundary: SolverBoundaryIdentity,
    pub adapter_id: String,
    pub external_boundary_handle: String,
    pub realized_boundary: RealizedBoundaryIdentity,
    pub solver_binding_verified: bool,
    pub physical_transport_unproven: bool,
    evidence_level: SolverBoundaryEvidenceLevel,
    solver_entity_fingerprint: Option<[u8; 32]>,
    solver_entity_mapping_digest: Option<[u8; 32]>,
}

impl SolverBoundaryBinding {
    /// Construct a verified binding from an actual candidate mesh.
    fn verified(
        interface: &PortInterface,
        adapter_id: impl Into<String>,
        external_boundary_handle: impl Into<String>,
        candidate_geometry_digest: [u8; 32],
        candidate_mesh: &TriangleMesh,
        boundary_patch: BoundaryPatchSelection,
        tolerance_mm: f64,
    ) -> Result<Self, SolverBindingError> {
        let adapter_id = adapter_id.into();
        let external_boundary_handle = external_boundary_handle.into();

        interface
            .validate(0.001)
            .map_err(|_| SolverBindingError::InvalidInterface)?;

        if adapter_id.trim().is_empty() {
            return Err(SolverBindingError::EmptyAdapterId);
        }
        if external_boundary_handle.trim().is_empty() {
            return Err(SolverBindingError::EmptyExternalBoundaryHandle);
        }

        let realized_boundary = RealizedBoundaryIdentity::new(
            candidate_geometry_digest,
            candidate_mesh,
            interface,
            &boundary_patch,
            tolerance_mm,
        )?;

        Ok(Self {
            port: interface.port,
            interface_digest: interface.digest(),
            solver_boundary: interface.solver_boundary,
            adapter_id,
            external_boundary_handle,
            realized_boundary,
            solver_binding_verified: true,
            physical_transport_unproven: true,
            evidence_level: SolverBoundaryEvidenceLevel::AdapterAttested,
            solver_entity_fingerprint: None,
            solver_entity_mapping_digest: None,
        })
    }

    /// Evidence level attached by the sealed construction path.
    pub fn evidence_level(&self) -> SolverBoundaryEvidenceLevel {
        self.evidence_level
    }

    /// Solver-side entity fingerprint, present only after explicit entity attestation.
    pub fn solver_entity_fingerprint(&self) -> Option<[u8; 32]> {
        self.solver_entity_fingerprint
    }

    /// Digest binding the solver-side entity fingerprint to this exact interface,
    /// semantic candidate, exact candidate mesh, and realized boundary patch.
    pub fn solver_entity_mapping_digest(&self) -> Option<[u8; 32]> {
        self.solver_entity_mapping_digest
    }

    /// Re-check that a binding still corresponds to the exact interface.
    pub fn validate_against(
        &self,
        interface: &PortInterface,
    ) -> Result<(), SolverBindingError> {
        if !self.solver_binding_verified {
            return Err(SolverBindingError::UnverifiedBinding);
        }
        if !self.physical_transport_unproven {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        if !self.evidence_level.is_verified() {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        match (self.evidence_level, self.solver_entity_fingerprint, self.solver_entity_mapping_digest) {
            (SolverBoundaryEvidenceLevel::AdapterAttested, None, None)
            | (SolverBoundaryEvidenceLevel::SolverEntityAttested, Some(_), Some(_)) => {}
            _ => return Err(SolverBindingError::InvalidEvidenceState),
        }
        if self.port != interface.port {
            return Err(SolverBindingError::PortMismatch);
        }
        if self.interface_digest != interface.digest() {
            return Err(SolverBindingError::InterfaceDigestMismatch);
        }
        if self.solver_boundary != interface.solver_boundary {
            return Err(SolverBindingError::SolverBoundaryMismatch);
        }
        Ok(())
    }

    /// Re-check the interface, semantic geometry identity, and exact candidate
    /// mesh identity.
    pub fn validate_against_candidate(
        &self,
        interface: &PortInterface,
        candidate_geometry_digest: [u8; 32],
        candidate: &TriangleMesh,
    ) -> Result<(), SolverBindingError> {
        self.validate_against(interface)?;
        if self.realized_boundary.candidate_geometry_digest() != candidate_geometry_digest {
            return Err(SolverBindingError::CandidateGeometryDigestMismatch);
        }
        if self.realized_boundary.candidate_mesh_digest() != digest_triangle_mesh(candidate) {
            return Err(SolverBindingError::CandidateMeshDigestMismatch);
        }
        Ok(())
    }

    /// Stable identity of the complete binding statement.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"passive-solver-boundary-binding:v3");
        hasher.update(&self.port.0.to_le_bytes());
        hasher.update(&self.interface_digest);
        hasher.update(&[domain_byte(self.solver_boundary.domain)]);
        hasher.update(&self.solver_boundary.id.to_le_bytes());
        hasher.update(self.adapter_id.as_bytes());
        hasher.update(&[0]);
        hasher.update(self.external_boundary_handle.as_bytes());
        hasher.update(&[0]);
        hasher.update(&self.realized_boundary.candidate_geometry_digest());
        hasher.update(&self.realized_boundary.candidate_mesh_digest());
        hasher.update(&self.realized_boundary.boundary_patch_digest());
        hasher.update(&(self.realized_boundary.boundary_edge_count() as u64).to_le_bytes());
        hasher.update(&self.realized_boundary.boundary_perimeter_micrometers().to_le_bytes());
        hasher.update(&self.realized_boundary.max_plane_residual_micrometers().to_le_bytes());
        hasher.update(&self.realized_boundary.max_radial_residual_micrometers().to_le_bytes());
        hasher.update(
            &self
                .realized_boundary
                .boundary_matching_tolerance_micrometers()
                .to_le_bytes(),
        );
        hasher.update(&[u8::from(self.solver_binding_verified)]);
        hasher.update(&[u8::from(self.physical_transport_unproven)]);
        hasher.update(&[match self.evidence_level {
            SolverBoundaryEvidenceLevel::AdapterAttested => 0,
            SolverBoundaryEvidenceLevel::SolverEntityAttested => 1,
        }]);
        hasher.update(&self.solver_entity_fingerprint.unwrap_or([0; 32]));
        hasher.update(&self.solver_entity_mapping_digest.unwrap_or([0; 32]));
        *hasher.finalize().as_bytes()
    }
}

/// Canonical digest of the exact TriangleMesh representation supplied to a
/// solver adapter.
///
/// This is deliberately byte-oriented rather than topology-normalized: a
/// changed coordinate, normal, index, ordering, or count is a different solver
/// input.
pub fn digest_triangle_mesh(mesh: &TriangleMesh) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"passive-candidate-mesh:v1");

    hasher.update(&(mesh.vertices.len() as u64).to_le_bytes());
    for vertex in &mesh.vertices {
        for value in vertex {
            hasher.update(&value.to_le_bytes());
        }
    }

    hasher.update(&(mesh.normals.len() as u64).to_le_bytes());
    for normal in &mesh.normals {
        for value in normal {
            hasher.update(&value.to_le_bytes());
        }
    }

    hasher.update(&(mesh.indices.len() as u64).to_le_bytes());
    for triangle in &mesh.indices {
        for index in triangle {
            hasher.update(&index.to_le_bytes());
        }
    }

    *hasher.finalize().as_bytes()
}

/// Draft returned by a solver adapter before Symthaea stamps adapter-attested evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverBoundaryBindingDraft {
    pub external_boundary_handle: String,
    pub boundary_patch: BoundaryPatchSelection,
}

impl SolverBoundaryBindingDraft {
    pub fn new(
        external_boundary_handle: impl Into<String>,
        boundary_patch: BoundaryPatchSelection,
    ) -> Result<Self, SolverBindingError> {
        let external_boundary_handle = external_boundary_handle.into();
        if external_boundary_handle.trim().is_empty() {
            return Err(SolverBindingError::EmptyExternalBoundaryHandle);
        }
        Ok(Self {
            external_boundary_handle,
            boundary_patch,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverBoundaryEvidenceLevel {
    /// Core accepted the adapter's mapping draft after independent candidate checks.
    AdapterAttested,
    /// The adapter additionally reported a concrete solver-side entity fingerprint
    /// and a digest binding that fingerprint to this exact candidate/interface rim.
    SolverEntityAttested,
}

impl SolverBoundaryEvidenceLevel {
    pub fn is_verified(self) -> bool {
        matches!(self, Self::AdapterAttested | Self::SolverEntityAttested)
    }

    /// Whether this evidence level satisfies a caller's minimum requirement.
    pub fn satisfies(self, minimum: Self) -> bool {
        match (self, minimum) {
            (_, Self::AdapterAttested) => true,
            (Self::SolverEntityAttested, Self::SolverEntityAttested) => true,
            (Self::AdapterAttested, Self::SolverEntityAttested) => false,
        }
    }
}

/// Solver-side entity evidence returned by a live-capable adapter.
///
/// This is intentionally a provenance claim, not an independent solver proof:
/// the neutral core verifies the cryptographic binding to the exact candidate
/// identity and realized rim, but cannot inspect vendor-specific solver state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverBoundaryEntityAttestation {
    pub external_boundary_handle: String,
    pub solver_entity_fingerprint: [u8; 32],
    pub solver_entity_mapping_digest: [u8; 32],
}

impl SolverBoundaryEntityAttestation {
    pub fn new(
        external_boundary_handle: impl Into<String>,
        solver_entity_fingerprint: [u8; 32],
        solver_entity_mapping_digest: [u8; 32],
    ) -> Result<Self, SolverBindingError> {
        let external_boundary_handle = external_boundary_handle.into();
        if external_boundary_handle.trim().is_empty() {
            return Err(SolverBindingError::EmptyExternalBoundaryHandle);
        }
        if solver_entity_fingerprint == [0; 32] {
            return Err(SolverBindingError::EmptySolverEntityFingerprint);
        }
        if solver_entity_mapping_digest == [0; 32] {
            return Err(SolverBindingError::EmptySolverEntityMappingDigest);
        }
        Ok(Self {
            external_boundary_handle,
            solver_entity_fingerprint,
            solver_entity_mapping_digest,
        })
    }
}

/// Canonical digest tying a solver-side entity fingerprint to the exact
/// interface, semantic candidate, exact mesh, realized candidate-surface rim,
/// and external solver handle.
pub fn solver_entity_mapping_digest(
    interface: &PortInterface,
    candidate_geometry_digest: [u8; 32],
    candidate_mesh_digest: [u8; 32],
    boundary_patch_digest: [u8; 32],
    external_boundary_handle: &str,
    solver_entity_fingerprint: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"passive-solver-entity-mapping:v1");
    hasher.update(&interface.digest());
    hasher.update(&candidate_geometry_digest);
    hasher.update(&candidate_mesh_digest);
    hasher.update(&boundary_patch_digest);
    hasher.update(external_boundary_handle.as_bytes());
    hasher.update(&[0]);
    hasher.update(&solver_entity_fingerprint);
    *hasher.finalize().as_bytes()
}

/// Optional extension for adapters able to inspect the concrete solver-side
/// boundary entity after the initial mapping has been created.
pub trait SolverBoundaryEntityIntrospector {
    fn attest_entity(
        &self,
        interface: &PortInterface,
        candidate: &TriangleMesh,
        binding: &SolverBoundaryBinding,
    ) -> Result<SolverBoundaryEntityAttestation, SolverBindingError>;
}

/// Promote an adapter-attested binding only when a solver-entity attestation
/// matches the exact binding handle and candidate identity.
pub fn promote_solver_entity_attestation(
    mut binding: SolverBoundaryBinding,
    interface: &PortInterface,
    candidate_geometry_digest: [u8; 32],
    candidate: &TriangleMesh,
    attestation: SolverBoundaryEntityAttestation,
) -> Result<SolverBoundaryBinding, SolverBindingError> {
    binding.validate_against_candidate(interface, candidate_geometry_digest, candidate)?;

    if binding.evidence_level != SolverBoundaryEvidenceLevel::AdapterAttested {
        return Err(SolverBindingError::InvalidEvidenceState);
    }
    if attestation.external_boundary_handle != binding.external_boundary_handle {
        return Err(SolverBindingError::SolverEntityHandleMismatch);
    }

    let expected_mapping_digest = solver_entity_mapping_digest(
        interface,
        candidate_geometry_digest,
        digest_triangle_mesh(candidate),
        binding.realized_boundary.boundary_patch_digest(),
        &attestation.external_boundary_handle,
        attestation.solver_entity_fingerprint,
    );
    if attestation.solver_entity_mapping_digest != expected_mapping_digest {
        return Err(SolverBindingError::SolverEntityMappingDigestMismatch);
    }

    binding.evidence_level = SolverBoundaryEvidenceLevel::SolverEntityAttested;
    binding.solver_entity_fingerprint = Some(attestation.solver_entity_fingerprint);
    binding.solver_entity_mapping_digest = Some(attestation.solver_entity_mapping_digest);

    Ok(binding)
}

/// Complete construction path for adapters that can introspect the live solver
/// entity they resolved.
pub fn bind_with_adapter_and_entity_attestation<
    A: SolverBoundaryBindingAdapter + SolverBoundaryEntityIntrospector,
>(
    adapter: &A,
    interface: &PortInterface,
    candidate: &TriangleMesh,
    candidate_geometry_digest: [u8; 32],
    tolerance_mm: f64,
) -> Result<SolverBoundaryBinding, SolverBindingError> {
    let binding = bind_with_adapter(
        adapter,
        interface,
        candidate,
        candidate_geometry_digest,
        tolerance_mm,
    )?;
    let attestation = adapter.attest_entity(interface, candidate, &binding)?;
    promote_solver_entity_attestation(
        binding,
        interface,
        candidate_geometry_digest,
        candidate,
        attestation,
    )
}

/// Contract implemented by concrete solver adapters.
///
/// Adapters return a draft. The public `bind_with_adapter` orchestration path
/// performs the checked candidate/interface construction and is the only path
/// that stamps adapter-attested verification into a binding.
pub trait SolverBoundaryBindingAdapter {
    fn adapter_id(&self) -> &str;

    fn bind(
        &self,
        interface: &PortInterface,
        candidate: &TriangleMesh,
        candidate_geometry_digest: [u8; 32],
    ) -> Result<SolverBoundaryBindingDraft, SolverBindingError>;
}

/// Public orchestration path that seals verified binding construction.
pub fn bind_with_adapter<A: SolverBoundaryBindingAdapter>(
    adapter: &A,
    interface: &PortInterface,
    candidate: &TriangleMesh,
    candidate_geometry_digest: [u8; 32],
    tolerance_mm: f64,
) -> Result<SolverBoundaryBinding, SolverBindingError> {
    let draft = adapter.bind(interface, candidate, candidate_geometry_digest)?;
    SolverBoundaryBinding::verified(
        interface,
        adapter.adapter_id(),
        draft.external_boundary_handle,
        candidate_geometry_digest,
        candidate,
        draft.boundary_patch,
        tolerance_mm,
    )
}

/// Deterministic registry-level validation for a complete set of bindings.
pub fn validate_binding_set(
    interfaces: &[PortInterface],
    bindings: &[SolverBoundaryBinding],
) -> Result<(), SolverBindingError> {
    if interfaces.len() != bindings.len() {
        return Err(SolverBindingError::BindingCountMismatch);
    }

    let mut interface_digests = std::collections::BTreeSet::new();
    let mut interface_ports = std::collections::BTreeSet::new();
    let mut solver_boundary_ids = std::collections::BTreeSet::new();
    let mut binding_digests = std::collections::BTreeSet::new();
    let mut binding_ports = std::collections::BTreeSet::new();
    let mut handles = std::collections::BTreeSet::new();
    let mut candidate_geometry_digest = None;
    let mut candidate_mesh_digest = None;

    for interface in interfaces {
        if !interface_ports.insert(interface.port) {
            return Err(SolverBindingError::DuplicateInterfacePort(interface.port));
        }
        if !solver_boundary_ids.insert((
            domain_byte(interface.solver_boundary.domain),
            interface.solver_boundary.id,
        )) {
            return Err(SolverBindingError::DuplicateSolverBoundaryIdentity(
                interface.solver_boundary,
            ));
        }
        if !interface_digests.insert(interface.digest()) {
            return Err(SolverBindingError::DuplicateInterface(interface.port));
        }
    }

    for binding in bindings {
        if !binding_ports.insert(binding.port) {
            return Err(SolverBindingError::DuplicateBindingPort(binding.port));
        }
        if !binding.solver_binding_verified {
            return Err(SolverBindingError::UnverifiedBinding);
        }
        if !binding.physical_transport_unproven {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        if !binding.evidence_level.is_verified() {
            return Err(SolverBindingError::InvalidEvidenceState);
        }
        match (binding.evidence_level, binding.solver_entity_fingerprint, binding.solver_entity_mapping_digest) {
            (SolverBoundaryEvidenceLevel::AdapterAttested, None, None)
            | (SolverBoundaryEvidenceLevel::SolverEntityAttested, Some(_), Some(_)) => {}
            _ => return Err(SolverBindingError::InvalidEvidenceState),
        }
        if let Some(expected) = candidate_geometry_digest {
            if expected != binding.realized_boundary.candidate_geometry_digest() {
                return Err(SolverBindingError::CandidateGeometryDigestMismatch);
            }
        } else {
            candidate_geometry_digest = Some(binding.realized_boundary.candidate_geometry_digest());
        }
        if let Some(expected) = candidate_mesh_digest {
            if expected != binding.realized_boundary.candidate_mesh_digest() {
                return Err(SolverBindingError::CandidateMeshDigestMismatch);
            }
        } else {
            candidate_mesh_digest = Some(binding.realized_boundary.candidate_mesh_digest());
        }
        if !binding_digests.insert(binding.digest()) {
            return Err(SolverBindingError::DuplicateBinding(binding.port));
        }
        if !handles.insert((
            binding.solver_boundary,
            binding.external_boundary_handle.clone(),
        )) {
            return Err(SolverBindingError::DuplicateExternalBoundaryHandle(
                binding.external_boundary_handle.clone(),
            ));
        }

        let interface = interfaces
            .iter()
            .find(|interface| interface.port == binding.port)
            .ok_or(SolverBindingError::PortMismatch)?;
        binding.validate_against(interface)?;
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverBindingError {
    InvalidInterface,
    EmptyAdapterId,
    EmptyExternalBoundaryHandle,
    UnverifiedBinding,
    InvalidEvidenceState,
    EmptyCandidateGeometryDigest,
    EmptySolverEntityFingerprint,
    EmptySolverEntityMappingDigest,
    SolverEntityHandleMismatch,
    SolverEntityMappingDigestMismatch,
    InsufficientEvidenceLevel {
        actual: SolverBoundaryEvidenceLevel,
        required: SolverBoundaryEvidenceLevel,
    },
    PortMismatch,
    InterfaceDigestMismatch,
    SolverBoundaryMismatch,
    CandidateGeometryDigestMismatch,
    CandidateMeshDigestMismatch,
    EmptyBoundaryPatchSelection,
    BoundaryPatchEdgeNotOnCandidate,
    BoundaryPatchDoesNotMatchInterface,
    BoundaryPatchIsNotSingleClosedLoop,
    BoundaryPatchSelectionIncomplete,
    BindingCountMismatch,
    DuplicateInterface(PortId),
    DuplicateInterfacePort(PortId),
    DuplicateSolverBoundaryIdentity(SolverBoundaryIdentity),
    DuplicateBinding(PortId),
    DuplicateBindingPort(PortId),
    DuplicateExternalBoundaryHandle(String),
    BoundaryEdgeIdentityCollision,
}

const fn domain_byte(domain: BoundaryConditionDomain) -> u8 {
    match domain {
        BoundaryConditionDomain::Mechanical => 0,
        BoundaryConditionDomain::Fluidic => 1,
        BoundaryConditionDomain::Thermal => 2,
        BoundaryConditionDomain::Acoustic => 3,
        BoundaryConditionDomain::Electromagnetic => 4,
        BoundaryConditionDomain::Optical => 5,
        BoundaryConditionDomain::Chemical => 6,
    }
}


/// Canonicalized selection of candidate-surface boundary edges forming one
/// typed interface rim.
///
/// Coordinates are quantized to 1 µm for identity comparison. This is a
/// geometry-side identity, not a solver face-number identity; the solver's
/// opaque boundary handle is carried separately by the binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryPatchSelection {
    edges: Vec<BoundaryEdgeKey>,
}

impl BoundaryPatchSelection {
    pub fn from_edges(mut edges: Vec<BoundaryEdgeKey>) -> Result<Self, SolverBindingError> {
        if edges.is_empty() {
            return Err(SolverBindingError::EmptyBoundaryPatchSelection);
        }
        edges.sort();
        edges.dedup();
        if edges.is_empty() {
            return Err(SolverBindingError::EmptyBoundaryPatchSelection);
        }
        Ok(Self { edges })
    }

    pub fn edges(&self) -> &[BoundaryEdgeKey] {
        &self.edges
    }
}

/// Stable, quantized boundary-edge identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundaryEdgeKey {
    pub a: [i64; 3],
    pub b: [i64; 3],
}

impl BoundaryEdgeKey {
    pub fn new(a_mm: [f32; 3], b_mm: [f32; 3]) -> Result<Self, SolverBindingError> {
        if a_mm.iter().chain(b_mm.iter()).any(|v| !v.is_finite()) {
            return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
        }
        let a = quantize_point(a_mm)?;
        let b = quantize_point(b_mm)?;
        if a == b {
            return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
        }
        if a <= b {
            Ok(Self { a, b })
        } else {
            Ok(Self { a: b, b: a })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MeshEdgeKey {
    a: u32,
    b: u32,
}

impl MeshEdgeKey {
    fn new(a: u32, b: u32) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct QuantizedBoundaryEdge {
    key: BoundaryEdgeKey,
    a_mm: [f64; 3],
    b_mm: [f64; 3],
    midpoint: [f64; 3],
    length_mm: f64,
}

/// Independently derived candidate-surface interface-rim evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryPatchCertificate {
    pub patch_digest: [u8; 32],
    pub boundary_edge_count: usize,
    pub boundary_perimeter_micrometers: u64,
    pub max_plane_residual_micrometers: u64,
    pub max_radial_residual_micrometers: u64,
}

/// Independently derive the complete boundary selection for a typed interface.
///
/// Adapters may use this helper when their solver boundary entity maps directly
/// to the candidate boundary. Solver-specific adapters may instead translate
/// their own selected entity back to BoundaryEdgeKey values and let
/// certify_boundary_patch validate that translation.
pub fn select_boundary_patch(
    interface: &PortInterface,
    candidate: &TriangleMesh,
    tolerance_mm: f64,
) -> Result<BoundaryPatchSelection, SolverBindingError> {
    interface
        .validate(0.001)
        .map_err(|_| SolverBindingError::InvalidInterface)?;
    if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
        return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
    }

    let report = symthaea_fabrication_kernel::validate::validate_mesh(candidate);
    if !report.is_valid() {
        return Err(SolverBindingError::BoundaryPatchEdgeNotOnCandidate);
    }

    let all_boundary = collect_boundary_edge_records(candidate)?;
    let expected: Vec<_> = all_boundary
        .iter()
        .filter(|edge| edge_matches_interface(edge, interface, tolerance_mm))
        .map(|edge| edge.key)
        .collect();

    if expected.is_empty() {
        return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
    }

    let selection = BoundaryPatchSelection::from_edges(expected)?;
    validate_closed_single_loop(selection.edges())?;
    validate_simple_boundary_loop(selection.edges(), &all_boundary, interface)?;
    Ok(selection)
}

/// Independently certify that the supplied selection is exactly the interface
/// rim present in the candidate mesh. This closes the gap where an adapter could
/// report a stale patch name or arbitrary digest.
pub fn certify_boundary_patch(
    interface: &PortInterface,
    candidate: &TriangleMesh,
    selection: &BoundaryPatchSelection,
    tolerance_mm: f64,
) -> Result<BoundaryPatchCertificate, SolverBindingError> {
    interface
        .validate(0.001)
        .map_err(|_| SolverBindingError::InvalidInterface)?;
    if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
        return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
    }

    let report = symthaea_fabrication_kernel::validate::validate_mesh(candidate);
    if !report.is_valid() {
        return Err(SolverBindingError::BoundaryPatchEdgeNotOnCandidate);
    }

    let all_boundary = collect_boundary_edge_records(candidate);
    let mut all_boundary_keys: std::collections::BTreeSet<BoundaryEdgeKey> =
        all_boundary.iter().map(|edge| edge.key).collect();

    for edge in selection.edges() {
        if !all_boundary_keys.remove(edge) {
            return Err(SolverBindingError::BoundaryPatchEdgeNotOnCandidate);
        }
    }

    let expected = select_boundary_patch(interface, candidate, tolerance_mm)?;

    if selection.edges != expected.edges {
        return Err(SolverBindingError::BoundaryPatchSelectionIncomplete);
    }

    validate_simple_boundary_loop(selection.edges(), &all_boundary, interface)?;

    let mut max_plane = 0.0f64;
    let mut max_radial = 0.0f64;
    let mut perimeter_mm = 0.0f64;
    for edge in &all_boundary {
        if !selection.edges.binary_search(&edge.key).is_ok() {
            continue;
        }
        max_plane = max_plane.max(
            plane_distance(
                edge.a_mm,
                interface.interface_plane.origin_mm,
                interface.interface_plane.normal_unit,
            )
            .abs(),
        );
        max_plane = max_plane.max(
            plane_distance(
                edge.b_mm,
                interface.interface_plane.origin_mm,
                interface.interface_plane.normal_unit,
            )
            .abs(),
        );
        max_radial = max_radial.max(
            (radial_distance(edge.a_mm, interface) - interface.radius_mm() as f64).abs(),
        );
        max_radial = max_radial.max(
            (radial_distance(edge.b_mm, interface) - interface.radius_mm() as f64).abs(),
        );
        perimeter_mm += edge.length_mm;
    }

    let boundary_perimeter_micrometers = (perimeter_mm * 1_000.0).round() as u64;
    let max_plane_residual_micrometers = (max_plane * 1_000.0).round() as u64;
    let max_radial_residual_micrometers = (max_radial * 1_000.0).round() as u64;
    let candidate_mesh_digest = digest_triangle_mesh(candidate);

    let mut hasher = Hasher::new();
    hasher.update(b"passive-boundary-patch-certificate:v1");
    hasher.update(&interface.digest());
    hasher.update(&candidate_mesh_digest);
    hasher.update(&(selection.edges.len() as u64).to_le_bytes());
    for edge in &selection.edges {
        for point in [edge.a, edge.b] {
            for value in point {
                hasher.update(&value.to_le_bytes());
            }
        }
    }
    hasher.update(&boundary_perimeter_micrometers.to_le_bytes());
    hasher.update(&max_plane_residual_micrometers.to_le_bytes());
    hasher.update(&max_radial_residual_micrometers.to_le_bytes());

    Ok(BoundaryPatchCertificate {
        patch_digest: *hasher.finalize().as_bytes(),
        boundary_edge_count: selection.edges.len(),
        boundary_perimeter_micrometers,
        max_plane_residual_micrometers,
        max_radial_residual_micrometers,
    })
}

fn collect_boundary_edge_keys(
    candidate: &TriangleMesh,
) -> Result<Vec<BoundaryEdgeKey>, SolverBindingError> {
    Ok(collect_boundary_edge_records(candidate)?
        .into_iter()
        .map(|edge| edge.key)
        .collect())
}

fn collect_boundary_edge_records(
    candidate: &TriangleMesh,
) -> Result<Vec<QuantizedBoundaryEdge>, SolverBindingError> {
    use std::collections::{BTreeMap, HashMap};

    let mut edge_counts: HashMap<
        MeshEdgeKey,
        (usize, [f64; 3], [f64; 3], [f64; 3], f64),
    > = HashMap::new();
    for triangle in &candidate.indices {
        if triangle
            .iter()
            .any(|index| (*index as usize) >= candidate.vertices.len())
        {
            continue;
        }
        let vertices = [
            candidate.vertices[triangle[0] as usize],
            candidate.vertices[triangle[1] as usize],
            candidate.vertices[triangle[2] as usize],
        ];
        for (a_index, b_index, a, b) in [
            (triangle[0], triangle[1], vertices[0], vertices[1]),
            (triangle[1], triangle[2], vertices[1], vertices[2]),
            (triangle[2], triangle[0], vertices[2], vertices[0]),
        ] {
            let key = MeshEdgeKey::new(a_index, b_index);
            let midpoint = [
                (a[0] as f64 + b[0] as f64) / 2.0,
                (a[1] as f64 + b[1] as f64) / 2.0,
                (a[2] as f64 + b[2] as f64) / 2.0,
            ];
            let dx = b[0] as f64 - a[0] as f64;
            let dy = b[1] as f64 - a[1] as f64;
            let dz = b[2] as f64 - a[2] as f64;
            let length_mm = (dx * dx + dy * dy + dz * dz).sqrt();
            let entry = edge_counts.entry(key).or_insert((
                0,
                [a[0] as f64, a[1] as f64, a[2] as f64],
                [b[0] as f64, b[1] as f64, b[2] as f64],
                midpoint,
                length_mm,
            ));
            entry.0 += 1;
        }
    }

    let mut portable_identity_sources = BTreeMap::<BoundaryEdgeKey, MeshEdgeKey>::new();
    let mut records = Vec::new();
    for (mesh_edge, (count, a_mm, b_mm, midpoint, length_mm)) in edge_counts {
        if count != 1 {
            continue;
        }
        let key = BoundaryEdgeKey::new(
            [a_mm[0] as f32, a_mm[1] as f32, a_mm[2] as f32],
            [b_mm[0] as f32, b_mm[1] as f32, b_mm[2] as f32],
        )?;
        if let Some(existing) = portable_identity_sources.insert(key, mesh_edge) {
            if existing != mesh_edge {
                return Err(SolverBindingError::BoundaryEdgeIdentityCollision);
            }
        }
        records.push(QuantizedBoundaryEdge {
            key,
            a_mm,
            b_mm,
            midpoint,
            length_mm,
        });
    }

    Ok(records)
}

fn edge_matches_interface(
    edge: &QuantizedBoundaryEdge,
    interface: &PortInterface,
    tolerance_mm: f64,
) -> bool {
    let points = [edge.a_mm, edge.b_mm, edge.midpoint];

    if points.iter().any(|point| {
        plane_distance(
            *point,
            interface.interface_plane.origin_mm,
            interface.interface_plane.normal_unit,
        )
        .abs()
            > tolerance_mm
    }) {
        return false;
    }

    let radial_tolerance = tolerance_mm;
    [edge.a_mm, edge.b_mm].iter().all(|point| {
        (radial_distance(*point, interface) - interface.radius_mm() as f64).abs()
            <= radial_tolerance
    })
}

fn validate_closed_single_loop(edges: &[BoundaryEdgeKey]) -> Result<(), SolverBindingError> {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    let mut degree = BTreeMap::<[i64; 3], usize>::new();
    let mut adjacency = BTreeMap::<[i64; 3], BTreeSet<[i64; 3]>>::new();

    for edge in edges {
        *degree.entry(edge.a).or_default() += 1;
        *degree.entry(edge.b).or_default() += 1;
        adjacency.entry(edge.a).or_default().insert(edge.b);
        adjacency.entry(edge.b).or_default().insert(edge.a);
    }

    if degree.values().any(|degree| *degree != 2) {
        return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
    }

    let start = *degree
        .keys()
        .next()
        .ok_or(SolverBindingError::EmptyBoundaryPatchSelection)?;
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        if !visited.insert(node) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            queue.extend(neighbors.iter().copied());
        }
    }

    if visited.len() != degree.len() {
        return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
    }

    Ok(())
}

fn validate_simple_boundary_loop(
    edges: &[BoundaryEdgeKey],
    records: &[QuantizedBoundaryEdge],
    interface: &PortInterface,
) -> Result<(), SolverBindingError> {
    use std::collections::{BTreeMap, BTreeSet};

    if edges.len() < 3 {
        return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
    }

    let raw: BTreeMap<BoundaryEdgeKey, ([f64; 3], [f64; 3])> = records
        .iter()
        .filter(|record| edges.binary_search(&record.key).is_ok())
        .map(|record| (record.key, (record.a_mm, record.b_mm)))
        .collect();

    if raw.len() != edges.len() {
        return Err(SolverBindingError::BoundaryPatchEdgeNotOnCandidate);
    }

    let mut adjacency = BTreeMap::<[i64; 3], Vec<BoundaryEdgeKey>>::new();
    for edge in edges {
        adjacency.entry(edge.a).or_default().push(*edge);
        adjacency.entry(edge.b).or_default().push(*edge);
    }

    let start = *adjacency
        .keys()
        .next()
        .ok_or(SolverBindingError::EmptyBoundaryPatchSelection)?;
    let mut used = BTreeSet::new();
    let mut current = start;
    let mut loop_points = Vec::with_capacity(edges.len());

    for step in 0..edges.len() {
        let incident = adjacency
            .get(&current)
            .ok_or(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop)?;
        let next_edge = incident
            .iter()
            .copied()
            .find(|edge| !used.contains(edge))
            .ok_or(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop)?;

        let (a_mm, b_mm) = *raw
            .get(&next_edge)
            .ok_or(SolverBindingError::BoundaryPatchEdgeNotOnCandidate)?;
        let next_point = if next_edge.a == current { b_mm } else { a_mm };

        if step == 0 {
            let start_point = if next_edge.a == current { a_mm } else { b_mm };
            loop_points.push(start_point);
        }
        loop_points.push(next_point);

        used.insert(next_edge);
        current = if next_edge.a == current {
            next_edge.b
        } else {
            next_edge.a
        };
    }

    if current != start || used.len() != edges.len() {
        return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
    }

    loop_points.pop();
    if loop_points.len() != edges.len() {
        return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
    }

    let normal = interface.interface_plane.normal_unit;
    let ax = (normal[0] as f64).abs();
    let ay = (normal[1] as f64).abs();
    let az = (normal[2] as f64).abs();
    let dropped_axis = if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    };

    let project = |point: [f64; 3]| -> [f64; 2] {
        match dropped_axis {
            0 => [point[1], point[2]],
            1 => [point[0], point[2]],
            _ => [point[0], point[1]],
        }
    };

    let points: Vec<[f64; 2]> = loop_points.into_iter().map(project).collect();
    let epsilon = 1.0e-12;

    for i in 0..points.len() {
        let a1 = points[i];
        let a2 = points[(i + 1) % points.len()];
        for j in (i + 1)..points.len() {
            if j == i + 1 || (i == 0 && j + 1 == points.len()) {
                continue;
            }
            let b1 = points[j];
            let b2 = points[(j + 1) % points.len()];
            if segments_intersect_2d(a1, a2, b1, b2, epsilon) {
                return Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop);
            }
        }
    }

    Ok(())
}

fn segments_intersect_2d(
    a1: [f64; 2],
    a2: [f64; 2],
    b1: [f64; 2],
    b2: [f64; 2],
    epsilon: f64,
) -> bool {
    fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    fn on_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2], epsilon: f64) -> bool {
        p[0] >= a[0].min(b[0]) - epsilon
            && p[0] <= a[0].max(b[0]) + epsilon
            && p[1] >= a[1].min(b[1]) - epsilon
            && p[1] <= a[1].max(b[1]) + epsilon
    }

    let o1 = orient(a1, a2, b1);
    let o2 = orient(a1, a2, b2);
    let o3 = orient(b1, b2, a1);
    let o4 = orient(b1, b2, a2);

    if ((o1 > epsilon && o2 < -epsilon) || (o1 < -epsilon && o2 > epsilon))
        && ((o3 > epsilon && o4 < -epsilon) || (o3 < -epsilon && o4 > epsilon))
    {
        return true;
    }

    (o1.abs() <= epsilon && on_segment(a1, a2, b1, epsilon))
        || (o2.abs() <= epsilon && on_segment(a1, a2, b2, epsilon))
        || (o3.abs() <= epsilon && on_segment(b1, b2, a1, epsilon))
        || (o4.abs() <= epsilon && on_segment(b1, b2, a2, epsilon))
}

fn micrometers_from_mm(tolerance_mm: f64) -> Result<u64, SolverBindingError> {
    if !tolerance_mm.is_finite() || tolerance_mm < 0.0 {
        return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
    }
    let micrometers = tolerance_mm * 1_000.0;
    if micrometers > u64::MAX as f64 {
        return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
    }
    Ok(micrometers.round() as u64)
}

fn quantize_point(point: [f32; 3]) -> Result<[i64; 3], SolverBindingError> {
    let mut quantized = [0i64; 3];
    for (index, value) in point.iter().copied().enumerate() {
        let rounded = (value as f64 * 1_000_000.0).round();
        if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded >= i64::MAX as f64 {
            return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
        }
        quantized[index] = rounded as i64;
    }
    Ok(quantized)
}

fn plane_distance(point: [f64; 3], origin: [f32; 3], normal: [f32; 3]) -> f64 {
    let delta = [
        point[0] - origin[0] as f64,
        point[1] - origin[1] as f64,
        point[2] - origin[2] as f64,
    ];
    delta[0] * normal[0] as f64
        + delta[1] * normal[1] as f64
        + delta[2] * normal[2] as f64
}

fn radial_distance(point: [f64; 3], interface: &PortInterface) -> f64 {
    let center = interface.position_mm;
    let normal = interface.outward_normal_unit;
    let delta = [
        point[0] - center[0] as f64,
        point[1] - center[1] as f64,
        point[2] - center[2] as f64,
    ];
    let axial = delta[0] * normal[0] as f64
        + delta[1] * normal[1] as f64
        + delta[2] * normal[2] as f64;
    let radial = [
        delta[0] - axial * normal[0] as f64,
        delta[1] - axial * normal[1] as f64,
        delta[2] - axial * normal[2] as f64,
    ];
    (radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2]).sqrt()
}


/// Validate a complete binding set against the exact candidate geometry and mesh.
///
/// This is the recommended final pre-dispatch gate: set structure is checked first,
/// then every binding is re-checked against the same semantic geometry digest and
/// exact TriangleMesh representation.
pub fn validate_binding_set_against_candidate(
    interfaces: &[PortInterface],
    bindings: &[SolverBoundaryBinding],
    candidate_geometry_digest: [u8; 32],
    candidate: &TriangleMesh,
) -> Result<(), SolverBindingError> {
    validate_binding_set_against_candidate_with_minimum_evidence(
        interfaces,
        bindings,
        candidate_geometry_digest,
        candidate,
        SolverBoundaryEvidenceLevel::AdapterAttested,
    )
}

/// Variant of the final pre-dispatch gate that makes the required evidence level explicit.
pub fn validate_binding_set_against_candidate_with_minimum_evidence(
    interfaces: &[PortInterface],
    bindings: &[SolverBoundaryBinding],
    candidate_geometry_digest: [u8; 32],
    candidate: &TriangleMesh,
    minimum_evidence: SolverBoundaryEvidenceLevel,
) -> Result<(), SolverBindingError> {
    validate_binding_set(interfaces, bindings)?;

    for binding in bindings {
        if !binding.evidence_level.satisfies(minimum_evidence) {
            return Err(SolverBindingError::InsufficientEvidenceLevel {
                actual: binding.evidence_level,
                required: minimum_evidence,
            });
        }

        let interface = interfaces
            .iter()
            .find(|interface| interface.port == binding.port)
            .ok_or(SolverBindingError::PortMismatch)?;
        binding.validate_against_candidate(interface, candidate_geometry_digest, candidate)?;
    }

    Ok(())
}

/// Deterministic identity for a complete, validated set of solver bindings.
///
/// The digest is order-independent with respect to the supplied binding slice,
/// while still committing to every per-binding digest and the common candidate
/// geometry/mesh identity.
pub fn digest_binding_set(
    interfaces: &[PortInterface],
    bindings: &[SolverBoundaryBinding],
) -> Result<[u8; 32], SolverBindingError> {
    validate_binding_set(interfaces, bindings)?;

    let mut digests: Vec<_> = bindings.iter().map(SolverBoundaryBinding::digest).collect();
    digests.sort_unstable();

    let mut hasher = Hasher::new();
    hasher.update(b"passive-solver-boundary-binding-set:v1");
    hasher.update(&(interfaces.len() as u64).to_le_bytes());

    if let Some(binding) = bindings.first() {
        hasher.update(&binding.realized_boundary.candidate_geometry_digest());
        hasher.update(&binding.realized_boundary.candidate_mesh_digest());
    } else {
        hasher.update(&[0; 32]);
        hasher.update(&[0; 32]);
    }

    for digest in digests {
        hasher.update(&digest);
    }

    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_passive_void_compiler::{
        BoundaryConditionDomain, InterfacePlane, PortAperture, SolverBoundaryIdentity,
    };

    fn interface(port: PortId, solver_id: u32) -> PortInterface {
        PortInterface::new(
            port,
            [0.0, 0.0, 0.0],
            PortAperture::Circular { radius_mm: 2.0 },
            [0.0, 0.0, 1.0],
            InterfacePlane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
            SolverBoundaryIdentity {
                domain: BoundaryConditionDomain::Fluidic,
                id: solver_id,
            },
        )
        .unwrap()
    }

    fn candidate() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
        }
    }

    fn boundary_edges(candidate: &TriangleMesh) -> Vec<BoundaryEdgeKey> {
        collect_boundary_edge_keys(candidate).unwrap()
    }

    struct FixtureAdapter;

    impl SolverBoundaryBindingAdapter for FixtureAdapter {
        fn adapter_id(&self) -> &str {
            "fixture-adapter/v1"
        }

        fn bind(
            &self,
            interface: &PortInterface,
            candidate: &TriangleMesh,
            candidate_geometry_digest: [u8; 32],
        ) -> Result<SolverBoundaryBindingDraft, SolverBindingError> {
            SolverBoundaryBindingDraft::new(
                "fixture:boundary-7",
                select_boundary_patch(interface, candidate, 0.05).unwrap(),
            )
        }
    }

    impl SolverBoundaryEntityIntrospector for FixtureAdapter {
        fn attest_entity(
            &self,
            interface: &PortInterface,
            candidate: &TriangleMesh,
            binding: &SolverBoundaryBinding,
        ) -> Result<SolverBoundaryEntityAttestation, SolverBindingError> {
            let fingerprint = [0x42; 32];
            let mapping_digest = solver_entity_mapping_digest(
                interface,
                binding.realized_boundary.candidate_geometry_digest(),
                digest_triangle_mesh(candidate),
                binding.realized_boundary.boundary_patch_digest(),
                &binding.external_boundary_handle,
                fingerprint,
            );
            SolverBoundaryEntityAttestation::new(
                binding.external_boundary_handle.clone(),
                fingerprint,
                mapping_digest,
            )
        }
    }

    #[test]
    fn adapter_orchestration_stamps_verified_evidence() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert_eq!(binding.evidence_level(), SolverBoundaryEvidenceLevel::AdapterAttested);
        assert!(binding.solver_binding_verified);
    }

    #[test]
    fn entity_attestation_orchestration_stamps_solver_entity_evidence() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter_and_entity_attestation(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert_eq!(
            binding.evidence_level(),
            SolverBoundaryEvidenceLevel::SolverEntityAttested
        );
        assert_eq!(binding.solver_entity_fingerprint(), Some([0x42; 32]));
        assert!(binding.solver_entity_mapping_digest().is_some());
        assert!(binding.validate_against_candidate(&interface, [7; 32], &candidate).is_ok());
    }

    #[test]
    fn entity_attestation_rejects_handle_mismatch() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        let attestation = SolverBoundaryEntityAttestation::new(
            "fixture:other-boundary",
            [0x42; 32],
            [1; 32],
        )
        .unwrap();

        assert_eq!(
            promote_solver_entity_attestation(
                binding,
                &interface,
                [7; 32],
                &candidate,
                attestation,
            ),
            Err(SolverBindingError::SolverEntityHandleMismatch)
        );
    }

    #[test]
    fn entity_attestation_rejects_mapping_digest_mismatch() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        let attestation = SolverBoundaryEntityAttestation::new(
            binding.external_boundary_handle.clone(),
            [0x42; 32],
            [1; 32],
        )
        .unwrap();

        assert_eq!(
            promote_solver_entity_attestation(
                binding,
                &interface,
                [7; 32],
                &candidate,
                attestation,
            ),
            Err(SolverBindingError::SolverEntityMappingDigestMismatch)
        );
    }

    #[test]
    fn distinct_boundary_edges_cannot_collapse_to_one_portable_identity() {
        let mesh = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0000004, 0.0000004, 0.0],
                [2.0000004, 0.0000004, 0.0],
                [0.0000004, 2.0000004, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 6],
            indices: vec![
                [0, 1, 2],
                [3, 4, 5],
            ],
        };

        assert!(matches!(
            collect_boundary_edge_records(&mesh),
            Err(SolverBindingError::BoundaryEdgeIdentityCollision)
        ));
    }

    #[test]
    fn entity_attestation_rejects_empty_fingerprint() {
        assert_eq!(
            SolverBoundaryEntityAttestation::new(
                "fixture:boundary-7",
                [0; 32],
                [1; 32],
            ),
            Err(SolverBindingError::EmptySolverEntityFingerprint)
        );
    }

    #[test]
    fn adapter_attestation_cannot_satisfy_solver_entity_requirement() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert_eq!(
            validate_binding_set_against_candidate_with_minimum_evidence(
                std::slice::from_ref(&interface),
                std::slice::from_ref(&binding),
                [7; 32],
                &candidate,
                SolverBoundaryEvidenceLevel::SolverEntityAttested,
            ),
            Err(SolverBindingError::InsufficientEvidenceLevel {
                actual: SolverBoundaryEvidenceLevel::AdapterAttested,
                required: SolverBoundaryEvidenceLevel::SolverEntityAttested,
            })
        );
    }

    #[test]
    fn solver_entity_attestation_satisfies_minimum_evidence_requirement() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter_and_entity_attestation(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert!(validate_binding_set_against_candidate_with_minimum_evidence(
            std::slice::from_ref(&interface),
            std::slice::from_ref(&binding),
            [7; 32],
            &candidate,
            SolverBoundaryEvidenceLevel::SolverEntityAttested,
        ).is_ok());
    }

    #[test]
    fn adapter_contract_binds_against_actual_candidate_mesh() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = bind_with_adapter(
            &FixtureAdapter,
            &interface,
            &candidate,
            [7; 32],
            0.05,
        )
        .unwrap();

        assert_eq!(binding.port, PortId(10));
        assert_eq!(binding.external_boundary_handle, "fixture:boundary-7");
        assert_eq!(
            binding.realized_boundary.candidate_mesh_digest(),
            digest_triangle_mesh(&candidate)
        );
    }

    #[test]
    fn adapter_draft_rejects_empty_handle() {
        let selection = BoundaryPatchSelection::from_edges(boundary_edges(&candidate())).unwrap();
        assert_eq!(
            SolverBoundaryBindingDraft::new("", selection),
            Err(SolverBindingError::EmptyExternalBoundaryHandle)
        );
    }

    #[test]
    fn verified_binding_carries_exact_interface_identity() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert!(binding.validate_against(&interface).is_ok());
        assert!(binding.solver_binding_verified);
        assert!(binding.physical_transport_unproven);
    }

    #[test]
    fn sub_micron_boundary_drift_is_not_erased_by_identity_quantization() {
        let interface = interface(PortId(10), 7);
        let mesh = TriangleMesh {
            vertices: vec![
                [1.0004, 0.0, 0.0],
                [0.0, 1.0004, 0.0],
                [-1.0004, 0.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };

        assert_eq!(
            select_boundary_patch(&interface, &mesh, 0.0001),
            Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface)
        );
    }

    #[test]
    fn binding_records_boundary_matching_tolerance() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let selection = select_boundary_patch(&interface, &candidate, 0.05).unwrap();

        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            selection,
            0.05,
        )
        .unwrap();

        assert_eq!(
            binding
                .realized_boundary
                .boundary_matching_tolerance_micrometers(),
            50
        );
    }

    #[test]
    fn binding_digest_commits_to_boundary_matching_tolerance() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let selection_a = select_boundary_patch(&interface, &candidate, 0.05).unwrap();
        let selection_b = select_boundary_patch(&interface, &candidate, 0.10).unwrap();

        let a = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            selection_a,
            0.05,
        )
        .unwrap();
        let b = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            selection_b,
            0.10,
        )
        .unwrap();

        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn candidate_mesh_drift_is_rejected() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        let mut changed = candidate.clone();
        changed.vertices[0][0] += 0.125;

        assert_eq!(
            binding.validate_against_candidate(&interface, [3; 32], &changed),
            Err(SolverBindingError::CandidateMeshDigestMismatch)
        );
        assert!(
            binding
                .validate_against_candidate(&interface, [3; 32], &candidate)
                .is_ok()
        );
    }

    #[test]
    fn candidate_geometry_identity_drift_is_rejected() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert_eq!(
            binding.validate_against_candidate(&interface, [4; 32], &candidate),
            Err(SolverBindingError::CandidateGeometryDigestMismatch)
        );
    }

    #[test]
    fn interface_drift_is_rejected() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &candidate,
            select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        let drifted = interface(PortId(10), 8);
        assert_eq!(
            binding.validate_against(&drifted),
            Err(SolverBindingError::InterfaceDigestMismatch)
        );
    }

    #[test]
    fn duplicate_external_handle_is_rejected() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let candidate = candidate();
        let bindings = vec![
            SolverBoundaryBinding::verified(
                &a,
                "test-adapter/v1",
                "patch:shared",
                [3; 32],
                &candidate,
                select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
            0.05,
            )
            .unwrap(),
            SolverBoundaryBinding::verified(
                &b,
                "test-adapter/v1",
                "patch:shared",
                [4; 32],
                &candidate,
                select_boundary_patch(&b, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
        ];

        assert_eq!(
            validate_binding_set(&[a, b], &bindings),
            Err(SolverBindingError::DuplicateExternalBoundaryHandle(
                "patch:shared".into()
            ))
        );
    }


    #[test]
    fn public_selector_rejects_out_of_range_mesh_coordinates() {
        let interface = interface(PortId(10), 7);
        let mesh = TriangleMesh {
            vertices: vec![
                [f32::MAX, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![[0, 1, 2]],
        };

        assert!(select_boundary_patch(&interface, &mesh, 0.05).is_err());
    }

    #[test]
    fn boundary_edge_key_rejects_out_of_range_coordinates() {
        assert_eq!(
            BoundaryEdgeKey::new([f32::MAX, 0.0, 0.0], [0.0, 0.0, 0.0]),
            Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface)
        );
    }

    #[test]
    fn boundary_edge_key_canonicalizes_reversed_endpoints() {
        let forward = BoundaryEdgeKey::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        let reverse = BoundaryEdgeKey::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]).unwrap();

        assert_eq!(forward, reverse);
        assert!(forward.a < forward.b);
    }

    #[test]
    fn self_intersecting_boundary_loop_is_rejected() {
        let interface = interface(PortId(10), 7);
        let raw_points = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let edges = vec![
            BoundaryEdgeKey::new(raw_points[0], raw_points[1]).unwrap(),
            BoundaryEdgeKey::new(raw_points[1], raw_points[2]).unwrap(),
            BoundaryEdgeKey::new(raw_points[2], raw_points[3]).unwrap(),
            BoundaryEdgeKey::new(raw_points[3], raw_points[0]).unwrap(),
        ];
        let selection = BoundaryPatchSelection::from_edges(edges.clone()).unwrap();
        let records = edges
            .into_iter()
            .map(|key| {
                let find = |point: [i64; 3]| {
                    [
                        point[0] as f64 / 1_000_000.0,
                        point[1] as f64 / 1_000_000.0,
                        point[2] as f64 / 1_000_000.0,
                    ]
                };
                let a_mm = find(key.a);
                let b_mm = find(key.b);
                QuantizedBoundaryEdge {
                    key,
                    a_mm,
                    b_mm,
                    midpoint: [
                        (a_mm[0] + b_mm[0]) / 2.0,
                        (a_mm[1] + b_mm[1]) / 2.0,
                        (a_mm[2] + b_mm[2]) / 2.0,
                    ],
                    length_mm: ((b_mm[0] - a_mm[0]).powi(2)
                        + (b_mm[1] - a_mm[1]).powi(2)
                        + (b_mm[2] - a_mm[2]).powi(2))
                    .sqrt(),
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(
            validate_simple_boundary_loop(selection.edges(), &records, &interface),
            Err(SolverBindingError::BoundaryPatchIsNotSingleClosedLoop)
        );
    }

    #[test]
    fn boundary_patch_selection_is_rejected_when_edge_is_not_on_candidate() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let selection = BoundaryPatchSelection::from_edges(vec![
            BoundaryEdgeKey::new([0.0, 0.0, 0.0], [0.75, 0.0, 0.0]).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            SolverBoundaryBinding::verified(
                &interface,
                "test-adapter/v1",
                "patch:inlet",
                [3; 32],
                &candidate,
                selection,
                0.05,
            ),
            Err(SolverBindingError::BoundaryPatchEdgeNotOnCandidate)
        );
    }

    #[test]
    fn boundary_patch_selection_must_be_complete() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let mut edges = boundary_edges(&candidate);
        edges.pop();
        let selection = BoundaryPatchSelection::from_edges(edges).unwrap();

        assert_eq!(
            SolverBoundaryBinding::verified(
                &interface,
                "test-adapter/v1",
                "patch:inlet",
                [3; 32],
                &candidate,
                selection,
                0.05,
            ),
            Err(SolverBindingError::BoundaryPatchSelectionIncomplete)
        );
    }

    #[test]
    fn near_size_reduction_is_rejected() {
        let interface = interface(PortId(10), 7);
        let candidate = TriangleMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [0.85, 0.0, 0.0],
                [0.0, 0.85, 0.0],
                [-0.85, 0.0, 0.0],
                [0.0, -0.85, 0.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
        };
        let result = select_boundary_patch(&interface, &candidate, 0.05);
        assert_eq!(
            result,
            Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface)
        );
    }

    #[test]
    fn boundary_patch_certificate_is_deterministic() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let selection = select_boundary_patch(&interface, &candidate, 0.05).unwrap();

        let a = certify_boundary_patch(&interface, &candidate, &selection, 0.05).unwrap();
        let b = certify_boundary_patch(&interface, &candidate, &selection, 0.05).unwrap();

        assert_eq!(a, b);
        assert_eq!(a.boundary_edge_count, 4);
        assert!(a.boundary_perimeter_micrometers > 0);
    }

    #[test]
    fn binding_set_candidate_gate_rejects_mesh_drift() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let mut changed = candidate.clone();
        changed.vertices[0][2] += 0.125;
        let binding = SolverBoundaryBinding::verified(
            &interface,
            "test-adapter/v1",
            "patch:inlet",
            [3; 32],
            &changed,
            select_boundary_patch(&interface, &changed, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert_eq!(
            validate_binding_set_against_candidate(
                &[interface],
                &[binding],
                [3; 32],
                &candidate,
            ),
            Err(SolverBindingError::CandidateMeshDigestMismatch)
        );
    }

    #[test]
    fn binding_set_rejects_mixed_candidate_geometry_identity() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let candidate = candidate();
        let bindings = vec![
            SolverBoundaryBinding::verified(
                &a,
                "test-adapter/v1",
                "patch:a",
                [3; 32],
                &candidate,
                select_boundary_patch(&a, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
            SolverBoundaryBinding::verified(
                &b,
                "test-adapter/v1",
                "patch:b",
                [4; 32],
                &candidate,
                select_boundary_patch(&b, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
        ];

        assert_eq!(
            validate_binding_set(&[a, b], &bindings),
            Err(SolverBindingError::CandidateGeometryDigestMismatch)
        );
    }

    #[test]
    fn binding_set_rejects_mixed_candidate_mesh_identity() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let candidate = candidate();
        let mut changed = candidate.clone();
        changed.vertices[0][2] += 0.125;
        let first = SolverBoundaryBinding::verified(
            &a,
            "test-adapter/v1",
            "patch:a",
            [3; 32],
            &candidate,
            select_boundary_patch(&a, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();
        let second = SolverBoundaryBinding::verified(
            &b,
            "test-adapter/v1",
            "patch:b",
            [3; 32],
            &changed,
            select_boundary_patch(&b, &changed, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert_eq!(
            validate_binding_set(&[a, b], &[first, second]),
            Err(SolverBindingError::CandidateMeshDigestMismatch)
        );
    }

    #[test]
    fn binding_set_digest_is_order_independent() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let candidate = candidate();
        let first = SolverBoundaryBinding::verified(
            &a,
            "test-adapter/v1",
            "patch:a",
            [3; 32],
            &candidate,
            select_boundary_patch(&a, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();
        let second = SolverBoundaryBinding::verified(
            &b,
            "test-adapter/v1",
            "patch:b",
            [3; 32],
            &candidate,
            select_boundary_patch(&b, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert_eq!(
            digest_binding_set(&[a.clone(), b.clone()], &[first.clone(), second.clone()]).unwrap(),
            digest_binding_set(&[b, a], &[second, first]).unwrap()
        );
    }

    #[test]
    fn binding_set_rejects_duplicate_interface_port() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(10), 8);
        let candidate = candidate();
        let bindings = vec![
            SolverBoundaryBinding::verified(
                &a,
                "test-adapter/v1",
                "patch:a",
                [3; 32],
                &candidate,
                select_boundary_patch(&a, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
            SolverBoundaryBinding::verified(
                &b,
                "test-adapter/v1",
                "patch:b",
                [4; 32],
                &candidate,
                select_boundary_patch(&b, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
        ];

        assert_eq!(
            validate_binding_set(&[a, b], &bindings),
            Err(SolverBindingError::DuplicateInterfacePort(PortId(10)))
        );
    }

    #[test]
    fn binding_set_rejects_duplicate_solver_boundary_identity() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 7);
        let candidate = candidate();
        let bindings = vec![
            SolverBoundaryBinding::verified(
                &a,
                "test-adapter/v1",
                "patch:a",
                [3; 32],
                &candidate,
                select_boundary_patch(&a, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
            SolverBoundaryBinding::verified(
                &b,
                "test-adapter/v1",
                "patch:b",
                [4; 32],
                &candidate,
                select_boundary_patch(&b, &candidate, 0.05).unwrap(),
                0.05,
            )
            .unwrap(),
        ];

        assert_eq!(
            validate_binding_set(&[a, b], &bindings),
            Err(SolverBindingError::DuplicateSolverBoundaryIdentity(
                a.solver_boundary
            ))
        );
    }

    #[test]
    fn binding_set_rejects_duplicate_binding_port() {
        let a = interface(PortId(10), 7);
        let b = interface(PortId(20), 8);
        let candidate = candidate();
        let first = SolverBoundaryBinding::verified(
            &a,
            "test-adapter/v1",
            "patch:a",
            [3; 32],
            &candidate,
            select_boundary_patch(&a, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();
        let second = SolverBoundaryBinding::verified(
            &a,
            "test-adapter/v1",
            "patch:b",
            [4; 32],
            &candidate,
            select_boundary_patch(&a, &candidate, 0.05).unwrap(),
            0.05,
        )
        .unwrap();

        assert_eq!(
            validate_binding_set(&[a, b], &[first, second]),
            Err(SolverBindingError::DuplicateBindingPort(PortId(10)))
        );
    }

    #[test]
    fn zero_digests_are_rejected() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();

        assert_eq!(
            SolverBoundaryBinding::verified(
                &interface,
                "test-adapter/v1",
                "patch:inlet",
                [0; 32],
                &candidate,
                select_boundary_patch(&interface, &candidate, 0.05).unwrap(),
                0.05,
            ),
            Err(SolverBindingError::EmptyCandidateGeometryDigest)
        );

        assert_eq!(
            BoundaryPatchSelection::from_edges(Vec::new()),
            Err(SolverBindingError::EmptyBoundaryPatchSelection)
        );
    }
}
