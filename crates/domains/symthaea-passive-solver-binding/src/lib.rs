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
}

impl SolverBoundaryBinding {
    /// Construct a verified binding from an actual candidate mesh.
    pub fn verified(
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
        })
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
        hasher.update(&[u8::from(self.solver_binding_verified)]);
        hasher.update(&[u8::from(self.physical_transport_unproven)]);
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

/// Contract implemented by concrete solver adapters.
///
/// The adapter must inspect the actual candidate mesh, resolve the typed
/// interface to solver-specific boundary entities, convert that actual selection
/// into BoundaryEdgeKey values, and construct the binding through the checked
/// constructor. The constructor independently re-validates completeness,
/// interface geometry, and exact mesh identity.
pub trait SolverBoundaryBindingAdapter {
    fn adapter_id(&self) -> &str;

    fn bind(
        &self,
        interface: &PortInterface,
        candidate: &TriangleMesh,
        candidate_geometry_digest: [u8; 32],
    ) -> Result<SolverBoundaryBinding, SolverBindingError>;
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
    EmptyBoundaryPatchDigest,
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


/// Canonicalized selection of boundary edges that make up one solver patch.
///
/// Coordinates are quantized to 1 µm before identity comparison. This is a
/// topology-level identity, not a solver face-number identity, so it remains
/// portable across solver implementations.
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
        let a = quantize_point(a_mm);
        let b = quantize_point(b_mm);
        if a == b {
            return Err(SolverBindingError::BoundaryPatchDoesNotMatchInterface);
        }
        Ok(Self { a, b: if a <= b { b } else { a } })
    }
}

#[derive(Debug, Clone, Copy)]
struct QuantizedBoundaryEdge {
    key: BoundaryEdgeKey,
    midpoint: [f64; 3],
    length_mm: f64,
}

/// Independently derived boundary-patch evidence.
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

    let all_boundary = collect_boundary_edge_records(candidate);
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

    let mut max_plane = 0.0f64;
    let mut max_radial = 0.0f64;
    let mut perimeter_mm = 0.0f64;
    for edge in &all_boundary {
        if !selection.edges.binary_search(&edge.key).is_ok() {
            continue;
        }
        max_plane = max_plane.max(
            plane_distance(
                dequantize_point(edge.key.a),
                interface.interface_plane.origin_mm,
                interface.interface_plane.normal_unit,
            )
            .abs(),
        );
        max_plane = max_plane.max(
            plane_distance(
                dequantize_point(edge.key.b),
                interface.interface_plane.origin_mm,
                interface.interface_plane.normal_unit,
            )
            .abs(),
        );
        max_radial = max_radial.max(
            (radial_distance(dequantize_point(edge.key.a), interface)
                - interface.radius_mm() as f64)
                .abs(),
        );
        max_radial = max_radial.max(
            (radial_distance(dequantize_point(edge.key.b), interface)
                - interface.radius_mm() as f64)
                .abs(),
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

fn collect_boundary_edge_keys(candidate: &TriangleMesh) -> Vec<BoundaryEdgeKey> {
    collect_boundary_edge_records(candidate)
        .into_iter()
        .map(|edge| edge.key)
        .collect()
}

fn collect_boundary_edge_records(candidate: &TriangleMesh) -> Vec<QuantizedBoundaryEdge> {
    use std::collections::HashMap;

    let mut edge_counts: HashMap<BoundaryEdgeKey, (usize, [f64; 3], f64)> = HashMap::new();
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
        for (a, b) in [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ] {
            let key = BoundaryEdgeKey::new(a, b)
                .expect("validated TriangleMesh contains finite non-degenerate edge endpoints");
            let midpoint = [
                (a[0] as f64 + b[0] as f64) / 2.0,
                (a[1] as f64 + b[1] as f64) / 2.0,
                (a[2] as f64 + b[2] as f64) / 2.0,
            ];
            let dx = b[0] as f64 - a[0] as f64;
            let dy = b[1] as f64 - a[1] as f64;
            let dz = b[2] as f64 - a[2] as f64;
            let length_mm = (dx * dx + dy * dy + dz * dz).sqrt();
            let entry = edge_counts.entry(key).or_insert((0, midpoint, length_mm));
            entry.0 += 1;
        }
    }

    edge_counts
        .into_iter()
        .filter_map(|(key, (count, midpoint, length_mm))| {
            (count == 1).then_some(QuantizedBoundaryEdge {
                key,
                midpoint,
                length_mm,
            })
        })
        .collect()
}

fn edge_matches_interface(
    edge: &QuantizedBoundaryEdge,
    interface: &PortInterface,
    tolerance_mm: f64,
) -> bool {
    let a = dequantize_point(edge.key.a);
    let b = dequantize_point(edge.key.b);
    let points = [a, b, edge.midpoint];

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
    [a, b].iter().all(|point| {
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

fn quantize_point(point: [f32; 3]) -> [i64; 3] {
    [
        (point[0] as f64 * 1_000_000.0).round() as i64,
        (point[1] as f64 * 1_000_000.0).round() as i64,
        (point[2] as f64 * 1_000_000.0).round() as i64,
    ]
}

fn dequantize_point(point: [i64; 3]) -> [f64; 3] {
    [
        point[0] as f64 / 1_000_000.0,
        point[1] as f64 / 1_000_000.0,
        point[2] as f64 / 1_000_000.0,
    ]
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
        collect_boundary_edge_keys(candidate)
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
        ) -> Result<SolverBoundaryBinding, SolverBindingError> {
            SolverBoundaryBinding::verified(
                interface,
                self.adapter_id(),
                "fixture:boundary-7",
                candidate_geometry_digest,
                candidate,
                select_boundary_patch(interface, candidate, 0.05),
                0.05,
            )
        }
    }

    #[test]
    fn adapter_contract_binds_against_actual_candidate_mesh() {
        let interface = interface(PortId(10), 7);
        let candidate = candidate();
        let binding = FixtureAdapter
            .bind(&interface, &candidate, [7; 32])
            .unwrap();

        assert_eq!(binding.port, PortId(10));
        assert_eq!(binding.external_boundary_handle, "fixture:boundary-7");
        assert_eq!(
            binding.realized_boundary.candidate_mesh_digest(),
            digest_triangle_mesh(&candidate)
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
            select_boundary_patch(&interface, &candidate, 0.05),
            0.05,
        )
        .unwrap();

        assert!(binding.validate_against(&interface).is_ok());
        assert!(binding.solver_binding_verified);
        assert!(binding.physical_transport_unproven);
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
            select_boundary_patch(&interface, &candidate, 0.05),
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
            select_boundary_patch(&interface, &candidate, 0.05),
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
            select_boundary_patch(&interface, &candidate, 0.05),
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
                select_boundary_patch(&interface, &candidate, 0.05),
            0.05,
            )
            .unwrap(),
            SolverBoundaryBinding::verified(
                &b,
                "test-adapter/v1",
                "patch:shared",
                [4; 32],
                &candidate,
                [3; 32],
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
        changed.vertices[1][0] += 0.125;
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
                [2; 32],
            ),
            Err(SolverBindingError::EmptyCandidateGeometryDigest)
        );

        assert_eq!(
            SolverBoundaryBinding::verified(
                &interface,
                "test-adapter/v1",
                "patch:inlet",
                [3; 32],
                &candidate,
                select_boundary_patch(&interface, &candidate, 0.05),
                0.05,
            ),
            Err(SolverBindingError::EmptyBoundaryPatchDigest)
        );
    }
}
