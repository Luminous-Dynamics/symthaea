// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STL and 3MF file export
//!
//! Binary STL uses the standard 80-byte header and 50-byte triangle records.
//! 3MF export can return either the model XML payload or a complete stored ZIP/OPC
//! package containing content types, relationships, and `/3D/3dmodel.model`.

use crate::attestation::AttestedFabricationManifest;
use crate::audit::{AuditJournal, digest_audit_journal};
use crate::mesh::TriangleMesh;
use crate::provenance::FabricationManifest;
use crate::trust::{TrustSnapshot, digest_trust_snapshot};

/// Export mesh as binary STL
pub fn export_stl(mesh: &TriangleMesh) -> Vec<u8> {
    let tri_count = mesh.indices.len() as u32;
    let mut buf = Vec::with_capacity(84 + tri_count as usize * 50);

    // 80-byte header
    let header = b"symthaea-fabrication-kernel STL output\0";
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 80 - 39]); // Pad to 80 bytes

    // Triangle count
    buf.extend_from_slice(&tri_count.to_le_bytes());

    // Per-triangle: normal (3×f32) + 3 vertices (3×f32 each) + attribute (u16)
    for tri in &mesh.indices {
        // Compute face normal from first triangle's vertices
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        let (nx, ny, nz) = if len > 1e-10 {
            (nx / len, ny / len, nz / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Normal
        buf.extend_from_slice(&nx.to_le_bytes());
        buf.extend_from_slice(&ny.to_le_bytes());
        buf.extend_from_slice(&nz.to_le_bytes());

        // Vertices
        for &idx in tri {
            let v = mesh.vertices[idx as usize];
            buf.extend_from_slice(&v[0].to_le_bytes());
            buf.extend_from_slice(&v[1].to_le_bytes());
            buf.extend_from_slice(&v[2].to_le_bytes());
        }

        // Attribute byte count
        buf.extend_from_slice(&0u16.to_le_bytes());
    }

    buf
}

/// Export the `/3D/3dmodel.model` XML payload used inside a 3MF package.
///
/// This function deliberately does **not** claim to produce a complete `.3mf`
/// file. A standards-conforming 3MF artifact is an OPC/ZIP package containing
/// relationships and content-type metadata in addition to this model XML.
pub fn export_3mf_model_xml(mesh: &TriangleMesh) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<model unit=\"millimeter\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n");
    xml.push_str("  <resources>\n");
    xml.push_str("    <object id=\"1\" type=\"model\">\n");
    xml.push_str("      <mesh>\n");

    // Vertices
    xml.push_str("        <vertices>\n");
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\" />\n",
            v[0], v[1], v[2]
        ));
    }
    xml.push_str("        </vertices>\n");

    // Triangles
    xml.push_str("        <triangles>\n");
    for tri in &mesh.indices {
        xml.push_str(&format!(
            "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" />\n",
            tri[0], tri[1], tri[2]
        ));
    }
    xml.push_str("        </triangles>\n");

    xml.push_str("      </mesh>\n");
    xml.push_str("    </object>\n");
    xml.push_str("  </resources>\n");
    xml.push_str("  <build>\n");
    xml.push_str("    <item objectid=\"1\" />\n");
    xml.push_str("  </build>\n");
    xml.push_str("</model>\n");

    xml
}

const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Default Extension="json" ContentType="application/json"/>
</Types>
"#;

const ROOT_RELATIONSHIPS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

const ROOT_RELATIONSHIPS_WITH_MANIFEST_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
  <Relationship Target="/Metadata/fabrication-manifest.json" Id="rel1" Type="https://luminousdynamics.org/relationships/fabrication-manifest"/>
</Relationships>
"#;

const ROOT_RELATIONSHIPS_WITH_ATTESTATION_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
  <Relationship Target="/Metadata/fabrication-manifest.json" Id="rel1" Type="https://luminousdynamics.org/relationships/fabrication-manifest"/>
  <Relationship Target="/Metadata/fabrication-attestation.json" Id="rel2" Type="https://luminousdynamics.org/relationships/fabrication-attestation"/>
  <Relationship Target="/Metadata/fabrication-manifest.sha256" Id="rel3" Type="https://luminousdynamics.org/relationships/fabrication-manifest-digest"/>
</Relationships>
"#;

const ROOT_RELATIONSHIPS_WITH_GOVERNANCE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
  <Relationship Target="/Metadata/fabrication-manifest.json" Id="rel1" Type="https://luminousdynamics.org/relationships/fabrication-manifest"/>
  <Relationship Target="/Metadata/fabrication-attestation.json" Id="rel2" Type="https://luminousdynamics.org/relationships/fabrication-attestation"/>
  <Relationship Target="/Metadata/fabrication-manifest.sha256" Id="rel3" Type="https://luminousdynamics.org/relationships/fabrication-manifest-digest"/>
  <Relationship Target="/Metadata/fabrication-trust-snapshot.json" Id="rel4" Type="https://luminousdynamics.org/relationships/fabrication-trust-snapshot"/>
  <Relationship Target="/Metadata/fabrication-audit-journal.json" Id="rel5" Type="https://luminousdynamics.org/relationships/fabrication-audit-journal"/>
</Relationships>
"#;

/// Export a complete core 3MF package as an uncompressed ZIP/OPC byte stream.
///
/// The package contains `[Content_Types].xml`, `/_rels/.rels`, and
/// `/3D/3dmodel.model`. Stored entries avoid an additional compression
/// dependency while remaining standards-compatible ZIP members.
pub fn export_3mf_package(mesh: &TriangleMesh) -> Vec<u8> {
    let model = export_3mf_model_xml(mesh);
    write_stored_zip(&[
        ("[Content_Types].xml", CONTENT_TYPES_XML.as_bytes()),
        ("_rels/.rels", ROOT_RELATIONSHIPS_XML.as_bytes()),
        ("3D/3dmodel.model", model.as_bytes()),
    ])
}

/// Export a complete core 3MF package with a deterministic fabrication manifest.
///
/// The manifest remains non-cryptographic provenance. Packaging it next to the
/// model prevents accidental separation of geometry from the policies and
/// machine-validated toolpath identity that produced the job.
pub fn export_3mf_package_with_manifest(
    mesh: &TriangleMesh,
    manifest: &FabricationManifest,
) -> Result<Vec<u8>, serde_json::Error> {
    let model = export_3mf_model_xml(mesh);
    let manifest_json = serde_json::to_vec_pretty(manifest)?;
    Ok(write_stored_zip(&[
        ("[Content_Types].xml", CONTENT_TYPES_XML.as_bytes()),
        (
            "_rels/.rels",
            ROOT_RELATIONSHIPS_WITH_MANIFEST_XML.as_bytes(),
        ),
        ("3D/3dmodel.model", model.as_bytes()),
        (
            "Metadata/fabrication-manifest.json",
            manifest_json.as_slice(),
        ),
    ]))
}

/// Export a 3MF package carrying a manifest, its SHA-256 digest, and detached signatures.
pub fn export_3mf_package_with_attestation(
    mesh: &TriangleMesh,
    attested: &AttestedFabricationManifest,
) -> Result<Vec<u8>, serde_json::Error> {
    let model = export_3mf_model_xml(mesh);
    let manifest_json = serde_json::to_vec_pretty(&attested.manifest)?;
    let attestation_json = serde_json::to_vec_pretty(attested)?;
    let digest_hex = attested.manifest_digest.to_hex();
    Ok(write_stored_zip(&[
        ("[Content_Types].xml", CONTENT_TYPES_XML.as_bytes()),
        (
            "_rels/.rels",
            ROOT_RELATIONSHIPS_WITH_ATTESTATION_XML.as_bytes(),
        ),
        ("3D/3dmodel.model", model.as_bytes()),
        (
            "Metadata/fabrication-manifest.json",
            manifest_json.as_slice(),
        ),
        (
            "Metadata/fabrication-attestation.json",
            attestation_json.as_slice(),
        ),
        (
            "Metadata/fabrication-manifest.sha256",
            digest_hex.as_bytes(),
        ),
    ]))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GovernedPackageBuildError {
    TrustSnapshot(String),
    AuditJournal(String),
    EmptyAuditJournal,
    Encoding(String),
}

/// Export one self-contained release package carrying lifecycle and audit evidence.
pub fn export_3mf_package_with_governance(
    mesh: &TriangleMesh,
    attested: &AttestedFabricationManifest,
    trust_snapshot: &TrustSnapshot,
    audit_journal: &AuditJournal,
) -> Result<Vec<u8>, GovernedPackageBuildError> {
    let trust_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| GovernedPackageBuildError::TrustSnapshot(format!("{error:?}")))?;
    let audit_report = audit_journal.verify();
    if !audit_report.intact() {
        return Err(GovernedPackageBuildError::AuditJournal(format!(
            "{:?}",
            audit_report.violations
        )));
    }
    let audit_head = audit_journal
        .head()
        .ok_or(GovernedPackageBuildError::EmptyAuditJournal)?;
    let audit_digest = digest_audit_journal(audit_journal)
        .map_err(|error| GovernedPackageBuildError::AuditJournal(format!("{error:?}")))?;

    let model = export_3mf_model_xml(mesh);
    let manifest_json = serde_json::to_vec_pretty(&attested.manifest)
        .map_err(|error| GovernedPackageBuildError::Encoding(error.to_string()))?;
    let attestation_json = serde_json::to_vec_pretty(attested)
        .map_err(|error| GovernedPackageBuildError::Encoding(error.to_string()))?;
    let trust_json = serde_json::to_vec_pretty(trust_snapshot)
        .map_err(|error| GovernedPackageBuildError::Encoding(error.to_string()))?;
    let audit_json = serde_json::to_vec_pretty(audit_journal)
        .map_err(|error| GovernedPackageBuildError::Encoding(error.to_string()))?;
    let manifest_digest_hex = attested.manifest_digest.to_hex();
    let trust_digest_hex = trust_digest.to_hex();
    let audit_digest_hex = audit_digest.to_hex();
    let audit_head_hex = audit_head.to_hex();

    Ok(write_stored_zip(&[
        ("[Content_Types].xml", CONTENT_TYPES_XML.as_bytes()),
        (
            "_rels/.rels",
            ROOT_RELATIONSHIPS_WITH_GOVERNANCE_XML.as_bytes(),
        ),
        ("3D/3dmodel.model", model.as_bytes()),
        (
            "Metadata/fabrication-manifest.json",
            manifest_json.as_slice(),
        ),
        (
            "Metadata/fabrication-attestation.json",
            attestation_json.as_slice(),
        ),
        (
            "Metadata/fabrication-manifest.sha256",
            manifest_digest_hex.as_bytes(),
        ),
        (
            "Metadata/fabrication-trust-snapshot.json",
            trust_json.as_slice(),
        ),
        (
            "Metadata/fabrication-trust-snapshot.sha256",
            trust_digest_hex.as_bytes(),
        ),
        (
            "Metadata/fabrication-audit-journal.json",
            audit_json.as_slice(),
        ),
        (
            "Metadata/fabrication-audit-journal.sha256",
            audit_digest_hex.as_bytes(),
        ),
        (
            "Metadata/fabrication-audit-head.sha256",
            audit_head_hex.as_bytes(),
        ),
    ]))
}

#[derive(Debug)]
struct CentralDirectoryEntry {
    name: String,
    crc32: u32,
    size: u32,
    local_header_offset: u32,
}

fn write_stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut directory = Vec::with_capacity(entries.len());

    for (name, data) in entries {
        let name_bytes = name.as_bytes();
        let size = u32::try_from(data.len()).expect("3MF ZIP entry exceeds 4 GiB");
        let name_length = u16::try_from(name_bytes.len()).expect("3MF ZIP path is too long");
        let local_header_offset =
            u32::try_from(output.len()).expect("3MF ZIP package exceeds 4 GiB");
        let checksum = crc32(data);

        push_u32(&mut output, 0x0403_4b50);
        push_u16(&mut output, 20);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u32(&mut output, checksum);
        push_u32(&mut output, size);
        push_u32(&mut output, size);
        push_u16(&mut output, name_length);
        push_u16(&mut output, 0);
        output.extend_from_slice(name_bytes);
        output.extend_from_slice(data);

        directory.push(CentralDirectoryEntry {
            name: (*name).to_string(),
            crc32: checksum,
            size,
            local_header_offset,
        });
    }

    let central_directory_offset =
        u32::try_from(output.len()).expect("3MF ZIP package exceeds 4 GiB");
    for entry in &directory {
        let name = entry.name.as_bytes();
        push_u32(&mut output, 0x0201_4b50);
        push_u16(&mut output, 20);
        push_u16(&mut output, 20);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u32(&mut output, entry.crc32);
        push_u32(&mut output, entry.size);
        push_u32(&mut output, entry.size);
        push_u16(&mut output, name.len() as u16);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u16(&mut output, 0);
        push_u32(&mut output, 0);
        push_u32(&mut output, entry.local_header_offset);
        output.extend_from_slice(name);
    }

    let central_directory_size = u32::try_from(output.len())
        .expect("3MF ZIP package exceeds 4 GiB")
        - central_directory_offset;
    let entry_count = u16::try_from(directory.len()).expect("too many 3MF ZIP entries");
    push_u32(&mut output, 0x0605_4b50);
    push_u16(&mut output, 0);
    push_u16(&mut output, 0);
    push_u16(&mut output, entry_count);
    push_u16(&mut output, entry_count);
    push_u32(&mut output, central_directory_size);
    push_u32(&mut output, central_directory_offset);
    push_u16(&mut output, 0);
    output
}

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

/// Compatibility alias for [`export_3mf_model_xml`].
///
/// The return value is model XML, not a complete binary `.3mf` package.
#[deprecated(
    since = "0.6.0",
    note = "use export_3mf_model_xml; export_3mf returns XML, not a packaged .3mf file"
)]
pub fn export_3mf(mesh: &TriangleMesh) -> String {
    export_3mf_model_xml(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::AttestedFabricationManifest;
    use crate::audit::{AuditJournal, digest_audit_journal};
    use crate::mesh::TriangleMesh;

    fn simple_triangle() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            indices: vec![[0, 1, 2]],
        }
    }

    #[test]
    fn test_stl_empty() {
        let mesh = TriangleMesh::empty();
        let stl = export_stl(&mesh);
        assert_eq!(stl.len(), 84); // header + count, no triangles
    }

    #[test]
    fn test_stl_size() {
        let mesh = simple_triangle();
        let stl = export_stl(&mesh);
        assert_eq!(stl.len(), 84 + 50); // 1 triangle
    }

    #[test]
    fn test_stl_triangle_count() {
        let stl = export_stl(&simple_triangle());
        let count = u32::from_le_bytes([stl[80], stl[81], stl[82], stl[83]]);
        assert_eq!(count, 1);
    }

    #[test]
    fn packaged_3mf_contains_required_opc_members() {
        let package = export_3mf_package(&simple_triangle());
        assert_eq!(&package[0..4], &0x0403_4b50u32.to_le_bytes());
        assert!(
            package
                .windows("[Content_Types].xml".len())
                .any(|window| window == b"[Content_Types].xml")
        );
        assert!(
            package
                .windows("_rels/.rels".len())
                .any(|window| window == b"_rels/.rels")
        );
        assert!(
            package
                .windows("3D/3dmodel.model".len())
                .any(|window| window == b"3D/3dmodel.model")
        );
        let end_of_central_directory = 0x0605_4b50u32.to_le_bytes();
        assert!(
            package
                .windows(4)
                .any(|window| window == &end_of_central_directory[..])
        );
    }

    #[test]
    fn crc32_matches_standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn test_3mf_xml() {
        let xml = export_3mf_model_xml(&simple_triangle());
        assert!(xml.contains("<vertex"));
        assert!(xml.contains("<triangle"));
        assert!(xml.contains("3dmanufacturing"));
    }

    #[test]
    fn packaged_3mf_can_bind_fabrication_manifest() {
        use crate::provenance::{FabricationManifest, StableFingerprint};

        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        let manifest = FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        };
        let package = export_3mf_package_with_manifest(&simple_triangle(), &manifest).unwrap();
        assert!(
            package
                .windows("Metadata/fabrication-manifest.json".len())
                .any(|window| window == b"Metadata/fabrication-manifest.json")
        );
        assert!(
            package
                .windows("symthaea.fabrication.manifest.v1".len())
                .any(|window| window == b"symthaea.fabrication.manifest.v1")
        );
        assert!(
            package
                .windows("fabrication-manifest".len())
                .any(|window| window == b"fabrication-manifest")
        );
    }

    #[test]
    fn attested_package_contains_digest_and_signature_envelope() {
        use crate::attestation::AttestedFabricationManifest;
        use crate::audit::{AuditJournal, digest_audit_journal};
        use crate::crypto_digest::sha256;
        use crate::provenance::{FabricationManifest, StableFingerprint};

        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        let manifest = FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        };
        let attested = AttestedFabricationManifest {
            schema_version: "symthaea.fabrication.attestation.v1".into(),
            manifest,
            manifest_digest: sha256(b"test"),
            signatures: Vec::new(),
        };
        let package = export_3mf_package_with_attestation(&simple_triangle(), &attested).unwrap();
        for path in [
            "Metadata/fabrication-manifest.json",
            "Metadata/fabrication-attestation.json",
            "Metadata/fabrication-manifest.sha256",
        ] {
            assert!(
                package
                    .windows(path.len())
                    .any(|window| window == path.as_bytes())
            );
        }
        assert!(
            package
                .windows(64)
                .any(|window| window == attested.manifest_digest.to_hex().as_bytes())
        );
    }
}
