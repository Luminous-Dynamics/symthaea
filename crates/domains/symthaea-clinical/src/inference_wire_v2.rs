// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Serializer-independent canonical binary wire contract for clinical inference v2.
//!
//! The v2 evidence semantics are intended for cross-system medical interoperability,
//! so the identity-bearing representation must not depend on JSON serializer behavior.
//! This module defines an explicit, bounded binary framing and a domain-separated digest.
//! It still carries **no clinical authority**.

use crate::claims::{
    ClinicalApplicability, ClinicalClaimKind, ClinicalClaimSemanticsV1, ClinicalEvidenceStage,
    ClinicalIntendedUseClass, CLINICAL_CLAIM_VOCABULARY_VERSION,
};
use crate::inference::{
    AlternativeClinicalHypothesisV1, ClinicalArtifactIdentityV1, ClinicalCalibrationStatusV1,
    ClinicalDigestAlgorithmV1, ClinicalDigestV1, ClinicalDistributionStatusV1,
    ClinicalEvidenceRoleV1, MissingClinicalEvidenceV1, MissingEvidenceCriticalityV1,
};
use crate::inference_v2::{
    ClinicalDistributionAssessmentV2, ClinicalEvidenceIdentityV2, ClinicalEvidenceRefV2,
    ClinicalExecutionIdentityV2, ClinicalInferenceEnvelopeV2, ClinicalInferenceEnvelopeV2Error,
    ClinicalModelIdentityV2, ClinicalSubjectBindingV2, ClinicalUncertaintyV2,
    CLINICAL_EVIDENCE_IDENTITY_V2_VERSION, CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
};

/// Version of this binary framing, independent from envelope schema version.
pub const CLINICAL_INFERENCE_WIRE_V2_VERSION: u16 = 1;

const MAGIC: &[u8; 8] = b"SYMCLN2\0";
const DERIVE_KEY_CONTEXT: &str = "symthaea.clinical.inference-wire-v2.v1";
const DOMAIN_TAG: &[u8] = b"symthaea/clinical-inference-wire/v2/v1";

const MAX_WIRE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TEXT_BYTES: usize = 256 * 1024;
const MAX_ID_BYTES: usize = 4 * 1024;
const MAX_VECTOR_ITEMS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClinicalInferenceWireDigestV2([u8; 32]);

impl ClinicalInferenceWireDigestV2 {
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    #[must_use]
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Validate and encode one v2 envelope into its unique canonical binary representation.
pub fn clinical_inference_wire_v2_bytes(
    envelope: &ClinicalInferenceEnvelopeV2,
) -> Result<Vec<u8>, ClinicalInferenceWireV2Error> {
    envelope
        .validate()
        .map_err(ClinicalInferenceWireV2Error::InvalidEnvelope)?;

    let mut writer = Writer::default();
    writer.raw(MAGIC)?;
    writer.u16(CLINICAL_INFERENCE_WIRE_V2_VERSION);
    encode_envelope(&mut writer, envelope)?;
    let bytes = writer.finish();
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(ClinicalInferenceWireV2Error::WireTooLarge);
    }
    Ok(bytes)
}

/// Parse the canonical binary representation, rejecting truncation, invalid tags,
/// excessive lengths, trailing bytes, and any envelope that fails semantic validation.
pub fn parse_clinical_inference_wire_v2(
    bytes: &[u8],
) -> Result<ClinicalInferenceEnvelopeV2, ClinicalInferenceWireV2Error> {
    if bytes.is_empty() || bytes.len() > MAX_WIRE_BYTES {
        return Err(ClinicalInferenceWireV2Error::InvalidWireLength);
    }
    let mut reader = Reader::new(bytes);
    let magic = reader.take_exact(MAGIC.len())?;
    if magic != MAGIC {
        return Err(ClinicalInferenceWireV2Error::WrongMagic);
    }
    let wire_version = reader.u16()?;
    if wire_version != CLINICAL_INFERENCE_WIRE_V2_VERSION {
        return Err(ClinicalInferenceWireV2Error::UnsupportedWireVersion(wire_version));
    }
    let envelope = decode_envelope(&mut reader)?;
    if !reader.is_finished() {
        return Err(ClinicalInferenceWireV2Error::TrailingBytes);
    }
    envelope
        .validate()
        .map_err(ClinicalInferenceWireV2Error::InvalidEnvelope)?;
    Ok(envelope)
}

/// Compute a domain-separated digest over exact validated canonical v2 bytes.
pub fn clinical_inference_wire_v2_digest(
    envelope: &ClinicalInferenceEnvelopeV2,
) -> Result<ClinicalInferenceWireDigestV2, ClinicalInferenceWireV2Error> {
    let bytes = clinical_inference_wire_v2_bytes(envelope)?;
    digest_canonical_bytes(&bytes)
}

/// Verify that supplied bytes are canonical by parsing and re-encoding them before digesting.
pub fn clinical_inference_wire_v2_digest_from_bytes(
    bytes: &[u8],
) -> Result<ClinicalInferenceWireDigestV2, ClinicalInferenceWireV2Error> {
    let envelope = parse_clinical_inference_wire_v2(bytes)?;
    let canonical = clinical_inference_wire_v2_bytes(&envelope)?;
    if canonical != bytes {
        return Err(ClinicalInferenceWireV2Error::NonCanonicalWire);
    }
    digest_canonical_bytes(bytes)
}

fn digest_canonical_bytes(
    bytes: &[u8],
) -> Result<ClinicalInferenceWireDigestV2, ClinicalInferenceWireV2Error> {
    if bytes.is_empty() || bytes.len() > MAX_WIRE_BYTES {
        return Err(ClinicalInferenceWireV2Error::InvalidWireLength);
    }
    let mut hasher = blake3::Hasher::new_derive_key(DERIVE_KEY_CONTEXT);
    hasher.update(&CLINICAL_INFERENCE_WIRE_V2_VERSION.to_be_bytes());
    hasher.update(&(DOMAIN_TAG.len() as u16).to_be_bytes());
    hasher.update(DOMAIN_TAG);
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
    Ok(ClinicalInferenceWireDigestV2(*hasher.finalize().as_bytes()))
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn raw(&mut self, value: &[u8]) -> Result<(), ClinicalInferenceWireV2Error> {
        self.ensure_room(value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn ensure_room(&self, additional: usize) -> Result<(), ClinicalInferenceWireV2Error> {
        if self.bytes.len().saturating_add(additional) > MAX_WIRE_BYTES {
            return Err(ClinicalInferenceWireV2Error::WireTooLarge);
        }
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), ClinicalInferenceWireV2Error> {
        self.raw(&[value])
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn fixed(&mut self, value: &[u8]) -> Result<(), ClinicalInferenceWireV2Error> {
        self.raw(value)
    }

    fn string(&mut self, value: &str, max: usize) -> Result<(), ClinicalInferenceWireV2Error> {
        if value.len() > max {
            return Err(ClinicalInferenceWireV2Error::FieldTooLarge);
        }
        let len = u32::try_from(value.len()).map_err(|_| ClinicalInferenceWireV2Error::LengthOverflow)?;
        self.u32(len);
        self.raw(value.as_bytes())
    }

    fn vector_len(&mut self, len: usize) -> Result<(), ClinicalInferenceWireV2Error> {
        if len > MAX_VECTOR_ITEMS {
            return Err(ClinicalInferenceWireV2Error::TooManyItems);
        }
        self.u32(u32::try_from(len).map_err(|_| ClinicalInferenceWireV2Error::LengthOverflow)?);
        Ok(())
    }

    fn option<T>(
        &mut self,
        value: Option<&T>,
        encode: impl FnOnce(&mut Self, &T) -> Result<(), ClinicalInferenceWireV2Error>,
    ) -> Result<(), ClinicalInferenceWireV2Error> {
        match value {
            Some(value) => {
                self.u8(1)?;
                encode(self, value)
            }
            None => self.u8(0),
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn take_exact(&mut self, len: usize) -> Result<&'a [u8], ClinicalInferenceWireV2Error> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ClinicalInferenceWireV2Error::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(ClinicalInferenceWireV2Error::TruncatedWire);
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ClinicalInferenceWireV2Error> {
        Ok(self.take_exact(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ClinicalInferenceWireV2Error> {
        let bytes: [u8; 2] = self
            .take_exact(2)?
            .try_into()
            .map_err(|_| ClinicalInferenceWireV2Error::TruncatedWire)?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, ClinicalInferenceWireV2Error> {
        let bytes: [u8; 4] = self
            .take_exact(4)?
            .try_into()
            .map_err(|_| ClinicalInferenceWireV2Error::TruncatedWire)?;
        Ok(u32::from_be_bytes(bytes))
    }

    fn i64(&mut self) -> Result<i64, ClinicalInferenceWireV2Error> {
        let bytes: [u8; 8] = self
            .take_exact(8)?
            .try_into()
            .map_err(|_| ClinicalInferenceWireV2Error::TruncatedWire)?;
        Ok(i64::from_be_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, ClinicalInferenceWireV2Error> {
        let bytes: [u8; 8] = self
            .take_exact(8)?
            .try_into()
            .map_err(|_| ClinicalInferenceWireV2Error::TruncatedWire)?;
        Ok(u64::from_be_bytes(bytes))
    }

    fn f64(&mut self) -> Result<f64, ClinicalInferenceWireV2Error> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], ClinicalInferenceWireV2Error> {
        self.take_exact(N)?
            .try_into()
            .map_err(|_| ClinicalInferenceWireV2Error::TruncatedWire)
    }

    fn string(&mut self, max: usize) -> Result<String, ClinicalInferenceWireV2Error> {
        let len = usize::try_from(self.u32()?).map_err(|_| ClinicalInferenceWireV2Error::LengthOverflow)?;
        if len > max {
            return Err(ClinicalInferenceWireV2Error::FieldTooLarge);
        }
        let bytes = self.take_exact(len)?;
        let value = std::str::from_utf8(bytes).map_err(|_| ClinicalInferenceWireV2Error::InvalidUtf8)?;
        Ok(value.to_owned())
    }

    fn vector_len(&mut self) -> Result<usize, ClinicalInferenceWireV2Error> {
        let len = usize::try_from(self.u32()?).map_err(|_| ClinicalInferenceWireV2Error::LengthOverflow)?;
        if len > MAX_VECTOR_ITEMS {
            return Err(ClinicalInferenceWireV2Error::TooManyItems);
        }
        Ok(len)
    }

    fn option<T>(
        &mut self,
        decode: impl FnOnce(&mut Self) -> Result<T, ClinicalInferenceWireV2Error>,
    ) -> Result<Option<T>, ClinicalInferenceWireV2Error> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(decode(self)?)),
            tag => Err(ClinicalInferenceWireV2Error::InvalidOptionTag(tag)),
        }
    }
}

fn encode_envelope(
    w: &mut Writer,
    value: &ClinicalInferenceEnvelopeV2,
) -> Result<(), ClinicalInferenceWireV2Error> {
    w.u16(value.schema_version);
    encode_semantics(w, &value.semantics)?;
    w.option(value.subject.as_ref(), encode_subject)?;
    w.string(&value.statement, MAX_TEXT_BYTES)?;
    w.vector_len(value.evidence.len())?;
    for item in &value.evidence {
        encode_evidence_ref(w, item)?;
    }
    w.vector_len(value.alternatives.len())?;
    for item in &value.alternatives {
        encode_alternative(w, item)?;
    }
    w.vector_len(value.missing_evidence.len())?;
    for item in &value.missing_evidence {
        encode_missing(w, item)?;
    }
    encode_uncertainty(w, &value.uncertainty)?;
    encode_distribution(w, &value.distribution)?;
    encode_execution(w, &value.execution)?;
    w.i64(value.generated_at_micros);
    Ok(())
}

fn decode_envelope(r: &mut Reader<'_>) -> Result<ClinicalInferenceEnvelopeV2, ClinicalInferenceWireV2Error> {
    let schema_version = r.u16()?;
    let semantics = decode_semantics(r)?;
    let subject = r.option(decode_subject)?;
    let statement = r.string(MAX_TEXT_BYTES)?;
    let evidence_len = r.vector_len()?;
    let mut evidence = Vec::with_capacity(evidence_len);
    for _ in 0..evidence_len {
        evidence.push(decode_evidence_ref(r)?);
    }
    let alternatives_len = r.vector_len()?;
    let mut alternatives = Vec::with_capacity(alternatives_len);
    for _ in 0..alternatives_len {
        alternatives.push(decode_alternative(r)?);
    }
    let missing_len = r.vector_len()?;
    let mut missing_evidence = Vec::with_capacity(missing_len);
    for _ in 0..missing_len {
        missing_evidence.push(decode_missing(r)?);
    }
    let uncertainty = decode_uncertainty(r)?;
    let distribution = decode_distribution(r)?;
    let execution = decode_execution(r)?;
    let generated_at_micros = r.i64()?;
    Ok(ClinicalInferenceEnvelopeV2 {
        schema_version,
        semantics,
        subject,
        statement,
        evidence,
        alternatives,
        missing_evidence,
        uncertainty,
        distribution,
        execution,
        generated_at_micros,
    })
}

fn encode_digest(w: &mut Writer, value: &ClinicalDigestV1) -> Result<(), ClinicalInferenceWireV2Error> {
    w.u8(match value.algorithm {
        ClinicalDigestAlgorithmV1::Blake3_256 => 0,
    })?;
    w.fixed(&value.value)
}

fn decode_digest(r: &mut Reader<'_>) -> Result<ClinicalDigestV1, ClinicalInferenceWireV2Error> {
    let algorithm = match r.u8()? {
        0 => ClinicalDigestAlgorithmV1::Blake3_256,
        tag => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("digest algorithm", tag)),
    };
    Ok(ClinicalDigestV1 {
        algorithm,
        value: r.fixed()?,
    })
}

fn encode_artifact(w: &mut Writer, value: &ClinicalArtifactIdentityV1) -> Result<(), ClinicalInferenceWireV2Error> {
    w.string(&value.name, MAX_ID_BYTES)?;
    w.string(&value.version, MAX_ID_BYTES)?;
    encode_digest(w, &value.digest)
}

fn decode_artifact(r: &mut Reader<'_>) -> Result<ClinicalArtifactIdentityV1, ClinicalInferenceWireV2Error> {
    Ok(ClinicalArtifactIdentityV1 {
        name: r.string(MAX_ID_BYTES)?,
        version: r.string(MAX_ID_BYTES)?,
        digest: decode_digest(r)?,
    })
}

fn encode_evidence_identity(w: &mut Writer, value: &ClinicalEvidenceIdentityV2) -> Result<(), ClinicalInferenceWireV2Error> {
    w.u16(value.identity_version);
    w.string(&value.namespace, MAX_ID_BYTES)?;
    w.string(&value.artifact_id, MAX_ID_BYTES)?;
    encode_digest(w, &value.digest)
}

fn decode_evidence_identity(r: &mut Reader<'_>) -> Result<ClinicalEvidenceIdentityV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalEvidenceIdentityV2 {
        identity_version: r.u16()?,
        namespace: r.string(MAX_ID_BYTES)?,
        artifact_id: r.string(MAX_ID_BYTES)?,
        digest: decode_digest(r)?,
    })
}

fn encode_semantics(w: &mut Writer, value: &ClinicalClaimSemanticsV1) -> Result<(), ClinicalInferenceWireV2Error> {
    w.u16(value.schema_version);
    w.u8(claim_kind_tag(value.claim_kind))?;
    w.u8(evidence_stage_tag(value.evidence_stage))?;
    w.u8(applicability_tag(value.applicability))?;
    w.u8(intended_use_tag(value.intended_use))?;
    Ok(())
}

fn decode_semantics(r: &mut Reader<'_>) -> Result<ClinicalClaimSemanticsV1, ClinicalInferenceWireV2Error> {
    let schema_version = r.u16()?;
    if schema_version != CLINICAL_CLAIM_VOCABULARY_VERSION {
        return Err(ClinicalInferenceWireV2Error::UnsupportedClaimVocabularyVersion(schema_version));
    }
    Ok(ClinicalClaimSemanticsV1 {
        schema_version,
        claim_kind: decode_claim_kind(r.u8()?)?,
        evidence_stage: decode_evidence_stage(r.u8()?)?,
        applicability: decode_applicability(r.u8()?)?,
        intended_use: decode_intended_use(r.u8()?)?,
    })
}

fn encode_subject(w: &mut Writer, value: &ClinicalSubjectBindingV2) -> Result<(), ClinicalInferenceWireV2Error> {
    w.string(&value.subject_namespace, MAX_ID_BYTES)?;
    w.string(&value.subject_id, MAX_ID_BYTES)?;
    encode_evidence_identity(w, &value.binding_evidence)
}

fn decode_subject(r: &mut Reader<'_>) -> Result<ClinicalSubjectBindingV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalSubjectBindingV2 {
        subject_namespace: r.string(MAX_ID_BYTES)?,
        subject_id: r.string(MAX_ID_BYTES)?,
        binding_evidence: decode_evidence_identity(r)?,
    })
}

fn encode_evidence_ref(w: &mut Writer, value: &ClinicalEvidenceRefV2) -> Result<(), ClinicalInferenceWireV2Error> {
    encode_evidence_identity(w, &value.identity)?;
    w.u8(evidence_role_tag(value.role))
}

fn decode_evidence_ref(r: &mut Reader<'_>) -> Result<ClinicalEvidenceRefV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalEvidenceRefV2 {
        identity: decode_evidence_identity(r)?,
        role: decode_evidence_role(r.u8()?)?,
    })
}

fn encode_alternative(w: &mut Writer, value: &AlternativeClinicalHypothesisV1) -> Result<(), ClinicalInferenceWireV2Error> {
    w.string(&value.statement, MAX_TEXT_BYTES)?;
    encode_semantics(w, &value.semantics)?;
    w.string(&value.rationale, MAX_TEXT_BYTES)
}

fn decode_alternative(r: &mut Reader<'_>) -> Result<AlternativeClinicalHypothesisV1, ClinicalInferenceWireV2Error> {
    Ok(AlternativeClinicalHypothesisV1 {
        statement: r.string(MAX_TEXT_BYTES)?,
        semantics: decode_semantics(r)?,
        rationale: r.string(MAX_TEXT_BYTES)?,
    })
}

fn encode_missing(w: &mut Writer, value: &MissingClinicalEvidenceV1) -> Result<(), ClinicalInferenceWireV2Error> {
    w.string(&value.requirement_id, MAX_ID_BYTES)?;
    w.string(&value.description, MAX_TEXT_BYTES)?;
    w.u8(missing_criticality_tag(value.criticality))
}

fn decode_missing(r: &mut Reader<'_>) -> Result<MissingClinicalEvidenceV1, ClinicalInferenceWireV2Error> {
    Ok(MissingClinicalEvidenceV1 {
        requirement_id: r.string(MAX_ID_BYTES)?,
        description: r.string(MAX_TEXT_BYTES)?,
        criticality: decode_missing_criticality(r.u8()?)?,
    })
}

fn encode_model(w: &mut Writer, value: &ClinicalModelIdentityV2) -> Result<(), ClinicalInferenceWireV2Error> {
    encode_artifact(w, &value.model)?;
    encode_digest(w, &value.input_schema_digest)?;
    encode_digest(w, &value.output_schema_digest)?;
    w.option(value.training_lineage.as_ref(), encode_evidence_identity)?;
    w.option(value.evaluation_lineage.as_ref(), encode_evidence_identity)?;
    w.option(value.calibration_evidence.as_ref(), encode_evidence_identity)
}

fn decode_model(r: &mut Reader<'_>) -> Result<ClinicalModelIdentityV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalModelIdentityV2 {
        model: decode_artifact(r)?,
        input_schema_digest: decode_digest(r)?,
        output_schema_digest: decode_digest(r)?,
        training_lineage: r.option(decode_evidence_identity)?,
        evaluation_lineage: r.option(decode_evidence_identity)?,
        calibration_evidence: r.option(decode_evidence_identity)?,
    })
}

fn encode_uncertainty(w: &mut Writer, value: &ClinicalUncertaintyV2) -> Result<(), ClinicalInferenceWireV2Error> {
    encode_optional_f64(w, value.epistemic)?;
    encode_optional_f64(w, value.aleatoric)?;
    encode_optional_f64(w, value.calibrated_probability)?;
    w.u8(calibration_status_tag(value.calibration_status))?;
    w.option(value.calibration_evidence.as_ref(), encode_evidence_identity)
}

fn decode_uncertainty(r: &mut Reader<'_>) -> Result<ClinicalUncertaintyV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalUncertaintyV2 {
        epistemic: decode_optional_f64(r)?,
        aleatoric: decode_optional_f64(r)?,
        calibrated_probability: decode_optional_f64(r)?,
        calibration_status: decode_calibration_status(r.u8()?)?,
        calibration_evidence: r.option(decode_evidence_identity)?,
    })
}

fn encode_distribution(w: &mut Writer, value: &ClinicalDistributionAssessmentV2) -> Result<(), ClinicalInferenceWireV2Error> {
    w.u8(distribution_status_tag(value.status))?;
    w.option(value.detector_evidence.as_ref(), encode_evidence_identity)
}

fn decode_distribution(r: &mut Reader<'_>) -> Result<ClinicalDistributionAssessmentV2, ClinicalInferenceWireV2Error> {
    Ok(ClinicalDistributionAssessmentV2 {
        status: decode_distribution_status(r.u8()?)?,
        detector_evidence: r.option(decode_evidence_identity)?,
    })
}

fn encode_execution(w: &mut Writer, value: &ClinicalExecutionIdentityV2) -> Result<(), ClinicalInferenceWireV2Error> {
    encode_artifact(w, &value.engine)?;
    encode_model(w, &value.model)?;
    encode_digest(w, &value.runtime_digest)?;
    encode_digest(w, &value.configuration_digest)?;
    w.vector_len(value.input_evidence.len())?;
    for input in &value.input_evidence {
        encode_evidence_identity(w, input)?;
    }
    w.string(&value.operation, MAX_ID_BYTES)?;
    w.i64(value.executed_at_micros);
    w.fixed(&value.execution_nonce)
}

fn decode_execution(r: &mut Reader<'_>) -> Result<ClinicalExecutionIdentityV2, ClinicalInferenceWireV2Error> {
    let engine = decode_artifact(r)?;
    let model = decode_model(r)?;
    let runtime_digest = decode_digest(r)?;
    let configuration_digest = decode_digest(r)?;
    let len = r.vector_len()?;
    let mut input_evidence = Vec::with_capacity(len);
    for _ in 0..len {
        input_evidence.push(decode_evidence_identity(r)?);
    }
    Ok(ClinicalExecutionIdentityV2 {
        engine,
        model,
        runtime_digest,
        configuration_digest,
        input_evidence,
        operation: r.string(MAX_ID_BYTES)?,
        executed_at_micros: r.i64()?,
        execution_nonce: r.fixed()?,
    })
}

fn encode_optional_f64(w: &mut Writer, value: Option<f64>) -> Result<(), ClinicalInferenceWireV2Error> {
    match value {
        Some(value) => {
            w.u8(1)?;
            w.f64(value);
            Ok(())
        }
        None => w.u8(0),
    }
}

fn decode_optional_f64(r: &mut Reader<'_>) -> Result<Option<f64>, ClinicalInferenceWireV2Error> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(r.f64()?)),
        tag => Err(ClinicalInferenceWireV2Error::InvalidOptionTag(tag)),
    }
}

fn claim_kind_tag(value: ClinicalClaimKind) -> u8 {
    match value {
        ClinicalClaimKind::CandidateSignal => 0,
        ClinicalClaimKind::Association => 1,
        ClinicalClaimKind::Prediction => 2,
        ClinicalClaimKind::RiskEstimate => 3,
        ClinicalClaimKind::CausalHypothesis => 4,
        ClinicalClaimKind::CausalEffectEstimate => 5,
        ClinicalClaimKind::DiagnosticSupport => 6,
        ClinicalClaimKind::TreatmentSupport => 7,
    }
}

fn decode_claim_kind(tag: u8) -> Result<ClinicalClaimKind, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalClaimKind::CandidateSignal,
        1 => ClinicalClaimKind::Association,
        2 => ClinicalClaimKind::Prediction,
        3 => ClinicalClaimKind::RiskEstimate,
        4 => ClinicalClaimKind::CausalHypothesis,
        5 => ClinicalClaimKind::CausalEffectEstimate,
        6 => ClinicalClaimKind::DiagnosticSupport,
        7 => ClinicalClaimKind::TreatmentSupport,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("claim kind", tag)),
    })
}

fn evidence_stage_tag(value: ClinicalEvidenceStage) -> u8 {
    match value {
        ClinicalEvidenceStage::MechanisticHypothesis => 0,
        ClinicalEvidenceStage::SyntheticDemonstration => 1,
        ClinicalEvidenceStage::RetrospectiveInternal => 2,
        ClinicalEvidenceStage::RetrospectiveExternal => 3,
        ClinicalEvidenceStage::ProspectiveShadow => 4,
        ClinicalEvidenceStage::ProspectiveClinicalStudy => 5,
        ClinicalEvidenceStage::ReplicatedClinicalEvidence => 6,
    }
}

fn decode_evidence_stage(tag: u8) -> Result<ClinicalEvidenceStage, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalEvidenceStage::MechanisticHypothesis,
        1 => ClinicalEvidenceStage::SyntheticDemonstration,
        2 => ClinicalEvidenceStage::RetrospectiveInternal,
        3 => ClinicalEvidenceStage::RetrospectiveExternal,
        4 => ClinicalEvidenceStage::ProspectiveShadow,
        5 => ClinicalEvidenceStage::ProspectiveClinicalStudy,
        6 => ClinicalEvidenceStage::ReplicatedClinicalEvidence,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("evidence stage", tag)),
    })
}

fn applicability_tag(value: ClinicalApplicability) -> u8 {
    match value {
        ClinicalApplicability::Unestablished => 0,
        ClinicalApplicability::EvaluatedCohortOnly => 1,
        ClinicalApplicability::DefinedTargetPopulation => 2,
        ClinicalApplicability::ValidatedTargetPopulation => 3,
    }
}

fn decode_applicability(tag: u8) -> Result<ClinicalApplicability, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalApplicability::Unestablished,
        1 => ClinicalApplicability::EvaluatedCohortOnly,
        2 => ClinicalApplicability::DefinedTargetPopulation,
        3 => ClinicalApplicability::ValidatedTargetPopulation,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("applicability", tag)),
    })
}

fn intended_use_tag(value: ClinicalIntendedUseClass) -> u8 {
    match value {
        ClinicalIntendedUseClass::ResearchOnly => 0,
        ClinicalIntendedUseClass::ClinicalDecisionSupport => 1,
    }
}

fn decode_intended_use(tag: u8) -> Result<ClinicalIntendedUseClass, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalIntendedUseClass::ResearchOnly,
        1 => ClinicalIntendedUseClass::ClinicalDecisionSupport,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("intended use", tag)),
    })
}

fn evidence_role_tag(value: ClinicalEvidenceRoleV1) -> u8 {
    match value {
        ClinicalEvidenceRoleV1::Supports => 0,
        ClinicalEvidenceRoleV1::Opposes => 1,
        ClinicalEvidenceRoleV1::Context => 2,
        ClinicalEvidenceRoleV1::Contraindication => 3,
    }
}

fn decode_evidence_role(tag: u8) -> Result<ClinicalEvidenceRoleV1, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalEvidenceRoleV1::Supports,
        1 => ClinicalEvidenceRoleV1::Opposes,
        2 => ClinicalEvidenceRoleV1::Context,
        3 => ClinicalEvidenceRoleV1::Contraindication,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("evidence role", tag)),
    })
}

fn calibration_status_tag(value: ClinicalCalibrationStatusV1) -> u8 {
    match value {
        ClinicalCalibrationStatusV1::NotAssessed => 0,
        ClinicalCalibrationStatusV1::Uncalibrated => 1,
        ClinicalCalibrationStatusV1::Calibrated => 2,
    }
}

fn decode_calibration_status(tag: u8) -> Result<ClinicalCalibrationStatusV1, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalCalibrationStatusV1::NotAssessed,
        1 => ClinicalCalibrationStatusV1::Uncalibrated,
        2 => ClinicalCalibrationStatusV1::Calibrated,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("calibration status", tag)),
    })
}

fn distribution_status_tag(value: ClinicalDistributionStatusV1) -> u8 {
    match value {
        ClinicalDistributionStatusV1::Unknown => 0,
        ClinicalDistributionStatusV1::InDistribution => 1,
        ClinicalDistributionStatusV1::OutOfDistribution => 2,
    }
}

fn decode_distribution_status(tag: u8) -> Result<ClinicalDistributionStatusV1, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => ClinicalDistributionStatusV1::Unknown,
        1 => ClinicalDistributionStatusV1::InDistribution,
        2 => ClinicalDistributionStatusV1::OutOfDistribution,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("distribution status", tag)),
    })
}

fn missing_criticality_tag(value: MissingEvidenceCriticalityV1) -> u8 {
    match value {
        MissingEvidenceCriticalityV1::Informational => 0,
        MissingEvidenceCriticalityV1::Important => 1,
        MissingEvidenceCriticalityV1::Critical => 2,
    }
}

fn decode_missing_criticality(tag: u8) -> Result<MissingEvidenceCriticalityV1, ClinicalInferenceWireV2Error> {
    Ok(match tag {
        0 => MissingEvidenceCriticalityV1::Informational,
        1 => MissingEvidenceCriticalityV1::Important,
        2 => MissingEvidenceCriticalityV1::Critical,
        _ => return Err(ClinicalInferenceWireV2Error::InvalidEnumTag("missing criticality", tag)),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClinicalInferenceWireV2Error {
    InvalidEnvelope(ClinicalInferenceEnvelopeV2Error),
    WrongMagic,
    UnsupportedWireVersion(u16),
    UnsupportedClaimVocabularyVersion(u16),
    InvalidWireLength,
    WireTooLarge,
    FieldTooLarge,
    TooManyItems,
    LengthOverflow,
    TruncatedWire,
    InvalidUtf8,
    InvalidOptionTag(u8),
    InvalidEnumTag(&'static str, u8),
    TrailingBytes,
    NonCanonicalWire,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{
        ClinicalApplicability, ClinicalClaimKind, ClinicalEvidenceStage,
    };
    use crate::inference_v2::{
        ClinicalEvidenceIdentityV2, CLINICAL_EVIDENCE_IDENTITY_V2_VERSION,
    };

    fn digest(byte: u8) -> ClinicalDigestV1 {
        ClinicalDigestV1::blake3([byte; 32])
    }

    fn artifact(name: &str, version: &str, byte: u8) -> ClinicalArtifactIdentityV1 {
        ClinicalArtifactIdentityV1 {
            name: name.into(),
            version: version.into(),
            digest: digest(byte),
        }
    }

    fn evidence(namespace: &str, artifact_id: &str, byte: u8) -> ClinicalEvidenceIdentityV2 {
        ClinicalEvidenceIdentityV2 {
            identity_version: CLINICAL_EVIDENCE_IDENTITY_V2_VERSION,
            namespace: namespace.into(),
            artifact_id: artifact_id.into(),
            digest: digest(byte),
        }
    }

    fn envelope() -> ClinicalInferenceEnvelopeV2 {
        let fact = evidence("mycelix/clinical-fact-snapshot/v1", "fact-1", 11);
        let subject_binding = evidence("mycelix/patient-subject-binding-evidence/v1", "binding-1", 10);
        let training = evidence("symthaea/model-training-lineage/v1", "train-1", 5);
        let evaluation = evidence("symthaea/model-evaluation-lineage/v1", "eval-1", 6);
        let calibration = evidence("symthaea/model-calibration-evidence/v1", "cal-1", 12);
        let detector = evidence("symthaea/ood-detector-evidence/v1", "ood-1", 13);
        ClinicalInferenceEnvelopeV2 {
            schema_version: CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
            semantics: ClinicalClaimSemanticsV1::new(
                ClinicalClaimKind::Prediction,
                ClinicalEvidenceStage::RetrospectiveExternal,
                ClinicalApplicability::DefinedTargetPopulation,
                ClinicalIntendedUseClass::ClinicalDecisionSupport,
            ),
            subject: Some(ClinicalSubjectBindingV2 {
                subject_namespace: "fhir/Patient".into(),
                subject_id: "patient-a".into(),
                binding_evidence: subject_binding.clone(),
            }),
            statement: "Candidate risk prediction".into(),
            evidence: vec![ClinicalEvidenceRefV2 {
                identity: fact.clone(),
                role: ClinicalEvidenceRoleV1::Supports,
            }],
            alternatives: vec![AlternativeClinicalHypothesisV1 {
                statement: "Alternative explanation".into(),
                semantics: ClinicalClaimSemanticsV1::new(
                    ClinicalClaimKind::CausalHypothesis,
                    ClinicalEvidenceStage::MechanisticHypothesis,
                    ClinicalApplicability::Unestablished,
                    ClinicalIntendedUseClass::ResearchOnly,
                ),
                rationale: "Preserve competing hypothesis".into(),
            }],
            missing_evidence: vec![],
            uncertainty: ClinicalUncertaintyV2 {
                epistemic: Some(0.2),
                aleatoric: Some(0.1),
                calibrated_probability: Some(0.7),
                calibration_status: ClinicalCalibrationStatusV1::Calibrated,
                calibration_evidence: Some(calibration.clone()),
            },
            distribution: ClinicalDistributionAssessmentV2 {
                status: ClinicalDistributionStatusV1::InDistribution,
                detector_evidence: Some(detector),
            },
            execution: ClinicalExecutionIdentityV2 {
                engine: artifact("symthaea", "0.1.0", 1),
                model: ClinicalModelIdentityV2 {
                    model: artifact("clinical-model", "1.0.0", 2),
                    input_schema_digest: digest(3),
                    output_schema_digest: digest(4),
                    training_lineage: Some(training),
                    evaluation_lineage: Some(evaluation),
                    calibration_evidence: Some(calibration),
                },
                runtime_digest: digest(7),
                configuration_digest: digest(8),
                input_evidence: vec![fact, subject_binding],
                operation: "evaluate".into(),
                executed_at_micros: 1_000,
                execution_nonce: [1u8; 16],
            },
            generated_at_micros: 1_001,
        }
    }

    #[test]
    fn canonical_binary_round_trip_is_exact() {
        let original = envelope();
        let bytes = clinical_inference_wire_v2_bytes(&original).unwrap();
        let decoded = parse_clinical_inference_wire_v2(&bytes).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(clinical_inference_wire_v2_bytes(&decoded).unwrap(), bytes);
    }

    #[test]
    fn canonical_binary_digest_is_deterministic() {
        let value = envelope();
        assert_eq!(
            clinical_inference_wire_v2_digest(&value).unwrap(),
            clinical_inference_wire_v2_digest(&value).unwrap()
        );
    }

    #[test]
    fn evidence_namespace_substitution_changes_wire_digest() {
        let left = envelope();
        let mut right = left.clone();
        let replacement = "hl7/fhir-r4/resource-canonical/v1".to_string();
        right.evidence[0].identity.namespace = replacement.clone();
        right.execution.input_evidence[0].namespace = replacement;
        assert_ne!(
            clinical_inference_wire_v2_digest(&left).unwrap(),
            clinical_inference_wire_v2_digest(&right).unwrap()
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = clinical_inference_wire_v2_bytes(&envelope()).unwrap();
        bytes.push(0);
        assert_eq!(
            parse_clinical_inference_wire_v2(&bytes),
            Err(ClinicalInferenceWireV2Error::TrailingBytes)
        );
    }

    #[test]
    fn truncated_wire_is_rejected() {
        let bytes = clinical_inference_wire_v2_bytes(&envelope()).unwrap();
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            parse_clinical_inference_wire_v2(truncated),
            Err(ClinicalInferenceWireV2Error::TruncatedWire)
        ));
    }

    #[test]
    fn wrong_magic_is_rejected() {
        let mut bytes = clinical_inference_wire_v2_bytes(&envelope()).unwrap();
        bytes[0] ^= 0xff;
        assert_eq!(
            parse_clinical_inference_wire_v2(&bytes),
            Err(ClinicalInferenceWireV2Error::WrongMagic)
        );
    }

    #[test]
    fn exact_float_bits_affect_wire_identity() {
        let positive = ClinicalInferenceEnvelopeV2 {
            uncertainty: ClinicalUncertaintyV2 {
                epistemic: Some(0.0),
                ..envelope().uncertainty
            },
            ..envelope()
        };
        let negative = ClinicalInferenceEnvelopeV2 {
            uncertainty: ClinicalUncertaintyV2 {
                epistemic: Some(-0.0),
                ..envelope().uncertainty
            },
            ..envelope()
        };
        assert_ne!(
            clinical_inference_wire_v2_digest(&positive).unwrap(),
            clinical_inference_wire_v2_digest(&negative).unwrap()
        );
    }

    #[test]
    fn digest_from_bytes_requires_valid_canonical_wire() {
        let bytes = clinical_inference_wire_v2_bytes(&envelope()).unwrap();
        assert_eq!(
            clinical_inference_wire_v2_digest_from_bytes(&bytes).unwrap(),
            clinical_inference_wire_v2_digest(&envelope()).unwrap()
        );
    }
}
