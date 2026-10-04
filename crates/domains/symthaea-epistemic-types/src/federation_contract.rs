//! Substrate-neutral contract for exporting a locally admitted epistemic claim
//! to a federated provenance fabric.
//!
//! This module intentionally contains no Holochain, database, network, or truth
//! semantics. It describes what a federation adapter must carry and structurally
//! validate when transporting an explicitly admitted claim.

use crate::{ClaimRepresentationIdentity, CanonicalAdmissionReceipt, CanonicalAdmissionSubject, CanonicalStatementIdentity, ProvenanceRelation, ProvenanceRelationKind, ProvenanceValidationReport, ProvenanceView, sha256_hex};

pub const FEDERATED_CLAIM_SCHEMA_VERSION: u16 = 1;
/// Version of the canonical federated-envelope digest encoding. Bump whenever
/// fields included in canonical_digest() or their canonical encoding changes.
pub const FEDERATED_CLAIM_DIGEST_VERSION: u16 = 2;

/// A typed dependency identity exposed to a federation adapter.
///
/// Keeping the dependency kind explicit prevents a plain string from losing
/// semantic information when the same identifier happens to be reused across
/// namespaces. Resolution remains the adapter's responsibility; this type does
/// not perform I/O or encode Holochain/DHT behavior.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum FederationDependency {
    Frontier(String),
    Derivation(String),
}

impl FederationDependency {
    pub fn reference(&self) -> &str {
        match self {
            Self::Frontier(reference) | Self::Derivation(reference) => reference,
        }
    }
}

/// Substrate-neutral validation classification for a federation adapter.
///
/// This mirrors the useful three-way boundary of definitive acceptance,
/// definitive rejection, and dependency resolution without importing any
/// substrate-specific validation type.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FederationValidationOutcome {
    Valid,
    Invalid(String),
    Unresolved(Vec<FederationDependency>),
}

/// Typed identity of the entity that authored a federated claim.
///
/// This remains an identifier only. It does not prove control of keys or authenticity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClaimAuthorIdentity(String);

impl ClaimAuthorIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err("claim author identity must be non-empty");
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.0.trim().is_empty() {
            Err("claim author identity must be non-empty")
        } else {
            Ok(())
        }
    }
}

/// Typed purpose for an authorship proof.
///
/// The value is intentionally extensible and substrate-neutral. Deployments may use
/// standards-defined identifiers such as the W3C assertionMethod URI.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClaimProofPurpose(String);

impl ClaimProofPurpose {
    pub fn new(value: impl Into<String>) -> Result<Self, &'static str> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err("claim proof purpose must be non-empty");
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.0.trim().is_empty() {
            Err("claim proof purpose must be non-empty")
        } else {
            Ok(())
        }
    }
}

/// Typed authorship context for a claim.
///
/// Structural validation binds the declared author to the proof context but does not
/// perform cryptographic verification. That belongs to the eventual adapter/cryptosuite.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ClaimAuthorship {
    author: ClaimAuthorIdentity,
    proof_purpose: ClaimProofPurpose,
    verification_method: Option<String>,
}

impl ClaimAuthorship {
    pub fn new(
        author: ClaimAuthorIdentity,
        proof_purpose: ClaimProofPurpose,
        verification_method: Option<String>,
    ) -> Result<Self, &'static str> {
        if verification_method
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("claim verification method must be non-empty when present");
        }
        Ok(Self {
            author,
            proof_purpose,
            verification_method,
        })
    }

    pub fn author(&self) -> &ClaimAuthorIdentity {
        &self.author
    }

    pub fn proof_purpose(&self) -> &ClaimProofPurpose {
        &self.proof_purpose
    }

    pub fn verification_method(&self) -> Option<&str> {
        self.verification_method.as_deref()
    }

    pub fn digest(&self) -> String {
        let encoded = (
            "symthaea:claim-authorship:v1",
            self.author.as_str(),
            self.proof_purpose.as_str(),
            &self.verification_method,
        );
        let bytes = serde_json::to_vec(&encoded)
            .expect("claim authorship is serializable");
        crate::sha256_hex(&bytes)
    }

    pub fn validate_structure(&self) -> Result<(), &'static str> {
        self.author.validate_structure()?;
        self.proof_purpose.validate_structure()?;
        if self
            .verification_method
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err("claim verification method must be non-empty when present");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FederatedClaim {
    pub schema_version: u16,
    pub claim_identity: String,
    pub canonical_identity: String,
    pub provenance_family: String,
    pub author: ClaimAuthorIdentity,
    pub authorship: Option<ClaimAuthorship>,
    pub statement_ref: String,
    pub source_event: Option<String>,
    pub frontier_ref: Option<String>,
    pub provenance_snapshot_digest: String,
    pub provenance_validation: ProvenanceValidationReport,
    pub admission_receipt: CanonicalAdmissionReceipt,
    pub epistemic_state: Option<String>,
    pub claim_ceiling: Option<String>,
    pub model_ref: Option<String>,
    pub derivation_refs: Vec<String>,
    pub relations: Vec<ProvenanceRelation>,
}

impl FederatedClaim {
    /// Typed projection of the canonical admission subject.
    /// This is intentionally separate from the claim/representation identity.
    pub fn admission_subject(&self) -> Result<CanonicalAdmissionSubject, &'static str> {
        CanonicalAdmissionSubject::new(
            self.canonical_identity.clone(),
            Some(self.provenance_family.clone()),
        )
    }

    /// Typed projection of this exported representation's local identity.
    pub fn representation_identity(&self) -> Result<ClaimRepresentationIdentity, &'static str> {
        ClaimRepresentationIdentity::new(self.claim_identity.clone())
    }

    pub fn new(
        claim_identity: impl Into<String>,
        canonical_identity: impl Into<String>,
        provenance_family: impl Into<String>,
        author: impl Into<String>,
        statement_ref: impl Into<String>,
        provenance_view: ProvenanceView,
        admission_receipt: CanonicalAdmissionReceipt,
    ) -> Result<Self, &'static str> {
        let representation = ClaimRepresentationIdentity::new(claim_identity)?;
        let author = ClaimAuthorIdentity::new(author)?;
        let subject = CanonicalAdmissionSubject::new(
            canonical_identity,
            Some(provenance_family.into()),
        )?;
        let statement = CanonicalStatementIdentity::new(subject.clone(), statement_ref)?;
        let claim = Self {
            schema_version: FEDERATED_CLAIM_SCHEMA_VERSION,
            claim_identity: representation.as_str().to_owned(),
            canonical_identity: subject.canonical_identity().to_owned(),
            provenance_family: subject.provenance_family().expect("claim subject family").to_owned(),
            author: author.clone(),
            authorship: None,
            statement_ref: statement.statement_ref().to_owned(),
            source_event: None,
            frontier_ref: admission_receipt.frontier_ref.clone(),
            provenance_snapshot_digest: provenance_view.snapshot_digest.clone(),
            provenance_validation: provenance_view.validation.clone(),
            admission_receipt,
            epistemic_state: None,
            claim_ceiling: None,
            model_ref: None,
            derivation_refs: Vec::new(),
            relations: provenance_view.relations.clone(),
        };
        if !claim.admission_receipt.binds_validation(&provenance_view.validation) {
            return Err("admission receipt must bind provenance view validation");
        }
        claim.validate_structure()?;
        Ok(claim)
    }

    /// Structural contract only. It does not establish truth, reliability, authorship
    /// cryptography, or scientific validity; those belong to the federation adapter
    /// and/or higher-level epistemic assessment.
    /// Deterministic digest of the complete federated envelope. This is an identity
    /// for the exported representation, not a truth or authority signal.
    pub fn canonical_digest(&self) -> String {
        fn field(bytes: &mut Vec<u8>, label: &str, value: Option<&str>) {
            bytes.extend_from_slice(&(label.len() as u64).to_be_bytes());
            bytes.extend_from_slice(label.as_bytes());
            match value {
                Some(value) => {
                    bytes.push(1);
                    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
                    bytes.extend_from_slice(value.as_bytes());
                }
                None => bytes.push(0),
            }
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"symthaea:federated-claim-digest:v2\0");
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        bytes.extend_from_slice(&FEDERATED_CLAIM_DIGEST_VERSION.to_be_bytes());
        field(&mut bytes, "claim_identity", Some(&self.claim_identity));
        field(&mut bytes, "canonical_identity", Some(&self.canonical_identity));
        field(&mut bytes, "provenance_family", Some(&self.provenance_family));
        field(&mut bytes, "author", Some(self.author.as_str()));
        field(
            &mut bytes,
            "proof_purpose",
            self.authorship
                .as_ref()
                .map(|authorship| authorship.proof_purpose().as_str()),
        );
        field(
            &mut bytes,
            "verification_method",
            self.authorship
                .as_ref()
                .and_then(|authorship| authorship.verification_method()),
        );
        field(&mut bytes, "statement_ref", Some(&self.statement_ref));
        field(&mut bytes, "source_event", self.source_event.as_deref());
        field(&mut bytes, "frontier_ref", self.frontier_ref.as_deref());
        field(&mut bytes, "provenance_snapshot_digest", Some(&self.provenance_snapshot_digest));
        field(&mut bytes, "validator_version", Some(&self.provenance_validation.validator_version));
        bytes.extend_from_slice(&self.provenance_validation.snapshot_schema_version.to_be_bytes());
        bytes.extend_from_slice(&(self.provenance_validation.relation_count as u64).to_be_bytes());
        bytes.push(self.provenance_validation.conforms as u8);

        bytes.extend_from_slice(&(self.provenance_validation.violations.len() as u64).to_be_bytes());
        for violation in &self.provenance_validation.violations {
            field(&mut bytes, "violation_code", Some(&violation.code));
            field(&mut bytes, "violation_source", violation.source_memory_id.as_deref());
            field(&mut bytes, "violation_target", violation.target_memory_id.as_deref());
            field(&mut bytes, "violation_message", Some(&violation.message));
        }

        field(&mut bytes, "admission_event", Some(&self.admission_receipt.admission_event));
        field(&mut bytes, "receipt_frontier_ref", self.admission_receipt.frontier_ref.as_deref());
        field(
            &mut bytes,
            "receipt_admitted_subject_digest",
            Some(&self.admission_receipt.admitted_subject_digest),
        );
        field(
            &mut bytes,
            "receipt_provenance_snapshot_digest",
            Some(&self.admission_receipt.provenance_snapshot_digest),
        );
        field(&mut bytes, "receipt_validator_version", Some(&self.admission_receipt.validator_version));
        bytes.extend_from_slice(&self.admission_receipt.snapshot_schema_version.to_be_bytes());
        field(&mut bytes, "epistemic_state", self.epistemic_state.as_deref());
        field(&mut bytes, "claim_ceiling", self.claim_ceiling.as_deref());
        field(&mut bytes, "model_ref", self.model_ref.as_deref());

        let mut derivation_refs = self.derivation_refs.clone();
        derivation_refs.sort();
        bytes.extend_from_slice(&(derivation_refs.len() as u64).to_be_bytes());
        for reference in derivation_refs {
            field(&mut bytes, "derivation_ref", Some(&reference));
        }

        let mut relations = self.relations.clone();
        relations.sort_by(|a, b| {
            (&a.source_memory_id, &a.target_memory_id, a.kind.stable_code(), &a.created_at)
                .cmp(&(&b.source_memory_id, &b.target_memory_id, b.kind.stable_code(), &b.created_at))
        });
        bytes.extend_from_slice(&(relations.len() as u64).to_be_bytes());
        for relation in relations {
            field(&mut bytes, "relation_source", Some(&relation.source_memory_id));
            field(&mut bytes, "relation_target", Some(&relation.target_memory_id));
            field(&mut bytes, "relation_kind", Some(relation.kind.stable_code()));
            field(&mut bytes, "relation_created_at", Some(&relation.created_at));
        }

        sha256_hex(&bytes)
    }

    pub fn validate_structure(&self) -> Result<(), &'static str> {
        if self.schema_version != FEDERATED_CLAIM_SCHEMA_VERSION {
            return Err("unsupported federated claim schema version");
        }
        for (name, value) in [
            ("claim_identity", self.claim_identity.as_str()),
            ("canonical_identity", self.canonical_identity.as_str()),
            ("provenance_family", self.provenance_family.as_str()),
            ("author", self.author.as_str()),
            ("statement_ref", self.statement_ref.as_str()),
            ("provenance_snapshot_digest", self.provenance_snapshot_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(match name {
                    "claim_identity" => "claim identity must be non-empty",
                    "canonical_identity" => "canonical identity must be non-empty",
                    "provenance_family" => "provenance family must be non-empty",
                    "author" => "author must be non-empty",
                    "statement_ref" => "statement reference must be non-empty",
                    _ => "provenance snapshot digest must be non-empty",
                });
            }
        }

        if self.admission_receipt.admission_event.trim().is_empty() {
            return Err("admission event must be non-empty");
        }
        if self.admission_receipt.frontier_ref.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err("admission receipt frontier reference must be non-empty when present");
        }
        if self.provenance_validation.validator_version.trim().is_empty() {
            return Err("provenance validator version must be non-empty");
        }
        if self.admission_receipt.validator_version.trim().is_empty() {
            return Err("admission receipt validator version must be non-empty");
        }
        if self.source_event.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.frontier_ref.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.epistemic_state.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.claim_ceiling.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.model_ref.as_deref().is_some_and(|v| v.trim().is_empty())
        {
            return Err("federated claim reference must be non-empty when present");
        }
        if !self.provenance_validation.conforms {
            return Err("federated claim provenance validation does not conform");
        }
        if !self.provenance_validation.violations.is_empty() {
            return Err("conforming federated claim cannot contain validation violations");
        }
        if self.provenance_validation.relation_count != self.relations.len() {
            return Err("claim relation count must match validation report");
        }
        if self.provenance_validation.snapshot_digest != self.provenance_snapshot_digest {
            return Err("claim snapshot digest must match validation report");
        }
        if ProvenanceView::from_relations(&self.relations, self.provenance_validation.clone()).is_err() {
            return Err("claim relations must match validation snapshot");
        }
        if !self.admission_receipt.binds_validation(&self.provenance_validation) {
            return Err("admission receipt must bind claim validation report");
        }
        if self.admission_receipt.provenance_snapshot_digest != self.provenance_snapshot_digest {
            return Err("claim snapshot digest must match validation report");
        }
        if !self.admission_receipt.binds_subject(
            &self.canonical_identity,
            Some(&self.provenance_family),
        ) {
            return Err("admission receipt must bind claim canonical subject");
        }
        self.author.validate_structure()?;
        if let Some(authorship) = &self.authorship {
            authorship.validate_structure()?;
            if authorship.author() != &self.author {
                return Err("claim authorship author must match declared claim author");
            }
        }
        if self.admission_receipt.frontier_ref != self.frontier_ref {
            return Err("claim frontier must match admission receipt");
        }
        // Force the proposition through the typed identity constructor at the
        // structural boundary, so a serialized/raw statement reference cannot
        // bypass the subject relationship.
        let statement = self.statement_identity()?;
        let subject = self.admission_subject()?;
        if statement.subject() != &subject {
            return Err("statement identity must bind canonical subject");
        }
        // `derivation_refs` is a set at the wire-contract level: ordering is
        // canonicalized for digesting, while duplicate references are rejected rather
        // than silently changing the representation identity.
        let unique_derivations: std::collections::HashSet<&str> =
            self.derivation_refs.iter().map(String::as_str).collect();
        if unique_derivations.len() != self.derivation_refs.len() {
            return Err("derivation references must be unique");
        }
        if self.derivation_refs.iter().any(|reference| reference.trim().is_empty()) {
            return Err("derivation references must be non-empty");
        }

        let mut seen_relations = std::collections::HashSet::with_capacity(self.relations.len());
        for relation in &self.relations {
            relation.validate()?;
            let relation_key = (
                relation.source_memory_id.as_str(),
                relation.target_memory_id.as_str(),
                relation.kind.stable_code(),
                relation.created_at.as_str(),
            );
            if !seen_relations.insert(relation_key) {
                return Err("provenance relations must be unique");
            }
            if relation.kind == ProvenanceRelationKind::RepresentationOf
                && relation.source_memory_id == relation.target_memory_id
            {
                return Err("representation relation cannot self-reference");
            }
        }

        // A federated snapshot is a directed provenance graph. Reject cycles in
        // semantic lineage edges so a derived/revised/superseding chain cannot
        // claim that an object is ultimately derived from itself. This remains
        // structural: temporal semantics and external event ordering belong to
        // the provenance adapter, not this substrate-neutral contract.
        let mut lineage = std::collections::BTreeMap::<&str, Vec<&str>>::new();
        for relation in &self.relations {
            if matches!(
                relation.kind,
                ProvenanceRelationKind::DerivedFrom
                    | ProvenanceRelationKind::RevisedFrom
                    | ProvenanceRelationKind::Supersedes
            ) {
                lineage
                    .entry(relation.target_memory_id.as_str())
                    .or_default()
                    .push(relation.source_memory_id.as_str());
            }
        }
        fn visit<'a>(
            node: &'a str,
            lineage: &std::collections::BTreeMap<&'a str, Vec<&'a str>>,
            visiting: &mut std::collections::BTreeSet<&'a str>,
            visited: &mut std::collections::BTreeSet<&'a str>,
        ) -> bool {
            if visiting.contains(node) {
                return true;
            }
            if !visited.insert(node) {
                return false;
            }
            visiting.insert(node);
            if let Some(parents) = lineage.get(node) {
                for parent in parents {
                    if visit(parent, lineage, visiting, visited) {
                        return true;
                    }
                }
            }
            visiting.remove(node);
            false
        }
        let mut visiting = std::collections::BTreeSet::new();
        let mut visited = std::collections::BTreeSet::new();
        for node in lineage.keys().copied() {
            if visit(node, &lineage, &mut visiting, &mut visited) {
                return Err("provenance lineage contains a cycle");
            }
        }

        Ok(())
    }

    /// Stable, typed dependency identities an adapter must resolve or otherwise
    /// account for before accepting this envelope. The method deliberately returns
    /// identifiers only; it performs no I/O and makes no assumptions about
    /// Holochain/DHT semantics.
    ///
    /// The admission event and provenance snapshot digest are intentionally not
    /// classified as resolution dependencies here: they are admission/binding
    /// identities. An adapter may authenticate or verify those bindings separately.
    pub fn dependency_bindings(&self) -> Vec<FederationDependency> {
        let mut dependencies = Vec::new();
        if let Some(frontier) = &self.frontier_ref {
            dependencies.push(FederationDependency::Frontier(frontier.clone()));
        }
        if let Some(frontier) = &self.admission_receipt.frontier_ref {
            dependencies.push(FederationDependency::Frontier(frontier.clone()));
        }
        dependencies.extend(
            self.derivation_refs
                .iter()
                .cloned()
                .map(FederationDependency::Derivation),
        );
        dependencies.sort();
        dependencies.dedup();
        dependencies
    }

    /// Compatibility projection for adapters that only need raw identifiers.
    /// Prefer dependency_bindings() when the dependency kind matters.
    pub fn dependency_refs(&self) -> Vec<String> {
        self.dependency_bindings()
            .into_iter()
            .map(|dependency| dependency.reference().to_owned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The canonical representation identity is also the idempotency key for a
    /// substrate adapter: retrying the exact same envelope must address the same
    /// representation, while any digest-bearing mutation creates a different key.
    pub fn replay_key(&self) -> String {
        self.canonical_digest()
    }

    /// Classify an adapter's dependency-resolution result after applying the
    /// substrate-neutral structural contract. Structural failure is definitive;
    /// otherwise non-empty unresolved dependencies remain retryable/accountable.
    pub fn validation_outcome<I>(&self, unresolved: I) -> FederationValidationOutcome
    where
        I: IntoIterator<Item = FederationDependency>,
    {
        if let Err(reason) = self.validate_structure() {
            return FederationValidationOutcome::Invalid(reason.to_owned());
        }

        let mut dependencies: Vec<_> = unresolved.into_iter().collect();
        dependencies.sort();
        dependencies.dedup();

        let declared: std::collections::BTreeSet<_> =
            self.dependency_bindings().into_iter().collect();
        if let Some(undeclared) = dependencies.iter().find(|dependency| !declared.contains(*dependency)) {
            return FederationValidationOutcome::Invalid(format!(
                "unresolved dependency is not declared: {}",
                undeclared.reference()
            ));
        }

        if dependencies.is_empty() {
            FederationValidationOutcome::Valid
        } else {
            FederationValidationOutcome::Unresolved(dependencies)
        }
    }

    /// Attach explicit authorship context to this claim.
    ///
    /// The author in the context must match the claim's declared author. Cryptographic
    /// verification remains outside this substrate-neutral structural type.
    pub fn with_authorship(mut self, authorship: ClaimAuthorship) -> Result<Self, &'static str> {
        authorship.validate_structure()?;
        if authorship.author() != &self.author {
            return Err("claim authorship author must match declared claim author");
        }
        self.authorship = Some(authorship);
        self.validate_structure()?;
        Ok(self)
    }

    pub fn authorship_binding(&self) -> Option<&ClaimAuthorship> {
        self.authorship.as_ref()
    }

    pub fn authorship_binding_is_valid(&self) -> bool {
        self.authorship
            .map(|authorship| {
                authorship.validate_structure().is_ok()
                    && authorship.author() == &self.author
            })
            .unwrap_or(false)
    }

    /// Typed proposition identity: this is distinct from both the representation
    /// identity and the subject-level admission identity.
    pub fn statement_identity(&self) -> Result<CanonicalStatementIdentity, &'static str> {
        CanonicalStatementIdentity::new(self.admission_subject()?, self.statement_ref.clone())
    }

    /// True when the statement reference is coupled to this claim's exact canonical
    /// subject. This proves identity structure only; it does not assert that the
    /// proposition itself was separately admitted.
    pub fn statement_subject_binding_is_valid(&self) -> bool {
        self.statement_identity()
            .map(|statement| {
                self.admission_subject()
                    .map(|subject| statement.subject() == &subject)
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Subject-level admission check. This intentionally does not mean that the
    /// proposition named by statement_ref was itself admitted.
    pub fn is_subject_admission_bound(
        &self,
        validation: &crate::ProvenanceValidationReport,
    ) -> bool {
        self.validate_structure().is_ok()
            && self.statement_subject_binding_is_valid()
            && validation.validate_against_relations(&self.relations).is_ok()
            && self.admission_receipt.binds_validation(validation)
            && self.provenance_snapshot_digest == validation.snapshot_digest
            && self.provenance_validation == *validation
    }

    /// Backwards-compatible alias for the historical subject-level check.
    ///
    /// Callers that need proposition semantics must use statement_identity() and
    /// separately obtain a statement-level admission binding; the subject receipt
    /// alone is insufficient.
    pub fn is_admission_bound(
        &self,
        validation: &crate::ProvenanceValidationReport,
    ) -> bool {
        self.is_subject_admission_bound(validation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProvenanceRelation, ProvenanceRelationKind, ProvenanceValidationReport};

    fn receipt() -> (CanonicalAdmissionReceipt, ProvenanceValidationReport) {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(&[relation]);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        (receipt, validation)
    }

    #[test]
    fn federated_claim_exposes_distinct_typed_identity_projections() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        ).unwrap();

        assert_eq!(claim.admission_subject().unwrap().canonical_identity(), "canonical:1");
        assert_eq!(claim.admission_subject().unwrap().provenance_family(), Some("family:1"));
        assert_eq!(claim.representation_identity().unwrap().as_str(), "claim:1");
        assert_ne!(
            claim.admission_subject().unwrap().digest(),
            sha256_hex(claim.representation_identity().unwrap().as_str().as_bytes())
        );
    }

    #[test]
    fn mutating_statement_reference_changes_proposition_identity_not_subject_admission() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        )
        .unwrap();

        let original_statement = claim.statement_identity().unwrap();
        claim.statement_ref = "statement:2".into();
        let changed_statement = claim.statement_identity().unwrap();

        assert_ne!(original_statement.digest(), changed_statement.digest());
        assert_eq!(changed_statement.subject(), &claim.admission_subject().unwrap());
        assert!(claim.validate_structure().is_ok());
        // The existing receipt remains a subject-level admission, not admission
        // of whichever proposition happens to be placed in statement_ref.
        assert!(claim.is_subject_admission_bound(&claim.provenance_validation));
    }

    #[test]
    fn typed_authorship_binds_author_purpose_and_optional_verification_method() {
        let author = ClaimAuthorIdentity::new("author:1").unwrap();
        let purpose = ClaimProofPurpose::new("assertionMethod").unwrap();
        let authorship =
            ClaimAuthorship::new(author.clone(), purpose, Some("https://example.test/key/1".into()))
                .unwrap();

        assert_eq!(authorship.author(), &author);
        assert_eq!(authorship.proof_purpose().as_str(), "assertionMethod");
        assert_eq!(
            authorship.verification_method(),
            Some("https://example.test/key/1")
        );
        assert!(!authorship.digest().is_empty());
        assert!(authorship.validate_structure().is_ok());
    }

    #[test]
    fn authorship_cannot_be_attached_to_a_different_declared_author() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        )
        .unwrap();

        let other = ClaimAuthorIdentity::new("author:2").unwrap();
        let authorship = ClaimAuthorship::new(
            other,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(
            claim.with_authorship(authorship).unwrap_err(),
            "claim authorship author must match declared claim author"
        );
    }

    #[test]
    fn authorship_changes_representation_digest_but_not_subject_identity() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let base = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        )
        .unwrap();

        let authorship = ClaimAuthorship::new(
            base.author.clone(),
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            Some("https://example.test/key/1".into()),
        )
        .unwrap();
        let with_authorship = base.clone().with_authorship(authorship).unwrap();

        assert_ne!(base.canonical_digest(), with_authorship.canonical_digest());
        assert_eq!(
            base.admission_subject().unwrap(),
            with_authorship.admission_subject().unwrap()
        );
        assert!(with_authorship.authorship_binding_is_valid());
    }

    #[test]
    fn claim_authorship_rejects_blank_verification_method() {
        let author = ClaimAuthorIdentity::new("author:1").unwrap();
        let purpose = ClaimProofPurpose::new("assertionMethod").unwrap();
        assert_eq!(
            ClaimAuthorship::new(author, purpose, Some("   ".into())).unwrap_err(),
            "claim verification method must be non-empty when present"
        );
    }

    #[test]
    fn federated_claim_statement_identity_binds_exact_subject_and_proposition() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        )
        .unwrap();

        let statement = claim.statement_identity().unwrap();
        assert_eq!(statement.subject(), &claim.admission_subject().unwrap());
        assert_eq!(statement.statement_ref(), "statement:1");
        assert!(claim.statement_subject_binding_is_valid());
    }

    #[test]
    fn alternate_statement_is_not_the_same_typed_identity_even_for_same_subject() {
        let (receipt, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view =
            ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            view,
            receipt,
        )
        .unwrap();

        let a = claim.statement_identity().unwrap();
        let b = CanonicalStatementIdentity::new(
            claim.admission_subject().unwrap(),
            "statement:2",
        )
        .unwrap();

        assert_ne!(a.digest(), b.digest());
        assert!(claim.validate_structure().is_ok());
        // The receipt proves subject-level admission only; statement identity is a
        // separate namespace and must not be inferred as separately admitted.
        assert!(claim.is_subject_admission_bound(&claim.provenance_validation));
    }

    #[test]
    fn federated_claim_preserves_admission_boundary() {
        let (r, validation) = receipt();
        let claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&[ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            }], validation).unwrap(),
            r,
        ).unwrap();
        assert_eq!(claim.schema_version, FEDERATED_CLAIM_SCHEMA_VERSION);
        assert_eq!(claim.frontier_ref.as_deref(), Some("frontier:1"));
        assert!(claim.validate_structure().is_ok());
        assert!(!claim.canonical_digest().is_empty());
    }

    #[test]
    fn federated_claim_rejects_snapshot_mismatch() {
        let (r, _valid) = receipt();
        let validation = ProvenanceValidationReport::from_relations(&[]);
        assert_eq!(
            FederatedClaim::new(
                "claim:1", "canonical:1", "family:1", "author:1", "statement:1",
                ProvenanceView::from_relations(&[], validation).unwrap(), r,
            ).unwrap_err(),
            "claim snapshot digest must match validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_lineage_cycle() {
        let relations = vec![
            ProvenanceRelation {
                source_memory_id: "b".into(),
                target_memory_id: "a".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            },
            ProvenanceRelation {
                source_memory_id: "a".into(),
                target_memory_id: "b".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:3".into(),
            },
        ];
        let validation = ProvenanceValidationReport::from_relations(&relations);
        let view = ProvenanceView::from_relations(&relations, validation).unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-cycle",
            Some("frontier:cycle".into()),
            "canonical:cycle",
            Some("family:cycle".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        )
        .unwrap();
        let claim = FederatedClaim {
            schema_version: FEDERATED_CLAIM_SCHEMA_VERSION,
            claim_identity: "claim:cycle".into(),
            canonical_identity: "canonical:cycle".into(),
            provenance_family: "family:cycle".into(),
            author: ClaimAuthorIdentity::new("author:1").unwrap(),
            authorship: None,
            statement_ref: "statement:cycle".into(),
            source_event: None,
            frontier_ref: Some("frontier:cycle".into()),
            provenance_snapshot_digest: view.snapshot_digest.clone(),
            provenance_validation: view.validation.clone(),
            admission_receipt: receipt,
            epistemic_state: None,
            claim_ceiling: None,
            model_ref: None,
            derivation_refs: Vec::new(),
            relations,
        };
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "provenance lineage contains a cycle"
        );
    }

    #[test]
    fn federated_claim_rejects_duplicate_provenance_relation() {
        let (r, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&[relation.clone()], validation).unwrap(),
            r,
        )
        .unwrap();
        claim.relations.push(relation);
        claim.provenance_validation.relation_count = 2;
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "claim snapshot digest must match validation report"
        );

        // Rebuild the validation snapshot around the duplicated relation. The
        // read-only view correctly rejects duplicates before a claim can cross it,
        // so the claim-level regression constructs the deserialized envelope directly.
        let duplicate_relations = vec![
            ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            },
            ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            },
        ];
        let duplicate_validation = ProvenanceValidationReport::from_relations(&duplicate_relations);
        let duplicate_receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:duplicate",
            Some("family:1".into()),
            duplicate_validation.snapshot_digest.clone(),
            duplicate_validation.validator_version.clone(),
            duplicate_validation.snapshot_schema_version,
        )
        .unwrap();
        let duplicate_claim = FederatedClaim {
            schema_version: FEDERATED_CLAIM_SCHEMA_VERSION,
            claim_identity: "claim:duplicate".into(),
            canonical_identity: "canonical:duplicate".into(),
            provenance_family: "family:1".into(),
            author: ClaimAuthorIdentity::new("author:1").unwrap(),
            authorship: None,
            statement_ref: "statement:1".into(),
            source_event: None,
            frontier_ref: Some("frontier:1".into()),
            provenance_snapshot_digest: duplicate_validation.snapshot_digest.clone(),
            provenance_validation: duplicate_validation,
            admission_receipt: duplicate_receipt,
            epistemic_state: None,
            claim_ceiling: None,
            model_ref: None,
            derivation_refs: Vec::new(),
            relations: duplicate_relations,
        };
        assert_eq!(
            duplicate_claim.validate_structure().unwrap_err(),
            "claim relations must match validation snapshot"
        );
    }

    #[test]
    fn federated_claim_rejects_provenance_family_subject_mismatch() {
        let (_, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:other".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        ).unwrap();

        assert_eq!(
            FederatedClaim::new(
                "claim:1",
                "canonical:1",
                "family:1",
                "author:1",
                "statement:1",
                view,
                receipt,
            ).unwrap_err(),
            "admission receipt must bind claim canonical subject"
        );
    }

    #[test]
    fn federated_claim_rejects_admission_subject_mismatch() {
        let (_, validation) = receipt();
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let view = ProvenanceView::from_relations(std::slice::from_ref(&relation), validation).unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:other",
            Some("family:1".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        ).unwrap();
        assert_eq!(
            FederatedClaim::new(
                "claim:1",
                "canonical:1",
                "family:1",
                "author:1",
                "statement:1",
                view,
                receipt,
            ).unwrap_err(),
            "admission receipt must bind claim canonical subject"
        );
    }

    #[test]
    fn federated_claim_rejects_frontier_mismatch() {
        let (r, validation) = receipt();
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&[ProvenanceRelation {
                source_memory_id: "derived".into(),
                target_memory_id: "source".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:2".into(),
            }], validation).unwrap(),
            r,
        ).unwrap();
        claim.frontier_ref = Some("frontier:other".into());
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "claim frontier must match admission receipt"
        );
    }
}


#[cfg(test)]
mod digest_tests {
    use super::*;

    pub(super) fn base_claim() -> FederatedClaim {
        let relation_a = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let relation_b = ProvenanceRelation {
            source_memory_id: "claim".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:3".into(),
        };
        let relations = vec![relation_b, relation_a];
        let validation = ProvenanceValidationReport::from_relations(&relations);
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-1",
            Some("frontier:1".into()),
            "canonical:1",
            Some("family:1".into()),
            validation.snapshot_digest.clone(),
            validation.validator_version.clone(),
            validation.snapshot_schema_version,
        ).unwrap();
        let mut claim = FederatedClaim::new(
            "claim:1",
            "canonical:1",
            "family:1",
            "author:1",
            "statement:1",
            ProvenanceView::from_relations(&relations, validation).unwrap(),
            receipt,
        ).unwrap();
        claim.derivation_refs = vec!["derivation:b".into(), "derivation:a".into()];
        claim.relations = relations;
        claim
    }

    #[test]
    fn canonical_digest_has_a_stable_golden_vector() {
        let claim = base_claim();
        assert_eq!(
            claim.canonical_digest(),
            "99ce0d23b4db8dba9fc0c31d7f7d47a01dbaff803c024c278691897685a7b124"
        );
    }

    #[test]
    fn canonical_digest_is_order_independent_for_sets() {
        let a = base_claim();
        let mut b = a.clone();
        b.derivation_refs.reverse();
        b.relations.reverse();
        assert_eq!(a.canonical_digest(), b.canonical_digest());
        assert!(a.validate_structure().is_ok());
    }

    #[test]
    fn canonical_digest_covers_validation_violations() {
        let mut a = base_claim();
        a.provenance_validation.conforms = false;
        a.provenance_validation.violations.push(ProvenanceValidationViolation {
            code: "synthetic".into(),
            source_memory_id: Some("source".into()),
            target_memory_id: Some("target".into()),
            message: "first".into(),
        });
        let mut b = a.clone();
        b.provenance_validation.violations[0].message = "second".into();
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn canonical_digest_changes_when_claim_content_changes() {
        let a = base_claim();
        let mut b = a.clone();
        b.author = ClaimAuthorIdentity::new("author:2").unwrap();
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn canonical_digest_covers_admitted_subject_binding() {
        let a = base_claim();
        let mut b = a.clone();
        b.admission_receipt.admitted_subject_digest = "0".repeat(64);
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn canonical_digest_covers_receipt_frontier_and_snapshot() {
        let a = base_claim();
        let mut frontier = a.clone();
        frontier.admission_receipt.frontier_ref = Some("frontier:2".into());
        assert_ne!(a.canonical_digest(), frontier.canonical_digest());

        let mut snapshot = a.clone();
        snapshot.admission_receipt.provenance_snapshot_digest = "snapshot:tampered".into();
        assert_ne!(a.canonical_digest(), snapshot.canonical_digest());
    }

    #[test]
    fn federated_claim_json_round_trip_preserves_digest_and_admission_binding() {
        let claim = base_claim();
        let digest = claim.canonical_digest();
        let encoded = serde_json::to_string(&claim).expect("claim should serialize");
        let decoded: FederatedClaim =
            serde_json::from_str(&encoded).expect("claim should deserialize");
        assert_eq!(decoded, claim);
        assert_eq!(decoded.canonical_digest(), digest);
        assert!(decoded.validate_structure().is_ok());
        assert!(decoded.statement_subject_binding_is_valid());
        assert!(decoded.is_subject_admission_bound(&decoded.provenance_validation));
        assert!(decoded.is_admission_bound(&decoded.provenance_validation));
    }

    #[test]
    fn federated_claim_digest_covers_admission_event_but_structure_does_not_claim_authenticity() {
        let mut claim = base_claim();
        let original = claim.canonical_digest();
        claim.admission_receipt.admission_event = "admission:event-tampered".into();
        assert_ne!(claim.canonical_digest(), original);
        assert!(claim.validate_structure().is_ok());
    }

    #[test]
    fn federated_claim_rejects_empty_receipt_frontier() {
        let mut claim = base_claim();
        claim.admission_receipt.frontier_ref = Some("".into());
        claim.frontier_ref = Some("".into());
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt frontier reference must be non-empty when present"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_validation_version() {
        let mut claim = base_claim();
        claim.provenance_validation.validator_version = "tampered-validator".into();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt must bind claim validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_schema_version() {
        let mut claim = base_claim();
        claim.provenance_validation.snapshot_schema_version += 1;
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt must bind claim validation report"
        );
    }

    #[test]
    fn federated_claim_rejects_mutated_relation_content() {
        let mut claim = base_claim();
        claim.relations[0].created_at = "cycle:999".into();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "claim snapshot digest must match validation report"
        );
    }

    #[test]
    fn canonical_digest_changes_when_admission_event_changes() {
        let a = base_claim();
        let mut b = a.clone();
        b.admission_receipt.admission_event = "admission:event-2".into();
        assert_ne!(a.canonical_digest(), b.canonical_digest());
    }

    #[test]
    fn validation_rejects_relation_snapshot_drift() {
        let a = base_claim();
        let mut b = a.clone();
        b.relations.clear();
        assert_eq!(
            b.validate_structure().unwrap_err(),
            "claim relation count must match validation report"
        );
    }

    #[test]
    fn validation_is_defensive_against_deserialized_blank_versions_and_references() {
        let mut claim = base_claim();

        claim.provenance_validation.validator_version.clear();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "provenance validator version must be non-empty"
        );

        claim = base_claim();
        claim.admission_receipt.validator_version.clear();
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "admission receipt validator version must be non-empty"
        );

        let mutations: &[fn(&mut FederatedClaim)] = &[
            |c| c.source_event = Some("   ".into()),
            |c| c.frontier_ref = Some("   ".into()),
            |c| c.epistemic_state = Some("   ".into()),
            |c| c.claim_ceiling = Some("   ".into()),
            |c| c.model_ref = Some("   ".into()),
        ];
        for mutation in mutations {
            claim = base_claim();
            mutation(&mut claim);
            assert_eq!(
                claim.validate_structure().unwrap_err(),
                "federated claim reference must be non-empty when present"
            );
        }
    }
}

#[cfg(test)]
mod adversarial_contract_tests {
    use super::*;

    fn claim() -> FederatedClaim {
        super::digest_tests::base_claim()
    }

    #[test]
    fn digest_changes_for_every_digest_bearing_scalar_field() {
        let cases: &[(&str, fn(&mut FederatedClaim))] = &[
            ("schema_version", |c| c.schema_version = 2),
            ("claim_identity", |c| c.claim_identity.push_str(":changed")),
            ("canonical_identity", |c| c.canonical_identity.push_str(":changed")),
            ("provenance_family", |c| c.provenance_family.push_str(":changed")),
            ("author", |c| c.author = ClaimAuthorIdentity::new("author:changed").unwrap()),
            ("statement_ref", |c| c.statement_ref.push_str(":changed")),
            ("source_event", |c| c.source_event = Some("event:changed".into())),
            ("frontier_ref", |c| c.frontier_ref = Some("frontier:changed".into())),
            ("provenance_snapshot_digest", |c| c.provenance_snapshot_digest.push('x')),
            ("validator_version", |c| c.provenance_validation.validator_version.push('x')),
            ("snapshot_schema_version", |c| c.provenance_validation.snapshot_schema_version += 1),
            ("relation_count", |c| c.provenance_validation.relation_count += 1),
            ("conforms", |c| c.provenance_validation.conforms = false),
            ("admission_event", |c| c.admission_receipt.admission_event.push_str(":changed")),
            ("receipt_frontier_ref", |c| c.admission_receipt.frontier_ref = Some("frontier:changed".into())),
            ("receipt_provenance_snapshot_digest", |c| c.admission_receipt.provenance_snapshot_digest.push('x')),
            ("receipt_validator_version", |c| c.admission_receipt.validator_version.push('x')),
            ("receipt_snapshot_schema_version", |c| c.admission_receipt.snapshot_schema_version += 1),
            ("epistemic_state", |c| c.epistemic_state = Some("Observed".into())),
            ("claim_ceiling", |c| c.claim_ceiling = Some("source-scoped".into())),
            ("model_ref", |c| c.model_ref = Some("model:changed".into())),
        ];

        for (name, mutate) in cases {
            let mut changed = claim();
            let original = changed.canonical_digest();
            mutate(&mut changed);
            assert_ne!(
                changed.canonical_digest(),
                original,
                "digest must cover {name}"
            );
        }
    }

    #[test]
    fn digest_changes_for_derivation_membership_and_relation_membership() {
        let base = claim();
        let original = base.canonical_digest();

        let mut derivation = base.clone();
        derivation.derivation_refs.push("derivation:c".into());
        assert_ne!(derivation.canonical_digest(), original);

        let mut relation = base.clone();
        relation.relations[0].created_at = "cycle:99".into();
        assert_ne!(relation.canonical_digest(), original);
    }

    #[test]
    fn derivation_refs_have_set_semantics_and_reject_duplicates() {
        let mut reordered = claim();
        reordered.derivation_refs.reverse();
        assert_eq!(reordered.canonical_digest(), claim().canonical_digest());
        assert!(reordered.validate_structure().is_ok());

        let mut duplicate = claim();
        duplicate.derivation_refs.push(duplicate.derivation_refs[0].clone());
        assert_eq!(
            duplicate.validate_structure().unwrap_err(),
            "derivation references must be unique"
        );
    }

    #[test]
    fn derivation_refs_reject_empty_members() {
        let mut claim = claim();
        claim.derivation_refs.push("   ".into());
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "derivation references must be non-empty"
        );
    }

    #[test]
    fn future_federated_schema_version_is_rejected_explicitly() {
        let mut claim = claim();
        claim.schema_version = FEDERATED_CLAIM_SCHEMA_VERSION + 1;
        assert_eq!(
            claim.validate_structure().unwrap_err(),
            "unsupported federated claim schema version"
        );
    }

    #[test]
    fn mutation_matrix_distinguishes_invalid_from_representation_only_changes() {
        // These mutations alter the canonical representation, but only some remain
        // structurally valid. The substrate-neutral layer must reject malformed state
        // rather than silently accepting a different envelope.
        let valid: &[(&str, fn(&mut FederatedClaim))] = &[
            ("author", |c| c.author.push_str(":changed")),
            ("source_event", |c| c.source_event = Some("event:changed".into())),
            ("epistemic_state", |c| c.epistemic_state = Some("Observed".into())),
            ("claim_ceiling", |c| c.claim_ceiling = Some("source-scoped".into())),
            ("model_ref", |c| c.model_ref = Some("model:changed".into())),
        ];
        for (name, mutate) in valid {
            let mut changed = claim();
            mutate(&mut changed);
            assert!(
                changed.validate_structure().is_ok(),
                "{name} should remain structurally valid"
            );
        }

        let invalid: &[(&str, fn(&mut FederatedClaim))] = &[
            ("schema_version", |c| c.schema_version += 1),
            ("relation_count", |c| c.provenance_validation.relation_count += 1),
            ("conforms", |c| c.provenance_validation.conforms = false),
            ("snapshot", |c| c.provenance_snapshot_digest.push('x')),
            ("receipt_snapshot", |c| c.admission_receipt.provenance_snapshot_digest.push('x')),
        ];
        for (name, mutate) in invalid {
            let mut changed = claim();
            mutate(&mut changed);
            assert!(
                changed.validate_structure().is_err(),
                "{name} should be structurally rejected"
            );
        }
    }

    #[test]
    fn dependency_bindings_preserve_kind_and_are_sorted_deduplicated() {
        let mut claim = claim();
        claim.derivation_refs.push("frontier:1".into());
        claim.derivation_refs.push("derivation:z".into());
        claim.derivation_refs.push("derivation:a".into());
        assert_eq!(
            claim.dependency_bindings(),
            vec![
                FederationDependency::Derivation("derivation:a".into()),
                FederationDependency::Derivation("derivation:b".into()),
                FederationDependency::Derivation("derivation:z".into()),
                FederationDependency::Derivation("frontier:1".into()),
                FederationDependency::Frontier("frontier:1".into()),
            ]
        );
    }

    #[test]
    fn dependency_refs_are_sorted_deduplicated_and_substrate_neutral() {
        let mut claim = claim();
        claim.derivation_refs.push("frontier:1".into());
        claim.derivation_refs.push("derivation:z".into());
        claim.derivation_refs.push("derivation:a".into());
        assert_eq!(
            claim.dependency_refs(),
            vec![
                "derivation:a".to_string(),
                "derivation:b".to_string(),
                "derivation:z".to_string(),
                "frontier:1".to_string(),
            ]
        );
    }

    #[test]
    fn validation_outcome_rejects_undeclared_unresolved_dependencies() {
        let claim = claim();
        assert_eq!(
            claim.validation_outcome([FederationDependency::Frontier("frontier:missing".into())]),
            FederationValidationOutcome::Invalid(
                "unresolved dependency is not declared: frontier:missing".into()
            )
        );
    }

    #[test]
    fn receipt_and_frontier_mutations_change_replay_identity() {
        let original = claim();

        let mut changed_receipt = original.clone();
        changed_receipt.admission_receipt.admission_event = "admission:event-2".into();
        assert_ne!(original.replay_key(), changed_receipt.replay_key());

        let mut changed_frontier = original.clone();
        changed_frontier.frontier_ref = Some("frontier:2".into());
        changed_frontier.admission_receipt.frontier_ref = Some("frontier:2".into());
        assert_ne!(original.replay_key(), changed_frontier.replay_key());
    }

    #[test]
    fn validation_outcome_has_neutral_valid_invalid_unresolved_states() {
        let claim = claim();
        assert_eq!(
            claim.validation_outcome(std::iter::empty()),
            FederationValidationOutcome::Valid
        );

        assert_eq!(
            claim.validation_outcome([FederationDependency::Frontier("frontier:1".into())]),
            FederationValidationOutcome::Unresolved(vec![
                FederationDependency::Frontier("frontier:1".into())
            ])
        );

        let mut malformed = claim;
        malformed.schema_version += 1;
        assert_eq!(
            malformed.validation_outcome(std::iter::empty()),
            FederationValidationOutcome::Invalid("unsupported federated claim schema version".into())
        );
    }

    #[test]
    fn replay_key_is_exact_representation_identity() {
        let a = claim();
        let mut reordered = a.clone();
        reordered.derivation_refs.reverse();
        reordered.relations.reverse();
        assert_eq!(a.replay_key(), reordered.replay_key());

        let mut changed = a.clone();
        changed.claim_identity.push_str(":retry-different");
        assert_ne!(a.replay_key(), changed.replay_key());
    }

    #[test]
    fn admission_binding_is_fail_closed_for_malformed_claims() {
        let mut claim = claim();
        claim.statement_ref = "   ".into();
        let validation = claim.provenance_validation.clone();
        assert!(!claim.is_admission_bound(&validation));
    }

    #[test]
    fn structural_layer_does_not_claim_authorship_authenticity() {
        let mut claim = claim();
        claim.admission_receipt.admission_event = "admission:event-forged".into();
        // The representation changes, but the substrate-neutral structural contract
        // cannot determine whether the admission event was genuinely authored.
        assert!(claim.validate_structure().is_ok());
        assert_ne!(claim.canonical_digest(), claim().canonical_digest());
    }
}

