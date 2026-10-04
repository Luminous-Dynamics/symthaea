
    /// Verify that a current report is bound to this exact attestation envelope payload.
    ///
    /// The report's payload fingerprint commits to the complete signed payload, including
    /// attester identity, verification method, proof purpose, validity interval, domain,
    /// and challenge. This helper rejects internally inconsistent report identity bindings
    /// and malformed envelope structure before checking report↔envelope identity. Its structural
    /// envelope check does not verify the detached proof or the report's full execution trace,
    /// so callers must still apply the complete verification contract.
    pub fn matches_attestation_envelope(&self, envelope: &ReceiptAttestationEnvelope) -> bool {
        self.has_consistent_identity_bindings()
            && envelope.validate().is_ok()
            && self.verifier_version == VERIFIER_VERSION
            && self.attestation_payload_fingerprint == envelope.payload_fingerprint()
            && self.receipt_fingerprint == envelope.receipt_fingerprint
    }
