#![no_main]

use risc0_zkvm::guest::env;
use sha2::{Digest, Sha256};
use symthaea_tlsn_risc0_probe_core::{
    AuthenticatedRangeV1, PresentationVerifyInputV1, PresentationVerifyJournalV1,
    MAX_AUTHENTICATED_RANGES_V1, MAX_PRESENTATION_BYTES_V1, MAX_TRUST_ROOTS_V1,
    MAX_TRUST_ROOT_DER_BYTES_V1, MAX_TRUST_ROOT_PROFILE_BYTES_V1,
    NOTARY_KEY_K256_V1, NOTARY_KEY_P256_V1, PROBE_CONTRACT_VERSION_V1,
    TLS_VERSION_1_2_V1, TLS_VERSION_1_3_V1,
};
use tlsn_attestation::{
    CryptoProvider,
    presentation::Presentation,
    signing::{KeyAlgId, SignatureVerifierProvider, SignerProvider, VerifyingKey},
};
use tlsn_core::{
    hash::HashProvider,
    webpki::{CertificateDer, RootCertStore, ServerCertVerifier},
};

risc0_zkvm::guest::entry!(main);

const TRUST_ROOT_DOMAIN: &[u8] = b"SYMTHAEA-TLSN-RISC0-ROOT-PROFILE-V1";
const NOTARY_KEY_DOMAIN: &[u8] = b"SYMTHAEA-TLSN-RISC0-NOTARY-KEY-V1";
const ATTESTATION_HEADER_DOMAIN: &[u8] = b"SYMTHAEA-TLSN-RISC0-HEADER-V1";

fn main() {
    let input: PresentationVerifyInputV1 = env::read();

    assert_eq!(input.contract_version, PROBE_CONTRACT_VERSION_V1);
    assert!(!input.serialized_presentation.is_empty());
    assert!(input.serialized_presentation.len() <= MAX_PRESENTATION_BYTES_V1);
    assert!(!input.expected_server_dns_name.is_empty());
    assert!(input.expected_server_dns_name.len() <= 253);
    assert!(input.expected_server_dns_name.is_ascii());
    assert!(
        input.expected_tls_version_code == TLS_VERSION_1_2_V1
            || input.expected_tls_version_code == TLS_VERSION_1_3_V1
    );
    assert!(!input.trust_root_der.is_empty());
    assert!(input.trust_root_der.len() <= MAX_TRUST_ROOTS_V1);

    let trust_root_bytes = input.trust_root_der.iter().fold(0usize, |total, root| {
        assert!(!root.is_empty());
        assert!(root.len() <= MAX_TRUST_ROOT_DER_BYTES_V1);
        total.checked_add(root.len()).expect("root byte count overflow")
    });
    assert!(trust_root_bytes <= MAX_TRUST_ROOT_PROFILE_BYTES_V1);

    let trust_root_profile_digest = digest_trust_roots(&input.trust_root_der);
    assert_eq!(trust_root_profile_digest, input.expected_trust_root_profile_digest);

    let roots = RootCertStore {
        roots: input
            .trust_root_der
            .iter()
            .cloned()
            .map(CertificateDer)
            .collect(),
    };
    let cert_verifier = ServerCertVerifier::new(&roots)
        .expect("verifier trust-root profile contains an invalid DER root");

    // Presentation verification does not need an attestation signer. The
    // signature verifier and hash provider are used to verify the proof.
    let crypto_provider = CryptoProvider {
        hash: HashProvider::default(),
        cert: cert_verifier,
        signer: SignerProvider::default(),
        signature: SignatureVerifierProvider::default(),
    };

    let expected_notary_key = VerifyingKey {
        alg: match input.expected_notary_key_algorithm_id {
            NOTARY_KEY_K256_V1 => KeyAlgId::K256,
            NOTARY_KEY_P256_V1 => KeyAlgId::P256,
            _ => panic!("unsupported expected notary key algorithm"),
        },
        data: input.expected_notary_key_bytes.clone(),
    };
    assert!(!expected_notary_key.data.is_empty());
    assert!(expected_notary_key.data.len() <= 128);

    let presentation: Presentation = bincode::deserialize(&input.serialized_presentation)
        .expect("malformed TLSNotary Presentation serialization");
    assert_eq!(presentation.verifying_key(), &expected_notary_key);

    // Verifies the notary signature/body root, server-identity proof, and
    // transcript proof using the pinned Beta.1 verifier implementation.
    let verified = presentation
        .verify(&crypto_provider)
        .expect("TLSNotary Beta.1 Presentation verification failed");

    assert_eq!(verified.attestation.body.verifying_key(), &expected_notary_key);

    let server_name = verified
        .server_name
        .as_ref()
        .expect("presentation omitted its server identity proof")
        .to_string();
    assert_eq!(server_name, input.expected_server_dns_name);

    let tls_version_code = match verified.connection_info.version {
        tlsn_core::connection::TlsVersion::V1_2 => TLS_VERSION_1_2_V1,
        tlsn_core::connection::TlsVersion::V1_3 => TLS_VERSION_1_3_V1,
    };
    assert_eq!(tls_version_code, input.expected_tls_version_code);
    assert_eq!(
        verified.connection_info.transcript_length.received,
        input.expected_received_transcript_bytes
    );

    let partial_transcript = verified
        .transcript
        .as_ref()
        .expect("presentation omitted its transcript proof");
    assert_eq!(
        partial_transcript.len_received(),
        input.expected_received_transcript_bytes as usize
    );

    let mut authenticated_received_ranges = Vec::new();
    for range in partial_transcript.received_authed().iter() {
        assert!(range.start < range.end);
        assert!(range.end <= input.expected_received_transcript_bytes as usize);
        assert!(authenticated_received_ranges.len() < MAX_AUTHENTICATED_RANGES_V1);
        authenticated_received_ranges.push(AuthenticatedRangeV1 {
            start: u32::try_from(range.start).expect("range start exceeds u32"),
            end: u32::try_from(range.end).expect("range end exceeds u32"),
        });
    }

    let header_bytes = bincode::serialize(&verified.attestation.header)
        .expect("attestation header serialization failed");
    let attestation_header_digest = digest_domain(ATTESTATION_HEADER_DOMAIN, &header_bytes);
    let notary_key_fingerprint = digest_notary_key(
        input.expected_notary_key_algorithm_id,
        &expected_notary_key.data,
    );

    let journal = PresentationVerifyJournalV1 {
        contract_version: PROBE_CONTRACT_VERSION_V1,
        attestation_id: verified.attestation.header.id.0,
        attestation_header_digest,
        notary_key_algorithm_id: input.expected_notary_key_algorithm_id,
        notary_key_fingerprint,
        server_dns_name: server_name,
        tls_version_code,
        connection_started_unix_seconds: verified.connection_info.time,
        sent_transcript_bytes: verified.connection_info.transcript_length.sent,
        received_transcript_bytes: verified.connection_info.transcript_length.received,
        trust_root_profile_digest,
        authenticated_received_ranges,
    };
    env::commit(&journal);
}

fn digest_trust_roots(roots: &[Vec<u8>]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(TRUST_ROOT_DOMAIN);
    hasher.update((roots.len() as u32).to_be_bytes());
    for root in roots {
        hasher.update((root.len() as u32).to_be_bytes());
        hasher.update(root);
    }
    hasher.finalize().into()
}

fn digest_notary_key(algorithm_id: u8, key_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(NOTARY_KEY_DOMAIN);
    hasher.update([algorithm_id]);
    hasher.update((key_bytes.len() as u32).to_be_bytes());
    hasher.update(key_bytes);
    hasher.finalize().into()
}

fn digest_domain(domain: &[u8], message: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((message.len() as u64).to_be_bytes());
    hasher.update(message);
    hasher.finalize().into()
}
