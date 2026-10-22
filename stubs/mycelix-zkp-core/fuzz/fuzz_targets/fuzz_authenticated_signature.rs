#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zkp_core::{
    dilithium::{verify_authenticated_signature, PUBLIC_KEY_SIZE, SIGNATURE_SIZE},
    domain::DomainTag,
    types::{AuthenticatedProof, BackendId, ProofMetadata},
};
use sha2::{Digest, Sha256};

const MAX_INPUT_BYTES: usize = 8 * 1024;
const MAX_MESSAGE_SUFFIX_BYTES: usize = 1024;

fn expand_to_len(input: &[u8], len: usize, offset: usize) -> Vec<u8> {
    let mut seed_hasher = Sha256::new();
    seed_hasher.update(b"SYMTHAEA-CRYPTO-FUZZ/AUTHENTICATED-ENVELOPE/v1");
    seed_hasher.update((input.len() as u64).to_le_bytes());
    seed_hasher.update(input);
    seed_hasher.update((offset as u64).to_le_bytes());
    let seed = seed_hasher.finalize();

    let mut output = Vec::with_capacity(len);
    let mut counter = 0_u64;
    while output.len() < len {
        let mut block_hasher = Sha256::new();
        block_hasher.update(b"SYMTHAEA-CRYPTO-FUZZ/BLOCK/v1");
        block_hasher.update(seed);
        block_hasher.update(counter.to_le_bytes());
        let block = block_hasher.finalize();
        let count = (len - output.len()).min(block.len());
        output.extend_from_slice(&block[..count]);
        counter = counter.wrapping_add(1);
    }
    output
}

fuzz_target!(|input: &[u8]| {
    let input = &input[..input.len().min(MAX_INPUT_BYTES)];
    let public_key = expand_to_len(input, PUBLIC_KEY_SIZE, 0);
    let signature_suffix = input.len().min(MAX_MESSAGE_SUFFIX_BYTES);
    let signature =
        expand_to_len(input, SIGNATURE_SIZE + signature_suffix, 1);

    // Match client_id to the exact public-key bytes so verification proceeds
    // to the public-key/signature parsers rather than stopping at the identity
    // guard. This fuzz target tests signature authentication only; it does not
    // claim to verify the underlying zero-knowledge proof.
    let client_id_digest = Sha256::digest(&public_key);
    let mut client_id = [0u8; 32];
    client_id.copy_from_slice(&client_id_digest);

    let nonce_bytes = expand_to_len(input, 32, 2);
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&nonce_bytes);

    let public_inputs_digest = Sha256::digest(input);
    let mut public_inputs_hash = [0u8; 32];
    public_inputs_hash.copy_from_slice(&public_inputs_digest);

    let timestamp_bytes = expand_to_len(input, 8, 3);
    let mut timestamp_array = [0u8; 8];
    timestamp_array.copy_from_slice(&timestamp_bytes);

    let energy_bytes = expand_to_len(input, 8, 4);
    let mut energy_array = [0u8; 8];
    energy_array.copy_from_slice(&energy_bytes);

    let backend = match input.first().copied().unwrap_or_default() % 3 {
        0 => BackendId::Risc0,
        1 => BackendId::Winterfell,
        _ => BackendId::Binius,
    };

    let proof = AuthenticatedProof {
        proof: input.to_vec(),
        signature,
        metadata: ProofMetadata {
            domain_tag: DomainTag::from_bytes(input.iter().copied().take(64).collect()),
            protocol_version: u32::from_le_bytes([
                timestamp_array[0],
                timestamp_array[1],
                timestamp_array[2],
                timestamp_array[3],
            ]),
            client_id,
            timestamp: u64::from_le_bytes(timestamp_array),
            nonce,
            backend,
        },
        public_inputs_hash,
        joules_consumed: f64::from_bits(u64::from_le_bytes(energy_array)),
    };

    let _ = verify_authenticated_signature(&proof, &public_key);
});
