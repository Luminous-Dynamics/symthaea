#![no_main]

//! Raw-byte fuzzing of the bounded REA claim request/result/Evidence replay path.
//!
//! Input frame:
//!   u32 LE claim length
//!   u32 LE result length
//!   u8 evidence count
//!   evidence_count × u32 LE Evidence length
//!   claim bytes || result bytes || Evidence bytes...
//!
//! The entire frame is capped before slicing. Each component receives a tighter
//! target-specific bound than the production CLI, making memory/time bounded.

use libfuzzer_sys::fuzz_target;
use rea_claim_replay_verifier::replay_claim_evaluation;

const MAX_FRAME_BYTES: usize = 256 * 1024;
const MAX_CLAIM_BYTES: usize = 64 * 1024;
const MAX_RESULT_BYTES: usize = 128 * 1024;
const MAX_EVIDENCE_BYTES: usize = 16 * 1024;
const MAX_EVIDENCE_COUNT: usize = 8;

fn read_u32(data: &[u8], cursor: &mut usize) -> Option<usize> {
    let end = cursor.checked_add(4)?;
    let bytes: [u8; 4] = data.get(*cursor..end)?.try_into().ok()?;
    *cursor = end;
    Some(u32::from_le_bytes(bytes) as usize)
}

fn read_part<'a>(data: &'a [u8], cursor: &mut usize, length: usize) -> Option<&'a [u8]> {
    let end = cursor.checked_add(length)?;
    let part = data.get(*cursor..end)?;
    *cursor = end;
    Some(part)
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FRAME_BYTES {
        return;
    }

    let mut cursor = 0usize;
    let Some(claim_length) = read_u32(data, &mut cursor) else {
        return;
    };
    let Some(result_length) = read_u32(data, &mut cursor) else {
        return;
    };
    let Some(count_byte) = data.get(cursor) else {
        return;
    };
    cursor += 1;

    let evidence_count = *count_byte as usize;
    if claim_length > MAX_CLAIM_BYTES
        || result_length > MAX_RESULT_BYTES
        || evidence_count > MAX_EVIDENCE_COUNT
    {
        return;
    }

    let mut evidence_lengths = Vec::with_capacity(evidence_count);
    let mut total_evidence_bytes = 0usize;
    for _ in 0..evidence_count {
        let Some(length) = read_u32(data, &mut cursor) else {
            return;
        };
        if length > MAX_EVIDENCE_BYTES {
            return;
        }
        let Some(total) = total_evidence_bytes.checked_add(length) else {
            return;
        };
        total_evidence_bytes = total;
        evidence_lengths.push(length);
    }

    let Some(claim_json) = read_part(data, &mut cursor, claim_length) else {
        return;
    };
    let Some(result_json) = read_part(data, &mut cursor, result_length) else {
        return;
    };

    let mut evidence = Vec::with_capacity(evidence_count);
    for length in evidence_lengths {
        let Some(record) = read_part(data, &mut cursor, length) else {
            return;
        };
        evidence.push(record);
    }

    // Trailing bytes are not silently ignored by the framing protocol.
    if cursor != data.len() {
        return;
    }

    let _ = replay_claim_evaluation(claim_json, &evidence, result_json);
});
