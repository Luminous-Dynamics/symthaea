//! Bounded ASCII length-framed input format for the claim replay fuzzer.

pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAX_RESULT_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_EVIDENCE_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_EVIDENCE_RECORDS: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct ReplayInput<'a> {
    pub claim: &'a [u8],
    pub result: &'a [u8],
    pub evidence: Vec<&'a [u8]>,
}

fn take_decimal_line(input: &[u8], cursor: &mut usize) -> Option<usize> {
    let tail = input.get(*cursor..)?;
    let newline = tail.iter().position(|byte| *byte == b'\n')?;
    let digits = std::str::from_utf8(&tail[..newline]).ok()?;
    if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) {
        return None;
    }
    if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value = digits.parse::<usize>().ok()?;
    *cursor = cursor.checked_add(newline + 1)?;
    Some(value)
}

fn take_payload<'a>(
    input: &'a [u8],
    cursor: &mut usize,
    length: usize,
    maximum: usize,
) -> Option<&'a [u8]> {
    if length > maximum {
        return None;
    }
    let end = cursor.checked_add(length)?;
    let payload = input.get(*cursor..end)?;
    *cursor = end;
    Some(payload)
}

/// Parse an exact frame:
/// <claim_len>\n<claim><result_len>\n<result><evidence_count>\n
/// followed by <evidence_len>\n<evidence> repeated evidence_count times.
///
/// Decimal lengths make corpus seeds inspectable. The parser rejects overflow,
/// truncation, excess counts, oversize fields and trailing bytes before the
/// production verifier is called.
pub fn parse_frame(input: &[u8]) -> Option<ReplayInput<'_>> {
    if input.len() > MAX_FRAME_BYTES {
        return None;
    }
    let mut cursor = 0usize;
    let claim_len = take_decimal_line(input, &mut cursor)?;
    let claim = take_payload(input, &mut cursor, claim_len, MAX_REQUEST_BYTES)?;

    let result_len = take_decimal_line(input, &mut cursor)?;
    let result = take_payload(input, &mut cursor, result_len, MAX_RESULT_BYTES)?;

    let evidence_count = take_decimal_line(input, &mut cursor)?;
    if evidence_count > MAX_EVIDENCE_RECORDS {
        return None;
    }

    let mut evidence = Vec::with_capacity(evidence_count);
    let mut total_evidence_bytes = 0usize;
    for _ in 0..evidence_count {
        let length = take_decimal_line(input, &mut cursor)?;
        total_evidence_bytes = total_evidence_bytes.checked_add(length)?;
        if total_evidence_bytes > MAX_EVIDENCE_BYTES {
            return None;
        }
        evidence.push(take_payload(
            input,
            &mut cursor,
            length,
            MAX_EVIDENCE_BYTES,
        )?);
    }

    if cursor != input.len() {
        return None;
    }
    Some(ReplayInput {
        claim,
        result,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exact_frame_and_preserves_payload_boundaries() {
        let frame = b"2\n{}\n2\n{}\n1\n3\nabc";
        let parsed = parse_frame(frame).unwrap();
        assert_eq!(parsed.claim, b"{}");
        assert_eq!(parsed.result, b"{}");
        assert_eq!(parsed.evidence, vec![&b"abc"[..]]);
    }

    #[test]
    fn rejects_truncation_trailing_bytes_and_noncanonical_lengths() {
        assert!(parse_frame(b"2\n{").is_none());
        assert!(parse_frame(b"2\n{}\n2\n{}\n0\nx").is_none());
        assert!(parse_frame(b"02\n{}\n2\n{}\n0\n").is_none());
        assert!(parse_frame(b"+2\n{}\n2\n{}\n0\n").is_none());
    }

    #[test]
    fn rejects_excessive_record_count_and_total_frame_size() {
        let too_many = format!("2\n{{}}\n2\n{{}}\n{}\n", MAX_EVIDENCE_RECORDS + 1);
        assert!(parse_frame(too_many.as_bytes()).is_none());

        let oversized = vec![b'x'; MAX_FRAME_BYTES + 1];
        assert!(parse_frame(&oversized).is_none());
    }
}
