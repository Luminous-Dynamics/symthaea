use std::ops::Range;

/// Strict limits for the first supported HTTP/1.1 response profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Http11ResponseLimits {
    pub max_header_bytes: usize,
    pub max_header_count: usize,
    pub max_body_bytes: usize,
}

impl Default for Http11ResponseLimits {
    fn default() -> Self {
        Self { max_header_bytes: 16 * 1024, max_header_count: 64, max_body_bytes: 1 << 20 }
    }
}

/// A body offset mapping for one strict HTTP/1.1 JSON response.
/// The body range is relative to the exact received application-data byte stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Http11JsonResponseMappingV1 {
    pub status_code: u16,
    pub content_length: usize,
    pub transcript_body_range: Range<usize>,
}

/// Errors from the strict HTTP/1.1 single-response mapper.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Http11ResponseError {
    #[error("HTTP response header block is empty, oversized, or not CRLF terminated")]
    InvalidHeaderBlock,
    #[error("HTTP response uses malformed or non-ASCII syntax")]
    InvalidSyntax,
    #[error("HTTP response header count exceeds its configured bound")]
    TooManyHeaders,
    #[error("HTTP response contains a duplicate header name")]
    DuplicateHeader,
    #[error("HTTP response uses obsolete folded header syntax")]
    ObsoleteFoldedHeader,
    #[error("HTTP response has an invalid status line or unsupported status code")]
    UnsupportedStatus,
    #[error("HTTP response must contain exactly one canonical Content-Length header")]
    InvalidContentLength,
    #[error("HTTP response must contain exactly one application/json Content-Type header with no parameters")]
    InvalidContentType,
    #[error("Transfer-Encoding is outside the admitted framing profile")]
    TransferEncodingUnsupported,
    #[error("Content-Encoding is outside the admitted identity representation profile")]
    ContentEncodingUnsupported,
    #[error("trailers, upgrades, and partial-content responses are outside the admitted profile")]
    UnsupportedFramingHeader,
    #[error("HTTP response body exceeds the configured limit")]
    BodyTooLarge,
    #[error("received transcript length does not equal header length plus Content-Length")]
    TranscriptLengthMismatch,
    #[error("body-relative field range is outside the mapped HTTP body")]
    BodyFieldRangeOutOfBounds,
    #[error("byte offset arithmetic overflow")]
    OffsetOverflow,
}

/// Parse an exact HTTP/1.1 response header block and map its JSON body.
///
/// Supported profile: status 200, one Content-Length, one Content-Type exactly
/// equal to application/json, no Transfer-Encoding, no Content-Encoding, no
/// Trailer, no Upgrade, and no trailing/pipelined bytes after the body.
///
/// `header_block` includes the final CRLF CRLF and contains no body bytes.
/// `received_transcript_len` is the total length of the exact response bytes in
/// the TLS application-data received direction. The caller must separately
/// prove request/response ordinal correspondence and authenticate the header
/// bytes through the TLSNotary transcript proof.
pub fn parse_http11_single_json_response_header_v1(
    header_block: &[u8],
    received_transcript_len: usize,
    limits: Http11ResponseLimits,
) -> Result<Http11JsonResponseMappingV1, Http11ResponseError> {
    if header_block.len() < 4
        || header_block.len() > limits.max_header_bytes
        || !header_block.ends_with(b"\r\n\r\n")
    {
        return Err(Http11ResponseError::InvalidHeaderBlock);
    }

    let section = &header_block[..header_block.len() - 4];
    let lines = split_crlf_lines(section)?;
    if lines.is_empty() {
        return Err(Http11ResponseError::InvalidHeaderBlock);
    }
    if lines.len().saturating_sub(1) > limits.max_header_count {
        return Err(Http11ResponseError::TooManyHeaders);
    }

    let status_line = std::str::from_utf8(lines[0]).map_err(|_| Http11ResponseError::InvalidSyntax)?;
    let mut status_parts = status_line.splitn(3, ' ');
    let version = status_parts.next().ok_or(Http11ResponseError::UnsupportedStatus)?;
    let status = status_parts.next().ok_or(Http11ResponseError::UnsupportedStatus)?;
    let reason = status_parts.next().unwrap_or("");
    if version != "HTTP/1.1" || status.len() != 3 || !status.bytes().all(|b| b.is_ascii_digit())
        || reason.bytes().any(|b| b < 0x20 || b == 0x7f)
    {
        return Err(Http11ResponseError::UnsupportedStatus);
    }
    let status_code = status.parse::<u16>().map_err(|_| Http11ResponseError::UnsupportedStatus)?;
    if status_code != 200 {
        return Err(Http11ResponseError::UnsupportedStatus);
    }

    let mut seen = std::collections::HashSet::<String>::new();
    let mut content_length: Option<usize> = None;
    let mut content_type: Option<Vec<u8>> = None;

    for line in lines.iter().skip(1) {
        if line.is_empty() {
            return Err(Http11ResponseError::InvalidSyntax);
        }
        if line.first().is_some_and(|byte| *byte == b' ' || *byte == b'\t') {
            return Err(Http11ResponseError::ObsoleteFoldedHeader);
        }
        let colon = line.iter().position(|b| *b == b':').ok_or(Http11ResponseError::InvalidSyntax)?;
        if colon == 0 || !line[..colon].iter().copied().all(is_tchar) {
            return Err(Http11ResponseError::InvalidSyntax);
        }
        let name = std::str::from_utf8(&line[..colon])
            .map_err(|_| Http11ResponseError::InvalidSyntax)?
            .to_ascii_lowercase();
        if !seen.insert(name.clone()) {
            return Err(Http11ResponseError::DuplicateHeader);
        }
        let raw_value = trim_ows(&line[colon + 1..]);
        if raw_value.iter().any(|b| (*b < 0x20 && *b != b'\t') || *b == 0x7f) {
            return Err(Http11ResponseError::InvalidSyntax);
        }

        match name.as_str() {
            "content-length" => {
                if raw_value.is_empty() || !raw_value.iter().all(u8::is_ascii_digit)
                    || (raw_value.len() > 1 && raw_value[0] == b'0')
                {
                    return Err(Http11ResponseError::InvalidContentLength);
                }
                let text = std::str::from_utf8(raw_value).map_err(|_| Http11ResponseError::InvalidContentLength)?;
                content_length = Some(text.parse::<usize>().map_err(|_| Http11ResponseError::InvalidContentLength)?);
            }
            "content-type" => content_type = Some(raw_value.to_vec()),
            "transfer-encoding" => return Err(Http11ResponseError::TransferEncodingUnsupported),
            "content-encoding" => return Err(Http11ResponseError::ContentEncodingUnsupported),
            "trailer" | "upgrade" | "content-range" => return Err(Http11ResponseError::UnsupportedFramingHeader),
            _ => {}
        }
    }

    let content_length = content_length.ok_or(Http11ResponseError::InvalidContentLength)?;
    let content_type = content_type.ok_or(Http11ResponseError::InvalidContentType)?;
    if !content_type.eq_ignore_ascii_case(b"application/json") {
        return Err(Http11ResponseError::InvalidContentType);
    }
    if content_length > limits.max_body_bytes {
        return Err(Http11ResponseError::BodyTooLarge);
    }
    let body_start = header_block.len();
    let body_end = body_start.checked_add(content_length).ok_or(Http11ResponseError::OffsetOverflow)?;
    if body_end != received_transcript_len {
        return Err(Http11ResponseError::TranscriptLengthMismatch);
    }

    Ok(Http11JsonResponseMappingV1 {
        status_code,
        content_length,
        transcript_body_range: body_start..body_end,
    })
}

/// Result: a verifier-selected hidden JSON field mapped into the received application-data stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Http11FieldRangeCandidateV1 {
    pub parser_profile_id: &'static str,
    pub json_pointer: String,
    pub response_body_range: Range<usize>,
    pub field_body_range: Range<usize>,
    pub hidden_value_body_range: Range<usize>,
    pub field_received_stream_range: Range<usize>,
    pub hidden_value_received_stream_range: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Http11FieldBindingError {
    #[error(transparent)]
    Http(#[from] Http11ResponseError),
    #[error(transparent)]
    Field(#[from] crate::FieldSelectorError),
    #[error("partial JSON body length does not match the HTTP Content-Length")]
    PartialBodyLengthMismatch,
}

/// Bind one hidden JSON string to the exact requested field and map it into the
/// received application-data stream.
///
/// The header block and total stream length must be anchored by the caller.
/// The body input must contain authenticated disclosed bytes as Some and exactly
/// one undisclosed field-value run as None. This returns offsets; it does not
/// itself verify TLSNotary commitments or transcript openings.
pub fn bind_hidden_json_field_to_received_stream_v1(
    response_header_block: &[u8],
    received_stream_len: usize,
    partial_json_body: &[Option<u8>],
    json_pointer: &str,
    response_limits: Http11ResponseLimits,
    selector_limits: crate::FieldSelectorLimits,
) -> Result<Http11FieldRangeCandidateV1, Http11FieldBindingError> {
    let response = parse_http11_single_json_response_header_v1(
        response_header_block, received_stream_len, response_limits,
    )?;
    if partial_json_body.len() != response.content_length {
        return Err(Http11FieldBindingError::PartialBodyLengthMismatch);
    }
    let field = crate::bind_single_hidden_string_field_v1(
        partial_json_body, json_pointer, selector_limits,
    )?;
    let field_received_stream_range = map_body_range_to_received_transcript_v1(
        &response, field.body_range.clone(),
    )?;
    let hidden_value_received_stream_range = map_body_range_to_received_transcript_v1(
        &response, field.hidden_content_range.clone(),
    )?;
    Ok(Http11FieldRangeCandidateV1 {
        parser_profile_id: field.parser_profile_id,
        json_pointer: field.json_pointer,
        response_body_range: response.transcript_body_range,
        field_body_range: field.body_range,
        hidden_value_body_range: field.hidden_content_range,
        field_received_stream_range,
        hidden_value_received_stream_range,
    })
}
/// Map a body-relative JSON value or hidden-string range into the received
/// application-data transcript direction.
pub fn map_body_range_to_received_transcript_v1(
    response: &Http11JsonResponseMappingV1,
    body_range: Range<usize>,
) -> Result<Range<usize>, Http11ResponseError> {
    if body_range.start > body_range.end || body_range.end > response.content_length {
        return Err(Http11ResponseError::BodyFieldRangeOutOfBounds);
    }
    let start = response.transcript_body_range.start
        .checked_add(body_range.start)
        .ok_or(Http11ResponseError::OffsetOverflow)?;
    let end = response.transcript_body_range.start
        .checked_add(body_range.end)
        .ok_or(Http11ResponseError::OffsetOverflow)?;
    if end > response.transcript_body_range.end {
        return Err(Http11ResponseError::BodyFieldRangeOutOfBounds);
    }
    Ok(start..end)
}

fn split_crlf_lines(bytes: &[u8]) -> Result<Vec<&[u8]>, Http11ResponseError> {
    if bytes.iter().enumerate().any(|(i, b)| {
        (*b == b'\n' && (i == 0 || bytes[i - 1] != b'\r'))
            || (*b == b'\r' && bytes.get(i + 1) != Some(&b'\n'))
    }) {
        return Err(Http11ResponseError::InvalidSyntax);
    }
    let mut lines = Vec::new();
    let mut start = 0usize;
    while let Some(relative) = find_subslice(&bytes[start..], b"\r\n") {
        let end = start.checked_add(relative).ok_or(Http11ResponseError::OffsetOverflow)?;
        lines.push(&bytes[start..end]);
        start = end + 2;
    }
    if start < bytes.len() {
        lines.push(&bytes[start..]);
    }
    Ok(lines)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn trim_ows(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(|byte| *byte == b' ' || *byte == b'\t') { value = &value[1..]; }
    while value.last().is_some_and(|byte| *byte == b' ' || *byte == b'\t') { value = &value[..value.len() - 1]; }
    value
}

fn is_tchar(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(response: &[u8]) -> &[u8] {
        let end = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        &response[..end]
    }

    #[test]
    fn maps_exact_json_body_start_and_field_range() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 20\r\n\r\n{\"dob\":\"1985-03-12\"}";
        let parsed = parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()).unwrap();
        let absolute = map_body_range_to_received_transcript_v1(&parsed, 8..19).unwrap();
        assert_eq!(&response[absolute], b"1985-03-12\"");
    }

    #[test]
    fn binds_hidden_json_field_to_exact_received_stream_range() {
        let body = br#"{"taxpayer":{"date_of_birth":"1985-03-12"},"other_date":"2020-01-01"}"#;
        let field = crate::select_json_field_v1(body, "/taxpayer/date_of_birth", crate::FieldSelectorLimits::default()).unwrap();
        let hidden = (field.body_range.start + 1)..(field.body_range.end - 1);
        let partial = body.iter().enumerate().map(|(i, b)| if hidden.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        let header = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len());
        let header_bytes = header.as_bytes();
        let received_len = header_bytes.len() + body.len();
        let result = bind_hidden_json_field_to_received_stream_v1(
            header_bytes, received_len, &partial, "/taxpayer/date_of_birth",
            Http11ResponseLimits::default(), crate::FieldSelectorLimits::default(),
        ).unwrap();
        assert_eq!(result.field_body_range, field.body_range);
        assert_eq!(result.hidden_value_body_range, hidden);
        assert_eq!(result.field_received_stream_range, (header_bytes.len() + field.body_range.start)..(header_bytes.len() + field.body_range.end));
        assert_eq!(result.hidden_value_received_stream_range, (header_bytes.len() + hidden.start)..(header_bytes.len() + hidden.end));
    }

    #[test]
    fn composite_binding_rejects_wrong_json_pointer() {
        let body = br#"{"dob":"1985-03-12","other_date":"2020-01-01"}"#;
        let field = crate::select_json_field_v1(body, "/dob", crate::FieldSelectorLimits::default()).unwrap();
        let hidden = (field.body_range.start + 1)..(field.body_range.end - 1);
        let partial = body.iter().enumerate().map(|(i, b)| if hidden.contains(&i) { None } else { Some(*b) }).collect::<Vec<_>>();
        let header = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n", body.len());
        assert!(bind_hidden_json_field_to_received_stream_v1(
            header.as_bytes(), header.len() + body.len(), &partial, "/other_date",
            Http11ResponseLimits::default(), crate::FieldSelectorLimits::default(),
        ).is_err());
    }

    #[test]
    fn rejects_duplicate_content_length() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::DuplicateHeader));
    }

    #[test]
    fn rejects_transfer_encoding_even_with_content_length() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::TransferEncodingUnsupported));
    }

    #[test]
    fn rejects_content_encoding() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Encoding: gzip\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::ContentEncodingUnsupported));
    }

    #[test]
    fn rejects_parameterized_content_type() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::InvalidContentType));
    }

    #[test]
    fn rejects_non_200_status() {
        let response = b"HTTP/1.1 206 Partial Content\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::UnsupportedStatus));
    }

    #[test]
    fn rejects_trailing_or_pipelined_bytes() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}NEXT";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::TranscriptLengthMismatch));
    }

    #[test]
    fn rejects_body_length_mismatch() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 3\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::TranscriptLengthMismatch));
    }

    #[test]
    fn rejects_obs_fold() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n X-Injected: yes\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()), Err(Http11ResponseError::ObsoleteFoldedHeader));
    }

    #[test]
    fn rejects_oversized_body() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let limits = Http11ResponseLimits { max_body_bytes: 1, ..Http11ResponseLimits::default() };
        assert_eq!(parse_http11_single_json_response_header_v1(headers(response), response.len(), limits), Err(Http11ResponseError::BodyTooLarge));
    }

    #[test]
    fn rejects_out_of_bounds_body_field_range() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let parsed = parse_http11_single_json_response_header_v1(headers(response), response.len(), Http11ResponseLimits::default()).unwrap();
        assert_eq!(map_body_range_to_received_transcript_v1(&parsed, 0..3), Err(Http11ResponseError::BodyFieldRangeOutOfBounds));
    }
}
