//! Independently recomputed evidence for maximum-independent-set candidates.
//!
//! This module deliberately accepts canonical input bytes instead of caller-supplied
//! digest strings. It recomputes SHA-256 over the exact graph and candidate bytes,
//! checks candidate bounds/uniqueness/independence, and constructs a deterministic
//! receipt. It does not sign the receipt or attest which machine executed this code.

use std::error::Error;
use std::fmt;

const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c,
    0x1f83_d9ab, 0x5be0_cd19,
];

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1,
    0x923f_82a4, 0xab1c_5ed5, 0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174, 0xe49b_69c1, 0xefbe_4786,
    0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147,
    0x06ca_6351, 0x1429_2967, 0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85, 0xa2bfe8a1, 0xa81a_664b,
    0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a,
    0x5b9c_ca4f, 0x682e_6ff3, 0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

/// Result category emitted only after candidate bytes have been independently checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisEvidenceVerdict {
    /// Every selected vertex exists and no selected pair shares an edge.
    VerifiedIndependentSet,
    /// At least one edge has both endpoints selected.
    VerifiedConflict,
}

/// Experiment identity bound into a verifier receipt.
///
/// The IDs are restricted to canonical single-token ASCII so they can be embedded
/// in receipt bytes without escaping ambiguity. This binds a receipt to a record
/// identity; it is not proof that the candidate came from the claimed process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisEvidenceContext {
    protocol_sha256: String,
    trial_id: String,
    pair_id: String,
    seed: u64,
}

impl MisEvidenceContext {
    /// Construct context after validating the protocol digest and canonical identifiers.
    pub fn new(
        protocol_sha256: &str,
        trial_id: &str,
        pair_id: &str,
        seed: u64,
    ) -> Result<Self, MisEvidenceError> {
        validate_lower_sha256(protocol_sha256)?;
        validate_context_id(trial_id)?;
        validate_context_id(pair_id)?;
        Ok(Self {
            protocol_sha256: protocol_sha256.to_owned(),
            trial_id: trial_id.to_owned(),
            pair_id: pair_id.to_owned(),
            seed,
        })
    }

    /// Frozen experiment-protocol SHA-256.
    pub fn protocol_sha256(&self) -> &str { &self.protocol_sha256 }
    /// Unique attempt identifier for this arm.
    pub fn trial_id(&self) -> &str { &self.trial_id }
    /// Paired attempt identifier shared by comparison arms.
    pub fn pair_id(&self) -> &str { &self.pair_id }
    /// Declared PRNG seed identity for this attempt.
    pub fn seed(&self) -> u64 { self.seed }
}

fn validate_context_id(value: &str) -> Result<(), MisEvidenceError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(MisEvidenceError::InvalidTrialContext);
    }
    Ok(())
}

/// A non-forgeable-by-safe-API result of checking canonical graph and candidate bytes.
///
/// Fields are private so callers cannot manufacture a successful receipt by filling
/// metadata fields. Construct this value with `verify_mis_candidate` or validate a
/// persisted receipt with `verify_mis_candidate_with_receipt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedMisCandidate {
    instance_sha256: String,
    candidate_sha256: String,
    verifier_sha256: String,
    receipt_sha256: String,
    receipt_bytes: Vec<u8>,
    target_size: usize,
    candidate_size: usize,
    conflict_count: usize,
    verdict: MisEvidenceVerdict,
    context: Option<MisEvidenceContext>,
}

impl VerifiedMisCandidate {
    /// SHA-256 recomputed from the exact canonical graph bytes.
    pub fn instance_sha256(&self) -> &str { &self.instance_sha256 }
    /// SHA-256 recomputed from the exact canonical candidate bytes.
    pub fn candidate_sha256(&self) -> &str { &self.candidate_sha256 }
    /// Caller-pinned digest identifying the verifier build/source used by this run.
    pub fn verifier_sha256(&self) -> &str { &self.verifier_sha256 }
    /// SHA-256 recomputed from the exact canonical receipt bytes.
    pub fn receipt_sha256(&self) -> &str { &self.receipt_sha256 }
    /// Canonical receipt bytes, suitable for persistence and external review.
    pub fn receipt_bytes(&self) -> &[u8] { &self.receipt_bytes }
    /// Target size against which this candidate was checked.
    pub fn target_size(&self) -> usize { self.target_size }
    /// Candidate cardinality independently counted from parsed candidate bytes.
    pub fn candidate_size(&self) -> usize { self.candidate_size }
    /// Number of graph edges whose two endpoints are both selected.
    pub fn conflict_count(&self) -> usize { self.conflict_count }
    /// Feasibility verdict from recomputation.
    pub fn verdict(&self) -> MisEvidenceVerdict { self.verdict }
    /// Context bound into the canonical receipt, absent for generic non-experiment verification.
    pub fn context(&self) -> Option<&MisEvidenceContext> { self.context.as_ref() }
    /// Whether this verified independent set reaches the frozen target.
    pub fn reaches_target(&self) -> bool {
        self.verdict == MisEvidenceVerdict::VerifiedIndependentSet
            && self.conflict_count == 0
            && self.candidate_size >= self.target_size
    }
}

/// Convert a QOBLIB-style DIMACS edge graph into the strict canonical graph encoding.
///
/// Supported input records are comment lines beginning with `c`, one problem line
/// `p edge N M`, and exactly M undirected edge rows `e U V` using DIMACS one-based
/// vertex labels. Comments and record order do not affect the canonical output, but
/// duplicate edges, self-loops, out-of-range labels, duplicate headers, malformed
/// records, and declared edge-count mismatch are rejected. Preserve and separately
/// hash the original source bytes for dataset provenance; the returned bytes are the
/// canonical semantic graph representation used by the verifier.
pub fn canonicalize_dimacs_graph(raw_bytes: &[u8]) -> Result<Vec<u8>, MisEvidenceError> {
    let text = std::str::from_utf8(raw_bytes).map_err(|_| MisEvidenceError::InvalidUtf8)?;
    let mut header: Option<(usize, usize)> = None;
    let mut raw_edges = Vec::new();
    let mut saw_edge = false;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "c" || line.starts_with("c ") || line.starts_with("c\\t") {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first().copied() {
            Some("p") => {
                if header.is_some() || saw_edge {
                    return Err(MisEvidenceError::InvalidDimacsHeaderPosition);
                }
                if fields.len() != 4 || fields[1] != "edge" {
                    return Err(MisEvidenceError::InvalidDimacsHeader);
                }
                let vertices = parse_decimal(fields[2])?;
                let edges = parse_decimal(fields[3])?;
                header = Some((vertices, edges));
            }
            Some("e") => {
                let (vertex_count, _) = header.ok_or(MisEvidenceError::DimacsEdgeBeforeHeader)?;
                if fields.len() != 3 {
                    return Err(MisEvidenceError::InvalidDimacsEdge);
                }
                saw_edge = true;
                let first = parse_decimal(fields[1])?;
                let second = parse_decimal(fields[2])?;
                if first == 0 || second == 0 || first > vertex_count || second > vertex_count || first == second {
                    return Err(MisEvidenceError::InvalidDimacsVertex);
                }
                let u = first - 1;
                let v = second - 1;
                raw_edges.push(if u < v { (u, v) } else { (v, u) });
            }
            _ => return Err(MisEvidenceError::InvalidDimacsRecord),
        }
    }

    let (vertex_count, declared_edges) = header.ok_or(MisEvidenceError::MissingDimacsHeader)?;
    if raw_edges.len() != declared_edges {
        return Err(MisEvidenceError::DimacsEdgeCountMismatch);
    }
    raw_edges.sort_unstable();
    if raw_edges.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(MisEvidenceError::DuplicateDimacsEdge);
    }

    let mut canonical = format!(
        "SYMTHEA-MIS-GRAPH-V1\\nvertices {vertex_count}\\nedges {declared_edges}\\n"
    );
    for (u, v) in raw_edges {
        canonical.push_str(&format!("{u} {v}\\n"));
    }
    Ok(canonical.into_bytes())
}

/// Parse, hash, and independently validate canonical graph and candidate bytes.
///
/// Graph format (UTF-8, LF line endings, final newline, zero-based vertices):
/// `SYMTHEA-MIS-GRAPH-V1\nvertices N\nedges M\nu v\n...`
///
/// Each edge must satisfy `u < v < N`, and edge rows must be strictly lexicographically
/// increasing. There are exactly M edge rows.
///
/// Candidate format:
/// `SYMTHEA-MIS-CANDIDATE-V1\ncount K\nv_0\n...\nv_(K-1)\n`
///
/// Candidate vertices must be strictly increasing and in range. These canonical formats
/// are purposely narrow: accepting multiple serializations for one logical instance
/// would make digest identity ambiguous.
pub fn verify_mis_candidate(
    graph_bytes: &[u8],
    candidate_bytes: &[u8],
    target_size: usize,
    verifier_sha256: &str,
) -> Result<VerifiedMisCandidate, MisEvidenceError> {
    verify_mis_candidate_internal(graph_bytes, candidate_bytes, target_size, verifier_sha256, None)
}

/// Verify a candidate and bind its receipt to the exact experiment-trial identity.
///
/// An unscoped receipt produced by `verify_mis_candidate` is intentionally insufficient
/// for a successful `MatchedTrialRecord`. The context is identity binding, not process
/// attestation or a cryptographic signature.
pub fn verify_mis_candidate_for_trial(
    graph_bytes: &[u8],
    candidate_bytes: &[u8],
    target_size: usize,
    verifier_sha256: &str,
    context: MisEvidenceContext,
) -> Result<VerifiedMisCandidate, MisEvidenceError> {
    verify_mis_candidate_internal(
        graph_bytes,
        candidate_bytes,
        target_size,
        verifier_sha256,
        Some(context),
    )
}

fn verify_mis_candidate_internal(
    graph_bytes: &[u8],
    candidate_bytes: &[u8],
    target_size: usize,
    verifier_sha256: &str,
    context: Option<MisEvidenceContext>,
) -> Result<VerifiedMisCandidate, MisEvidenceError> {
    if target_size == 0 {
        return Err(MisEvidenceError::ZeroTarget);
    }
    validate_lower_sha256(verifier_sha256)?;
    let (vertex_count, edges) = parse_canonical_graph(graph_bytes)?;
    let vertices = parse_canonical_candidate(candidate_bytes, vertex_count)?;
    let selected_count = vertices.len();
    let mut conflicts = 0usize;
    for &(u, v) in &edges {
        if vertices.binary_search(&u).is_ok() && vertices.binary_search(&v).is_ok() {
            conflicts += 1;
        }
    }

    let verdict = if conflicts == 0 {
        MisEvidenceVerdict::VerifiedIndependentSet
    } else {
        MisEvidenceVerdict::VerifiedConflict
    };
    let instance_sha256 = sha256_hex(graph_bytes);
    let candidate_sha256 = sha256_hex(candidate_bytes);
    let receipt_bytes = canonical_receipt_bytes(
        &instance_sha256,
        verifier_sha256,
        &candidate_sha256,
        target_size,
        selected_count,
        conflicts,
        verdict,
        context.as_ref(),
    );
    let receipt_sha256 = sha256_hex(&receipt_bytes);
    Ok(VerifiedMisCandidate {
        instance_sha256,
        candidate_sha256,
        verifier_sha256: verifier_sha256.to_owned(),
        receipt_sha256,
        receipt_bytes,
        target_size,
        candidate_size: selected_count,
        conflict_count: conflicts,
        verdict,
        context,
    })
}

/// Re-verify graph and candidate bytes and fail unless persisted receipt bytes are exact.
///
/// The expected receipt is reconstructed from recomputed digests and verifier output;
/// merely supplying a well-formed digest or matching self-described metadata is not enough.
pub fn verify_mis_candidate_with_receipt(
    graph_bytes: &[u8],
    candidate_bytes: &[u8],
    target_size: usize,
    verifier_sha256: &str,
    persisted_receipt_bytes: &[u8],
) -> Result<VerifiedMisCandidate, MisEvidenceError> {
    let evidence = verify_mis_candidate(graph_bytes, candidate_bytes, target_size, verifier_sha256)?;
    if evidence.receipt_bytes() != persisted_receipt_bytes {
        return Err(MisEvidenceError::ReceiptBytesMismatch);
    }
    if sha256_hex(persisted_receipt_bytes) != evidence.receipt_sha256() {
        return Err(MisEvidenceError::ReceiptDigestMismatch);
    }
    Ok(evidence)
}

/// Re-verify a trial-scoped receipt and fail unless its exact bytes are recomputed.
pub fn verify_mis_candidate_for_trial_with_receipt(
    graph_bytes: &[u8],
    candidate_bytes: &[u8],
    target_size: usize,
    verifier_sha256: &str,
    context: MisEvidenceContext,
    persisted_receipt_bytes: &[u8],
) -> Result<VerifiedMisCandidate, MisEvidenceError> {
    let evidence = verify_mis_candidate_for_trial(
        graph_bytes,
        candidate_bytes,
        target_size,
        verifier_sha256,
        context,
    )?;
    if evidence.receipt_bytes() != persisted_receipt_bytes {
        return Err(MisEvidenceError::ReceiptBytesMismatch);
    }
    if sha256_hex(persisted_receipt_bytes) != evidence.receipt_sha256() {
        return Err(MisEvidenceError::ReceiptDigestMismatch);
    }
    Ok(evidence)
}

/// SHA-256 of the exact supplied bytes, returned as 64 lowercase hexadecimal digits.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(bytes.len().saturating_add(72));
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = SHA256_INITIAL_STATE;
    for block in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let x = schedule[index - 15];
            let y = schedule[index - 2];
            let sigma0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let sigma1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(sigma1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(sigma1)
                .wrapping_add(choose)
                .wrapping_add(SHA256_ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = sigma0.wrapping_add(majority);
            h = g; g = f; f = e; e = d.wrapping_add(t1);
            d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (target, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *target = target.wrapping_add(value);
        }
    }
    let mut output = String::with_capacity(64);
    for word in state {
        use std::fmt::Write as _;
        let _ = write!(output, "{word:08x}");
    }
    output
}

fn parse_canonical_graph(bytes: &[u8]) -> Result<(usize, Vec<(usize, usize)>), MisEvidenceError> {
    let text = std::str::from_utf8(bytes).map_err(|_| MisEvidenceError::InvalidUtf8)?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err(MisEvidenceError::NonCanonicalLineEndings);
    }
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    if lines.len() < 3 || lines[0] != "SYMTHEA-MIS-GRAPH-V1" {
        return Err(MisEvidenceError::InvalidGraphHeader);
    }
    let vertex_count = parse_count_line(lines[1], "vertices ")?;
    let edge_count = parse_count_line(lines[2], "edges ")?;
    if lines.len() != 3usize.saturating_add(edge_count) {
        return Err(MisEvidenceError::EdgeCountMismatch);
    }
    let mut edges = Vec::with_capacity(edge_count);
    let mut previous = None;
    for line in &lines[3..] {
        let mut parts = line.split(' ');
        let u_text = parts.next().ok_or(MisEvidenceError::InvalidEdge)?;
        let v_text = parts.next().ok_or(MisEvidenceError::InvalidEdge)?;
        if parts.next().is_some() || u_text.is_empty() || v_text.is_empty() {
            return Err(MisEvidenceError::InvalidEdge);
        }
        let u = parse_decimal(u_text)?;
        let v = parse_decimal(v_text)?;
        if u >= v || v >= vertex_count {
            return Err(MisEvidenceError::InvalidEdgeEndpoint);
        }
        let edge = (u, v);
        if previous.is_some_and(|old| old >= edge) {
            return Err(MisEvidenceError::EdgesNotStrictlySorted);
        }
        previous = Some(edge);
        edges.push(edge);
    }
    Ok((vertex_count, edges))
}

fn parse_canonical_candidate(bytes: &[u8], vertex_count: usize) -> Result<Vec<usize>, MisEvidenceError> {
    let text = std::str::from_utf8(bytes).map_err(|_| MisEvidenceError::InvalidUtf8)?;
    if !text.ends_with('\n') || text.contains('\r') {
        return Err(MisEvidenceError::NonCanonicalLineEndings);
    }
    let lines: Vec<&str> = text.split_terminator('\n').collect();
    if lines.len() < 2 || lines[0] != "SYMTHEA-MIS-CANDIDATE-V1" {
        return Err(MisEvidenceError::InvalidCandidateHeader);
    }
    let count = parse_count_line(lines[1], "count ")?;
    if lines.len() != 2usize.saturating_add(count) {
        return Err(MisEvidenceError::CandidateCountMismatch);
    }
    let mut vertices = Vec::with_capacity(count);
    let mut previous = None;
    for line in &lines[2..] {
        let vertex = parse_decimal(line)?;
        if vertex >= vertex_count {
            return Err(MisEvidenceError::CandidateVertexOutOfRange(vertex));
        }
        if previous.is_some_and(|old| old >= vertex) {
            return Err(MisEvidenceError::CandidateVerticesNotStrictlySorted);
        }
        previous = Some(vertex);
        vertices.push(vertex);
    }
    Ok(vertices)
}

fn parse_count_line(line: &str, prefix: &str) -> Result<usize, MisEvidenceError> {
    let value = line.strip_prefix(prefix).ok_or(MisEvidenceError::InvalidCountLine)?;
    parse_decimal(value)
}

fn parse_decimal(value: &str) -> Result<usize, MisEvidenceError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(MisEvidenceError::NonCanonicalDecimal);
    }
    value.parse::<usize>().map_err(|_| MisEvidenceError::DecimalOverflow)
}

fn validate_lower_sha256(value: &str) -> Result<(), MisEvidenceError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(MisEvidenceError::InvalidVerifierDigest);
    }
    Ok(())
}

fn canonical_receipt_bytes(
    instance_sha256: &str,
    verifier_sha256: &str,
    candidate_sha256: &str,
    target_size: usize,
    candidate_size: usize,
    conflict_count: usize,
    verdict: MisEvidenceVerdict,
    context: Option<&MisEvidenceContext>,
) -> Vec<u8> {
    let (context_present, protocol_sha256, trial_id, pair_id, seed) = match context {
        Some(context) => (
            "1",
            context.protocol_sha256(),
            context.trial_id(),
            context.pair_id(),
            context.seed().to_string(),
        ),
        None => ("0", "-", "-", "-", "-".to_owned()),
    };
    let verdict = match verdict {
        MisEvidenceVerdict::VerifiedIndependentSet => "verified-independent-set",
        MisEvidenceVerdict::VerifiedConflict => "verified-conflict",
    };
    format!(
        "SYMTHEA-MIS-RECEIPT-V1\ninstance_sha256 {instance_sha256}\nverifier_sha256 {verifier_sha256}\ncandidate_sha256 {candidate_sha256}\ntarget_size {target_size}\ncandidate_size {candidate_size}\nconflict_count {conflict_count}\nverdict {verdict}\ncontext_present {context_present}\nprotocol_sha256 {protocol_sha256}\ntrial_id {trial_id}\npair_id {pair_id}\nseed {seed}\n"
    ).into_bytes()
}

/// Errors raised when canonical artifacts are malformed or fail independent checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisEvidenceError {
    InvalidUtf8,
    NonCanonicalLineEndings,
    InvalidGraphHeader,
    InvalidCandidateHeader,
    InvalidCountLine,
    NonCanonicalDecimal,
    DecimalOverflow,
    InvalidEdge,
    InvalidEdgeEndpoint,
    EdgeCountMismatch,
    EdgesNotStrictlySorted,
    CandidateCountMismatch,
    CandidateVertexOutOfRange(usize),
    CandidateVerticesNotStrictlySorted,
    InvalidVerifierDigest,
    InvalidDimacsHeader,
    InvalidDimacsHeaderPosition,
    MissingDimacsHeader,
    InvalidDimacsEdge,
    InvalidDimacsVertex,
    InvalidDimacsRecord,
    DimacsEdgeBeforeHeader,
    DimacsEdgeCountMismatch,
    DuplicateDimacsEdge,
    ZeroTarget,
    InvalidTrialContext,
    ReceiptBytesMismatch,
    ReceiptDigestMismatch,
}

impl fmt::Display for MisEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => write!(f, "artifact is not UTF-8"),
            Self::NonCanonicalLineEndings => write!(f, "artifact must use LF line endings and end with LF"),
            Self::InvalidGraphHeader => write!(f, "invalid canonical MIS graph header"),
            Self::InvalidCandidateHeader => write!(f, "invalid canonical MIS candidate header"),
            Self::InvalidCountLine => write!(f, "invalid count line"),
            Self::NonCanonicalDecimal => write!(f, "integer is not in canonical decimal form"),
            Self::DecimalOverflow => write!(f, "integer does not fit in usize"),
            Self::InvalidEdge => write!(f, "edge line must contain exactly two decimal vertex IDs"),
            Self::InvalidEdgeEndpoint => write!(f, "edge endpoints must be distinct, in-range, and ordered u < v"),
            Self::EdgeCountMismatch => write!(f, "declared edge count does not match edge rows"),
            Self::EdgesNotStrictlySorted => write!(f, "edges must be strictly sorted with no duplicates"),
            Self::CandidateCountMismatch => write!(f, "declared candidate count does not match vertex rows"),
            Self::CandidateVertexOutOfRange(v) => write!(f, "candidate vertex {v} is outside the graph"),
            Self::CandidateVerticesNotStrictlySorted => write!(f, "candidate vertices must be strictly sorted with no duplicates"),
            Self::InvalidVerifierDigest => write!(f, "verifier digest must be 64 lowercase hexadecimal characters"),
            Self::InvalidDimacsHeader => write!(f, "DIMACS header must be exactly 'p edge N M'"),
            Self::InvalidDimacsHeaderPosition => write!(f, "DIMACS problem header is duplicated or appears after an edge"),
            Self::MissingDimacsHeader => write!(f, "DIMACS problem header is missing"),
            Self::InvalidDimacsEdge => write!(f, "DIMACS edge must be exactly 'e U V'"),
            Self::InvalidDimacsVertex => write!(f, "DIMACS edge has a zero, self-loop, or out-of-range vertex"),
            Self::InvalidDimacsRecord => write!(f, "unsupported or malformed DIMACS record"),
            Self::DimacsEdgeBeforeHeader => write!(f, "DIMACS edge appears before its problem header"),
            Self::DimacsEdgeCountMismatch => write!(f, "DIMACS declared edge count does not match edge rows"),
            Self::DuplicateDimacsEdge => write!(f, "DIMACS input contains a duplicate undirected edge"),
            Self::ZeroTarget => write!(f, "target cardinality must be positive"),
            Self::InvalidTrialContext => write!(f, "trial context IDs must be non-empty canonical single-token ASCII"),
            Self::ReceiptBytesMismatch => write!(f, "persisted receipt bytes differ from recomputed receipt"),
            Self::ReceiptDigestMismatch => write!(f, "persisted receipt digest does not match its bytes"),
        }
    }
}

impl Error for MisEvidenceError {}

#[cfg(test)]
mod tests {
    use super::*;

    const VERIFIER: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const GRAPH: &[u8] = b"SYMTHEA-MIS-GRAPH-V1\nvertices 4\nedges 2\n0 1\n2 3\n";

    fn candidate(vertices: &[usize]) -> Vec<u8> {
        let mut text = format!("SYMTHEA-MIS-CANDIDATE-V1\ncount {}\n", vertices.len());
        for vertex in vertices {
            text.push_str(&format!("{vertex}\n"));
        }
        text.into_bytes()
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(sha256_hex(b"The quick brown fox jumps over the lazy dog"), "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592");
        let long_message = vec![b'a'; 10_000];
        assert_eq!(
            sha256_hex(&long_message),
            "27dd1f61b867b6a0f6e9d8a41c43231de52107e53ae424de8f847b821db4b711"
        );
        let padding_boundaries = [
            (55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"),
            (56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"),
            (63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"),
            (64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"),
            (65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"),
        ];
        for (length, expected) in padding_boundaries {
            assert_eq!(sha256_hex(&vec![b'a'; length]), expected, "length {length}");
        }
    }

    #[test]
    fn canonicalizes_dimacs_graph_with_one_based_vertices_and_unsorted_edges() {
        let raw = b"c sample graph\\np edge 4 2\\ne 4 3\\ne 2 1\\n";
        let expected = b"SYMTHEA-MIS-GRAPH-V1\\nvertices 4\\nedges 2\\n0 1\\n2 3\\n";
        let canonical = canonicalize_dimacs_graph(raw).unwrap();
        assert_eq!(canonical, expected);
        assert_eq!(sha256_hex(&canonical), sha256_hex(expected));
        let with_comments_and_reordered_edges = b"c reordered\\np edge 4 2\\ne 1 2\\nc between rows\\ne 3 4\\n";
        assert_eq!(canonicalize_dimacs_graph(with_comments_and_reordered_edges).unwrap(), expected);
    }

    #[test]
    fn rejects_bad_dimacs_header_counts_duplicate_edges_and_invalid_vertices() {
        assert!(matches!(
            canonicalize_dimacs_graph(b"e 1 2\\np edge 3 1\\n"),
            Err(MisEvidenceError::DimacsEdgeBeforeHeader)
        ));
        assert!(matches!(
            canonicalize_dimacs_graph(b"p edge 3 2\\ne 1 2\\n"),
            Err(MisEvidenceError::DimacsEdgeCountMismatch)
        ));
        assert!(matches!(
            canonicalize_dimacs_graph(b"p edge 3 2\\ne 1 2\\ne 2 1\\n"),
            Err(MisEvidenceError::DuplicateDimacsEdge)
        ));
        assert!(matches!(
            canonicalize_dimacs_graph(b"p edge 3 1\\ne 0 2\\n"),
            Err(MisEvidenceError::InvalidDimacsVertex)
        ));
        assert!(matches!(
            canonicalize_dimacs_graph(b"p edge 3 1\\ne 2 2\\n"),
            Err(MisEvidenceError::InvalidDimacsVertex)
        ));
        assert!(matches!(
            canonicalize_dimacs_graph(b"p edge 3 1\\ne 1 4\\n"),
            Err(MisEvidenceError::InvalidDimacsVertex)
        ));
    }

    #[test]
    fn verifies_candidate_and_binds_exact_artifact_bytes() {
        let bytes = candidate(&[0, 2]);
        let evidence = verify_mis_candidate(GRAPH, &bytes, 2, VERIFIER).unwrap();
        assert_eq!(evidence.verdict(), MisEvidenceVerdict::VerifiedIndependentSet);
        assert_eq!(evidence.conflict_count(), 0);
        assert!(evidence.reaches_target());
        assert_eq!(evidence.instance_sha256(), sha256_hex(GRAPH));
        assert_eq!(evidence.candidate_sha256(), sha256_hex(&bytes));
        assert_eq!(evidence.receipt_sha256(), sha256_hex(evidence.receipt_bytes()));
        let reread = verify_mis_candidate_with_receipt(GRAPH, &bytes, 2, VERIFIER, evidence.receipt_bytes()).unwrap();
        assert_eq!(reread, evidence);
    }

    #[test]
    fn detects_conflicting_selected_vertices_without_fabricating_a_hit() {
        let bytes = candidate(&[0, 1, 2]);
        let evidence = verify_mis_candidate(GRAPH, &bytes, 2, VERIFIER).unwrap();
        assert_eq!(evidence.verdict(), MisEvidenceVerdict::VerifiedConflict);
        assert_eq!(evidence.conflict_count(), 1);
        assert!(!evidence.reaches_target());
    }

    #[test]
    fn valid_but_below_target_is_not_a_target_hit() {
        let evidence = verify_mis_candidate(GRAPH, &candidate(&[0]), 2, VERIFIER).unwrap();
        assert_eq!(evidence.verdict(), MisEvidenceVerdict::VerifiedIndependentSet);
        assert!(!evidence.reaches_target());
    }

    #[test]
    fn rejects_tampered_candidate_graph_and_receipt_bytes() {
        let bytes = candidate(&[0, 2]);
        let evidence = verify_mis_candidate(GRAPH, &bytes, 2, VERIFIER).unwrap();
        let mut changed_candidate = bytes.clone();
        changed_candidate[changed_candidate.len() - 2] = b'1';
        assert_ne!(sha256_hex(&changed_candidate), evidence.candidate_sha256());
        assert!(matches!(
            verify_mis_candidate_with_receipt(GRAPH, &changed_candidate, 2, VERIFIER, evidence.receipt_bytes()),
            Err(MisEvidenceError::ReceiptBytesMismatch)
        ));
        let mut changed_graph = GRAPH.to_vec();
        changed_graph[changed_graph.len() - 2] = b'2';
        assert!(verify_mis_candidate(&changed_graph, &bytes, 2, VERIFIER).is_err());
        let mut changed_receipt = evidence.receipt_bytes().to_vec();
        changed_receipt[0] ^= 1;
        assert!(matches!(
            verify_mis_candidate_with_receipt(GRAPH, &bytes, 2, VERIFIER, &changed_receipt),
            Err(MisEvidenceError::ReceiptBytesMismatch)
        ));
    }

    #[test]
    fn rejects_noncanonical_or_ambiguous_edge_serializations() {
        assert!(matches!(
            verify_mis_candidate(
                b"SYMTHEA-MIS-GRAPH-V1\\nvertices 4\\nedges 2\\n2 3\\n0 1\\n",
                &candidate(&[0, 2]),
                2,
                VERIFIER
            ),
            Err(MisEvidenceError::EdgesNotStrictlySorted)
        ));
        assert!(matches!(
            verify_mis_candidate(
                b"SYMTHEA-MIS-GRAPH-V1\\nvertices 4\\nedges 2\\n0 1\\n0 1\\n",
                &candidate(&[0, 2]),
                2,
                VERIFIER
            ),
            Err(MisEvidenceError::EdgesNotStrictlySorted)
        ));
        assert!(matches!(
            verify_mis_candidate(
                b"SYMTHEA-MIS-GRAPH-V1\\nvertices 4\\nedges 1\\n1 1\\n",
                &candidate(&[0, 2]),
                2,
                VERIFIER
            ),
            Err(MisEvidenceError::InvalidEdgeEndpoint)
        ));
    }

    #[test]
    fn rejects_noncanonical_graph_candidate_and_bad_digest() {
        assert!(matches!(
            verify_mis_candidate(b"SYMTHEA-MIS-GRAPH-V1\r\nvertices 4\r\nedges 2\r\n0 1\r\n2 3\r\n", &candidate(&[0, 2]), 2, VERIFIER),
            Err(MisEvidenceError::NonCanonicalLineEndings)
        ));
        assert!(matches!(
            verify_mis_candidate(GRAPH, b"SYMTHEA-MIS-CANDIDATE-V1\ncount 2\n1\n1\n", 2, VERIFIER),
            Err(MisEvidenceError::CandidateVerticesNotStrictlySorted)
        ));
        assert!(matches!(
            verify_mis_candidate(GRAPH, &candidate(&[0, 4]), 2, VERIFIER),
            Err(MisEvidenceError::CandidateVertexOutOfRange(4))
        ));
        assert!(matches!(
            verify_mis_candidate(GRAPH, &candidate(&[0, 2]), 2, &"C".repeat(64)),
            Err(MisEvidenceError::InvalidVerifierDigest)
        ));
    }
    #[test]
    fn exhaustive_small_graphs_match_independent_conflict_oracle() {
        // Exhaust every simple undirected graph through four vertices and every
        // candidate subset. This checks the verifier against a separate bit-mask
        // oracle rather than reusing its parser or conflict-count implementation.
        for vertex_count in 1usize..=4 {
            let mut possible_edges = Vec::new();
            for u in 0..vertex_count {
                for v in (u + 1)..vertex_count {
                    possible_edges.push((u, v));
                }
            }

            for edge_mask in 0usize..(1usize << possible_edges.len()) {
                let selected_edges: Vec<(usize, usize)> = possible_edges
                    .iter()
                    .enumerate()
                    .filter_map(|(bit, &edge)| ((edge_mask >> bit) & 1 == 1).then_some(edge))
                    .collect();
                let mut graph = format!(
                    "SYMTHEA-MIS-GRAPH-V1\\nvertices {vertex_count}\\nedges {}\\n",
                    selected_edges.len()
                );
                for (u, v) in &selected_edges {
                    graph.push_str(&format!("{u} {v}\\n"));
                }
                let graph = graph.into_bytes();

                for vertex_mask in 0usize..(1usize << vertex_count) {
                    let vertices: Vec<usize> = (0..vertex_count)
                        .filter(|vertex| (vertex_mask >> vertex) & 1 == 1)
                        .collect();
                    let candidate_bytes = candidate(&vertices);
                    let mut oracle_conflicts = 0usize;
                    for &(u, v) in &selected_edges {
                        if ((vertex_mask >> u) & 1 == 1) && ((vertex_mask >> v) & 1 == 1) {
                            oracle_conflicts += 1;
                        }
                    }

                    let evidence =
                        verify_mis_candidate(&graph, &candidate_bytes, 1, VERIFIER).unwrap();
                    assert_eq!(evidence.conflict_count(), oracle_conflicts);
                    assert_eq!(
                        evidence.verdict(),
                        if oracle_conflicts == 0 {
                            MisEvidenceVerdict::VerifiedIndependentSet
                        } else {
                            MisEvidenceVerdict::VerifiedConflict
                        }
                    );
                    assert_eq!(
                        evidence.reaches_target(),
                        oracle_conflicts == 0 && !vertices.is_empty()
                    );
                }
            }
        }
    }

}
