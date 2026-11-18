//! Small, deterministic, lossless byte codec used by CDL-Bench E0.
//!
//! This is a research reference implementation, not a production compression library.
//! The adaptive model is reconstructed from decoded symbols; no model state is hidden
//! in the payload. Every frame includes source and payload CRC-32 checksums.

pub mod bounded_error;
pub mod causal;
pub mod scm_predictor;

pub use bounded_error::{compress_bounded_i16, decompress_bounded_i16, BoundedError, BoundedPredictor};
pub mod numeric;

pub use numeric::{compress_i16, decompress_i16, NumericError, NumericTransform};

pub use causal::{e3_mechanism_splits, generate_fsm, generate_scm, generate_scm_with_config, FsmTrace, GeneratorError, ScmIntervention, ScmMechanismConfig, ScmResponse, ScmTrace};
pub use scm_predictor::{ScmFieldPredictor, ScmLagPredictor, ScmParentPredictor};

use std::fmt;

const MAGIC: &[u8; 4] = b"CDL1";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
pub const BYTE_SYMBOLS: usize = 256;
const SYMBOLS: usize = BYTE_SYMBOLS;
pub type ByteCdf = [u32; BYTE_SYMBOLS + 1];
const MAX_TOTAL: u32 = 1 << 15;

const TOP: u64 = 1 << 32;
const HALF: u64 = TOP >> 1;
const FIRST_QUARTER: u64 = TOP >> 2;
const THIRD_QUARTER: u64 = FIRST_QUARTER * 3;

/// The adaptive probability model serialized by its small stable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModelKind {
    /// One global 256-symbol frequency table.
    Order0 = 0,
    /// One frequency table per previous byte, plus a start-of-stream context.
    Order1 = 1,
}

/// Deterministic adaptive byte predictor used by the shared entropy coder.
///
/// The returned CDF must contain 257 monotonically nondecreasing entries,
/// start at zero, and have a positive final total. To encode a byte, its
/// interval is [cdf[symbol], cdf[symbol + 1]).
///
/// The decoder starts from the same predictor configuration and calls
/// observe() only after decoding each byte. Implementations must therefore
/// update deterministically from decoded symbols alone. The model ID must be
/// stable and unique within the caller's codec/container protocol; custom
/// models may use IDs 2..=255, while 0 and 1 are reserved by CDL's built-ins.
pub trait BytePredictor {
    fn model_id(&self) -> u8;
    fn cumulative_frequency(&self) -> ByteCdf;
    fn observe(&mut self, symbol: u8);
}

impl ModelKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "order0" | "0" => Some(Self::Order0),
            "order1" | "1" => Some(Self::Order1),
            _ => None,
        }
    }

    fn from_id(value: u8) -> Result<Self, CodecError> {
        match value {
            0 => Ok(Self::Order0),
            1 => Ok(Self::Order1),
            _ => Err(CodecError::UnsupportedModel(value)),
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }
}

/// Errors returned by the framing and decoding layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    TruncatedHeader,
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedModel(u8),
    UnsupportedFlags(u8),
    LengthOverflow,
    LengthMismatch,
    OutputLimitExceeded { declared: u64, limit: usize },
    PayloadChecksumMismatch,
    SourceChecksumMismatch,
    InvalidInterval,
    PredictorMismatch { frame_model: u8, supplied_model: u8 },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "truncated CDL frame header"),
            Self::BadMagic => write!(f, "invalid CDL frame magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported CDL frame version {v}"),
            Self::UnsupportedModel(v) => write!(f, "unsupported predictor model id {v}"),
            Self::UnsupportedFlags(v) => write!(f, "unsupported CDL frame flags {v:#04x}"),
            Self::LengthOverflow => write!(f, "CDL frame length overflows this platform"),
            Self::LengthMismatch => write!(f, "CDL frame length does not match its header"),
            Self::OutputLimitExceeded { declared, limit } => {
                write!(f, "declared output length {declared} exceeds limit {limit}")
            }
            Self::PayloadChecksumMismatch => write!(f, "CDL payload checksum mismatch"),
            Self::SourceChecksumMismatch => write!(f, "decoded source checksum mismatch"),
            Self::InvalidInterval => write!(f, "arithmetic coder interval became invalid"),
            Self::PredictorMismatch { frame_model, supplied_model } => write!(f, "frame predictor id {frame_model} does not match supplied predictor id {supplied_model}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Runtime model state. For Order1, context 256 is the start-of-stream context.
#[derive(Clone)]
struct AdaptiveByteModel {
    kind: ModelKind,
    counts: Vec<u16>,
    totals: Vec<u32>,
    context: usize,
}

impl AdaptiveByteModel {
    fn new(kind: ModelKind) -> Self {
        let context_count = match kind {
            ModelKind::Order0 => 1,
            ModelKind::Order1 => SYMBOLS + 1,
        };
        Self {
            kind,
            counts: vec![1; context_count * SYMBOLS],
            totals: vec![SYMBOLS as u32; context_count],
            context: if kind == ModelKind::Order1 { SYMBOLS } else { 0 },
        }
    }

    fn interval(&self, symbol: u8) -> (u64, u64, u64) {
        let base = self.context * SYMBOLS;
        let symbol_index = symbol as usize;
        let cumulative_low: u64 = self.counts[base..base + symbol_index]
            .iter()
            .map(|&count| count as u64)
            .sum();
        let frequency = self.counts[base + symbol_index] as u64;
        let total = self.totals[self.context] as u64;
        (cumulative_low, cumulative_low + frequency, total)
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        let base = self.context * SYMBOLS;
        let mut cdf = [0u32; BYTE_SYMBOLS + 1];
        for symbol in 0..SYMBOLS {
            cdf[symbol + 1] = cdf[symbol] + self.counts[base + symbol] as u32;
        }
        cdf
    }

    fn update(&mut self, symbol: u8) {
        let context = self.context;
        let base = context * SYMBOLS;
        if self.totals[context] >= MAX_TOTAL {
            let slice = &mut self.counts[base..base + SYMBOLS];
            let mut new_total = 0u32;
            for count in slice.iter_mut() {
                *count = ((*count as u32 + 1) / 2).max(1) as u16;
                new_total += *count as u32;
            }
            self.totals[context] = new_total;
        }

        let index = base + symbol as usize;
        self.counts[index] += 1;
        self.totals[context] += 1;

        if self.kind == ModelKind::Order1 {
            self.context = symbol as usize;
        }
    }

    /// Bytes occupied by the model's frequency and total arrays, excluding Vec metadata.
    fn state_bytes(&self) -> usize {
        self.counts.len() * std::mem::size_of::<u16>()
            + self.totals.len() * std::mem::size_of::<u32>()
            + std::mem::size_of::<usize>()
    }
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn write_bit(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used != 0 {
            self.current <<= 8 - self.used;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_index: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_index: 0 }
    }

    // Arithmetic coding has a finite termination code. Zero-extension here is safe
    // because the outer frame verifies exact byte length and payload checksum first.
    fn read_bit(&mut self) -> bool {
        let byte_index = self.bit_index / 8;
        let bit_offset = 7 - (self.bit_index % 8);
        self.bit_index += 1;
        self.bytes
            .get(byte_index)
            .map(|byte| ((byte >> bit_offset) & 1) != 0)
            .unwrap_or(false)
    }
}

struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending: u64,
    writer: BitWriter,
}

impl ArithmeticEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            high: TOP - 1,
            pending: 0,
            writer: BitWriter::default(),
        }
    }

    fn emit_with_pending(&mut self, bit: bool) {
        self.writer.write_bit(bit);
        for _ in 0..self.pending {
            self.writer.write_bit(!bit);
        }
        self.pending = 0;
    }

    fn encode(
        &mut self,
        cumulative_low: u64,
        cumulative_high: u64,
        total: u64,
    ) -> Result<(), CodecError> {
        if total == 0 || cumulative_low >= cumulative_high || cumulative_high > total {
            return Err(CodecError::InvalidInterval);
        }

        let range = self.high - self.low + 1;
        self.high = self.low + (range * cumulative_high) / total - 1;
        self.low += (range * cumulative_low) / total;
        if self.low > self.high {
            return Err(CodecError::InvalidInterval);
        }

        loop {
            if self.high < HALF {
                self.emit_with_pending(false);
            } else if self.low >= HALF {
                self.emit_with_pending(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QUARTER && self.high < THIRD_QUARTER {
                self.pending += 1;
                self.low -= FIRST_QUARTER;
                self.high -= FIRST_QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < FIRST_QUARTER {
            self.emit_with_pending(false);
        } else {
            self.emit_with_pending(true);
        }
        self.writer.finish()
    }
}

struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    reader: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(payload: &'a [u8]) -> Self {
        let mut reader = BitReader::new(payload);
        let mut value = 0u64;
        for _ in 0..32 {
            value = (value << 1) | u64::from(reader.read_bit());
        }
        Self {
            low: 0,
            high: TOP - 1,
            value,
            reader,
        }
    }

    fn decode(&mut self, model: &AdaptiveByteModel) -> Result<u8, CodecError> {
        let cdf = model.cumulative_frequency();
        self.decode_with_cdf(&cdf)
    }

    fn decode_with_cdf(&mut self, cdf: &ByteCdf) -> Result<u8, CodecError> {
        let total = validate_cdf(cdf)?;
        if total == 0 {
            return Err(CodecError::InvalidInterval);
        }
        let range = self.high - self.low + 1;
        let scaled = (((self.value - self.low + 1) * total - 1) / range) as u32;

        let mut selected = None;
        for symbol in 0..SYMBOLS {
            let cumulative_low = cdf[symbol] as u64;
            let cumulative_high = cdf[symbol + 1] as u64;
            if (scaled as u64) >= cumulative_low && (scaled as u64) < cumulative_high {
                selected = Some((symbol as u8, cumulative_low, cumulative_high));
                break;
            }
        }
        let (symbol, cumulative_low, cumulative_high) =
            selected.ok_or(CodecError::InvalidInterval)?;

        self.high = self.low + (range * cumulative_high) / total - 1;
        self.low += (range * cumulative_low) / total;
        if self.low > self.high || self.value < self.low || self.value > self.high {
            return Err(CodecError::InvalidInterval);
        }

        loop {
            if self.high < HALF {
                // No translation is needed.
            } else if self.low >= HALF {
                self.value -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QUARTER && self.high < THIRD_QUARTER {
                self.value -= FIRST_QUARTER;
                self.low -= FIRST_QUARTER;
                self.high -= FIRST_QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | u64::from(self.reader.read_bit());
        }
        Ok(symbol)
    }
}

/// Estimate the adaptive model's frequency-array and context storage in bytes.
/// This excludes allocator metadata and the codec executable.
pub fn predictor_state_bytes(kind: ModelKind) -> usize {
    let contexts = match kind {
        ModelKind::Order0 => 1,
        ModelKind::Order1 => SYMBOLS + 1,
    };
    contexts * SYMBOLS * std::mem::size_of::<u16>()
        + contexts * std::mem::size_of::<u32>()
        + std::mem::size_of::<usize>()
}

impl BytePredictor for AdaptiveByteModel {
    fn model_id(&self) -> u8 {
        self.kind.id()
    }

    fn cumulative_frequency(&self) -> ByteCdf {
        AdaptiveByteModel::cumulative_frequency(self)
    }

    fn observe(&mut self, symbol: u8) {
        self.update(symbol);
    }
}

fn validate_cdf(cdf: &ByteCdf) -> Result<u64, CodecError> {
    if cdf[0] != 0 || cdf[BYTE_SYMBOLS] == 0 {
        return Err(CodecError::InvalidInterval);
    }
    if cdf.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(CodecError::InvalidInterval);
    }
    Ok(cdf[BYTE_SYMBOLS] as u64)
}

/// CRC-32/ISO-HDLC, implemented locally so the reference crate has no dependencies.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Encode bytes using a caller-supplied deterministic adaptive predictor.
///
/// The predictor is mutated as symbols are consumed. The resulting CDL1 frame
/// carries the predictor's model ID, the source length, and both source and
/// payload CRC-32 values. The decoder must be supplied a fresh predictor with
/// the same initial configuration. Custom model configuration bytes are not
/// embedded and remain the caller's side-information/accounting responsibility.
pub fn compress_with_predictor<P: BytePredictor>(
    input: &[u8],
    predictor: &mut P,
) -> Result<Vec<u8>, CodecError> {
    let model_id = predictor.model_id();
    if model_id < 2 {
        return Err(CodecError::UnsupportedModel(model_id));
    }
    let mut coder = ArithmeticEncoder::new();

    for &symbol in input {
        let cdf = predictor.cumulative_frequency();
        let total = validate_cdf(&cdf)?;
        let low = cdf[symbol as usize] as u64;
        let high = cdf[symbol as usize + 1] as u64;
        coder.encode(low, high, total)?;
        predictor.observe(symbol);
    }

    let payload = coder.finish();
    let raw_len = u64::try_from(input.len()).map_err(|_| CodecError::LengthOverflow)?;
    let payload_len = u64::try_from(payload.len()).map_err(|_| CodecError::LengthOverflow)?;
    let mut frame = Vec::with_capacity(
        HEADER_LEN
            .checked_add(payload.len())
            .ok_or(CodecError::LengthOverflow)?,
    );
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION.to_le_bytes());
    frame.push(model_id);
    frame.push(0);
    frame.extend_from_slice(&raw_len.to_le_bytes());
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&crc32(input).to_le_bytes());
    frame.extend_from_slice(&crc32(&payload).to_le_bytes());
    debug_assert_eq!(frame.len(), HEADER_LEN);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decode a custom-predictor CDL1 frame using a fresh matching predictor.
///
/// Custom model parameters and predictor executable size are not carried in
/// this frame. The caller must provide a matching predictor and account for
/// any required model, checkpoint, configuration, or initial state.
pub fn decompress_with_predictor<P: BytePredictor>(
    frame: &[u8],
    max_output_bytes: usize,
    predictor: &mut P,
) -> Result<Vec<u8>, CodecError> {
    if frame.len() < HEADER_LEN {
        return Err(CodecError::TruncatedHeader);
    }
    if &frame[0..4] != MAGIC {
        return Err(CodecError::BadMagic);
    }
    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(CodecError::UnsupportedVersion(version));
    }
    let supplied_model = predictor.model_id();
    if supplied_model < 2 {
        return Err(CodecError::UnsupportedModel(supplied_model));
    }
    let frame_model = frame[6];
    if frame_model != supplied_model {
        return Err(CodecError::PredictorMismatch {
            frame_model,
            supplied_model,
        });
    }
    if frame[7] != 0 {
        return Err(CodecError::UnsupportedFlags(frame[7]));
    }

    let raw_len = u64::from_le_bytes(
        frame[8..16].try_into().map_err(|_| CodecError::LengthMismatch)?,
    );
    let payload_len = u64::from_le_bytes(
        frame[16..24].try_into().map_err(|_| CodecError::LengthMismatch)?,
    );
    let source_crc = u32::from_le_bytes(
        frame[24..28].try_into().map_err(|_| CodecError::LengthMismatch)?,
    );
    let payload_crc = u32::from_le_bytes(
        frame[28..32].try_into().map_err(|_| CodecError::LengthMismatch)?,
    );
    if raw_len > max_output_bytes as u64 {
        return Err(CodecError::OutputLimitExceeded {
            declared: raw_len,
            limit: max_output_bytes,
        });
    }
    let raw_len = usize::try_from(raw_len).map_err(|_| CodecError::LengthOverflow)?;
    let payload_len = usize::try_from(payload_len).map_err(|_| CodecError::LengthOverflow)?;
    let expected_len = HEADER_LEN
        .checked_add(payload_len)
        .ok_or(CodecError::LengthOverflow)?;
    if expected_len != frame.len() {
        return Err(CodecError::LengthMismatch);
    }
    let payload = &frame[HEADER_LEN..];
    if crc32(payload) != payload_crc {
        return Err(CodecError::PayloadChecksumMismatch);
    }

    let mut decoder = ArithmeticDecoder::new(payload);
    let mut output = Vec::with_capacity(raw_len);
    for _ in 0..raw_len {
        let cdf = predictor.cumulative_frequency();
        let _ = validate_cdf(&cdf)?;
        let symbol = decoder.decode_with_cdf(&cdf)?;
        output.push(symbol);
        predictor.observe(symbol);
    }
    if crc32(&output) != source_crc {
        return Err(CodecError::SourceChecksumMismatch);
    }
    Ok(output)
}

/// Encode a byte slice into a versioned, checksummed CDL frame.
pub fn compress(input: &[u8], kind: ModelKind) -> Result<Vec<u8>, CodecError> {
    let mut model = AdaptiveByteModel::new(kind);
    let mut coder = ArithmeticEncoder::new();

    for &symbol in input {
        let (low, high, total) = model.interval(symbol);
        coder.encode(low, high, total)?;
        model.update(symbol);
    }

    let payload = coder.finish();
    let raw_len = u64::try_from(input.len()).map_err(|_| CodecError::LengthOverflow)?;
    let payload_len = u64::try_from(payload.len()).map_err(|_| CodecError::LengthOverflow);

    let mut frame = Vec::with_capacity(
        HEADER_LEN
            .checked_add(payload.len())
            .ok_or(CodecError::LengthOverflow)?,
    );
    frame.extend_from_slice(MAGIC);
    frame.extend_from_slice(&VERSION.to_le_bytes());
    frame.push(kind.id());
    frame.push(0); // flags reserved; version 1 requires zero.
    frame.extend_from_slice(&raw_len.to_le_bytes());
    frame.extend_from_slice(&payload_len?.to_le_bytes());
    frame.extend_from_slice(&crc32(input).to_le_bytes());
    frame.extend_from_slice(&crc32(&payload).to_le_bytes());
    debug_assert_eq!(frame.len(), HEADER_LEN);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decode a CDL frame, rejecting malformed framing, checksum mismatches and oversized output.
pub fn decompress(frame: &[u8], max_output_bytes: usize) -> Result<Vec<u8>, CodecError> {
    if frame.len() < HEADER_LEN {
        return Err(CodecError::TruncatedHeader);
    }
    if &frame[0..4] != MAGIC {
        return Err(CodecError::BadMagic);
    }

    let version = u16::from_le_bytes([frame[4], frame[5]]);
    if version != VERSION {
        return Err(CodecError::UnsupportedVersion(version));
    }

    let kind = ModelKind::from_id(frame[6])?;
    if frame[7] != 0 {
        return Err(CodecError::UnsupportedFlags(frame[7]));
    }

    let raw_len = u64::from_le_bytes(frame[8..16].try_into().map_err(|_| CodecError::LengthMismatch)?);
    let payload_len = u64::from_le_bytes(frame[16..24].try_into().map_err(|_| CodecError::LengthMismatch)?);
    let source_crc = u32::from_le_bytes(frame[24..28].try_into().map_err(|_| CodecError::LengthMismatch)?);
    let payload_crc = u32::from_le_bytes(frame[28..32].try_into().map_err(|_| CodecError::LengthMismatch)?);

    if raw_len > max_output_bytes as u64 {
        return Err(CodecError::OutputLimitExceeded {
            declared: raw_len,
            limit: max_output_bytes,
        });
    }

    let raw_len_usize = usize::try_from(raw_len).map_err(|_| CodecError::LengthOverflow)?;
    let payload_len_usize = usize::try_from(payload_len).map_err(|_| CodecError::LengthOverflow)?;
    let expected_len = HEADER_LEN
        .checked_add(payload_len_usize)
        .ok_or(CodecError::LengthOverflow)?;
    if expected_len != frame.len() {
        return Err(CodecError::LengthMismatch);
    }

    let payload = &frame[HEADER_LEN..];
    if crc32(payload) != payload_crc {
        return Err(CodecError::PayloadChecksumMismatch);
    }

    let mut model = AdaptiveByteModel::new(kind);
    let mut decoder = ArithmeticDecoder::new(payload);
    let mut output = Vec::with_capacity(raw_len_usize);
    for _ in 0..raw_len_usize {
        let symbol = decoder.decode(&model)?;
        output.push(symbol);
        model.update(symbol);
    }

    if crc32(&output) != source_crc {
        return Err(CodecError::SourceChecksumMismatch);
    }
    Ok(output)
}

/// Deterministic E0 source families used to exercise the coder and adaptive baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E0Kind {
    IidUniform,
    IidSkewed,
    MarkovFirstOrder,
    Periodic,
    Bursty,
    LongRuns,
    Incompressible,
}

impl E0Kind {
    pub const ALL: [Self; 7] = [
        Self::IidUniform,
        Self::IidSkewed,
        Self::MarkovFirstOrder,
        Self::Periodic,
        Self::Bursty,
        Self::LongRuns,
        Self::Incompressible,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::IidUniform => "iid-uniform",
            Self::IidSkewed => "iid-skewed",
            Self::MarkovFirstOrder => "markov-order1",
            Self::Periodic => "periodic",
            Self::Bursty => "bursty",
            Self::LongRuns => "long-runs",
            Self::Incompressible => "incompressible",
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Generate one deterministic E0 stream without external random-number dependencies.
pub fn generate_e0(kind: E0Kind, len: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(len);

    match kind {
        E0Kind::IidUniform | E0Kind::Incompressible => {
            for _ in 0..len {
                out.push(rng.next() as u8);
            }
        }
        E0Kind::IidSkewed => {
            for _ in 0..len {
                let draw = rng.next() % 1000;
                out.push(match draw {
                    0..=899 => b'A',
                    900..=949 => b'B',
                    950..=979 => b'C',
                    _ => (rng.next() >> 8) as u8,
                });
            }
        }
        E0Kind::MarkovFirstOrder => {
            let alphabet = [b'A', b'B', b'C', b'D'];
            let mut state = (rng.next() % 4) as usize;
            for _ in 0..len {
                out.push(alphabet[state]);
                if rng.next() % 100 < 82 {
                    // Persist the current state; otherwise transition to another state.
                } else {
                    let mut next = (rng.next() % 3) as usize;
                    if next >= state {
                        next += 1;
                    }
                    state = next;
                }
            }
        }
        E0Kind::Periodic => {
            const PATTERN: &[u8] = b"CDL-BENCH::CAUSAL::";
            for index in 0..len {
                out.push(PATTERN[index % PATTERN.len()]);
            }
        }
        E0Kind::Bursty => {
            let mut block = 0u8;
            for index in 0..len {
                if index % 4096 == 0 {
                    block = (rng.next() % 8) as u8;
                }
                let draw = rng.next() % 100;
                out.push(if draw < 94 {
                    b'A' + block
                } else {
                    (rng.next() >> 16) as u8
                });
            }
        }
        E0Kind::LongRuns => {
            while out.len() < len {
                let run = 1 + (rng.next() % 2048) as usize;
                let symbol = b'A' + (rng.next() % 8) as u8;
                let remaining = len - out.len();
                out.extend(std::iter::repeat(symbol).take(run.min(remaining)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bytes() -> Vec<Vec<u8>> {
        let all_symbols: Vec<u8> = (0..=255).collect();
        let mut x = SplitMix64::new(0xCD1_B_EA_5E);
        let random: Vec<u8> = (0..32_768).map(|_| x.next() as u8).collect();
        vec![
            Vec::new(),
            vec![0xA5],
            vec![b'A'; 4096],
            all_symbols,
            b"abracadabra abracadabra abracadabra".to_vec(),
            generate_e0(E0Kind::MarkovFirstOrder, 8192, 42),
            generate_e0(E0Kind::Bursty, 8192, 43),
            random,
        ]
    }

    #[derive(Clone)]
    struct TestAdaptiveModel {
        counts: [u32; BYTE_SYMBOLS],
    }

    impl TestAdaptiveModel {
        fn new() -> Self {
            Self {
                counts: [1; BYTE_SYMBOLS],
            }
        }
    }

    impl BytePredictor for TestAdaptiveModel {
        fn model_id(&self) -> u8 {
            42
        }

        fn cumulative_frequency(&self) -> ByteCdf {
            let mut cdf = [0u32; BYTE_SYMBOLS + 1];
            for index in 0..BYTE_SYMBOLS {
                cdf[index + 1] = cdf[index] + self.counts[index];
            }
            cdf
        }

        fn observe(&mut self, symbol: u8) {
            self.counts[symbol as usize] += 1;
            if self.counts.iter().sum::<u32>() >= MAX_TOTAL {
                for count in &mut self.counts {
                    *count = (*count + 1) / 2;
                }
            }
        }
    }

    struct InvalidCdfModel;

    impl BytePredictor for InvalidCdfModel {
        fn model_id(&self) -> u8 {
            43
        }

        fn cumulative_frequency(&self) -> ByteCdf {
            [0; BYTE_SYMBOLS + 1]
        }

        fn observe(&mut self, _symbol: u8) {}
    }

    #[test]
    fn caller_supplied_predictor_round_trips_with_shared_coder() {
        let input = generate_e0(E0Kind::MarkovFirstOrder, 32_768, 0xC0_DE);
        let mut encoder_model = TestAdaptiveModel::new();
        let frame = compress_with_predictor(&input, &mut encoder_model).unwrap();
        assert_eq!(frame[6], 42);

        let mut decoder_model = TestAdaptiveModel::new();
        let decoded = decompress_with_predictor(&frame, 1_000_000, &mut decoder_model).unwrap();
        assert_eq!(decoded, input);

        let wrong_model = TestAdaptiveModel::new();
        let mut wrong_id = WrongModelId(wrong_model);
        assert!(matches!(
            decompress_with_predictor(&frame, 1_000_000, &mut wrong_id),
            Err(CodecError::PredictorMismatch { .. })
        ));
    }

    struct WrongModelId(TestAdaptiveModel);

    impl BytePredictor for WrongModelId {
        fn model_id(&self) -> u8 {
            44
        }
        fn cumulative_frequency(&self) -> ByteCdf {
            self.0.cumulative_frequency()
        }
        fn observe(&mut self, symbol: u8) {
            self.0.observe(symbol);
        }
    }

    #[test]
    fn invalid_predictor_cdf_rejects_before_coding() {
        assert!(matches!(
            compress_with_predictor(b"x", &mut InvalidCdfModel),
            Err(CodecError::InvalidInterval)
        ));
    }

    #[test]
    fn exact_round_trip_for_all_models_and_fixture_families() {
        for kind in [ModelKind::Order0, ModelKind::Order1] {
            for input in sample_bytes() {
                let frame = compress(&input, kind).expect("encode");
                let decoded = decompress(&frame, 1_000_000).expect("decode");
                assert_eq!(decoded, input, "round trip failed for {kind:?}");
            }
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let input = generate_e0(E0Kind::IidSkewed, 16_384, 99);
        assert_eq!(
            compress(&input, ModelKind::Order0).unwrap(),
            compress(&input, ModelKind::Order0).unwrap()
        );
        assert_eq!(
            compress(&input, ModelKind::Order1).unwrap(),
            compress(&input, ModelKind::Order1).unwrap()
        );
    }

    #[test]
    fn every_truncated_prefix_is_rejected() {
        let frame = compress(
            &generate_e0(E0Kind::MarkovFirstOrder, 2048, 7),
            ModelKind::Order1,
        )
        .unwrap();
        for end in 0..frame.len() {
            assert!(
                decompress(&frame[..end], 1_000_000).is_err(),
                "truncated prefix of length {end} was accepted"
            );
        }
    }

    #[test]
    fn every_single_bit_mutation_is_rejected() {
        let frame = compress(
            &generate_e0(E0Kind::IidSkewed, 1024, 1234),
            ModelKind::Order1,
        )
        .unwrap();
        for index in 0..frame.len() {
            for bit in 0..8 {
                let mut mutated = frame.clone();
                mutated[index] ^= 1 << bit;
                assert!(
                    decompress(&mutated, 1_000_000).is_err(),
                    "mutation at byte {index}, bit {bit} was accepted"
                );
            }
        }
    }

    #[test]
    fn output_limit_is_enforced_before_allocation() {
        let frame = compress(b"this is a bounded output test", ModelKind::Order0).unwrap();
        assert!(matches!(
            decompress(&frame, 4),
            Err(CodecError::OutputLimitExceeded { .. })
        ));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut frame = compress(b"trailing bytes", ModelKind::Order0).unwrap();
        frame.push(0);
        assert_eq!(decompress(&frame, 1024), Err(CodecError::LengthMismatch));
    }

    #[test]
    fn all_e0_generators_are_deterministic_and_sized() {
        for kind in E0Kind::ALL {
            let a = generate_e0(kind, 10_000, 11);
            let b = generate_e0(kind, 10_000, 11);
            assert_eq!(a, b);
            assert_eq!(a.len(), 10_000);
        }
    }

    #[test]
    fn crc32_matches_standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
