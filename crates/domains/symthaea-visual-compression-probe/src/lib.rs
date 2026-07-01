//! Symthaea Visual Compression Probe.
//!
//! This crate is a dependency-light experiment in **cognitive visual compression**:
//! sparse spectral reconstruction, HDC signatures for query-without-decode, and
//! topology fingerprints for durable structure.
//!
//! The first supported input/output image format is grayscale PGM (`P2` or `P5`).

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

/// Result type used by the crate.
pub type Result<T> = std::result::Result<T, ProbeError>;

/// Current readable prototype packet magic.
pub const SVMP_MAGIC: &str = "SVMP 0.1";

/// Human-readable crate experiment version.
pub const PROBE_EXPERIMENT_VERSION: &str = "0.1.0-alpha.5";

/// Errors returned by the probe crate.
#[derive(Debug)]
pub enum ProbeError {
    Io(io::Error),
    InvalidFormat(String),
    InvalidArgs(String),
}

impl Display for ProbeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::InvalidArgs(msg) => write!(f, "invalid arguments: {msg}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<io::Error> for ProbeError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Grayscale image with normalized samples in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl GrayImage {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ProbeError::InvalidArgs(
                "image dimensions must be nonzero".into(),
            ));
        }
        if pixels.len() != width * height {
            return Err(ProbeError::InvalidArgs(format!(
                "pixel count {} does not match {}x{}",
                pixels.len(),
                width,
                height
            )));
        }
        let pixels = pixels.into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.pixels[y * self.width + x] = value.clamp(0.0, 1.0);
    }

    pub fn read_pgm(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        parse_pgm(&bytes)
    }

    pub fn write_pgm(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut out = Vec::with_capacity(self.width * self.height + 64);
        write!(&mut out, "P5\n{} {}\n255\n", self.width, self.height).unwrap();
        for &px in &self.pixels {
            let byte = (px.clamp(0.0, 1.0) * 255.0).round() as u8;
            out.push(byte);
        }
        fs::write(path, out)?;
        Ok(())
    }
}

/// Parse P2/P5 PGM with comments.
pub fn parse_pgm(bytes: &[u8]) -> Result<GrayImage> {
    let mut cursor = 0usize;
    let magic = next_token(bytes, &mut cursor)
        .ok_or_else(|| ProbeError::InvalidFormat("missing magic".into()))?;
    let width: usize = parse_token(next_token(bytes, &mut cursor), "width")?;
    let height: usize = parse_token(next_token(bytes, &mut cursor), "height")?;
    let max_value: usize = parse_token(next_token(bytes, &mut cursor), "max value")?;
    if max_value == 0 || max_value > 65535 {
        return Err(ProbeError::InvalidFormat(
            "PGM max value must be 1..65535".into(),
        ));
    }

    match magic.as_str() {
        "P2" => {
            let mut pixels = Vec::with_capacity(width * height);
            for _ in 0..(width * height) {
                let value: usize = parse_token(next_token(bytes, &mut cursor), "P2 pixel")?;
                pixels.push((value as f32 / max_value as f32).clamp(0.0, 1.0));
            }
            GrayImage::new(width, height, pixels)
        }
        "P5" => {
            // P5 binary data begins after a single whitespace delimiter following max_value.
            // Do not call skip_ws_and_comments here: valid pixel bytes may themselves be
            // ASCII whitespace values such as 0x0a or 0x20.
            if cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let expected = width * height;
            let remaining = bytes.len().saturating_sub(cursor);
            if max_value <= 255 {
                if remaining < expected {
                    return Err(ProbeError::InvalidFormat(format!(
                        "P5 data too short: expected {expected}, found {remaining}"
                    )));
                }
                let pixels = bytes[cursor..cursor + expected]
                    .iter()
                    .map(|&b| b as f32 / max_value as f32)
                    .collect();
                GrayImage::new(width, height, pixels)
            } else {
                let expected_bytes = expected * 2;
                if remaining < expected_bytes {
                    return Err(ProbeError::InvalidFormat(format!(
                        "P5 16-bit data too short: expected {expected_bytes}, found {remaining}"
                    )));
                }
                let mut pixels = Vec::with_capacity(expected);
                for chunk in bytes[cursor..cursor + expected_bytes].chunks_exact(2) {
                    let value = u16::from_be_bytes([chunk[0], chunk[1]]) as usize;
                    pixels.push((value as f32 / max_value as f32).clamp(0.0, 1.0));
                }
                GrayImage::new(width, height, pixels)
            }
        }
        other => Err(ProbeError::InvalidFormat(format!(
            "unsupported PGM magic {other}"
        ))),
    }
}

fn parse_token<T: std::str::FromStr>(token: Option<String>, name: &str) -> Result<T> {
    let token = token.ok_or_else(|| ProbeError::InvalidFormat(format!("missing {name}")))?;
    token
        .parse::<T>()
        .map_err(|_| ProbeError::InvalidFormat(format!("could not parse {name}: {token}")))
}

fn next_token(bytes: &[u8], cursor: &mut usize) -> Option<String> {
    skip_ws_and_comments(bytes, cursor);
    if *cursor >= bytes.len() {
        return None;
    }
    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() && bytes[*cursor] != b'#' {
        *cursor += 1;
    }
    Some(String::from_utf8_lossy(&bytes[start..*cursor]).to_string())
}

fn skip_ws_and_comments(bytes: &[u8], cursor: &mut usize) {
    loop {
        while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }
        if *cursor < bytes.len() && bytes[*cursor] == b'#' {
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
        } else {
            break;
        }
    }
}

/// Explicit encode parameters used by integration callers.
///
/// This is intentionally small and dependency-light.  It exists so Symthaea
/// perception/memory code can pass one typed value instead of a loose pair of
/// CLI-style integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodingParams {
    pub block_size: usize,
    pub keep_coeffs: usize,
    pub topology_levels: usize,
}

impl Default for EncodingParams {
    fn default() -> Self {
        Self {
            block_size: 8,
            keep_coeffs: 10,
            topology_levels: 16,
        }
    }
}

impl EncodingParams {
    pub fn new(block_size: usize, keep_coeffs: usize, topology_levels: usize) -> Result<Self> {
        let params = Self {
            block_size,
            keep_coeffs,
            topology_levels,
        };
        params.validate()?;
        Ok(params)
    }

    pub fn validate(&self) -> Result<()> {
        if self.block_size == 0 {
            return Err(ProbeError::InvalidArgs("block_size must be nonzero".into()));
        }
        if self.keep_coeffs == 0 {
            return Err(ProbeError::InvalidArgs(
                "keep_coeffs must be nonzero".into(),
            ));
        }
        if self.keep_coeffs > self.block_size * self.block_size {
            return Err(ProbeError::InvalidArgs(format!(
                "keep_coeffs {} exceeds block capacity {}",
                self.keep_coeffs,
                self.block_size * self.block_size
            )));
        }
        if self.topology_levels == 0 {
            return Err(ProbeError::InvalidArgs(
                "topology_levels must be nonzero".into(),
            ));
        }
        Ok(())
    }

    pub fn to_pretty_text(&self) -> String {
        format!(
            "block_size={}\nkeep_coeffs={}\ntopology_levels={}",
            self.block_size, self.keep_coeffs, self.topology_levels
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"block_size\":{},\"keep_coeffs\":{},\"topology_levels\":{}}}",
            self.block_size, self.keep_coeffs, self.topology_levels
        )
    }
}

/// One sparse spectral coefficient in one image block.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseCoeff {
    pub index: usize,
    pub value: f32,
}

/// Encoded block.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedBlock {
    pub block_x: usize,
    pub block_y: usize,
    pub coeffs: Vec<SparseCoeff>,
    pub hdc: BinaryHv,
}

/// Topology summary at one threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct TopologySample {
    pub threshold: f32,
    pub beta0: usize,
    pub beta1: usize,
}

/// Summary metrics for one visual memory packet.
#[derive(Clone, Debug, PartialEq)]
pub struct PacketMetrics {
    pub width: usize,
    pub height: usize,
    pub block_size: usize,
    pub keep_coeffs: usize,
    pub blocks: usize,
    pub dense_coefficients: usize,
    pub stored_coefficients: usize,
    pub nonzero_hdc_bits: u32,
    pub topology_samples: usize,
    pub raw_grayscale_bytes: usize,
    pub prototype_text_bytes: usize,
    pub coefficient_density: f32,
    pub text_to_raw_ratio: f32,
}

impl PacketMetrics {
    pub fn to_pretty_text(&self) -> String {
        format!(
            "dims={}x{}\nblock_size={}\nkeep_coeffs={}\nblocks={}\ndense_coefficients={}\nstored_coefficients={}\ncoefficient_density={:.6}\nnonzero_hdc_bits={}\ntopology_samples={}\nraw_grayscale_bytes={}\nprototype_text_bytes={}\ntext_to_raw_ratio={:.6}",
            self.width,
            self.height,
            self.block_size,
            self.keep_coeffs,
            self.blocks,
            self.dense_coefficients,
            self.stored_coefficients,
            self.coefficient_density,
            self.nonzero_hdc_bits,
            self.topology_samples,
            self.raw_grayscale_bytes,
            self.prototype_text_bytes,
            self.text_to_raw_ratio,
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"width\":{},\"height\":{},\"block_size\":{},\"keep_coeffs\":{},\"blocks\":{},\"dense_coefficients\":{},\"stored_coefficients\":{},\"coefficient_density\":{:.8},\"nonzero_hdc_bits\":{},\"topology_samples\":{},\"raw_grayscale_bytes\":{},\"prototype_text_bytes\":{},\"text_to_raw_ratio\":{:.8}}}",
            self.width,
            self.height,
            self.block_size,
            self.keep_coeffs,
            self.blocks,
            self.dense_coefficients,
            self.stored_coefficients,
            self.coefficient_density,
            self.nonzero_hdc_bits,
            self.topology_samples,
            self.raw_grayscale_bytes,
            self.prototype_text_bytes,
            self.text_to_raw_ratio,
        )
    }
}

/// Similarity report comparing two cognitive visual packets.
#[derive(Clone, Debug, PartialEq)]
pub struct PacketSimilarity {
    pub hdc_similarity: f32,
    pub topology_similarity: f32,
    pub combined_similarity: f32,
}

impl PacketSimilarity {
    pub fn to_pretty_text(&self) -> String {
        format!(
            "hdc_similarity={:.6}\ntopology_similarity={:.6}\ncombined_similarity={:.6}",
            self.hdc_similarity, self.topology_similarity, self.combined_similarity
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"hdc_similarity\":{:.8},\"topology_similarity\":{:.8},\"combined_similarity\":{:.8}}}",
            self.hdc_similarity, self.topology_similarity, self.combined_similarity
        )
    }
}

/// Encode/decode benchmark result for one image and parameter set.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub metrics: PacketMetrics,
    pub mse: f32,
    pub psnr_db: f32,
    pub self_similarity: PacketSimilarity,
}

impl BenchmarkReport {
    pub fn to_pretty_text(&self) -> String {
        format!(
            "{}\nmse={:.8}\npsnr_db={:.6}\n{}",
            self.metrics.to_pretty_text(),
            self.mse,
            self.psnr_db,
            self.self_similarity.to_pretty_text(),
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"metrics\":{},\"mse\":{:.8},\"psnr_db\":{},\"self_similarity\":{}}}",
            self.metrics.to_json(),
            self.mse,
            json_float(self.psnr_db),
            self.self_similarity.to_json(),
        )
    }
}

fn json_float(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.8}")
    } else if value.is_infinite() && value.is_sign_positive() {
        "\"inf\"".to_string()
    } else if value.is_infinite() && value.is_sign_negative() {
        "\"-inf\"".to_string()
    } else {
        "\"nan\"".to_string()
    }
}

/// High-level visual-memory summary for a scan.
///
/// This is not a scientific proof of perception. It is a compact report that
/// helps downstream memory/retrieval systems decide how a packet should be
/// indexed, compared, and triaged.
#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveScanSummary {
    pub packet_hash64: u64,
    pub image_hash64: u64,
    pub edge_energy: f32,
    pub structural_density: f32,
    pub topology_complexity: f32,
    pub hdc_activation_ratio: f32,
    pub reconstruction_psnr_db: f32,
    pub memory_class: String,
    pub params: EncodingParams,
}

impl CognitiveScanSummary {
    pub fn to_pretty_text(&self) -> String {
        format!(
            "packet_hash64={:016x}\nimage_hash64={:016x}\nedge_energy={:.8}\nstructural_density={:.8}\ntopology_complexity={:.8}\nhdc_activation_ratio={:.8}\nreconstruction_psnr_db={}\nmemory_class={}\n{}",
            self.packet_hash64,
            self.image_hash64,
            self.edge_energy,
            self.structural_density,
            self.topology_complexity,
            self.hdc_activation_ratio,
            json_float(self.reconstruction_psnr_db),
            self.memory_class,
            self.params.to_pretty_text(),
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"packet_hash64\":\"{:016x}\",\"image_hash64\":\"{:016x}\",\"edge_energy\":{:.8},\"structural_density\":{:.8},\"topology_complexity\":{:.8},\"hdc_activation_ratio\":{:.8},\"reconstruction_psnr_db\":{},\"memory_class\":\"{}\",\"params\":{}}}",
            self.packet_hash64,
            self.image_hash64,
            self.edge_energy,
            self.structural_density,
            self.topology_complexity,
            self.hdc_activation_ratio,
            json_float(self.reconstruction_psnr_db),
            json_escape_str(&self.memory_class),
            self.params.to_json(),
        )
    }
}

/// Ranked packet result for query-without-decode retrieval.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedPacket {
    pub label: String,
    pub packet_hash64: u64,
    pub similarity: PacketSimilarity,
}

impl RankedPacket {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"label\":\"{}\",\"packet_hash64\":\"{:016x}\",\"similarity\":{}}}",
            json_escape_str(&self.label),
            self.packet_hash64,
            self.similarity.to_json()
        )
    }
}

/// Prototype visual memory packet.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualMemoryPacket {
    pub width: usize,
    pub height: usize,
    pub block_size: usize,
    pub keep_coeffs: usize,
    pub blocks: Vec<EncodedBlock>,
    pub topology: Vec<TopologySample>,
}

impl VisualMemoryPacket {
    pub fn encode(image: &GrayImage, block_size: usize, keep_coeffs: usize) -> Result<Self> {
        Self::encode_with_params(
            image,
            EncodingParams {
                block_size,
                keep_coeffs,
                topology_levels: 16,
            },
        )
    }

    pub fn encode_with_params(image: &GrayImage, params: EncodingParams) -> Result<Self> {
        params.validate()?;
        let block_size = params.block_size;
        let keep_coeffs = params.keep_coeffs;
        let blocks_x = image.width.div_ceil(block_size);
        let blocks_y = image.height.div_ceil(block_size);
        let mut blocks = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block = extract_block(image, bx, by, block_size);
                let coeffs = dct2(&block, block_size);
                let sparse = keep_top_coeffs(&coeffs, keep_coeffs.min(block_size * block_size));
                let hdc = BinaryHv::from_coeffs(&sparse, bx, by);
                blocks.push(EncodedBlock {
                    block_x: bx,
                    block_y: by,
                    coeffs: sparse,
                    hdc,
                });
            }
        }
        let topology = topology_signature(image, params.topology_levels);
        Ok(Self {
            width: image.width,
            height: image.height,
            block_size,
            keep_coeffs,
            blocks,
            topology,
        })
    }

    /// Validate internal packet consistency before use in experiments.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(ProbeError::InvalidFormat(
                "packet dimensions must be nonzero".into(),
            ));
        }
        if self.block_size == 0 || self.keep_coeffs == 0 {
            return Err(ProbeError::InvalidFormat(
                "packet block_size and keep_coeffs must be nonzero".into(),
            ));
        }
        let blocks_x = self.width.div_ceil(self.block_size);
        let blocks_y = self.height.div_ceil(self.block_size);
        let expected_blocks = blocks_x * blocks_y;
        if self.blocks.len() != expected_blocks {
            return Err(ProbeError::InvalidFormat(format!(
                "packet has {} blocks, expected {expected_blocks}",
                self.blocks.len()
            )));
        }
        let mut seen = vec![false; expected_blocks];
        for block in &self.blocks {
            if block.block_x >= blocks_x || block.block_y >= blocks_y {
                return Err(ProbeError::InvalidFormat(format!(
                    "block coordinate out of range: {},{}",
                    block.block_x, block.block_y
                )));
            }
            let slot = block.block_y * blocks_x + block.block_x;
            if seen[slot] {
                return Err(ProbeError::InvalidFormat(format!(
                    "duplicate block coordinate: {},{}",
                    block.block_x, block.block_y
                )));
            }
            seen[slot] = true;
            if block.coeffs.len() > self.keep_coeffs {
                return Err(ProbeError::InvalidFormat(format!(
                    "block {},{} stores {} coeffs, keep_coeffs is {}",
                    block.block_x,
                    block.block_y,
                    block.coeffs.len(),
                    self.keep_coeffs
                )));
            }
            for coeff in &block.coeffs {
                if coeff.index >= self.block_size * self.block_size {
                    return Err(ProbeError::InvalidFormat(format!(
                        "coefficient index {} exceeds block capacity",
                        coeff.index
                    )));
                }
                if !coeff.value.is_finite() {
                    return Err(ProbeError::InvalidFormat(
                        "non-finite coefficient value".into(),
                    ));
                }
            }
        }
        let mut last_threshold = -1.0f32;
        for sample in &self.topology {
            if !sample.threshold.is_finite() {
                return Err(ProbeError::InvalidFormat(
                    "non-finite topology threshold".into(),
                ));
            }
            if sample.threshold < last_threshold {
                return Err(ProbeError::InvalidFormat(
                    "topology thresholds are not monotonic".into(),
                ));
            }
            last_threshold = sample.threshold;
        }
        Ok(())
    }

    /// Stable 64-bit checksum for regression tests and corpus indexing.
    ///
    /// This is intentionally based on the canonical prototype packet text rather
    /// than raw in-memory `f32` bits. The `.svmp` format rounds floating-point
    /// coefficients for readability, so hashing raw coefficient bits makes a
    /// packet hash change after a valid write/read roundtrip. Hashing the
    /// canonical text makes the checksum match the artifact the CLI actually
    /// persists. This is not a cryptographic hash.
    pub fn stable_hash64(&self) -> u64 {
        hash_bytes64(self.to_text().as_bytes())
    }

    pub fn decode(&self) -> Result<GrayImage> {
        let mut image =
            GrayImage::new(self.width, self.height, vec![0.0; self.width * self.height])?;
        for block in &self.blocks {
            let mut coeffs = vec![0.0; self.block_size * self.block_size];
            for coeff in &block.coeffs {
                if coeff.index < coeffs.len() {
                    coeffs[coeff.index] = coeff.value;
                }
            }
            let samples = idct2(&coeffs, self.block_size);
            write_block(
                &mut image,
                block.block_x,
                block.block_y,
                self.block_size,
                &samples,
            );
        }
        Ok(image)
    }

    /// Compute summary metrics for this packet.
    pub fn metrics(&self) -> PacketMetrics {
        let dense_coefficients = self.blocks.len() * self.block_size * self.block_size;
        let stored_coefficients: usize = self.blocks.iter().map(|block| block.coeffs.len()).sum();
        let nonzero_hdc_bits: u32 = self
            .blocks
            .iter()
            .flat_map(|block| block.hdc.words)
            .map(|word| word.count_ones())
            .sum();
        let raw_grayscale_bytes = self.width * self.height;
        let prototype_text_bytes = self.to_text().len();
        let coefficient_density = if dense_coefficients == 0 {
            0.0
        } else {
            stored_coefficients as f32 / dense_coefficients as f32
        };
        let text_to_raw_ratio = if raw_grayscale_bytes == 0 {
            0.0
        } else {
            prototype_text_bytes as f32 / raw_grayscale_bytes as f32
        };
        PacketMetrics {
            width: self.width,
            height: self.height,
            block_size: self.block_size,
            keep_coeffs: self.keep_coeffs,
            blocks: self.blocks.len(),
            dense_coefficients,
            stored_coefficients,
            nonzero_hdc_bits,
            topology_samples: self.topology.len(),
            raw_grayscale_bytes,
            prototype_text_bytes,
            coefficient_density,
            text_to_raw_ratio,
        }
    }

    /// Write prototype packet text to disk.
    pub fn write_text(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_text())?;
        Ok(())
    }

    pub fn read_text(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_text(&fs::read_to_string(path)?)
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(SVMP_MAGIC);
        s.push('\n');
        s.push_str(&format!("dims {} {}\n", self.width, self.height));
        s.push_str(&format!("block_size {}\n", self.block_size));
        s.push_str(&format!("keep_coeffs {}\n", self.keep_coeffs));
        s.push_str(&format!("topology {}\n", self.topology.len()));
        for t in &self.topology {
            s.push_str(&format!("t {:.6} {} {}\n", t.threshold, t.beta0, t.beta1));
        }
        s.push_str(&format!("blocks {}\n", self.blocks.len()));
        for block in &self.blocks {
            s.push_str(&format!(
                "b {} {} {}",
                block.block_x,
                block.block_y,
                block.coeffs.len()
            ));
            for coeff in &block.coeffs {
                s.push_str(&format!(" {}:{:.8}", coeff.index, coeff.value));
            }
            s.push_str(" hdc");
            for word in block.hdc.words {
                s.push_str(&format!(" {word:016x}"));
            }
            s.push('\n');
        }
        s
    }

    pub fn from_text(input: &str) -> Result<Self> {
        let mut lines = input.lines();
        let magic = lines
            .next()
            .ok_or_else(|| ProbeError::InvalidFormat("empty SVMP".into()))?;
        if magic.trim() != SVMP_MAGIC {
            return Err(ProbeError::InvalidFormat(format!(
                "unsupported SVMP magic: {magic}"
            )));
        }

        let dims_line = lines
            .next()
            .ok_or_else(|| ProbeError::InvalidFormat("missing dims".into()))?;
        let dims = words(dims_line);
        if dims.len() != 3 || dims[0] != "dims" {
            return Err(ProbeError::InvalidFormat("expected dims line".into()));
        }
        let width = dims[1]
            .parse::<usize>()
            .map_err(|_| ProbeError::InvalidFormat("bad width".into()))?;
        let height = dims[2]
            .parse::<usize>()
            .map_err(|_| ProbeError::InvalidFormat("bad height".into()))?;

        let block_size = parse_named_usize(lines.next(), "block_size")?;
        let keep_coeffs = parse_named_usize(lines.next(), "keep_coeffs")?;
        let topology_count = parse_named_usize(lines.next(), "topology")?;
        let mut topology = Vec::with_capacity(topology_count);
        for _ in 0..topology_count {
            let line = lines
                .next()
                .ok_or_else(|| ProbeError::InvalidFormat("missing topology sample".into()))?;
            let ws = words(line);
            if ws.len() != 4 || ws[0] != "t" {
                return Err(ProbeError::InvalidFormat("bad topology sample".into()));
            }
            topology.push(TopologySample {
                threshold: ws[1]
                    .parse()
                    .map_err(|_| ProbeError::InvalidFormat("bad threshold".into()))?,
                beta0: ws[2]
                    .parse()
                    .map_err(|_| ProbeError::InvalidFormat("bad beta0".into()))?,
                beta1: ws[3]
                    .parse()
                    .map_err(|_| ProbeError::InvalidFormat("bad beta1".into()))?,
            });
        }
        let block_count = parse_named_usize(lines.next(), "blocks")?;
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let line = lines
                .next()
                .ok_or_else(|| ProbeError::InvalidFormat("missing block".into()))?;
            let ws = words(line);
            if ws.len() < 6 || ws[0] != "b" {
                return Err(ProbeError::InvalidFormat(format!("bad block line: {line}")));
            }
            let block_x = ws[1]
                .parse::<usize>()
                .map_err(|_| ProbeError::InvalidFormat("bad block_x".into()))?;
            let block_y = ws[2]
                .parse::<usize>()
                .map_err(|_| ProbeError::InvalidFormat("bad block_y".into()))?;
            let coeff_count = ws[3]
                .parse::<usize>()
                .map_err(|_| ProbeError::InvalidFormat("bad coeff count".into()))?;
            let mut coeffs = Vec::with_capacity(coeff_count);
            let mut idx = 4;
            for _ in 0..coeff_count {
                let Some((i, v)) = ws[idx].split_once(':') else {
                    return Err(ProbeError::InvalidFormat("bad coeff token".into()));
                };
                coeffs.push(SparseCoeff {
                    index: i
                        .parse()
                        .map_err(|_| ProbeError::InvalidFormat("bad coeff index".into()))?,
                    value: v
                        .parse()
                        .map_err(|_| ProbeError::InvalidFormat("bad coeff value".into()))?,
                });
                idx += 1;
            }
            if ws.get(idx) != Some(&"hdc") {
                return Err(ProbeError::InvalidFormat("missing hdc marker".into()));
            }
            idx += 1;
            if ws.len() < idx + HDC_WORDS {
                return Err(ProbeError::InvalidFormat("not enough hdc words".into()));
            }
            let mut hdc_words = [0u64; HDC_WORDS];
            for word in &mut hdc_words {
                *word = u64::from_str_radix(ws[idx], 16)
                    .map_err(|_| ProbeError::InvalidFormat("bad hdc word".into()))?;
                idx += 1;
            }
            blocks.push(EncodedBlock {
                block_x,
                block_y,
                coeffs,
                hdc: BinaryHv { words: hdc_words },
            });
        }
        Ok(Self {
            width,
            height,
            block_size,
            keep_coeffs,
            blocks,
            topology,
        })
    }
}

fn parse_named_usize(line: Option<&str>, expected: &str) -> Result<usize> {
    let line = line.ok_or_else(|| ProbeError::InvalidFormat(format!("missing {expected}")))?;
    let ws = words(line);
    if ws.len() != 2 || ws[0] != expected {
        return Err(ProbeError::InvalidFormat(format!(
            "expected {expected} line"
        )));
    }
    ws[1]
        .parse::<usize>()
        .map_err(|_| ProbeError::InvalidFormat(format!("bad {expected}")))
}

fn words(line: &str) -> Vec<&str> {
    line.split_ascii_whitespace().collect()
}

fn extract_block(image: &GrayImage, bx: usize, by: usize, block_size: usize) -> Vec<f32> {
    let mut block = vec![0.0; block_size * block_size];
    for y in 0..block_size {
        for x in 0..block_size {
            let ix = (bx * block_size + x).min(image.width - 1);
            let iy = (by * block_size + y).min(image.height - 1);
            block[y * block_size + x] = image.get(ix, iy) - 0.5;
        }
    }
    block
}

fn write_block(image: &mut GrayImage, bx: usize, by: usize, block_size: usize, block: &[f32]) {
    for y in 0..block_size {
        for x in 0..block_size {
            let ix = bx * block_size + x;
            let iy = by * block_size + y;
            if ix < image.width && iy < image.height {
                image.set(ix, iy, block[y * block_size + x] + 0.5);
            }
        }
    }
}

/// Naive 2D DCT-II for prototype correctness, not speed.
pub fn dct2(block: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for v in 0..n {
        for u in 0..n {
            let au = if u == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            let av = if v == 0 {
                (1.0 / n as f32).sqrt()
            } else {
                (2.0 / n as f32).sqrt()
            };
            let mut sum = 0.0;
            for y in 0..n {
                for x in 0..n {
                    let cx = ((PI * (2 * x + 1) as f32 * u as f32) / (2.0 * n as f32)).cos();
                    let cy = ((PI * (2 * y + 1) as f32 * v as f32) / (2.0 * n as f32)).cos();
                    sum += block[y * n + x] * cx * cy;
                }
            }
            out[v * n + u] = au * av * sum;
        }
    }
    out
}

/// Naive 2D inverse DCT-III for prototype correctness, not speed.
pub fn idct2(coeffs: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0;
            for v in 0..n {
                for u in 0..n {
                    let au = if u == 0 {
                        (1.0 / n as f32).sqrt()
                    } else {
                        (2.0 / n as f32).sqrt()
                    };
                    let av = if v == 0 {
                        (1.0 / n as f32).sqrt()
                    } else {
                        (2.0 / n as f32).sqrt()
                    };
                    let cx = ((PI * (2 * x + 1) as f32 * u as f32) / (2.0 * n as f32)).cos();
                    let cy = ((PI * (2 * y + 1) as f32 * v as f32) / (2.0 * n as f32)).cos();
                    sum += au * av * coeffs[v * n + u] * cx * cy;
                }
            }
            out[y * n + x] = sum;
        }
    }
    out
}

/// Keep the largest coefficients by absolute value.
pub fn keep_top_coeffs(coeffs: &[f32], keep: usize) -> Vec<SparseCoeff> {
    let mut indexed: Vec<(usize, f32)> = coeffs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(keep)
        .filter(|(_, value)| value.abs() > 1.0e-7)
        .map(|(index, value)| SparseCoeff { index, value })
        .collect()
}

const HDC_BITS: usize = 1024;
const HDC_WORDS: usize = HDC_BITS / 64;

/// Small deterministic binary hypervector for visual block signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryHv {
    pub words: [u64; HDC_WORDS],
}

impl BinaryHv {
    pub fn zero() -> Self {
        Self {
            words: [0; HDC_WORDS],
        }
    }

    pub fn from_coeffs(coeffs: &[SparseCoeff], bx: usize, by: usize) -> Self {
        let mut hv = Self::zero();
        for coeff in coeffs {
            let magnitude_bucket = (coeff.value.abs() * 10_000.0).round() as i64;
            let sign = if coeff.value >= 0.0 { 1i64 } else { -1i64 };
            let seed = mix64(
                (coeff.index as u64)
                    ^ ((magnitude_bucket.unsigned_abs()) << 8)
                    ^ ((sign as u64) << 47)
                    ^ ((bx as u64) << 16)
                    ^ ((by as u64) << 32),
            );
            for k in 0..4 {
                let bit =
                    (mix64(seed ^ (k as u64).wrapping_mul(0x9e3779b97f4a7c15)) as usize) % HDC_BITS;
                hv.words[bit / 64] ^= 1u64 << (bit % 64);
            }
        }
        hv
    }

    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - self.hamming_distance(other) as f32 / HDC_BITS as f32
    }
}

fn hash_bytes64(bytes: &[u8]) -> u64 {
    let mut h = mix64(bytes.len() as u64);
    for (i, &b) in bytes.iter().enumerate() {
        h ^= mix64((b as u64) ^ ((i as u64).wrapping_mul(0x9e3779b97f4a7c15)));
    }
    h
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Compute a simple threshold topology signature: beta0 foreground components and
/// beta1 foreground holes approximated by background components not touching image border.
pub fn topology_signature(image: &GrayImage, levels: usize) -> Vec<TopologySample> {
    let levels = levels.max(2);
    let mut out = Vec::with_capacity(levels);
    for i in 0..levels {
        let threshold = i as f32 / (levels - 1) as f32;
        let fg = threshold_mask(image, threshold, true);
        let bg = threshold_mask(image, threshold, false);
        let beta0 = count_components(image.width, image.height, &fg).components;
        let bg_components = count_components(image.width, image.height, &bg);
        let beta1 = bg_components
            .components
            .saturating_sub(bg_components.border_components);
        out.push(TopologySample {
            threshold,
            beta0,
            beta1,
        });
    }
    out
}

fn threshold_mask(image: &GrayImage, threshold: f32, foreground: bool) -> Vec<bool> {
    image
        .pixels
        .iter()
        .map(|&v| {
            if foreground {
                v >= threshold
            } else {
                v < threshold
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ComponentCount {
    components: usize,
    border_components: usize,
}

fn count_components(width: usize, height: usize, mask: &[bool]) -> ComponentCount {
    let mut seen = vec![false; mask.len()];
    let mut components = 0usize;
    let mut border_components = 0usize;
    let mut q = VecDeque::new();
    for y in 0..height {
        for x in 0..width {
            let start = y * width + x;
            if !mask[start] || seen[start] {
                continue;
            }
            components += 1;
            let mut touches_border = false;
            seen[start] = true;
            q.push_back((x, y));
            while let Some((cx, cy)) = q.pop_front() {
                if cx == 0 || cy == 0 || cx + 1 == width || cy + 1 == height {
                    touches_border = true;
                }
                let neighbors = [
                    (cx.wrapping_sub(1), cy, cx > 0),
                    (cx + 1, cy, cx + 1 < width),
                    (cx, cy.wrapping_sub(1), cy > 0),
                    (cx, cy + 1, cy + 1 < height),
                ];
                for (nx, ny, valid) in neighbors {
                    if !valid {
                        continue;
                    }
                    let ni = ny * width + nx;
                    if mask[ni] && !seen[ni] {
                        seen[ni] = true;
                        q.push_back((nx, ny));
                    }
                }
            }
            if touches_border {
                border_components += 1;
            }
        }
    }
    ComponentCount {
        components,
        border_components,
    }
}

/// Stable 64-bit checksum for normalized image samples.
/// This is intended for deterministic regression fixtures, not security.
pub fn image_hash64(image: &GrayImage) -> u64 {
    let mut h = mix64(image.width as u64) ^ mix64((image.height as u64) << 1);
    for (i, px) in image.pixels.iter().enumerate() {
        let q = (px.clamp(0.0, 1.0) * 65535.0).round() as u64;
        h ^= mix64(q ^ ((i as u64).wrapping_mul(0x9e3779b97f4a7c15)));
    }
    h
}

/// Edge energy proxy based on adjacent pixel differences.
pub fn edge_energy(image: &GrayImage) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for y in 0..image.height {
        for x in 0..image.width {
            let v = image.get(x, y);
            if x + 1 < image.width {
                total += (v - image.get(x + 1, y)).abs();
                count += 1;
            }
            if y + 1 < image.height {
                total += (v - image.get(x, y + 1)).abs();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

/// Mean squared error.
pub fn mse(a: &GrayImage, b: &GrayImage) -> Result<f32> {
    if a.width != b.width || a.height != b.height {
        return Err(ProbeError::InvalidArgs(
            "image dimensions must match".into(),
        ));
    }
    let sum: f32 = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Ok(sum / a.pixels.len() as f32)
}

/// Peak signal-to-noise ratio for normalized grayscale images.
pub fn psnr(a: &GrayImage, b: &GrayImage) -> Result<f32> {
    let err = mse(a, b)?;
    if err <= f32::EPSILON {
        return Ok(f32::INFINITY);
    }
    Ok(10.0 * (1.0 / err).log10())
}

/// Lightweight global similarity by averaging block HDC similarities in order.
pub fn packet_hdc_similarity(a: &VisualMemoryPacket, b: &VisualMemoryPacket) -> f32 {
    let n = a.blocks.len().min(b.blocks.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = a
        .blocks
        .iter()
        .zip(b.blocks.iter())
        .take(n)
        .map(|(x, y)| x.hdc.similarity(&y.hdc))
        .sum();
    sum / n as f32
}

/// Similarity of two topology signatures. `1.0` means identical summaries.
pub fn topology_similarity(a: &[TopologySample], b: &[TopologySample]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for (x, y) in a.iter().zip(b.iter()).take(n) {
        let beta0_scale = x.beta0.max(y.beta0).max(1) as f32;
        let beta1_scale = x.beta1.max(y.beta1).max(1) as f32;
        let d0 = (x.beta0 as f32 - y.beta0 as f32).abs() / beta0_scale;
        let d1 = (x.beta1 as f32 - y.beta1 as f32).abs() / beta1_scale;
        total += 1.0 - ((d0 + d1) * 0.5).clamp(0.0, 1.0);
    }
    total / n as f32
}

/// Combined packet similarity for query-without-decode experiments.
pub fn packet_similarity(a: &VisualMemoryPacket, b: &VisualMemoryPacket) -> PacketSimilarity {
    let hdc_similarity = packet_hdc_similarity(a, b);
    let topology_similarity = topology_similarity(&a.topology, &b.topology);
    let combined_similarity = (0.7 * hdc_similarity) + (0.3 * topology_similarity);
    PacketSimilarity {
        hdc_similarity,
        topology_similarity,
        combined_similarity,
    }
}

/// Run an encode/decode benchmark using the crate's own cognitive packet path.
pub fn benchmark_image(
    image: &GrayImage,
    block_size: usize,
    keep_coeffs: usize,
) -> Result<BenchmarkReport> {
    let packet = VisualMemoryPacket::encode(image, block_size, keep_coeffs)?;
    let reconstructed = packet.decode()?;
    let mse_value = mse(image, &reconstructed)?;
    let psnr_db = psnr(image, &reconstructed)?;
    let recon_packet = VisualMemoryPacket::encode(&reconstructed, block_size, keep_coeffs)?;
    let self_similarity = packet_similarity(&packet, &recon_packet);
    Ok(BenchmarkReport {
        metrics: packet.metrics(),
        mse: mse_value,
        psnr_db,
        self_similarity,
    })
}

/// Produce a high-level scan summary from one image.
pub fn visual_summary(image: &GrayImage, params: EncodingParams) -> Result<CognitiveScanSummary> {
    params.validate()?;
    let packet = VisualMemoryPacket::encode_with_params(image, params)?;
    let reconstructed = packet.decode()?;
    let reconstruction_psnr_db = psnr(image, &reconstructed)?;
    let metrics = packet.metrics();
    let topology_complexity = topology_complexity(&packet.topology);
    let structural_density = edge_energy(image);
    let hdc_capacity = (packet.blocks.len() * HDC_WORDS * 64) as f32;
    let hdc_activation_ratio = if hdc_capacity <= 0.0 {
        0.0
    } else {
        metrics.nonzero_hdc_bits as f32 / hdc_capacity
    };
    let memory_class = classify_visual_memory(
        structural_density,
        topology_complexity,
        metrics.coefficient_density,
    );
    Ok(CognitiveScanSummary {
        packet_hash64: packet.stable_hash64(),
        image_hash64: image_hash64(image),
        edge_energy: structural_density,
        structural_density,
        topology_complexity,
        hdc_activation_ratio,
        reconstruction_psnr_db,
        memory_class,
        params,
    })
}

/// Simple scalar proxy for how topologically busy a signature is.
pub fn topology_complexity(samples: &[TopologySample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let total: f32 = samples
        .iter()
        .map(|s| (s.beta0 as f32).ln_1p() + 2.0 * (s.beta1 as f32).ln_1p())
        .sum();
    total / samples.len() as f32
}

/// Rank a corpus of packets without decoding pixels.
pub fn rank_packets<'a, I>(query: &VisualMemoryPacket, corpus: I, top: usize) -> Vec<RankedPacket>
where
    I: IntoIterator<Item = (&'a str, &'a VisualMemoryPacket)>,
{
    let mut rows: Vec<RankedPacket> = corpus
        .into_iter()
        .map(|(label, packet)| RankedPacket {
            label: label.to_string(),
            packet_hash64: packet.stable_hash64(),
            similarity: packet_similarity(query, packet),
        })
        .collect();
    rows.sort_by(|a, b| {
        b.similarity
            .combined_similarity
            .partial_cmp(&a.similarity.combined_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(top);
    rows
}

/// Produce a deterministic TSV manifest row for one packet.
pub fn packet_manifest_row(label: &str, packet: &VisualMemoryPacket) -> String {
    let m = packet.metrics();
    format!(
        "{}\t{:016x}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{}\t{:.8}",
        label.replace('\t', " "),
        packet.stable_hash64(),
        m.width,
        m.height,
        m.block_size,
        m.keep_coeffs,
        m.blocks,
        m.stored_coefficients,
        m.coefficient_density,
        m.prototype_text_bytes,
        m.text_to_raw_ratio,
    )
}

pub fn packet_manifest_header() -> &'static str {
    "label\tpacket_hash64\twidth\theight\tblock_size\tkeep_coeffs\tblocks\tstored_coefficients\tcoefficient_density\tprototype_text_bytes\ttext_to_raw_ratio"
}

fn classify_visual_memory(
    edge_energy: f32,
    topology_complexity: f32,
    coefficient_density: f32,
) -> String {
    if edge_energy < 0.005 && topology_complexity < 0.5 {
        "low-structure".to_string()
    } else if topology_complexity > 2.5 || edge_energy > 0.16 {
        "high-structure".to_string()
    } else if coefficient_density < 0.10 {
        "sparse-structured".to_string()
    } else {
        "structured".to_string()
    }
}

fn json_escape_str(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_image() -> GrayImage {
        let mut px = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                px.push(((x + y) as f32 / 14.0).clamp(0.0, 1.0));
            }
        }
        GrayImage::new(8, 8, px).unwrap()
    }

    #[test]
    fn dct_roundtrip_full_coefficients() {
        let img = tiny_image();
        let block = extract_block(&img, 0, 0, 8);
        let coeffs = dct2(&block, 8);
        let recon = idct2(&coeffs, 8);
        let err: f32 = block
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(err < 0.001, "roundtrip error too high: {err}");
    }

    #[test]
    fn encode_decode_packet() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 8, 16).unwrap();
        let recon = packet.decode().unwrap();
        assert_eq!(recon.width, 8);
        assert_eq!(recon.height, 8);
        let score = psnr(&img, &recon).unwrap();
        assert!(!score.is_nan(), "PSNR must never be NaN");
        assert!(
            score.is_infinite() || score >= 40.0,
            "expected high-quality reconstruction for full 8x8 coefficient packet, got PSNR={score}"
        );
    }

    #[test]
    fn psnr_identical_images_is_infinite() {
        let img = tiny_image();
        let score = psnr(&img, &img).unwrap();
        assert!(
            score.is_infinite(),
            "identical images have zero MSE, so PSNR should be +inf"
        );
    }

    #[test]
    fn packet_text_roundtrip() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 4, 6).unwrap();
        let text = packet.to_text();
        let decoded = VisualMemoryPacket::from_text(&text).unwrap();
        assert_eq!(packet.width, decoded.width);
        assert_eq!(packet.height, decoded.height);
        assert_eq!(packet.blocks.len(), decoded.blocks.len());
    }

    #[test]
    fn topology_runs() {
        let img = tiny_image();
        let topo = topology_signature(&img, 8);
        assert_eq!(topo.len(), 8);
    }

    #[test]
    fn hdc_similarity_identical() {
        let coeffs = vec![
            SparseCoeff {
                index: 0,
                value: 1.0,
            },
            SparseCoeff {
                index: 3,
                value: -0.5,
            },
        ];
        let a = BinaryHv::from_coeffs(&coeffs, 0, 0);
        let b = BinaryHv::from_coeffs(&coeffs, 0, 0);
        assert_eq!(a.similarity(&b), 1.0);
    }

    #[test]
    fn metrics_and_similarity_work() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 4, 4).unwrap();
        let metrics = packet.metrics();
        assert_eq!(metrics.width, 8);
        assert!(metrics.stored_coefficients > 0);
        let similarity = packet_similarity(&packet, &packet);
        assert!((similarity.combined_similarity - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn benchmark_runs() {
        let img = tiny_image();
        let report = benchmark_image(&img, 4, 6).unwrap();
        assert!(report.metrics.stored_coefficients > 0);
        assert!(report.mse >= 0.0);
    }

    #[test]
    fn packet_hash_survives_text_roundtrip() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 8, 10).unwrap();
        let roundtrip = VisualMemoryPacket::from_text(&packet.to_text()).unwrap();
        assert_eq!(packet.stable_hash64(), roundtrip.stable_hash64());
    }

    #[test]
    fn validate_and_hash_are_stable() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 4, 6).unwrap();
        packet.validate().unwrap();
        assert_ne!(packet.stable_hash64(), 0);
        assert_ne!(image_hash64(&img), 0);
        assert!(edge_energy(&img) > 0.0);
    }

    #[test]
    fn encode_params_and_summary_work() {
        let img = tiny_image();
        let params = EncodingParams::new(4, 6, 8).unwrap();
        let packet = VisualMemoryPacket::encode_with_params(&img, params).unwrap();
        assert_eq!(packet.topology.len(), 8);
        let summary = visual_summary(&img, params).unwrap();
        assert_ne!(summary.packet_hash64, 0);
        assert!(!summary.memory_class.is_empty());
        assert!(summary.to_json().contains("memory_class"));
    }

    #[test]
    fn rank_packets_orders_identical_first() {
        let img = tiny_image();
        let packet = VisualMemoryPacket::encode(&img, 4, 6).unwrap();
        let mut altered = packet.clone();
        altered.topology.clear();
        let corpus = vec![("altered", &altered), ("same", &packet)];
        let ranked = rank_packets(&packet, corpus, 2);
        assert_eq!(ranked[0].label, "same");
    }
}
