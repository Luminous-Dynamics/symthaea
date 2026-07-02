// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio processing pipelines

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::mpsc;

use crate::{audio, codec, CoreError, Result};
use codec::{OpusEncoder, Mp3Encoder, VorbisEncoder};

/// A processing node in the pipeline
#[async_trait]
pub trait PipelineNode: Send + Sync {
    /// Process audio through this node
    async fn process(&self, input: audio::AudioBuffer) -> Result<audio::AudioBuffer>;

    /// Get the node name
    fn name(&self) -> &str;
}

/// Audio processing pipeline
pub struct AudioPipeline {
    nodes: Vec<Arc<dyn PipelineNode>>,
    sample_rate: u32,
    channels: usize,
}

impl AudioPipeline {
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            nodes: Vec::new(),
            sample_rate,
            channels,
        }
    }

    /// Add a processing node
    pub fn add_node(&mut self, node: Arc<dyn PipelineNode>) {
        self.nodes.push(node);
    }

    /// Process audio through the pipeline
    pub async fn process(&self, input: audio::AudioBuffer) -> Result<audio::AudioBuffer> {
        let mut buffer = input;
        for node in &self.nodes {
            buffer = node.process(buffer).await?;
        }
        Ok(buffer)
    }

    /// Get pipeline info
    pub fn info(&self) -> PipelineInfo {
        PipelineInfo {
            node_count: self.nodes.len(),
            node_names: self.nodes.iter().map(|n| n.name().to_string()).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineInfo {
    pub node_count: usize,
    pub node_names: Vec<String>,
    pub sample_rate: u32,
    pub channels: usize,
}

/// Gain adjustment node
pub struct GainNode {
    gain: f32,
}

impl GainNode {
    pub fn new(gain_db: f32) -> Self {
        Self {
            gain: 10.0_f32.powf(gain_db / 20.0),
        }
    }
}

#[async_trait]
impl PipelineNode for GainNode {
    async fn process(&self, mut input: audio::AudioBuffer) -> Result<audio::AudioBuffer> {
        for sample in &mut input.samples {
            *sample *= self.gain;
        }
        Ok(input)
    }

    fn name(&self) -> &str {
        "Gain"
    }
}

/// Limiter node to prevent clipping
pub struct LimiterNode {
    threshold: f32,
}

impl LimiterNode {
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold: 10.0_f32.powf(threshold_db / 20.0),
        }
    }
}

#[async_trait]
impl PipelineNode for LimiterNode {
    async fn process(&self, mut input: audio::AudioBuffer) -> Result<audio::AudioBuffer> {
        for sample in &mut input.samples {
            if sample.abs() > self.threshold {
                *sample = self.threshold * sample.signum();
            }
        }
        Ok(input)
    }

    fn name(&self) -> &str {
        "Limiter"
    }
}

/// Normalization node
pub struct NormalizeNode {
    target_peak: f32,
}

impl NormalizeNode {
    pub fn new(target_db: f32) -> Self {
        Self {
            target_peak: 10.0_f32.powf(target_db / 20.0),
        }
    }
}

#[async_trait]
impl PipelineNode for NormalizeNode {
    async fn process(&self, mut input: audio::AudioBuffer) -> Result<audio::AudioBuffer> {
        let max = input.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max > 0.0 {
            let scale = self.target_peak / max;
            for sample in &mut input.samples {
                *sample *= scale;
            }
        }
        Ok(input)
    }

    fn name(&self) -> &str {
        "Normalize"
    }
}

/// Encoder node (outputs encoded bytes)
pub struct EncoderNode {
    codec: String,
    quality: codec::QualityPreset,
}

impl EncoderNode {
    pub fn new(codec: impl Into<String>, quality: codec::QualityPreset) -> Self {
        Self {
            codec: codec.into(),
            quality,
        }
    }

    pub async fn encode(&self, input: &audio::AudioBuffer) -> Result<Bytes> {
        let config = match self.codec.to_lowercase().as_str() {
            "opus" => codec::EncoderConfig::opus(self.quality),
            "mp3" => codec::EncoderConfig::mp3(self.quality),
            "vorbis" | "ogg" => codec::EncoderConfig::vorbis(self.quality),
            _ => return Err(CoreError::Config(format!("Unknown codec: {}", self.codec))),
        };

        let mut encoder: Box<dyn codec::Encoder> = match self.codec.to_lowercase().as_str() {
            "opus" => Box::new(OpusEncoder::new(config)?),
            "mp3" => Box::new(Mp3Encoder::new(config)?),
            "vorbis" | "ogg" => Box::new(VorbisEncoder::new(config)?),
            _ => return Err(CoreError::Config(format!("Unknown codec: {}", self.codec))),
        };
        let mut output = encoder.encode(&input.samples)?.to_vec();
        output.extend(encoder.flush()?);

        Ok(Bytes::from(output))
    }
}

#[async_trait]
impl PipelineNode for EncoderNode {
    async fn process(&self, input: audio::AudioBuffer) -> Result<audio::AudioBuffer> {
        // Encoder doesn't transform AudioBuffer, use encode() directly
        Ok(input)
    }

    fn name(&self) -> &str {
        "Encoder"
    }
}

/// Streaming pipeline for real-time processing
pub struct StreamingPipeline {
    input_tx: mpsc::Sender<audio::AudioBuffer>,
    output_rx: mpsc::Receiver<audio::AudioBuffer>,
}

impl StreamingPipeline {
    pub fn new(pipeline: AudioPipeline, buffer_size: usize) -> Self {
        let (input_tx, mut input_rx) = mpsc::channel::<audio::AudioBuffer>(buffer_size);
        let (output_tx, output_rx) = mpsc::channel::<audio::AudioBuffer>(buffer_size);

        let pipeline_arc = Arc::new(pipeline);

        tokio::spawn(async move {
            while let Some(input) = input_rx.recv().await {
                match pipeline_arc.process(input).await {
                    Ok(output) => {
                        if output_tx.send(output).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Pipeline error: {}", e);
                    }
                }
            }
        });

        Self {
            input_tx,
            output_rx,
        }
    }

    pub async fn send(&self, buffer: audio::AudioBuffer) -> Result<()> {
        self.input_tx
            .send(buffer)
            .await
            .map_err(|_| CoreError::Pipeline("Input channel closed".to_string()))
    }

    pub async fn recv(&mut self) -> Option<audio::AudioBuffer> {
        self.output_rx.recv().await
    }
}

/// Pipeline builder
pub struct PipelineBuilder {
    sample_rate: u32,
    channels: usize,
    nodes: Vec<Arc<dyn PipelineNode>>,
}

impl PipelineBuilder {
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            nodes: Vec::new(),
        }
    }

    pub fn add<N: PipelineNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Arc::new(node));
        self
    }

    pub fn gain(self, db: f32) -> Self {
        self.add(GainNode::new(db))
    }

    pub fn limiter(self, threshold_db: f32) -> Self {
        self.add(LimiterNode::new(threshold_db))
    }

    pub fn normalize(self, target_db: f32) -> Self {
        self.add(NormalizeNode::new(target_db))
    }

    pub fn build(self) -> AudioPipeline {
        let mut pipeline = AudioPipeline::new(self.sample_rate, self.channels);
        for node in self.nodes {
            pipeline.add_node(node);
        }
        pipeline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gain_node() {
        let node = GainNode::new(6.0); // +6dB
        let input = audio::AudioBuffer {
            samples: vec![0.5; 100],
            sample_rate: 48000,
            channels: 1,
            layout: audio::ChannelLayout::Mono,
        };

        let output = node.process(input).await.unwrap();
        assert!(output.samples[0] > 0.5); // Should be louder
    }

    #[tokio::test]
    async fn test_pipeline_builder() {
        let pipeline = PipelineBuilder::new(48000, 2)
            .gain(-3.0)
            .limiter(-1.0)
            .build();

        assert_eq!(pipeline.info().node_count, 2);
        assert_eq!(pipeline.info().node_names[0], "Gain");
        assert_eq!(pipeline.info().node_names[1], "Limiter");
    }
}
