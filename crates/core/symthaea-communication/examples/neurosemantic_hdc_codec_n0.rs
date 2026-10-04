use serde_json::Value;
use symthaea_communication::hdc_codec::{
    measure_bit_corruption, quantize_continuous, HdcBinaryFrame, HdcCodecDescriptor,
    HDC_CODEC_SCHEMA_VERSION,
};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

fn main() -> Result<(), String> {
    let seed = 42_u64;
    let continuous = ContinuousHV::random(HDC_DIMENSION, seed);

    let (binary, metrics) = quantize_continuous(&continuous)?;
    let frame = HdcBinaryFrame::from_binary(&binary);
    let encoded = serde_json::to_vec(&frame).map_err(|error| error.to_string())?;
    let decoded_frame: HdcBinaryFrame =
        serde_json::from_slice(&encoded).map_err(|error| error.to_string())?;
    let restored_binary = decoded_frame.to_binary()?;

    if binary != restored_binary {
        return Err("binary frame round-trip changed the HDC code".into());
    }

    let (repeat_binary, repeat_metrics) = quantize_continuous(&continuous)?;
    if binary != repeat_binary || metrics != repeat_metrics {
        return Err("HDC quantization is not deterministic".into());
    }

    if !metrics.validates() || metrics.schema_version != HDC_CODEC_SCHEMA_VERSION {
        return Err("HDC codec metrics failed validation".into());
    }

    let codec_descriptor = HdcCodecDescriptor::v1();
    if !codec_descriptor.validates() {
        return Err("HDC codec descriptor failed validation".into());
    }

    let corruption = [0.0_f32, 0.001, 0.01, 0.05, 0.10]
        .into_iter()
        .map(|flip_probability| measure_bit_corruption(&continuous, flip_probability, 9001))
        .collect::<Result<Vec<_>, _>>()?;

    let report = serde_json::json!({
        "benchmark": "neurosemantic-hdc-codec-n0",
        "claim_boundary": "continuous_binary_quantization_only",
        "codec_schema_version": HDC_CODEC_SCHEMA_VERSION,
        "codec_descriptor": codec_descriptor,
        "dimension": HDC_DIMENSION,
        "seed": seed,
        "metrics": metrics,
        "binary_frame_json_bytes": encoded.len(),
        "binary_frame_roundtrip_exact": true,
        "deterministic_repeat_exact": true,
        "semantic_decoder_invoked": false,
        "bit_corruption_observations": corruption,
    });

    // Ensure the report is valid JSON before printing it.
    let _: Value = report.clone();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}
