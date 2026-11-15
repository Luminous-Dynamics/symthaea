use mel_audio_boundary_integrity_harness::{
    AudioBoundaryRequestV1, AudioBoundaryReceiptV1, AudioPcmBufferV1,
    evaluate_audio_edit_boundary_v1, verify_audio_boundary_receipt_v1,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    request: AudioBoundaryRequestV1,
    source: AudioPcmBufferV1,
    output: AudioPcmBufferV1,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/audio-boundary-known-answer-v1.json"
    ))?;
    let receipt: AudioBoundaryReceiptV1 =
        evaluate_audio_edit_boundary_v1(&fixture.request, &fixture.source, &fixture.output)?;
    if !receipt.payload.passed || receipt.payload.changed_frame_count != 2 {
        return Err("known-answer PCM boundary fixture did not pass".into());
    }
    verify_audio_boundary_receipt_v1(
        &fixture.request,
        &fixture.source,
        &fixture.output,
        &receipt,
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
