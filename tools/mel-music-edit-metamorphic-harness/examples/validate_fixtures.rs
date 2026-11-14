use mel_music_edit_metamorphic_harness::{
    music_edit_metamorphic_pair_digest_v1, validate_music_edit_metamorphic_pair_v1,
    MusicEditMetamorphicPairV1, MusicEditMetamorphicRelationV1,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureSet {
    schema_version: String,
    pairs: Vec<MusicEditMetamorphicPairV1>,
}

const FIXTURE: &str =
    include_str!("../fixtures/synthetic-music-edit-metamorphic-pairs-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture: FixtureSet = serde_json::from_str(FIXTURE)?;
    if fixture.schema_version != "mel-music-edit-metamorphic-fixture-set-v1" || fixture.pairs.len() != 4 {
        return Err(std::io::Error::other("wrong metamorphic fixture-set identity or coverage").into());
    }
    let mut counts = [0_usize; 4];
    for pair in &fixture.pairs {
        let issues = validate_music_edit_metamorphic_pair_v1(pair);
        if !issues.is_empty() {
            return Err(std::io::Error::other(format!("{} rejected: {issues:?}", pair.pair_id)).into());
        }
        music_edit_metamorphic_pair_digest_v1(pair)?;
        let slot = match pair.expected_relation {
            MusicEditMetamorphicRelationV1::Invariant => 0,
            MusicEditMetamorphicRelationV1::Equivariant => 1,
            MusicEditMetamorphicRelationV1::NonInvariant => 2,
            MusicEditMetamorphicRelationV1::AbstainExpected => 3,
        };
        counts[slot] += 1;
    }
    if counts != [1, 1, 1, 1] {
        return Err(std::io::Error::other(format!("relation coverage mismatch: {counts:?}")).into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-edit-metamorphic-fixture-check-v1\",\"pair_count\":{},\"invariant\":{},\"equivariant\":{},\"non_invariant\":{},\"abstain_expected\":{},\"audio_transforms_executed\":false,\"editor_runs_executed\":false}}",
        fixture.pairs.len(), counts[0], counts[1], counts[2], counts[3]
    );
    Ok(())
}
