//! Explicit, versioned HARTH label protocols.
//!
//! Raw parser codes remain numeric identities. The UCI dataset metadata and
//! original paper use different descriptions for codes 130/140; only this
//! named paper-reproduction profile applies the original paper's nine-class
//! merge. The raw parser never bakes this interpretation into ingestion.

use super::digest::CanonicalHasher;
use super::harth_stream::HarthActivity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarthLabelProfile {
    /// Preserve all twelve original numeric label IDs as twelve target classes.
    NativeTwelve,
    /// Reproduce the original paper's explicit twelve-to-nine class merge.
    PaperMergedNineV1,
}

impl HarthLabelProfile {
    pub const fn profile_id(self) -> &'static str {
        match self {
            Self::NativeTwelve => "harth.native-12.v1",
            Self::PaperMergedNineV1 => "harth.paper-merged-9.v1",
        }
    }

    pub const fn class_count(self) -> u8 {
        match self {
            Self::NativeTwelve => 12,
            Self::PaperMergedNineV1 => 9,
        }
    }

    /// Map one parsed raw label to a stable target class ID.
    ///
    /// NativeTwelve returns contiguous class IDs 1..=12 in the declared raw-code order.
    /// PaperMergedNineV1 uses the original paper's reported label aggregation:
    /// 3/6/140 -> Standing and 7/130 -> Sitting.
    pub fn target_class_id(self, raw: HarthActivity) -> u8 {
        match self {
            Self::NativeTwelve => match raw {
                HarthActivity::Walking => 1,
                HarthActivity::Running => 2,
                HarthActivity::Shuffling => 3,
                HarthActivity::StairsAscending => 4,
                HarthActivity::StairsDescending => 5,
                HarthActivity::Standing => 6,
                HarthActivity::Sitting => 7,
                HarthActivity::Lying => 8,
                HarthActivity::CyclingSitting => 9,
                HarthActivity::CyclingStanding => 10,
                HarthActivity::Label130 => 11,
                HarthActivity::Label140 => 12,
            },
            Self::PaperMergedNineV1 => match raw {
                HarthActivity::Walking => 1,
                HarthActivity::Running => 2,
                HarthActivity::Shuffling
                | HarthActivity::Standing
                | HarthActivity::Label140 => 3,
                HarthActivity::Sitting | HarthActivity::Label130 => 4,
                HarthActivity::StairsAscending => 5,
                HarthActivity::StairsDescending => 6,
                HarthActivity::Lying => 7,
                HarthActivity::CyclingSitting => 8,
                HarthActivity::CyclingStanding => 9,
            },
        }
    }
    /// Domain-separated digest of the profile's complete raw-code mapping.
    pub fn mapping_sha256(self) -> [u8; 32] {
        let mut hasher = CanonicalHasher::new(b"symthaea.temporal.harth-label-profile.v1");
        hasher.bytes(self.profile_id().as_bytes());
        hasher.u8(self.class_count());
        for raw in [
            HarthActivity::Walking,
            HarthActivity::Running,
            HarthActivity::Shuffling,
            HarthActivity::StairsAscending,
            HarthActivity::StairsDescending,
            HarthActivity::Standing,
            HarthActivity::Sitting,
            HarthActivity::Lying,
            HarthActivity::CyclingSitting,
            HarthActivity::CyclingStanding,
            HarthActivity::Label130,
            HarthActivity::Label140,
        ] {
            hasher.u16(raw.code());
            hasher.u8(self.target_class_id(raw));
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_profile_maps_twelve_source_ids_to_distinct_contiguous_target_classes() {
        for code in [1_u16, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140] {
            let raw = HarthActivity::from_code(code).unwrap();
            assert_eq!(
                HarthLabelProfile::NativeTwelve.target_class_id(raw),
                [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][
                    [1_u16, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]
                        .iter()
                        .position(|candidate| *candidate == code)
                        .unwrap()
                ]
            );
        }
        assert_eq!(HarthLabelProfile::NativeTwelve.class_count(), 12);
    }

    #[test]
    fn paper_merged_nine_profile_applies_only_documented_merges() {
        let profile = HarthLabelProfile::PaperMergedNineV1;
        let class = |code| profile.target_class_id(HarthActivity::from_code(code).unwrap());
        assert_eq!(class(3), 3);
        assert_eq!(class(6), 3);
        assert_eq!(class(140), 3);
        assert_eq!(class(7), 4);
        assert_eq!(class(130), 4);
        assert_eq!(profile.class_count(), 9);
    }

    #[test]
    fn distinct_label_protocols_have_distinct_stable_digests() {
        assert_ne!(
            HarthLabelProfile::NativeTwelve.mapping_sha256(),
            HarthLabelProfile::PaperMergedNineV1.mapping_sha256()
        );
        assert_eq!(
            HarthLabelProfile::NativeTwelve.mapping_sha256(),
            HarthLabelProfile::NativeTwelve.mapping_sha256()
        );
    }

    #[test]
    fn every_native_code_maps_into_the_paper_nine_class_space() {
        let profile = HarthLabelProfile::PaperMergedNineV1;
        for code in [1_u16, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140] {
            let mapped = profile.target_class_id(HarthActivity::from_code(code).unwrap());
            assert!((1..=9).contains(&mapped));
        }
    }
}
