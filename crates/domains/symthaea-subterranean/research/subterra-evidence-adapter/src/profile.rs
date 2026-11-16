//! Typed adapter for Subterra non-uniform one-dimensional survey profiles.
//!
//! This module preserves unobserved intervals explicitly. Its summary computes
//! simple length-weighted observed dimensions only; it does not reconstruct a
//! 3D cave, infer geometry inside missing spans, assess stability, or replace
//! the separate Symtropy morphometry reference.

use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};

fn required_nullable<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileMeasurement {
    pub value: f64,
    #[serde(deserialize_with = "required_nullable")]
    pub standard_uncertainty: Option<f64>,
    #[serde(deserialize_with = "required_nullable")]
    pub correlation_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionObservation {
    pub width_m: ProfileMeasurement,
    pub height_m: ProfileMeasurement,
    #[serde(deserialize_with = "required_nullable")]
    pub roof_thickness_m: Option<ProfileMeasurement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSegment {
    pub segment_id: String,
    pub start_m: f64,
    pub end_m: f64,
    /// Null is an explicit unobserved interval. The field itself is required.
    #[serde(deserialize_with = "required_nullable")]
    pub cross_section: Option<CrossSectionObservation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubterraProfileRecord {
    pub profile_id: String,
    pub total_length_m: f64,
    pub segments: Vec<ProfileSegment>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileIssue {
    EmptyProfileId,
    InvalidTotalLength,
    NoSegments,
    EmptySegmentId,
    DuplicateSegmentId(String),
    InvalidStationBounds(String),
    NonContiguousInterval(String),
    IncompleteProfileCoverage,
    InvalidMeasurement { segment_id: String, field: &'static str },
    InvalidUncertainty { segment_id: String, field: &'static str },
    EmptyCorrelationGroup { segment_id: String, field: &'static str },
    NumericalOverflow(&'static str),
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl ProfileMeasurement {
    fn validate(
        &self,
        segment_id: &str,
        field: &'static str,
        issues: &mut Vec<ProfileIssue>,
    ) {
        if !self.value.is_finite() || self.value <= 0.0 {
            issues.push(ProfileIssue::InvalidMeasurement {
                segment_id: segment_id.to_string(),
                field,
            });
        }
        if self
            .standard_uncertainty
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            issues.push(ProfileIssue::InvalidUncertainty {
                segment_id: segment_id.to_string(),
                field,
            });
        }
        if self
            .correlation_group
            .as_deref()
            .is_some_and(|value| !nonempty(value))
        {
            issues.push(ProfileIssue::EmptyCorrelationGroup {
                segment_id: segment_id.to_string(),
                field,
            });
        }
    }
}

impl SubterraProfileRecord {
    /// Validate ordered, contiguous station coverage and preserve null
    /// cross-sections as unknown geometry rather than substituting zero.
    pub fn validate(&self) -> Result<(), Vec<ProfileIssue>> {
        let mut issues = Vec::new();
        if !nonempty(&self.profile_id) {
            issues.push(ProfileIssue::EmptyProfileId);
        }
        if !self.total_length_m.is_finite() || self.total_length_m <= 0.0 {
            issues.push(ProfileIssue::InvalidTotalLength);
        }
        if self.segments.is_empty() {
            issues.push(ProfileIssue::NoSegments);
        }

        let mut seen_ids = BTreeSet::new();
        let mut previous_end = 0.0_f64;

        for segment in &self.segments {
            if !nonempty(&segment.segment_id) {
                issues.push(ProfileIssue::EmptySegmentId);
            }
            if !seen_ids.insert(segment.segment_id.as_str()) {
                issues.push(ProfileIssue::DuplicateSegmentId(
                    segment.segment_id.clone(),
                ));
            }

            if !segment.start_m.is_finite()
                || !segment.end_m.is_finite()
                || segment.start_m < 0.0
                || segment.end_m <= segment.start_m
                || segment.end_m > self.total_length_m
            {
                issues.push(ProfileIssue::InvalidStationBounds(
                    segment.segment_id.clone(),
                ));
                continue;
            }

            if (segment.start_m - previous_end).abs() > 1.0e-6 {
                issues.push(ProfileIssue::NonContiguousInterval(
                    segment.segment_id.clone(),
                ));
            }
            previous_end = segment.end_m;

            if let Some(section) = &segment.cross_section {
                section.width_m.validate(
                    &segment.segment_id,
                    "width_m",
                    &mut issues,
                );
                section.height_m.validate(
                    &segment.segment_id,
                    "height_m",
                    &mut issues,
                );
                if let Some(roof) = &section.roof_thickness_m {
                    roof.validate(
                        &segment.segment_id,
                        "roof_thickness_m",
                        &mut issues,
                    );
                }
            }
        }

        if self.segments.is_empty()
            || (previous_end - self.total_length_m).abs() > 1.0e-6
        {
            issues.push(ProfileIssue::IncompleteProfileCoverage);
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    pub fn observed_summary(&self) -> Result<ProfileSummary, Vec<ProfileIssue>> {
        self.validate()?;

        let mut observed_length = 0.0_f64;
        let mut roof_length = 0.0_f64;
        let mut weighted_width = 0.0_f64;
        let mut weighted_height = 0.0_f64;
        let mut width_values = Vec::new();
        let mut height_values = Vec::new();

        for segment in &self.segments {
            let Some(section) = &segment.cross_section else {
                continue;
            };

            let span = segment.end_m - segment.start_m;
            observed_length += span;
            weighted_width += section.width_m.value * span;
            weighted_height += section.height_m.value * span;

            width_values.push(section.width_m.value);
            height_values.push(section.height_m.value);

            if section.roof_thickness_m.is_some() {
                roof_length += span;
            }
        }

        if !observed_length.is_finite()
            || !weighted_width.is_finite()
            || !weighted_height.is_finite()
            || !roof_length.is_finite()
        {
            return Err(vec![ProfileIssue::NumericalOverflow(
                "weighted profile summary",
            )]);
        }

        let coverage_fraction =
            (observed_length / self.total_length_m).clamp(0.0, 1.0);
        let roof_thickness_coverage_fraction =
            (roof_length / self.total_length_m).clamp(0.0, 1.0);

        let (mean_width_m, mean_height_m, min_width_m, max_width_m, min_height_m, max_height_m) =
            if observed_length > 0.0 {
                (
                    Some(weighted_width / observed_length),
                    Some(weighted_height / observed_length),
                    width_values.iter().copied().reduce(f64::min),
                    width_values.iter().copied().reduce(f64::max),
                    height_values.iter().copied().reduce(f64::min),
                    height_values.iter().copied().reduce(f64::max),
                )
            } else {
                (None, None, None, None, None, None)
            };

        let coverage_status = if observed_length <= 0.0 {
            ProfileCoverageStatus::Unquantified
        } else if coverage_fraction >= 1.0 - (1.0e-6 / self.total_length_m) {
            ProfileCoverageStatus::FullyObserved
        } else {
            ProfileCoverageStatus::PartiallyObserved
        };

        Ok(ProfileSummary {
            total_length_m: self.total_length_m,
            observed_length_m: observed_length,
            coverage_fraction,
            coverage_status,
            mean_width_m,
            mean_height_m,
            min_observed_width_m: min_width_m,
            max_observed_width_m: max_width_m,
            min_observed_height_m: min_height_m,
            max_observed_height_m: max_height_m,
            roof_thickness_coverage_fraction,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileCoverageStatus {
    FullyObserved,
    PartiallyObserved,
    Unquantified,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSummary {
    pub total_length_m: f64,
    pub observed_length_m: f64,
    pub coverage_fraction: f64,
    pub coverage_status: ProfileCoverageStatus,
    /// Length-weighted over observed intervals only. Unknown spans are omitted,
    /// not filled with zero or interpolated.
    pub mean_width_m: Option<f64>,
    pub mean_height_m: Option<f64>,
    pub min_observed_width_m: Option<f64>,
    pub max_observed_width_m: Option<f64>,
    pub min_observed_height_m: Option<f64>,
    pub max_observed_height_m: Option<f64>,
    pub roof_thickness_coverage_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileIntakeError {
    Json(String),
    Semantic(Vec<ProfileIssue>),
}

impl std::fmt::Display for ProfileIntakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ProfileIntakeError {}

pub fn parse_profile_record(input: &str) -> Result<SubterraProfileRecord, ProfileIntakeError> {
    let profile: SubterraProfileRecord =
        serde_json::from_str(input).map_err(|error| ProfileIntakeError::Json(error.to_string()))?;
    profile.validate().map_err(ProfileIntakeError::Semantic)?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE_FIXTURE: &str =
        include_str!("../fixtures/nonuniform-profile-synthetic.json");

    #[test]
    fn fixture_preserves_explicit_unobserved_interval() {
        let profile = parse_profile_record(PROFILE_FIXTURE).unwrap();
        assert_eq!(profile.segments[2].cross_section, None);

        let summary = profile.observed_summary().unwrap();
        assert_eq!(summary.coverage_status, ProfileCoverageStatus::PartiallyObserved);
        assert!((summary.coverage_fraction - 0.8).abs() < 1.0e-12);
        assert!((summary.mean_width_m.unwrap() - 8.75).abs() < 1.0e-12);
        assert!((summary.mean_height_m.unwrap() - 3.75).abs() < 1.0e-12);
        assert!((summary.roof_thickness_coverage_fraction - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn unobserved_span_is_not_counted_as_zero_width() {
        let profile = parse_profile_record(PROFILE_FIXTURE).unwrap();
        let summary = profile.observed_summary().unwrap();
        assert_eq!(summary.min_observed_width_m, Some(6.0));
        assert_eq!(summary.max_observed_width_m, Some(12.0));
    }

    #[test]
    fn missing_field_is_not_silently_treated_as_null() {
        let invalid = PROFILE_FIXTURE.replace(
            "\"cross_section\": null",
            "\"unexpected\": null",
        );
        assert!(parse_profile_record(&invalid).is_err());
    }

    #[test]
    fn negative_uncertainty_is_rejected() {
        let mut profile = parse_profile_record(PROFILE_FIXTURE).unwrap();
        profile.segments[0]
            .cross_section
            .as_mut()
            .unwrap()
            .width_m
            .standard_uncertainty = Some(-0.1);
        assert!(profile.validate().unwrap_err().contains(
            &ProfileIssue::InvalidUncertainty {
                segment_id: "section-a".into(),
                field: "width_m"
            }
        ));
    }

    #[test]
    fn noncontiguous_gap_must_be_explicit() {
        let mut profile = parse_profile_record(PROFILE_FIXTURE).unwrap();
        profile.segments.remove(2);
        assert!(profile.validate().unwrap_err().iter().any(|issue| matches!(
            issue,
            ProfileIssue::NonContiguousInterval(_)
                | ProfileIssue::IncompleteProfileCoverage
        )));
    }

    #[test]
    fn empty_cross_section_profile_is_unquantified() {
        let profile = SubterraProfileRecord {
            profile_id: "all-unknown".into(),
            total_length_m: 10.0,
            segments: vec![ProfileSegment {
                segment_id: "unknown".into(),
                start_m: 0.0,
                end_m: 10.0,
                cross_section: None,
            }],
            notes: "synthetic all-unknown profile".into(),
        };
        let summary = profile.observed_summary().unwrap();
        assert_eq!(summary.coverage_status, ProfileCoverageStatus::Unquantified);
        assert_eq!(summary.coverage_fraction, 0.0);
        assert!(summary.mean_width_m.is_none());
    }
}
