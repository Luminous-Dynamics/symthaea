#![forbid(unsafe_code)]

//! Exact symbolic action fixtures for task-indexed musical identity.
//!
//! This standalone crate checks a bounded C12 transposition action and
//! positive-rational tempo scaling. It does not authenticate upstream
//! certificates, process audio, or authorize production search pruning.

use std::collections::BTreeMap;

pub const PITCH_CLASS_COUNT: u8 = 12;

/// Runtime search pruning is deliberately disabled in this first adapter.
pub const RUNTIME_PRUNING_AUTHORIZED_V0_1: bool = false;

/// A validated pitch class in C12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PitchClass(u8);

impl PitchClass {
    /// Construct a pitch class, rejecting values outside 0..12.
    pub const fn new(value: u8) -> Option<Self> {
        if value < PITCH_CLASS_COUNT {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Return the canonical pitch-class integer.
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Apply a C12 transposition.
    pub const fn transpose(self, action: Transposition) -> Self {
        Self((self.0 + action.semitones) % PITCH_CLASS_COUNT)
    }
}

/// An element of the cyclic transposition group C12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Transposition {
    semitones: u8,
}

impl Transposition {
    /// Construct the canonical C12 element represented by any integer offset.
    pub const fn new(semitones: u8) -> Self {
        Self {
            semitones: semitones % PITCH_CLASS_COUNT,
        }
    }

    /// The identity element.
    pub const fn identity() -> Self {
        Self { semitones: 0 }
    }

    /// Return the canonical semitone displacement.
    pub const fn semitones(self) -> u8 {
        self.semitones
    }

    /// Compose in execution order: apply this, then next.
    pub const fn compose(self, next: Self) -> Self {
        Self::new(self.semitones + next.semitones)
    }

    /// Return the group inverse.
    pub const fn inverse(self) -> Self {
        Self::new(PITCH_CLASS_COUNT - self.semitones)
    }
}

/// Four exact triad-quality anchors used by the finite fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TriadQuality {
    /// Major triad: 0, 4, 7 semitones.
    Major,
    /// Minor triad: 0, 3, 7 semitones.
    Minor,
    /// Diminished triad: 0, 3, 6 semitones.
    Diminished,
    /// Augmented triad: 0, 4, 8 semitones.
    Augmented,
}

impl TriadQuality {
    /// Return the root-relative interval signature.
    pub const fn relative_intervals(self) -> [u8; 3] {
        match self {
            Self::Major => [0, 4, 7],
            Self::Minor => [0, 3, 7],
            Self::Diminished => [0, 3, 6],
            Self::Augmented => [0, 4, 8],
        }
    }
}

/// A symbolic triad with an absolute pitch-class root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chord {
    root: PitchClass,
    quality: TriadQuality,
}

impl Chord {
    /// Construct a chord from a validated root and triad quality.
    pub const fn new(root: PitchClass, quality: TriadQuality) -> Self {
        Self { root, quality }
    }

    /// Return the absolute root pitch class.
    pub const fn root(self) -> PitchClass {
        self.root
    }

    /// Return the quality label.
    pub const fn quality(self) -> TriadQuality {
        self.quality
    }

    /// Return the exact root-relative interval signature.
    pub const fn relative_intervals(self) -> [u8; 3] {
        self.quality.relative_intervals()
    }

    /// Apply a global transposition.
    pub const fn transpose(self, action: Transposition) -> Self {
        Self {
            root: self.root.transpose(action),
            quality: self.quality,
        }
    }
}

/// A positive rational duration multiplier, stored in reduced form.
///
/// Scaling event durations by p/q scales pulse period by p/q and pulse rate
/// (BPM) by q/p. The representation is bounded by u64; checked composition
/// returns None rather than overflowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TempoScale {
    numerator: u64,
    denominator: u64,
}

impl TempoScale {
    /// Construct a positive, reduced rational duration multiplier.
    pub fn new(numerator: u64, denominator: u64) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        let divisor = gcd(numerator, denominator);
        Some(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    /// The identity scale 1/1.
    pub const fn identity() -> Self {
        Self {
            numerator: 1,
            denominator: 1,
        }
    }

    /// Return the reduced numerator.
    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    /// Return the reduced denominator.
    pub const fn denominator(self) -> u64 {
        self.denominator
    }

    /// Compose duration scales in execution order.
    pub fn compose(self, next: Self) -> Option<Self> {
        let cross_left = gcd(self.numerator, next.denominator);
        let cross_right = gcd(next.numerator, self.denominator);

        let left_numerator = self.numerator / cross_left;
        let right_denominator = next.denominator / cross_left;
        let right_numerator = next.numerator / cross_right;
        let left_denominator = self.denominator / cross_right;

        let numerator = left_numerator.checked_mul(right_numerator)?;
        let denominator = left_denominator.checked_mul(right_denominator)?;
        Self::new(numerator, denominator)
    }

    /// Return the reciprocal duration scale.
    pub const fn inverse(self) -> Self {
        Self {
            numerator: self.denominator,
            denominator: self.numerator,
        }
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Product action C12 × Q+ acting on pitch root and duration scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicalAction {
    transposition: Transposition,
    tempo_scale: TempoScale,
}

impl MusicalAction {
    /// Construct a product action.
    pub const fn new(transposition: Transposition, tempo_scale: TempoScale) -> Self {
        Self {
            transposition,
            tempo_scale,
        }
    }

    /// The product-group identity.
    pub const fn identity() -> Self {
        Self::new(Transposition::identity(), TempoScale::identity())
    }

    /// Compose in execution order.
    pub fn compose(self, next: Self) -> Option<Self> {
        Some(Self::new(
            self.transposition.compose(next.transposition),
            self.tempo_scale.compose(next.tempo_scale)?,
        ))
    }

    /// Return the product-group inverse.
    pub const fn inverse(self) -> Self {
        Self::new(self.transposition.inverse(), self.tempo_scale.inverse())
    }

    /// Transform a symbolic chord.
    pub const fn apply_chord(self, chord: Chord) -> Chord {
        chord.transpose(self.transposition)
    }

    /// Return the multiplicative duration factor for a rhythm query.
    pub const fn duration_scale(self) -> TempoScale {
        self.tempo_scale
    }
}

/// Musical action family covered by the first classification slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// Exact cyclic pitch-class transposition.
    C12Transposition,
    /// Exact positive multiplicative scaling of every event duration.
    ConstantTempoScaling,
    /// Expressive timing variation that is not a single global scale.
    Rubato,
    /// A lossy crop that removes part of the evidence.
    LossyCrop,
    /// A substitution that changes source timbre.
    InstrumentSubstitution,
    /// A new performer/take, not an invertible group action by default.
    PerformerTakeChange,
    /// Chord inversion; not admitted as a group action by this v0.
    ChordInversion,
}

/// Target property in a musical identity query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicalProperty {
    /// Major/minor/diminished/augmented quality.
    ChordQuality,
    /// Absolute root pitch class.
    RootPitchClass,
    /// Ordered root-relative interval signature.
    RelativeIntervals,
    /// Ratios among inter-onset intervals.
    RhythmicRatios,
    /// Pulse frequency / beats per minute.
    PulseRate,
    /// Absolute event duration.
    AbsoluteDuration,
    /// Instrument or source timbre.
    Timbre,
    /// Identity of the performer or performance take.
    PerformanceIdentity,
}

/// Evidence class supplied by the upstream qualification owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceClass {
    /// Exact finite action witness, e.g. a complete C12 table.
    ExactFiniteAction,
    /// Exact continuous action witness, e.g. positive rational scaling.
    ExactContinuousAction,
    /// Invertible only on a restricted declared subdomain.
    RestrictedPartialIsomorphism,
    /// Human-calibrated relation, not a formal action certificate.
    HumanCalibratedNuisance,
    /// Sampled or learned proposal only.
    SampledProposal,
    /// One-way or non-invertible relation.
    OneWayNonInvertible,
}

/// Coverage of the declared candidate domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainCoverage {
    /// All candidates in the declared query domain are covered.
    Complete,
    /// Only a subset is covered.
    Partial,
    /// Coverage is not established.
    Unknown,
}

/// Disposition of one action/property/domain combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Exact symbolic fixture is mathematically eligible for quotient testing.
    ///
    /// This is not permission to prune a production search.
    FixtureHardQuotientEligible,
    /// May be used for retrieval/soft grouping only.
    SoftRelationOnly,
    /// The property transforms equivariantly and is not invariant.
    EquivariantNotInvariant,
    /// The action can remove or destroy the evidence needed for the query.
    NotEstimable,
    /// The requested preservation relation is unsupported.
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PropertyEffect {
    Invariant,
    Equivariant,
    EmpiricalOnly,
    ErasesEvidence,
    Unsupported,
}

/// Classify a declared action against one query property and domain.
///
/// The result is deliberately conservative. Human and sampled relations
/// cannot qualify for hard quotienting. Partial/unknown domains cannot qualify
/// for hard quotienting. Upstream digests are not authenticated here.
pub fn classify_action(
    action: ActionKind,
    property: MusicalProperty,
    evidence: EvidenceClass,
    domain: DomainCoverage,
) -> Disposition {
    let effect = match (action, property) {
        (
            ActionKind::C12Transposition,
            MusicalProperty::ChordQuality | MusicalProperty::RelativeIntervals,
        ) => PropertyEffect::Invariant,
        (ActionKind::C12Transposition, MusicalProperty::RootPitchClass) => {
            PropertyEffect::Equivariant
        }
        (ActionKind::ConstantTempoScaling, MusicalProperty::RhythmicRatios) => {
            PropertyEffect::Invariant
        }
        (
            ActionKind::ConstantTempoScaling,
            MusicalProperty::PulseRate | MusicalProperty::AbsoluteDuration,
        ) => PropertyEffect::Equivariant,
        (
            ActionKind::PerformerTakeChange,
            MusicalProperty::ChordQuality
            | MusicalProperty::RelativeIntervals
            | MusicalProperty::RhythmicRatios,
        )
        | (ActionKind::Rubato, MusicalProperty::RhythmicRatios) => {
            PropertyEffect::EmpiricalOnly
        }
        (ActionKind::LossyCrop, _) => PropertyEffect::ErasesEvidence,
        _ => PropertyEffect::Unsupported,
    };

    match effect {
        PropertyEffect::Equivariant => Disposition::EquivariantNotInvariant,
        PropertyEffect::ErasesEvidence => Disposition::NotEstimable,
        PropertyEffect::EmpiricalOnly => {
            if evidence == EvidenceClass::HumanCalibratedNuisance {
                Disposition::SoftRelationOnly
            } else {
                Disposition::Unsupported
            }
        }
        PropertyEffect::Invariant => {
            let exact_pair = matches!(
                (action, evidence),
                (
                    ActionKind::C12Transposition,
                    EvidenceClass::ExactFiniteAction
                ) | (
                    ActionKind::ConstantTempoScaling,
                    EvidenceClass::ExactContinuousAction
                )
            );
            if !exact_pair {
                return match evidence {
                    EvidenceClass::HumanCalibratedNuisance
                    | EvidenceClass::SampledProposal
                    | EvidenceClass::RestrictedPartialIsomorphism => {
                        Disposition::SoftRelationOnly
                    }
                    EvidenceClass::OneWayNonInvertible => Disposition::NotEstimable,
                    EvidenceClass::ExactFiniteAction
                    | EvidenceClass::ExactContinuousAction => Disposition::Unsupported,
                };
            }
            match domain {
                DomainCoverage::Complete => Disposition::FixtureHardQuotientEligible,
                DomainCoverage::Partial => Disposition::SoftRelationOnly,
                DomainCoverage::Unknown => Disposition::Unsupported,
            }
        }
        PropertyEffect::Unsupported => Disposition::Unsupported,
    }
}

/// A witness that a C12 quotient representative lifts to an original chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChordLiftWitness {
    transposition: Transposition,
}

impl ChordLiftWitness {
    /// Return the representative-to-member transposition.
    pub const fn transposition(self) -> Transposition {
        self.transposition
    }
}

/// One original candidate and its reversible lift witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChordOrbitMember {
    /// Original, unquotiented candidate.
    pub candidate: Chord,
    /// Exact map from the canonical representative to this candidate.
    pub witness: ChordLiftWitness,
}

/// A complete finite orbit for one triad quality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChordQualityOrbit {
    /// Root-C representative with the same chord quality.
    pub representative: Chord,
    /// All twelve original roots and their lift witnesses.
    pub members: Vec<ChordOrbitMember>,
}

/// Failure to construct an exhaustive finite orbit partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitError {
    /// The input contains the same candidate more than once.
    DuplicateCandidate,
    /// An expected quality has no candidate group.
    MissingQuality(TriadQuality),
    /// A quality group does not contain all twelve pitch classes.
    IncompleteOrbit {
        quality: TriadQuality,
        observed: usize,
    },
    /// The input contains a quality not in the declared target set.
    UnexpectedQuality(TriadQuality),
}

/// Verify one representative/member lift against the original candidate.
pub const fn verify_chord_lift(
    representative: Chord,
    witness: ChordLiftWitness,
    candidate: Chord,
) -> bool {
    representative.transpose(witness.transposition) == candidate
}

/// Build exact C12 chord-quality orbits over a fully enumerated candidate set.
///
/// Every expected quality must contain exactly one candidate for each pitch
/// class. This is an exhaustive finite fixture, not a real-audio certificate.
pub fn build_complete_quality_orbits(
    candidates: &[Chord],
    expected_qualities: &[TriadQuality],
) -> Result<Vec<ChordQualityOrbit>, OrbitError> {
    let mut groups: BTreeMap<TriadQuality, BTreeMap<u8, Chord>> = BTreeMap::new();

    for candidate in candidates {
        let roots = groups.entry(candidate.quality()).or_default();
        if roots.insert(candidate.root().value(), *candidate).is_some() {
            return Err(OrbitError::DuplicateCandidate);
        }
    }

    let mut orbits = Vec::with_capacity(expected_qualities.len());
    for quality in expected_qualities {
        let roots = groups
            .remove(quality)
            .ok_or(OrbitError::MissingQuality(*quality))?;
        if roots.len() != usize::from(PITCH_CLASS_COUNT) {
            return Err(OrbitError::IncompleteOrbit {
                quality: *quality,
                observed: roots.len(),
            });
        }

        let representative = Chord::new(PitchClass(0), *quality);
        let mut members = Vec::with_capacity(usize::from(PITCH_CLASS_COUNT));
        for pitch_class in 0..PITCH_CLASS_COUNT {
            let candidate = *roots
                .get(&pitch_class)
                .ok_or(OrbitError::IncompleteOrbit {
                    quality: *quality,
                    observed: roots.len(),
                })?;
            let witness = ChordLiftWitness {
                transposition: Transposition::new(pitch_class),
            };
            if !verify_chord_lift(representative, witness, candidate) {
                return Err(OrbitError::IncompleteOrbit {
                    quality: *quality,
                    observed: roots.len(),
                });
            }
            members.push(ChordOrbitMember {
                candidate,
                witness,
            });
        }
        orbits.push(ChordQualityOrbit {
            representative,
            members,
        });
    }

    if let Some((quality, _)) = groups.into_iter().next() {
        return Err(OrbitError::UnexpectedQuality(quality));
    }
    Ok(orbits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUALITIES: [TriadQuality; 4] = [
        TriadQuality::Major,
        TriadQuality::Minor,
        TriadQuality::Diminished,
        TriadQuality::Augmented,
    ];

    fn chord(root: u8, quality: TriadQuality) -> Chord {
        Chord::new(PitchClass::new(root).expect("valid pitch class"), quality)
    }

    #[test]
    fn c12_identity_inverse_and_closure_hold_exhaustively() {
        let identity = Transposition::identity();
        for left in 0..PITCH_CLASS_COUNT {
            let action = Transposition::new(left);
            assert_eq!(action.compose(identity), action);
            assert_eq!(identity.compose(action), action);
            assert_eq!(action.compose(action.inverse()), identity);
            assert_eq!(action.inverse().compose(action), identity);
            for right in 0..PITCH_CLASS_COUNT {
                let composed = action.compose(Transposition::new(right));
                assert!(composed.semitones() < PITCH_CLASS_COUNT);
            }
        }
    }

    #[test]
    fn transposition_preserves_quality_and_relative_intervals_but_moves_root() {
        for quality in QUALITIES {
            let original = chord(2, quality);
            for offset in 0..PITCH_CLASS_COUNT {
                let transformed = original.transpose(Transposition::new(offset));
                assert_eq!(transformed.quality(), original.quality());
                assert_eq!(transformed.relative_intervals(), original.relative_intervals());
                assert_eq!(
                    transformed.root().value(),
                    (original.root().value() + offset) % PITCH_CLASS_COUNT
                );
            }
        }
    }

    #[test]
    fn pitch_class_constructor_rejects_out_of_domain_values() {
        assert!(PitchClass::new(0).is_some());
        assert!(PitchClass::new(11).is_some());
        assert!(PitchClass::new(12).is_none());
    }

    #[test]
    fn positive_rational_tempo_scales_compose_and_invert_exactly() {
        let scales = [
            TempoScale::new(1, 1).unwrap(),
            TempoScale::new(3, 2).unwrap(),
            TempoScale::new(5, 4).unwrap(),
            TempoScale::new(7, 6).unwrap(),
            TempoScale::new(12, 9).unwrap(),
        ];
        for left in scales {
            assert_eq!(left.compose(TempoScale::identity()), Some(left));
            assert_eq!(left.compose(left.inverse()), Some(TempoScale::identity()));
            assert_eq!(left.inverse().compose(left), Some(TempoScale::identity()));
            for right in scales {
                let composed = left.compose(right).unwrap();
                assert_eq!(composed.compose(right.inverse()), Some(left));
            }
        }
    }

    #[test]
    fn tempo_scale_rejects_zero_and_checked_overflow() {
        assert!(TempoScale::new(0, 1).is_none());
        assert!(TempoScale::new(1, 0).is_none());
        let huge = TempoScale::new(u64::MAX, 1).unwrap();
        assert!(huge.compose(huge).is_none());
    }

    #[test]
    fn independent_pitch_and_tempo_actions_commute() {
        let source = chord(7, TriadQuality::Minor);
        let pitch_only = MusicalAction::new(
            Transposition::new(5),
            TempoScale::identity(),
        );
        let tempo_only =
            MusicalAction::new(Transposition::identity(), TempoScale::new(3, 2).unwrap());
        let left = pitch_only.compose(tempo_only).unwrap();
        let right = tempo_only.compose(pitch_only).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.apply_chord(source), right.apply_chord(source));
        assert_eq!(left.duration_scale(), right.duration_scale());
        assert_eq!(
            left.compose(left.inverse()),
            Some(MusicalAction::identity())
        );
    }

    #[test]
    fn exact_action_property_matrix_is_query_relative() {
        assert_eq!(
            classify_action(
                ActionKind::C12Transposition,
                MusicalProperty::ChordQuality,
                EvidenceClass::ExactFiniteAction,
                DomainCoverage::Complete,
            ),
            Disposition::FixtureHardQuotientEligible
        );
        assert_eq!(
            classify_action(
                ActionKind::C12Transposition,
                MusicalProperty::RootPitchClass,
                EvidenceClass::ExactFiniteAction,
                DomainCoverage::Complete,
            ),
            Disposition::EquivariantNotInvariant
        );
        assert_eq!(
            classify_action(
                ActionKind::ConstantTempoScaling,
                MusicalProperty::RhythmicRatios,
                EvidenceClass::ExactContinuousAction,
                DomainCoverage::Complete,
            ),
            Disposition::FixtureHardQuotientEligible
        );
        assert_eq!(
            classify_action(
                ActionKind::ConstantTempoScaling,
                MusicalProperty::PulseRate,
                EvidenceClass::ExactContinuousAction,
                DomainCoverage::Complete,
            ),
            Disposition::EquivariantNotInvariant
        );
    }

    #[test]
    fn sampled_human_and_partial_relations_never_become_hard_quotients() {
        for (evidence, domain) in [
            (EvidenceClass::SampledProposal, DomainCoverage::Complete),
            (
                EvidenceClass::HumanCalibratedNuisance,
                DomainCoverage::Complete,
            ),
            (EvidenceClass::ExactFiniteAction, DomainCoverage::Partial),
        ] {
            let result = classify_action(
                ActionKind::C12Transposition,
                MusicalProperty::ChordQuality,
                evidence,
                domain,
            );
            assert_ne!(result, Disposition::FixtureHardQuotientEligible);
        }
        assert_eq!(
            classify_action(
                ActionKind::PerformerTakeChange,
                MusicalProperty::ChordQuality,
                EvidenceClass::HumanCalibratedNuisance,
                DomainCoverage::Complete,
            ),
            Disposition::SoftRelationOnly
        );
    }

    #[test]
    fn noninvertible_or_unsupported_edits_fail_closed() {
        assert_eq!(
            classify_action(
                ActionKind::LossyCrop,
                MusicalProperty::ChordQuality,
                EvidenceClass::OneWayNonInvertible,
                DomainCoverage::Complete,
            ),
            Disposition::NotEstimable
        );
        assert_eq!(
            classify_action(
                ActionKind::Rubato,
                MusicalProperty::RhythmicRatios,
                EvidenceClass::ExactContinuousAction,
                DomainCoverage::Complete,
            ),
            Disposition::Unsupported
        );
        assert_eq!(
            classify_action(
                ActionKind::ChordInversion,
                MusicalProperty::ChordQuality,
                EvidenceClass::ExactFiniteAction,
                DomainCoverage::Complete,
            ),
            Disposition::Unsupported
        );
        assert_eq!(
            classify_action(
                ActionKind::InstrumentSubstitution,
                MusicalProperty::Timbre,
                EvidenceClass::ExactFiniteAction,
                DomainCoverage::Complete,
            ),
            Disposition::Unsupported
        );
    }

    #[test]
    fn exhaustive_chord_quality_quotient_has_reversible_witnesses() {
        let candidates: Vec<Chord> = QUALITIES
            .into_iter()
            .flat_map(|quality| (0..PITCH_CLASS_COUNT).map(move |root| chord(root, quality)))
            .collect();
        let orbits = build_complete_quality_orbits(&candidates, &QUALITIES).unwrap();

        assert_eq!(candidates.len(), 48);
        assert_eq!(orbits.len(), 4);
        assert!(orbits.iter().all(|orbit| orbit.members.len() == 12));

        let mut lifted = Vec::new();
        for orbit in &orbits {
            for member in &orbit.members {
                assert!(verify_chord_lift(
                    orbit.representative,
                    member.witness,
                    member.candidate
                ));
                lifted.push(member.candidate);
            }
        }
        lifted.sort();
        let mut original = candidates;
        original.sort();
        assert_eq!(lifted, original);
    }

    #[test]
    fn incomplete_or_duplicate_orbit_is_rejected() {
        let incomplete: Vec<Chord> = (0..11)
            .map(|root| chord(root, TriadQuality::Major))
            .collect();
        assert_eq!(
            build_complete_quality_orbits(&incomplete, &[TriadQuality::Major]),
            Err(OrbitError::IncompleteOrbit {
                quality: TriadQuality::Major,
                observed: 11,
            })
        );

        let duplicate = [
            chord(0, TriadQuality::Major),
            chord(0, TriadQuality::Major),
        ];
        assert_eq!(
            build_complete_quality_orbits(&duplicate, &[TriadQuality::Major]),
            Err(OrbitError::DuplicateCandidate)
        );
    }

    #[test]
    fn v0_never_authorizes_production_pruning() {
        assert!(!RUNTIME_PRUNING_AUTHORIZED_V0_1);
    }
}
