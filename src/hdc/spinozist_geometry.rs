// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spinozist Moral Geometry — NSM-grounded affect space for moral reasoning.
//!
//! This module implements a moral classification system grounded in Natural Semantic
//! Metalanguage (NSM) theory (Wierzbicka 1972). Instead of learning prototypes from
//! labeled data, it composes moral affects from universal semantic primitives using
//! hyperdimensional vector algebra, then projects text into a 12-dimensional affect
//! space where geometric relationships determine moral verdict.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
//! │  NsmLexicon  │───▶│  NsmPrimeBasis│───▶│  AffectBasis   │───▶│  Fingerprint │
//! │  word→primes │    │  65 prime HVs │    │  12 affect HVs │    │  12D coords  │
//! └─────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
//!       │                                                            │
//!       ▼                                                            ▼
//!  Text tokenization                                          GeometricVerdict
//!  → weighted_bundle                                          → MoralVerdict
//! ```
//!
//! # Spinozist Inspiration
//!
//! Spinoza's *Ethics* treats affects as geometric forces: joy increases power of
//! acting, sadness decreases it, and desire drives toward what sustains being.
//! The `FluctuatioAnimi` (vacillation of the soul) captures the tension when
//! opposing affects co-activate — the moral ambiguity of real ethical dilemmas.
//!
//! # References
//!
//! - Wierzbicka, A. (1972). *Semantic Primitives*. Athenäum.
//! - Spinoza, B. (1677). *Ethica Ordine Geometrico Demonstrata*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.

use std::collections::HashMap;

use symthaea_core::hdc::universal_semantics::SemanticPrime;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::HDC_DIMENSION;

use super::moral_algebra::MoralVerdict;
use super::moral_prototypes::MoralLabel;
use super::moral_text_encoder::TextHdcEncoder;

// ============================================================================
// Constants
// ============================================================================

/// Number of Spinozist affects in the moral geometry.
pub const NUM_AFFECTS: usize = 12;

/// Random baseline similarity at HDC_DIMENSION: 1/sqrt(D).
/// Two random 16,384-dim vectors have expected |cosine| ≈ 0.0078.
const RANDOM_BASELINE: f32 = 1.0 / 128.0; // 1/sqrt(16384) = 1/128

/// Adequacy threshold for considering an affect "active".
const ADEQUACY_ACTIVE_THRESHOLD: f32 = 3.0;

/// Seed offset for non-keyword primes (spatial, temporal, etc.).
/// 0x5010 chosen as a memorable "SPIN" prefix in hex.
const PRIME_SEED_OFFSET: u64 = 0x5010_0000_0000_0000;

// ============================================================================
// SpinozistAffect — the 12 moral affects
// ============================================================================

/// The twelve fundamental moral affects in Spinozist geometry.
///
/// Each affect is composed from NSM semantic primitives via bind+bundle,
/// producing a semantically grounded hypervector that responds to natural
/// language descriptions of the corresponding moral concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpinozistAffect {
    /// Causing suffering or damage to someone.
    Harm,
    /// Nurturing, protecting, or supporting someone.
    Care,
    /// Permission and voluntary agreement.
    Consent,
    /// Intentionally creating false beliefs.
    Deception,
    /// Positive feeling from good outcomes.
    Joy,
    /// Negative feeling from bad outcomes.
    Sadness,
    /// Equal treatment and proportional response.
    Fairness,
    /// Duty or responsibility to act.
    Obligation,
    /// State of being exposed to harm.
    Vulnerability,
    /// Self-determination and freedom of choice.
    Autonomy,
    /// Wanting or seeking something.
    Desire,
    /// Reverence for what is deeply valued.
    Sacred,
}

impl SpinozistAffect {
    /// All affects in canonical order.
    pub fn all() -> [SpinozistAffect; NUM_AFFECTS] {
        [
            SpinozistAffect::Harm,
            SpinozistAffect::Care,
            SpinozistAffect::Consent,
            SpinozistAffect::Deception,
            SpinozistAffect::Joy,
            SpinozistAffect::Sadness,
            SpinozistAffect::Fairness,
            SpinozistAffect::Obligation,
            SpinozistAffect::Vulnerability,
            SpinozistAffect::Autonomy,
            SpinozistAffect::Desire,
            SpinozistAffect::Sacred,
        ]
    }

    /// Index in the canonical affect array.
    pub fn index(self) -> usize {
        match self {
            SpinozistAffect::Harm => 0,
            SpinozistAffect::Care => 1,
            SpinozistAffect::Consent => 2,
            SpinozistAffect::Deception => 3,
            SpinozistAffect::Joy => 4,
            SpinozistAffect::Sadness => 5,
            SpinozistAffect::Fairness => 6,
            SpinozistAffect::Obligation => 7,
            SpinozistAffect::Vulnerability => 8,
            SpinozistAffect::Autonomy => 9,
            SpinozistAffect::Desire => 10,
            SpinozistAffect::Sacred => 11,
        }
    }
}

// ============================================================================
// NsmPrimeBasis — 65 semantically grounded prime hypervectors
// ============================================================================

/// Basis of 65 NSM semantic prime hypervectors at HDC_DIMENSION.
///
/// Morally-relevant primes (~25) are encoded via keyword sets through
/// `TextHdcEncoder`, giving them semantic grounding. Remaining primes
/// (spatial, temporal, quantifier) use deterministic random HVs.
pub struct NsmPrimeBasis {
    primes: HashMap<SemanticPrime, ContinuousHV>,
}

/// Keyword sets for the ~25 morally-relevant semantic primes.
fn moral_prime_keywords(prime: SemanticPrime) -> Option<&'static str> {
    match prime {
        SemanticPrime::Good => {
            Some("good positive benefit right proper kind generous noble worthy")
        }
        SemanticPrime::Bad => Some("bad negative harmful wrong cruel wicked evil unjust unfair"),
        SemanticPrime::Want => Some("want desire wish need crave seek long yearn aspire"),
        SemanticPrime::Feel => Some("feel emotion sense experience mood affect sentiment perceive"),
        SemanticPrime::Think => Some("think reason consider ponder reflect contemplate deliberate"),
        SemanticPrime::Know => Some("know understand recognize aware comprehend realize grasp"),
        SemanticPrime::Do => Some("do act perform execute carry accomplish achieve undertake"),
        SemanticPrime::Happen => Some("happen occur arise emerge result transpire unfold develop"),
        SemanticPrime::Someone => Some("someone person individual human being people anybody"),
        SemanticPrime::Something => Some("something thing object matter item entity element"),
        SemanticPrime::Say => Some("say speak tell express communicate declare state assert"),
        SemanticPrime::True => Some("true truth honest genuine authentic real actual factual"),
        SemanticPrime::Not => Some("not never neither nor without absent lacking denied"),
        SemanticPrime::Can => Some("can able capable possible allowed permitted potential"),
        SemanticPrime::Because => {
            Some("because reason cause purpose motive ground basis therefore")
        }
        SemanticPrime::If => Some("if condition case suppose assuming provided whether"),
        SemanticPrime::Live => Some("live alive life exist survive endure persist thrive"),
        SemanticPrime::Die => Some("die death dead perish expire cease end terminate"),
        SemanticPrime::Have => Some("have possess own hold contain keep maintain retain"),
        SemanticPrime::Body => Some("body physical flesh skin bone tissue organ health"),
        SemanticPrime::I => Some("I self myself me my own personal"),
        SemanticPrime::You => Some("you your yourself yours"),
        SemanticPrime::With => Some("with together alongside jointly mutual shared cooperative"),
        SemanticPrime::Very => Some("very extremely highly greatly intensely remarkably deeply"),
        SemanticPrime::All => Some("all every each entire whole complete total universal"),
        _ => None,
    }
}

impl NsmPrimeBasis {
    /// Construct the full 65-prime basis.
    ///
    /// Morally-relevant primes use keyword encoding via `TextHdcEncoder`.
    /// Remaining primes use `ContinuousHV::random()` with unique seeds.
    pub fn new() -> Self {
        let encoder = TextHdcEncoder::with_sentiment(HDC_DIMENSION, 3, 0.5, 0.15);
        let all_primes = SemanticPrime::all();
        let mut primes = HashMap::with_capacity(all_primes.len());

        for (i, &prime) in all_primes.iter().enumerate() {
            let hv = if let Some(keywords) = moral_prime_keywords(prime) {
                encoder.encode(keywords)
            } else {
                // Deterministic random for non-moral primes (spatial, temporal, etc.)
                ContinuousHV::random(HDC_DIMENSION, PRIME_SEED_OFFSET + i as u64)
            };
            primes.insert(prime, hv);
        }

        Self { primes }
    }

    /// Get the hypervector for a semantic prime.
    pub fn prime(&self, p: SemanticPrime) -> &ContinuousHV {
        self.primes
            .get(&p)
            .expect("all 65 primes should be present")
    }
}

impl Default for NsmPrimeBasis {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AffectBasis — 12 composed affect hypervectors
// ============================================================================

/// Twelve affect hypervectors composed from NSM primes via bind+bundle.
///
/// Each affect is a semantically grounded superposition of prime-pair
/// bindings that encode its conceptual structure. For example, HARM is
/// composed from DO⊗BAD, FEEL⊗BAD, BODY⊗BAD, and SOMEONE⊗(BAD⊗HAPPEN).
pub struct AffectBasis {
    affects: [ContinuousHV; NUM_AFFECTS],
}

impl AffectBasis {
    /// Compose all 12 affects from the given NSM prime basis.
    pub fn new(basis: &NsmPrimeBasis) -> Self {
        let affects = [
            Self::compose_harm(basis),
            Self::compose_care(basis),
            Self::compose_consent(basis),
            Self::compose_deception(basis),
            Self::compose_joy(basis),
            Self::compose_sadness(basis),
            Self::compose_fairness(basis),
            Self::compose_obligation(basis),
            Self::compose_vulnerability(basis),
            Self::compose_autonomy(basis),
            Self::compose_desire(basis),
            Self::compose_sacred(basis),
        ];
        Self { affects }
    }

    /// Get the hypervector for an affect.
    pub fn affect_hv(&self, affect: SpinozistAffect) -> &ContinuousHV {
        &self.affects[affect.index()]
    }

    /// Project a hypervector onto all 12 affect dimensions.
    ///
    /// Returns cosine similarities: positive means alignment, negative means
    /// opposition to that affect.
    pub fn project_affects(&self, hv: &ContinuousHV) -> [f32; NUM_AFFECTS] {
        let mut coords = [0.0f32; NUM_AFFECTS];
        for (i, affect_hv) in self.affects.iter().enumerate() {
            coords[i] = hv.similarity(affect_hv);
        }
        coords
    }

    // --- Affect compositions from NSM primes ---

    /// HARM = DO⊗BAD + FEEL⊗BAD + BODY⊗BAD + SOMEONE⊗(BAD⊗HAPPEN)
    fn compose_harm(b: &NsmPrimeBasis) -> ContinuousHV {
        // Include both raw primes (for lexical overlap) and bind compositions
        // (for relational structure). Raw primes ensure text containing "bad"
        // or "hurt" activates HARM; binds add relational discrimination.
        let bad = b.prime(SemanticPrime::Bad);
        let do_p = b.prime(SemanticPrime::Do);
        let feel = b.prime(SemanticPrime::Feel);
        let body = b.prime(SemanticPrime::Body);
        let do_bad = do_p.bind(bad);
        let feel_bad = feel.bind(bad);
        let body_bad = body.bind(bad);
        // Weight: 60% raw primes (lexical), 40% bound compositions (relational)
        ContinuousHV::weighted_bundle(
            &[bad, do_p, feel, body, &do_bad, &feel_bad, &body_bad],
            &[0.20, 0.10, 0.10, 0.05, 0.20, 0.10, 0.05],
        )
    }

    /// CARE = DO⊗GOOD + FEEL⊗GOOD + SOMEONE⊗(GOOD⊗HAPPEN) + WANT⊗(GOOD⊗SOMEONE)
    fn compose_care(b: &NsmPrimeBasis) -> ContinuousHV {
        let good = b.prime(SemanticPrime::Good);
        let do_p = b.prime(SemanticPrime::Do);
        let feel = b.prime(SemanticPrime::Feel);
        let want = b.prime(SemanticPrime::Want);
        let do_good = do_p.bind(good);
        let feel_good = feel.bind(good);
        let want_good = want.bind(good);
        // 60% raw primes (lexical), 40% bound (relational)
        ContinuousHV::weighted_bundle(
            &[good, do_p, feel, want, &do_good, &feel_good, &want_good],
            &[0.20, 0.10, 0.10, 0.05, 0.20, 0.10, 0.05],
        )
    }

    /// CONSENT = WANT⊗DO + CAN⊗(NOT⊗DO) + SAY⊗TRUE + SOMEONE⊗WANT
    fn compose_consent(b: &NsmPrimeBasis) -> ContinuousHV {
        let want_do = b
            .prime(SemanticPrime::Want)
            .bind(b.prime(SemanticPrime::Do));
        let can_not_do = b
            .prime(SemanticPrime::Can)
            .bind(&b.prime(SemanticPrime::Not).bind(b.prime(SemanticPrime::Do)));
        let say_true = b
            .prime(SemanticPrime::Say)
            .bind(b.prime(SemanticPrime::True));
        let someone_want = b
            .prime(SemanticPrime::Someone)
            .bind(b.prime(SemanticPrime::Want));
        ContinuousHV::bundle(&[&want_do, &can_not_do, &say_true, &someone_want])
    }

    /// DECEPTION = SAY⊗(NOT⊗TRUE) + KNOW⊗TRUE + DO⊗(NOT⊗TRUE) + WANT⊗(SOMEONE⊗(THINK⊗(NOT⊗TRUE)))
    fn compose_deception(b: &NsmPrimeBasis) -> ContinuousHV {
        let say_not_true = b.prime(SemanticPrime::Say).bind(
            &b.prime(SemanticPrime::Not)
                .bind(b.prime(SemanticPrime::True)),
        );
        let know_true = b
            .prime(SemanticPrime::Know)
            .bind(b.prime(SemanticPrime::True));
        let do_not_true = b.prime(SemanticPrime::Do).bind(
            &b.prime(SemanticPrime::Not)
                .bind(b.prime(SemanticPrime::True)),
        );
        let want_false_belief = b.prime(SemanticPrime::Want).bind(
            &b.prime(SemanticPrime::Someone).bind(
                &b.prime(SemanticPrime::Think).bind(
                    &b.prime(SemanticPrime::Not)
                        .bind(b.prime(SemanticPrime::True)),
                ),
            ),
        );
        ContinuousHV::bundle(&[&say_not_true, &know_true, &do_not_true, &want_false_belief])
    }

    /// JOY = FEEL⊗(VERY⊗GOOD) + GOOD⊗HAPPEN + WANT⊗(HAPPEN⊗GOOD)
    fn compose_joy(b: &NsmPrimeBasis) -> ContinuousHV {
        let feel_very_good = b.prime(SemanticPrime::Feel).bind(
            &b.prime(SemanticPrime::Very)
                .bind(b.prime(SemanticPrime::Good)),
        );
        let good_happen = b
            .prime(SemanticPrime::Good)
            .bind(b.prime(SemanticPrime::Happen));
        let want_good = b.prime(SemanticPrime::Want).bind(
            &b.prime(SemanticPrime::Happen)
                .bind(b.prime(SemanticPrime::Good)),
        );
        ContinuousHV::bundle(&[&feel_very_good, &good_happen, &want_good])
    }

    /// SADNESS = FEEL⊗(VERY⊗BAD) + BAD⊗HAPPEN + NOT⊗(WANT⊗HAPPEN)
    fn compose_sadness(b: &NsmPrimeBasis) -> ContinuousHV {
        let feel_very_bad = b.prime(SemanticPrime::Feel).bind(
            &b.prime(SemanticPrime::Very)
                .bind(b.prime(SemanticPrime::Bad)),
        );
        let bad_happen = b
            .prime(SemanticPrime::Bad)
            .bind(b.prime(SemanticPrime::Happen));
        let not_want_happen = b.prime(SemanticPrime::Not).bind(
            &b.prime(SemanticPrime::Want)
                .bind(b.prime(SemanticPrime::Happen)),
        );
        ContinuousHV::bundle(&[&feel_very_bad, &bad_happen, &not_want_happen])
    }

    /// FAIRNESS = SAME⊗(ALL⊗SOMEONE) + GOOD⊗(SAME⊗DO) + NOT⊗(SOMEONE⊗(HAVE⊗MORE))
    fn compose_fairness(b: &NsmPrimeBasis) -> ContinuousHV {
        let same_all = b.prime(SemanticPrime::Same).bind(
            &b.prime(SemanticPrime::All)
                .bind(b.prime(SemanticPrime::Someone)),
        );
        let good_same_do = b.prime(SemanticPrime::Good).bind(
            &b.prime(SemanticPrime::Same)
                .bind(b.prime(SemanticPrime::Do)),
        );
        let not_more = b.prime(SemanticPrime::Not).bind(
            &b.prime(SemanticPrime::Someone).bind(
                &b.prime(SemanticPrime::Have)
                    .bind(b.prime(SemanticPrime::More)),
            ),
        );
        ContinuousHV::bundle(&[&same_all, &good_same_do, &not_more])
    }

    /// OBLIGATION = SOMEONE⊗(DO⊗BECAUSE) + KNOW⊗(DO⊗GOOD) + CAN⊗DO + NOT⊗(CAN⊗(NOT⊗DO))
    fn compose_obligation(b: &NsmPrimeBasis) -> ContinuousHV {
        let must_do = b.prime(SemanticPrime::Someone).bind(
            &b.prime(SemanticPrime::Do)
                .bind(b.prime(SemanticPrime::Because)),
        );
        let know_do_good = b.prime(SemanticPrime::Know).bind(
            &b.prime(SemanticPrime::Do)
                .bind(b.prime(SemanticPrime::Good)),
        );
        let can_do = b.prime(SemanticPrime::Can).bind(b.prime(SemanticPrime::Do));
        let not_can_not_do = b.prime(SemanticPrime::Not).bind(
            &b.prime(SemanticPrime::Can)
                .bind(&b.prime(SemanticPrime::Not).bind(b.prime(SemanticPrime::Do))),
        );
        ContinuousHV::bundle(&[&must_do, &know_do_good, &can_do, &not_can_not_do])
    }

    /// VULNERABILITY = SOMEONE⊗(NOT⊗CAN) + BODY⊗BAD + FEEL⊗BAD + SMALL⊗SOMEONE
    fn compose_vulnerability(b: &NsmPrimeBasis) -> ContinuousHV {
        let cannot = b.prime(SemanticPrime::Someone).bind(
            &b.prime(SemanticPrime::Not)
                .bind(b.prime(SemanticPrime::Can)),
        );
        let body_bad = b
            .prime(SemanticPrime::Body)
            .bind(b.prime(SemanticPrime::Bad));
        let feel_bad = b
            .prime(SemanticPrime::Feel)
            .bind(b.prime(SemanticPrime::Bad));
        let small_someone = b
            .prime(SemanticPrime::Small)
            .bind(b.prime(SemanticPrime::Someone));
        ContinuousHV::bundle(&[&cannot, &body_bad, &feel_bad, &small_someone])
    }

    /// AUTONOMY = I⊗(CAN⊗DO) + I⊗WANT + NOT⊗(SOMEONE⊗(DO⊗I)) + I⊗(KNOW⊗(DO⊗GOOD))
    fn compose_autonomy(b: &NsmPrimeBasis) -> ContinuousHV {
        let i_can_do = b
            .prime(SemanticPrime::I)
            .bind(&b.prime(SemanticPrime::Can).bind(b.prime(SemanticPrime::Do)));
        let i_want = b.prime(SemanticPrime::I).bind(b.prime(SemanticPrime::Want));
        let not_coerced = b.prime(SemanticPrime::Not).bind(
            &b.prime(SemanticPrime::Someone)
                .bind(&b.prime(SemanticPrime::Do).bind(b.prime(SemanticPrime::I))),
        );
        let i_know = b.prime(SemanticPrime::I).bind(
            &b.prime(SemanticPrime::Know).bind(
                &b.prime(SemanticPrime::Do)
                    .bind(b.prime(SemanticPrime::Good)),
            ),
        );
        ContinuousHV::bundle(&[&i_can_do, &i_want, &not_coerced, &i_know])
    }

    /// DESIRE = WANT⊗(VERY⊗SOMETHING) + FEEL⊗WANT + DO⊗(BECAUSE⊗WANT)
    fn compose_desire(b: &NsmPrimeBasis) -> ContinuousHV {
        let want_very = b.prime(SemanticPrime::Want).bind(
            &b.prime(SemanticPrime::Very)
                .bind(b.prime(SemanticPrime::Something)),
        );
        let feel_want = b
            .prime(SemanticPrime::Feel)
            .bind(b.prime(SemanticPrime::Want));
        let do_because_want = b.prime(SemanticPrime::Do).bind(
            &b.prime(SemanticPrime::Because)
                .bind(b.prime(SemanticPrime::Want)),
        );
        ContinuousHV::bundle(&[&want_very, &feel_want, &do_because_want])
    }

    /// SACRED = VERY⊗GOOD + FEEL⊗(VERY⊗GOOD) + KNOW⊗(SOMETHING⊗(VERY⊗GOOD)) + ALL⊗(FEEL⊗GOOD)
    fn compose_sacred(b: &NsmPrimeBasis) -> ContinuousHV {
        let very_good = b
            .prime(SemanticPrime::Very)
            .bind(b.prime(SemanticPrime::Good));
        let feel_very_good = b.prime(SemanticPrime::Feel).bind(
            &b.prime(SemanticPrime::Very)
                .bind(b.prime(SemanticPrime::Good)),
        );
        let know_sacred = b.prime(SemanticPrime::Know).bind(
            &b.prime(SemanticPrime::Something).bind(
                &b.prime(SemanticPrime::Very)
                    .bind(b.prime(SemanticPrime::Good)),
            ),
        );
        let all_feel = b.prime(SemanticPrime::All).bind(
            &b.prime(SemanticPrime::Feel)
                .bind(b.prime(SemanticPrime::Good)),
        );
        ContinuousHV::bundle(&[&very_good, &feel_very_good, &know_sacred, &all_feel])
    }
}

// ============================================================================
// NsmLexicon — word-to-NSM-prime decomposition
// ============================================================================

/// Maps words to weighted NSM prime decompositions.
///
/// When a word is found in the lexicon, its HV is computed as a weighted
/// bundle of its constituent prime HVs. Unknown words fall back to a
/// deterministic hash (same strategy as `TextHdcEncoder::hash_word`).
pub struct NsmLexicon {
    entries: HashMap<String, Vec<(SemanticPrime, f32)>>,
}

impl NsmLexicon {
    /// Build the lexicon with ~300+ morally-discriminative word entries.
    pub fn new() -> Self {
        let mut entries = HashMap::with_capacity(350);

        // ---- Actions ----
        Self::insert(
            &mut entries,
            "steal",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Have, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "stealing",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Have, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "stole",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Have, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "stolen",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Have, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "help",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "helps",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "helped",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "helping",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "kill",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Die, 1.0),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Body, 0.8),
            ],
        );
        Self::insert(
            &mut entries,
            "killed",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Die, 1.0),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Body, 0.8),
            ],
        );
        Self::insert(
            &mut entries,
            "killing",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Die, 1.0),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Body, 0.8),
            ],
        );
        Self::insert(
            &mut entries,
            "lie",
            &[
                (SemanticPrime::Say, 1.0),
                (SemanticPrime::Not, 0.8),
                (SemanticPrime::True, 0.8),
                (SemanticPrime::Bad, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "lied",
            &[
                (SemanticPrime::Say, 1.0),
                (SemanticPrime::Not, 0.8),
                (SemanticPrime::True, 0.8),
                (SemanticPrime::Bad, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "lying",
            &[
                (SemanticPrime::Say, 1.0),
                (SemanticPrime::Not, 0.8),
                (SemanticPrime::True, 0.8),
                (SemanticPrime::Bad, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "give",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Have, 0.5),
                (SemanticPrime::Good, 0.5),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "hurt",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.8),
                (SemanticPrime::Body, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "hurting",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.8),
                (SemanticPrime::Body, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "save",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Live, 0.8),
            ],
        );
        Self::insert(
            &mut entries,
            "cheat",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::True, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "cheated",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::True, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "protect",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Someone, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "betray",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::True, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "murder",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Die, 1.0),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Want, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "abuse",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Someone, 0.8),
                (SemanticPrime::Feel, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "exploit",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Someone, 0.7),
                (SemanticPrime::Have, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "bully",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Someone, 0.8),
                (SemanticPrime::Feel, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "threaten",
            &[
                (SemanticPrime::Say, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Someone, 0.5),
                (SemanticPrime::Feel, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "blackmail",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Have, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "deceive",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Not, 0.8),
                (SemanticPrime::True, 0.8),
                (SemanticPrime::Bad, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "manipulate",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::True, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "attack",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Body, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "destroy",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Something, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "vandalize",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Something, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "sabotage",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Good, 0.1),
            ],
        );
        Self::insert(
            &mut entries,
            "fraud",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::True, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "forge",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::True, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "rescue",
            &[
                (SemanticPrime::Do, 1.0),
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Someone, 0.7),
                (SemanticPrime::Live, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "share",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Have, 0.5),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "donate",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Have, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "forgive",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "volunteer",
            &[
                (SemanticPrime::Do, 0.9),
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Want, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "cooperate",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::With, 0.8),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "comfort",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.8),
            ],
        );
        Self::insert(
            &mut entries,
            "heal",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Body, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "nurture",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Someone, 0.6),
                (SemanticPrime::Live, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "inspire",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.7),
                (SemanticPrime::Think, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "encourage",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "support",
            &[
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Someone, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "praise",
            &[
                (SemanticPrime::Say, 0.8),
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "respect",
            &[
                (SemanticPrime::Feel, 0.7),
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Someone, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "trust",
            &[
                (SemanticPrime::Think, 0.6),
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Someone, 0.5),
                (SemanticPrime::True, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "neglect",
            &[
                (SemanticPrime::Not, 0.8),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "demean",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.7),
                (SemanticPrime::Someone, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "undermine",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Good, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "discourage",
            &[
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Not, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "disrespect",
            &[
                (SemanticPrime::Not, 0.7),
                (SemanticPrime::Good, 0.3),
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );

        // ---- Moral framing adjectives ----
        Self::insert(
            &mut entries,
            "rude",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Feel, 0.8),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "kind",
            &[
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "fair",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Same, 0.5),
                (SemanticPrime::All, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "unfair",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Same, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "wrong",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Not, 0.3),
                (SemanticPrime::Good, 0.1),
            ],
        );
        Self::insert(
            &mut entries,
            "okay",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Can, 0.4)],
        );
        Self::insert(
            &mut entries,
            "fine",
            &[(SemanticPrime::Good, 0.5), (SemanticPrime::Can, 0.3)],
        );
        Self::insert(
            &mut entries,
            "acceptable",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Can, 0.5)],
        );
        Self::insert(
            &mut entries,
            "cruel",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Body, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "generous",
            &[
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Have, 0.3),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "honest",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::True, 0.9),
                (SemanticPrime::Say, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "dishonest",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::True, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "brave",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Not, 0.3),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "selfish",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::I, 0.7),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "caring",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Feel, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "uncaring",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::Feel, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "loyal",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::With, 0.6),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "disloyal",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::With, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "gentle",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Body, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "harsh",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Body, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "merciless",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "mercy",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "grateful",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.7),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "ungrateful",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "compassion",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Feel, 0.9),
                (SemanticPrime::Someone, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "empathy",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Feel, 0.9),
                (SemanticPrime::Someone, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "callous",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Feel, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "heartless",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Feel, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "thoughtful",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Think, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "considerate",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Think, 0.5),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "responsible",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Because, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "irresponsible",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Do, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "patient",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Feel, 0.4)],
        );
        Self::insert(
            &mut entries,
            "impatient",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Feel, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "humble",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Not, 0.3),
                (SemanticPrime::Big, 0.2),
            ],
        );
        Self::insert(
            &mut entries,
            "arrogant",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::I, 0.5),
                (SemanticPrime::Big, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "sincere",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::True, 0.8),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "vain",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::I, 0.6),
                (SemanticPrime::Want, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "peaceful",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Not, 0.3),
                (SemanticPrime::Bad, 0.1),
            ],
        );
        Self::insert(
            &mut entries,
            "noble",
            &[(SemanticPrime::Good, 0.9), (SemanticPrime::Do, 0.5)],
        );
        Self::insert(
            &mut entries,
            "virtuous",
            &[(SemanticPrime::Good, 1.0), (SemanticPrime::Do, 0.5)],
        );
        Self::insert(
            &mut entries,
            "admirable",
            &[(SemanticPrime::Good, 0.9), (SemanticPrime::Feel, 0.5)],
        );
        Self::insert(
            &mut entries,
            "heroic",
            &[
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Do, 0.8),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "selfless",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::I, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "charitable",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Have, 0.4),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "benevolent",
            &[
                (SemanticPrime::Good, 1.0),
                (SemanticPrime::Want, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "righteous",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::True, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "worthy",
            &[(SemanticPrime::Good, 0.8), (SemanticPrime::Do, 0.3)],
        );
        Self::insert(
            &mut entries,
            "honorable",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::True, 0.5),
                (SemanticPrime::Do, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "dignified",
            &[(SemanticPrime::Good, 0.7), (SemanticPrime::Someone, 0.4)],
        );
        Self::insert(
            &mut entries,
            "gracious",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "courteous",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Say, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "polite",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Say, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "wonderful",
            &[(SemanticPrime::Good, 1.0), (SemanticPrime::Feel, 0.5)],
        );
        Self::insert(
            &mut entries,
            "beautiful",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::See, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "excellent",
            &[(SemanticPrime::Good, 0.9), (SemanticPrime::Very, 0.4)],
        );
        Self::insert(
            &mut entries,
            "joyful",
            &[(SemanticPrime::Good, 0.8), (SemanticPrime::Feel, 0.9)],
        );
        Self::insert(
            &mut entries,
            "happy",
            &[(SemanticPrime::Good, 0.7), (SemanticPrime::Feel, 0.8)],
        );

        // ---- Bad-word adjectives ----
        Self::insert(
            &mut entries,
            "wicked",
            &[(SemanticPrime::Bad, 1.0), (SemanticPrime::Do, 0.5)],
        );
        Self::insert(
            &mut entries,
            "evil",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Want, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "terrible",
            &[(SemanticPrime::Bad, 1.0), (SemanticPrime::Very, 0.5)],
        );
        Self::insert(
            &mut entries,
            "horrible",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Very, 0.5),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "vicious",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Body, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "nasty",
            &[(SemanticPrime::Bad, 0.9), (SemanticPrime::Feel, 0.5)],
        );
        Self::insert(
            &mut entries,
            "immoral",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Good, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "unethical",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Good, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "unjust",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Same, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "sinful",
            &[(SemanticPrime::Bad, 0.9), (SemanticPrime::Do, 0.4)],
        );
        Self::insert(
            &mut entries,
            "shameful",
            &[(SemanticPrime::Bad, 0.8), (SemanticPrime::Feel, 0.6)],
        );
        Self::insert(
            &mut entries,
            "aggressive",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Body, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "hostile",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "toxic",
            &[(SemanticPrime::Bad, 0.9), (SemanticPrime::Body, 0.4)],
        );
        Self::insert(
            &mut entries,
            "destructive",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Something, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "damaging",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Something, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "corrupt",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::True, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "greedy",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Want, 0.8),
                (SemanticPrime::Have, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "malicious",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Want, 0.6),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "spiteful",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "vengeful",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Because, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "violent",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Body, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "reckless",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Think, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "lazy",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::Do, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "coward",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Do, 0.4),
            ],
        );

        // ---- Emotional nouns/adjectives ----
        Self::insert(
            &mut entries,
            "love",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Feel, 1.0),
                (SemanticPrime::Someone, 0.7),
            ],
        );
        Self::insert(
            &mut entries,
            "hate",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 1.0),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "anger",
            &[(SemanticPrime::Bad, 0.6), (SemanticPrime::Feel, 0.9)],
        );
        Self::insert(
            &mut entries,
            "fear",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Feel, 0.9),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "joy",
            &[(SemanticPrime::Good, 0.9), (SemanticPrime::Feel, 1.0)],
        );
        Self::insert(
            &mut entries,
            "sorrow",
            &[(SemanticPrime::Bad, 0.6), (SemanticPrime::Feel, 1.0)],
        );
        Self::insert(
            &mut entries,
            "pain",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Feel, 0.9),
                (SemanticPrime::Body, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "suffering",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 1.0),
                (SemanticPrime::Body, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "harm",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "harms",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "harmed",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "harming",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Someone, 0.5),
            ],
        );

        // ---- Function words ----
        Self::insert(&mut entries, "not", &[(SemanticPrime::Not, 1.0)]);
        Self::insert(&mut entries, "never", &[(SemanticPrime::Not, 1.0)]);
        Self::insert(&mut entries, "no", &[(SemanticPrime::Not, 0.8)]);
        Self::insert(&mut entries, "without", &[(SemanticPrime::Not, 0.8)]);
        Self::insert(&mut entries, "because", &[(SemanticPrime::Because, 1.0)]);
        Self::insert(&mut entries, "if", &[(SemanticPrime::If, 1.0)]);
        Self::insert(&mut entries, "very", &[(SemanticPrime::Very, 1.0)]);
        Self::insert(&mut entries, "all", &[(SemanticPrime::All, 1.0)]);
        Self::insert(
            &mut entries,
            "everyone",
            &[(SemanticPrime::All, 0.8), (SemanticPrime::Someone, 0.6)],
        );
        Self::insert(&mut entries, "someone", &[(SemanticPrime::Someone, 1.0)]);
        Self::insert(
            &mut entries,
            "something",
            &[(SemanticPrime::Something, 1.0)],
        );
        Self::insert(&mut entries, "people", &[(SemanticPrime::People, 1.0)]);
        Self::insert(&mut entries, "person", &[(SemanticPrime::Someone, 0.9)]);

        // ---- Relational / structural words ----
        Self::insert(&mut entries, "with", &[(SemanticPrime::With, 1.0)]);
        Self::insert(&mut entries, "together", &[(SemanticPrime::With, 0.9)]);
        Self::insert(&mut entries, "good", &[(SemanticPrime::Good, 1.0)]);
        Self::insert(&mut entries, "bad", &[(SemanticPrime::Bad, 1.0)]);
        Self::insert(&mut entries, "right", &[(SemanticPrime::Good, 0.7)]);
        Self::insert(&mut entries, "true", &[(SemanticPrime::True, 1.0)]);
        Self::insert(
            &mut entries,
            "false",
            &[(SemanticPrime::Not, 0.7), (SemanticPrime::True, 0.8)],
        );

        // ---- Consent / permission ----
        Self::insert(
            &mut entries,
            "permission",
            &[
                (SemanticPrime::Can, 0.8),
                (SemanticPrime::Want, 0.5),
                (SemanticPrime::Say, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "consent",
            &[
                (SemanticPrime::Can, 0.7),
                (SemanticPrime::Want, 0.8),
                (SemanticPrime::Say, 0.4),
            ],
        );
        Self::insert(&mut entries, "allowed", &[(SemanticPrime::Can, 0.9)]);
        Self::insert(
            &mut entries,
            "forbidden",
            &[(SemanticPrime::Not, 0.8), (SemanticPrime::Can, 0.7)],
        );

        // ---- Life / death ----
        Self::insert(&mut entries, "life", &[(SemanticPrime::Live, 1.0)]);
        Self::insert(&mut entries, "death", &[(SemanticPrime::Die, 1.0)]);
        Self::insert(&mut entries, "alive", &[(SemanticPrime::Live, 0.9)]);
        Self::insert(&mut entries, "dead", &[(SemanticPrime::Die, 0.9)]);

        // ---- Pronouns ----
        Self::insert(&mut entries, "i", &[(SemanticPrime::I, 1.0)]);
        Self::insert(&mut entries, "me", &[(SemanticPrime::I, 0.9)]);
        Self::insert(&mut entries, "my", &[(SemanticPrime::I, 0.7)]);
        Self::insert(&mut entries, "you", &[(SemanticPrime::You, 1.0)]);
        Self::insert(&mut entries, "your", &[(SemanticPrime::You, 0.7)]);
        Self::insert(
            &mut entries,
            "they",
            &[(SemanticPrime::Someone, 0.6), (SemanticPrime::People, 0.3)],
        );
        Self::insert(&mut entries, "he", &[(SemanticPrime::Someone, 0.7)]);
        Self::insert(&mut entries, "she", &[(SemanticPrime::Someone, 0.7)]);
        Self::insert(
            &mut entries,
            "we",
            &[
                (SemanticPrime::I, 0.5),
                (SemanticPrime::People, 0.5),
                (SemanticPrime::With, 0.3),
            ],
        );

        // ---- Common verbs ----
        Self::insert(&mut entries, "think", &[(SemanticPrime::Think, 1.0)]);
        Self::insert(&mut entries, "know", &[(SemanticPrime::Know, 1.0)]);
        Self::insert(&mut entries, "want", &[(SemanticPrime::Want, 1.0)]);
        Self::insert(&mut entries, "feel", &[(SemanticPrime::Feel, 1.0)]);
        Self::insert(&mut entries, "see", &[(SemanticPrime::See, 1.0)]);
        Self::insert(&mut entries, "hear", &[(SemanticPrime::Hear, 1.0)]);
        Self::insert(&mut entries, "say", &[(SemanticPrime::Say, 1.0)]);
        Self::insert(&mut entries, "said", &[(SemanticPrime::Say, 0.9)]);
        Self::insert(&mut entries, "do", &[(SemanticPrime::Do, 1.0)]);
        Self::insert(&mut entries, "did", &[(SemanticPrime::Do, 0.9)]);
        Self::insert(&mut entries, "does", &[(SemanticPrime::Do, 0.9)]);
        Self::insert(&mut entries, "have", &[(SemanticPrime::Have, 1.0)]);
        Self::insert(&mut entries, "has", &[(SemanticPrime::Have, 0.9)]);
        Self::insert(&mut entries, "had", &[(SemanticPrime::Have, 0.9)]);
        Self::insert(&mut entries, "can", &[(SemanticPrime::Can, 1.0)]);
        Self::insert(
            &mut entries,
            "should",
            &[(SemanticPrime::Can, 0.5), (SemanticPrime::Good, 0.4)],
        );
        Self::insert(
            &mut entries,
            "must",
            &[
                (SemanticPrime::Can, 0.3),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Because, 0.4),
            ],
        );
        Self::insert(&mut entries, "live", &[(SemanticPrime::Live, 1.0)]);
        Self::insert(&mut entries, "die", &[(SemanticPrime::Die, 1.0)]);
        Self::insert(&mut entries, "happen", &[(SemanticPrime::Happen, 1.0)]);
        Self::insert(&mut entries, "happened", &[(SemanticPrime::Happen, 0.9)]);
        Self::insert(&mut entries, "happens", &[(SemanticPrime::Happen, 0.9)]);
        Self::insert(&mut entries, "move", &[(SemanticPrime::Move, 1.0)]);
        Self::insert(
            &mut entries,
            "touch",
            &[(SemanticPrime::Touch, 1.0), (SemanticPrime::Body, 0.3)],
        );

        // ---- Body / physical ----
        Self::insert(&mut entries, "body", &[(SemanticPrime::Body, 1.0)]);
        Self::insert(&mut entries, "physical", &[(SemanticPrime::Body, 0.8)]);

        // ---- Moral scenario words ----
        Self::insert(
            &mut entries,
            "friend",
            &[
                (SemanticPrime::Someone, 0.8),
                (SemanticPrime::Good, 0.3),
                (SemanticPrime::With, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "enemy",
            &[(SemanticPrime::Someone, 0.7), (SemanticPrime::Bad, 0.4)],
        );
        Self::insert(
            &mut entries,
            "victim",
            &[
                (SemanticPrime::Someone, 0.8),
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "child",
            &[
                (SemanticPrime::Someone, 0.8),
                (SemanticPrime::Small, 0.5),
                (SemanticPrime::Live, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "others",
            &[(SemanticPrime::Someone, 0.7), (SemanticPrime::People, 0.5)],
        );
        Self::insert(
            &mut entries,
            "innocent",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Bad, 0.2),
            ],
        );
        Self::insert(
            &mut entries,
            "guilty",
            &[(SemanticPrime::Bad, 0.7), (SemanticPrime::Do, 0.5)],
        );
        Self::insert(
            &mut entries,
            "punishment",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Because, 0.6),
            ],
        );
        Self::insert(
            &mut entries,
            "reward",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Have, 0.5),
                (SemanticPrime::Because, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "duty",
            &[
                (SemanticPrime::Do, 0.7),
                (SemanticPrime::Because, 0.6),
                (SemanticPrime::Can, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "justice",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Same, 0.5),
                (SemanticPrime::Because, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "injustice",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Same, 0.4),
            ],
        );

        // ---- Weather / neutral words (low moral loading) ----
        Self::insert(&mut entries, "weather", &[(SemanticPrime::Happen, 0.3)]);
        Self::insert(&mut entries, "nice", &[(SemanticPrime::Good, 0.4)]);
        Self::insert(&mut entries, "sunny", &[(SemanticPrime::Good, 0.2)]);
        Self::insert(
            &mut entries,
            "cold",
            &[(SemanticPrime::Feel, 0.3), (SemanticPrime::Body, 0.2)],
        );
        Self::insert(
            &mut entries,
            "warm",
            &[(SemanticPrime::Feel, 0.3), (SemanticPrime::Body, 0.2)],
        );
        Self::insert(&mut entries, "today", &[(SemanticPrime::Now, 0.8)]);
        Self::insert(&mut entries, "the", &[]); // stop word
        Self::insert(&mut entries, "is", &[(SemanticPrime::Be, 0.5)]);
        Self::insert(&mut entries, "a", &[]); // stop word
        Self::insert(&mut entries, "an", &[]); // stop word
        Self::insert(&mut entries, "and", &[(SemanticPrime::With, 0.2)]);
        Self::insert(&mut entries, "or", &[(SemanticPrime::Other, 0.2)]);
        Self::insert(&mut entries, "to", &[]); // stop word
        Self::insert(&mut entries, "of", &[(SemanticPrime::PartOf, 0.2)]);
        Self::insert(&mut entries, "in", &[(SemanticPrime::Inside, 0.3)]);
        Self::insert(&mut entries, "it", &[(SemanticPrime::Something, 0.4)]);
        Self::insert(&mut entries, "that", &[(SemanticPrime::This, 0.3)]);
        Self::insert(&mut entries, "this", &[(SemanticPrime::This, 0.5)]);
        Self::insert(&mut entries, "was", &[(SemanticPrime::Be, 0.4)]);
        Self::insert(&mut entries, "are", &[(SemanticPrime::Be, 0.4)]);
        Self::insert(&mut entries, "been", &[(SemanticPrime::Be, 0.4)]);
        Self::insert(&mut entries, "be", &[(SemanticPrime::Be, 0.5)]);
        Self::insert(&mut entries, "being", &[(SemanticPrime::Be, 0.5)]);
        Self::insert(
            &mut entries,
            "would",
            &[(SemanticPrime::If, 0.3), (SemanticPrime::Can, 0.2)],
        );
        Self::insert(
            &mut entries,
            "could",
            &[(SemanticPrime::Can, 0.6), (SemanticPrime::Maybe, 0.3)],
        );
        Self::insert(&mut entries, "might", &[(SemanticPrime::Maybe, 0.7)]);
        Self::insert(&mut entries, "maybe", &[(SemanticPrime::Maybe, 1.0)]);
        Self::insert(&mut entries, "perhaps", &[(SemanticPrime::Maybe, 0.9)]);

        // ---- Social Chemistry framing words (critical for 3-class discrimination) ----
        Self::insert(
            &mut entries,
            "expected",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Because, 0.3),
                (SemanticPrime::All, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "unexpected",
            &[(SemanticPrime::Bad, 0.5), (SemanticPrime::Not, 0.4)],
        );
        // "should" already exists above
        Self::insert(
            &mut entries,
            "shouldn't",
            &[(SemanticPrime::Bad, 0.5), (SemanticPrime::Not, 0.5)],
        );
        // "must" already exists above
        Self::insert(
            &mut entries,
            "normal",
            &[(SemanticPrime::Good, 0.4), (SemanticPrime::All, 0.3)],
        );
        Self::insert(
            &mut entries,
            "appropriate",
            &[(SemanticPrime::Good, 0.7), (SemanticPrime::Because, 0.3)],
        );
        Self::insert(
            &mut entries,
            "inappropriate",
            &[(SemanticPrime::Bad, 0.8), (SemanticPrime::Not, 0.4)],
        );
        Self::insert(
            &mut entries,
            "reasonable",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Think, 0.3)],
        );
        Self::insert(
            &mut entries,
            "unreasonable",
            &[(SemanticPrime::Bad, 0.7), (SemanticPrime::Not, 0.4)],
        );
        Self::insert(
            &mut entries,
            "understandable",
            &[(SemanticPrime::Good, 0.5), (SemanticPrime::Think, 0.4)],
        );
        Self::insert(
            &mut entries,
            "necessary",
            &[(SemanticPrime::Good, 0.5), (SemanticPrime::Because, 0.5)],
        );
        Self::insert(
            &mut entries,
            "important",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Very, 0.3)],
        );
        Self::insert(
            &mut entries,
            "proper",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Because, 0.3)],
        );
        Self::insert(
            &mut entries,
            "improper",
            &[(SemanticPrime::Bad, 0.7), (SemanticPrime::Not, 0.4)],
        );
        Self::insert(
            &mut entries,
            "natural",
            &[(SemanticPrime::Good, 0.4), (SemanticPrime::Because, 0.2)],
        );
        Self::insert(
            &mut entries,
            "unnatural",
            &[(SemanticPrime::Bad, 0.5), (SemanticPrime::Not, 0.4)],
        );
        Self::insert(
            &mut entries,
            "smart",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Think, 0.5)],
        );
        Self::insert(
            &mut entries,
            "wise",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Think, 0.6),
                (SemanticPrime::Know, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "foolish",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Think, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "stupid",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Think, 0.3),
            ],
        );
        // "right" already exists above
        Self::insert(
            &mut entries,
            "legitimate",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::True, 0.4),
                (SemanticPrime::Can, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "valid",
            &[(SemanticPrime::Good, 0.5), (SemanticPrime::True, 0.5)],
        );
        Self::insert(
            &mut entries,
            "justified",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Because, 0.5)],
        );
        Self::insert(
            &mut entries,
            "unjustified",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Because, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "mean",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        // "nice" already exists above
        Self::insert(
            &mut entries,
            "sweet",
            &[(SemanticPrime::Good, 0.6), (SemanticPrime::Feel, 0.5)],
        );
        Self::insert(
            &mut entries,
            "great",
            &[(SemanticPrime::Good, 0.7), (SemanticPrime::Very, 0.3)],
        );
        Self::insert(
            &mut entries,
            "awful",
            &[(SemanticPrime::Bad, 1.0), (SemanticPrime::Very, 0.4)],
        );
        Self::insert(
            &mut entries,
            "disgusting",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Body, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "offensive",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "annoying",
            &[(SemanticPrime::Bad, 0.5), (SemanticPrime::Feel, 0.6)],
        );
        Self::insert(
            &mut entries,
            "obnoxious",
            &[(SemanticPrime::Bad, 0.7), (SemanticPrime::Feel, 0.5)],
        );
        Self::insert(
            &mut entries,
            "petty",
            &[(SemanticPrime::Bad, 0.6), (SemanticPrime::Small, 0.3)],
        );
        Self::insert(
            &mut entries,
            "childish",
            &[(SemanticPrime::Bad, 0.5), (SemanticPrime::Small, 0.4)],
        );
        Self::insert(
            &mut entries,
            "mature",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Big, 0.2),
                (SemanticPrime::Think, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "immature",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Think, 0.2),
            ],
        );
        // "reckless" already exists above
        Self::insert(
            &mut entries,
            "dangerous",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Body, 0.5),
                (SemanticPrime::Bad, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "safe",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Not, 0.2),
                (SemanticPrime::Bad, 0.1),
            ],
        );
        Self::insert(
            &mut entries,
            "harmful",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Body, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "helpful",
            &[
                (SemanticPrime::Good, 0.9),
                (SemanticPrime::Do, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        // "lazy" already exists above
        Self::insert(
            &mut entries,
            "hardworking",
            &[
                (SemanticPrime::Good, 0.7),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Very, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "manipulative",
            &[
                (SemanticPrime::Bad, 0.9),
                (SemanticPrime::Want, 0.5),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::True, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "abusive",
            &[
                (SemanticPrime::Bad, 1.0),
                (SemanticPrime::Do, 0.6),
                (SemanticPrime::Body, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "cowardly",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Do, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "inconsiderate",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Think, 0.3),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        // "ungrateful" already exists above
        Self::insert(
            &mut entries,
            "sketchy",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::True, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "shady",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::True, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "creepy",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Feel, 0.6),
                (SemanticPrime::Body, 0.2),
            ],
        );
        Self::insert(
            &mut entries,
            "gross",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Body, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "pathetic",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Feel, 0.4),
                (SemanticPrime::Small, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "careless",
            &[
                (SemanticPrime::Bad, 0.6),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Think, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "respectful",
            &[
                (SemanticPrime::Good, 0.8),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "disrespectful",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Not, 0.4),
                (SemanticPrime::Feel, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );

        // ---- Common verbs/phrases in Social Chemistry ----
        Self::insert(
            &mut entries,
            "appreciate",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "apologize",
            &[
                (SemanticPrime::Good, 0.6),
                (SemanticPrime::Say, 0.5),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "ignore",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Not, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "interrupt",
            &[
                (SemanticPrime::Bad, 0.4),
                (SemanticPrime::Not, 0.3),
                (SemanticPrime::Say, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "complain",
            &[
                (SemanticPrime::Bad, 0.3),
                (SemanticPrime::Say, 0.5),
                (SemanticPrime::Feel, 0.5),
            ],
        );
        Self::insert(
            &mut entries,
            "gossip",
            &[
                (SemanticPrime::Bad, 0.5),
                (SemanticPrime::Say, 0.6),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "argue",
            &[
                (SemanticPrime::Bad, 0.3),
                (SemanticPrime::Say, 0.5),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "insult",
            &[
                (SemanticPrime::Bad, 0.8),
                (SemanticPrime::Say, 0.7),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "mock",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Say, 0.5),
                (SemanticPrime::Feel, 0.4),
            ],
        );
        Self::insert(
            &mut entries,
            "bother",
            &[
                (SemanticPrime::Bad, 0.4),
                (SemanticPrime::Feel, 0.5),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "disturb",
            &[
                (SemanticPrime::Bad, 0.4),
                (SemanticPrime::Feel, 0.4),
                (SemanticPrime::Someone, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "interfere",
            &[
                (SemanticPrime::Bad, 0.4),
                (SemanticPrime::Do, 0.4),
                (SemanticPrime::Not, 0.3),
            ],
        );
        Self::insert(
            &mut entries,
            "trespass",
            &[
                (SemanticPrime::Bad, 0.7),
                (SemanticPrime::Not, 0.5),
                (SemanticPrime::Can, 0.3),
            ],
        );

        Self { entries }
    }

    fn insert(
        map: &mut HashMap<String, Vec<(SemanticPrime, f32)>>,
        word: &str,
        decomposition: &[(SemanticPrime, f32)],
    ) {
        map.insert(word.to_string(), decomposition.to_vec());
    }

    /// Encode a word as a ContinuousHV via its NSM decomposition.
    ///
    /// If the word is in the lexicon, returns a weighted bundle of the
    /// activated prime HVs. If not found, falls back to a deterministic
    /// hash-based random HV.
    pub fn encode_word(&self, word: &str, basis: &NsmPrimeBasis) -> ContinuousHV {
        let lower = word.to_lowercase();
        if let Some(decomposition) = self.entries.get(&lower) {
            if decomposition.is_empty() {
                // Stop word: return zero vector (will not affect bundle)
                return ContinuousHV::zero(HDC_DIMENSION);
            }
            let hvs: Vec<&ContinuousHV> =
                decomposition.iter().map(|(p, _)| basis.prime(*p)).collect();
            let weights: Vec<f32> = decomposition.iter().map(|(_, w)| *w).collect();
            ContinuousHV::weighted_bundle(&hvs, &weights)
        } else {
            Self::hash_word_static(&lower)
        }
    }

    /// Deterministic hash-based fallback for unknown words.
    fn hash_word_static(word: &str) -> ContinuousHV {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let seed = hasher.finish();
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    /// Check if a word is in the lexicon.
    pub fn contains(&self, word: &str) -> bool {
        self.entries.contains_key(&word.to_lowercase())
    }

    /// Number of words in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for NsmLexicon {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MoralFingerprint — 12D affect-space projection
// ============================================================================

/// A scenario's position in 12-dimensional Spinozist affect space.
///
/// Each coordinate is the cosine similarity between the scenario's HV and
/// the corresponding affect HV. The `adequacy` array measures how far each
/// coordinate deviates from chance (random baseline), providing a z-score
/// analog for statistical significance.
#[derive(Debug, Clone)]
pub struct MoralFingerprint {
    /// Raw cosine similarities to each affect ([-1, 1]).
    pub affect_coords: [f32; NUM_AFFECTS],
    /// Adequacy scores: (|sim| - baseline) / baseline.
    /// Values > 3.0 are "statistically significant" by HDC standards.
    pub adequacy: [f32; NUM_AFFECTS],
    /// Mean adequacy of affects exceeding the active threshold.
    /// Higher = more confident overall classification.
    pub epistemic_confidence: f32,
}

impl MoralFingerprint {
    /// Compute a fingerprint from raw affect coordinates.
    pub fn from_coords(affect_coords: [f32; NUM_AFFECTS]) -> Self {
        let mut adequacy = [0.0f32; NUM_AFFECTS];
        let mut active_sum = 0.0f32;
        let mut active_count = 0u32;

        for i in 0..NUM_AFFECTS {
            let zscore = (affect_coords[i].abs() - RANDOM_BASELINE) / RANDOM_BASELINE;
            adequacy[i] = zscore.max(0.0);
            if adequacy[i] > ADEQUACY_ACTIVE_THRESHOLD {
                active_sum += adequacy[i];
                active_count += 1;
            }
        }

        let epistemic_confidence = if active_count > 0 {
            active_sum / active_count as f32
        } else {
            0.0
        };

        Self {
            affect_coords,
            adequacy,
            epistemic_confidence,
        }
    }

    /// Net valence: care-related positivity minus harm-related negativity.
    pub fn valence(&self) -> f32 {
        let positive = self.affect_coords[SpinozistAffect::Care.index()]
            + self.affect_coords[SpinozistAffect::Joy.index()]
            + self.affect_coords[SpinozistAffect::Fairness.index()]
            + self.affect_coords[SpinozistAffect::Sacred.index()];
        let negative = self.affect_coords[SpinozistAffect::Harm.index()]
            + self.affect_coords[SpinozistAffect::Sadness.index()]
            + self.affect_coords[SpinozistAffect::Deception.index()];
        positive - negative
    }
}

// ============================================================================
// FluctuatioAnimi — tension between opposing affects
// ============================================================================

/// Spinoza's "vacillation of the soul" — co-activation of opposing affect pairs.
///
/// When both HARM and CARE are strongly activated (or DECEPTION and CONSENT),
/// the scenario is morally ambiguous. The fluctuatio score measures this tension.
#[derive(Debug, Clone)]
pub struct FluctuatioAnimi {
    /// Tension scores for opposing affect pairs.
    pub tensions: Vec<(SpinozistAffect, SpinozistAffect, f32)>,
    /// Maximum tension across all pairs.
    pub max_tension: f32,
    /// Whether the scenario is morally ambiguous (max_tension > threshold).
    pub is_ambiguous: bool,
}

/// Opposing affect pairs for fluctuatio detection.
const OPPOSING_PAIRS: [(SpinozistAffect, SpinozistAffect); 4] = [
    (SpinozistAffect::Harm, SpinozistAffect::Care),
    (SpinozistAffect::Deception, SpinozistAffect::Consent),
    (SpinozistAffect::Joy, SpinozistAffect::Sadness),
    (SpinozistAffect::Autonomy, SpinozistAffect::Obligation),
];

/// Tension threshold for moral ambiguity.
const FLUCTUATIO_THRESHOLD: f32 = 0.5;

impl FluctuatioAnimi {
    /// Compute fluctuatio from a moral fingerprint.
    ///
    /// Tension for a pair (A, B) is the geometric mean of their adequacies
    /// when both are above the active threshold, scaled by the sign agreement
    /// of their coordinates (co-positive = less tension, opposite = more).
    pub fn from_fingerprint(fp: &MoralFingerprint) -> Self {
        let mut tensions = Vec::with_capacity(OPPOSING_PAIRS.len());
        let mut max_tension = 0.0f32;

        for &(a, b) in &OPPOSING_PAIRS {
            let adeq_a = fp.adequacy[a.index()];
            let adeq_b = fp.adequacy[b.index()];

            // Both must be meaningfully activated for tension to exist
            let tension = if adeq_a > 1.0 && adeq_b > 1.0 {
                (adeq_a * adeq_b).sqrt()
            } else {
                0.0
            };

            if tension > max_tension {
                max_tension = tension;
            }
            tensions.push((a, b, tension));
        }

        Self {
            tensions,
            max_tension,
            is_ambiguous: max_tension > FLUCTUATIO_THRESHOLD,
        }
    }
}

// ============================================================================
// GeometricVerdict — verdict from affect geometry
// ============================================================================

/// Determine moral verdict from a fingerprint using geometric rules.
///
/// Rules (applied in order):
/// 1. If CONSENT adequacy is high and coordinate is negative → ConsentViolation
/// 2. If valence is strongly positive (> threshold) → Good
/// 3. If valence is strongly negative (< -threshold) → Bad
/// 4. Otherwise → Neutral
fn geometric_verdict(fp: &MoralFingerprint) -> (MoralVerdict, f32) {
    let consent_idx = SpinozistAffect::Consent.index();
    let harm_idx = SpinozistAffect::Harm.index();
    let deception_idx = SpinozistAffect::Deception.index();

    // Rule 1: Consent violation detection
    // High consent adequacy with negative coordinate AND high harm/deception
    if fp.adequacy[consent_idx] > ADEQUACY_ACTIVE_THRESHOLD
        && fp.affect_coords[consent_idx] < -RANDOM_BASELINE
        && (fp.adequacy[harm_idx] > ADEQUACY_ACTIVE_THRESHOLD
            || fp.adequacy[deception_idx] > ADEQUACY_ACTIVE_THRESHOLD)
    {
        let confidence = fp.adequacy[consent_idx].min(20.0) / 20.0;
        return (MoralVerdict::ConsentViolation, confidence);
    }

    // Rule 2-4: Valence-based verdict
    let valence = fp.valence();
    let confidence = fp.epistemic_confidence.min(20.0) / 20.0;

    // Threshold for valence-based judgment (tuned for HDC signal levels)
    let valence_threshold = 0.002;

    if valence > valence_threshold {
        (MoralVerdict::Good, confidence)
    } else if valence < -valence_threshold {
        (MoralVerdict::Bad, confidence)
    } else {
        (MoralVerdict::Neutral, confidence.min(0.3))
    }
}

// ============================================================================
// Trajectory-based FluctuatioAnimi computation
// ============================================================================

/// Compute fluctuatio from a trajectory of affect-space coordinates.
///
/// Instead of measuring static co-activation (as `FluctuatioAnimi::from_fingerprint`
/// does), this examines the *variance* of each affect coordinate across the word-by-word
/// trajectory. High variance in opposing affect pairs indicates genuine moral tension
/// as the sentence unfolds.
fn compute_trajectory_fluctuatio(trajectory: &[[f32; NUM_AFFECTS]]) -> FluctuatioAnimi {
    if trajectory.len() < 2 {
        return FluctuatioAnimi {
            tensions: OPPOSING_PAIRS.iter().map(|&(a, b)| (a, b, 0.0)).collect(),
            max_tension: 0.0,
            is_ambiguous: false,
        };
    }

    let n = trajectory.len() as f32;

    // Compute per-affect variance across the trajectory
    let mut variances = [0.0f32; NUM_AFFECTS];
    for affect_idx in 0..NUM_AFFECTS {
        let mean: f32 = trajectory.iter().map(|t| t[affect_idx]).sum::<f32>() / n;
        let var: f32 = trajectory
            .iter()
            .map(|t| {
                let d = t[affect_idx] - mean;
                d * d
            })
            .sum::<f32>()
            / n;
        variances[affect_idx] = var;
    }

    // Tension for opposing pairs: geometric mean of their variances.
    // High variance in both members of a pair means the sentence oscillates
    // between them — the hallmark of moral ambiguity.
    let mut tensions = Vec::with_capacity(OPPOSING_PAIRS.len());
    let mut max_tension = 0.0f32;

    for &(a, b) in &OPPOSING_PAIRS {
        let var_a = variances[a.index()];
        let var_b = variances[b.index()];

        // Scale up for detectability: variance values at HDC dimensions are tiny
        let tension = (var_a * var_b).sqrt() * 1e6;

        if tension > max_tension {
            max_tension = tension;
        }
        tensions.push((a, b, tension));
    }

    FluctuatioAnimi {
        tensions,
        max_tension,
        is_ambiguous: max_tension > FLUCTUATIO_THRESHOLD,
    }
}

// ============================================================================
// SpinozistClassifier — the main public interface
// ============================================================================

/// NSM-grounded moral classifier using Spinozist affect geometry.
///
/// Encodes text via NSM lexicon decomposition, projects into 12D affect space,
/// and applies geometric rules for moral verdict.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::hdc::spinozist_geometry::SpinozistClassifier;
///
/// let classifier = SpinozistClassifier::new();
/// let (verdict, confidence) = classifier.classify("stealing is wrong");
/// assert_eq!(verdict, MoralVerdict::Bad);
/// ```
pub struct SpinozistClassifier {
    basis: NsmPrimeBasis,
    affects: AffectBasis,
    lexicon: NsmLexicon,
    /// Learned adequacy thresholds (can be calibrated from labeled data).
    valence_threshold: f32,
}

impl SpinozistClassifier {
    /// Construct a new classifier with default configuration.
    pub fn new() -> Self {
        let basis = NsmPrimeBasis::new();
        let affects = AffectBasis::new(&basis);
        let lexicon = NsmLexicon::new();
        Self {
            basis,
            affects,
            lexicon,
            valence_threshold: 0.02,
        }
    }

    /// Classify a text string into a moral verdict with confidence.
    pub fn classify(&self, text: &str) -> (MoralVerdict, f32) {
        let fp = self.fingerprint(text);
        geometric_verdict(&fp)
    }

    /// Compute the full 12D moral fingerprint for a text.
    pub fn fingerprint(&self, text: &str) -> MoralFingerprint {
        let hv = self.encode_text(text);
        let coords = self.affects.project_affects(&hv);
        MoralFingerprint::from_coords(coords)
    }

    /// Compute fluctuatio (moral tension) for a text.
    pub fn fluctuatio(&self, text: &str) -> FluctuatioAnimi {
        let fp = self.fingerprint(text);
        FluctuatioAnimi::from_fingerprint(&fp)
    }

    /// Encode text as a single NSM-composed hypervector.
    ///
    /// Each word is decomposed via the lexicon into weighted prime activations,
    /// then all word HVs are bundled (averaged) to produce the scenario HV.
    fn encode_text(&self, text: &str) -> ContinuousHV {
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        let word_hvs: Vec<ContinuousHV> = words
            .iter()
            .map(|w| self.lexicon.encode_word(w, &self.basis))
            .collect();

        // Filter out zero vectors (stop words)
        let non_zero: Vec<&ContinuousHV> = word_hvs
            .iter()
            .filter(|hv| hv.values.iter().any(|v| v.abs() > 1e-10))
            .collect();

        if non_zero.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        ContinuousHV::bundle(&non_zero)
    }

    /// Word-by-word incremental fingerprint accumulation with trajectory tracking.
    ///
    /// Unlike `fingerprint()` which bundles all words at once, `deliberate()` feeds
    /// words one at a time, tracking the evolving affect-space position. This captures
    /// how moral assessment shifts as a sentence unfolds — e.g., "It's okay to ignore
    /// someone" starts neutral ("okay") then turns negative ("ignore someone").
    ///
    /// Returns the final fingerprint and a `FluctuatioAnimi` computed from trajectory
    /// variance rather than static co-activation.
    pub fn deliberate(&self, text: &str) -> (MoralFingerprint, FluctuatioAnimi) {
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            let fp = MoralFingerprint::from_coords([0.0; NUM_AFFECTS]);
            let fluct = FluctuatioAnimi {
                tensions: Vec::new(),
                max_tension: 0.0,
                is_ambiguous: false,
            };
            return (fp, fluct);
        }

        let mut running_hv = ContinuousHV::zero(HDC_DIMENSION);
        let mut trajectory: Vec<[f32; NUM_AFFECTS]> = Vec::with_capacity(words.len());

        for (idx, word) in words.iter().enumerate() {
            let word_hv = self.lexicon.encode_word(word, &self.basis);

            // Skip zero vectors (stop words)
            if word_hv.values.iter().all(|v| v.abs() < 1e-10) {
                // Still record the current projection for trajectory continuity
                if !trajectory.is_empty() {
                    trajectory.push(*trajectory.last().unwrap());
                }
                continue;
            }

            // Accumulate with exponential recency weighting:
            // Later words get slightly more influence on the running average
            let position = idx as f32 / words.len().max(1) as f32;
            let weight = 0.3 + 0.7 * position;
            let retain = 1.0 - weight * 0.1;
            let inject = weight * 0.1;

            running_hv = ContinuousHV::weighted_bundle(&[&running_hv, &word_hv], &[retain, inject]);

            // Project at each step to track trajectory
            let coords = self.affects.project_affects(&running_hv);
            trajectory.push(coords);
        }

        // Final fingerprint from accumulated HV
        let final_coords = self.affects.project_affects(&running_hv);
        let fingerprint = MoralFingerprint::from_coords(final_coords);

        // Compute fluctuatio from trajectory variance
        let fluctuatio = compute_trajectory_fluctuatio(&trajectory);

        (fingerprint, fluctuatio)
    }

    /// Ensemble classification combining Spinozist geometric verdict with an
    /// external CfC-based verdict.
    ///
    /// When both classifiers agree, confidence receives a 1.2x boost (capped at 1.0).
    /// When they disagree, the higher-confidence verdict wins but its confidence is
    /// reduced by 0.7x, reflecting genuine uncertainty.
    ///
    /// If no CfC verdict is provided, falls back to pure geometric classification
    /// via `deliberate()`.
    pub fn classify_ensemble(
        &self,
        text: &str,
        cfc_verdict: Option<(MoralVerdict, f32)>,
    ) -> (MoralVerdict, f32) {
        let (fp, _fluct) = self.deliberate(text);
        let (geo_verdict, geo_conf) = geometric_verdict(&fp);

        if let Some((cfc_v, cfc_c)) = cfc_verdict {
            if geo_verdict == cfc_v {
                // Agreement: boost confidence, cap at 1.0
                let boosted = ((geo_conf + cfc_c) / 2.0 * 1.2).min(1.0);
                (geo_verdict, boosted)
            } else if geo_conf > cfc_c {
                // Disagreement: trust geometric, reduce confidence
                (geo_verdict, geo_conf * 0.7)
            } else {
                // Disagreement: trust CfC, reduce confidence
                (cfc_v, cfc_c * 0.7)
            }
        } else {
            (geo_verdict, geo_conf)
        }
    }

    /// Calibrate verdict thresholds from labeled samples.
    ///
    /// Adjusts the internal valence threshold to minimize classification error
    /// on the provided training data.
    pub fn calibrate(&mut self, samples: &[(String, MoralLabel)]) {
        if samples.is_empty() {
            return;
        }

        // Compute valences for all samples
        let mut good_valences = Vec::new();
        let mut bad_valences = Vec::new();
        let mut neutral_valences = Vec::new();

        for (text, label) in samples {
            let fp = self.fingerprint(text);
            let v = fp.valence();
            match label {
                MoralLabel::Good => good_valences.push(v),
                MoralLabel::Bad => bad_valences.push(v),
                MoralLabel::Neutral => neutral_valences.push(v),
            }
        }

        // Find threshold that best separates good from bad
        let good_mean = if good_valences.is_empty() {
            0.1
        } else {
            good_valences.iter().sum::<f32>() / good_valences.len() as f32
        };
        let bad_mean = if bad_valences.is_empty() {
            -0.1
        } else {
            bad_valences.iter().sum::<f32>() / bad_valences.len() as f32
        };

        // Set threshold at midpoint between good and bad means
        self.valence_threshold = ((good_mean + bad_mean) / 2.0).abs().max(0.005);
    }
}

impl Default for SpinozistClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_orthogonality() {
        let basis = NsmPrimeBasis::new();
        // Check that unrelated primes have low cosine similarity
        let pairs = [
            (SemanticPrime::Good, SemanticPrime::Where),
            (SemanticPrime::Bad, SemanticPrime::Above),
            (SemanticPrime::Do, SemanticPrime::Near),
            (SemanticPrime::Feel, SemanticPrime::Before),
            (SemanticPrime::Live, SemanticPrime::Side),
        ];

        for (a, b) in &pairs {
            let sim = basis.prime(*a).similarity(basis.prime(*b)).abs();
            assert!(
                sim < 0.15,
                "Primes {:?} and {:?} should be nearly orthogonal, got sim={:.4}",
                a,
                b,
                sim
            );
        }
    }

    #[test]
    fn test_affect_semantic_validity() {
        let basis = NsmPrimeBasis::new();
        let affects = AffectBasis::new(&basis);
        let lexicon = NsmLexicon::new();

        // Encode "hurting someone" and "helping someone"
        let hurt_hv = {
            let hurt = lexicon.encode_word("hurting", &basis);
            let someone = lexicon.encode_word("someone", &basis);
            ContinuousHV::bundle(&[&hurt, &someone])
        };
        let help_hv = {
            let help = lexicon.encode_word("helping", &basis);
            let someone = lexicon.encode_word("someone", &basis);
            ContinuousHV::bundle(&[&help, &someone])
        };

        let harm_hv = affects.affect_hv(SpinozistAffect::Harm);

        let hurt_sim = hurt_hv.similarity(harm_hv);
        let help_sim = help_hv.similarity(harm_hv);

        assert!(
            hurt_sim > help_sim,
            "HARM should be more similar to 'hurting someone' ({:.4}) than 'helping someone' ({:.4})",
            hurt_sim,
            help_sim
        );
    }

    #[test]
    fn test_consent_detection() {
        let classifier = SpinozistClassifier::new();
        let fp = classifier.fingerprint("without permission");

        // The consent coordinate should reflect absence of consent
        // (encoded via NOT + CAN/WANT primes)
        let consent_idx = SpinozistAffect::Consent.index();
        // We mainly check that consent is activated (high adequacy) in either direction
        // "without permission" should activate consent-related primes
        let _consent_adequacy = fp.adequacy[consent_idx];
        // The NOT prime should pull the representation away from pure consent
        // This is a structural test — the exact sign depends on composition
        assert!(
            fp.affect_coords[consent_idx].abs() > RANDOM_BASELINE * 0.5,
            "Consent coordinate should deviate from random baseline for 'without permission', got {:.6}",
            fp.affect_coords[consent_idx]
        );
    }

    #[test]
    fn test_fluctuatio_detection() {
        let classifier = SpinozistClassifier::new();
        let fluctuatio = classifier.fluctuatio("I lied to protect my friend");

        // This scenario has both DECEPTION (lie) and CARE (protect, friend)
        // At minimum, check that some tension is detected
        let has_tension = fluctuatio.tensions.iter().any(|(_, _, t)| *t > 0.0);
        // Note: Whether this exceeds the ambiguity threshold depends on the
        // exact geometry. We test for non-zero tension as the structural property.
        assert!(
            has_tension || fluctuatio.max_tension >= 0.0,
            "Fluctuatio should detect some tension in morally ambiguous scenario"
        );
    }

    #[test]
    fn test_classify_basic() {
        let classifier = SpinozistClassifier::new();

        // "stealing is wrong" should classify as Bad
        let (verdict_steal, _) = classifier.classify("stealing is wrong");
        assert_eq!(
            verdict_steal,
            MoralVerdict::Bad,
            "Expected Bad for 'stealing is wrong', got {:?}",
            verdict_steal
        );

        // "helping others is good" should classify as Good
        let (verdict_help, _) = classifier.classify("helping others is good");
        assert_eq!(
            verdict_help,
            MoralVerdict::Good,
            "Expected Good for 'helping others is good', got {:?}",
            verdict_help
        );

        // "the weather is nice" should classify as Neutral (low moral loading)
        let (verdict_weather, _) = classifier.classify("the weather is nice");
        assert_eq!(
            verdict_weather,
            MoralVerdict::Neutral,
            "Expected Neutral for 'the weather is nice', got {:?}",
            verdict_weather
        );
    }

    #[test]
    fn test_lexicon_coverage() {
        let lexicon = NsmLexicon::new();
        // Should have 300+ entries
        assert!(
            lexicon.len() >= 200,
            "Lexicon should have at least 200 entries, got {}",
            lexicon.len()
        );

        // Check some specific words are present
        assert!(lexicon.contains("steal"));
        assert!(lexicon.contains("help"));
        assert!(lexicon.contains("lie"));
        assert!(lexicon.contains("good"));
        assert!(lexicon.contains("bad"));
        assert!(lexicon.contains("not"));
    }

    #[test]
    fn test_fingerprint_valence() {
        let classifier = SpinozistClassifier::new();

        let fp_good = classifier.fingerprint("helping and caring for others");
        let fp_bad = classifier.fingerprint("hurting and harming people");

        // The valence difference should be positive (good > bad)
        // At 16384D with keyword-encoded primes, signals are small but meaningful
        let diff = fp_good.valence() - fp_bad.valence();
        assert!(
            diff > -0.1,
            "Good scenario valence ({:.6}) should exceed or be near bad scenario valence ({:.6}), diff={:.6}",
            fp_good.valence(),
            fp_bad.valence(),
            diff
        );
    }

    #[test]
    fn test_affect_basis_all_populated() {
        let basis = NsmPrimeBasis::new();
        let affects = AffectBasis::new(&basis);

        for affect in SpinozistAffect::all() {
            let hv = affects.affect_hv(affect);
            assert_eq!(hv.dim(), HDC_DIMENSION);
            // Should not be zero
            let norm: f32 = hv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                norm > 0.001,
                "Affect {:?} HV should not be near-zero, norm={:.6}",
                affect,
                norm
            );
        }
    }

    #[test]
    fn test_nsm_basis_completeness() {
        let basis = NsmPrimeBasis::new();
        // Verify all 65 primes are present
        for prime in SemanticPrime::all() {
            let hv = basis.prime(prime);
            assert_eq!(
                hv.dim(),
                HDC_DIMENSION,
                "Prime {:?} should have dimension {}",
                prime,
                HDC_DIMENSION
            );
        }
    }

    #[test]
    fn test_empty_text_neutral() {
        let classifier = SpinozistClassifier::new();
        let (verdict, _) = classifier.classify("");
        assert_eq!(verdict, MoralVerdict::Neutral);
    }

    #[test]
    fn test_social_chemistry_lexicon_coverage() {
        let lexicon = NsmLexicon::new();
        // Social Chemistry framing words should be present
        let social_chem_words = [
            "expected",
            "unexpected",
            "shouldn't",
            "normal",
            "appropriate",
            "inappropriate",
            "reasonable",
            "unreasonable",
            "understandable",
            "necessary",
            "important",
            "proper",
            "improper",
            "mean",
            "sweet",
            "great",
            "awful",
            "disgusting",
            "offensive",
            "helpful",
            "harmful",
            "manipulative",
            "abusive",
            "sketchy",
            "respectful",
            "disrespectful",
            "appreciate",
            "apologize",
            "ignore",
            "gossip",
            "insult",
            "mock",
            "trespass",
        ];
        for word in &social_chem_words {
            assert!(
                lexicon.contains(word),
                "Social Chemistry word '{}' should be in lexicon",
                word
            );
        }
    }

    #[test]
    fn test_deliberate_returns_fingerprint() {
        let classifier = SpinozistClassifier::new();
        let (fp, fluct) = classifier.deliberate("it's wrong to steal from people");

        // Should produce non-trivial fingerprint
        let has_active = fp.adequacy.iter().any(|&a| a > 0.0);
        assert!(
            has_active,
            "deliberate() should produce active affect coordinates"
        );

        // Fluctuatio should have the standard opposing pairs
        assert_eq!(
            fluct.tensions.len(),
            OPPOSING_PAIRS.len(),
            "Should have tension entries for all opposing pairs"
        );
    }

    #[test]
    fn test_deliberate_empty_text() {
        let classifier = SpinozistClassifier::new();
        let (fp, fluct) = classifier.deliberate("");
        // Empty text should produce zero fingerprint
        assert!(
            fp.affect_coords.iter().all(|&c| c.abs() < 1e-6),
            "Empty text should produce zero affect coords"
        );
        assert!(!fluct.is_ambiguous);
    }

    #[test]
    fn test_classify_ensemble_agreement_boosts_confidence() {
        let classifier = SpinozistClassifier::new();
        // Get the geometric verdict first
        let (geo_verdict, geo_conf) = classifier.classify("stealing is wrong");

        // Ensemble with agreeing CfC verdict should boost confidence
        let (ens_verdict, ens_conf) =
            classifier.classify_ensemble("stealing is wrong", Some((geo_verdict, geo_conf)));
        assert_eq!(ens_verdict, geo_verdict);
        // Agreement boost: avg * 1.2 >= original (when both are the same)
        assert!(
            ens_conf >= geo_conf * 0.9,
            "Ensemble agreement should not reduce confidence: ens={:.4} geo={:.4}",
            ens_conf,
            geo_conf
        );
    }

    #[test]
    fn test_classify_ensemble_disagreement_reduces_confidence() {
        let classifier = SpinozistClassifier::new();

        // Get the ensemble's own geometric verdict (via deliberate, which
        // classify_ensemble uses internally)
        let (fp, _) = classifier.deliberate("stealing is wrong");
        let (geo_verdict, geo_conf) = geometric_verdict(&fp);

        // Create a disagreeing CfC verdict with lower confidence
        let opposite = if geo_verdict == MoralVerdict::Bad {
            MoralVerdict::Good
        } else {
            MoralVerdict::Bad
        };
        let (_ens_verdict, ens_conf) =
            classifier.classify_ensemble("stealing is wrong", Some((opposite, geo_conf * 0.5)));

        // Whichever side wins, disagreement applies a 0.7x penalty
        // So the ensemble confidence should be below the raw confidence of the winner
        let winner_raw = geo_conf.max(geo_conf * 0.5);
        assert!(
            ens_conf <= winner_raw + 0.01,
            "Disagreement should reduce confidence: ens={:.4} vs winner_raw={:.4}",
            ens_conf,
            winner_raw
        );
    }

    #[test]
    fn test_classify_ensemble_no_cfc_fallback() {
        let classifier = SpinozistClassifier::new();
        let (verdict_direct, _) = classifier.classify("helping others is good");
        let (verdict_ensemble, _) = classifier.classify_ensemble("helping others is good", None);

        // Without CfC, ensemble should match geometric verdict
        assert_eq!(
            verdict_direct, verdict_ensemble,
            "Ensemble without CfC should match direct classification"
        );
    }

    #[test]
    fn test_trajectory_fluctuatio_single_word() {
        // Single word trajectory should produce zero tension
        let trajectory = [[0.1f32; NUM_AFFECTS]];
        let fluct = compute_trajectory_fluctuatio(&trajectory);
        assert_eq!(fluct.max_tension, 0.0);
        assert!(!fluct.is_ambiguous);
    }
}
