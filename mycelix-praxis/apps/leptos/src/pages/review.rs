// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Spaced Repetition Review page.
//!
//! Implements a flashcard review flow using the SM-2 quality scale (0-5).
//! Integrated with the adaptivity engine: card content is adapted based on
//! cognitive state (text complexity, modality indicators), suggestions slide
//! in when the student struggles, and metacognitive prompts appear between
//! cards.

use leptos::prelude::*;

use crate::adaptivity_provider::use_adaptivity;
use crate::cognitive_adaptivity::*;
use crate::components::suggestion_overlay::SuggestionOverlay;
use crate::curriculum::{use_progress, use_set_progress};

use crate::curriculum::display_subject;

// ---------------------------------------------------------------------------
// Review state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum CardSource {
    CoreMath,
    CorePhysics,
    CoreChemistry,
    Foundations,
    DynamicSubject(String), // Generated from curriculum graph for any subject
}

#[derive(Clone, Debug, PartialEq)]
enum ReviewState {
    Loading,
    NoDueCards,
    ShowingFront { card_index: usize },
    Predicting { card_index: usize },
    ShowingBack { card_index: usize },
    SessionComplete { reviewed: usize, correct: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
enum Confidence {
    KnowIt,
    Maybe,
    NoIdea,
}

impl Confidence {
    fn is_calibrated(self, correct: bool) -> bool {
        match (self, correct) {
            (Confidence::KnowIt, true) | (Confidence::NoIdea, false) | (Confidence::Maybe, _) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Mock flashcard data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MockFlashcard {
    front: &'static str,
    back: &'static str,
    tags: &'static str,
    mastery_permille: u16,
}

/// Dynamic flashcard generated from curriculum graph nodes
#[derive(Clone, Debug)]
struct DynamicFlashcard {
    front: String,
    back: String,
    tags: String,
    mastery_permille: u16,
}

/// Generate flashcards from curriculum nodes for a given subject
fn generate_dynamic_cards(subject: &str) -> Vec<MockFlashcard> {
    // We can't return MockFlashcard (static refs) from dynamic data,
    // so we'll use a leaked string approach for compatibility
    let graph = crate::curriculum::curriculum_graph();
    let progress = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress")
        .unwrap_or_default();

    let mut cards: Vec<MockFlashcard> = Vec::new();

    for node in &graph.nodes {
        if node.subject_area != subject { continue; }

        let bkt = progress.bkt(&node.id);
        let mastery = (bkt.p_mastery * 1000.0) as u16;

        // Card 1: What is [topic]?
        let front = format!("What is {}?", node.title);
        let back = if node.description.len() > 150 {
            format!("{}...", &node.description[..147])
        } else {
            node.description.clone()
        };
        let tags = format!("{} · {}", node.subject_area,
            node.grade_levels.first().cloned().unwrap_or_default());

        // Leak strings so they have 'static lifetime (acceptable for single-session app)
        let front_static: &'static str = Box::leak(front.into_boxed_str());
        let back_static: &'static str = Box::leak(back.into_boxed_str());
        let tags_static: &'static str = Box::leak(tags.into_boxed_str());

        cards.push(MockFlashcard {
            front: front_static,
            back: back_static,
            tags: tags_static,
            mastery_permille: mastery,
        });
    }

    // Shuffle by mastery (weakest first — prioritize what needs review)
    cards.sort_by_key(|c| c.mastery_permille);

    // Cap at 20 cards per session
    cards.truncate(20);
    cards
}

/// Generate flashcards for Core Mathematics.
fn generate_curriculum_cards() -> Vec<MockFlashcard> {
    vec![
        // Algebra
        MockFlashcard { front: "State the quadratic formula.", back: "x = (\u{2212}b \u{00b1} \u{221a}(b\u{00b2}\u{2212}4ac)) / 2a", tags: "Algebra", mastery_permille: 400 },
        MockFlashcard { front: "What does the discriminant \u{0394} = b\u{00b2}\u{2212}4ac tell us?", back: "\u{0394} > 0: two distinct real roots. \u{0394} = 0: two equal roots. \u{0394} < 0: no real roots.", tags: "Algebra", mastery_permille: 350 },
        // Sequences & Series
        MockFlashcard { front: "When does a geometric series converge?", back: "|r| < 1. Sum to infinity: a/(1\u{2212}r)", tags: "Sequences", mastery_permille: 300 },
        MockFlashcard { front: "What is Tn for an arithmetic sequence?", back: "Tn = a + (n\u{2212}1)d, where a = first term, d = common difference", tags: "Sequences", mastery_permille: 350 },
        // Finance
        MockFlashcard { front: "What is the present value annuity formula?", back: "P = x[1 \u{2212} (1+i)\u{207b}\u{207f}]/i", tags: "Finance", mastery_permille: 250 },
        MockFlashcard { front: "Sinking fund or loan? Regular payments to accumulate a target amount.", back: "Sinking fund (future value annuity). Loans use present value.", tags: "Finance", mastery_permille: 300 },
        // Functions
        MockFlashcard { front: "What is the turning point of y = a(x\u{2212}p)\u{00b2} + q?", back: "(p, q). If a > 0 it's a minimum; if a < 0 it's a maximum.", tags: "Functions", mastery_permille: 400 },
        // Calculus
        MockFlashcard { front: "What is the derivative of f(x) = ax\u{207f}?", back: "f'(x) = nax\u{207f}\u{207b}\u{00b9} (power rule)", tags: "Calculus", mastery_permille: 350 },
        MockFlashcard { front: "How do you find turning points of a cubic?", back: "Set f'(x) = 0. Solve for x. Substitute back into f(x) for y-values.", tags: "Calculus", mastery_permille: 250 },
        MockFlashcard { front: "What is the first principles definition of the derivative?", back: "f'(x) = lim[h\u{2192}0] (f(x+h) \u{2212} f(x))/h", tags: "Calculus", mastery_permille: 200 },
        // Counting & Probability
        MockFlashcard { front: "What is the formula for combinations (nCr)?", back: "nCr = n! / [r!(n\u{2212}r)!]", tags: "Counting", mastery_permille: 300 },
        MockFlashcard { front: "Permutations vs Combinations: when does order matter?", back: "Permutations: order matters (arrangements). Combinations: order doesn't (selections).", tags: "Counting", mastery_permille: 350 },
        // Trigonometry
        MockFlashcard { front: "Write cos2\u{03b1} in three forms.", back: "cos\u{00b2}\u{03b1} \u{2212} sin\u{00b2}\u{03b1} = 1 \u{2212} 2sin\u{00b2}\u{03b1} = 2cos\u{00b2}\u{03b1} \u{2212} 1", tags: "Trigonometry", mastery_permille: 250 },
        MockFlashcard { front: "sin(A+B) = ?", back: "sinA\u{00b7}cosB + cosA\u{00b7}sinB. NOT sinA + sinB!", tags: "Trigonometry", mastery_permille: 300 },
        // Geometry
        MockFlashcard { front: "Opposite angles of a cyclic quadrilateral sum to?", back: "180\u{00b0} (supplementary)", tags: "Geometry", mastery_permille: 350 },
        // Analytical Geometry
        MockFlashcard { front: "Equation of a circle with centre (a,b) and radius r?", back: "(x\u{2212}a)\u{00b2} + (y\u{2212}b)\u{00b2} = r\u{00b2}", tags: "Analytical Geometry", mastery_permille: 400 },
        // Statistics
        MockFlashcard { front: "What is the formula for standard deviation?", back: "\u{03c3} = \u{221a}[\u{03a3}(xi \u{2212} x\u{0304})\u{00b2}/n]", tags: "Statistics", mastery_permille: 300 },
    ]
}

/// Generate flashcards for Core Physics (Paper 1).
fn generate_science_cards() -> Vec<MockFlashcard> {
    vec![
        // Momentum
        MockFlashcard { front: "State the impulse-momentum theorem.", back: "Fnet\u{0394}t = \u{0394}p = m(vf \u{2212} vi). Net impulse equals change in momentum.", tags: "Momentum", mastery_permille: 350 },
        MockFlashcard { front: "Is momentum conserved in all collisions?", back: "Only in isolated systems (no external net force). Kinetic energy is only conserved in elastic collisions.", tags: "Momentum", mastery_permille: 300 },
        // Work, Energy, Power
        MockFlashcard { front: "State the work-energy theorem.", back: "Wnet = \u{0394}Ek. The net work done equals the change in kinetic energy.", tags: "Energy", mastery_permille: 350 },
        MockFlashcard { front: "What is the formula for power?", back: "P = W/\u{0394}t = Fv (work per unit time, or force times velocity)", tags: "Energy", mastery_permille: 400 },
        // Doppler Effect
        MockFlashcard { front: "Write the Doppler effect formula for sound.", back: "fL = fs(v \u{00b1} vL)/(v \u{00b1} vs). Approaching = higher frequency.", tags: "Doppler", mastery_permille: 250 },
        // Electricity
        MockFlashcard { front: "What is the relationship between EMF, terminal voltage, and internal resistance?", back: "\u{03b5} = Vterminal + Ir, or \u{03b5} = I(R + r)", tags: "Electricity", mastery_permille: 350 },
        MockFlashcard { front: "State Faraday's law of electromagnetic induction.", back: "\u{03b5} = \u{2212}N\u{0394}\u{03a6}/\u{0394}t. An EMF is induced when magnetic flux through a coil changes.", tags: "Electrodynamics", mastery_permille: 250 },
        // Photoelectric Effect
        MockFlashcard { front: "State Einstein's photoelectric equation.", back: "E = W0 + Ek(max), or hf = hf0 + \u{00bd}mv\u{00b2}(max)", tags: "Modern Physics", mastery_permille: 250 },
        MockFlashcard { front: "Why can't wave theory explain the photoelectric effect?", back: "Wave theory predicts any frequency should cause emission given enough intensity, and there should be a delay. Both are wrong.", tags: "Modern Physics", mastery_permille: 200 },
    ]
}

/// Generate flashcards for Core Chemistry (Paper 2).
fn generate_chemistry_cards() -> Vec<MockFlashcard> {
    vec![
        // Organic Chemistry
        MockFlashcard { front: "What is the product of an esterification reaction?", back: "An ester + water. Alcohol + carboxylic acid \u{2192} ester + H2O.", tags: "Organic Chemistry", mastery_permille: 300 },
        MockFlashcard { front: "What are the three types of organic reactions for alkenes?", back: "Addition reactions: HX, X2, H2O, H2 add across the C=C double bond.", tags: "Organic Chemistry", mastery_permille: 250 },
        // Reaction Rates
        MockFlashcard { front: "How does a catalyst increase reaction rate?", back: "Provides an alternative pathway with lower activation energy. NOT consumed in the reaction.", tags: "Reaction Rates", mastery_permille: 350 },
        // Equilibrium
        MockFlashcard { front: "Does a catalyst shift equilibrium?", back: "No. It speeds up both forward and reverse equally. Kc is unchanged.", tags: "Equilibrium", mastery_permille: 350 },
        MockFlashcard { front: "State Le Chatelier's principle.", back: "A system at equilibrium shifts to counteract any imposed change (concentration, temperature, pressure).", tags: "Equilibrium", mastery_permille: 300 },
        MockFlashcard { front: "What are the conditions for the Haber process?", back: "N2 + 3H2 \u{21cc} 2NH3. 450\u{00b0}C, 200 atm, iron catalyst.", tags: "Equilibrium", mastery_permille: 250 },
        // Acids & Bases
        MockFlashcard { front: "Strong acid vs weak acid?", back: "Strong: fully ionised (HCl). Weak: partially ionised (CH3COOH). Strength \u{2260} concentration.", tags: "Acids & Bases", mastery_permille: 350 },
        MockFlashcard { front: "What is pH?", back: "pH = \u{2212}log[H3O+]. pH 7 = neutral, < 7 = acidic, > 7 = basic.", tags: "Acids & Bases", mastery_permille: 400 },
        // Electrochemistry
        MockFlashcard { front: "How do you calculate E\u{00b0}cell?", back: "E\u{00b0}cell = E\u{00b0}cathode \u{2212} E\u{00b0}anode. Positive = spontaneous.", tags: "Electrochemistry", mastery_permille: 300 },
        MockFlashcard { front: "AN OX, RED CAT \u{2014} what does it mean?", back: "Anode = Oxidation, Reduction = Cathode. In galvanic cells, anode is negative.", tags: "Electrochemistry", mastery_permille: 350 },
    ]
}

/// Generate flashcards for Grade 9 foundations (cumulative prerequisites).
fn generate_foundations_cards() -> Vec<MockFlashcard> {
    vec![
        // Integers & Operations
        MockFlashcard { front: "What is BODMAS?", back: "Brackets, Orders (exponents), Division, Multiplication, Addition, Subtraction \u{2014} the order of operations.", tags: "Numbers", mastery_permille: 600 },
        MockFlashcard { front: "Why is it dark at night and bright during the day?", back: "Earth spins! When our side faces the Sun, it\u{2019}s day. When it faces away, it\u{2019}s night.", tags: "Space \u{00b7} Foundational", mastery_permille: 700 },
        MockFlashcard { front: "Is the Sun a star?", back: "Yes! The Sun is a star \u{2014} the closest star to Earth. It looks bigger and brighter because it\u{2019}s much closer than other stars.", tags: "Space \u{00b7} Foundational", mastery_permille: 650 },
        MockFlashcard { front: "How many planets are in our solar system?", back: "8 planets! Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. (Pluto is a dwarf planet.)", tags: "Space \u{00b7} Foundational", mastery_permille: 600 },

        // Developing: Grade 3-5
        MockFlashcard { front: "The Moon is about 384,400 km from Earth. Is that more or less than 1 million km?", back: "Less! It\u{2019}s about 384 thousand km \u{2014} not even half a million. But that\u{2019}s still really far!", tags: "Space \u{00b7} Developing", mastery_permille: 450 },
        MockFlashcard { front: "Jupiter is about 11 times wider than Earth. If Earth is a grape, how big is Jupiter?", back: "About the size of a basketball! Jupiter is the biggest planet in our solar system.", tags: "Space \u{00b7} Developing", mastery_permille: 400 },
        MockFlashcard { front: "Why don\u{2019}t we float off Earth into space?", back: "Gravity! Earth pulls everything toward its center. The bigger an object, the stronger its pull.", tags: "Space \u{00b7} Developing", mastery_permille: 500 },
        MockFlashcard { front: "Why does the Moon seem to change shape each month?", back: "It doesn\u{2019}t change shape! We see different amounts of the lit-up side as the Moon orbits Earth. These are called phases.", tags: "Space \u{00b7} Developing", mastery_permille: 350 },
        MockFlashcard { front: "Why do we have seasons?", back: "Earth is tilted! When your part of Earth tilts toward the Sun, you get summer. When it tilts away, winter.", tags: "Space \u{00b7} Developing", mastery_permille: 400 },

        // Proficient: Grade 6-8
        MockFlashcard { front: "Light travels at 300,000 km/s. The Sun is 150 million km away. How long does sunlight take to reach Earth?", back: "500 seconds = about 8 minutes and 20 seconds (150,000,000 \u{00f7} 300,000 = 500)", tags: "Space \u{00b7} Proficient", mastery_permille: 250 },
        MockFlashcard { front: "A light-year is the distance light travels in one year. Why do astronomers use this unit instead of kilometers?", back: "Because space is so vast! The nearest star (Proxima Centauri) is 4.24 light-years away \u{2014} that\u{2019}s about 40 TRILLION km. Light-years are easier to work with.", tags: "Space \u{00b7} Proficient", mastery_permille: 200 },
        MockFlashcard { front: "How does a rocket work in the vacuum of space where there\u{2019}s nothing to push against?", back: "Newton\u{2019}s 3rd Law! The rocket pushes exhaust gas backward, and the gas pushes the rocket forward. Action and reaction \u{2014} no air needed.", tags: "Space \u{00b7} Proficient", mastery_permille: 200 },
        MockFlashcard { front: "What\u{2019}s the difference between mass and weight?", back: "Mass is how much stuff you\u{2019}re made of (same everywhere). Weight is the force of gravity on your mass. You\u{2019}d weigh less on the Moon but your mass stays the same!", tags: "Space \u{00b7} Proficient", mastery_permille: 300 },

        // Advanced: Grade 9-12
        MockFlashcard { front: "The ISS orbits at 408 km altitude and 7.66 km/s. Why doesn\u{2019}t it fall down?", back: "It IS falling \u{2014} but it\u{2019}s moving forward so fast that Earth\u{2019}s surface curves away at the same rate. Orbit = falling around the planet.", tags: "Space \u{00b7} Advanced", mastery_permille: 150 },
        MockFlashcard { front: "Kepler\u{2019}s 3rd Law says T\u{00b2} \u{221d} a\u{00b3} (period squared is proportional to semi-major axis cubed). Mars is 1.52 AU from the Sun. What\u{2019}s its orbital period?", back: "T\u{00b2} = (1.52)\u{00b3} = 3.51, so T = \u{221a}3.51 \u{2248} 1.87 years (actual: 1.88 years)", tags: "Space \u{00b7} Advanced", mastery_permille: 100 },
        MockFlashcard { front: "What is the escape velocity from Earth\u{2019}s surface?", back: "About 11.2 km/s. This is the minimum speed needed to escape Earth\u{2019}s gravity without further propulsion. v = \u{221a}(2GM/r)", tags: "Space \u{00b7} Advanced", mastery_permille: 100 },

        // Expert: University
        // Exponents
        MockFlashcard { front: "Simplify: a\u{00b3} \u{00d7} a\u{2074}", back: "a\u{2077} (add the exponents when bases are the same)", tags: "Exponents", mastery_permille: 600 },
        MockFlashcard { front: "What is a\u{2070}?", back: "1 (any non-zero number to the power 0 equals 1)", tags: "Exponents", mastery_permille: 550 },
        // Algebra fundamentals
        MockFlashcard { front: "Factorise: x\u{00b2} \u{2212} 9", back: "(x + 3)(x \u{2212} 3) \u{2014} difference of two squares", tags: "Algebra", mastery_permille: 500 },
        MockFlashcard { front: "Solve: 3x + 7 = 22", back: "3x = 15, so x = 5", tags: "Algebra", mastery_permille: 600 },
        MockFlashcard { front: "Expand: (x + 3)(x \u{2212} 2)", back: "x\u{00b2} + x \u{2212} 6", tags: "Algebra", mastery_permille: 500 },
        // Geometry
        MockFlashcard { front: "State the theorem of Pythagoras.", back: "a\u{00b2} + b\u{00b2} = c\u{00b2}, where c is the hypotenuse of a right-angled triangle.", tags: "Geometry", mastery_permille: 550 },
        MockFlashcard { front: "Angles on a straight line sum to?", back: "180\u{00b0}", tags: "Geometry", mastery_permille: 700 },
        MockFlashcard { front: "Vertically opposite angles are?", back: "Equal", tags: "Geometry", mastery_permille: 650 },
        // Forces & Motion
        MockFlashcard { front: "What is the formula for speed?", back: "speed = distance / time", tags: "Forces", mastery_permille: 600 },
        MockFlashcard { front: "What is the formula for weight?", back: "w = mg (mass \u{00d7} gravitational acceleration, g \u{2248} 9.8 m/s\u{00b2})", tags: "Forces", mastery_permille: 500 },
        // Matter
        MockFlashcard { front: "What is the difference between an element and a compound?", back: "Element: one type of atom. Compound: two or more elements chemically bonded.", tags: "Matter", mastery_permille: 550 },
        MockFlashcard { front: "Name the three states of matter.", back: "Solid, liquid, gas. Particles are closest together in solids and furthest apart in gases.", tags: "Matter", mastery_permille: 650 },
    ]
}

/// Kid-friendly rating options: emoji, label, and mapped SM-2 quality value.
struct KidRating {
    emoji: &'static str,
    label: &'static str,
    quality: u8,
    css_class: &'static str,
}

const KID_RATINGS: &[KidRating] = &[
    KidRating { emoji: "\u{2716}", label: "Blackout", quality: 1, css_class: "kid-rate-red" },
    KidRating { emoji: "\u{2753}", label: "Hard", quality: 2, css_class: "kid-rate-orange" },
    KidRating { emoji: "\u{2714}", label: "Good", quality: 4, css_class: "kid-rate-green" },
    KidRating { emoji: "\u{26a1}", label: "Easy", quality: 5, css_class: "kid-rate-gold" },
];

// ---------------------------------------------------------------------------
// Helper: apply text complexity to card content
// ---------------------------------------------------------------------------

fn adapt_card_text(text: &str, adaptation: &ContentAdaptation, sovereignty: &SovereigntyLevel) -> String {
    let rewrite = suggest_rewrite(sovereignty, text, &adaptation.text_complexity, 5);
    match rewrite {
        RewriteResult::Applied { rewritten, .. } => rewritten,
        RewriteResult::Offered { rewritten, .. } => {
            // In Guide mode, show the rewritten version (student can toggle back)
            rewritten
        }
        RewriteResult::Available { original } => original,
    }
}

/// Modality indicator text for the current adaptation.
fn modality_indicator(modality: &Modality) -> &'static str {
    match modality {
        Modality::Text => "Reading mode",
        Modality::Visual => "Visual mode",
        Modality::Auditory => "Listen mode",
        Modality::Kinesthetic => "Hands-on mode",
        Modality::MultiModal => "",
    }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn ReviewPage() -> impl IntoView {
    let adaptivity = use_adaptivity();

    // Cards are populated lazily when a topic is selected
    let cards = StoredValue::new(Vec::<MockFlashcard>::new());

    // Extract signals for closures (signals are Copy)
    let adaptation_sig = adaptivity.adaptation;
    let sovereignty_sig = adaptivity.sovereignty;
    let adaptivity_for_rate = adaptivity.clone();

    // Topic selection signal — None means show the topic picker
    let (card_source, set_card_source) = signal(Option::<CardSource>::None);

    // State signals
    let (state, set_state) = signal(ReviewState::Loading);
    let (ratings, set_ratings) = signal(Vec::<u8>::new());
    let (_start_time, set_start_time) = signal(0.0_f64);
    let (card_start_time, set_card_start_time) = signal(0.0_f64);
    let (_total_time_secs, set_total_time_secs) = signal(0.0_f64);
    let (predictions, set_predictions) = signal(Vec::<(Confidence, bool)>::new());
    let (current_prediction, set_current_prediction) = signal(Option::<Confidence>::None);

    // Select a topic: populate cards, reset state, and start the loading transition
    let select_source = {
        let adaptivity = adaptivity.clone();
        move |source: CardSource| {
            let deck = match &source {
                CardSource::CoreMath => generate_curriculum_cards(),
                CardSource::CorePhysics => generate_science_cards(),
                CardSource::CoreChemistry => generate_chemistry_cards(),
                CardSource::Foundations => generate_foundations_cards(),
                CardSource::DynamicSubject(subject) => generate_dynamic_cards(subject),
            };
            cards.set_value(deck);
            set_card_source.set(Some(source));
            set_state.set(ReviewState::Loading);
            set_ratings.set(Vec::new());
            set_predictions.set(Vec::new());
            set_current_prediction.set(None);

            // Simulate loading -> show first card
            let adaptivity = adaptivity.clone();
            set_timeout(
                move || {
                    cards.with_value(|deck| {
                        if deck.is_empty() {
                            set_state.set(ReviewState::NoDueCards);
                        } else {
                            set_start_time.set(js_sys::Date::now());
                            set_card_start_time.set(js_sys::Date::now());
                            let card = &deck[0];
                            adaptivity.set_skill(card.tags, card.mastery_permille);
                            set_state.set(ReviewState::ShowingFront { card_index: 0 });
                        }
                    });
                },
                std::time::Duration::from_millis(300),
            );
        }
    };

    // Reveal: go to confidence prediction (or straight to back in sandbox/autonomous)
    let reveal = move |_| {
        if let ReviewState::ShowingFront { card_index } = state.get() {
            let sov = sovereignty_sig.get();
            if sov.sandbox_active || sov.mode() == InteractionMode::Autonomous {
                set_current_prediction.set(None);
                set_state.set(ReviewState::ShowingBack { card_index });
            } else {
                set_state.set(ReviewState::Predicting { card_index });
            }
        }
    };
    let predict = move |confidence: Confidence| {
        if let ReviewState::Predicting { card_index } = state.get() {
            set_current_prediction.set(Some(confidence));
            set_state.set(ReviewState::ShowingBack { card_index });
        }
    };
    let skip_predict = move |_| {
        if let ReviewState::Predicting { card_index } = state.get() {
            set_current_prediction.set(None);
            set_state.set(ReviewState::ShowingBack { card_index });
        }
    };

    // Store adaptivity ctx for rate closure (StoredValue is Copy)
    let rate_ctx = StoredValue::new(adaptivity_for_rate);

    let set_progress_for_srs = use_set_progress();

    // Rate and advance to next card
    let rate = move |quality: u8| {
        if let ReviewState::ShowingBack { card_index } = state.get() {
            // Record the rating
            set_ratings.update(|r| r.push(quality));

            // Persist SRS state
            let card_id = format!("review:{}", card_index);
            set_progress_for_srs.update(|p| {
                let srs = p.srs_cards.entry(card_id).or_default();
                srs.update(quality);
            });

            // Tell adaptivity engine about the attempt
            let correct = quality >= 3;
            rate_ctx.with_value(|ctx| ctx.record_attempt(correct));
            if let Some(pred) = current_prediction.get() {
                set_predictions.update(|p| p.push((pred, correct)));
            }
            set_current_prediction.set(None);

            // Accumulate time for this card
            let elapsed = (js_sys::Date::now() - card_start_time.get()) / 1000.0;
            set_total_time_secs.update(|t| *t += elapsed);

            let next_index = card_index + 1;
            cards.with_value(|deck| {
                if next_index < deck.len() {
                    set_card_start_time.set(js_sys::Date::now());
                    // Tell adaptivity about the new card
                    let card = &deck[next_index];
                    rate_ctx.with_value(|ctx| ctx.set_skill(card.tags, card.mastery_permille));
                    set_state.set(ReviewState::ShowingFront { card_index: next_index });
                } else {
                    // Session complete
                    let r = ratings.get();
                    let correct_count = r.iter().filter(|&&q| q >= 3).count();
                    let correct_count = if quality >= 3 { correct_count.max(1) } else { correct_count };
                    let reviewed = r.len();
                    set_state.set(ReviewState::SessionComplete { reviewed, correct: correct_count });
                }
            });
        }
    };

    view! {
        <div class="review-page">
            // Suggestion overlay -- slides in from bottom, never blocks
            <SuggestionOverlay />

            {move || {
                // Show topic selector when no source is selected
                if card_source.get().is_none() {
                    let select_math = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::CoreMath)
                    };
                    let select_physics = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::CorePhysics)
                    };
                    let select_chem = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::CoreChemistry)
                    };
                    let select_foundations = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::Foundations)
                    };
                    return view! {
                        <div class="topic-selector">
                            <h2>"What do you want to review?"</h2>
                            <p class="topic-selector-subtitle">"Choose a subject to start your flashcard session."</p>
                            <div class="source-cards">
                                <button class="source-card source-math" on:click=select_math>
                                    <span class="source-icon">"\u{1f4d0}"</span>
                                    <span class="source-label">"Core Maths"</span>
                                    <span class="source-meta">"17 cards"</span>
                                </button>
                                <button class="source-card source-science" on:click=select_physics>
                                    <span class="source-icon">"\u{269b}\u{fe0f}"</span>
                                    <span class="source-label">"Core Physics"</span>
                                    <span class="source-meta">"9 cards"</span>
                                </button>
                                <button class="source-card source-science" on:click=select_chem>
                                    <span class="source-icon">"\u{2697}\u{fe0f}"</span>
                                    <span class="source-label">"Core Chemistry"</span>
                                    <span class="source-meta">"10 cards"</span>
                                </button>
                                <button class="source-card" on:click=select_foundations style="border-color: var(--grade-9)">
                                    <span class="source-icon">"\u{1f9f1}"</span>
                                    <span class="source-label">"Gr9 Foundations"</span>
                                    <span class="source-meta">"12 cards"</span>
                                </button>
                            </div>
                            // Dynamic decks from curriculum subjects
                            <h3 style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 1.5rem; margin-bottom: 0.75rem">"More Subjects"</h3>
                            <div class="source-cards" style="gap: 0.5rem">
                                {
                                    let graph = crate::curriculum::curriculum_graph();
                                    let mut subjects: Vec<(String, usize)> = graph.subjects().iter()
                                        .filter(|s| !["Mathematics", "PhysicalSciences", "NaturalSciences"].contains(s))
                                        .map(|s| {
                                            let count = graph.nodes.iter().filter(|n| n.subject_area == *s).count();
                                            (s.to_string(), count)
                                        })
                                        .filter(|(_, c)| *c >= 2)
                                        .collect();
                                    subjects.sort_by(|a, b| b.1.cmp(&a.1));
                                    subjects.truncate(12);

                                    subjects.into_iter().map(|(subject, count)| {
                                        let subj = subject.clone();
                                        let display = display_subject(&subject);
                                        let label = if display.len() > 22 { format!("{}...", &display[..20]) } else { display };
                                        let select = select_source.clone();
                                        view! {
                                            <button
                                                class="source-card"
                                                style="padding: 0.6rem 0.75rem; min-height: auto"
                                                on:click=move |_| select(CardSource::DynamicSubject(subj.clone()))
                                            >
                                                <span class="source-label" style="font-size: 0.8rem">{label}</span>
                                                <span class="source-meta">{count}" topics"</span>
                                            </button>
                                        }
                                    }).collect::<Vec<_>>()
                                }
                            </div>
                        </div>
                    }.into_any();
                }

                view! { <div></div> }.into_any()
            }}

            // Difficulty/modality indicator bar
            {move || {
                if card_source.get().is_none() {
                    return view! { <div></div> }.into_any();
                }
                let adapt = adaptation_sig.get();
                let mod_text = modality_indicator(&adapt.modality);
                let diff = adapt.difficulty_delta;
                let diff_label = if diff > 0.1 { "Harder" }
                    else if diff < -0.1 { "Easier" }
                    else { "" };

                if mod_text.is_empty() && diff_label.is_empty() {
                    view! { <div class="adaptation-indicator hidden"></div> }.into_any()
                } else {
                    view! {
                        <div class="adaptation-indicator visible">
                            {if !mod_text.is_empty() {
                                view! { <span class="modality-badge">{mod_text}</span> }.into_any()
                            } else {
                                view! { <span></span> }.into_any()
                            }}
                            {if !diff_label.is_empty() {
                                view! { <span class="difficulty-badge">{diff_label}</span> }.into_any()
                            } else {
                                view! { <span></span> }.into_any()
                            }}
                        </div>
                    }.into_any()
                }
            }}

            {move || {
                if card_source.get().is_none() {
                    return view! { <div></div> }.into_any();
                }
                let current = state.get();
                match current {
                    ReviewState::Loading => view! {
                        <div class="review-loading">
                            <div class="review-spinner"></div>
                            <p class="review-loading-text">"Loading due cards..."</p>
                        </div>
                    }.into_any(),

                    ReviewState::NoDueCards => view! {
                        <div class="review-empty">
                            <div class="review-empty-icon">"--"</div>
                            <h2>"All caught up!"</h2>
                            <p>"No cards are due for review right now."</p>
                            <div class="review-stats-mini">
                                <div class="stat-item">
                                    <span class="stat-value">"0"</span>
                                    <span class="stat-label">"Due today"</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-value">"5"</span>
                                    <span class="stat-label">"Total cards"</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-value">"3d"</span>
                                    <span class="stat-label">"Streak"</span>
                                </div>
                            </div>
                        </div>
                    }.into_any(),

                    ReviewState::ShowingFront { card_index } => {
                        let (card_front, card_tags, total) = cards.with_value(|deck| {
                            let card = &deck[card_index];
                            (card.front.to_string(), card.tags.to_string(), deck.len())
                        });
                        let progress_pct = ((card_index as f64 / total as f64) * 100.0) as u32;

                        // Apply text complexity adaptation to card content
                        let adapt = adaptation_sig.get();
                        let sov = sovereignty_sig.get();
                        let adapted_front = adapt_card_text(&card_front, &adapt, &sov);

                        view! {
                            <div class="review-session">
                                <div class="review-progress">
                                    <span class="progress-text">
                                        {format!("Card {} of {}", card_index + 1, total)}
                                    </span>
                                    <div class="progress-bar">
                                        <div class="progress-fill"
                                            style:width=format!("{}%", progress_pct)>
                                        </div>
                                    </div>
                                </div>
                                <div class="flashcard">
                                    <div class="flashcard-inner flashcard-front">
                                        <span class="card-tag">{card_tags}</span>
                                        <div class="card-content">
                                            <p>{adapted_front}</p>
                                        </div>
                                        // Show rewrite explanation in Guardian mode
                                        {move || {
                                            let adapt = adaptation_sig.get();
                                            if adapt.text_complexity != TextComplexity::Standard {
                                                let sov = sovereignty_sig.get();
                                                if sov.mode() == InteractionMode::Guardian {
                                                    return view! {
                                                        <p class="rewrite-note">
                                                            <small>"(Simplified to help you focus on the math)"</small>
                                                        </p>
                                                    }.into_any();
                                                }
                                            }
                                            view! { <span></span> }.into_any()
                                        }}
                                        <button class="reveal-btn" on:click=reveal>
                                            "Show Answer"
                                        </button>
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::Predicting { card_index } => {
                        let (card_front, card_tags, total) = cards.with_value(|deck| {
                            let card = &deck[card_index];
                            (card.front.to_string(), card.tags.to_string(), deck.len())
                        });
                        let progress_pct = ((card_index as f64 / total as f64) * 100.0) as u32;
                        let adapt = adaptation_sig.get();
                        let sov = sovereignty_sig.get();
                        let adapted_front = adapt_card_text(&card_front, &adapt, &sov);

                        view! {
                            <div class="review-session">
                                <div class="review-progress">
                                    <span class="progress-text">{format!("Card {} of {}", card_index + 1, total)}</span>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style:width=format!("{}%", progress_pct)></div>
                                    </div>
                                </div>
                                <div class="flashcard">
                                    <div class="flashcard-inner flashcard-front">
                                        <span class="card-tag">{card_tags}</span>
                                        <div class="card-content"><p>{adapted_front}</p></div>
                                    </div>
                                </div>
                                <div class="confidence-prediction">
                                    <p class="confidence-prompt">"Before you look \u{2014} how sure are you?"</p>
                                    <div class="confidence-grid">
                                        <button class="confidence-btn confidence-know"
                                            on:click=move |_| predict(Confidence::KnowIt)>
                                            <span class="confidence-emoji">"\u{1f4aa}"</span>
                                            <span class="confidence-label">"I know this!"</span>
                                        </button>
                                        <button class="confidence-btn confidence-maybe"
                                            on:click=move |_| predict(Confidence::Maybe)>
                                            <span class="confidence-emoji">"\u{1f914}"</span>
                                            <span class="confidence-label">"Maybe..."</span>
                                        </button>
                                        <button class="confidence-btn confidence-noidea"
                                            on:click=move |_| predict(Confidence::NoIdea)>
                                            <span class="confidence-emoji">"\u{1f937}"</span>
                                            <span class="confidence-label">"No idea"</span>
                                        </button>
                                    </div>
                                    <button class="confidence-skip" on:click=skip_predict>"Just show me"</button>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::ShowingBack { card_index } => {
                        let (card_front, card_back, card_tags, total) = cards.with_value(|deck| {
                            let card = &deck[card_index];
                            (card.front.to_string(), card.back.to_string(), card.tags.to_string(), deck.len())
                        });
                        let progress_pct = (((card_index + 1) as f64 / total as f64) * 100.0) as u32;
                        view! {
                            <div class="review-session">
                                <div class="review-progress">
                                    <span class="progress-text">
                                        {format!("Card {} of {}", card_index + 1, total)}
                                    </span>
                                    <div class="progress-bar">
                                        <div class="progress-fill"
                                            style:width=format!("{}%", progress_pct)>
                                        </div>
                                    </div>
                                </div>
                                <div class="flashcard">
                                    <div class="flashcard-inner flashcard-back">
                                        <span class="card-tag">{card_tags}</span>
                                        <div class="card-content card-content-split">
                                            <div class="card-question">
                                                <span class="label">"Q: "</span>
                                                {card_front}
                                            </div>
                                            <hr class="card-divider" />
                                            <div class="card-answer">
                                                <span class="label">"A: "</span>
                                                {card_back}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="rating-buttons">
                                    <p class="rating-prompt">"How did you do?"</p>
                                    <div class="kid-rating-grid">
                                        {KID_RATINGS.iter().map(|r| {
                                            let q = r.quality;
                                            let class = format!("kid-rate-btn {}", r.css_class);
                                            let emoji = r.emoji;
                                            let label = r.label;
                                            view! {
                                                <button
                                                    class=class
                                                    on:click=move |_| rate(q)
                                                >
                                                    <span class="kid-rate-emoji">{emoji}</span>
                                                    <span class="kid-rate-label">{label}</span>
                                                </button>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }

                    ReviewState::SessionComplete { reviewed, correct } => {
                        let xp = correct * 10 + (reviewed - correct) * 2;
                        let stars_filled = "\u{2b50}".repeat(correct.min(reviewed));
                        let stars_empty = "\u{2606}".repeat(reviewed.saturating_sub(correct));

                        // Show sovereignty growth summary
                        let sov = sovereignty_sig.get();
                        let mode_label = match sov.mode() {
                            InteractionMode::Guardian => "Helper Mode",
                            InteractionMode::Guide => "Guide Mode",
                            InteractionMode::Mirror => "Mirror Mode",
                            InteractionMode::Autonomous => "Independent Mode",
                        };

                        view! {
                            <div class="review-complete kid-complete">
                                <div class="kid-celebration">"\u{1f389}"</div>
                                <h2>"Great job!"</h2>
                                <p class="kid-summary-text">
                                    "You reviewed " {reviewed} " cards"
                                </p>
                                <div class="kid-stars">
                                    <span class="kid-stars-filled">{stars_filled}</span>
                                    <span class="kid-stars-empty">{stars_empty}</span>
                                </div>
                                <p class="kid-stars-label">
                                    {correct} " out of " {reviewed} " correct"
                                </p>
                                <div class="kid-xp-earned">
                                    <span class="kid-xp-badge">{format!("+{} XP earned!", xp)}</span>
                                </div>
                                // Calibration: "Do you know what you know?"
                                {move || {
                                    let preds = predictions.get();
                                    if preds.is_empty() {
                                        return view! { <div></div> }.into_any();
                                    }
                                    let total = preds.len();
                                    let calibrated = preds.iter()
                                        .filter(|(p, c)| p.is_calibrated(*c))
                                        .count();
                                    let underconfident = preds.iter()
                                        .filter(|(p, c)| *p == Confidence::NoIdea && *c)
                                        .count();
                                    let overconfident = preds.iter()
                                        .filter(|(p, c)| *p == Confidence::KnowIt && !*c)
                                        .count();
                                    let msg = if calibrated as f64 / total as f64 >= 0.8 {
                                        format!("You predicted {} out of {} right! You really know what you know.", calibrated, total)
                                    } else if underconfident > overconfident {
                                        format!("You predicted {} out of {}. You know more than you think!", calibrated, total)
                                    } else {
                                        format!("You predicted {} out of {}. Some tricky ones surprised you!", calibrated, total)
                                    };
                                    view! {
                                        <div class="calibration-summary">
                                            <span class="calibration-icon">"\u{1f52e}"</span>
                                            <p class="calibration-message">{msg}</p>
                                        </div>
                                    }.into_any()
                                }}
                                // Sovereignty status
                                <div class="sovereignty-summary">
                                    <span class="sovereignty-badge">{mode_label}</span>
                                    <span class="sovereignty-level">
                                        {format!("Independence: {}/1000", sov.level)}
                                    </span>
                                </div>
                                <div class="kid-complete-actions">
                                    <button class="btn-primary kid-btn"
                                        on:click=move |_| set_card_source.set(None)>
                                        "Keep Going"
                                    </button>
                                    <a href="/" class="btn-secondary kid-btn">"Done for now"</a>
                                </div>
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
