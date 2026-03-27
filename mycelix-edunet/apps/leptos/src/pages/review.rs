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

// ---------------------------------------------------------------------------
// Review state machine
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum CardSource {
    Grade3Math,
    Grade3Science,
    SpaceExplorer,
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

/// Generate real flashcards from Common Core Grade 3 Math standards.
/// Covers all 5 domains: OA, NBT, NF, MD, G.
fn generate_curriculum_cards() -> Vec<MockFlashcard> {
    vec![
        // === Operations & Algebraic Thinking (3.OA) ===
        // 3.OA.A.1: Interpret products
        MockFlashcard { front: "3 \u{00d7} 6 means 3 groups of 6. How many total?", back: "18", tags: "Multiplication", mastery_permille: 700 },
        MockFlashcard { front: "Draw an array for 4 \u{00d7} 5", back: "4 rows of 5 = 20", tags: "Multiplication", mastery_permille: 650 },
        // 3.OA.A.2: Interpret quotients
        MockFlashcard { front: "24 \u{00f7} 6 = ? (How many groups of 6 make 24?)", back: "4", tags: "Division", mastery_permille: 400 },
        MockFlashcard { front: "Share 15 stickers equally among 3 friends. How many each?", back: "5 stickers each (15 \u{00f7} 3 = 5)", tags: "Division", mastery_permille: 450 },
        // 3.OA.A.3: Multiply and divide word problems
        MockFlashcard { front: "A garden has 5 rows with 7 flowers in each row. How many flowers?", back: "35 flowers (5 \u{00d7} 7 = 35)", tags: "Word Problems", mastery_permille: 350 },
        MockFlashcard { front: "Mom bought 32 cookies for 8 children. How many each?", back: "4 cookies each (32 \u{00f7} 8 = 4)", tags: "Word Problems", mastery_permille: 300 },
        // 3.OA.A.4: Unknown in multiplication
        MockFlashcard { front: "? \u{00d7} 8 = 48", back: "6 (because 6 \u{00d7} 8 = 48)", tags: "Multiplication", mastery_permille: 500 },
        MockFlashcard { front: "7 \u{00d7} ? = 56", back: "8 (because 7 \u{00d7} 8 = 56)", tags: "Multiplication", mastery_permille: 480 },
        // 3.OA.B.5: Properties of multiplication
        MockFlashcard { front: "If 3 \u{00d7} 5 \u{00d7} 2 is hard, can you rearrange? Try (3 \u{00d7} 2) \u{00d7} 5", back: "6 \u{00d7} 5 = 30. Same answer! (Associative property)", tags: "Multiplication", mastery_permille: 400 },
        // 3.OA.B.6: Division as unknown factor
        MockFlashcard { front: "56 \u{00f7} 8 = ? Think: 8 \u{00d7} ? = 56", back: "7", tags: "Division", mastery_permille: 420 },
        // 3.OA.C.7: Fluently multiply/divide within 100
        MockFlashcard { front: "9 \u{00d7} 7 = ?", back: "63", tags: "Multiplication", mastery_permille: 600 },
        MockFlashcard { front: "8 \u{00d7} 6 = ?", back: "48", tags: "Multiplication", mastery_permille: 620 },
        MockFlashcard { front: "72 \u{00f7} 9 = ?", back: "8", tags: "Division", mastery_permille: 380 },
        MockFlashcard { front: "7 \u{00d7} 4 = ?", back: "28", tags: "Multiplication", mastery_permille: 700 },
        // 3.OA.D.8: Two-step word problems
        MockFlashcard { front: "You have $20. You buy 3 books at $4 each. How much left?", back: "$8 (3 \u{00d7} $4 = $12, $20 - $12 = $8)", tags: "Word Problems", mastery_permille: 250 },
        MockFlashcard { front: "A baker makes 6 trays of 8 muffins, then eats 3. How many left?", back: "45 (6 \u{00d7} 8 = 48, 48 - 3 = 45)", tags: "Word Problems", mastery_permille: 200 },
        // 3.OA.D.9: Patterns in addition/multiplication tables
        MockFlashcard { front: "Look at the pattern: 4, 8, 12, 16, 20. What comes next?", back: "24 (adding 4 each time \u{2014} the 4\u{00d7} table!)", tags: "Patterns", mastery_permille: 550 },

        // === Number & Operations in Base Ten (3.NBT) ===
        // 3.NBT.A.1: Round to nearest 10/100
        MockFlashcard { front: "Round 67 to the nearest ten", back: "70", tags: "Rounding", mastery_permille: 600 },
        MockFlashcard { front: "Round 345 to the nearest hundred", back: "300", tags: "Rounding", mastery_permille: 500 },
        MockFlashcard { front: "Round 851 to the nearest hundred", back: "900", tags: "Rounding", mastery_permille: 480 },
        // 3.NBT.A.2: Fluently add/subtract within 1000
        MockFlashcard { front: "456 + 237 = ?", back: "693", tags: "Addition", mastery_permille: 750 },
        MockFlashcard { front: "803 - 256 = ?", back: "547", tags: "Subtraction", mastery_permille: 680 },
        // 3.NBT.A.3: Multiply one-digit x multiples of 10
        MockFlashcard { front: "6 \u{00d7} 70 = ?", back: "420 (6 \u{00d7} 7 = 42, then add a zero)", tags: "Multiplication", mastery_permille: 350 },

        // === Number & Operations -- Fractions (3.NF) ===
        // 3.NF.A.1: Understand fractions
        MockFlashcard { front: "A pizza is cut into 8 equal slices. You eat 3. What fraction did you eat?", back: "3/8", tags: "Fractions", mastery_permille: 400 },
        MockFlashcard { front: "What does the 4 mean in 3/4?", back: "The whole is divided into 4 equal parts", tags: "Fractions", mastery_permille: 350 },
        // 3.NF.A.2: Fractions on number line
        MockFlashcard { front: "Where does 1/2 go on a number line from 0 to 1?", back: "Exactly in the middle, halfway between 0 and 1", tags: "Fractions", mastery_permille: 450 },
        // 3.NF.A.3: Equivalent fractions, comparing
        MockFlashcard { front: "Which is bigger: 1/3 or 1/4?", back: "1/3 is bigger (fewer parts = bigger pieces)", tags: "Fractions", mastery_permille: 300 },
        MockFlashcard { front: "Are 2/4 and 1/2 the same?", back: "Yes! 2/4 = 1/2 (equivalent fractions)", tags: "Fractions", mastery_permille: 320 },

        // === Measurement & Data (3.MD) ===
        // 3.MD.C.5-7: Area
        MockFlashcard { front: "A rectangle is 4 units wide and 3 units tall. What's the area?", back: "12 square units (4 \u{00d7} 3 = 12)", tags: "Geometry", mastery_permille: 250 },
        MockFlashcard { front: "A square has sides of 5. What's its area?", back: "25 square units (5 \u{00d7} 5 = 25)", tags: "Geometry", mastery_permille: 280 },
        // 3.MD.A.1: Tell time
        MockFlashcard { front: "What time is 15 minutes after 2:45?", back: "3:00", tags: "Time", mastery_permille: 550 },
        // 3.MD.B.3: Picture graphs/bar graphs
        MockFlashcard { front: "A bar graph shows: Red=5, Blue=8, Green=3. How many more blue than green?", back: "5 more (8 - 3 = 5)", tags: "Data", mastery_permille: 500 },
        // 3.MD.D.8: Perimeter
        MockFlashcard { front: "A rectangle is 6 long and 4 wide. What's the perimeter?", back: "20 units (6 + 4 + 6 + 4 = 20)", tags: "Geometry", mastery_permille: 300 },

        // === Geometry (3.G) ===
        // 3.G.A.1: Shapes and attributes
        MockFlashcard { front: "How many sides does a hexagon have?", back: "6 sides", tags: "Geometry", mastery_permille: 600 },
        MockFlashcard { front: "A rhombus has 4 equal sides. Is it always a square?", back: "No \u{2014} a square has 4 equal sides AND 4 right angles. A rhombus might not.", tags: "Geometry", mastery_permille: 220 },
        // 3.G.A.2: Partition shapes into equal parts
        MockFlashcard { front: "Split a rectangle into 4 equal parts. What fraction is each part?", back: "1/4 (each part is one-fourth of the whole)", tags: "Fractions", mastery_permille: 380 },
    ]
}

/// Generate Grade 3 Science flashcards based on NGSS standards.
/// Covers Life Science (LS1-LS4), Physical Science (PS2), Earth & Space Science (ESS2-ESS3).
fn generate_science_cards() -> Vec<MockFlashcard> {
    vec![
        // Life Science: 3-LS1 (Inheritance & Variation of Traits)
        MockFlashcard { front: "Do baby animals always look exactly like their parents?", back: "No! They look similar but not identical. A puppy might have a different fur color than its mom.", tags: "Life Science", mastery_permille: 500 },
        MockFlashcard { front: "Why do some flowers in a garden look different from each other even though they\u{2019}re the same kind?", back: "Plants (and animals) of the same kind can have different traits. This is called variation.", tags: "Life Science", mastery_permille: 450 },

        // Life Science: 3-LS2 (Ecosystems)
        MockFlashcard { front: "What would happen to a pond ecosystem if all the frogs disappeared?", back: "The insects frogs eat would increase a lot, and the animals that eat frogs (like herons) would have less food.", tags: "Ecosystems", mastery_permille: 350 },
        MockFlashcard { front: "Can an animal survive if its environment changes?", back: "Some can adapt or move to a new place. Others might not survive. That\u{2019}s why protecting habitats matters!", tags: "Ecosystems", mastery_permille: 400 },

        // Life Science: 3-LS3 (Heredity)
        MockFlashcard { front: "Why do you look a bit like your parents?", back: "You inherit traits from both parents! Eye color, hair type, and even how tall you\u{2019}ll grow.", tags: "Heredity", mastery_permille: 500 },

        // Life Science: 3-LS4 (Evolution: Fossils)
        MockFlashcard { front: "What can fossils tell us about animals that lived long ago?", back: "Fossils show us what ancient animals looked like, what they ate, and how they\u{2019}ve changed over time.", tags: "Fossils", mastery_permille: 300 },
        MockFlashcard { front: "If you found a fossil of a seashell on top of a mountain, what might that tell you?", back: "That area was probably underwater millions of years ago! The land changed over a very long time.", tags: "Fossils", mastery_permille: 250 },

        // Physical Science: 3-PS2 (Motion and Stability: Forces)
        MockFlashcard { front: "If you push a ball harder, what happens?", back: "It goes faster and farther! A bigger push (force) means more motion.", tags: "Forces", mastery_permille: 600 },
        MockFlashcard { front: "Two magnets are near each other. One flips around. What happened?", back: "The magnets\u{2019} poles switched! Like poles (N-N or S-S) push apart, opposite poles (N-S) pull together.", tags: "Forces", mastery_permille: 400 },
        MockFlashcard { front: "Can something move without being touched?", back: "Yes! Magnets can push or pull without touching. Gravity pulls things down without touching them too.", tags: "Forces", mastery_permille: 350 },

        // Earth & Space Science: 3-ESS2 (Earth's Systems: Weather)
        MockFlashcard { front: "Is weather the same as climate?", back: "No! Weather is what\u{2019}s happening outside right now. Climate is the typical weather pattern over many years.", tags: "Weather", mastery_permille: 450 },
        MockFlashcard { front: "What causes wind?", back: "The sun heats the ground unevenly. Warm air rises and cool air rushes in to fill the space. That moving air is wind!", tags: "Weather", mastery_permille: 300 },

        // Earth & Space Science: 3-ESS2 (Earth's Systems: Water)
        MockFlashcard { front: "Where does rain come from?", back: "Water evaporates from oceans and lakes, forms clouds, and falls back down as rain. This is the water cycle!", tags: "Water Cycle", mastery_permille: 500 },

        // Earth & Space Science: 3-ESS3 (Earth and Human Activity)
        MockFlashcard { front: "Can people change the land? Give an example.", back: "Yes! Building cities, farming, cutting forests, and mining all change the land. Some changes help, some hurt.", tags: "Human Impact", mastery_permille: 400 },
        MockFlashcard { front: "What\u{2019}s one way we can reduce the impact of a natural hazard like flooding?", back: "Build levees, plant trees to absorb water, move buildings away from flood zones, or create wetlands.", tags: "Natural Hazards", mastery_permille: 350 },
    ]
}

/// Generate Space Explorer pathway cards spanning PreK through university.
/// Interest-driven pathway crossing Math and Science boundaries.
fn generate_space_pathway_cards() -> Vec<MockFlashcard> {
    vec![
        // Foundational: PreK-Grade 2
        MockFlashcard { front: "Look up at the sky at night. What do you see?", back: "Stars! They look tiny but they\u{2019}re actually HUGE \u{2014} they\u{2019}re just very, very far away.", tags: "Space \u{00b7} Foundational", mastery_permille: 800 },
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
        MockFlashcard { front: "In the Hertzsprung-Russell diagram, where are main sequence stars located?", back: "Along a diagonal band from hot/bright (upper left) to cool/dim (lower right). Our Sun is a G2V star in the middle.", tags: "Space \u{00b7} Expert", mastery_permille: 50 },
        MockFlashcard { front: "What is the Schwarzschild radius of a black hole with 10 solar masses?", back: "r\u{209b} = 2GM/c\u{00b2} = 2(6.674\u{00d7}10\u{207b}\u{00b9}\u{00b9})(10\u{00d7}1.989\u{00d7}10\u{00b3}\u{2070})/(3\u{00d7}10\u{2078})\u{00b2} \u{2248} 29.5 km", tags: "Space \u{00b7} Expert", mastery_permille: 30 },
        MockFlashcard { front: "Why does time pass slower near a massive object?", back: "General Relativity: mass curves spacetime. Clocks in stronger gravitational fields tick slower. GPS satellites correct for this daily!", tags: "Space \u{00b7} Expert", mastery_permille: 50 },
        MockFlashcard { front: "Explain the twin paradox in special relativity.", back: "A twin traveling near light speed ages less than the twin on Earth. This isn\u{2019}t a paradox \u{2014} the traveling twin accelerated (broke inertial symmetry), making the situation asymmetric.", tags: "Space \u{00b7} Expert", mastery_permille: 30 },
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
    KidRating { emoji: "\u{1f61f}", label: "I don't know this yet", quality: 1, css_class: "kid-rate-red" },
    KidRating { emoji: "\u{1f914}", label: "I'm still learning", quality: 2, css_class: "kid-rate-orange" },
    KidRating { emoji: "\u{1f60a}", label: "I got it!", quality: 4, css_class: "kid-rate-green" },
    KidRating { emoji: "\u{1f31f}", label: "Too easy!", quality: 5, css_class: "kid-rate-gold" },
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
                CardSource::Grade3Math => generate_curriculum_cards(),
                CardSource::Grade3Science => generate_science_cards(),
                CardSource::SpaceExplorer => generate_space_pathway_cards(),
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

    // Rate and advance to next card
    let rate = move |quality: u8| {
        if let ReviewState::ShowingBack { card_index } = state.get() {
            // Record the rating
            set_ratings.update(|r| r.push(quality));

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
                        move |_| select_source(CardSource::Grade3Math)
                    };
                    let select_science = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::Grade3Science)
                    };
                    let select_space = {
                        let select_source = select_source.clone();
                        move |_| select_source(CardSource::SpaceExplorer)
                    };
                    return view! {
                        <div class="topic-selector">
                            <h2>"What do you want to study?"</h2>
                            <p class="topic-selector-subtitle">"Pick a topic and let\u{2019}s go!"</p>
                            <div class="source-cards">
                                <button class="source-card source-math" on:click=select_math>
                                    <span class="source-icon">"\u{1f522}"</span>
                                    <span class="source-label">"Grade 3 Math"</span>
                                    <span class="source-meta">"35 cards"</span>
                                </button>
                                <button class="source-card source-science" on:click=select_science>
                                    <span class="source-icon">"\u{1f52c}"</span>
                                    <span class="source-label">"Grade 3 Science"</span>
                                    <span class="source-meta">"15 cards"</span>
                                </button>
                                <button class="source-card source-space" on:click=select_space>
                                    <span class="source-icon">"\u{1f680}"</span>
                                    <span class="source-label">"Space Explorer"</span>
                                    <span class="source-meta">"PreK \u{2192} University"</span>
                                </button>
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
