// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Retention Challenge Banks
//!
//! Question banks for Living Credential vitality refresh. Each credential
//! needs 5-10 quick-check questions to verify ongoing competency.
//!
//! These are NOT full assessments — they're 2-minute retention probes
//! designed to catch genuine forgetting vs. maintained knowledge.

use serde::{Deserialize, Serialize};

/// A retention challenge question.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionQuestion {
    /// The question text
    pub question: String,
    /// Correct answer
    pub correct_answer: String,
    /// Plausible distractors (for multiple choice)
    pub distractors: Vec<String>,
    /// Which credential/skill this tests
    pub skill_tag: String,
    /// Bloom level of the question
    pub bloom_level: String,
}

/// A retention challenge bank for a specific credential domain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetentionBank {
    /// Domain this bank covers
    pub domain: String,
    /// Questions in the bank
    pub questions: Vec<RetentionQuestion>,
}

/// Get the built-in retention challenge banks.
pub fn built_in_banks() -> Vec<RetentionBank> {
    vec![
        caps_math_gr12_bank(),
        caps_physics_gr12_bank(),
        caps_chemistry_gr12_bank(),
        caps_life_sciences_bank(),
        caps_history_bank(),
        caps_geography_bank(),
        ib_biology_bank(),
        rust_programming_bank(),
        holochain_development_bank(),
    ]
}

fn caps_math_gr12_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS Mathematics Grade 12".to_string(),
        questions: vec![
            RetentionQuestion {
                question: "What is the derivative of f(x) = x³ + 2x?".to_string(),
                correct_answer: "f'(x) = 3x² + 2".to_string(),
                distractors: vec![
                    "f'(x) = 3x²".to_string(),
                    "f'(x) = x² + 2".to_string(),
                    "f'(x) = 3x³ + 2x".to_string(),
                ],
                skill_tag: "calculus".to_string(),
                bloom_level: "Apply".to_string(),
            },
            RetentionQuestion {
                question: "For the quadratic y = a(x-p)² + q, what does the value of 'a' determine?".to_string(),
                correct_answer: "The direction (up/down) and width of the parabola".to_string(),
                distractors: vec![
                    "The x-coordinate of the vertex".to_string(),
                    "The y-intercept".to_string(),
                    "The axis of symmetry".to_string(),
                ],
                skill_tag: "functions".to_string(),
                bloom_level: "Understand".to_string(),
            },
            RetentionQuestion {
                question: "Solve: 2x² - 5x - 3 = 0".to_string(),
                correct_answer: "x = 3 or x = -½".to_string(),
                distractors: vec![
                    "x = 3 or x = ½".to_string(),
                    "x = -3 or x = ½".to_string(),
                    "x = 5 or x = -3".to_string(),
                ],
                skill_tag: "algebra".to_string(),
                bloom_level: "Apply".to_string(),
            },
            RetentionQuestion {
                question: "What is the sum of the first 20 terms of the arithmetic sequence: 3, 7, 11, 15, ...?".to_string(),
                correct_answer: "820".to_string(),
                distractors: vec!["400".to_string(), "760".to_string(), "880".to_string()],
                skill_tag: "sequences".to_string(),
                bloom_level: "Apply".to_string(),
            },
            RetentionQuestion {
                question: "In a compound interest formula A = P(1 + i)^n, what does 'n' represent?".to_string(),
                correct_answer: "The number of compounding periods".to_string(),
                distractors: vec![
                    "The interest rate".to_string(),
                    "The principal amount".to_string(),
                    "The final amount".to_string(),
                ],
                skill_tag: "finance".to_string(),
                bloom_level: "Remember".to_string(),
            },
        ],
    }
}

fn caps_physics_gr12_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS Physical Sciences Grade 12 (Physics)".to_string(),
        questions: vec![
            RetentionQuestion {
                question: "State Newton's Second Law of Motion.".to_string(),
                correct_answer: "The net force acting on an object is equal to the product of its mass and acceleration (F_net = ma).".to_string(),
                distractors: vec![
                    "Every action has an equal and opposite reaction.".to_string(),
                    "An object at rest stays at rest unless acted upon by an external force.".to_string(),
                    "Force equals mass times velocity.".to_string(),
                ],
                skill_tag: "mechanics".to_string(),
                bloom_level: "Remember".to_string(),
            },
            RetentionQuestion {
                question: "What is the relationship between EMF, terminal voltage, and lost volts?".to_string(),
                correct_answer: "EMF = V_terminal + V_lost, where V_lost = I × r (internal resistance)".to_string(),
                distractors: vec![
                    "EMF = V_terminal - V_lost".to_string(),
                    "EMF = V_terminal × V_lost".to_string(),
                    "EMF and terminal voltage are always equal".to_string(),
                ],
                skill_tag: "electricity".to_string(),
                bloom_level: "Understand".to_string(),
            },
            RetentionQuestion {
                question: "A 2 kg ball is dropped from 10 m. What is its kinetic energy just before hitting the ground? (g = 10 m/s²)".to_string(),
                correct_answer: "200 J".to_string(),
                distractors: vec!["100 J".to_string(), "20 J".to_string(), "400 J".to_string()],
                skill_tag: "energy".to_string(),
                bloom_level: "Apply".to_string(),
            },
            RetentionQuestion {
                question: "What is the Doppler effect?".to_string(),
                correct_answer: "The change in frequency (or wavelength) of a wave perceived by an observer moving relative to the source.".to_string(),
                distractors: vec![
                    "The bending of light around obstacles.".to_string(),
                    "The splitting of white light into colors.".to_string(),
                    "The reinforcement of waves.".to_string(),
                ],
                skill_tag: "doppler".to_string(),
                bloom_level: "Remember".to_string(),
            },
            RetentionQuestion {
                question: "Calculate the momentum of a 5 kg object moving at 3 m/s.".to_string(),
                correct_answer: "15 kg·m/s".to_string(),
                distractors: vec!["8 kg·m/s".to_string(), "1.67 kg·m/s".to_string(), "45 kg·m/s".to_string()],
                skill_tag: "momentum".to_string(),
                bloom_level: "Apply".to_string(),
            },
        ],
    }
}

fn caps_chemistry_gr12_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS Physical Sciences Grade 12 (Chemistry)".to_string(),
        questions: vec![
            RetentionQuestion {
                question: "What is Le Chatelier's principle?".to_string(),
                correct_answer: "If a system at equilibrium is disturbed, it will shift to counteract the disturbance and re-establish equilibrium.".to_string(),
                distractors: vec![
                    "Reactions always proceed in the forward direction.".to_string(),
                    "The rate of reaction doubles with every 10°C increase.".to_string(),
                    "All reactions are reversible.".to_string(),
                ],
                skill_tag: "equilibrium".to_string(),
                bloom_level: "Remember".to_string(),
            },
            RetentionQuestion {
                question: "What is the pH of a 0.01 M HCl solution?".to_string(),
                correct_answer: "pH = 2".to_string(),
                distractors: vec!["pH = 1".to_string(), "pH = 7".to_string(), "pH = 0.01".to_string()],
                skill_tag: "acids".to_string(),
                bloom_level: "Apply".to_string(),
            },
            RetentionQuestion {
                question: "In an electrochemical cell, at which electrode does oxidation occur?".to_string(),
                correct_answer: "The anode".to_string(),
                distractors: vec!["The cathode".to_string(), "Both electrodes".to_string(), "Neither electrode".to_string()],
                skill_tag: "electrochemistry".to_string(),
                bloom_level: "Remember".to_string(),
            },
        ],
    }
}

fn rust_programming_bank() -> RetentionBank {
    RetentionBank {
        domain: "Rust Programming".to_string(),
        questions: vec![
            RetentionQuestion {
                question: "What is the difference between &str and String in Rust?".to_string(),
                correct_answer: "&str is a borrowed string slice (reference to UTF-8 data), String is an owned, heap-allocated, growable string.".to_string(),
                distractors: vec![
                    "They are the same type.".to_string(),
                    "&str is mutable, String is immutable.".to_string(),
                    "String is stack-allocated, &str is heap-allocated.".to_string(),
                ],
                skill_tag: "rust-types".to_string(),
                bloom_level: "Understand".to_string(),
            },
            RetentionQuestion {
                question: "What does the borrow checker enforce?".to_string(),
                correct_answer: "At any time, you can have either one mutable reference OR any number of immutable references to a value, but not both.".to_string(),
                distractors: vec![
                    "All variables must be mutable.".to_string(),
                    "References cannot outlive the data they point to.".to_string(),
                    "Only one variable can exist at a time.".to_string(),
                ],
                skill_tag: "rust-ownership".to_string(),
                bloom_level: "Understand".to_string(),
            },
            RetentionQuestion {
                question: "What is the Result<T, E> type used for?".to_string(),
                correct_answer: "Representing operations that can succeed (Ok(T)) or fail (Err(E)), enabling explicit error handling without exceptions.".to_string(),
                distractors: vec![
                    "Storing multiple return values.".to_string(),
                    "Implementing async/await.".to_string(),
                    "Type erasure.".to_string(),
                ],
                skill_tag: "rust-error-handling".to_string(),
                bloom_level: "Remember".to_string(),
            },
        ],
    }
}

fn holochain_development_bank() -> RetentionBank {
    RetentionBank {
        domain: "Holochain Development".to_string(),
        questions: vec![
            RetentionQuestion {
                question: "Why should you never mutate a shared entry to count votes in Holochain?".to_string(),
                correct_answer: "Concurrent mutations create branching revision histories. Use link-counting instead: commit standalone entries and create links, then count links to tally.".to_string(),
                distractors: vec![
                    "Holochain doesn't support entry updates.".to_string(),
                    "It would be too slow.".to_string(),
                    "Entries are encrypted and can't be modified.".to_string(),
                ],
                skill_tag: "holochain-patterns".to_string(),
                bloom_level: "Understand".to_string(),
            },
            RetentionQuestion {
                question: "What is the difference between an integrity zome and a coordinator zome?".to_string(),
                correct_answer: "Integrity zomes define entry/link types and validation rules (immutable). Coordinator zomes implement business logic and can be upgraded without breaking data.".to_string(),
                distractors: vec![
                    "They are the same thing with different names.".to_string(),
                    "Integrity zomes handle networking, coordinators handle storage.".to_string(),
                    "Coordinator zomes are optional.".to_string(),
                ],
                skill_tag: "holochain-architecture".to_string(),
                bloom_level: "Remember".to_string(),
            },
            RetentionQuestion {
                question: "Why must time-dependent calculations (like Ebbinghaus decay) only run in coordinators, never in integrity zomes?".to_string(),
                correct_answer: "Every node's clock is slightly different. If integrity validation uses continuous time, nodes will compute different results and reject each other's validations, causing network forks.".to_string(),
                distractors: vec![
                    "Integrity zomes don't have access to the system clock.".to_string(),
                    "Time functions are too expensive for validation.".to_string(),
                    "Coordinator zomes are faster.".to_string(),
                ],
                skill_tag: "holochain-guardrails".to_string(),
                bloom_level: "Analyze".to_string(),
            },
        ],
    }
}

fn caps_life_sciences_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS Life Sciences".to_string(),
        questions: vec![
            RetentionQuestion { question: "What is the function of mitochondria?".to_string(), correct_answer: "Produce ATP (energy) through cellular respiration.".to_string(), distractors: vec!["Store DNA".to_string(), "Photosynthesis".to_string(), "Protein synthesis".to_string()], skill_tag: "cell-biology".to_string(), bloom_level: "Remember".to_string() },
            RetentionQuestion { question: "What is osmosis?".to_string(), correct_answer: "Movement of water across a semi-permeable membrane from high to low water concentration.".to_string(), distractors: vec!["Movement of any substance".to_string(), "Active transport of water".to_string(), "Diffusion of gases".to_string()], skill_tag: "transport".to_string(), bloom_level: "Understand".to_string() },
            RetentionQuestion { question: "What are the base pairing rules in DNA?".to_string(), correct_answer: "Adenine pairs with Thymine (A-T), Guanine pairs with Cytosine (G-C).".to_string(), distractors: vec!["A-G and T-C".to_string(), "A-C and T-G".to_string(), "A-U and T-C".to_string()], skill_tag: "genetics".to_string(), bloom_level: "Remember".to_string() },
            RetentionQuestion { question: "Why is the Cape Floral Kingdom globally significant?".to_string(), correct_answer: "Smallest of 6 floral kingdoms but most diverse per area. ~9000 species, 70% endemic (fynbos).".to_string(), distractors: vec!["Largest kingdom".to_string(), "Most trees".to_string(), "Oldest plants".to_string()], skill_tag: "biodiversity".to_string(), bloom_level: "Understand".to_string() },
        ],
    }
}

fn caps_history_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS History".to_string(),
        questions: vec![
            RetentionQuestion { question: "When did the Dutch arrive at the Cape?".to_string(), correct_answer: "1652 — Jan van Riebeeck established a refreshment station for the VOC.".to_string(), distractors: vec!["1806".to_string(), "1488".to_string(), "1994".to_string()], skill_tag: "colonialism".to_string(), bloom_level: "Remember".to_string() },
            RetentionQuestion { question: "What was the 'Scramble for Africa'?".to_string(), correct_answer: "European nations dividing and colonising Africa (1881-1914), formalised at the Berlin Conference (1884).".to_string(), distractors: vec!["African independence movements".to_string(), "A gold rush".to_string(), "A trade route".to_string()], skill_tag: "imperialism".to_string(), bloom_level: "Understand".to_string() },
            RetentionQuestion { question: "What is the difference between a primary and secondary source?".to_string(), correct_answer: "Primary: from the time period (diary, photo). Secondary: written later analysing primary sources (textbook, documentary).".to_string(), distractors: vec!["Same thing".to_string(), "Primary=important, secondary=not".to_string(), "Primary=government, secondary=private".to_string()], skill_tag: "source-analysis".to_string(), bloom_level: "Understand".to_string() },
        ],
    }
}

fn caps_geography_bank() -> RetentionBank {
    RetentionBank {
        domain: "CAPS Geography".to_string(),
        questions: vec![
            RetentionQuestion { question: "Name South Africa's two main ocean currents.".to_string(), correct_answer: "Benguela (cold, west coast) and Agulhas (warm, east coast).".to_string(), distractors: vec!["Gulf Stream and Kuroshio".to_string(), "Only one current".to_string(), "Benguela and Atlantic".to_string()], skill_tag: "climate".to_string(), bloom_level: "Remember".to_string() },
            RetentionQuestion { question: "What is the lapse rate?".to_string(), correct_answer: "Temperature drops ~6.5°C per 1000m altitude increase.".to_string(), distractors: vec!["Speed of wind".to_string(), "Rate of rainfall".to_string(), "Temperature increase per km".to_string()], skill_tag: "atmosphere".to_string(), bloom_level: "Remember".to_string() },
            RetentionQuestion { question: "Explain the rain shadow effect.".to_string(), correct_answer: "Mountains force air up (rain on windward side). Air descends on the other side, warms, and creates dry conditions (leeward/rain shadow).".to_string(), distractors: vec!["Mountains block all rain".to_string(), "Shadow of clouds".to_string(), "Only happens at night".to_string()], skill_tag: "geomorphology".to_string(), bloom_level: "Understand".to_string() },
        ],
    }
}

fn ib_biology_bank() -> RetentionBank {
    RetentionBank {
        domain: "IB Biology".to_string(),
        questions: vec![
            RetentionQuestion { question: "What happens to enzymes above 40°C?".to_string(), correct_answer: "Denaturation: heat breaks hydrogen bonds, destroying the active site shape. Substrate can no longer fit.".to_string(), distractors: vec!["They work faster".to_string(), "They dissolve".to_string(), "Nothing changes".to_string()], skill_tag: "enzymes".to_string(), bloom_level: "Understand".to_string() },
            RetentionQuestion { question: "What is the difference between condensation and hydrolysis?".to_string(), correct_answer: "Condensation: monomers join, releasing H₂O (builds polymers). Hydrolysis: water breaks a bond (breaks polymers).".to_string(), distractors: vec!["Same reaction".to_string(), "Condensation uses water".to_string(), "Hydrolysis builds polymers".to_string()], skill_tag: "molecular-biology".to_string(), bloom_level: "Understand".to_string() },
            RetentionQuestion { question: "Why is water called the 'universal solvent'?".to_string(), correct_answer: "Water's polarity allows it to dissolve many ionic and polar substances. Most biochemical reactions occur in aqueous solution.".to_string(), distractors: vec!["It dissolves everything".to_string(), "It's the most common liquid".to_string(), "It has no charge".to_string()], skill_tag: "water-properties".to_string(), bloom_level: "Understand".to_string() },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_built_in_banks_not_empty() {
        let banks = built_in_banks();
        assert!(banks.len() >= 5, "Should have at least 5 domain banks");
        for bank in &banks {
            assert!(!bank.questions.is_empty(), "Bank '{}' has no questions", bank.domain);
            assert!(bank.questions.len() >= 3, "Bank '{}' needs at least 3 questions", bank.domain);
        }
    }

    #[test]
    fn test_all_questions_have_distractors() {
        for bank in built_in_banks() {
            for q in &bank.questions {
                assert!(
                    q.distractors.len() >= 2,
                    "Question '{}' needs at least 2 distractors",
                    q.question
                );
            }
        }
    }

    #[test]
    fn test_all_questions_have_bloom_level() {
        let valid_blooms = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"];
        for bank in built_in_banks() {
            for q in &bank.questions {
                assert!(
                    valid_blooms.contains(&q.bloom_level.as_str()),
                    "Invalid Bloom level '{}' in question '{}'",
                    q.bloom_level, q.question
                );
            }
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let banks = built_in_banks();
        let json = serde_json::to_string(&banks).unwrap();
        let parsed: Vec<RetentionBank> = serde_json::from_str(&json).unwrap();
        assert_eq!(banks.len(), parsed.len());
    }

    #[test]
    fn test_total_question_count() {
        let total: usize = built_in_banks().iter().map(|b| b.questions.len()).sum();
        assert!(total >= 19, "Should have at least 19 total retention questions, got {}", total);
    }
}
