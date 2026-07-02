// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::TopicContent;

macro_rules! stub_topic {
    ($name:ident, $title:expr, $expl:expr, $vocab_term:expr) => {
        pub(crate) struct $name;
        impl TopicContent for $name {
            fn explanation(&self) -> String { $expl.to_string() }
            fn worked_example(&self, _index: usize) -> String {
                format!("Solve a {} problem.\nStep 1: Identify the given information -> read carefully\nStep 2: Apply the relevant formula or method -> show working\nStep 3: State the answer with units -> verify\nAnswer: See worked solution", $title)
            }
            fn practice_problem(&self, _difficulty: u16) -> String {
                format!("Practice: {}\nAnswer\nThe answer follows from applying the correct method.\nWrong answer 1,Wrong answer 2,Wrong answer 3\nThink about what method applies here.,Review the key formula for this topic.", $title)
            }
            fn hint(&self, level: u8) -> String {
                match level { 1 => format!("What concept from {} applies here?", $title), 2 => format!("Apply the method for {} step by step.", $title), _ => format!("Start by writing down what you know about {}.", $title) }
            }
            fn misconception(&self) -> String {
                format!("WRONG: A common error in {}.\nRIGHT: The correct understanding.\nWHY: This mistake is common because students often skip a key step.", $title)
            }
            fn vocabulary(&self) -> String { format!("{}: A key term | Used in this topic.\nconcept: An important idea | Fundamental to understanding.", $vocab_term) }
            fn flashcard(&self) -> String { format!("What is the key idea in {}? | {}", $title, $expl.split('.').next().unwrap_or($title)) }
            fn assessment_item(&self, _context: &str) -> String { format!("Explain {}.\nKey concept answer.\n3\nThink about what you learned.", $title) }
        }
    };
}

stub_topic!(Gr10KinematicsTopic, "Motion in One Dimension",
    "Kinematics describes motion without considering forces. Key concepts: displacement (change in position, can be negative), velocity (rate of change of displacement), acceleration (rate of change of velocity). Equations of motion: v = u + at, s = ut + ½at², v² = u² + 2as. Motion graphs: x-t graphs show position over time (gradient = velocity), v-t graphs show velocity (gradient = acceleration, area = displacement).",
    "displacement");

stub_topic!(Gr10EnergyTopic, "Energy",
    "Gravitational potential energy Ep = mgh depends on mass, gravitational acceleration (g = 9.8 m/s²), and height. Kinetic energy Ek = ½mv² depends on mass and speed. The law of conservation of mechanical energy states that in the absence of friction, total mechanical energy (Ep + Ek) remains constant. Work is done when a force moves an object: W = FΔx cos θ.",
    "kinetic energy");

stub_topic!(Gr10WavesTopic, "Waves — Transverse and Longitudinal",
    "A wave transfers energy without transferring matter. Transverse waves: particles vibrate perpendicular to wave direction (e.g., light, water waves). Longitudinal waves: particles vibrate parallel to wave direction (e.g., sound). Key properties: wavelength (λ) is the distance between two consecutive crests, frequency (f) is the number of waves per second, wave speed v = fλ, period T = 1/f.",
    "wavelength");

stub_topic!(Gr10SoundTopic, "Sound",
    "Sound is a longitudinal wave that needs a medium to travel (cannot travel in a vacuum). Speed of sound depends on the medium: fastest in solids, slowest in gases (~340 m/s in air). Pitch is determined by frequency (higher frequency = higher pitch). Loudness is determined by amplitude (larger amplitude = louder). Echoes occur when sound reflects off surfaces.",
    "pitch");

stub_topic!(Gr10EMRadiationTopic, "Electromagnetic Radiation",
    "The electromagnetic spectrum (from lowest to highest frequency): radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, gamma rays. All EM waves travel at the speed of light (c = 3 × 10⁸ m/s) in a vacuum, are transverse, and do not need a medium. Higher frequency means higher energy and more penetrating. The relationship c = fλ connects speed, frequency, and wavelength.",
    "electromagnetic");
