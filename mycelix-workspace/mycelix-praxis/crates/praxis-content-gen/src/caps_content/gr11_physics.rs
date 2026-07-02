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

stub_topic!(Gr11VectorsTopic, "Vectors in Two Dimensions",
    "Vectors have both magnitude and direction. To add vectors: use the head-to-tail method or resolve into components. Resolution: Fx = F cos θ, Fy = F sin θ (where θ is measured from the horizontal). The resultant = √(Rx² + Ry²) at angle tan⁻¹(Ry/Rx). On inclined planes: weight resolves into a component parallel to the slope (mg sin θ) and perpendicular (mg cos θ).",
    "resultant");

stub_topic!(Gr11NewtonsLawsTopic, "Newton's Laws and Application",
    "Newton's 1st Law: An object remains at rest or moves at constant velocity unless acted on by a net force. 2nd Law: Fnet = ma — acceleration is proportional to net force and inversely proportional to mass. 3rd Law: For every action, there is an equal and opposite reaction (forces act on different objects). Draw free-body diagrams showing all forces. Friction: f = μN where μ is the coefficient of friction and N is the normal force.",
    "Newton");

stub_topic!(Gr11WorkEnergyPowerTopic, "Work, Energy, and Power",
    "Work W = FΔx cos θ (only the component of force in the direction of motion does work). Work-energy theorem: Wnet = ΔEk. Conservation of energy with friction: Wnc = ΔEk + ΔEp (non-conservative work equals total energy change). Power P = W/Δt = Fv (power is the rate of doing work). Efficiency = useful work output / total energy input × 100%.",
    "power");

stub_topic!(Gr11OpticsTopic, "Geometrical Optics",
    "Refraction: light bends when passing between media of different optical density. Snell's law: n₁ sin θ₁ = n₂ sin θ₂. Total internal reflection occurs when light travels from a denser to a less dense medium at an angle greater than the critical angle: sin θc = n₂/n₁. Lenses: converging (convex) and diverging (concave). Thin lens equation: 1/f = 1/do + 1/di. Magnification m = -di/do.",
    "refraction");
