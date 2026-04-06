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

stub_topic!(Gr9MaterialPropertiesTopic, "Properties of Materials",
    "Materials are classified by their properties: metals are shiny, conduct heat and electricity, and are malleable. Non-metals are dull, poor conductors, and brittle. Acids taste sour and turn litmus red (pH < 7). Bases feel slippery and turn litmus blue (pH > 7). Physical changes (melting, dissolving) are reversible; chemical changes (burning, rusting) produce new substances and are usually irreversible.",
    "acid");

stub_topic!(Gr9ParticleModelTopic, "Particle Model of Matter",
    "All matter is made of tiny particles (atoms and molecules). Elements contain only one type of atom. Compounds contain two or more elements chemically bonded — they have different properties from their elements. Mixtures contain substances that are physically combined and can be separated by physical methods (filtration, distillation, chromatography). In solids, particles are closely packed; in liquids, they can slide; in gases, they move freely.",
    "atom");

stub_topic!(Gr9ChemicalReactionsTopic, "Chemical Reactions",
    "When metals react with oxygen, they form metal oxides (e.g., 2Mg + O₂ → 2MgO). More reactive metals react with water (e.g., Na + H₂O → NaOH + H₂). Metals react with acids to produce a salt and hydrogen gas (e.g., Zn + 2HCl → ZnCl₂ + H₂). The law of conservation of mass states that the total mass of reactants equals the total mass of products — atoms are rearranged, not created or destroyed.",
    "reactant");

stub_topic!(Gr9EnergyElectricityTopic, "Energy and Electricity",
    "Energy can be transferred (from one object to another) or transformed (from one type to another). In electrical circuits, a battery provides energy to move charges through a circuit. Current (I) is the rate of flow of charge. Potential difference (V) is the energy per unit charge. Resistance (R) opposes current flow. Cost of electricity is measured in kilowatt-hours (kWh).",
    "current");

stub_topic!(Gr9ForcesMotionTopic, "Forces and Motion",
    "Contact forces require physical contact: friction, applied force, tension, normal force. Non-contact forces act at a distance: gravity, magnetic force, electrostatic force. Weight is the gravitational force on an object: w = mg (mass × gravitational acceleration, g ≈ 9.8 m/s²). Mass is the amount of matter (measured in kg); weight is a force (measured in N). Speed = distance/time.",
    "force");

stub_topic!(Gr9CircuitsTopic, "Electric Cells and Circuits",
    "A battery provides the energy to push electrons through a circuit. In a series circuit, components are connected one after another — the same current flows through all. In a parallel circuit, components are on separate branches — the voltage is the same across each branch. Resistance is measured in ohms (Ω). Adding resistors in series increases total resistance; adding in parallel decreases it.",
    "circuit");

stub_topic!(Gr9WavesSoundLightTopic, "Waves, Sound, and Light",
    "Waves transfer energy without transferring matter. Transverse waves have particles vibrating perpendicular to the wave direction (e.g., light). Longitudinal waves have particles vibrating parallel (e.g., sound). Reflection: angle of incidence = angle of reflection. Refraction: light bends when it enters a different medium because its speed changes. Sound needs a medium to travel — it cannot travel through a vacuum.",
    "reflection");
