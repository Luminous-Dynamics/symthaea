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

stub_topic!(Gr10AtomicStructureTopic, "Atomic Structure",
    "Atoms consist of protons and neutrons in the nucleus, with electrons in energy levels around it. Atomic number (Z) = number of protons = number of electrons (in a neutral atom). Mass number (A) = protons + neutrons. Isotopes are atoms of the same element with different numbers of neutrons. Electron configuration follows the Aufbau principle: fill lower energy levels first (2, 8, 18, 32 electrons per level).",
    "atomic number");

stub_topic!(Gr10BondingTopic, "Chemical Bonding",
    "Ionic bonding: metal atoms transfer electrons to non-metal atoms, forming positive and negative ions that attract. Covalent bonding: non-metal atoms share electron pairs. Draw Lewis dot structures to show bonding. VSEPR theory predicts molecular shapes: 2 bonding pairs = linear, 3 = trigonal planar, 4 = tetrahedral, 2 bonding + 2 lone pairs = bent/V-shape. Metallic bonding: positive metal ions in a 'sea' of delocalised electrons.",
    "ionic bond");

stub_topic!(Gr10IntermolecularForcesTopic, "Intermolecular Forces",
    "Intermolecular forces (IMFs) are forces between molecules — NOT the same as chemical bonds within molecules. Types: London/dispersion forces (weakest, all molecules), dipole-dipole forces (polar molecules), hydrogen bonds (strongest IMF — when H is bonded to N, O, or F). Stronger IMFs = higher boiling point, higher viscosity, higher surface tension. Water has unusually high boiling point because of hydrogen bonding.",
    "intermolecular");

stub_topic!(Gr10StoichiometryTopic, "Stoichiometry",
    "The mole is the chemist's counting unit: 1 mol = 6.022 × 10²³ particles (Avogadro's number). Molar mass M (g/mol) is found from the periodic table. Key formulas: n = m/M (moles from mass), n = cV (moles from concentration and volume in dm³), n = N/NA (moles from number of particles). In balanced equations, coefficients give mole ratios. The limiting reagent is used up first and determines how much product forms.",
    "mole");

stub_topic!(Gr10TypesOfReactionsTopic, "Types of Reactions",
    "Acid-base (neutralisation): acid + base → salt + water. Redox: involves transfer of electrons — oxidation is loss of electrons (OIL), reduction is gain of electrons (RIG). Assign oxidation numbers to identify what is oxidised and reduced. Combustion: substance reacts with oxygen, releasing energy. Synthesis: A + B → AB. Decomposition: AB → A + B.",
    "oxidation");

stub_topic!(Gr10WaterChemistryTopic, "Water Chemistry",
    "Water is essential for life and is treated to make it safe for drinking. Water purification steps: screening (removes large debris), flocculation (chemicals cause particles to clump), sedimentation (clumps settle), filtration (removes remaining particles), chlorination (kills bacteria). Water pollution sources: industrial waste, agricultural runoff (fertilisers cause eutrophication), sewage. pH testing measures water acidity/alkalinity.",
    "purification");
