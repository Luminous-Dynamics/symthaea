// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NGSS Next Generation Science Standards (US).
//! Cross-cutting concepts + disciplinary core ideas + science practices.

use crate::caps_content::TopicContent;

/// MS-PS1: Matter and Its Interactions (Middle School Physical Science)
pub struct NgssMs1Matter;
impl TopicContent for NgssMs1Matter {
    fn explanation(&self) -> String {
        "All matter is made of atoms — tiny particles too small to see. \
         Atoms combine to form molecules. Water (H₂O) has 2 hydrogen atoms and 1 oxygen atom. \
         Matter exists in three states: solid (fixed shape), liquid (takes container shape), gas (fills space). \
         Changing state (melting, freezing, boiling) is a physical change — atoms rearrange but don't change type. \
         Chemical changes create NEW substances (like rust or burning).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Is melting ice a physical or chemical change?\nIce (solid H₂O) → Water (liquid H₂O)\nThe substance is still H₂O — only the state changed.\nPhysical change (reversible: freeze it back!).".to_string(),
            1 => "Why does a balloon expand when heated?\nHeat gives gas molecules more energy → they move faster.\nFaster molecules push harder on balloon walls → expansion.\nThis is Charles's Law: volume increases with temperature.".to_string(),
            _ => "Explain why sugar dissolves in water but sand doesn't.\nSugar molecules are attracted to water molecules → they separate and mix.\nSand particles are too strongly bonded to each other → water can't pull them apart.\nSolubility depends on molecular interactions.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Name the 3 states of matter.\nSolid, Liquid, Gas\nSolids: fixed shape. Liquids: flow. Gases: fill space.\nSolid, Liquid only\nThink of water: ice, water, steam.\nIce=solid, water=liquid, steam=gas.".to_string(),
            301..=600 => "Is baking a cake physical or chemical change? Explain.\nChemical change\nNew substances form (you can't unbake a cake). Color, texture, and taste all change permanently.\nPhysical change\nCan you reverse it? Can you get flour and eggs back?\nIrreversible + new substances = chemical change.".to_string(),
            _ => "A sealed container holds 2L of gas at 20°C. If heated to 40°C, what happens to pressure?\nPressure increases\nGay-Lussac's Law: pressure ∝ temperature (at constant volume). Molecules move faster → hit walls harder.\nDecreases, Stays same\nMore heat = more molecular energy.\nFaster molecules = more force on container walls.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Think about what happens to the particles.".to_string(), 2 => "Physical change: same substance. Chemical change: new substance.".to_string(), _ => "Key question: can you reverse the change?".to_string() } }
    fn misconception(&self) -> String { "WRONG: When water boils, the bubbles are air.\nRIGHT: The bubbles are water vapor (gaseous H₂O), not air.\nWHY: At 100°C, liquid water gains enough energy to become gas. The bubbles are water escaping as steam.".to_string() }
    fn vocabulary(&self) -> String { "atom: Smallest unit of an element | A gold atom is the smallest piece of gold.\nmolecule: Two or more atoms bonded together | H₂O is a molecule (2H + 1O).\nphysical change: Change in form but not substance | Ice melting to water.\nchemical change: New substance forms | Iron rusting.".to_string() }
    fn flashcard(&self) -> String { "Physical or chemical: cutting paper? | Physical (still paper, just smaller pieces)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Design an experiment to test if temperature affects dissolving speed.\nHypothesis: Sugar dissolves faster in hot water. Procedure: same sugar amount, same water volume, different temps. Measure time to dissolve.\n3\nControl variables: sugar amount, water volume, stirring. Change: temperature.".to_string() }
}

/// HS-LS1: From Molecules to Organisms (High School Life Science)
pub struct NgssHsLifeScience;
impl TopicContent for NgssHsLifeScience {
    fn explanation(&self) -> String {
        "All living organisms are made of cells — the basic unit of life. \
         Cells need energy: plants get it from sunlight (photosynthesis), \
         animals get it from food (cellular respiration). \
         Photosynthesis: 6CO₂ + 6H₂O + sunlight → C₆H₁₂O₆ + 6O₂ \
         Respiration: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP (energy) \
         These are opposite reactions — connected in the carbon cycle.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Why do plants need both sunlight AND carbon dioxide?\nPhotosynthesis: CO₂ + H₂O + light → sugar + O₂\nCO₂ provides carbon atoms for building sugar molecules.\nLight provides energy to break apart CO₂ and H₂O.".to_string(),
            1 => "How does energy flow from the sun to a wolf?\nSun → grass (photosynthesis) → rabbit (eats grass) → wolf (eats rabbit)\nEach transfer loses ~90% as heat.\nOnly ~10% of energy passes to the next level.".to_string(),
            _ => "Explain why exercise makes you breathe harder.\nMuscles need more ATP → more cellular respiration.\nMore respiration needs more O₂ and produces more CO₂.\nYour body breathes faster to get more O₂ and remove CO₂.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "What is the basic unit of life?\nThe cell\nAll living things are made of one or more cells.\nAtom, Molecule, Organ\nThink: smallest LIVING structure.\nAtoms are chemistry. Cells are biology.".to_string(),
            301..=600 => "Why do animals depend on plants, even if they don't eat plants directly?\nPlants produce oxygen through photosynthesis.\nAll animals need O₂ for respiration. Also: plants start every food chain.\nThey don't, Only herbivores depend\nWhat gas do all animals breathe?\nO₂ comes from photosynthesis.".to_string(),
            _ => "A terrarium is sealed with plants and small animals inside. Explain how it sustains itself.\nPlants photosynthesize (CO₂→O₂+sugar). Animals respire (sugar+O₂→CO₂+H₂O). The gases cycle between them.\nIt can't sustain itself, Only plants survive\nWhat do plants produce that animals need? What do animals produce that plants need?\nO₂↔CO₂ cycle. Water also cycles via transpiration and respiration.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Photosynthesis and respiration are opposite reactions.".to_string(), 2 => "Photosynthesis: light energy → chemical energy (sugar).".to_string(), _ => "Plants: CO₂ in, O₂ out. Animals: O₂ in, CO₂ out.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Plants only do photosynthesis. They don't respire.\nRIGHT: Plants do BOTH. They photosynthesize during the day AND respire 24/7.\nWHY: Plants need energy from respiration too — for growth, repair, reproduction.".to_string() }
    fn vocabulary(&self) -> String { "photosynthesis: Making food from light, CO₂, and H₂O | 6CO₂+6H₂O+light→C₆H₁₂O₆+6O₂.\ncellular respiration: Breaking food for energy | C₆H₁₂O₆+6O₂→6CO₂+6H₂O+ATP.\nATP: Cell's energy currency | Powers muscle contraction, nerve signals, etc.".to_string() }
    fn flashcard(&self) -> String { "Products of photosynthesis? | Glucose (C₆H₁₂O₆) and oxygen (O₂)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Explain why deforestation affects atmospheric CO₂ levels.\nFewer trees = less CO₂ absorbed by photosynthesis. Burning trees releases stored carbon. Both increase atmospheric CO₂.\n3\nTrees remove CO₂. What happens when they're gone?".to_string() }
}
