// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS Life Sciences Grade 10 — Chemistry of Life, Cells, Cell Division.

use super::TopicContent;

pub(crate) struct Gr10CellBiology;
impl TopicContent for Gr10CellBiology {
    fn explanation(&self) -> String {
        "All living things are made of cells. Plant cells have a cell wall, chloroplasts, and a large vacuole. \
         Animal cells have no cell wall, no chloroplasts, and smaller vacuoles. \
         The cell membrane controls what enters and exits. The nucleus contains DNA — the instructions for life. \
         Mitochondria are the 'powerhouses' — they produce ATP through cellular respiration.".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Name 3 differences between plant and animal cells.\n1. Cell wall: plants YES, animals NO\n2. Chloroplasts: plants YES, animals NO\n3. Vacuole: plants LARGE central, animals SMALL/many".to_string(),
            1 => "Why can't animal cells photosynthesize?\nThey lack chloroplasts — the organelle containing chlorophyll.\nChlorophyll is the green pigment that captures light energy.\nWithout it, no photosynthesis is possible.".to_string(),
            _ => "A student looks at a cell under a microscope. She sees a cell wall and chloroplasts. What type?\nPlant cell — both structures are unique to plants.\nAnimal cells have neither cell walls nor chloroplasts.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "Which organelle is the 'powerhouse' of the cell?\nMitochondria\nMitochondria produce ATP through cellular respiration.\nNucleus, Ribosome, Cell membrane\nWhat produces energy?\nATP = energy currency. Mitochondria make it.".to_string(),
            301..=600 => "Explain why red blood cells have no nucleus.\nThey need maximum space for haemoglobin to carry oxygen.\nNo nucleus = more room for haemoglobin = more efficient O₂ transport.\nThey lost it by accident, All cells need nuclei\nWhat is the main job of red blood cells?\nCarrying oxygen. More haemoglobin = better at the job.".to_string(),
            _ => "A cell is placed in a concentrated salt solution. Predict what happens and explain.\nThe cell shrinks (crenation/plasmolysis). Water moves OUT by osmosis.\nWater moves from high concentration (inside cell) to low concentration (salty solution outside) through the semi-permeable membrane.\nIt expands, Nothing happens\nWhich side has more water molecules?\nInside the cell has MORE water. Water moves to where there's LESS water.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Think about what each organelle DOES.".to_string(), 2 => "Structure determines function: shape and parts relate to job.".to_string(), _ => "Osmosis: water moves from dilute to concentrated solution.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Only plant cells have a cell membrane.\nRIGHT: ALL cells have a cell membrane. Plants have a cell wall OUTSIDE the membrane.\nWHY: The membrane controls entry/exit for ALL cells. The wall provides extra support for plants.".to_string() }
    fn vocabulary(&self) -> String { "organelle: A structure inside a cell with a specific function | Mitochondria, nucleus, ribosomes.\nosmosis: Movement of water across a semi-permeable membrane from dilute to concentrated | Water leaves cells in salty solutions.\nATP: Adenosine triphosphate — the cell's energy molecule | Made by mitochondria.".to_string() }
    fn flashcard(&self) -> String { "Function of the cell membrane? | Controls what enters and exits the cell (selectively permeable)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Compare active transport and osmosis.\nOsmosis: passive (no energy), water only, high→low concentration. Active transport: requires ATP, any substance, low→high concentration.\n3\nThink about energy, direction, and what moves.".to_string() }
}
