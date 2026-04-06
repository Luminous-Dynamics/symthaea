// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cambridge IGCSE — Mathematics, Biology, Chemistry, Physics.

use crate::caps_content::TopicContent;

pub struct IgcseMathAlgebra;
impl TopicContent for IgcseMathAlgebra {
    fn explanation(&self) -> String { "IGCSE algebra: simultaneous equations (two unknowns, two equations), quadratics (ax²+bx+c=0), inequalities. Quadratic formula: x=(-b±√(b²-4ac))/2a.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "Solve: 2x+y=7, x-y=2\nAdd: 3x=9→x=3. y=7-6=1.".to_string(), 1 => "x²+3x-10=0 → (x+5)(x-2)=0 → x=-5,2.".to_string(), _ => "2x²-5x+1=0: x=(5±√17)/4 ≈ 2.28, 0.22.".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "3x+4=19\nx=5\n3x=15→x=5\n3,15,23\n19-4=15, 15÷3.\n15÷3=5.".to_string(), _ => "Sum=20, product=96. Find the numbers.\n8,12\nx²-20x+96=0→(x-8)(x-12)=0.\n6,14\nForm quadratic.\nx(20-x)=96.".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "Elimination: add/subtract to remove a variable.".to_string(), _ => "Discriminant: b²-4ac determines root count.".to_string() } }
    fn misconception(&self) -> String { "WRONG: x²=9 means x=3.\nRIGHT: x=±3. Don't forget the negative!\nWHY: (-3)²=9 too.".to_string() }
    fn vocabulary(&self) -> String { "discriminant: b²-4ac | >0: 2 roots, =0: 1 root, <0: no real roots.".to_string() }
    fn flashcard(&self) -> String { "Discriminant of x²+2x+5? | 4-20=-16 (no real roots)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Prove x²+4x+5>0 for all x.\n(x+2)²+1≥1>0.\n3\nComplete the square.".to_string() }
}

pub struct IgcseBiology;
impl TopicContent for IgcseBiology {
    fn explanation(&self) -> String { "Cambridge IGCSE Biology: Cells, enzymes, nutrition, transport, respiration, reproduction, genetics, ecology. Cells: plant (wall+chloroplasts+vacuole) vs animal. Active transport uses ATP to move against concentration gradient.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "Explain osmosis in a potato chip placed in salt solution.\nWater moves from potato (high water conc.) to solution (low water conc.) through cell membrane.\nPotato loses water → becomes flaccid/soft.".to_string(), _ => "Food test for starch: add iodine solution.\nStarch present → turns blue-black.\nNo starch → stays yellow-brown.".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "Name the organelle for photosynthesis.\nChloroplast\nContains chlorophyll.\nMitochondria\nGreen pigment captures light.\nOnly in plant cells.".to_string(), _ => "Red blood cells in pure water. What happens?\nThey swell and burst (lysis/haemolysis).\nWater enters by osmosis (water conc. higher outside).\nShrink, Nothing\nWhich direction does water move?\nFrom high water conc. (pure water) to low (inside cell).".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "Osmosis: water moves from dilute to concentrated.".to_string(), _ => "Diffusion: particles move from high to low concentration.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Plants only respire at night.\nRIGHT: Plants respire 24/7. During the day, photosynthesis rate exceeds respiration.\nWHY: Both processes happen simultaneously in light.".to_string() }
    fn vocabulary(&self) -> String { "osmosis: Water movement through a semi-permeable membrane.\ndiffusion: Net movement of particles from high to low concentration.".to_string() }
    fn flashcard(&self) -> String { "What is the test for glucose? | Benedict's solution: heat → orange/red precipitate if glucose present".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Design an experiment to show osmosis using visking tubing.\nFill tubing with sugar solution, place in water, measure mass change over time. Sugar solution gains water → mass increases.\n3\nControl: same tubing in sugar solution (no gradient).".to_string() }
}

pub struct IgcsePhysics;
impl TopicContent for IgcsePhysics {
    fn explanation(&self) -> String { "Cambridge IGCSE Physics: Forces & motion, energy, waves, electricity, magnetism, nuclear physics. Speed=distance/time. Acceleration=(v-u)/t. Weight=mass×gravity. Ohm's Law: V=IR.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "A 1200Ω resistor carries 0.05A. Voltage?\nV=IR=1200×0.05=60V".to_string(), _ => "Car 0→30m/s in 10s. Acceleration?\na=(30-0)/10=3 m/s²".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "Speed of 100m in 12.5s?\n8 m/s\n100÷12.5=8\n12.5, 112.5, 1250\nSpeed=distance÷time.\n100/12.5.".to_string(), _ => "Two resistors 6Ω and 3Ω in parallel. Total?\n2Ω\n1/R=1/6+1/3=1/6+2/6=3/6=1/2. R=2Ω.\n9, 4.5, 18\n1/R_total = 1/R₁ + 1/R₂.\nFind common denominator.".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "V=IR, P=IV, E=Pt.".to_string(), _ => "Parallel: 1/R=1/R₁+1/R₂. Series: R=R₁+R₂.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Current is 'used up' in a circuit.\nRIGHT: Current is the SAME everywhere in a series circuit. Energy is transferred, not current.\nWHY: Electrons flow in a loop — they're not consumed.".to_string() }
    fn vocabulary(&self) -> String { "current: Flow of charge (amperes) | I=Q/t.\nvoltage: Energy per unit charge (volts) | V=W/Q.\nresistance: Opposition to current flow (ohms) | R=V/I.".to_string() }
    fn flashcard(&self) -> String { "Ohm's Law? | V = I × R (voltage = current × resistance)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A kettle (2kW) boils water in 3 minutes. Energy used?\n360 kJ\n3\nE=Pt=2000×180=360000J=360kJ.".to_string() }
}
