// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IB International Baccalaureate — Math, Biology, Chemistry, Physics.

use crate::caps_content::TopicContent;

pub struct IbMathFunctions;
impl TopicContent for IbMathFunctions {
    fn explanation(&self) -> String { "A function maps each input to exactly one output: f(x) = 2x + 3. Domain: valid inputs. Range: possible outputs. Key types: linear, quadratic, exponential. Transformations: f(x)+k shifts up, f(x-h) shifts right.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "f(x)=x²-4x+3. Vertex: complete square → (x-2)²-1. Vertex: (2,-1).".to_string(), 1 => "Inverse of f(x)=3x-7: swap x,y → x=3y-7 → f⁻¹(x)=(x+7)/3.".to_string(), _ => "g(x)=f(x-3)+2: shift right 3, up 2.".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "f(x)=2x+1, find f(3).\n7\n2(3)+1=7\n5,9,6\nSubstitute x=3.\n2×3+1.".to_string(), 301..=600 => "Zeros of x²-5x+6?\nx=2,3\n(x-2)(x-3)=0\n-2,-3\nFactor.\nProduct=6, sum=5.".to_string(), _ => "P(t)=500·1.03ᵗ. When doubles?\n≈23.4 yrs\nt=ln2/ln1.03\n33,50\nSet P=1000, solve.\nln(2)/ln(1.03).".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "f(a): replace x with a.".to_string(), _ => "Zeros: set f(x)=0.".to_string() } }
    fn misconception(&self) -> String { "WRONG: f(x+3) shifts RIGHT.\nRIGHT: f(x+3) shifts LEFT. f(x-3) shifts RIGHT.\nWHY: f(x+3)=0 when x=-3 (zero moved left).".to_string() }
    fn vocabulary(&self) -> String { "domain: valid inputs | f(x)=1/x: x≠0.\nrange: all outputs | f(x)=x²: [0,∞).".to_string() }
    fn flashcard(&self) -> String { "Domain of √x? | x ≥ 0".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Sketch |x-2|-1. Vertex and zeros?\nVertex (2,-1). Zeros x=1,3.\n3\n|x-2|=1→x=1,3.".to_string() }
}

pub struct IbBiology;
impl TopicContent for IbBiology {
    fn explanation(&self) -> String { "IB Biology Topic 2: Molecular Biology. Water is polar — hydrogen bonds give it special properties: high specific heat (temperature stability), cohesion (surface tension), and it's the universal solvent. Carbohydrates (energy), lipids (membranes/storage), proteins (enzymes/structure), nucleic acids (DNA/RNA).".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "Draw a water molecule showing polarity.\nO is slightly negative (δ-), H atoms slightly positive (δ+).\nBent shape (~104.5°). Hydrogen bonds form between molecules.".to_string(), 1 => "Compare condensation and hydrolysis.\nCondensation: two monomers join, releasing H₂O (builds polymers).\nHydrolysis: water breaks a bond (breaks polymers into monomers).\nThey're reverse reactions.".to_string(), _ => "Enzyme lock-and-key model.\nSubstrate fits into active site like a key in a lock.\nEnzyme-substrate complex forms → products released.\nEnzyme unchanged → can be reused.".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "Name 4 biological macromolecules.\nCarbohydrates, Lipids, Proteins, Nucleic acids\nBuilding blocks of life.\nVitamins, Minerals\nThink CLPN.\nC=energy, L=membranes, P=enzymes, N=DNA.".to_string(), _ => "Why does increasing temperature beyond 40°C reduce enzyme activity?\nDenaturation: heat breaks hydrogen bonds maintaining the active site shape. Substrate no longer fits.\nMore heat=faster reaction always\nWhat happens to protein shape when heated?\nEgg whites turn solid when cooked — same principle.".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "Think about molecular structure.".to_string(), _ => "Enzymes: shape determines function.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Enzymes are 'used up' in reactions.\nRIGHT: Enzymes are catalysts — they speed up reactions without being consumed. One enzyme can catalyse millions of reactions.\nWHY: The enzyme's shape is unchanged after each reaction cycle.".to_string() }
    fn vocabulary(&self) -> String { "enzyme: Biological catalyst (protein) | Amylase breaks down starch.\ndenaturation: Loss of protein shape due to heat/pH | Cooked egg whites.".to_string() }
    fn flashcard(&self) -> String { "What type of bond holds water molecules together? | Hydrogen bonds (between O and H of neighbouring molecules)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Explain why water is essential for life. Give 3 properties.\nHigh specific heat (temperature regulation), cohesion (transport in plants), solvent (chemical reactions occur in solution).\n3\nLink each property to a biological function.".to_string() }
}

pub struct IbChemistry;
impl TopicContent for IbChemistry {
    fn explanation(&self) -> String { "IB Chemistry Topic 1: Stoichiometry. The mole (6.022×10²³ particles) bridges atoms and grams. Molar mass = atomic mass in g/mol. Balancing equations: same atoms on both sides. Limiting reagent: the reactant that runs out first determines the product amount.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "How many moles in 36g of water (H₂O)?\nMolar mass: 2(1)+16=18 g/mol\nn = mass/M = 36/18 = 2 mol".to_string(), 1 => "Balance: _Fe + _O��� → _Fe₂O₃\n4Fe + 3O₂ → 2Fe₂O₃\nCheck: Fe: 4=4✓, O: 6=6✓".to_string(), _ => "2H₂ + O₂ → 2H₂O. 4 mol H₂ + 2 mol O₂ = ?\nRatio: 2:1. 4 mol H₂ needs 2 mol O₂. Neither is limiting.\nProduct: 4 mol H₂O.".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "Molar mass of CO₂?\n44 g/mol\n12+2(16)=44\n28, 32, 60\nAdd atomic masses.\nC=12, O=16.".to_string(), _ => "10g CaCO₃ decomposes. Mass of CO₂ produced?\n4.4g\nCaCO₃→CaO+CO₂. M(CaCO₃)=100. n=0.1mol. n(CO₂)=0.1mol. m=0.1×44=4.4g.\n10, 44, 2.2\nn=mass/M, use ratio, m=n×M.\n10/100=0.1 mol.".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "n = mass / molar mass.".to_string(), _ => "Balance equations: atoms in = atoms out.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Balancing changes the formulas.\nRIGHT: Only change coefficients (numbers in front), NEVER subscripts.\nWHY: Changing subscripts changes the substance. H₂O₂ is hydrogen peroxide, not water!".to_string() }
    fn vocabulary(&self) -> String { "mole: 6.022×10²³ particles | 1 mol of carbon = 12g = 6.022×10²³ atoms.\nstoichiometry: Quantitative relationships in chemical reactions.".to_string() }
    fn flashcard(&self) -> String { "Avogadro's number? | 6.022 × 10²³ (particles per mole)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "25.0 cm³ of 0.100 mol/dm³ NaOH neutralises HCl. Find HCl concentration if 20.0 cm³ used.\n0.125 mol/dm³\n3\nn(NaOH)=0.025×0.1=0.0025mol. 1:1 ratio. c(HCl)=0.0025/0.020=0.125.".to_string() }
}

pub struct IbPhysics;
impl TopicContent for IbPhysics {
    fn explanation(&self) -> String { "IB Physics Topic 2: Mechanics. Newton's Laws: 1) Inertia (no net force=constant velocity), 2) F=ma (net force=mass×acceleration), 3) Action-reaction (equal and opposite). Kinematics equations: v=u+at, s=ut+½at², v²=u²+2as.".to_string() }
    fn worked_example(&self, i: usize) -> String { match i { 0 => "A 5kg box is pushed with 30N force. Friction=10N. Acceleration?\nF_net=30-10=20N\na=F/m=20/5=4 m/s²".to_string(), 1 => "Ball dropped from 20m. Time to hit ground? (g=10m/s²)\ns=½gt²\n20=½(10)t²\nt²=4\nt=2s".to_string(), _ => "Car brakes from 30m/s to 0 in 5s. Braking distance?\nv²=u²+2as\n0=900+2a(s)\na=(0-30)/5=-6m/s²\ns=900/12=75m".to_string() } }
    fn practice_problem(&self, d: u16) -> String { match d { 0..=300 => "F=ma. m=10kg, a=3m/s². Force?\n30N\n10×3=30\n13, 3.3, 33\nMultiply mass by acceleration.\n10×3=?".to_string(), _ => "Projectile launched at 20m/s at 45°. Maximum height? (g=10)\n10m\nv_y=20sin45°=14.1m/s. h=v²/2g=200/20=10m.\n20, 5, 40\nVertical component only.\nv_y²=2gh → h=v_y²/2g.".to_string() } }
    fn hint(&self, l: u8) -> String { match l { 1 => "F=ma: force, mass, acceleration.".to_string(), _ => "Projectiles: split into horizontal (constant v) and vertical (accelerating).".to_string() } }
    fn misconception(&self) -> String { "WRONG: Heavier objects fall faster.\nRIGHT: In a vacuum, ALL objects fall at the same rate (g≈9.8m/s²). Air resistance causes differences.\nWHY: Galileo proved this. F=mg means a=g regardless of m.".to_string() }
    fn vocabulary(&self) -> String { "inertia: Resistance to change in motion | Newton's 1st Law.\nacceleration: Rate of change of velocity | a=Δv/Δt, units: m/s².".to_string() }
    fn flashcard(&self) -> String { "Newton's 3rd Law? | Every action has an equal and opposite reaction".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "A 60kg person in a lift accelerating up at 2m/s². What does the scale read?\n720N\n3\nF=m(g+a)=60(10+2)=720N. Apparent weight increases in upward acceleration.".to_string() }
}
