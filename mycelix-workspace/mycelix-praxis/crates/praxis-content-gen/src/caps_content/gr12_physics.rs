// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 12 Physics topic content.

use super::TopicContent;

pub(crate) struct MechanicsTopic;
impl TopicContent for MechanicsTopic {
    fn explanation(&self) -> String { "Grade 12 mechanics covers momentum, impulse, and work-energy-power. \
         Momentum p = mv is conserved in isolated systems: Σpᵢ = Σpf. \
         Impulse FΔt = Δp = m(vf − vi). The work-energy theorem: Wnet = ΔEk. \
         Conservation of energy with friction: Wnc = ΔEk + ΔEp. Power P = W/Δt = Fv.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "A 2 kg ball moving at 6 m/s collides with a stationary 4 kg ball. After collision they move together. Find their velocity.\n\
         Step 1: Before: p = m₁v₁ + m₂v₂ = 2(6) + 4(0) = 12 kg·m/s -> total initial momentum\n\
         Step 2: After: p = (m₁ + m₂)vf = 6vf (they stick together) -> combined mass × final velocity\n\
         Step 3: Conservation: 12 = 6vf → vf = 2 m/s -> apply conservation of momentum\n\
         Answer: Combined velocity = 2 m/s in the original direction".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "A 1500 kg car accelerates from rest to 20 m/s in 10 s. Calculate the average power.\n\
         30 000 W (30 kW)\n\
         Ek = ½mv² = ½(1500)(400) = 300 000 J. P = W/t = 300 000/10 = 30 000 W.\n\
         15 000 W,3000 W,300 000 W\n\
         Find the kinetic energy gained first.,Power = work done ÷ time taken.".to_string() }
    fn hint(&self, _level: u8) -> String { "Draw a diagram showing before and after. List known quantities. Choose the right principle: momentum conservation or work-energy theorem.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think momentum is always conserved.\n\
         RIGHT: Momentum is only conserved in an isolated system (no external net force). Kinetic energy is only conserved in elastic collisions.\n\
         WHY: External forces like friction change the total momentum of the system.".to_string() }
    fn vocabulary(&self) -> String { "impulse: The product FΔt, equal to the change in momentum | A longer collision time reduces the force (e.g., airbags).\nelastic collision: A collision where kinetic energy is conserved | Billiard ball collisions are approximately elastic.\ninelastic collision: A collision where kinetic energy is NOT conserved (but momentum is) | A car crash is inelastic — energy goes to deformation.".to_string() }
    fn flashcard(&self) -> String { "State the impulse-momentum theorem. | FnetΔt = Δp = m(vf − vi). The net impulse equals the change in momentum.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Explain why a cricket player pulls their hands back when catching a fast ball.\n\
         Pulling hands back increases Δt, reducing the force (FΔt = Δp, same Δp but larger Δt means smaller F).\n3\nThink about impulse.".to_string() }
}

pub(crate) struct DopplerTopic;
impl TopicContent for DopplerTopic {
    fn explanation(&self) -> String { "The Doppler effect is the change in observed frequency when a source and listener move relative to each other. \
         Formula: fL = fs × (v ± vL)/(v ± vs), where v = speed of sound, vL = listener speed, vs = source speed. \
         Convention: use + in numerator when listener moves toward source; + in denominator when source moves away.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "An ambulance siren emits sound at 500 Hz and approaches at 30 m/s. Speed of sound = 340 m/s. Find the frequency heard.\n\
         Step 1: Source approaches → use minus in denominator -> identify relative motion\n\
         Step 2: fL = 500 × 340/(340 − 30) = 500 × 340/310 -> substitute\n\
         Step 3: fL = 548.4 Hz -> calculate\n\
         Answer: The observer hears 548.4 Hz (higher pitch as ambulance approaches)".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "A car horn (frequency 400 Hz) moves away from you at 25 m/s. Speed of sound = 340 m/s. What frequency do you hear?\n\
         372.6 Hz\n\
         Source moving away: fL = 400 × 340/(340 + 25) = 400 × 340/365 ≈ 372.6 Hz.\n\
         400 Hz,430.8 Hz,350 Hz\n\
         Is the source moving toward or away from you?,When source moves away, the denominator has a + sign.".to_string() }
    fn hint(&self, _level: u8) -> String { "Remember: approaching → higher frequency. Receding → lower frequency. Use the sign convention carefully.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think the Doppler effect changes the actual frequency emitted by the source.\n\
         RIGHT: Only the observed frequency changes. The source continues emitting at its original frequency.\n\
         WHY: The wave crests are compressed or stretched in the medium, but the source vibrates at the same rate.".to_string() }
    fn vocabulary(&self) -> String { "Doppler effect: The apparent change in frequency due to relative motion | An approaching siren sounds higher-pitched.\nred shift: The decrease in frequency of light from objects moving away | Distant galaxies show red shift, indicating the universe is expanding.".to_string() }
    fn flashcard(&self) -> String { "Write the Doppler effect formula for sound. | fL = fs(v ± vL)/(v ± vs). Top sign: listener toward source (+) or away (−). Bottom sign: source toward listener (−) or away (+).".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "A bat emits ultrasound at 50 kHz and flies at 10 m/s toward a wall. Speed of sound = 340 m/s. Find the frequency of the echo.\n\
         fecho ≈ 53.03 kHz (apply Doppler twice: bat→wall, then wall→bat)\n5\nThe wall acts as a stationary 'listener' first, then a 'source' of the reflected wave.".to_string() }
}

pub(crate) struct ElectricityTopic;
impl TopicContent for ElectricityTopic {
    fn explanation(&self) -> String { "Grade 12 electricity covers internal resistance and electrodynamics. \
         EMF (ε) = I(R + r), where r is internal resistance. The 'lost volts' = Ir. \
         Terminal voltage = ε − Ir. For electrodynamics: Faraday's law ε = −NΔΦ/Δt \
         (changing magnetic flux induces an EMF). AC: Vrms = Vmax/√2, Pavg = Vrms × Irms.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "A battery with ε = 12 V and r = 0.5 Ω is connected to a 5.5 Ω resistor. Find the current and terminal voltage.\n\
         Step 1: ε = I(R + r) → 12 = I(5.5 + 0.5) = 6I -> apply formula\n\
         Step 2: I = 12/6 = 2 A -> solve for current\n\
         Step 3: V_terminal = ε − Ir = 12 − 2(0.5) = 11 V -> or V = IR = 2(5.5) = 11 V\n\
         Answer: I = 2 A, terminal voltage = 11 V".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "When a 3 Ω resistor is connected to a battery, the current is 2 A. When a 7 Ω resistor is used, the current is 1 A. Find ε and r.\n\
         ε = 8 V, r = 1 Ω\n\
         ε = I₁(R₁ + r) = 2(3 + r) and ε = I₂(R₂ + r) = 1(7 + r). So 6 + 2r = 7 + r → r = 1 Ω. ε = 2(4) = 8 V.\n\
         ε = 6 V and r = 0,ε = 10 V and r = 2 Ω,ε = 7 V and r = 0.5 Ω\n\
         Write two equations using ε = I(R + r) for each scenario.,Set the two expressions for ε equal and solve for r.".to_string() }
    fn hint(&self, _level: u8) -> String { "Use ε = I(R + r). With two different setups, you get two equations with two unknowns (ε and r).".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think the terminal voltage equals the EMF.\n\
         RIGHT: Terminal voltage = ε − Ir. It's less than the EMF when current flows due to energy lost in the battery.\n\
         WHY: Internal resistance causes a voltage drop inside the battery. Only when I = 0 does V_terminal = ε.".to_string() }
    fn vocabulary(&self) -> String { "EMF: The maximum potential difference a battery can provide (when no current flows) | ε = 12 V for this battery.\ninternal resistance: The resistance within the battery itself | Internal resistance causes the terminal voltage to drop under load.".to_string() }
    fn flashcard(&self) -> String { "What is the relationship between EMF, terminal voltage, and internal resistance? | ε = V_terminal + Ir, or equivalently ε = I(R + r)".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Explain why the headlights of a car dim slightly when the starter motor is engaged.\n\
         The starter motor draws a large current, increasing the Ir drop across internal resistance, reducing terminal voltage.\n3\nThink about internal resistance and the EMF equation.".to_string() }
}

pub(crate) struct ModernPhysicsTopic;
impl TopicContent for ModernPhysicsTopic {
    fn explanation(&self) -> String { "The photoelectric effect demonstrates the particle nature of light. \
         Photon energy E = hf = hc/λ (h = 6.63 × 10⁻³⁴ J·s). \
         Einstein's equation: E = W₀ + Ek(max), where W₀ = hf₀ is the work function. \
         Below threshold frequency f₀, no electrons are emitted regardless of intensity. \
         Emission spectra: atoms emit photons at specific frequencies when electrons drop energy levels: E = hf.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Light of wavelength 400 nm strikes a metal with work function 3.0 eV. Find the maximum kinetic energy of ejected electrons.\n\
         Step 1: E = hc/λ = (6.63×10⁻³⁴)(3×10⁸)/(400×10⁻⁹) = 4.97 × 10⁻¹⁹ J -> photon energy\n\
         Step 2: Convert to eV: 4.97×10⁻¹⁹/1.6×10⁻¹⁹ = 3.11 eV -> convert units\n\
         Step 3: Ek(max) = E − W₀ = 3.11 − 3.0 = 0.11 eV -> apply Einstein's equation\n\
         Answer: Ek(max) = 0.11 eV (≈ 1.76 × 10⁻²⁰ J)".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "The threshold frequency of sodium is 5.6 × 10¹⁴ Hz. Calculate the work function in eV.\n\
         2.32 eV\n\
         W₀ = hf₀ = (6.63×10⁻³⁴)(5.6×10¹⁴) = 3.71×10⁻¹⁹ J = 3.71×10⁻¹⁹/1.6×10⁻¹⁹ = 2.32 eV.\n\
         5.6 eV,0.37 eV,3.71 eV\n\
         Use W₀ = hf₀.,Remember to convert from joules to eV by dividing by 1.6 × 10⁻¹⁹.".to_string() }
    fn hint(&self, _level: u8) -> String { "Photon energy = hf or hc/λ. If E > W₀, photoelectrons are emitted with Ek = E − W₀.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think increasing light intensity below the threshold frequency will eventually cause emission.\n\
         RIGHT: Below f₀, no electrons are emitted regardless of intensity. Each photon must individually have enough energy (hf ≥ W₀).\n\
         WHY: One photon interacts with one electron. Many low-energy photons can't combine their energy.".to_string() }
    fn vocabulary(&self) -> String { "photoelectric effect: The emission of electrons when light strikes a metal surface | The photoelectric effect proves light behaves as particles.\nwork function: The minimum energy needed to remove an electron from a metal surface | W₀ = hf₀.\nthreshold frequency: The minimum frequency of light needed to cause photoemission | Below f₀, no electrons are emitted.".to_string() }
    fn flashcard(&self) -> String { "State Einstein's photoelectric equation. | E = W₀ + Ek(max), or hf = hf₀ + ½mv²(max)".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Explain why the photoelectric effect cannot be explained by the wave theory of light.\n\
         Wave theory predicts: (1) any frequency should cause emission given enough intensity, (2) delay before emission as energy accumulates. \
         Both are contradicted experimentally.\n4\nThink about what wave theory predicts vs what actually happens.".to_string() }
}

// ---- Chemistry Topics ----
