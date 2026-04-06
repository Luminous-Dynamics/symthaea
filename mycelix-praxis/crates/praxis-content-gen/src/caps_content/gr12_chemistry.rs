// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grade 12 Chemistry topic content.

use super::TopicContent;

pub(crate) struct OrganicChemTopic;
impl TopicContent for OrganicChemTopic {
    fn explanation(&self) -> String { "Organic chemistry studies carbon compounds. Key functional groups: alkanes (C−C), alkenes (C═C), \
         alcohols (−OH), carboxylic acids (−COOH), esters (−COO−), amines (−NH₂), alkyl halides (−X). \
         IUPAC naming: longest carbon chain + position of functional group + suffix. \
         Key reactions: substitution (haloalkanes), elimination, addition (alkenes), esterification, combustion.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Name CH₃CH₂CH(OH)CH₃ using IUPAC nomenclature.\n\
         Step 1: Longest chain with −OH group = 4 carbons → butane backbone -> find longest chain\n\
         Step 2: −OH on carbon 2 (number from the end closest to OH) -> locate functional group\n\
         Step 3: Suffix: −ol (alcohol) -> identify suffix\n\
         Answer: Butan-2-ol".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "Draw the structural formula and name the organic product when ethanol reacts with ethanoic acid in the presence of H₂SO₄.\n\
         Ethyl ethanoate (CH₃COOCH₂CH₃)\n\
         Esterification: alcohol + carboxylic acid → ester + water. CH₃COOH + CH₃CH₂OH → CH₃COOCH₂CH₃ + H₂O.\n\
         Ethanol,Ethanoic anhydride,Diethyl ether\n\
         What type of reaction occurs between an alcohol and a carboxylic acid?,The product is an ester. Name it as: acid part (−yl) + alcohol part (−anoate).".to_string() }
    fn hint(&self, _level: u8) -> String { "For naming: find the longest chain, number from the end nearest the functional group, add the correct suffix.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students number the carbon chain from left to right regardless of functional group position.\n\
         RIGHT: Number from the end that gives the lowest locant to the functional group (or first point of difference).\n\
         WHY: IUPAC rules require the lowest possible numbers for substituents and functional groups.".to_string() }
    fn vocabulary(&self) -> String { "functional group: The atom or group of atoms that determines a molecule's chemical properties | The −OH group makes a molecule an alcohol.\nesterification: The reaction between an alcohol and a carboxylic acid to form an ester and water | Esterification requires an acid catalyst.\nisomer: Molecules with the same molecular formula but different structural arrangements | Butan-1-ol and butan-2-ol are positional isomers.".to_string() }
    fn flashcard(&self) -> String { "What is the product of an addition reaction between ethene and HBr? | Bromoethane (CH₃CH₂Br). The H and Br add across the double bond.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Compare addition and substitution reactions. Give an example of each using a named organic compound.\n\
         Addition: C₂H₄ + Br₂ → C₂H₄Br₂ (1,2-dibromoethane). Substitution: CH₃Br + NaOH → CH₃OH + NaBr.\n4\nWhich type of compound undergoes each reaction?".to_string() }
}

pub(crate) struct ReactionRateTopic;
impl TopicContent for ReactionRateTopic {
    fn explanation(&self) -> String { "Reaction rate measures how fast reactants are consumed or products formed. \
         Rate = −Δ[reactant]/Δt = +Δ[product]/Δt. Factors: concentration (more collisions), \
         temperature (faster molecules, more exceed Ea), surface area (more exposed particles), \
         catalysts (lower Ea, provide alternative pathway). Collision theory: reactions only occur \
         when particles collide with sufficient energy (≥ Ea) and correct orientation.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "The concentration of H₂O₂ decreases from 0.80 mol/dm³ to 0.60 mol/dm³ in 20 seconds. Calculate the average rate.\n\
         Step 1: Δ[H₂O₂] = 0.60 − 0.80 = −0.20 mol/dm³ -> change in concentration\n\
         Step 2: Rate = −Δ[H₂O₂]/Δt = −(−0.20)/20 = 0.01 mol/dm³/s -> apply formula\n\
         Answer: Rate = 0.01 mol·dm⁻³·s⁻¹".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "Explain why a lump of zinc reacts slower with HCl than zinc powder.\n\
         Zinc powder has a greater surface area, exposing more zinc atoms for collision with HCl molecules.\n\
         Surface area affects the frequency of effective collisions.\n\
         Zinc powder is less reactive,The lump has more atoms,Temperature is different\n\
         Think about how many particles are exposed to the acid.,Which form has a greater surface area to volume ratio?".to_string() }
    fn hint(&self, _level: u8) -> String { "Reaction rate depends on the frequency of effective collisions. What factors increase collision frequency?".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think catalysts are 'used up' in the reaction.\n\
         RIGHT: Catalysts provide an alternative pathway with lower activation energy but are regenerated — they are not consumed.\n\
         WHY: The catalyst participates in an intermediate step but returns to its original form by the end.".to_string() }
    fn vocabulary(&self) -> String { "activation energy: The minimum energy needed for a reaction to occur | Catalysts lower the activation energy.\ncollision theory: Reactions occur when particles collide with enough energy and correct orientation | Not all collisions lead to reactions.\ncatalyst: A substance that speeds up a reaction without being consumed | MnO₂ catalyses the decomposition of H₂O₂.".to_string() }
    fn flashcard(&self) -> String { "How does increasing temperature affect reaction rate? | Higher temperature → particles move faster → more collisions with energy ≥ Ea → faster rate.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Sketch and label energy profiles for a reaction with and without a catalyst.\n\
         Both curves start and end at the same energy levels. The catalysed curve has a lower peak (lower Ea).\n4\nRemember: a catalyst changes the pathway, not the energy of reactants or products.".to_string() }
}

pub(crate) struct EquilibriumTopic;
impl TopicContent for EquilibriumTopic {
    fn explanation(&self) -> String { "Dynamic equilibrium occurs in a closed system when the rate of the forward reaction equals the rate of the reverse reaction. \
         Le Chatelier's principle: if a system at equilibrium is disturbed, it shifts to counteract the disturbance. \
         Kc = [products]ᵖ/[reactants]ʳ at equilibrium. A large Kc means products are favoured. \
         A catalyst speeds up both directions equally — it does NOT shift equilibrium.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "For N₂(g) + 3H₂(g) ⇌ 2NH₃(g) (ΔH < 0), predict the effect of increasing temperature.\n\
         Step 1: The forward reaction is exothermic (ΔH < 0) -> identify reaction type\n\
         Step 2: Increasing temperature favours the endothermic direction (Le Chatelier) -> apply principle\n\
         Step 3: Equilibrium shifts LEFT (reverse), producing more N₂ and H₂ -> determine shift\n\
         Step 4: Kc decreases (less product, more reactant at new equilibrium) -> effect on Kc\n\
         Answer: Equilibrium shifts left. [NH₃] decreases. Kc decreases.".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "For 2SO₂(g) + O₂(g) ⇌ 2SO₃(g), at equilibrium [SO₂] = 0.2, [O₂] = 0.1, [SO₃] = 0.6. Calculate Kc.\n\
         Kc = 90\n\
         Kc = [SO₃]²/([SO₂]²[O₂]) = (0.6)²/((0.2)²(0.1)) = 0.36/(0.04 × 0.1) = 0.36/0.004 = 90.\n\
         9,900,0.9\n\
         Write the Kc expression from the balanced equation.,Remember: coefficients become exponents.".to_string() }
    fn hint(&self, _level: u8) -> String { "For Le Chatelier: the system shifts to oppose the change. Add reactant → shifts right. Increase temp on exothermic → shifts left.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think adding a catalyst shifts equilibrium to the right.\n\
         RIGHT: A catalyst speeds up both forward and reverse reactions equally. It does NOT change the position of equilibrium or Kc.\n\
         WHY: It helps the system reach equilibrium faster, but doesn't change where equilibrium lies.".to_string() }
    fn vocabulary(&self) -> String { "dynamic equilibrium: Macroscopic properties are constant, but both forward and reverse reactions continue at equal rates | At equilibrium, concentrations don't change.\nLe Chatelier's principle: A system at equilibrium will shift to counteract any imposed change | Adding more reactant shifts equilibrium to the right.".to_string() }
    fn flashcard(&self) -> String { "Does a catalyst change the value of Kc? | No. A catalyst speeds up both forward and reverse reactions equally. It does not affect Kc or equilibrium position.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Explain why the Haber process uses high pressure (200 atm) but moderate temperature (450°C) despite the reaction being exothermic.\n\
         High pressure shifts equilibrium right (fewer gas moles on product side). Low temperature would favour NH₃ but make the reaction too slow. 450°C is a compromise between yield and rate.\n5\nConsider both equilibrium and rate.".to_string() }
}

pub(crate) struct AcidBaseTopic;
impl TopicContent for AcidBaseTopic {
    fn explanation(&self) -> String { "Brønsted-Lowry: acids donate H⁺, bases accept H⁺. Strong acids (HCl, HNO₃, H₂SO₄) ionise completely; \
         weak acids (CH₃COOH) ionise partially. pH = −log[H₃O⁺]. For water: Kw = [H₃O⁺][OH⁻] = 10⁻¹⁴ at 25°C. \
         Titration curves show pH change as base is added to acid. The equivalence point is where moles acid = moles base.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Calculate the pH of a 0.01 mol/dm³ HCl solution.\n\
         Step 1: HCl is a strong acid → fully ionised → [H₃O⁺] = 0.01 mol/dm³ -> identify acid type\n\
         Step 2: pH = −log[H₃O⁺] = −log(0.01) = −log(10⁻²) -> apply formula\n\
         Step 3: pH = 2 -> calculate\n\
         Answer: pH = 2".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "If [OH⁻] = 2 × 10⁻³ mol/dm³, find the pH at 25°C.\n\
         pH = 11.3\n\
         [H₃O⁺] = Kw/[OH⁻] = 10⁻¹⁴/(2×10⁻³) = 5×10⁻¹². pH = −log(5×10⁻¹²) = 11.3.\n\
         3.0,2.7,8.7\n\
         Find [H₃O⁺] first using Kw = [H₃O⁺][OH⁻] = 10⁻¹⁴.,Then use pH = −log[H₃O⁺].".to_string() }
    fn hint(&self, _level: u8) -> String { "Remember: pH + pOH = 14 at 25°C. If you know [OH⁻], find pOH = −log[OH⁻], then pH = 14 − pOH.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think a weak acid has a high pH.\n\
         RIGHT: A weak acid can still have a low pH if concentrated enough. 'Weak' refers to degree of ionisation, not pH.\n\
         WHY: Concentrated CH₃COOH (weak) can have a lower pH than dilute HCl (strong).".to_string() }
    fn vocabulary(&self) -> String { "conjugate acid-base pair: Two species that differ by one proton (H⁺) | NH₃ and NH₄⁺ are a conjugate pair.\nequivalence point: The point in a titration where moles of acid = moles of base | At the equivalence point, the indicator changes colour.\nhydrolysis: The reaction of a salt with water to produce acidic or basic solution | NaCH₃COO is basic by hydrolysis (weak acid salt).".to_string() }
    fn flashcard(&self) -> String { "What is the difference between a strong and weak acid? | Strong acid: fully ionised (e.g., HCl → H⁺ + Cl⁻). Weak acid: partially ionised (e.g., CH₃COOH ⇌ H⁺ + CH₃COO⁻).".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Sketch the titration curve for adding NaOH to CH₃COOH. Label the equivalence point and state which indicator to use.\n\
         S-shaped curve starting ~pH 3, equivalence point above pH 7 (~pH 8.7). Use phenolphthalein (range 8.2-10).\n5\nWeak acid + strong base → equivalence point above 7.".to_string() }
}

pub(crate) struct ElectrochemTopic;
impl TopicContent for ElectrochemTopic {
    fn explanation(&self) -> String { "Galvanic cells convert chemical energy to electrical energy using spontaneous redox reactions. \
         The more reactive metal is the anode (oxidation), the less reactive is the cathode (reduction). \
         E°cell = E°cathode − E°anode (must be positive for spontaneous). \
         Electrolytic cells use external voltage to drive non-spontaneous reactions. \
         In electrolysis: anode = oxidation (+), cathode = reduction (−) — opposite to galvanic.".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "Calculate E°cell for Zn|Zn²⁺||Cu²⁺|Cu. Given: E°(Zn²⁺/Zn) = −0.76 V, E°(Cu²⁺/Cu) = +0.34 V.\n\
         Step 1: Zn is more reactive → anode (oxidation). Cu²⁺ → Cu at cathode (reduction) -> identify electrodes\n\
         Step 2: E°cell = E°cathode − E°anode = 0.34 − (−0.76) -> apply formula\n\
         Step 3: E°cell = 0.34 + 0.76 = 1.10 V -> calculate\n\
         Answer: E°cell = 1.10 V (spontaneous, since E° > 0)".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "In the electrolysis of CuSO₄ solution with copper electrodes, what happens at each electrode?\n\
         Anode: Cu → Cu²⁺ + 2e⁻ (copper dissolves). Cathode: Cu²⁺ + 2e⁻ → Cu (copper deposits).\n\
         Copper transfers from anode to cathode. Mass of anode decreases, mass of cathode increases.\n\
         Both electrodes dissolve,Oxygen produced at cathode,Nothing happens\n\
         What are the ions present in solution?,At which electrode does oxidation occur?".to_string() }
    fn hint(&self, _level: u8) -> String { "Remember: AN OX, RED CAT — Anode = Oxidation, Reduction = Cathode. In galvanic cells, anode is negative; in electrolytic, anode is positive.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think the anode is always negative.\n\
         RIGHT: In a galvanic cell, the anode IS negative. But in an electrolytic cell, the anode is POSITIVE (connected to + terminal of battery).\n\
         WHY: The signs are different because the energy source is different: chemical (galvanic) vs external battery (electrolytic).".to_string() }
    fn vocabulary(&self) -> String { "anode: The electrode where oxidation occurs | Zn → Zn²⁺ + 2e⁻ at the anode.\ncathode: The electrode where reduction occurs | Cu²⁺ + 2e⁻ → Cu at the cathode.\nsalt bridge: Allows ion flow to maintain electrical neutrality | Without a salt bridge, the cell would stop working.\nstandard electrode potential: The voltage of a half-cell measured against the standard hydrogen electrode (SHE) | E°(Cu²⁺/Cu) = +0.34 V.".to_string() }
    fn flashcard(&self) -> String { "How do you calculate E°cell? | E°cell = E°cathode − E°anode. If E°cell > 0, the reaction is spontaneous.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Compare galvanic and electrolytic cells in terms of: energy conversion, spontaneity, and anode polarity.\n\
         Galvanic: chemical→electrical, spontaneous, anode negative. Electrolytic: electrical→chemical, non-spontaneous, anode positive.\n4\nThink about where the energy comes from in each case.".to_string() }
}

pub(crate) struct FertiliserTopic;
impl TopicContent for FertiliserTopic {
    fn explanation(&self) -> String { "Plants need N, P, and K for growth. The fertiliser industry produces these from chemical processes: \
         Haber process: N₂ + 3H₂ ⇌ 2NH₃ (high pressure, moderate temperature, iron catalyst). \
         Ostwald process: converts NH₃ → NO → NO₂ → HNO₃. \
         Contact process: S + O₂ → SO₂ → SO₃ → H₂SO₄. \
         Environmental concerns: eutrophication (excess nutrients → algal blooms → oxygen depletion).".to_string() }
    fn worked_example(&self, _variant: usize) -> String { "A fertiliser bag is labelled NPK 3:2:1 (28). If the bag weighs 50 kg, how much nitrogen does it contain?\n\
         Step 1: Total ratio = 3+2+1 = 6 parts -> sum ratio parts\n\
         Step 2: The (28) means 28% of the bag is active nutrients -> identify nutrient percentage\n\
         Step 3: N fraction = 3/6 = ½ of the nutrients -> nitrogen fraction\n\
         Step 4: N mass = 50 × 0.28 × 0.5 = 7 kg -> calculate\n\
         Answer: 7 kg nitrogen".to_string() }
    fn practice_problem(&self, _difficulty: u16) -> String { "Explain why eutrophication occurs when excess fertiliser runs off into rivers.\n\
         Excess N and P stimulate algal growth → algae die → decomposition by bacteria uses dissolved O₂ → aquatic organisms suffocate.\n\
         This process is called eutrophication and leads to 'dead zones'.\n\
         Fish eat the algae,The water becomes too acidic,Fertiliser is toxic to fish\n\
         What do algae need to grow?,What happens when large amounts of organic matter decompose in water?".to_string() }
    fn hint(&self, _level: u8) -> String { "Follow the chain: nutrients → algal bloom → decomposition → oxygen depletion → fish death.".to_string() }
    fn misconception(&self) -> String { "WRONG: Students think fertiliser directly kills fish.\n\
         RIGHT: Fertiliser feeds algae. When algae die and decompose, bacteria consume dissolved oxygen. Fish die from lack of oxygen, not from the fertiliser itself.\n\
         WHY: It's an indirect chain of events — the fertiliser triggers a cascade.".to_string() }
    fn vocabulary(&self) -> String { "eutrophication: The enrichment of water with nutrients, causing excessive plant growth and oxygen depletion | Agricultural runoff causes eutrophication.\nHaber process: Industrial synthesis of ammonia from N₂ and H₂ | The Haber process operates at 450°C, 200 atm with an iron catalyst.".to_string() }
    fn flashcard(&self) -> String { "What are the conditions for the Haber process? | N₂ + 3H₂ ⇌ 2NH₃. Conditions: 450°C, 200 atm, iron catalyst.".to_string() }
    fn assessment_item(&self, _context: &str) -> String { "Discuss the environmental impact of excessive fertiliser use and suggest two ways to reduce it.\n\
         Eutrophication leads to algal blooms and oxygen depletion. Mitigation: use slow-release fertilisers, apply precise amounts based on soil testing.\n4\nThink about both the problem and practical solutions.".to_string() }
}

