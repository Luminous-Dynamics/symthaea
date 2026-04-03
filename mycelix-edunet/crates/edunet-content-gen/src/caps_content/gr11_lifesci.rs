// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS Life Sciences Grade 11 — Biodiversity, Human Systems, Population Ecology.

use super::TopicContent;

pub(crate) struct Gr11Biodiversity;
impl TopicContent for Gr11Biodiversity {
    fn explanation(&self) -> String {
        "Biodiversity means the variety of life on Earth — different species, genes, and ecosystems. \
         South Africa is one of the most biodiverse countries, with the Cape Floral Kingdom (fynbos) \
         being one of only 6 floral kingdoms on Earth. Classification organises life into groups: \
         Domain → Kingdom → Phylum → Class → Order → Family → Genus → Species. \
         Binomial naming: Homo sapiens (Genus + species, always italicised).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "Classify a lion: Panthera leo\nDomain: Eukarya\nKingdom: Animalia\nPhylum: Chordata\nClass: Mammalia\nOrder: Carnivora\nFamily: Felidae\nGenus: Panthera\nSpecies: P. leo".to_string(),
            1 => "Why is biodiversity important?\n1. Ecosystem services: pollination, water purification, oxygen production\n2. Medicine: 25% of drugs come from plants\n3. Food security: genetic diversity protects against crop failure\n4. Cultural value: many species are sacred in SA cultures".to_string(),
            _ => "Human impact on biodiversity:\nHabitat destruction (deforestation, urbanisation)\nPollution (water, air, plastic)\nOverexploitation (overfishing, poaching rhinos)\nClimate change (coral bleaching, shifting habitats)\nInvasive species (eucalyptus, lantana in SA)".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "What does the binomial name Canis lupus tell us?\nGenus: Canis (dog family). Species: lupus (wolf). It's a grey wolf.\nIt's a type of cat, It means 'big dog'\nBinomial = two names: Genus + species.\nThe first word is the genus, the second is the species.".to_string(),
            301..=600 => "Why is the Cape Floral Kingdom globally significant?\nIt's one of only 6 floral kingdoms on Earth, despite being the smallest. It has ~9000 plant species, 70% found NOWHERE else (endemic). Fynbos is adapted to fire and poor soils.\nIt's not significant, Only pretty flowers\nWhat makes it unique compared to other regions?\nEndemic = found only in one place. 70% endemic = irreplaceable.".to_string(),
            _ => "A farmer uses ONE type of maize. A new disease kills all crops. How does biodiversity relate?\nMonoculture (one type) = vulnerability. If that type is susceptible, ALL are lost.\nBiodiversity solution: grow MULTIPLE varieties. Different genes = different disease resistance.\nIrrelevant to farming\nWhat if some varieties are naturally resistant to the disease?\nGenetic diversity = insurance. More variety = more chance of survival.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "Remember: King Philip Came Over For Good Spaghetti (Kingdom→Species).".to_string(), 2 => "Biodiversity exists at 3 levels: species, genetic, ecosystem.".to_string(), _ => "Endemic species are found NOWHERE else — extinction is permanent.".to_string() } }
    fn misconception(&self) -> String { "WRONG: Evolution means 'survival of the fittest' = strongest.\nRIGHT: 'Fittest' means best adapted to the ENVIRONMENT, not strongest. A small, fast mouse is 'fitter' than a strong, slow one if predators are the main threat.\nWHY: Fitness is about reproduction success in a specific environment.".to_string() }
    fn vocabulary(&self) -> String { "biodiversity: Variety of life (species, genetic, ecosystem diversity).\nendemic: Found only in one specific area | Fynbos is endemic to the Western Cape.\nclassification: Organising life into hierarchical groups | Domain → Kingdom → ... → Species.\nbinomial nomenclature: Two-word scientific naming system | Homo sapiens.".to_string() }
    fn flashcard(&self) -> String { "How many floral kingdoms are there on Earth? | 6 (Cape Floral Kingdom is the smallest and most diverse per area)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Explain why preserving biodiversity is important for medicine.\nMany drugs come from natural compounds (aspirin from willow bark, quinine from cinchona). Unknown species may hold future cures. Destroying habitats = destroying potential medicines before we discover them.\n3\nWhat percentage of drugs come from natural sources? What about undiscovered species?".to_string() }
}
