// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CAPS Life Sciences Grade 12 — DNA, Genetics, Evolution, Human Evolution.

use super::TopicContent;

pub(crate) struct Gr12Genetics;
impl TopicContent for Gr12Genetics {
    fn explanation(&self) -> String {
        "DNA (deoxyribonucleic acid) is the molecule that carries genetic information. \
         It has a double helix structure (discovered by Watson & Crick, with crucial work by Rosalind Franklin). \
         DNA is made of nucleotides: sugar + phosphate + base (A, T, G, C). \
         Base pairing: A-T and G-C (Chargaff's rules). \
         A gene is a section of DNA that codes for a protein. \
         Protein synthesis: DNA → mRNA (transcription) → protein (translation).".to_string()
    }
    fn worked_example(&self, index: usize) -> String {
        match index {
            0 => "DNA strand: TAC GGA ATT\nFind the mRNA: Transcription changes T→A, A→U, G→C, C→G\nDNA: TAC GGA ATT\nmRNA: AUG CCU UAA\nAUG = start codon (methionine), CCU = proline, UAA = stop codon".to_string(),
            1 => "Cross: Bb × Bb (brown eyes × brown eyes)\nPunnett square:\n    B    b\nB  BB   Bb\nb  Bb   bb\nRatio: 3 brown : 1 blue (75% brown, 25% blue)".to_string(),
            _ => "A mutation changes one DNA base: TAC → TAA in a coding region.\nTAC codes for methionine (start). TAA on mRNA = AUU → different amino acid.\nThis is a point mutation (substitution) that changes the protein.\nCould cause genetic disease if the protein doesn't function.".to_string(),
        }
    }
    fn practice_problem(&self, difficulty: u16) -> String {
        match difficulty {
            0..=300 => "If one DNA strand is ATGCCG, what is the complementary strand?\nTACGGC\nA pairs with T, G pairs with C.\nATGCCG, UACGGC, GCCGTA\nUse Chargaff's rules: A-T and G-C.\nA→T, T→A, G→C, C→G.".to_string(),
            301..=600 => "Parents: Tt × Tt (tall × tall). What fraction of offspring will be short (tt)?\n1/4 (25%)\nPunnett: TT, Tt, Tt, tt → 1 out of 4 is tt.\n1/2, 3/4, 0\nDraw a Punnett square.\nT and t from each parent → 4 possible combinations.".to_string(),
            _ => "Sickle cell anaemia is caused by ONE base change in the haemoglobin gene. The 'wrong' amino acid makes haemoglobin molecules stick together. Explain why carriers (heterozygous) have an advantage in malaria regions.\nCarriers (HbAHbS): enough normal haemoglobin to function + sickle cells that kill malaria parasites. Homozygous normal (HbAHbA): vulnerable to malaria. Homozygous sickle (HbSHbS): severe anaemia.\nThis is heterozygote advantage — natural selection maintains BOTH alleles in the population.\nNo advantage exists\nWhat happens to malaria parasites inside sickle-shaped cells?\nSickle cells rupture → kills the parasite. Carriers get mild sickling = malaria protection.".to_string(),
        }
    }
    fn hint(&self, level: u8) -> String { match level { 1 => "DNA base pairs: A-T and G-C. Always.".to_string(), 2 => "Punnett squares show ALL possible offspring combinations.".to_string(), _ => "Transcription: DNA→mRNA in the nucleus. Translation: mRNA→protein at ribosomes.".to_string() } }
    fn misconception(&self) -> String { "WRONG: We only use 10% of our DNA.\nRIGHT: While only ~2% codes for proteins, much of the rest has regulatory functions (turning genes on/off). Recent research shows most DNA has some function.\nWHY: 'Junk DNA' was named before we understood gene regulation. It's not junk — it's control instructions.".to_string() }
    fn vocabulary(&self) -> String { "gene: Section of DNA coding for a protein | The gene for eye colour is on chromosome 15.\nallele: Different versions of the same gene | Brown (B) and blue (b) are alleles for eye colour.\nmutation: A change in the DNA base sequence | Can be harmful, neutral, or beneficial.\nheterozygous: Having two DIFFERENT alleles (e.g., Bb) | Carriers of recessive traits.".to_string() }
    fn flashcard(&self) -> String { "What is Chargaff's rule? | A=T and G=C (adenine pairs with thymine, guanine with cytosine)".to_string() }
    fn assessment_item(&self, _c: &str) -> String { "Explain why identical twins can look slightly different despite having the same DNA.\nEpigenetics: environment affects which genes are expressed. Diet, exercise, sun exposure, and even random developmental variations cause differences. DNA is the recipe, but 'cooking conditions' vary.\n3\nSame genes ≠ same gene expression. Think about what turns genes on/off.".to_string() }
}
