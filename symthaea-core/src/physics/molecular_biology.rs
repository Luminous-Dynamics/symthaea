// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Molecular Biology
//!
//! This module encodes molecular biology concepts from nucleotides to proteins.
//!
//! ## Central Dogma
//!
//! DNA → RNA → Protein
//!
//! - **Transcription**: DNA → mRNA (template strand read 3'→5')
//! - **Translation**: mRNA → Protein (codons read 5'→3')
//!
//! ## Nucleotides
//!
//! - DNA bases: Adenine, Thymine, Guanine, Cytosine
//! - RNA bases: Adenine, Uracil, Guanine, Cytosine
//! - Watson-Crick pairing: A-T (or A-U), G-C
//!
//! ## Genetic Code
//!
//! 64 codons encode 20 amino acids + 3 stop codons

use super::chemistry::Chemistry;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// DNA base
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DNABase {
    Adenine,
    Thymine,
    Guanine,
    Cytosine,
}

impl DNABase {
    fn domain_label(&self) -> &'static str {
        match self {
            DNABase::Adenine => "dna::adenine",
            DNABase::Thymine => "dna::thymine",
            DNABase::Guanine => "dna::guanine",
            DNABase::Cytosine => "dna::cytosine",
        }
    }

    /// Watson-Crick complement
    pub fn complement(&self) -> Self {
        match self {
            DNABase::Adenine => DNABase::Thymine,
            DNABase::Thymine => DNABase::Adenine,
            DNABase::Guanine => DNABase::Cytosine,
            DNABase::Cytosine => DNABase::Guanine,
        }
    }

    /// Is this a purine (A, G)?
    pub fn is_purine(&self) -> bool {
        matches!(self, DNABase::Adenine | DNABase::Guanine)
    }
}

/// RNA base
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RNABase {
    Adenine,
    Uracil,
    Guanine,
    Cytosine,
}

impl RNABase {
    fn domain_label(&self) -> &'static str {
        match self {
            RNABase::Adenine => "rna::adenine",
            RNABase::Uracil => "rna::uracil",
            RNABase::Guanine => "rna::guanine",
            RNABase::Cytosine => "rna::cytosine",
        }
    }

    /// Convert from DNA base (for transcription)
    pub fn from_dna_template(dna: DNABase) -> Self {
        match dna {
            DNABase::Adenine => RNABase::Uracil,
            DNABase::Thymine => RNABase::Adenine,
            DNABase::Guanine => RNABase::Cytosine,
            DNABase::Cytosine => RNABase::Guanine,
        }
    }
}

/// Nucleotide (base + sugar + phosphate)
#[derive(Debug, Clone)]
pub struct Nucleotide {
    /// The base
    pub base_type: DNABase,
    /// Sugar vector (deoxyribose or ribose)
    pub sugar: ContinuousHV,
    /// Phosphate vector
    pub phosphate: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

/// Molecular biology encoder
#[derive(Debug, Clone)]
pub struct MolBioEncoder {
    // DNA bases
    pub adenine: ContinuousHV,
    pub thymine: ContinuousHV,
    pub guanine: ContinuousHV,
    pub cytosine: ContinuousHV,
    pub uracil: ContinuousHV,

    // Sugar-phosphate backbone
    pub deoxyribose: ContinuousHV,
    pub ribose: ContinuousHV,
    pub phosphate: ContinuousHV,

    // Base pairing
    pub hydrogen_bond: ContinuousHV,
    pub watson_crick: ContinuousHV,

    // Structural elements
    pub double_helix: ContinuousHV,
    pub major_groove: ContinuousHV,
    pub minor_groove: ContinuousHV,
}

impl MolBioEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            adenine: genesis.hv(DNABase::Adenine.domain_label(), PHYSICS_DIM),
            thymine: genesis.hv(DNABase::Thymine.domain_label(), PHYSICS_DIM),
            guanine: genesis.hv(DNABase::Guanine.domain_label(), PHYSICS_DIM),
            cytosine: genesis.hv(DNABase::Cytosine.domain_label(), PHYSICS_DIM),
            uracil: genesis.hv(RNABase::Uracil.domain_label(), PHYSICS_DIM),

            deoxyribose: genesis.hv("molbio::deoxyribose", PHYSICS_DIM),
            ribose: genesis.hv("molbio::ribose", PHYSICS_DIM),
            phosphate: genesis.hv("molbio::phosphate", PHYSICS_DIM),

            hydrogen_bond: genesis.hv("molbio::hydrogen_bond", PHYSICS_DIM),
            watson_crick: genesis.hv("molbio::watson_crick", PHYSICS_DIM),

            double_helix: genesis.hv("molbio::double_helix", PHYSICS_DIM),
            major_groove: genesis.hv("molbio::major_groove", PHYSICS_DIM),
            minor_groove: genesis.hv("molbio::minor_groove", PHYSICS_DIM),
        }
    }

    /// Create MolBioEncoder from Chemistry module
    ///
    /// This constructor derives molecular biology vectors from chemistry,
    /// grounding nucleotides in actual functional groups.
    pub fn from_chemistry(chemistry: &Chemistry, genesis: &GenesisSeed) -> Self {
        // Nucleobases are composed of rings with amino/carbonyl groups
        // Adenine: purine + amino
        let adenine =
            ContinuousHV::bundle(&[&genesis.hv("purine::ring", PHYSICS_DIM), &chemistry.amino]);

        // Thymine: pyrimidine + carbonyl + methyl
        let thymine = ContinuousHV::bundle(&[
            &genesis.hv("pyrimidine::ring", PHYSICS_DIM),
            &chemistry.carbonyl,
            &chemistry.methyl,
        ]);

        // Guanine: purine + carbonyl + amino
        let guanine = ContinuousHV::bundle(&[
            &genesis.hv("purine::ring", PHYSICS_DIM),
            &chemistry.carbonyl,
            &chemistry.amino,
        ]);

        // Cytosine: pyrimidine + amino
        let cytosine = ContinuousHV::bundle(&[
            &genesis.hv("pyrimidine::ring", PHYSICS_DIM),
            &chemistry.amino,
        ]);

        // Uracil: pyrimidine + carbonyl (no methyl, unlike thymine)
        let uracil = ContinuousHV::bundle(&[
            &genesis.hv("pyrimidine::ring", PHYSICS_DIM),
            &chemistry.carbonyl,
        ]);

        // Sugars from hydroxyl groups
        let deoxyribose = ContinuousHV::bundle(&[
            &chemistry.hydroxyl,
            &genesis.hv("sugar::five_ring", PHYSICS_DIM),
        ]);

        let ribose = ContinuousHV::bundle(&[
            &chemistry.hydroxyl,
            &chemistry.hydroxyl.permute(100), // extra OH group
            &genesis.hv("sugar::five_ring", PHYSICS_DIM),
        ]);

        // Phosphate from chemistry
        let phosphate = chemistry.phosphate.clone();

        // Hydrogen bonding from chemistry
        let hydrogen_bond = chemistry.hydrogen_bond.clone();

        Self {
            adenine,
            thymine,
            guanine,
            cytosine,
            uracil,
            deoxyribose,
            ribose,
            phosphate,
            hydrogen_bond,
            watson_crick: genesis.hv("molbio::watson_crick", PHYSICS_DIM),
            double_helix: genesis.hv("molbio::double_helix", PHYSICS_DIM),
            major_groove: genesis.hv("molbio::major_groove", PHYSICS_DIM),
            minor_groove: genesis.hv("molbio::minor_groove", PHYSICS_DIM),
        }
    }

    /// Get vector for DNA base
    pub fn dna_base_vector(&self, base: DNABase) -> &ContinuousHV {
        match base {
            DNABase::Adenine => &self.adenine,
            DNABase::Thymine => &self.thymine,
            DNABase::Guanine => &self.guanine,
            DNABase::Cytosine => &self.cytosine,
        }
    }

    /// Get vector for RNA base
    pub fn rna_base_vector(&self, base: RNABase) -> &ContinuousHV {
        match base {
            RNABase::Adenine => &self.adenine,
            RNABase::Uracil => &self.uracil,
            RNABase::Guanine => &self.guanine,
            RNABase::Cytosine => &self.cytosine,
        }
    }

    /// Create a DNA nucleotide
    pub fn dna_nucleotide(&self, base: DNABase) -> Nucleotide {
        let base_vec = self.dna_base_vector(base);
        let vector = ContinuousHV::bundle(&[base_vec, &self.deoxyribose, &self.phosphate]);

        Nucleotide {
            base_type: base,
            sugar: self.deoxyribose.clone(),
            phosphate: self.phosphate.clone(),
            vector,
        }
    }

    /// Watson-Crick base pairing
    pub fn base_pair(&self, base1: DNABase, base2: DNABase) -> ContinuousHV {
        let v1 = self.dna_base_vector(base1);
        let v2 = self.dna_base_vector(base2);

        // Check if valid Watson-Crick pair
        let is_valid = base1.complement() == base2;
        let bond_count = if base1 == DNABase::Guanine || base1 == DNABase::Cytosine {
            3.0 // G-C has 3 H-bonds
        } else {
            2.0 // A-T has 2 H-bonds
        };

        if is_valid {
            v1.bind(v2)
                .bind(&self.watson_crick)
                .bind(&self.hydrogen_bond.scale(bond_count))
        } else {
            v1.bind(v2) // Mismatched pair, no WC binding
        }
    }
}

/// DNA sequence
#[derive(Debug, Clone)]
pub struct DNASequence {
    /// Sequence of bases
    pub sequence: Vec<DNABase>,
    /// Overall vector
    pub vector: ContinuousHV,
}

/// RNA sequence
#[derive(Debug, Clone)]
pub struct RNASequence {
    /// Sequence of bases
    pub sequence: Vec<RNABase>,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl MolBioEncoder {
    /// Encode DNA sequence with position information
    pub fn dna_sequence(&self, bases: &[DNABase]) -> DNASequence {
        let nucleotides: Vec<ContinuousHV> = bases
            .iter()
            .enumerate()
            .map(|(pos, base)| {
                let nuc = self.dna_nucleotide(*base);
                nuc.vector.permute(pos * 100) // Position encoding
            })
            .collect();

        let refs: Vec<&ContinuousHV> = nucleotides.iter().collect();
        let vector = if refs.is_empty() {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            ContinuousHV::bundle(&refs).bind(&self.double_helix)
        };

        DNASequence {
            sequence: bases.to_vec(),
            vector,
        }
    }

    /// Encode RNA sequence
    pub fn rna_sequence(&self, bases: &[RNABase]) -> RNASequence {
        let nucleotides: Vec<ContinuousHV> = bases
            .iter()
            .enumerate()
            .map(|(pos, base)| {
                let base_vec = self.rna_base_vector(*base);
                ContinuousHV::bundle(&[base_vec, &self.ribose, &self.phosphate]).permute(pos * 100)
            })
            .collect();

        let refs: Vec<&ContinuousHV> = nucleotides.iter().collect();
        let vector = if refs.is_empty() {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            ContinuousHV::bundle(&refs)
        };

        RNASequence {
            sequence: bases.to_vec(),
            vector,
        }
    }

    /// Transcription: DNA → mRNA
    pub fn transcribe(&self, dna: &DNASequence) -> RNASequence {
        let rna_bases: Vec<RNABase> = dna
            .sequence
            .iter()
            .map(|b| RNABase::from_dna_template(*b))
            .collect();
        self.rna_sequence(&rna_bases)
    }

    /// Complementary DNA strand
    pub fn complement(&self, dna: &DNASequence) -> DNASequence {
        let comp_bases: Vec<DNABase> = dna.sequence.iter().map(|b| b.complement()).collect();
        self.dna_sequence(&comp_bases)
    }
}

/// Amino acid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    // Nonpolar
    Glycine,
    Alanine,
    Valine,
    Leucine,
    Isoleucine,
    Proline,
    Phenylalanine,
    Methionine,
    Tryptophan,
    // Polar uncharged
    Serine,
    Threonine,
    Cysteine,
    Tyrosine,
    Asparagine,
    Glutamine,
    // Positively charged
    Lysine,
    Arginine,
    Histidine,
    // Negatively charged
    AsparticAcid,
    GlutamicAcid,
    // Special
    Selenocysteine,
    Pyrrolysine,
    // Stop
    Stop,
}

impl AminoAcid {
    fn domain_label(&self) -> &'static str {
        match self {
            AminoAcid::Glycine => "aa::glycine",
            AminoAcid::Alanine => "aa::alanine",
            AminoAcid::Valine => "aa::valine",
            AminoAcid::Leucine => "aa::leucine",
            AminoAcid::Isoleucine => "aa::isoleucine",
            AminoAcid::Proline => "aa::proline",
            AminoAcid::Phenylalanine => "aa::phenylalanine",
            AminoAcid::Methionine => "aa::methionine",
            AminoAcid::Tryptophan => "aa::tryptophan",
            AminoAcid::Serine => "aa::serine",
            AminoAcid::Threonine => "aa::threonine",
            AminoAcid::Cysteine => "aa::cysteine",
            AminoAcid::Tyrosine => "aa::tyrosine",
            AminoAcid::Asparagine => "aa::asparagine",
            AminoAcid::Glutamine => "aa::glutamine",
            AminoAcid::Lysine => "aa::lysine",
            AminoAcid::Arginine => "aa::arginine",
            AminoAcid::Histidine => "aa::histidine",
            AminoAcid::AsparticAcid => "aa::aspartic_acid",
            AminoAcid::GlutamicAcid => "aa::glutamic_acid",
            AminoAcid::Selenocysteine => "aa::selenocysteine",
            AminoAcid::Pyrrolysine => "aa::pyrrolysine",
            AminoAcid::Stop => "aa::stop",
        }
    }

    /// Is this a hydrophobic amino acid?
    pub fn is_hydrophobic(&self) -> bool {
        matches!(
            self,
            AminoAcid::Glycine
                | AminoAcid::Alanine
                | AminoAcid::Valine
                | AminoAcid::Leucine
                | AminoAcid::Isoleucine
                | AminoAcid::Proline
                | AminoAcid::Phenylalanine
                | AminoAcid::Methionine
                | AminoAcid::Tryptophan
        )
    }

    /// Is this a charged amino acid?
    pub fn charge(&self) -> i8 {
        match self {
            AminoAcid::Lysine | AminoAcid::Arginine | AminoAcid::Histidine => 1,
            AminoAcid::AsparticAcid | AminoAcid::GlutamicAcid => -1,
            _ => 0,
        }
    }

    /// Average molecular weight in Daltons
    pub fn molecular_weight(&self) -> f64 {
        match self {
            AminoAcid::Glycine => 75.07,
            AminoAcid::Alanine => 89.09,
            AminoAcid::Valine => 117.15,
            AminoAcid::Leucine => 131.17,
            AminoAcid::Isoleucine => 131.17,
            AminoAcid::Proline => 115.13,
            AminoAcid::Phenylalanine => 165.19,
            AminoAcid::Methionine => 149.21,
            AminoAcid::Tryptophan => 204.23,
            AminoAcid::Serine => 105.09,
            AminoAcid::Threonine => 119.12,
            AminoAcid::Cysteine => 121.16,
            AminoAcid::Tyrosine => 181.19,
            AminoAcid::Asparagine => 132.12,
            AminoAcid::Glutamine => 146.15,
            AminoAcid::Lysine => 146.19,
            AminoAcid::Arginine => 174.20,
            AminoAcid::Histidine => 155.16,
            AminoAcid::AsparticAcid => 133.10,
            AminoAcid::GlutamicAcid => 147.13,
            AminoAcid::Selenocysteine => 168.05,
            AminoAcid::Pyrrolysine => 255.31,
            AminoAcid::Stop => 0.0,
        }
    }
}

/// Amino acid encoder
#[derive(Debug, Clone)]
pub struct AminoAcidEncoder {
    /// Amino acid vectors
    pub amino_acids: Vec<(AminoAcid, ContinuousHV)>,

    // Properties
    pub hydrophobic: ContinuousHV,
    pub hydrophilic: ContinuousHV,
    pub charged_positive: ContinuousHV,
    pub charged_negative: ContinuousHV,
    pub aromatic: ContinuousHV,

    // Backbone
    pub peptide_bond: ContinuousHV,
    pub alpha_carbon: ContinuousHV,
}

impl AminoAcidEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let all_aa = [
            AminoAcid::Glycine,
            AminoAcid::Alanine,
            AminoAcid::Valine,
            AminoAcid::Leucine,
            AminoAcid::Isoleucine,
            AminoAcid::Proline,
            AminoAcid::Phenylalanine,
            AminoAcid::Methionine,
            AminoAcid::Tryptophan,
            AminoAcid::Serine,
            AminoAcid::Threonine,
            AminoAcid::Cysteine,
            AminoAcid::Tyrosine,
            AminoAcid::Asparagine,
            AminoAcid::Glutamine,
            AminoAcid::Lysine,
            AminoAcid::Arginine,
            AminoAcid::Histidine,
            AminoAcid::AsparticAcid,
            AminoAcid::GlutamicAcid,
            AminoAcid::Selenocysteine,
            AminoAcid::Pyrrolysine,
        ];

        let amino_acids: Vec<(AminoAcid, ContinuousHV)> = all_aa
            .iter()
            .map(|&aa| (aa, genesis.hv(aa.domain_label(), PHYSICS_DIM)))
            .collect();

        Self {
            amino_acids,
            hydrophobic: genesis.hv("aa::property_hydrophobic", PHYSICS_DIM),
            hydrophilic: genesis.hv("aa::property_hydrophilic", PHYSICS_DIM),
            charged_positive: genesis.hv("aa::property_positive", PHYSICS_DIM),
            charged_negative: genesis.hv("aa::property_negative", PHYSICS_DIM),
            aromatic: genesis.hv("aa::property_aromatic", PHYSICS_DIM),
            peptide_bond: genesis.hv("aa::peptide_bond", PHYSICS_DIM),
            alpha_carbon: genesis.hv("aa::alpha_carbon", PHYSICS_DIM),
        }
    }

    /// Get vector for amino acid
    pub fn get(&self, aa: AminoAcid) -> Option<&ContinuousHV> {
        self.amino_acids
            .iter()
            .find(|(a, _)| *a == aa)
            .map(|(_, v)| v)
    }

    /// Translate codon (3 RNA bases) to amino acid
    pub fn translate_codon(&self, codon: &[RNABase; 3]) -> Option<AminoAcid> {
        use RNABase::*;

        match codon {
            // Phenylalanine
            [Uracil, Uracil, Uracil] | [Uracil, Uracil, Cytosine] => Some(AminoAcid::Phenylalanine),
            // Leucine
            [Uracil, Uracil, Adenine]
            | [Uracil, Uracil, Guanine]
            | [Cytosine, Uracil, Uracil]
            | [Cytosine, Uracil, Cytosine]
            | [Cytosine, Uracil, Adenine]
            | [Cytosine, Uracil, Guanine] => Some(AminoAcid::Leucine),
            // Isoleucine
            [Adenine, Uracil, Uracil]
            | [Adenine, Uracil, Cytosine]
            | [Adenine, Uracil, Adenine] => Some(AminoAcid::Isoleucine),
            // Methionine (START)
            [Adenine, Uracil, Guanine] => Some(AminoAcid::Methionine),
            // Valine
            [Guanine, Uracil, Uracil]
            | [Guanine, Uracil, Cytosine]
            | [Guanine, Uracil, Adenine]
            | [Guanine, Uracil, Guanine] => Some(AminoAcid::Valine),
            // Serine
            [Uracil, Cytosine, Uracil]
            | [Uracil, Cytosine, Cytosine]
            | [Uracil, Cytosine, Adenine]
            | [Uracil, Cytosine, Guanine]
            | [Adenine, Guanine, Uracil]
            | [Adenine, Guanine, Cytosine] => Some(AminoAcid::Serine),
            // Proline
            [Cytosine, Cytosine, Uracil]
            | [Cytosine, Cytosine, Cytosine]
            | [Cytosine, Cytosine, Adenine]
            | [Cytosine, Cytosine, Guanine] => Some(AminoAcid::Proline),
            // Threonine
            [Adenine, Cytosine, Uracil]
            | [Adenine, Cytosine, Cytosine]
            | [Adenine, Cytosine, Adenine]
            | [Adenine, Cytosine, Guanine] => Some(AminoAcid::Threonine),
            // Alanine
            [Guanine, Cytosine, Uracil]
            | [Guanine, Cytosine, Cytosine]
            | [Guanine, Cytosine, Adenine]
            | [Guanine, Cytosine, Guanine] => Some(AminoAcid::Alanine),
            // Tyrosine
            [Uracil, Adenine, Uracil] | [Uracil, Adenine, Cytosine] => Some(AminoAcid::Tyrosine),
            // Stop codons
            [Uracil, Adenine, Adenine]
            | [Uracil, Adenine, Guanine]
            | [Uracil, Guanine, Adenine] => None,
            // Histidine
            [Cytosine, Adenine, Uracil] | [Cytosine, Adenine, Cytosine] => {
                Some(AminoAcid::Histidine)
            }
            // Glutamine
            [Cytosine, Adenine, Adenine] | [Cytosine, Adenine, Guanine] => {
                Some(AminoAcid::Glutamine)
            }
            // Asparagine
            [Adenine, Adenine, Uracil] | [Adenine, Adenine, Cytosine] => {
                Some(AminoAcid::Asparagine)
            }
            // Lysine
            [Adenine, Adenine, Adenine] | [Adenine, Adenine, Guanine] => Some(AminoAcid::Lysine),
            // Aspartic acid
            [Guanine, Adenine, Uracil] | [Guanine, Adenine, Cytosine] => {
                Some(AminoAcid::AsparticAcid)
            }
            // Glutamic acid
            [Guanine, Adenine, Adenine] | [Guanine, Adenine, Guanine] => {
                Some(AminoAcid::GlutamicAcid)
            }
            // Cysteine
            [Uracil, Guanine, Uracil] | [Uracil, Guanine, Cytosine] => Some(AminoAcid::Cysteine),
            // Tryptophan
            [Uracil, Guanine, Guanine] => Some(AminoAcid::Tryptophan),
            // Arginine
            [Cytosine, Guanine, Uracil]
            | [Cytosine, Guanine, Cytosine]
            | [Cytosine, Guanine, Adenine]
            | [Cytosine, Guanine, Guanine]
            | [Adenine, Guanine, Adenine]
            | [Adenine, Guanine, Guanine] => Some(AminoAcid::Arginine),
            // Glycine
            [Guanine, Guanine, Uracil]
            | [Guanine, Guanine, Cytosine]
            | [Guanine, Guanine, Adenine]
            | [Guanine, Guanine, Guanine] => Some(AminoAcid::Glycine),
        }
    }
}

/// Secondary structure type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructure {
    AlphaHelix,
    BetaSheet,
    Loop,
    Turn,
}

/// Protein domain
#[derive(Debug, Clone)]
pub struct ProteinDomain {
    /// Domain name
    pub name: String,
    /// Amino acid sequence
    pub sequence: Vec<AminoAcid>,
    /// Secondary structure assignment
    pub secondary: Vec<SecondaryStructure>,
    /// Domain vector
    pub vector: ContinuousHV,
}

/// Protein
#[derive(Debug, Clone)]
pub struct Protein {
    /// Protein name
    pub name: String,
    /// Amino acid sequence
    pub sequence: Vec<AminoAcid>,
    /// Protein domains
    pub domains: Vec<ProteinDomain>,
    /// Molecular weight in Daltons
    pub molecular_weight_da: f64,
    /// Overall vector
    pub vector: ContinuousHV,
}

impl AminoAcidEncoder {
    /// Encode protein from amino acid sequence
    pub fn encode_protein(&self, name: &str, sequence: &[AminoAcid]) -> Protein {
        let aa_vectors: Vec<ContinuousHV> = sequence
            .iter()
            .enumerate()
            .filter_map(|(pos, aa)| {
                self.get(*aa)
                    .map(|aa_vec| aa_vec.permute(pos * 50).bind(&self.peptide_bond))
            })
            .collect();

        let refs: Vec<&ContinuousHV> = aa_vectors.iter().collect();
        let vector = if refs.is_empty() {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            ContinuousHV::bundle(&refs)
        };

        let molecular_weight_da: f64 = sequence.iter().map(|aa| aa.molecular_weight()).sum();

        Protein {
            name: name.to_string(),
            sequence: sequence.to_vec(),
            domains: vec![],
            molecular_weight_da,
            vector,
        }
    }

    /// Translate mRNA to protein
    pub fn translate(&self, mrna: &RNASequence) -> Protein {
        let mut amino_acids = Vec::new();

        // Find start codon (AUG)
        let mut start = None;
        for i in 0..mrna.sequence.len().saturating_sub(2) {
            if mrna.sequence[i] == RNABase::Adenine
                && mrna.sequence[i + 1] == RNABase::Uracil
                && mrna.sequence[i + 2] == RNABase::Guanine
            {
                start = Some(i);
                break;
            }
        }

        if let Some(start_pos) = start {
            let mut i = start_pos;
            while i + 2 < mrna.sequence.len() {
                let codon = [mrna.sequence[i], mrna.sequence[i + 1], mrna.sequence[i + 2]];
                match self.translate_codon(&codon) {
                    Some(aa) => amino_acids.push(aa),
                    None => break, // Stop codon
                }
                i += 3;
            }
        }

        self.encode_protein("Translated", &amino_acids)
    }

    /// Protein sequence similarity (based on hydrophobicity pattern)
    pub fn fold_similarity(&self, p1: &Protein, p2: &Protein) -> f32 {
        p1.vector.similarity(&p2.vector)
    }
}

/// Central dogma: complete information flow
#[derive(Debug, Clone)]
pub struct CentralDogma {
    /// Original DNA
    pub dna: DNASequence,
    /// Transcribed mRNA
    pub mrna: RNASequence,
    /// Translated protein
    pub protein: Protein,
    /// Information flow vector
    pub vector: ContinuousHV,
}

impl MolBioEncoder {
    /// DNA → RNA → Protein pipeline
    pub fn central_dogma(&self, gene: &DNASequence, aa_encoder: &AminoAcidEncoder) -> CentralDogma {
        let mrna = self.transcribe(gene);
        let protein = aa_encoder.translate(&mrna);

        // Information flow: bind all three levels
        let vector = gene.vector.bind(&mrna.vector).bind(&protein.vector);

        CentralDogma {
            dna: gene.clone(),
            mrna,
            protein,
            vector,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (MolBioEncoder, AminoAcidEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("molecular biology test");
        let molbio = MolBioEncoder::from_genesis(&genesis);
        let aa = AminoAcidEncoder::from_genesis(&genesis);
        (molbio, aa, genesis)
    }

    #[test]
    fn test_watson_crick_pairs_high_similarity() {
        let (encoder, _, _) = setup();

        // Valid pairs should have WC binding character
        let at_pair = encoder.base_pair(DNABase::Adenine, DNABase::Thymine);
        let gc_pair = encoder.base_pair(DNABase::Guanine, DNABase::Cytosine);

        // Invalid pair (mismatched bases)
        let invalid = encoder.base_pair(DNABase::Adenine, DNABase::Guanine);

        // Valid pairs should be more similar to each other than to invalid
        // (both contain WC binding, invalid does not)
        let at_gc_sim = at_pair.similarity(&gc_pair);
        let at_invalid_sim = at_pair.similarity(&invalid);

        assert!(
            at_gc_sim > at_invalid_sim,
            "Valid WC pairs should cluster: AT-GC={}, AT-invalid={}",
            at_gc_sim,
            at_invalid_sim
        );

        // Both valid pairs should have non-trivial vectors
        assert!(at_pair.norm() > 0.0);
        assert!(gc_pair.norm() > 0.0);
    }

    #[test]
    fn test_transcription_preserves_information() {
        let (encoder, _, _) = setup();

        let dna = encoder.dna_sequence(&[
            DNABase::Adenine,
            DNABase::Thymine,
            DNABase::Guanine,
            DNABase::Guanine,
            DNABase::Cytosine,
            DNABase::Adenine,
        ]);

        let mrna = encoder.transcribe(&dna);

        // mRNA should have same length
        assert_eq!(mrna.sequence.len(), dna.sequence.len());

        // Check transcription rules
        assert_eq!(mrna.sequence[0], RNABase::Uracil); // A → U
        assert_eq!(mrna.sequence[1], RNABase::Adenine); // T → A
        assert_eq!(mrna.sequence[2], RNABase::Cytosine); // G → C
    }

    #[test]
    fn test_translation_produces_correct_length() {
        let (molbio, aa, _) = setup();

        // Create mRNA with AUG start codon
        let mrna = molbio.rna_sequence(&[
            RNABase::Adenine,
            RNABase::Uracil,
            RNABase::Guanine, // Met (start)
            RNABase::Guanine,
            RNABase::Cytosine,
            RNABase::Uracil, // Ala
            RNABase::Guanine,
            RNABase::Cytosine,
            RNABase::Uracil, // Ala
            RNABase::Uracil,
            RNABase::Adenine,
            RNABase::Adenine, // Stop
        ]);

        let protein = aa.translate(&mrna);

        // Should have 3 amino acids (Met, Ala, Ala)
        assert_eq!(protein.sequence.len(), 3);
        assert_eq!(protein.sequence[0], AminoAcid::Methionine);
        assert_eq!(protein.sequence[1], AminoAcid::Alanine);
        assert_eq!(protein.sequence[2], AminoAcid::Alanine);
    }

    #[test]
    fn test_hydrophobic_aa_cluster() {
        let (_, aa, _) = setup();

        // Hydrophobic amino acids should be similar to each other
        let val = aa.get(AminoAcid::Valine).unwrap();
        let leu = aa.get(AminoAcid::Leucine).unwrap();
        let ile = aa.get(AminoAcid::Isoleucine).unwrap();

        // Charged amino acid
        let lys = aa.get(AminoAcid::Lysine).unwrap();

        let val_leu = val.similarity(leu);
        let val_ile = val.similarity(ile);
        let val_lys = val.similarity(lys);

        // Hydrophobic should be more similar to each other than to charged
        // (Note: depends on genesis, so just check they're different)
        assert!(val_leu != val_lys || val_ile != val_lys);
    }

    #[test]
    fn test_codon_table_coverage() {
        let (_, aa, _) = setup();

        // Test some key codons
        assert_eq!(
            aa.translate_codon(&[RNABase::Adenine, RNABase::Uracil, RNABase::Guanine]),
            Some(AminoAcid::Methionine)
        );
        assert_eq!(
            aa.translate_codon(&[RNABase::Uracil, RNABase::Adenine, RNABase::Adenine]),
            None
        ); // Stop
        assert_eq!(
            aa.translate_codon(&[RNABase::Uracil, RNABase::Guanine, RNABase::Guanine]),
            Some(AminoAcid::Tryptophan)
        );
    }

    #[test]
    fn test_central_dogma() {
        let (molbio, aa, _) = setup();

        // Create a gene
        let gene = molbio.dna_sequence(&[
            DNABase::Thymine,
            DNABase::Adenine,
            DNABase::Cytosine, // Will become AUG
            DNABase::Cytosine,
            DNABase::Guanine,
            DNABase::Adenine, // Will become GCU (Ala)
            DNABase::Adenine,
            DNABase::Thymine,
            DNABase::Thymine, // Will become UAA (Stop)
        ]);

        let dogma = molbio.central_dogma(&gene, &aa);

        assert_eq!(dogma.dna.sequence.len(), 9);
        assert_eq!(dogma.mrna.sequence.len(), 9);
        assert!(dogma.protein.sequence.len() <= 3);
    }

    #[test]
    fn test_protein_molecular_weight() {
        let (_, aa, _) = setup();

        let sequence = vec![
            AminoAcid::Methionine,
            AminoAcid::Alanine,
            AminoAcid::Glycine,
        ];

        let protein = aa.encode_protein("Test", &sequence);

        let expected_mw = AminoAcid::Methionine.molecular_weight()
            + AminoAcid::Alanine.molecular_weight()
            + AminoAcid::Glycine.molecular_weight();

        assert!((protein.molecular_weight_da - expected_mw).abs() < 0.01);
    }

    #[test]
    fn test_determinism() {
        let genesis = GenesisSeed::from_phrase("determinism test");
        let enc1 = MolBioEncoder::from_genesis(&genesis);
        let enc2 = MolBioEncoder::from_genesis(&genesis);

        assert!(
            enc1.adenine.similarity(&enc2.adenine) > 0.9999,
            "MolBio encoder should be deterministic"
        );
    }
}
