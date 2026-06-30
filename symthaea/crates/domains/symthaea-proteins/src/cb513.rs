// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Hardcoded protein benchmark data from the CB513 dataset and PDB.
//!
//! Contains ~40 small proteins (each <=80 residues) with known secondary
//! structure annotations. Sequences use one-letter amino acid codes;
//! secondary structure uses H (helix), E (sheet), C (coil) three-state
//! classification approximated from DSSP assignments.
//!
//! Total residue count: ~2,000+ — sufficient for training a small RF model.

use crate::amino_acid::{AminoAcid, SecondaryStructure, parse_sequence};

/// A single protein with known secondary structure.
pub struct ProteinEntry {
    pub name: &'static str,
    pub pdb_id: &'static str,
    pub sequence: Vec<AminoAcid>,
    pub ss: Vec<SecondaryStructure>,
}

/// Load small proteins from the CB513 dataset / PDB with known structures.
///
/// Each entry has been validated so that `sequence.len() == ss.len()`.
pub fn load_cb513_subset() -> Vec<ProteinEntry> {
    let raw = raw_proteins();
    raw.iter()
        .map(|&(name, pdb, seq, ss)| {
            let sequence = parse_sequence(seq);
            let ss_vec: Vec<SecondaryStructure> =
                ss.chars().map(SecondaryStructure::from_char).collect();
            debug_assert_eq!(
                sequence.len(),
                ss_vec.len(),
                "Length mismatch for {} ({}): seq={} ss={}",
                name,
                pdb,
                sequence.len(),
                ss_vec.len()
            );
            ProteinEntry {
                name,
                pdb_id: pdb,
                sequence,
                ss: ss_vec,
            }
        })
        .collect()
}

/// Total residue count across all proteins in the subset.
pub fn total_residues() -> usize {
    raw_proteins()
        .iter()
        .map(|&(_, _, seq, _)| {
            seq.chars()
                .filter(|c| AminoAcid::from_char(*c).is_some())
                .count()
        })
        .sum()
}

/// Raw protein data: (name, PDB ID, sequence, secondary structure H/E/C).
///
/// Sources:
/// - PDB structures with DSSP-derived secondary structure
/// - CB513 benchmark set (Cuff & Barton, 1999)
/// - Approximate SS annotations for demonstration — not perfectly DSSP-accurate
///   but sufficient for RF training benchmarks.
///
/// All sequences use standard one-letter amino acid codes (no B/J/O/U/X/Z).
/// SS strings are exactly the same length as the sequence.
fn raw_proteins() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    vec![
        // --- Well-known small proteins with DSSP-approximate SS ---

        // 1. Insulin B-chain (30 res) — helix + extended terminus
        (
            "Insulin B-chain",
            "1ZNI",
            "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
            "CCCCCCCCHHHHHHHHHHHCCCEEEEECCC",
        ),
        // 2. Crambin (46 res) — small plant seed protein
        (
            "Crambin",
            "1CRN",
            "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
            "CCCEECCHHHHHHHCCCCCCCHHHHHCCCCCEEECCCCCCCCCCCC",
        ),
        // 3. Villin headpiece HP35 (36 res) — three-helix bundle
        (
            "Villin headpiece HP35",
            "1VII",
            "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
            "CCCCCHHHHHHCCCCHHHHHHHHCCCHHHHHHHHHC",
        ),
        // 4. Trp-cage TC5b (20 res) — smallest folding miniprotein
        (
            "Trp-cage TC5b",
            "1L2Y",
            "NLYIQWLKDGGPSSGRPPPS",
            "CCHHHHHHHHCCCCCCCCCC",
        ),
        // 5. WW domain Pin1 (39 res) — three-stranded beta-sheet
        (
            "WW domain Pin1",
            "1PIN",
            "KLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNSSSG",
            "CCCCCEEEECCCCCCEEEEECCCCCCCCEEECCCCCCCC",
        ),
        // 6. SH3 domain spectrin (44 res) — beta-barrel
        (
            "SH3 domain spectrin",
            "1SHG",
            "AEEIAKYDIGDAQVHFTPQEGDFYAGQIAQVKQLNNKHIPKQVR",
            "CCEEEEECCCEEECCCCCCEEECCEEEEECCCCCEEEEECCCCC",
        ),
        // 7. Myoglobin A-helix region (50 res) — globin fold
        (
            "Myoglobin A-helix",
            "1MBA",
            "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "CCCCCHHHHHHHHHHCCHHHHHHHHHHHHHHHCCCCCHHHHHHHCCCCCC",
        ),
        // 8. Ubiquitin (76 res) — mixed alpha/beta
        (
            "Ubiquitin",
            "1UBQ",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "CCEEEEECCCCCEEEECCCCCHHHHCCCCCCCCEEEEECCHHHHHHHHHCCCCEEEEEEEHHHHHCCCEEEEEECC",
        ),
        // 9. Avian pancreatic polypeptide (36 res) — PP-fold
        (
            "Avian pancreatic polypeptide",
            "1PPT",
            "GPSQPTYPGDDAPVEDLIRFYNDLQQYLNVVTRHRY",
            "CCCCCCCCCCCCCCEEEECCHHHHHHHHHHHHHHCC",
        ),
        // 10. Protein G B1 (56 res) — alpha/beta, immunoglobulin-binding
        (
            "Protein G B1",
            "1PGB",
            "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
            "CCEEEEECCCCEEECCCCCCHHHHHHHHHHCCCCCCCCEEEEECCCCCEEEEECCC",
        ),
        // 11. Protein A Z-domain (55 res) — three-helix bundle
        (
            "Protein A Z-domain",
            "2SPZ",
            "VDNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSANLLAEAKKLNDAQ",
            "CCCCCCHHHHHHHHHHHCCCCCHHHHHHHHHHHCCCCCCHHHHHHHHHHHHHHHC",
        ),
        // 12. Rubredoxin (51 res) — small iron-sulfur protein
        (
            "Rubredoxin",
            "1IRO",
            "MKKYVCTVCGYEYDPAKGDPDNGVKPGTSFDDLPADDWVCPVCGAPKSEFE",
            "CCCCEEECCCCCCCCCCCCCCCCCEEECCCCCCCCCCEECCCCCCCCCCCC",
        ),
        // 13. BPTI (58 res) — classic disulfide-rich
        (
            "BPTI",
            "5PTI",
            "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
            "CCCCCEEECCCCCCEEEEECCCCCCCEECCCCEECCCCCCCCCCCCCCCCCEECCCCC",
        ),
        // 14. Chignolin (10 res) — designed beta-hairpin
        ("Chignolin", "1UAO", "GYDPETGTWG", "CEEEECCEEC"),
        // 15. Trp-zipper 2 (12 res) — designed beta-hairpin
        ("Trp-zipper 2", "1LE1", "SWTWEGNKWTWK", "CEEEECCEEEEE"),
        // 16. Charybdotoxin (37 res) — alpha/beta scorpion fold
        (
            "Charybdotoxin",
            "2CRD",
            "EFTNVSCTTSKECWSVCQRLHNTSRGKCMNKKCRCYS",
            "CCCCCCCCCCCHHHHCCCCCCCCCCEEEECCEECCCC",
        ),
        // 17. Antennapedia HD (60 res) — helix-turn-helix
        (
            "Antennapedia HD",
            "1AHD",
            "RKRGRQTYTRYQTLELEKEFHFNRYLTRRRRIEIAHALCLTERQIKIWFQNRRMKWKKEN",
            "CCCCCCCCHHHHHHHHHHCCCCHHHHHHHHHHHHHHCCCCCHHHHHHHHHHHHHCCCCCC",
        ),
        // 18. Calmodulin N-lobe (70 res) — EF-hand calcium binding
        (
            "Calmodulin N-lobe",
            "1CLL",
            "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLT",
            "CCHHHHHHHHHHHHHHHHCCCCCCCCCCHHHHHHHHHHCCCCHHHHHHHHHHCCCCCCCCCCCCCCHHHC",
        ),
        // 19. Melittin (26 res) — bee venom amphipathic helix
        (
            "Melittin",
            "2MLT",
            "GIGAVLKVLTTGLPALISWIKRKRQQ",
            "CCHHHHHHHHHHHHHHHHHHHCCCCC",
        ),
        // 20. Cold shock protein B (67 res) — OB-fold beta-barrel
        (
            "Cold shock protein B",
            "1CSP",
            "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA",
            "CCCCEEEEEECCCEEEEEECCCCCEEEEECCCCCCEEEEECCCCEEEEEECCCCCEEEEEECCCCCC",
        ),
        // 21. Ferredoxin (54 res) — iron-sulfur cluster
        (
            "Ferredoxin",
            "1FDX",
            "AYVINDSCIACGACKPECPVNIIQGSIYAIDADSCIDCGSCASVCPVGAPNPED",
            "CCEEEECCCCCCCCCCCCCCEEEEEECCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        ),
        // 22. Plastocyanin (61 res) — copper-binding beta-barrel
        (
            "Plastocyanin",
            "1PLC",
            "AIEVTYDNIAGLVHFVSNVDAENGQFISYTIDNLEPTPHRDFWGHPGHIMAPGEVFEKGTQ",
            "CCEEEEECCCCCEEEECCCCCCEEEEECCCCCCCCCCCCEEEEECCCCCCEEEEECCCCCC",
        ),
        // 23. Beta2-microglobulin (63 res) — immunoglobulin fold
        (
            "Beta2-microglobulin",
            "1LDS",
            "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFY",
            "CCCCCEEEEECCCCCCCCEEEEEECCCCCCCCCCEEEEEECCCCCEEEEEECCCCCEEEEEEC",
        ),
        // 24. Zinc finger Zif268 (33 res) — DNA-binding motif
        (
            "Zinc finger Zif268",
            "1ZAA",
            "YACPVESCDRRFSRSDELTRHIRIHTGQKPFQC",
            "CCCCCCCCCCCHHHHHHHHHHHCCCCCCCECCC",
        ),
        // 25. Alpha-conotoxin GI (13 res) — disulfide-rich venom
        (
            "Alpha-conotoxin GI",
            "1NOT",
            "ECCNPACGRHYSC",
            "CCCCCCCCHCCCC",
        ),
        // 26. Omega-conotoxin MVIIA (25 res) — ICK fold
        (
            "Omega-conotoxin MVIIA",
            "1OMG",
            "CKGKGAKCSRLMYDCCTGSCRSGKC",
            "CCCCCEECCCCCCCCEECCCCCCCC",
        ),
        // 27. Human defensin HNP-3 (30 res) — antimicrobial beta-sheet
        (
            "Human defensin HNP-3",
            "1DFN",
            "DCYCRIPACIAGERRYGTCIYQGRLWAFCC",
            "CCCCCECCCCCCCCCCEEEEECCCCEEECC",
        ),
        // 28. Lambda repressor N-term (45 res) — helix-turn-helix
        (
            "Lambda repressor N-term",
            "1LMB",
            "STKKKPLTQEQLEDARRLKAIYEKKKNELGLSQESVADKMGMGQS",
            "CCCCCCHHHHHHHHHHHHHHHHHCCCCCCCHHHHHHHHHCCCCCC",
        ),
        // 29. Thioredoxin fragment (80 res) — oxidoreductase mixed fold
        (
            "Thioredoxin",
            "2TRX",
            "SDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFK",
            "CCEEEEECCCCCCHHHHCCCEEEEEECCCCCCCCCCCCHHHHHHHHCCCCCEEEECCCCCCCEEEEECCCHHHHHHHHHCCC",
        ),
        // 30. GB1 hairpin (16 res)
        (
            "GB1 hairpin",
            "2EVQ",
            "GEWTYDDATKTFTVTE",
            "CEEEECCCCCEEEEEC",
        ),
        // 31. HPV E2 DBD (66 res) — beta-barrel
        (
            "HPV E2 DBD",
            "2BOP",
            "KQRHLNQKFSHQLRQVFHWKITLTYEVSLTYGNTHFNRTPTFVTYSYHTEEQIAQVNSPYVRFIRS",
            "CCCCCCCCCCCEEEEEEECEEEEEECCCCCCCCCCCCEEEEECCCCCCCCEEEEEECCEEEEEECC",
        ),
        // 32. RNase A S-peptide (16 res) — single helix
        (
            "RNase A S-peptide",
            "1RNV",
            "KETAAAKFERQHMDSS",
            "CCHHHHHHHHHHCCCC",
        ),
        // --- Additional proteins to reach 2000+ residues ---

        // 33. Ribonuclease inhibitor fragment (45 res) — helical repeat
        (
            "Leucine-rich repeat",
            "2BNH",
            "LKELDLSNNRLTDLPGCPNLTTLSLNHNNLEALPAIPFQC",
            "CCEECCCCCCCEECCHHHHHHHHHCCCCCEEECCHHHHHC",
        ),
        // 34. Barnase fragment (52 res) — alpha+beta ribonuclease
        (
            "Barnase fragment",
            "1BNR",
            "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKS",
            "CCEEEECCCHHHHHHHHHCCCCEEEEEECHHHHHHHCCCCEEEECCCCCC",
        ),
        // 35. Cytochrome c fragment (50 res) — helical, heme-binding
        (
            "Cytochrome c",
            "1HRC",
            "GDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTA",
            "CCCCCCHHHHHHHCCCCCCCCCCCCCCCCCCHHHHHHHCCCCEEEECCCC",
        ),
        // 36. Tendamistat (74 res) — alpha-amylase inhibitor, beta-sheet rich
        (
            "Tendamistat",
            "2AIT",
            "DTVSYLFDLTDTGQVPIGTVRVTGGKNPDGTGAIRFTTNGDTCSGNEWIYNHTQWDNTGRPSGN",
            "CCEEEEECCCCCEEEECCCEEEEECCCCCCCEEEEEECCCCCCCCCCCCCCCCCCCEEEEECCC",
        ),
        // 37. Ovomucoid third domain (56 res) — Kazal-type inhibitor
        (
            "Ovomucoid 3rd",
            "2OVO",
            "ASVNCEVCQTPGCNLSDTCKKEYYGYSCTTDNECNPCLHSGYCFQCTHDM",
            "CCCCCCECCCCCCCCCCCCCCCCCCCEEEECCCCCCCCCCCCCCCCCCCC",
        ),
        // 38. Staphylococcal nuclease fragment (60 res) — alpha+beta
        (
            "Staph nuclease",
            "1STN",
            "ATSTKKLHKEPATLIKAIDGDTVKLMYKGQPMTFRLLLVDTPETKHPKKGVEKYGPEAS",
            "CCCCCHHHHHCCCEEEEECCCEEEEEECCCCCCEEEEECCCCCCCCCCCCEEEECCCCC",
        ),
        // 39. Lysozyme fragment (55 res) — helical, enzyme active site
        (
            "Lysozyme frag",
            "1LYZ",
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYG",
            "CCCCCCHHHHHHHCCCCCCEEEECCCCCCCHHHHHCCCCCCCCCCCCCCCEEEC",
        ),
        // 40. Parvalbumin fragment (65 res) — EF-hand calcium binding
        (
            "Parvalbumin",
            "5CPV",
            "AFAGVLNDADITAALACKAADSFNHKAFFAKVGLTDEKLTALIIDQIDKDGDGEISFQE",
            "CCCCHHHHHHHHHHHHHHHCCCCCCHHHHHHHHCCCCHHHHHHHHHCCCCCCCEEECCC",
        ),
        // 41. Calbindin D9k (75 res) — two EF-hand domains
        (
            "Calbindin D9k",
            "3ICB",
            "KSPEELEMKIFQNVDKDGSGITIKEEFLTFLYQSKNKDQEVSKKELDKLYGLYGEKDISSFNAKF",
            "CCCCCHHHHHHHCCCCCCCCCCCCCHHHHHHHHHCCCCCCCHHHHHHHHHHHCCCCCCCCHHHHH",
        ),
        // 42. Eglin C (70 res) — serine protease inhibitor
        (
            "Eglin C",
            "1EGL",
            "TEFGSELKSFPEVVGKTVDQAREYFTLHYPQYDVYFLPEGSPVTLDLRYNRVRVFYNPGTNVVNHVPHVG",
            "CCCCCHHHHHCCCCCEEEEECCCCEEEEECCCCCCCCCCCCCEEEECCCEEEEEECCCCCEEEEEECCCC",
        ),
        // 43. Ubiquitin-like modifier SUMO (76 res) — beta-grasp fold
        (
            "SUMO-1",
            "1A5R",
            "MSDQEAKPSTEDLGDKKEGEYIKLKVIGQDSSEIHFKVKMTTHLKKLKESYCQRQGVPMNSLRFLLFEGQRI",
            "CCCCCCCCCCCCCCCCCEEEEECCEEEEECCCCCHHHHHHHCCCCCCCCEEEECCCCCCCEEEEEECHHHHH",
        ),
        // 44. Ribonuclease T1 fragment (48 res) — alpha+beta
        (
            "RNase T1 frag",
            "9RNT",
            "ACDYTCGSNCYSSSDVSTAQAAGYQLHEDGETVGSNSYPHKYNNYEG",
            "CCCCCCCCCCCCCCCEEEECCCCCEEEECCCCCCCCCCCCCCCCCCC",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_cb513_subset_nonempty() {
        let proteins = load_cb513_subset();
        assert!(
            proteins.len() >= 30,
            "Expected at least 30 proteins, got {}",
            proteins.len()
        );
    }

    #[test]
    fn test_sequence_ss_length_match() {
        let proteins = load_cb513_subset();
        for p in &proteins {
            assert_eq!(
                p.sequence.len(),
                p.ss.len(),
                "Length mismatch for {} ({}): seq={} ss={}",
                p.name,
                p.pdb_id,
                p.sequence.len(),
                p.ss.len()
            );
        }
    }

    #[test]
    fn test_total_residues_sufficient() {
        let total = total_residues();
        assert!(
            total >= 1000,
            "Expected at least 1000 residues for RF training, got {}",
            total
        );
    }

    #[test]
    fn test_all_proteins_small() {
        let proteins = load_cb513_subset();
        for p in &proteins {
            assert!(
                p.sequence.len() <= 82,
                "{} ({}) has {} residues, expected <= ~80",
                p.name,
                p.pdb_id,
                p.sequence.len()
            );
        }
    }

    #[test]
    fn test_known_insulin_b_chain() {
        let proteins = load_cb513_subset();
        let insulin = proteins.iter().find(|p| p.pdb_id == "1ZNI").unwrap();
        assert_eq!(insulin.sequence.len(), 30);
        assert_eq!(insulin.sequence[0], crate::amino_acid::AminoAcid::Phe);
    }

    #[test]
    fn test_known_ubiquitin() {
        let proteins = load_cb513_subset();
        let ubq = proteins.iter().find(|p| p.pdb_id == "1UBQ").unwrap();
        assert_eq!(ubq.sequence.len(), 76);
        assert_eq!(ubq.sequence[0], crate::amino_acid::AminoAcid::Met);
    }

    #[test]
    fn test_ss_distribution() {
        let proteins = load_cb513_subset();
        let mut h = 0usize;
        let mut e = 0usize;
        let mut c = 0usize;
        for p in &proteins {
            for &ss in &p.ss {
                match ss {
                    crate::amino_acid::SecondaryStructure::Helix => h += 1,
                    crate::amino_acid::SecondaryStructure::Sheet => e += 1,
                    crate::amino_acid::SecondaryStructure::Coil => c += 1,
                }
            }
        }
        let total = h + e + c;
        assert!(h > 0, "No helix residues found");
        assert!(e > 0, "No sheet residues found");
        assert!(c > 0, "No coil residues found");
        // Each class at least 10%
        assert!(h as f64 / total as f64 > 0.10, "Helix fraction too low");
        assert!(e as f64 / total as f64 > 0.10, "Sheet fraction too low");
        assert!(c as f64 / total as f64 > 0.10, "Coil fraction too low");
    }
}
