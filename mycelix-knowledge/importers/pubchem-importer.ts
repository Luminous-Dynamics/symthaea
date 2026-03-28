#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * PubChem → Mycelix Claims Importer
 *
 * Fetches chemical compound data from PubChem's PUG REST API and converts
 * to Claims with E4/N0/M2 classification (established chemical facts).
 *
 * PubChem API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
 * License: Public domain (US Government work)
 *
 * Usage:
 *   npx ts-node importers/pubchem-importer.ts [--limit N]
 *
 * Output:
 *   seed-data/claims/pubchem-compounds.json
 */

import {
  buildClaim,
  filterValid,
  writeSeedFile,
  enm,
  type Claim,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const PUBCHEM_API = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug';
const DEFAULT_LIMIT = 500;
const REQUEST_DELAY_MS = 300; // PubChem requests max 5/sec

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const CLASSIFICATION = enm('E4', 'N0', 'M2');

// ---------------------------------------------------------------------------
// Well-known compounds to seed (CID list)
// ---------------------------------------------------------------------------

/** Curated list of important chemical compounds by PubChem CID */
const CURATED_CIDS = [
  // Elements & simple molecules
  962,    // Water (H2O)
  280,    // Carbon dioxide (CO2)
  947,    // Molecular nitrogen (N2)
  977,    // Molecular oxygen (O2)
  783,    // Molecular hydrogen (H2)
  24823,  // Sulfuric acid (H2SO4)
  313,    // Hydrochloric acid (HCl)
  1049,   // Sodium hydroxide (NaOH)
  5462310, // Carbon (C)
  23925,  // Iron (Fe)
  23978,  // Gold (Au)
  23952,  // Copper (Cu)
  5462222, // Silicon (Si)
  23994,  // Aluminum (Al)
  23963,  // Titanium (Ti)

  // Organic chemistry fundamentals
  297,    // Methane (CH4)
  261,    // Ethanol (C2H5OH)
  176,    // Acetic acid (CH3COOH)
  702,    // Ethanol
  180,    // Acetone
  241,    // Benzene
  6325,   // Toluene
  7923,   // Phenol

  // Biochemistry
  5793,   // Glucose (C6H12O6)
  5988,   // Sucrose
  6137,   // L-Alanine
  5960,   // L-Glutamic acid
  5862,   // L-Cysteine
  5950,   // L-Tryptophan
  764,    // Glycerol
  5957,   // Adenosine triphosphate (ATP)
  5461108, // Cholesterol
  5280343, // Quercetin

  // Pharmaceuticals
  2244,   // Aspirin (acetylsalicylic acid)
  1983,   // Acetaminophen (paracetamol)
  3672,   // Ibuprofen
  2519,   // Caffeine
  5284371, // Penicillin G
  5311501, // Amoxicillin
  2157,   // Metformin
  60823,  // Atorvastatin (Lipitor)
  3821,   // Morphine
  5288826, // Omeprazole

  // Materials & industrial
  5462309, // Diamond (C allotrope)
  14917,  // Ethylene
  7501,   // Polyethylene glycol
  6049,   // Urea
  24261,  // Sodium chloride (NaCl)
  23665706, // Calcium carbonate (CaCO3)
  516892, // Sodium bicarbonate
  24524,  // Potassium chloride
  5234,   // Citric acid
  1140,   // Nitric acid

  // Vitamins
  54670067, // Ascorbic acid (Vitamin C)
  5280793,  // Retinol (Vitamin A)
  5280795,  // Cholecalciferol (Vitamin D3)
  14985,    // Alpha-tocopherol (Vitamin E)
  6037,     // Niacin (Vitamin B3)
  6019,     // Riboflavin (Vitamin B2)
  1054,     // Pyridoxine (Vitamin B6)
];

// ---------------------------------------------------------------------------
// PubChem API
// ---------------------------------------------------------------------------

interface CompoundProperty {
  CID: number;
  MolecularFormula?: string;
  MolecularWeight?: number;
  IUPACName?: string;
  InChI?: string;
  IsomericSMILES?: string;
}

async function fetchCompoundProperties(cids: number[]): Promise<CompoundProperty[]> {
  const cidStr = cids.join(',');
  const url = `${PUBCHEM_API}/compound/cid/${cidStr}/property/MolecularFormula,MolecularWeight,IUPACName,InChI,IsomericSMILES/JSON`;

  const resp = await fetch(url, {
    headers: {
      Accept: 'application/json',
      'User-Agent': 'MycelixKnowledgeImporter/1.0',
    },
  });

  if (!resp.ok) {
    throw new Error(`PubChem API error: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.PropertyTable?.Properties ?? [];
}

interface CompoundSynonym {
  CID: number;
  Synonym: string[];
}

async function fetchCompoundSynonyms(cids: number[]): Promise<Map<number, string>> {
  const nameMap = new Map<number, string>();
  // Fetch in smaller batches for synonyms
  const batchSize = 20;

  for (let i = 0; i < cids.length; i += batchSize) {
    const batch = cids.slice(i, i + batchSize);
    const cidStr = batch.join(',');
    const url = `${PUBCHEM_API}/compound/cid/${cidStr}/synonyms/JSON`;

    try {
      const resp = await fetch(url, {
        headers: { Accept: 'application/json', 'User-Agent': 'MycelixKnowledgeImporter/1.0' },
      });

      if (resp.ok) {
        const data = await resp.json();
        const items = data.InformationList?.Information ?? [];
        for (const item of items) {
          // First synonym is typically the common name
          if (item.Synonym && item.Synonym.length > 0) {
            nameMap.set(item.CID, item.Synonym[0]);
          }
        }
      }
    } catch {
      // Continue on error
    }

    await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
  }

  return nameMap;
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

function compoundToClaims(prop: CompoundProperty, commonName: string | undefined): Claim[] {
  const claims: Claim[] = [];
  const name = commonName || prop.IUPACName || `CID ${prop.CID}`;
  const sourceUri = `https://pubchem.ncbi.nlm.nih.gov/compound/${prop.CID}`;

  // Molecular formula claim
  if (prop.MolecularFormula) {
    claims.push(
      buildClaim({
        source: 'pubchem',
        content: `${name} has the molecular formula ${prop.MolecularFormula}.`,
        classification: CLASSIFICATION,
        claimType: 'Fact',
        confidence: 0.99,
        sources: [sourceUri],
        tags: ['chemistry', 'molecular-formula', 'compounds'],
      }),
    );
  }

  // Molecular weight claim
  if (prop.MolecularWeight) {
    claims.push(
      buildClaim({
        source: 'pubchem',
        content: `${name} has a molecular weight of approximately ${prop.MolecularWeight.toFixed(2)} g/mol.`,
        classification: CLASSIFICATION,
        claimType: 'Fact',
        confidence: 0.99,
        sources: [sourceUri],
        tags: ['chemistry', 'molecular-weight', 'compounds'],
      }),
    );
  }

  // IUPAC name claim (if different from common name)
  if (prop.IUPACName && commonName && prop.IUPACName.toLowerCase() !== commonName.toLowerCase()) {
    claims.push(
      buildClaim({
        source: 'pubchem',
        content: `The IUPAC name of ${commonName} is ${prop.IUPACName}.`,
        classification: CLASSIFICATION,
        claimType: 'Definition',
        confidence: 0.99,
        sources: [sourceUri],
        tags: ['chemistry', 'iupac', 'nomenclature'],
      }),
    );
  }

  return claims;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== PubChem → Mycelix Claims Importer ===');
  console.log(`Curated compounds: ${CURATED_CIDS.length}`);
  console.log(`Target: ~${limit} claims`);
  console.log();

  const cidsToFetch = CURATED_CIDS.slice(0, Math.min(CURATED_CIDS.length, Math.ceil(limit / 2)));

  // Fetch synonyms (common names) first
  console.log('--- Fetching common names ---');
  const nameMap = await fetchCompoundSynonyms(cidsToFetch);
  console.log(`  Got names for ${nameMap.size} compounds`);
  console.log();

  // Fetch properties in batches of 50
  console.log('--- Fetching compound properties ---');
  const allClaims: Claim[] = [];
  const batchSize = 50;

  for (let i = 0; i < cidsToFetch.length; i += batchSize) {
    const batch = cidsToFetch.slice(i, i + batchSize);
    console.log(`  Batch ${Math.floor(i / batchSize) + 1}: CIDs ${batch[0]}..${batch[batch.length - 1]}`);

    try {
      const props = await fetchCompoundProperties(batch);
      for (const prop of props) {
        const claims = compoundToClaims(prop, nameMap.get(prop.CID));
        allClaims.push(...claims);
      }
      console.log(`    ${props.length} compounds → ${allClaims.length} total claims`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`    [ERROR] ${msg}`);
    }

    await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));

    if (allClaims.length >= limit) break;
  }

  console.log();

  const valid = filterValid(allClaims.slice(0, limit));
  const outFile = writeSeedFile('pubchem-compounds.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
