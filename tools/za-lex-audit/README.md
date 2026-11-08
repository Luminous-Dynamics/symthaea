# ZA_LEX resource provenance audit

This audit verifies a pinned external snapshot of [NWU-MuST/za_lex](https://github.com/NWU-MuST/za_lex). It checks the exact upstream commit, Git blob SHA-1 and byte size for every manifest-listed file, and internal consistency of the language/resource/license registry.

It intentionally **does not copy or vendor** third-party dictionaries, G2P rules, grapheme inventories or phoneme sets into Symthaea. The report contains only paths, source hashes, sizes and profile IDs.

## Pinned source

- Repository: \`NWU-MuST/za_lex\`
- Branch at audit time: \`master\`
- Commit: \`59a93768d72b5804f8b87bc3dd409320c8f0c328\`
- Last upstream push in the inspected repository metadata: 14 December 2017.
- Manifest: \`resource-manifest.json\`

The pin is immutable. Do not float to \`master\`; re-audit an updated commit before changing the pin.

## Candidate profiles

| Profile | Resource types | License metadata | Important limitation |
|---|---|---|---|
| \`af-ZA\` | pronunciation dictionary, graphemes, phone inventory/map | CC BY-SA 2.5 South Africa | derived RCRL dictionary; stress markings are work-in-progress |
| \`en-ZA\` | limited manual dictionary, graphemes, phone inventory/map | CC BY 4.0 plus derived NCHLT CC BY 3.0 attribution | README says one pronunciation per entry; heteronyms need variants |
| \`st-ZA\` | NCHLT dictionary, context-sensitive rules, phone inventory/map | CC BY 3.0 | historical rule-match statistic is not independent accuracy |
| \`tn-ZA\` | NCHLT dictionary, context-sensitive rules, phone inventory/map | CC BY 3.0 | reported match uses simplified vowel diacritics |
| \`xh-ZA\` | NCHLT dictionary, context-sensitive rules, click-capable inventory | CC BY 3.0 | NCHLT 2013 inventory does not encode tone or vowel length |
| \`zu-ZA\` | NCHLT dictionary, context-sensitive rules, click-capable inventory, morphological rules | CC BY 3.0 | rule header has likely copy/paste error; tone not encoded |

The manifest records each file's license notice and upstream Git blob identity. The Afrikaans dictionary carries CC BY-SA 2.5 ZA. The South African English directory identifies CC BY 4.0 and derived NCHLT CC BY 3.0 source material. Sesotho, Setswana, isiXhosa and isiZulu records identify CC BY 3.0 and require attribution to DAC, CSIR and NWU.

## Historical rule-match notes

The pinned upstream rule comments report 14,941/15,000 for Sesotho, 14,811/15,000 for Setswana under simplified vowel-diacritic matching, 14,999/15,000 for isiXhosa, and 14,942/15,000 for isiZulu (with a documented alternative rule affecting that count). These compare rules with the associated NCHLT-inlang dictionaries used during rule development. They are **not independent held-out pronunciation accuracy**, singer intelligibility, or dialect coverage.

## Source limitations

- The source tree is old (last push in the inspected metadata was 2017). It needs a current community/linguistic review.
- \`data/zul/g2p.translit.txt\` describes itself in one header sentence as transcribing isiXhosa despite being in the isiZulu directory and otherwise describing isiZulu. This is recorded as an upstream documentation defect.
- isiXhosa and isiZulu require real click-phone units and place/phonation contrasts; they cannot be handled by ASCII character splitting.
- The NCHLT 2013 rules note that tone is not represented in the standard phone sets. A singing profile needs separate tone/context evidence.
- isiZulu morphology resources are distinct from pronunciation rules; they must be integrated and validated separately.
- The source phone sets contain syllabic consonants and ejective/aspirated contrasts, which the current generic English VocalPerformance v2 schema does not fully express.

## Rights boundary

The upstream repository notice licenses scripts under Apache-2.0 or MIT but explicitly assigns data files their own notices. This manifest records the data notices by language and source. It is not legal advice and does not establish that a particular downstream bundling, adaptation, model or derivative weight is automatically permitted. Retain attribution, licence notices and ShareAlike requirements where applicable; obtain a separate legal review before distributing adapted dictionaries or model assets. No voice-synthesis or singer-training permission is inferred from a pronunciation dictionary.

## Verify

From a local clone of Symthaea and a separately checked out copy of the pinned upstream source:

\`\`\`bash
python3 tools/za-lex-audit/audit.py \\
  --root /path/to/za_lex \\
  --manifest tools/za-lex-audit/resource-manifest.json \\
  --expected-commit 59a93768d72b5804f8b87bc3dd409320c8f0c328 \\
  --output za-lex-audit-report.json

python3 -m unittest discover -s tools/za-lex-audit -p 'test_*.py' -v
\`\`\`

The verifier recomputes Git blob SHA-1 values using Git's blob object format. It reports metadata only.
