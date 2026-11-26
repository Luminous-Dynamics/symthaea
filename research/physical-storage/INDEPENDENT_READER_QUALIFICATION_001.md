# Independent Reader Qualification — v0.1

**Programme:** INDEPENDENT-READER-QUALIFICATION-001  
**Tracking:** [Symthaea #13734](https://github.com/Luminous-Dynamics/symthaea/issues/13734)  
**Capsule profile:** [Symtropy #7571](https://github.com/Luminous-Dynamics/symtropy/issues/7571)  
**Capsule draft:** [Symtropy PR #7572](https://github.com/Luminous-Dynamics/symtropy/pull/7572)

## Purpose

This track owns the independent scoring and qualification side of the Physical Recovery Capsule. It does not own optical/material physics or physical reader control. Symtropy produces the capsule and raw reader-observation contract; Symthaea validates the recovery attempt and independently compares the restored bytes with the scorer-held original.

## Trust boundary

The decoder must not certify itself. Separate:
- package/BagIt validator;
- raw reader/observation adapter;
- decoder A and decoder B;
- scorer-only original payload and expected digest;
- independent receipt generator and verifier.

Reference bytes and challenge labels are not supplied to the decoders during blind reconstruction. The scorer alone holds the expected bytes/digest. Decoder outputs are untrusted until byte equality and independent digest checks pass.

## Receipt

`recovery_attempt_receipt_v0_1.schema.json` records:
- capsule id, challenge mode and evidence class;
- BagIt package and manifest validation states;
- reader identity, raw observation digest and calibration refs;
- decoder source commit/digest, build recipe, dependency/runtime digests and execution state;
- whether network and vendor runtime were absent;
- output length and digest;
- byte equality between decoder outputs and against the scorer-held reference;
- shared dependencies, independent review and Mycelix dependency-profile status;
- negative controls and resource measurements.

The supplied example is `fixture_only`; it intentionally has no decoder outputs, physical reader, package-validator pass or independence finding.

## Qualification ladder

1. **Fixture-only:** schema and receipt structure checks using synthetic artifacts.
2. **Capsule structure:** independent BagIt validator verifies bagit.txt, every payload manifest and tag manifest, schema, path safety and referenced files.
3. **Digital decoder:** two independently developed decoders run on a digital known-answer fixture and restore exact bytes with no network/vendor service.
4. **Physical reader replacement:** independent team uses actual physical media with vendor reader application withheld; reader adapter emits the declared raw-observation contract; both decoders pass blind exact recovery.
5. **Preservation service:** repeat reader replacement after media migration and a declared ageing/environmental period; retain PREMIS migration lineage, fixity and trust records.

These states must not collapse into a single success boolean. Digital decoder qualification is not physical reader qualification; package fixity is not authenticity; two decoder implementations are not independent failure domains.

## Negative controls

At minimum:
- flip one payload bit;
- remove an expected payload manifest entry;
- corrupt the capsule descriptor or format version;
- provide the wrong interleaver/code parameters;
- make calibration stale or inconsistent;
- force decoder A and B to disagree;
- substitute a decoder source or build dependency;
- attempt to require the original vendor runtime or network;
- supply incomplete shared-dependency closure.

Any mismatch or missing critical evidence fails closed or remains incomplete. It must never be silently promoted to qualified.

## Run/validation status

The schema and fixture are source artifacts. A full Draft 2020-12 validator, BagIt validator, source/build validator and independent digest scorer remain to be implemented and run. No decoder, reader, physical restoration or production qualification pass is claimed.
