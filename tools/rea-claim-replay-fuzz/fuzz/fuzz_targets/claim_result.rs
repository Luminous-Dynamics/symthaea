#![no_main]

use libfuzzer_sys::fuzz_target;
use rea_claim_replay_verifier::replay_claim_evaluation;

// This fixed request is valid and deliberately has no premises. Arbitrary
// fuzz bytes therefore reach the result parser whenever its input bounds allow.
const CLAIM: &[u8] = br#"{
  "schema":"luminous.rea.claim.v0",
  "claim_id":"fuzz-result-parser",
  "predicate":"producer_authenticity",
  "scope":{
    "subject_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "operation":"inspect_artifact",
    "platform":null,
    "platform_version":null,
    "environment_id":null
  },
  "asserted_value":null,
  "source_predicate_type":null,
  "evidence_ids":[],
  "assumptions":[],
  "falsification_condition":"an independent authenticity proof is present"
}"#;

fuzz_target!(|input: &[u8]| {
    let _ = replay_claim_evaluation(CLAIM, &[], input);
});
