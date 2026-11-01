// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Differential tests for pinned PyPhi IIT 4.0 (2023) primitive outputs.

use symthaea_iit::{primitive_repertoire_request_sha256, BinaryCausalModel, FormalismProfile, PrimitiveRepertoireOperation};

const TPM: [f64; 24] = [0.11920292202211757, 0.76852478349901776, 0.16798161486607552, 0.026596993576865863, 0.99888746396713979, 0.5, 0.97340300642313404, 0.40131233988754816, 0.16798161486607552, 0.88079707797788231, 0.99451370110054949, 0.5, 0.11920292202211757, 0.0054862988994504045, 0.5, 0.026596993576865863, 0.59868766011245189, 0.83201838513392445, 0.97340300642313404, 0.0011125360328603216, 0.5, 0.88079707797788231, 0.23147521650098227, 0.83201838513392445];
const CM: [bool; 9] = [true,true,true,true,true,false,false,true,true];
const STATE: [bool; 3] = [false,true,true];
struct Case { id: &'static str, candidate: &'static [usize], mechanism: &'static [usize], cause_request: &'static str, cause: &'static [f64], effect_request: &'static str, effect: &'static [f64], forward_request: &'static str, forward: &'static [f64] }
const CASES: &[Case] = &[
 Case {
  id: "AB-mechanism-empty", candidate: &[0, 1], mechanism: &[],
  cause_request: "0547a8ef542880e28a680b3f1f60a54e674159c3d818e5604502298955246f2f", cause: &[0.25, 0.25, 0.25, 0.25],
  effect_request: "4c09985d102e0971e92ad5976bc17d88de0d779026dfab3a98fa7084099c292f", effect: &[0.39540478605678203, 0.39540478605678192, 0.10459521394321813, 0.10459521394321811],
  forward_request: "a6ca845a658cc699c41b2982b304ce0854c200c6edc6b4691fdff1ba01b56692", forward: &[1, 1, 1, 1],
 },
 Case {
  id: "AB-mechanism-0-1", candidate: &[0, 1], mechanism: &[0, 1],
  cause_request: "eaffd44abaf437afcaa38dea1fff6c1892090c1039a29fe276dd292e8ec2859f", cause: &[0.3020496745917145, 0.63114186507841885, 0.0047468034154662421, 0.062061656914400257],
  effect_request: "6b3dc532baca1f219b8361c6a7e910f1095f4d8ce7621a8b25ca431cf3d6db74", effect: &[0.026567403463145938, 0.97232006050399378, 2.9590113720017906e-05, 0.0010829459191403037],
  forward_request: "f44788c73cc2d8901ebb65d460b1d51e5e4e8ea0f1f16d790f8254d431213e6e", forward: &[0.38410595109898188, 0.80260091884562257, 0.0060363430056400748, 0.078921627007567441],
 },
 Case {
  id: "ABC-mechanism-empty", candidate: &[0, 1, 2], mechanism: &[],
  cause_request: "f37b2b85c2b87e3fb4b5b2f8c16df713f6a85024c596416fa6b676aacf35a099", cause: &[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  effect_request: "86a55234087f3dd09a4104a41d46e2a002302c9e45350c28afcf4a067143b97c", effect: &[0.12500000000000003, 0.125, 0.12500000000000003, 0.125, 0.12500000000000003, 0.125, 0.12500000000000003, 0.125],
  forward_request: "da787377fccddfea086c40e444d32e2ae73129503202d968a1ae0e3843e086ca", forward: &[1, 1, 1, 1, 1, 1, 1, 1],
 },

 Case {
  id: "ABC-mechanism-0-1-2", candidate: &[0, 1, 2], mechanism: &[0, 1, 2],
  cause_request: "269132ddd98271b84d797fa089f4a16f3db3a8acbec2f9816c99acc9ee23c58d", cause: &[0.097088099845556969, 0.41509715549960591, 0.0015309017564716857, 0.050610215126307936, 0.0020629839123675569, 0.41399626212502294, 1.2632437131578487e-05, 0.019601749297535408],
  effect_request: "878718fc166839d6eb9f4f189b8770877404b6d004bbf7ef5772bf58678ad84", effect: &[0.013283701731572969, 0.48616003025199689, 1.4795056860008953e-05, 0.00054147295957015183, 0.013283701731572969, 0.48616003025199689, 1.4795056860008953e-05, 0.00054147295957015183],
  forward_request: "f62daa2dfff7b295303808591f7c62a614a70d1f9e24894ff0fd87b412c2e097", forward: &[0.11370917129320028, 0.486160030251997, 0.0017929856525837279, 0.059274469581108227, 0.0024161580197745941, 0.48487066859485117, 1.4795056860008953e-05, 0.022957485748154979],
 },
];
fn check(id: &str, op: &str, actual: &[f64], expected: &[f64]) {
 assert_eq!(actual.len(), expected.len(), "{id} {op} length");
 for (i,(a,e)) in actual.iter().zip(expected).enumerate() {
  assert!(a.is_finite(), "{id} {op}[{i}] non-finite");
  assert_eq!(*a == 0.0, *e == 0.0, "{id} {op}[{i}] support");
  assert!((a-e).abs() <= 3e-12, "{id} {op}[{i}] actual={a:.17} expected={e:.17}");
 }
}
#[test]
fn pinned_pyphi_primitive_vectors_match_rust_on_shared_fig1a_model() {
 let m = BinaryCausalModel::new(3, TPM.to_vec(), CM.to_vec()).unwrap();
 assert_eq!(m.model_sha256(), "6d67ba617b5d2ccbba711cb44f4753286a45ae6ded16fc1a544a77dc8e931ac1");
 assert_eq!(m.evaluation_sha256(&STATE).unwrap(), "1442d185bc5d2c23e3c6fc92e94f21d1c492cbe41a451cebd1d51fa36e862744");
 for c in CASES {
  let rq = primitive_repertoire_request_sha256(&m,&STATE,c.candidate,c.mechanism,c.candidate,FormalismProfile::Iit4_0_2023,PrimitiveRepertoireOperation::CauseRepertoire).unwrap();
  assert_eq!(rq,c.cause_request,"{} cause request",c.id);
  let a=m.cause_repertoire_iit4(c.candidate,c.mechanism,&STATE,c.candidate).unwrap(); check(c.id,"cause",&a,c.cause);
  let rq = primitive_repertoire_request_sha256(&m,&STATE,c.candidate,c.mechanism,c.candidate,FormalismProfile::Iit4_0_2023,PrimitiveRepertoireOperation::EffectRepertoire).unwrap();
  assert_eq!(rq,c.effect_request,"{} effect request",c.id);
  let a=m.effect_repertoire_iit4(c.candidate,c.mechanism,&STATE,c.candidate).unwrap(); check(c.id,"effect",&a,c.effect);
  let rq = primitive_repertoire_request_sha256(&m,&STATE,.candidate,c.mechanism,c.candidate,FormalismProfile::Iit4_0_2023,PrimitiveRepertoireOperation::ForwardCauseLikelihood).unwrap();
  assert_eq!(rq,c.forward_request,"{} forward request",c.id);
  let a=m.forward_cause_likelihood_iit4(c.candidate,c.mechanism,&STATE,c.candidate).unwrap(); check(c.id,"forward-caause",&a,c.forward);
 }
 assert_eq!(CASES.len()*3,12);
}
