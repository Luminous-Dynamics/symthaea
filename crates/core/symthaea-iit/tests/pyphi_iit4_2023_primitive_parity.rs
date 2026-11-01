// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full differential corpus for pinned PyPhi IIT 4.0 (2023) primitive outputs.
//! Includes every mechanism/purview control emitted by the wheel-attested generator.

#![allow(clippy::excessive_precision)]

use sha2::{Digest, Sha256};
use symthaea_iit::{
    primitive_repertoire_request_sha256, BinaryCausalModel, FormalismProfile,
    PrimitiveRepertoireOperation,
};

const TPM: [f64; 24] = [
    0.11920292202211757, 0.7685247834990178, 0.16798161486607552,
    0.026596993576865863, 0.9988874639671398, 0.5,
    0.973403006423134, 0.40131233988754816, 0.16798161486607552,
    0.8807970779778823, 0.9945137011005495, 0.5,
    0.11920292202211757, 0.0054862988994504045, 0.5,
    0.026596993576865863, 0.5986876601124519, 0.8320183851339245,
    0.973403006423134, 0.0011125360328603216, 0.5,
    0.8807970779778823, 0.23147521650098227, 0.8320183851339245,
];
const CONNECTIVITY: [bool; 9] = [
    true, true, true,
    true, true, false,
    false, true, true,
];
const STATE: [bool; 3] = [false, true, true];

struct PrimitiveCase {
    id: &'static str,
    candidate: &'static [usize],
    mechanism: &'static [usize],
    cause_request: &'static str,
    cause: &'static [f64],
    cause_vector_sha256: &'static str,
    effect_request: &'static str,
    effect: &'static [f64],
    effect_vector_sha256: &'static str,
    forward_request: &'static str,
    forward: &'static [f64],
    forward_vector_sha256: &'static str,
}

const CASES: &[PrimitiveCase] = &[
    PrimitiveCase {
        id: "AB-mechanism-empty",
        candidate: &[0, 1],
        mechanism: &[],
        cause_request: "0547a8ef542880e28a680b3f1f60a54e674159c3d818e5604502298955246f2f",
        cause: &[0.25, 0.25, 0.25, 0.25],
        cause_vector_sha256: "4b1223dd691162ee03d91bafb6bb2a9e175af92a3ac738c3b0489867abf84394",
        effect_request: "4c09985d102e0971e92ad5976bc17d88de0d779026dfab3a98fa7084099c292f",
        effect: &[0.39540478605678203, 0.3954047860567819, 0.10459521394321813, 0.10459521394321811],
        effect_vector_sha256: "6a7cbc63d9496daa8287509dcd360cc47cbaf4eb173f562c32ace9b9262b11b6",
        forward_request: "a6ca845a658cc699c41b2982b304ce0854c200c6edc6b4691fdff1ba01b56692",
        forward: &[1.0, 1.0, 1.0, 1.0],
        forward_vector_sha256: "85c527ba839ea75ea18a763deaec1ff10ffa23b16facc08b716332ed49d19eeb",
    },
    PrimitiveCase {
        id: "AB-mechanism-0",
        candidate: &[0, 1],
        mechanism: &[0],
        cause_request: "af0a1aaf95fe9c328e16da40848d71084874b9e81002c05848505bb716938694",
        cause: &[0.4403985389889411, 0.486701503211567, 0.013298496788432975, 0.05960146101105883],
        cause_vector_sha256: "c028b5a411093fae28304073b603699460d864c50cb166e222e70cf8d60a2646",
        effect_request: "a8638c31f5adeb7e03674b2384e4c1bb837711867ebbe572978d436b0d193994",
        effect: &[0.45220009985318743, 0.5445004826806572, 0.0014969359241867831, 0.0018024815419685801],
        effect_vector_sha256: "4b77879eaa109fe396513d4b2f000696f18221fc1fedf670ebf2f98c02aedee6",
        forward_request: "542301a83cb86febcb3860afb3755929aceb565c012df8118c5d31dcdbf7cce7",
        forward: &[0.8807970779778824, 0.9734030064231343, 0.026596993576865957, 0.11920292202211769],
        forward_vector_sha256: "e07032ab1f79327469d278b9863556dde87861ce35a1a4c11560c249238082e4",
    },
    PrimitiveCase {
        id: "AB-mechanism-1",
        candidate: &[0, 1],
        mechanism: &[1],
        cause_request: "a4a2a776ff13a6a86e6771129839999e961960d71850ab1fd26adafded47b176",
        cause: &[0.20286477717278842, 0.3835645368818352, 0.10557785616065384, 0.30799282978472253],
        cause_vector_sha256: "a986091d0cb3f0e40f1c6a0d3ce9405f0899f83d8d9381e5528a1efee238be08",
        effect_request: "125c5efe624f5e60ee3730809e41d56f3bb0227c045ba213a12179c4357ca7b0",
        effect: &[0.06442213912729394, 0.8192839846057849, 0.008477818672197886, 0.10781605759472342],
        effect_vector_sha256: "593fb9cbf81371f2cfd933214fcd98fceffa8bb9cedd124810c265b92b01edd4",
        forward_request: "05dbab255b52f878e28c08bf9d2cefbca573bf08533749a89c97bcf2a631a035",
        forward: &[0.4360890387838311, 0.8245309635881022, 0.2269558395085105, 0.6620779563853628],
        forward_vector_sha256: "8c8b665a43e5d253a2d992163c639787f2a9a957c5efccd1bebec060c9c723e4",
    },
    PrimitiveCase {
        id: "AB-mechanism-0-1",
        candidate: &[0, 1],
        mechanism: &[0, 1],
        cause_request: "eaffd44abaf437afcaa38dea1fff6c1892090c1039a29fe276dd292e8ec2859f",
        cause: &[0.3020496745917145, 0.6311418650784189, 0.004746803415466242, 0.06206165691440026],
        cause_vector_sha256: "1742051297ddaa37b31f61a38ec9921536ccc0aeabd5f3a2f6d0935d1dee4a7a",
        effect_request: "6b3dc532baca1f219b8361c6a7e910f1095f4d8ce7621a8b25ca431cf3d6db74",
        effect: &[0.026567403463145938, 0.9723200605039938, 2.9590113720017906e-05, 0.0010829459191403037],
        effect_vector_sha256: "0eeaf495e508f13428f8945b3b5d1d1d59f3177e9dec8d04b0b42927fea6acf4",
        forward_request: "f44788c73cc2d8901ebb65d460b1d51e5e4e8ea0f1f16d790f8254d431213e6e",
        forward: &[0.3841059510989819, 0.8026009188456226, 0.006036343005640075, 0.07892162700756744],
        forward_vector_sha256: "a16450ecedfd152e4e8fe871c2ebfbd1ea32c140a23dc3a83ec8c930270d8950",
    },
    PrimitiveCase {
        id: "ABC-mechanism-empty",
        candidate: &[0, 1, 2],
        mechanism: &[],
        cause_request: "f37b2b85c2b87e3fb4b5b2f8c16df713f6a85024c596416fa6b676aacf35a099",
        cause: &[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        cause_vector_sha256: "5e50b926711c213e4f276aae70f98276a9a8967310ced3622d506c65702edac9",
        effect_request: "86a55234087f3dd09a4104a41d46e2a002302c9e45350c28afcf4a067143b97c",
        effect: &[0.12500000000000003, 0.125, 0.12500000000000003, 0.125, 0.12500000000000003, 0.125, 0.12500000000000003, 0.125],
        effect_vector_sha256: "a13a9ea11d005b6f3155e1a79231451c880c2742888b249fc3f6caada23c2dc9",
        forward_request: "da787377fccddfea086c40e444d32e2ae73129503202d968a1ae0e3843e086ca",
        forward: &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        forward_vector_sha256: "b51aaae8fc647c443b6660db4f638fdb642a6198c40d7c9a279987ecad03fe4c",
    },
    PrimitiveCase {
        id: "ABC-mechanism-0",
        candidate: &[0, 1, 2],
        mechanism: &[0],
        cause_request: "02e0443b605834efe042f0095f4c7edd6b8fc9a1c237fd19d633e49130feecad",
        cause: &[0.2201992694944706, 0.24335075160578354, 0.006649248394216489, 0.02980073050552942, 0.2201992694944706, 0.24335075160578354, 0.006649248394216489, 0.02980073050552942],
        cause_vector_sha256: "e0e5a9cb43feaadfbeae2c7be8aeca20e8ed186214844dbdb2118cd2c4fa7676",
        effect_request: "0ea85e76bad4f109cb3f4ca17af861d1f0726842c597db1106d5c258d594d12b",
        effect: &[0.21329654291793157, 0.25683335897235676, 0.08886985355018158, 0.10700943712649227, 0.10696411609164538, 0.12879699243834708, 0.04456652321761563, 0.05366317568542966],
        effect_vector_sha256: "5c712ed03bbc0667dce3d1cd4424cf4549865ce842faf41c700867201feaf816",
        forward_request: "a149bb8b5c177edd21e956197c0bb302a558a7f9e69f502e0893eda74831cb94",
        forward: &[0.8807970779778824, 0.9734030064231342, 0.026596993576865957, 0.11920292202211769, 0.8807970779778824, 0.9734030064231342, 0.026596993576865957, 0.11920292202211769],
        forward_vector_sha256: "cfc5746b3a43a04d31f13f4b6edb3034a864672a1bccb569d2eb0d7f629f76bd",
    },
    PrimitiveCase {
        id: "ABC-mechanism-1",
        candidate: &[0, 1, 2],
        mechanism: &[1],
        cause_request: "7b2f3bea15301328283537eb173c701803ebe4b01d615c14586887700fec7d69",
        cause: &[0.19213119587475444, 0.24972186599178495, 0.10032808497188704, 0.24862842527513737, 0.0013715747248626011, 0.14967191502811297, 0.0002781340082150804, 0.05786880412524557],
        cause_vector_sha256: "f3ad3ae9485d4478acc3ecb364a1cbe0b487ff5fd6349b0fc5d82cbc2d177509",
        effect_request: "d32c19fa67fed4bdfba510d0db6cdb0eef38a82cac95d5aa570a0f359678dee7",
        effect: &[0.021611066796263432, 0.27483720901349407, 0.014838912103482479, 0.18871281208676008, 0.021611066796263432, 0.27483720901349407, 0.014838912103482479, 0.18871281208676008],
        effect_vector_sha256: "48b90211dbcc556f32e4260e84dacc2a43dbb1033c5305c647604b385f955151",
        forward_request: "3b0bc141a27d1cfde38dc74d18996de9fd6f215e81e7da421189347cda65ee9b",
        forward: &[0.7685247834990178, 0.9988874639671398, 0.40131233988754816, 0.9945137011005495, 0.0054862988994504045, 0.5986876601124519, 0.0011125360328603216, 0.23147521650098227],
        forward_vector_sha256: "b661f7ed28064b99435a85472f16d770a9822e699f169a3323666dccae3322a2",
    },
    PrimitiveCase {
        id: "ABC-mechanism-2",
        candidate: &[0, 1, 2],
        mechanism: &[2],
        cause_request: "b92f8befc1ccec94b08fb415a8ec00b342e751653de239bba6dcf13c0b442716",
        cause: &[0.04199540371651888, 0.125, 0.04199540371651888, 0.125, 0.125, 0.2080045962834811, 0.125, 0.2080045962834811],
        cause_vector_sha256: "22164ad3344e6783419e1f852237b89387e64d18e04e35d93ddff5be76d59773",
        effect_request: "86371372a633882fe3d93e9753067eb3b1f01a6f2b80327081f905a7a56c473b",
        effect: &[0.1320615637579922, 0.13206156375799216, 0.03493383995852675, 0.034933839958526745, 0.26334322229878987, 0.26334322229878976, 0.06966137398469138, 0.06966137398469135],
        effect_vector_sha256: "f46611880949eec5a50767d6625583e3c36c5503ed267cfa6eadbcf23a17ae28",
        forward_request: "d1d40eb5ff1a342cf85dbbd08f12be1f541d0e62ed655f6d040c34917f34aea0",
        forward: &[0.16798161486607552, 0.5, 0.16798161486607552, 0.5, 0.5, 0.8320183851339245, 0.5, 0.8320183851339245],
        forward_vector_sha256: "5ae3f41b159201dfb514048a0ac07e24b4d575a6edfd414a226487c09375cd95",
    },
    PrimitiveCase {
        id: "ABC-mechanism-0-1",
        candidate: &[0, 1, 2],
        mechanism: &[0, 1],
        cause_request: "6b04f6fe7ed7e688905d12a8d94edd0f848a636ac602a290a7bbbd8d40603a1d",
        cause: &[0.2827928329181155, 0.40620372539068755, 0.004459125735498252, 0.0495258944918256, 0.002018784613486837, 0.24346001592338715, 1.2361787970902402e-05, 0.011527259139028268],
        cause_vector_sha256: "99990810cbc5cfa4fe01a9587fa474e676c8cd4035a9f0a6c80533ee93f5b2fe",
        effect_request: "ae6524939e691388a5215601d9fca427679e7132a022a025aab1526f93e6b43c",
        effect: &[0.01414959683874498, 0.5178502623879117, 0.003564245378092197, 0.1304450879622134, 0.007095750818107886, 0.25969195199503126, 0.0017874005419208948, 0.06541570407797774],
        effect_vector_sha256: "deacc4926a880aab8a26761f3c007d34b80b54fd0b086da393b4ef49342819fc",
        forward_request: "5bc80de89598a60149e24ccf7ec309634cfa2a12f3d9df5e9fa9dc0cba67075a",
        forward: &[0.6769143836595195, 0.972320060503994, 0.010673701726306167, 0.11854893916221645, 0.004832316039549188, 0.5827643682618922, 2.9590113720017906e-05, 0.027592522182619397],
        forward_vector_sha256: "fe7a840fa189dea24854c1024f62719a639153027e9c5c8bb5e4d0d019bc8545",
    },
    PrimitiveCase {
        id: "ABC-mechanism-0-2",
        candidate: &[0, 1, 2],
        mechanism: &[0, 2],
        cause_request: "b9d58f4361c597e7fa1a1d7be762c62f19d83bf8f93bc08e37afe44355e44d99",
        cause: &[0.0717720905671446, 0.23609167148194274, 0.002167266308597497, 0.028911783629215426, 0.21363079115641734, 0.39286522249995004, 0.006450903303690005, 0.048110271053042504],
        cause_vector_sha256: "0dfb13bc86a7b9c0208d2ad68ba3d124929a849e5d8a08077f12f0aaeadaee0c",
        effect_request: "bc69dc917ea844931342632978807bea96aa01b759b5697b47c2af4f6de52ea5",
        effect: &[0.22610004992659372, 0.2722502413403286, 0.0007484679620933916, 0.0009012407709842901, 0.22610004992659372, 0.2722502413403286, 0.0007484679620933916, 0.0009012407709842901],
        effect_vector_sha256: "32ce61cc7584873aa92bffe8baf8a626f6e1b16fe9cd5e22c5f2487d1998a646",
        forward_request: "2c90c79f99c45f72670e3a6ce1f45e9dcbd04b9cbd0186825c80bc5eccb94c51",
        forward: &[0.14795771552804532, 0.4867015032115671, 0.004467805931624581, 0.05960146101105884, 0.4403985389889412, 0.8098891974886832, 0.013298496788432979, 0.09917902268408747],
        forward_vector_sha256: "ae7d9dbb09fde303007d641e4e2bcf458ff51ca65028c2480411779152167228",
    },
    PrimitiveCase {
        id: "ABC-mechanism-1-2",
        candidate: &[0, 1, 2],
        mechanism: &[1, 2],
        cause_request: "a9692fa5fed49822cbbb36509476807e56ad4bcac116bf604e50ea47832d2c65",
        cause: &[0.06840638816605557, 0.26464494219252294, 0.03572080990438317, 0.2634861587832332, 0.0014535383689065079, 0.2639430682827247, 0.0002947549596168566, 0.10205033934255715],
        cause_vector_sha256: "524927178cac443d76799d788385e2ef95052b0e241dfa1a34da198b65212d3b",
        effect_request: "395a15362aac4e6da2f935676a2614f7769ac22512d908aae1b5fb9f4402727d",
        effect: &[0.021516402263688398, 0.27363331953544257, 0.002831513503598256, 0.03600957213030858, 0.04290573686360554, 0.5456506650703423, 0.00564630516859963, 0.07180648546441484],
        effect_vector_sha256: "7364b5ceb7ac20989bd778e41649a9c9157a865a9cd652244607a7ba2bda44a5",
        forward_request: "32c2e4283247427acafa130bbd096a91bed45a10d1265103dcd27ad90f72a1ad",
        forward: &[0.12909803419676608, 0.4994437319835699, 0.06741309491999371, 0.49725685055027474, 0.0027431494497252022, 0.49811914016637004, 0.0005562680164301608, 0.1925916358316728],
        forward_vector_sha256: "6251eef683b0b3e444be0bcb80ef1678a9980157ab1cba10041363a0a4084644",
    },
    PrimitiveCase {
        id: "ABC-mechanism-0-1-2",
        candidate: &[0, 1, 2],
        mechanism: &[0, 1, 2],
        cause_request: "269132ddd98271b84d797fa089f4a16f3db3a8acbec2f9816c99acc99e23c58d",
        cause: &[0.09708809984555697, 0.4150971554996059, 0.0015309017564716857, 0.050610215126307936, 0.002062983912367557, 0.41399626212502294, 1.2632437131578487e-05, 0.019601749297535408],
        cause_vector_sha256: "c8ebef408cb70fbe653d7634461bb1b8a9f2f28ce6ec3226e553ab8b5dacb638",
        effect_request: "878718fc166839d6eb9f4f189b8770877404b6d004bbf77ef5772bf58678ad84",
        effect: &[0.013283701731572969, 0.4861600302519969, 1.4795056860008953e-05, 0.0005414729595701518, 0.013283701731572969, 0.4861600302519969, 1.4795056860008953e-05, 0.0005414729595701518],
        effect_vector_sha256: "955f9418e856e969ddd4c304680ad353bee489df1ee0e2dd697688b7c1e12d71",
        forward_request: "f62daa2dfff7b295303808591f7c62a614a70d1f9e24894ff0fd87b412c2e097",
        forward: &[0.11370917129320028, 0.486160030251997, 0.001792985652583728, 0.05927446958110823, 0.002416158019774594, 0.4848706685948512, 1.4795056860008953e-05, 0.02295748574815498],
        forward_vector_sha256: "9e5d394c10f0e5009e9d10345554c991d604099d9b25b6d74d48ba829ffd9d00",
    },
];

fn vector_sha256(values: &[f64]) -> String {
    let mut bytes = b"SYMTHAEA-IIT-PRIMITIVE-VECTOR\0v0.1\0".to_vec();
    for value in values {
        let canonical = if *value == 0.0 { 0.0 } else { *value };
        bytes.extend_from_slice(&canonical.to_bits().to_le_bytes());
    }
    let digest = Sha256::digest(&bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn assert_vector(case_id: &str, operation: &str, actual: &[f64], expected: &[f64], expected_sha: &str) {
    assert_eq!(actual.len(), expected.len(), "{case_id} {operation} vector length");
    assert_eq!(vector_sha256(expected), expected_sha, "{case_id} {operation} oracle vector digest");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{case_id} {operation}[{index}] non-finite");
        assert_eq!(*actual == 0.0, *expected == 0.0, "{case_id} {operation}[{index}] structural support");
        assert!((actual - expected).abs() <= 3e-12, "{case_id} {operation}[{index}] actual={actual:.17} expected={expected:.17}");
    }
}

#[test]
fn all_pinned_pyphi_iit4_2023_primitive_vectors_match_rust() {
    let model = BinaryCausalModel::new(3, TPM.to_vec(), CONNECTIVITY.to_vec()).unwrap();
    assert_eq!(model.model_sha256(), "6d67ba617b5d2ccbba711cb44f4753286a45ae6ded16fc1a544a77dc8e931ac1");
    assert_eq!(model.evaluation_sha256(&STATE).unwrap(), "1442d185bc5d2c23e3c6fc92e94f21d1c492cbe41a451cebd1d51fa36e862744");
    let mut operation_count = 0usize;
    for case in CASES {
        assert!(!case.candidate.is_empty());
        assert!(case.mechanism.iter().all(|unit| case.candidate.contains(unit)));
        for (operation, request, actual, expected, expected_sha, name) in [
            (PrimitiveRepertoireOperation::CauseRepertoire, case.cause_request,
                model.cause_repertoire_iit4(case.candidate, case.mechanism, &STATE, case.candidate).unwrap(),
                case.cause, case.cause_vector_sha256, "cause"),
            (PrimitiveRepertoireOperation::EffectRepertoire, case.effect_request,
                model.effect_repertoire_iit4(case.candidate, case.mechanism, &STATE, case.candidate).unwrap(),
                case.effect, case.effect_vector_sha256, "effect"),
            (PrimitiveRepertoireOperation::ForwardCauseLikelihood, case.forward_request,
                model.forward_cause_likelihood_iit4(case.candidate, case.mechanism, &STATE, case.candidate).unwrap(),
                case.forward, case.forward_vector_sha256, "forward-cause"),
        ] {
            let request_sha = primitive_repertoire_request_sha256(
                &model, &STATE, case.candidate, case.mechanism, case.candidate,
                FormalismProfile::Iit4_0_2023, operation,
            ).unwrap();
            assert_eq!(request_sha, request, "{} {name} request identity", case.id);
            assert_vector(case.id, name, &actual, expected, expected_sha);
            if name == "cause" || name == "effect" {
                let mass = actual.iter().sum::<f64>();
                assert!((mass - 1.0).abs() <= 1e-12, "{} {name} normalization={mass:.17}", case.id);
            }
            operation_count += 1;
        }
    }
    assert_eq!(CASES.len(), 12);
    assert_eq!(operation_count, 36);
}
