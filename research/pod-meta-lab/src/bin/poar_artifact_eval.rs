use pod_meta_lab::graph::{evaluate_poar_artifacts_v1, CaseScore, DatasetSplit};
use std::{env, fs, path::Path};

const K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

fn sha256(input: &[u8]) -> String {
    let mut bytes = input.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    bytes.push(0x80);
    while bytes.len() % 64 != 56 { bytes.push(0); }
    bytes.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667u32,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    ];
    for chunk in bytes.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0],word[1],word[2],word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }
        let (mut a,mut b,mut c,mut d,mut e,mut f,mut g,mut hh) =
            (h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh=g; g=f; f=e; e=d.wrapping_add(t1); d=c; c=b; b=a; a=t1.wrapping_add(t2);
        }
        for (slot,value) in h.iter_mut().zip([a,b,c,d,e,f,g,hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = String::with_capacity(64);
    for word in h { use std::fmt::Write as _; write!(&mut out, "{word:08x}").expect("String write"); }
    out
}

fn sum(report: &pod_meta_lab::graph::GraphEvaluationReport, f: fn(&CaseScore) -> usize) -> usize {
    report.case_scores.iter().map(f).sum()
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 6 {
        return Err(format!("usage: {} <planner> <oracle> <predictions> <evaluator-revision> <receipt>", args[0]));
    }
    let planner_path=Path::new(&args[1]); let oracle_path=Path::new(&args[2]);
    let predictions_path=Path::new(&args[3]); let receipt_path=Path::new(&args[5]);
    if planner_path==oracle_path || planner_path==predictions_path || planner_path==receipt_path
        || oracle_path==predictions_path || oracle_path==receipt_path || predictions_path==receipt_path {
        return Err("artifact and receipt paths must be distinct".into());
    }
    let planner=fs::read(planner_path).map_err(|e|format!("read planner: {e}"))?;
    let oracle=fs::read(oracle_path).map_err(|e|format!("read oracle: {e}"))?;
    let predictions=fs::read(predictions_path).map_err(|e|format!("read predictions: {e}"))?;
    let planner_sha=sha256(&planner); let oracle_sha=sha256(&oracle); let predictions_sha=sha256(&predictions);
    let evaluated=evaluate_poar_artifacts_v1(&planner,&oracle,&predictions,&planner_sha)?;
    let report=&evaluated.report;
    let targets=sum(report,|s|s.target_claims); let correct=sum(report,|s|s.correct_dispositions);
    let false_supported=report.false_supported_claims(); let false_refuted=report.false_refuted_claims();
    let invalid=report.invalid_citations(); let relevant=sum(report,|s|s.temporal_relevant_evidence);
    let found=sum(report,|s|s.temporal_relevant_retrieved_at_k);
    let retrieved=sum(report,|s|s.temporal_retrieved_at_k);
    let coverage=if relevant==0{"-".into()}else{format!("{:.6}",found as f64/relevant as f64)};
    let precision=if retrieved==0{"-".into()}else{format!("{:.6}",found as f64/retrieved as f64)};
    let split=match evaluated.prediction_identity.split {
        DatasetSplit::Development=>"dev",DatasetSplit::Calibration=>"cal",DatasetSplit::Test=>"test"
    };
    let output=format!(
        "POAR-EVAL-RECEIPT/1\nevaluator_revision={}\nplanner_sha256={}\noracle_sha256={}\npredictions_sha256={}\nmodel_id={}\nseed={}\nsplit={}\ncases={}\ntarget_claims={}\ncorrect_dispositions={}\nfalse_supported_claims={}\nfalse_refuted_claims={}\ninvalid_citations={}\ntemporal_coverage_at_3={}\ntemporal_precision_at_3={}\nhard_invariants={}\nreceipt_signed=false\n",
        args[4],planner_sha,oracle_sha,predictions_sha,evaluated.prediction_identity.model_id,
        evaluated.prediction_identity.seed,split,report.case_count(),targets,correct,
        false_supported,false_refuted,invalid,coverage,precision,report.all_hard_invariants_hold()
    );
    fs::write(receipt_path,output.as_bytes()).map_err(|e|format!("write receipt: {e}"))?;
    print!("{output}");
    Ok(())
}

fn main() {
    if let Err(error)=run() { eprintln!("poar-artifact-eval: {error}"); std::process::exit(1); }
}

#[cfg(test)]
mod sha256_tests {
    use super::sha256;

    #[test]
    fn sha256_matches_empty_input_vector() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_matches_abc_vector() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
