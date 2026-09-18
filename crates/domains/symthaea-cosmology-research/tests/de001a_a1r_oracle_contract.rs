use serde_json::Value;

const A0: &str = include_str!("../references/de001a_a0_artifacts_v1.json");
const A1R: &str = include_str!("../references/de001a_a1r_oracle_v1.json");

fn a0_artifact<'a>(a0: &'a Value, role: &str) -> &'a Value {
    a0["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["role"] == role)
        .unwrap()
}

#[test]
fn a1r_point_and_authority_are_frozen() {
    let a1r: Value = serde_json::from_str(A1R).unwrap();
    assert_eq!(a1r["schema_version"], 1);
    assert_eq!(a1r["protocol"], "DE-001A1R-RUST-ORACLE-v1");
    assert_eq!(a1r["scientific_claim"], "NONE");
    assert_eq!(a1r["authority"], "fixed-point-reproduction-sanity-only");
    assert_eq!(a1r["parameters"]["omega_m"], 0.29717787);
    assert_eq!(a1r["parameters"]["h_r_d_mpc"], 101.54786);
    assert_eq!(a1r["reference"]["chi2_bao"], 10.282299);
    assert_eq!(a1r["reference"]["absolute_tolerance"], 0.01);
    assert_eq!(a1r["numerics"]["simpson_subdivisions"], 1024);
}

#[test]
fn a1r_sources_are_exactly_bound_to_a0() {
    let a0: Value = serde_json::from_str(A0).unwrap();
    let a1r: Value = serde_json::from_str(A1R).unwrap();

    let mean = a0_artifact(&a0, "dataset-mean");
    assert_eq!(a1r["data"]["mean_size"], mean["expected_size"]);
    assert_eq!(a1r["data"]["mean_sha256"], mean["sha256"]);

    let covariance = a0_artifact(&a0, "dataset-covariance");
    assert_eq!(a1r["data"]["covariance_size"], covariance["expected_size"]);
    assert_eq!(a1r["data"]["covariance_sha256"], covariance["sha256"]);

    let bestfit = a0_artifact(&a0, "reference-bestfit-text");
    assert_eq!(a1r["source"]["bestfit_size"], bestfit["expected_size"]);
    assert_eq!(a1r["source"]["bestfit_sha256"], bestfit["sha256"]);
}

#[test]
fn a1r_independence_claim_is_narrow() {
    let a1r: Value = serde_json::from_str(A1R).unwrap();
    assert_eq!(a1r["independence"]["measurement_data_independent"], false);
    assert_eq!(a1r["independence"]["covariance_independent"], false);
    assert_eq!(
        a1r["independence"]["background_implementation_independent"],
        true
    );
    assert_eq!(
        a1r["independence"]["gaussian_likelihood_implementation_independent"],
        true
    );
}
