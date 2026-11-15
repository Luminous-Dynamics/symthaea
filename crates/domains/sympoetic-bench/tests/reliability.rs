use sympoetic_bench::reliability::{
    cohens_kappa_nominal, krippendorff_alpha_nominal,
};

#[test]
fn cohens_kappa_matches_known_nominal_example() {
    let rater_a = [Some("A"), Some("B"), Some("C"), Some("A")];
    let rater_b = [Some("A"), Some("B"), Some("C"), Some("B")];
    let kappa = cohens_kappa_nominal(&rater_a, &rater_b).unwrap().unwrap();
    assert!((kappa - 7.0 / 11.0).abs() < 1e-12);
}

#[test]
fn cohens_kappa_handles_pairwise_missing_labels() {
    let rater_a = [Some("A"), Some("B"), None, Some("C")];
    let rater_b = [Some("A"), None, Some("B"), Some("C")];
    assert_eq!(
        cohens_kappa_nominal(&rater_a, &rater_b).unwrap(),
        Some(1.0)
    );
}

#[test]
fn cohens_kappa_returns_unavailable_for_unidentifiable_agreement() {
    let no_overlap = [Some("A"), None];
    let no_overlap_b = [None, Some("B")];
    assert_eq!(
        cohens_kappa_nominal(&no_overlap, &no_overlap_b).unwrap(),
        None
    );

    let single_category = [Some("A"), Some("A"), Some("A")];
    assert_eq!(
        cohens_kappa_nominal(&single_category, &single_category).unwrap(),
        None
    );
}

#[test]
fn cohens_kappa_rejects_length_mismatch_and_empty_labels() {
    assert!(cohens_kappa_nominal(&[Some("A")], &[Some("A"), Some("B")]).is_err());
    assert!(cohens_kappa_nominal(&[Some("")], &[Some("A")]).is_err());
}

#[test]
fn krippendorff_alpha_matches_nominal_pairwise_disagreement_example() {
    let ratings = vec![
        vec![Some("A"), Some("A")],
        vec![Some("B"), Some("B")],
        vec![Some("A"), Some("B")],
        vec![Some("B"), Some("B")],
    ];
    let alpha = krippendorff_alpha_nominal(&ratings).unwrap().unwrap();
    assert!((alpha - (1.0 - 14.0 / 30.0)).abs() < 1e-12);
}

#[test]
fn krippendorff_alpha_is_one_for_perfect_agreement() {
    let ratings = vec![
        vec![Some("A"), Some("A"), Some("A")],
        vec![Some("B"), Some("B"), Some("B")],
        vec![Some("A"), Some("A"), Some("A")],
    ];
    assert_eq!(krippendorff_alpha_nominal(&ratings).unwrap(), Some(1.0));
}

#[test]
fn krippendorff_alpha_excludes_singleton_items_from_pairwise_marginals() {
    let pairable = vec![
        vec![Some("A"), Some("A")],
        vec![Some("B"), Some("B")],
        vec![Some("A"), Some("B")],
        vec![Some("B"), Some("B")],
    ];
    let with_singletons = vec![
        vec![Some("A"), Some("A")],
        vec![Some("B"), Some("B")],
        vec![Some("A"), Some("B")],
        vec![Some("B"), Some("B")],
        vec![Some("C"), None],
        vec![None, Some("D")],
    ];
    assert_eq!(
        krippendorff_alpha_nominal(&pairable).unwrap(),
        krippendorff_alpha_nominal(&with_singletons).unwrap()
    );
}

#[test]
fn krippendorff_alpha_preserves_worse_than_chance_agreement() {
    let ratings = vec![
        vec![Some("A"), Some("B")],
        vec![Some("A"), Some("B")],
    ];
    let alpha = krippendorff_alpha_nominal(&ratings).unwrap().unwrap();
    assert!((alpha + 0.5).abs() < 1e-12);
}

#[test]
fn krippendorff_alpha_returns_unavailable_without_pairable_units() {
    let ratings = vec![
        vec![Some("A"), None],
        vec![None, Some("B")],
        vec![Some("C"), None],
    ];
    assert_eq!(krippendorff_alpha_nominal(&ratings).unwrap(), None);
}

#[test]
fn krippendorff_alpha_rejects_ragged_matrices_and_empty_labels() {
    assert!(krippendorff_alpha_nominal(&[vec![Some("A")], vec![Some("A"), Some("B")]]).is_err());
    assert!(krippendorff_alpha_nominal(&[vec![Some(""), Some("A")]]).is_err());
}
