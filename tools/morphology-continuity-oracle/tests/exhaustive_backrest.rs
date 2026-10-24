use morphology_continuity_oracle::{
    Disposition, TargetEvidence, canonical_templates, evaluate_episode, template,
};

/// Exhaustively enumerate the 2^4 deletion patterns for the chair's four-cell
/// backrest. The ambiguity boundary must occur only when every backrest cell
/// has disappeared; intermediate shapes must not be misclassified as a table.
#[test]
fn every_backrest_deletion_subset_has_the_expected_identifiability_result() {
    let templates = canonical_templates();
    let chair = &template(&templates, "chair").expect("chair template").cells;
    let table = &template(&templates, "table").expect("table template").cells;
    let backrest: Vec<_> = chair.difference(table).copied().collect();

    assert_eq!(table.len(), 17);
    assert_eq!(chair.len(), 21);
    assert_eq!(backrest.len(), 4);

    let mut exact_table_alias_count = 0;
    let pattern_count = 1_u8 << backrest.len();

    for mask in 0_u8..pattern_count {
        let deleted_count = mask.count_ones() as usize;
        let mut observed = chair.clone();

        for (index, cell) in backrest.iter().enumerate() {
            if mask & (1_u8 << index) != 0 {
                assert!(observed.remove(cell));
            }
        }

        assert_eq!(observed.len(), chair.len() - deleted_count);
        let report = evaluate_episode(
            "chair",
            &observed,
            &templates,
            backrest.len(),
            TargetEvidence::Missing,
        );

        if observed == *table {
            exact_table_alias_count += 1;
            assert_eq!(
                mask,
                pattern_count - 1,
                "only all-backrest deletion should alias table"
            );
            assert_eq!(report.current_exact_matches, vec!["table"]);
            assert_eq!(report.compatible_histories, vec!["chair", "table"]);
            assert_eq!(report.disposition, Disposition::AbstainAmbiguousHistory);
        } else if mask == 0 {
            assert_eq!(report.current_exact_matches, vec!["chair"]);
            assert_eq!(report.compatible_histories, vec!["chair"]);
            assert_eq!(
                report.disposition,
                Disposition::AbstainInvalidTargetEvidence
            );
        } else {
            assert!(
                report.current_exact_matches.is_empty(),
                "partial backrest deletion mask {mask:04b} must not match an intact template"
            );
            assert_eq!(report.compatible_histories, vec!["chair"]);
            assert_eq!(
                report.disposition,
                Disposition::AbstainInvalidTargetEvidence
            );
        }
    }

    assert_eq!(pattern_count, 16);
    assert_eq!(exact_table_alias_count, 1);
}
