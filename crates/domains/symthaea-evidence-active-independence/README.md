# symthaea-evidence-active-independence

Exact-use composition for ASSURE-015.

This crate combines two opaque, lifecycle-current `ActiveVerifierProfile` capabilities with the exact verifier-profile records and exact relation-aware independence evidence that those records bind.

The compositor recomputes common-cause separation rather than trusting a persisted `Separated` flag. A qualification is minted only when:

- both active capabilities bind the exact supplied profile records;
- both records and capabilities bind the exact supplied graph and relation-completeness map;
- both lifecycle assessments were performed at the exact composition use-time;
- the exact relation-aware policy recomputes to `Separated`.

The resulting `ActiveIndependenceQualification` is deliberately non-serializable. The serializable report is evidence only and grants no physical authority.

Left/right ordering is intentionally preserved in qualification identity so downstream protocols can assign distinct roles such as campaign verifier and obligation verifier without role-swapping ambiguity.
