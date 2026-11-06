//! Deterministic update scenario for the public CEB top-k development fixture.

use ceb_topk_core::TopK;
use ceb_topk_rank_contract::RankedItem;

/// Update the same key twice with different scores.
///
/// Under the current intentionally defective source, both versions can remain
/// in the returned top-k set. This function is a source-level scenario only;
/// it has not been compiled or executed.
pub fn stale_update_scenario() -> Vec<RankedItem> {
    let mut topk = TopK::new(4);
    topk.upsert(RankedItem {
        score: 99,
        key: "account-7".to_owned(),
    });
    topk.upsert(RankedItem {
        score: 12,
        key: "account-7".to_owned(),
    });
    topk.top_k()
}
