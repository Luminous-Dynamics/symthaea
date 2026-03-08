//! # Currency Mint (Currency Factory) Integration Tests
//!
//! 56 sweettest scenarios covering all 30 coordinator externs.
//!
//! All coordinator guard/branch paths are tested — suite is COMPLETE.
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --release --test currency_mint_test -- --include-ignored --test-threads=1 --nocapture
//! ```
//!
//! ## Extern → Test Coverage Matrix (30/30)
//!
//! ### Lifecycle (10 externs) → lifecycle.rs
//!   create_currency          → 1.1, 1.2, 1.3
//!   activate_currency        → 1.1, 1.2, 14.2
//!   suspend_currency         → 1.2, 14.2
//!   reactivate_currency      → 1.2, 14.2
//!   retire_currency          → 1.2, 1.3
//!   get_currency             → 1.1
//!   get_dao_currencies       → 5.1 (balances), E11 (balances)
//!   list_active_currencies   → 10.1 (discovery)
//!   search_currencies        → 10.1 (discovery)
//!   amend_currency_params    → 9.1 (disputes), E10 (disputes), E12 (disputes)
//!
//! ### Exchanges (8 externs) → exchanges.rs
//!   record_minted_exchange   → 2.1, 2.2, 12.1, E1, E2, E4, E5, E14, E15, E16, E17, E20
//!   confirm_minted_exchange  → 3.1, 3.2, 13.1, E13, E18, E24
//!   list_pending_exchanges   → 13.1
//!   list_pending_for_receiver → 13.1
//!   cancel_expired_exchange  → 14.1 (lifecycle), E26, E28
//!   get_exchange             → 15.1
//!   get_currency_exchanges   → 18.2
//!   get_member_exchanges     → 7.1 (balances), 18.2
//!
//! ### Balances (2 externs) → balances.rs
//!   get_minted_balance       → 2.1, 3.1, 13.1, 16.1 (demurrage), E18, E23
//!   get_member_portfolio     → 11.1, 18.3
//!
//! ### Stats (4 externs) → stats.rs, balances.rs, demurrage.rs
//!   get_currency_stats       → 4.1, 17.1, E8, E9 (demurrage)
//!   list_currency_members    → 11.1, E6 (balances)
//!   get_demurrage_report     → 16.1, 18.1 (demurrage)
//!   get_compost_balance      → 6.1, 16.1, E7 (demurrage)
//!
//! ### Demurrage (3 externs) → demurrage.rs
//!   apply_minted_demurrage   → 16.1, E7, E23
//!   apply_demurrage_all      → 18.1, E9, E23
//!   redistribute_compost     → 16.2, E8, E25, E29
//!
//! ### Disputes (3 externs) → disputes.rs
//!   open_minted_dispute      → 8.1, E3, E19, E21, E27
//!   resolve_minted_dispute   → 8.2, E22
//!   get_dispute              → 8.2

mod common;

mod balances;
mod demurrage;
mod discovery;
mod disputes;
mod exchanges;
mod lifecycle;
mod stats;
