# Mycelix Finance — Link Topology

Complete mapping of all DHT link types, anchor key patterns, entry connections, and coordinator functions across the 7 zomes in mycelix-finance.

---

## Link Key Format Reference

Quick lookup of all anchor key patterns used across zomes.

| Anchor Key Format | Zome | LinkType | Purpose |
|---|---|---|---|
| `"sap:{member_did}"` | payments | DidToSapBalance | Member SAP balance |
| `"{mint_id}"` | payments | MintIdToMintRecord | Mint record by ID |
| `"mints:{recipient_did}"` | payments | DidToMintRecords | Member's mint history |
| `"{from_did}"` | payments | SenderToPayments | Sender's payment history |
| `"{to_did}"` | payments | ReceiverToPayments | Receiver's payment history |
| `"payment:{from_did}:{timestamp}"` | payments | PaymentIdToPayment | Payment by ID |
| `"{channel_id}"` | payments | ChannelIdToChannel | Payment channel by ID |
| `"{party_a_did}"` | payments | ChannelPartyA | Party A's channels |
| `"{party_b_did}"` | payments | ChannelPartyB | Party B's channels |
| `"{member_did}"` | payments | MemberToExitRecord | Member's exit record |
| `"hearth-sap:{hearth_did}"` | payments | HearthDidToSapPool | Hearth SAP pool |
| `"provider:{dao_did}:{provider_did}"` | tend | ProviderToExchanges | Provider's exchanges in a DAO |
| `"receiver:{dao_did}:{receiver_did}"` | tend | ReceiverToExchanges | Receiver's exchanges in a DAO |
| `"balance:{dao_did}:{member_did}"` | tend | MemberToBalance | Member's TEND balance in a DAO |
| `"dao:{dao_did}"` | tend | DaoToExchanges | All exchanges in a DAO |
| `"exchange:{exchange_id}"` | tend | ExchangeIdToExchange | Exchange by ID |
| `"listings:{dao_did}"` | tend | DaoToListings | Service listings in a DAO |
| `"requests:{dao_did}"` | tend | DaoToRequests | Service requests in a DAO |
| `"my_listings:{provider_did}"` | tend | ProviderToListings | Provider's listings |
| `"category:{dao_did}:{category}"` | tend | CategoryToListings | Listings by category |
| `"rating:{exchange_id}"` | tend | ExchangeToRating | Rating for an exchange |
| `"ratings_for:{provider_did}"` | tend | ExchangeToRating | All ratings for a provider |
| `"dispute_for_exchange:{exchange_id}"` | tend | ExchangeToDispute | Dispute for an exchange |
| `"dispute:{dispute_id}"` | tend | ExchangeToDispute | Dispute by ID |
| `"disputes_for:{member_did}"` | tend | MemberToDisputes | Member's disputes |
| `"bilateral:{dao_a}:{dao_b}"` | tend | DaoToBilateralBalance | Inter-DAO bilateral balance |
| `"settlements:{dao_a}:{dao_b}"` | tend | SettlementRegistry | Settlement records between DAOs |
| `"tend:oracle_state"` | tend | AnchorLinks | Oracle state singleton |
| `"tier:apprentice:{member_did}"` | tend | AnchorLinks | Apprentice tier marker |
| `"hearth-balance:{hearth_did}:{member_did}"` | tend | HearthToBalances | Hearth TEND balance |
| `"my-hearths:{member_did}"` | tend | MemberToHearthBalance | Member's hearth balances |
| `"alias:{dao_did}"` | tend | DaoToAlias | Currency alias for a DAO |
| `GOVERNANCE_AGENTS_ANCHOR` | tend | GovernanceAgents | Governance agent registry |
| `"recognizer:{did}"` | recognition | RecognizerToEvents | Events by recognizer |
| `"recipient:{did}"` | recognition | RecipientToEvents | Events by recipient |
| `"cycle:{cycle_id}"` | recognition | CycleToEvents | Events in a recognition cycle |
| `"mycel:{did}"` | recognition | MemberToMycelState | Member's MYCEL score |
| `"alloc:{did}:{cycle_id}"` | recognition | RecognizerCycleToAllocation | Allocation for a recognizer in a cycle |
| `GOVERNANCE_AGENTS_ANCHOR` | recognition | GovernanceAgents | Governance agent registry |
| `"staker:{did}"` | staking | StakerToStake | Member's stakes |
| `"active_stakes"` | staking | ActiveStakes | All active stakes |
| `"{stake_id}"` | staking | StakeIdToStake | Stake by ID |
| `"stake:{stake_id}"` | staking | StakeToSlashing | Slashing events for a stake |
| `"depositor:{did}"` | staking | DepositorToEscrow | Escrows by depositor |
| `"beneficiary:{did}"` | staking | BeneficiaryToEscrow | Escrows by beneficiary |
| `"{escrow_id}"` | staking | EscrowIdToEscrow | Escrow by ID |
| `GOVERNANCE_AGENTS_ANCHOR` | staking | GovernanceAgents | Governance agent registry |
| `"{treasury_id}"` | treasury | TreasuryIdToTreasury | Treasury by ID |
| `"{manager_did}"` | treasury | ManagerToTreasury | Treasuries by manager |
| `"{treasury_id}"` | treasury | TreasuryToContributions | Contributions to a treasury |
| `"{contributor_did}"` | treasury | ContributorToContributions | Contributions by a member |
| `"{treasury_id}"` | treasury | TreasuryToAllocations | Allocations from a treasury |
| `"{allocation_id}"` | treasury | AllocationIdToAllocation | Allocation by ID |
| `"{pool_id}"` | treasury | PoolIdToPool | Savings pool by ID |
| `"{treasury_id}"` | treasury | TreasuryToPools | Pools under a treasury |
| `"{member_did}"` | treasury | MemberToPool | Member's pools |
| `"{pool_id}"` | treasury | CommonsPoolIdToPool | Commons pool by ID |
| `"{dao_did}"` | treasury | DaoToCommonsPool | DAO's commons pool |
| `"{commons_pool_id}"` | treasury | CommonsPoolToCompost | Compost receivals for a pool |
| `"dao-currencies:{dao_did}"` | currency-mint | DaoToCurrencies | DAO's minted currencies |
| `"all-active-currencies"` | currency-mint | DaoToCurrencies | Global currency index |
| `"currency:{currency_id}"` | currency-mint | CurrencyIdToDefinition | Currency by ID |
| `"mex:{currency_id}"` | currency-mint | CurrencyToExchanges | Exchanges for a currency |
| `"mex-id:{exchange_id}"` | currency-mint | CurrencyToExchanges | Minted exchange by ID |
| `"receiver-pending:{receiver_did}"` | currency-mint | CurrencyToExchanges | Pending exchanges for receiver |
| `"mbal:{currency_id}:{member_did}"` | currency-mint | CurrencyMemberToBalance | Member's minted currency balance |
| `"member-currencies:{member_did}"` | currency-mint | AnchorLinks | Member's currency memberships |
| `"dispute:{exchange_id}"` | currency-mint | ExchangeToDispute | Dispute for a minted exchange |
| `"{from_did}"` | bridge | DidToPayments | Cross-hApp payment by DID |
| `"{source_happ}"` | bridge | HappToPayments | Cross-hApp payments by source |
| `"collateral_registry"` | bridge | CollateralRegistry | All collateral registrations |
| `"recent_events"` | bridge | RecentEvents | Recent finance bridge events |
| `"{depositor_did}"` | bridge | DidToDeposits | Collateral deposits by depositor |
| `"{deposit_id}"` | bridge | DepositIdToDeposit | Collateral deposit by ID |

---

## Bridge Zome

**Integrity**: `bridge_integrity` | **Coordinator**: `finance_bridge`

### Entry Types

| Entry Type | Description |
|---|---|
| CrossHappPayment | Payment record bridging across hApps |
| CollateralRegistration | External collateral registration |
| FinanceBridgeEvent | Event log for cross-cluster notifications |
| CollateralBridgeDeposit | Collateral deposit for SAP backing |

### Link Types (6)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| DidToPayments | `"{from_did}"` | Anchor → CrossHappPayment | `record_cross_happ_payment` | `get_cross_happ_payments` |
| HappToPayments | `"{source_happ}"` | Anchor → CrossHappPayment | `record_cross_happ_payment` | — |
| CollateralRegistry | `"collateral_registry"` | Anchor → CollateralRegistration | `register_collateral` | `get_collateral_registry` |
| RecentEvents | `"recent_events"` | Anchor → FinanceBridgeEvent | `broadcast_finance_event` | `get_recent_events` |
| DidToDeposits | `"{depositor_did}"` | Anchor → CollateralBridgeDeposit | `deposit_collateral` | `get_deposits` |
| DepositIdToDeposit | `"{deposit_id}"` | Anchor → CollateralBridgeDeposit | `deposit_collateral` | `get_deposit`, `redeem_collateral` |

---

## Currency-Mint Zome

**Integrity**: `currency_mint_integrity` | **Coordinator**: `currency_mint`

### Entry Types

| Entry Type | Description |
|---|---|
| CurrencyDefinition | DAO-minted community currency definition |
| MintedBalance | Member's balance in a minted currency |
| MintedExchange | Exchange record for a minted currency |
| MintedExchangeConfirmation | Confirmation of a minted exchange |
| MintedDispute | Dispute on a minted exchange |
| Anchor | DHT anchor placeholder |

### Link Types (7)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| DaoToCurrencies | `"dao-currencies:{dao_did}"`, `"all-active-currencies"` | Anchor → CurrencyDefinition | `create_currency` (lifecycle.rs) | `get_dao_currencies`, `get_all_currencies` (lifecycle.rs) |
| CurrencyIdToDefinition | `"currency:{currency_id}"` | Anchor → CurrencyDefinition | `create_currency` (lifecycle.rs) | `get_currency_inner` (helpers.rs) |
| CurrencyMemberToBalance | `"mbal:{currency_id}:{member_did}"` | Anchor → MintedBalance | `mutate_balance` (helpers.rs) | `get_balance_inner`, `get_my_balances` (helpers.rs, balances.rs) |
| CurrencyToExchanges | `"mex:{currency_id}"`, `"mex-id:{exchange_id}"`, `"receiver-pending:{receiver_did}"` | Anchor → MintedExchange | `record_minted_exchange` (exchanges.rs) | `get_currency_exchanges`, `find_minted_exchange` (helpers.rs, exchanges.rs) |
| ExchangeToConfirmation | (direct ActionHash link) | MintedExchange → MintedExchangeConfirmation | `confirm_minted_exchange` (exchanges.rs) | — |
| ExchangeToDispute | `"dispute:{exchange_id}"` | Anchor → MintedDispute | `open_minted_dispute` (disputes.rs) | `get_dispute` (disputes.rs) |
| AnchorLinks | `"member-currencies:{member_did}"` | Anchor → CurrencyDefinition | `mutate_balance` (helpers.rs) | `get_my_balances` (balances.rs) |

---

## Payments Zome

**Integrity**: `payments_integrity` | **Coordinator**: `payments`

### Entry Types

| Entry Type | Description |
|---|---|
| Payment | SAP or TEND payment record |
| PaymentChannel | Bidirectional payment channel |
| Receipt | Signed payment receipt |
| SapBalance | Member's on-chain SAP balance with demurrage |
| SapMintRecord | Immutable governance mint audit trail |
| ExitRecord | Member exit record (MYCEL dissolution + SAP succession + TEND forgiveness) |
| HearthSapPool | Shared household SAP fund |

### Link Types (12)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| DidToSapBalance | `"sap:{member_did}"` | Anchor → SapBalance | `initialize_sap_balance`, `credit_sap` | `find_sap_balance_record` (internal), `get_sap_balance`, `apply_demurrage`, `debit_sap`, `contribute_to_hearth_pool`, `withdraw_from_hearth_pool` |
| MintIdToMintRecord | `"{mint_id}"` | Anchor → SapMintRecord | `mint_sap_from_governance` | — |
| DidToMintRecords | `"mints:{recipient_did}"` | Anchor → SapMintRecord | `mint_sap_from_governance` | `get_mint_records` |
| SenderToPayments | `"{from_did}"` | Anchor → Payment | `send_payment`, `refund_payment`, `create_escrow` | `get_payment_history` |
| ReceiverToPayments | `"{to_did}"` | Anchor → Payment | `send_payment`, `refund_payment`, `create_escrow` | `get_payment_history` |
| PaymentIdToPayment | `"{payment_id}"` | Anchor → Payment | `send_payment` | `get_payment`, `get_payment_record` (internal), `get_receipt` |
| PaymentToReceipt | (direct ActionHash link) | Payment → Receipt | `send_payment` | `get_receipt` |
| ChannelPartyA | `"{party_a_did}"` | Anchor → PaymentChannel | `open_payment_channel` | `get_channels` |
| ChannelPartyB | `"{party_b_did}"` | Anchor → PaymentChannel | `open_payment_channel` | `get_channels` |
| ChannelIdToChannel | `"{channel_id}"` | Anchor → PaymentChannel | `open_payment_channel` | `get_channel_record` (internal), `channel_transfer`, `close_payment_channel` |
| MemberToExitRecord | `"{member_did}"` | Anchor → ExitRecord | `initiate_exit` | — |
| HearthDidToSapPool | `"hearth-sap:{hearth_did}"` | Anchor → HearthSapPool | `get_or_create_hearth_pool` (internal) | `get_hearth_sap_pool`, `contribute_to_hearth_pool`, `withdraw_from_hearth_pool`, `apply_hearth_demurrage` |

---

## Recognition Zome

**Integrity**: `recognition_integrity` | **Coordinator**: `recognition`

### Entry Types

| Entry Type | Description |
|---|---|
| RecognitionEvent | Peer recognition event (gratitude token) |
| MemberMycelState | Member's MYCEL score (soulbound reputation) |
| RecognitionAllocation | Per-cycle recognition allocation |
| Anchor | DHT anchor placeholder |

### Link Types (7)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| RecognizerToEvents | `"recognizer:{did}"` | Anchor → RecognitionEvent | `recognize_member` | — |
| RecipientToEvents | `"recipient:{did}"` | Anchor → RecognitionEvent | `recognize_member` | `get_recognitions_for` |
| CycleToEvents | `"cycle:{cycle_id}"` | Anchor → RecognitionEvent | `recognize_member` | — |
| MemberToMycelState | `"mycel:{did}"` | Anchor → MemberMycelState | `update_mycel_score` | `get_mycel_score` |
| RecognizerCycleToAllocation | `"alloc:{did}:{cycle_id}"` | Anchor → RecognitionAllocation | `allocate_recognition` | `get_allocation` |
| AnchorLinks | — | — | — | — |
| GovernanceAgents | `GOVERNANCE_AGENTS_ANCHOR` | Anchor → AgentPubKey | `register_governance_agent` | `verify_governance_agent` |

---

## Staking Zome

**Integrity**: `staking_integrity` | **Coordinator**: `staking`

### Entry Types

| Entry Type | Description |
|---|---|
| CollateralStake | SAP-backed collateral stake |
| SlashingEvent | Stake slashing penalty |
| CryptoEscrow | Trustless crypto escrow contract |
| RewardDistribution | Epoch-based reward distribution |

### Link Types (9)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| StakerToStake | `"staker:{did}"` | Anchor → CollateralStake | `create_stake` | `get_stakes_for_member` |
| ActiveStakes | `"active_stakes"` | Anchor → CollateralStake | `create_stake` | `get_active_stakes` |
| StakeIdToStake | `"{stake_id}"` | Anchor → CollateralStake | `create_stake` | `get_stake`, `slash_stake`, `release_stake` |
| StakeToSlashing | `"stake:{stake_id}"` | Anchor → SlashingEvent | `slash_stake` | `get_slashing_events` |
| DepositorToEscrow | `"depositor:{did}"` | Anchor → CryptoEscrow | `create_escrow` | `get_escrows_for_depositor` |
| BeneficiaryToEscrow | `"beneficiary:{did}"` | Anchor → CryptoEscrow | `create_escrow` | `get_escrows_for_beneficiary` |
| EscrowIdToEscrow | `"{escrow_id}"` | Anchor → CryptoEscrow | `create_escrow` | `get_escrow`, `release_escrow`, `refund_escrow` |
| EpochToRewards | — | — | — | — |
| GovernanceAgents | `GOVERNANCE_AGENTS_ANCHOR` | Anchor → AgentPubKey | `register_governance_agent` | `verify_governance_agent` |

---

## TEND Zome

**Integrity**: `tend_integrity` | **Coordinator**: `tend`

### Entry Types

| Entry Type | Description |
|---|---|
| TendExchange | Time exchange record (mutual credit) |
| TendBalance | Member's TEND balance in a DAO |
| ServiceListing | Service offer (marketplace) |
| ServiceRequest | Service request (marketplace) |
| QualityRating | 1-5 star rating on a confirmed exchange |
| DisputeCase | Dispute with 3-stage escalation |
| OracleState | Network vitality oracle (drives dynamic limits) |
| BilateralBalance | Inter-DAO TEND clearing balance |
| BilateralSettlement | Two-phase settlement of bilateral balance |
| HearthTendBalance | Hearth-scoped TEND balance |
| CurrencyAliasEntry | Cultural alias for TEND currency |
| Anchor | DHT anchor placeholder |

### Link Types (19)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| ProviderToExchanges | `"provider:{dao_did}:{provider_did}"` | Anchor → TendExchange | `record_exchange` | `get_my_exchanges` |
| ReceiverToExchanges | `"receiver:{dao_did}:{receiver_did}"` | Anchor → TendExchange | `record_exchange` | `get_my_exchanges` |
| MemberToBalance | `"balance:{dao_did}:{member_did}"` | Anchor → TendBalance | `get_or_create_balance` (internal) | `find_balance`, `update_balance_after_exchange`, `get_balance` |
| DaoToExchanges | `"dao:{dao_did}"` | Anchor → TendExchange | `record_exchange` | — |
| DaoToListings | `"listings:{dao_did}"` | Anchor → ServiceListing | `create_listing` | `get_dao_listings` |
| DaoToRequests | `"requests:{dao_did}"` | Anchor → ServiceRequest | `create_request` | `get_dao_requests` |
| ProviderToListings | `"my_listings:{provider_did}"` | Anchor → ServiceListing | `create_listing` | — |
| CategoryToListings | `"category:{dao_did}:{category}"` | Anchor → ServiceListing | `create_listing` | — |
| ExchangeIdToExchange | `"exchange:{exchange_id}"` | Anchor → TendExchange | `record_exchange`, `record_hearth_exchange` | `find_exchange_by_id`, `update_exchange_entry`, `confirm_exchange`, `dispute_exchange`, `cancel_exchange` |
| AnchorLinks | `"tend:oracle_state"`, `"tier:apprentice:{member_did}"` | Anchor → OracleState / Anchor | `update_oracle_state` | `read_current_tier`, `get_oracle_state`, `get_dynamic_tend_limit`, `get_effective_limit_for_member` |
| ExchangeToRating | `"rating:{exchange_id}"`, `"ratings_for:{provider_did}"` | Anchor → QualityRating | `rate_exchange` | `rate_exchange` (duplicate check), `get_validation_score` |
| MemberToDisputes | `"disputes_for:{member_did}"` | Anchor → DisputeCase | `open_dispute` | — |
| ExchangeToDispute | `"dispute_for_exchange:{exchange_id}"`, `"dispute:{dispute_id}"` | Anchor → DisputeCase | `open_dispute` | `open_dispute` (duplicate check), `find_dispute_by_id`, `escalate_dispute`, `resolve_dispute` |
| DaoToBilateralBalance | `"bilateral:{dao_a}:{dao_b}"` | Anchor → BilateralBalance | `record_cross_dao_exchange` | `get_bilateral_balance`, `settle_bilateral_balance` |
| SettlementRegistry | `"settlements:{dao_a}:{dao_b}"` | Anchor → BilateralSettlement | `settle_bilateral_balance` | — |
| GovernanceAgents | `GOVERNANCE_AGENTS_ANCHOR` | Anchor → AgentPubKey | `register_governance_agent` | `verify_governance_or_bootstrap` |
| HearthToBalances | `"hearth-balance:{hearth_did}:{member_did}"` | Anchor → HearthTendBalance | `get_or_create_hearth_balance` (internal) | `get_hearth_balance`, `update_hearth_balance` |
| MemberToHearthBalance | `"my-hearths:{member_did}"` | Anchor → HearthTendBalance | `get_or_create_hearth_balance` (internal) | — |
| DaoToAlias | `"alias:{dao_did}"` | Anchor → CurrencyAliasEntry | `register_currency_alias` | `get_currency_alias` |

---

## Treasury Zome

**Integrity**: `treasury_integrity` | **Coordinator**: `treasury`

### Entry Types

| Entry Type | Description |
|---|---|
| Treasury | Multi-sig treasury account |
| Contribution | Member contribution to a treasury |
| Allocation | Governance-approved treasury allocation |
| SavingsPool | Member savings pool |
| CommonsPool | DAO-level commons pool (receives compost/demurrage) |
| CompostReceival | Record of compost redistribution into a commons pool |

### Link Types (12)

| LinkType | Anchor Key | Base → Target | Created By | Read By |
|---|---|---|---|---|
| TreasuryIdToTreasury | `"{treasury_id}"` | Anchor → Treasury | `create_treasury` | `get_treasury` |
| ManagerToTreasury | `"{manager_did}"` | Anchor → Treasury | `create_treasury` | `get_managed_treasuries` |
| TreasuryToContributions | `"{treasury_id}"` | Anchor → Contribution | `contribute_to_treasury` | `get_contributions` |
| ContributorToContributions | `"{contributor_did}"` | Anchor → Contribution | `contribute_to_treasury` | `get_my_contributions` |
| TreasuryToAllocations | `"{treasury_id}"` | Anchor → Allocation | `create_allocation` | `get_allocations` |
| AllocationIdToAllocation | `"{allocation_id}"` | Anchor → Allocation | `create_allocation` | `get_allocation`, `execute_allocation` |
| PoolIdToPool | `"{pool_id}"` | Anchor → SavingsPool | `create_savings_pool` | `get_pool` |
| TreasuryToPools | `"{treasury_id}"` | Anchor → SavingsPool | `create_savings_pool` | `get_treasury_pools` |
| MemberToPool | `"{member_did}"` | Anchor → SavingsPool | `create_savings_pool` | `get_my_pools` |
| CommonsPoolIdToPool | `"{pool_id}"` | Anchor → CommonsPool | `create_commons_pool` | `get_commons_pool` |
| DaoToCommonsPool | `"{dao_did}"` | Anchor → CommonsPool | `create_commons_pool` | `get_dao_commons_pools` |
| CommonsPoolToCompost | `"{commons_pool_id}"` | Anchor → CompostReceival | `receive_compost` | `get_compost_history` |

---

## Cross-Zome Call Topology

Key cross-zome dependencies between finance zomes:

| Caller Zome | Callee Zome | Function Called | Purpose |
|---|---|---|---|
| payments | tend | `verify_governance_agent` | SAP mint authorization |
| payments | finance_bridge | `get_member_fee_tier` | Progressive fee calculation |
| payments | recognition | `get_mycel_score` | Fee tier fallback |
| payments | treasury | `receive_compost` | Demurrage redistribution |
| payments | tend | `forgive_balance` | TEND forgiveness on exit |
| payments | recognition | `dissolve_mycel` | MYCEL dissolution on exit |
| payments | hearth_bridge (OtherRole) | `is_hearth_member` | Hearth pool membership |
| tend | recognition | `get_mycel_score` | Mediator eligibility check |
| tend | treasury | `transfer_commons_sap` | Bilateral settlement |
| tend | hearth_bridge (OtherRole) | `is_hearth_member` | Hearth exchange membership |
| bridge | recognition | `get_mycel_score` | Fee tier lookup |

---

## Shared Infrastructure

### `GOVERNANCE_AGENTS_ANCHOR`

Defined in `mycelix-finance-shared`. Used identically across 3 zomes (tend, recognition, staking) for governance agent authorization. Each zome has its own `GovernanceAgents` LinkType pointing to `AgentPubKey` targets from the shared anchor. Bootstrap mode allows any agent when no governance agents are registered.

### `anchor_hash(key: &str)`

Defined in `mycelix-finance-shared`. Deterministic EntryHash from a string key, used as the base for all anchor-pattern links. Creates a one-way mapping from human-readable keys to DHT addresses.

### `follow_update_chain(action_hash: ActionHash)`

Defined in `mycelix-finance-shared`. Follows Holochain's update chain from an initial create action to the latest version. Used universally for mutable entries (balances, exchanges, channels, oracle state, etc.).

### `verify_caller_is_did(did: &str)`

Defined in `mycelix-finance-shared`. Verifies that `agent_info().agent_initial_pubkey` matches the DID suffix, preventing DID spoofing in payment and exit operations.

---

## Statistics

| Metric | Count |
|---|---|
| Total LinkType variants | 72 |
| Unique anchor key patterns | 55+ |
| Zomes with GovernanceAgents | 3 (tend, recognition, staking) |
| Entry types across all zomes | 35 |
| Cross-zome call paths | 11 |
