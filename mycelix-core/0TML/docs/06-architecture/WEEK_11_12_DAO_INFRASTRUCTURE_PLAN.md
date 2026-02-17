# Week 11-12: DAO Infrastructure & Treasury Management

**Version**: 1.0
**Date**: November 11, 2025
**Status**: Planning Phase
**Dependencies**: Week 9-10 Production Deployment (Polygon L2 Integration)

---

## Executive Summary

Build complete DAO infrastructure on Polygon L2 with:
- **Treasury Management**: Multi-sig treasury controlled by governance
- **Automated Execution**: On-chain execution of approved proposals
- **Token Economics**: Optional governance token for additional features
- **Budget Proposals**: Community-driven resource allocation

**Timeline**: 16-24 days
**Cost**: ~$500-2000/month operational costs

---

## Track 2.1: Treasury Smart Contracts (5-7 days)

### Objective

Create secure, governance-controlled treasury for Zero-TrustML network.

### Architecture

```
Governance Vote on Polygon
        ↓
   Proposal Approved
        ↓
Treasury Contract (Multi-sig)
        ↓
Automated Fund Distribution
        ↓
Participants / Projects / Infrastructure
```

### Smart Contracts

#### 1. Treasury Contract (3-4 days)

```solidity
// Treasury.sol
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract ZeroTrustMLTreasury is AccessControl, ReentrancyGuard {
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");

    struct BudgetProposal {
        bytes32 proposalId;
        address recipient;
        uint256 amount;
        address token;  // ERC20 token address (address(0) for MATIC)
        string purpose;
        uint256 executionTime;
        bool executed;
        bytes32 governanceProposalId;  // Link to governance proposal
    }

    mapping(bytes32 => BudgetProposal) public budgetProposals;
    mapping(address => uint256) public budgetAllocations;

    event FundsDeposited(address indexed from, uint256 amount, address token);
    event BudgetProposalCreated(bytes32 indexed proposalId, address recipient, uint256 amount);
    event BudgetProposalExecuted(bytes32 indexed proposalId, address recipient, uint256 amount);
    event EmergencyWithdrawal(address indexed to, uint256 amount, address token);

    constructor(address governanceContract) {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(GOVERNANCE_ROLE, governanceContract);
    }

    // Receive deposits
    receive() external payable {
        emit FundsDeposited(msg.sender, msg.value, address(0));
    }

    function depositToken(address token, uint256 amount) external {
        require(IERC20(token).transferFrom(msg.sender, address(this), amount), "Transfer failed");
        emit FundsDeposited(msg.sender, amount, token);
    }

    // Create budget proposal (called by governance)
    function createBudgetProposal(
        bytes32 proposalId,
        address recipient,
        uint256 amount,
        address token,
        string calldata purpose,
        bytes32 governanceProposalId
    ) external onlyRole(GOVERNANCE_ROLE) {
        require(budgetProposals[proposalId].recipient == address(0), "Proposal exists");

        budgetProposals[proposalId] = BudgetProposal({
            proposalId: proposalId,
            recipient: recipient,
            amount: amount,
            token: token,
            purpose: purpose,
            executionTime: block.timestamp + 48 hours,  // 48-hour timelock
            executed: false,
            governanceProposalId: governanceProposalId
        });

        emit BudgetProposalCreated(proposalId, recipient, amount);
    }

    // Execute approved budget proposal
    function executeBudgetProposal(bytes32 proposalId) external nonReentrant onlyRole(EXECUTOR_ROLE) {
        BudgetProposal storage proposal = budgetProposals[proposalId];
        require(proposal.recipient != address(0), "Proposal not found");
        require(!proposal.executed, "Already executed");
        require(block.timestamp >= proposal.executionTime, "Timelock not expired");

        proposal.executed = true;

        if (proposal.token == address(0)) {
            // Transfer MATIC
            require(address(this).balance >= proposal.amount, "Insufficient MATIC");
            (bool success, ) = proposal.recipient.call{value: proposal.amount}("");
            require(success, "Transfer failed");
        } else {
            // Transfer ERC20 token
            require(IERC20(proposal.token).transfer(proposal.recipient, proposal.amount), "Token transfer failed");
        }

        emit BudgetProposalExecuted(proposalId, proposal.recipient, proposal.amount);
    }

    // Emergency multi-sig withdrawal (requires 3/5 guardians)
    function emergencyWithdraw(
        address payable to,
        uint256 amount,
        address token
    ) external onlyRole(GOVERNANCE_ROLE) {
        if (token == address(0)) {
            require(address(this).balance >= amount, "Insufficient balance");
            (bool success, ) = to.call{value: amount}("");
            require(success, "Transfer failed");
        } else {
            require(IERC20(token).transfer(to, amount), "Token transfer failed");
        }

        emit EmergencyWithdrawal(to, amount, token);
    }

    // View functions
    function getBalance(address token) external view returns (uint256) {
        if (token == address(0)) {
            return address(this).balance;
        } else {
            return IERC20(token).balanceOf(address(this));
        }
    }
}
```

#### 2. Automated Rewards Distribution (2-3 days)

```solidity
// RewardsDistributor.sol
pragma solidity ^0.8.20;

contract RewardsDistributor {
    struct RewardBatch {
        bytes32 batchId;
        uint256 roundNumber;
        address[] recipients;
        uint256[] amounts;
        uint256 totalAmount;
        bool distributed;
    }

    mapping(bytes32 => RewardBatch) public rewardBatches;

    event RewardBatchCreated(bytes32 indexed batchId, uint256 roundNumber, uint256 totalAmount);
    event RewardsDistributed(bytes32 indexed batchId, uint256 numRecipients);

    function createRewardBatch(
        bytes32 batchId,
        uint256 roundNumber,
        address[] calldata recipients,
        uint256[] calldata amounts
    ) external {
        require(recipients.length == amounts.length, "Length mismatch");

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        rewardBatches[batchId] = RewardBatch({
            batchId: batchId,
            roundNumber: roundNumber,
            recipients: recipients,
            amounts: amounts,
            totalAmount: totalAmount,
            distributed: false
        });

        emit RewardBatchCreated(batchId, roundNumber, totalAmount);
    }

    function distributeRewards(bytes32 batchId) external {
        RewardBatch storage batch = rewardBatches[batchId];
        require(!batch.distributed, "Already distributed");

        batch.distributed = true;

        for (uint256 i = 0; i < batch.recipients.length; i++) {
            (bool success, ) = batch.recipients[i].call{value: batch.amounts[i]}("");
            require(success, "Transfer failed");
        }

        emit RewardsDistributed(batchId, batch.recipients.length);
    }
}
```

### Python Integration (1 day)

```python
from web3 import Web3

class TreasuryManager:
    def __init__(self, w3: Web3, treasury_address: str, governance_contract):
        self.w3 = w3
        self.treasury = w3.eth.contract(address=treasury_address, abi=TREASURY_ABI)
        self.governance = governance_contract

    async def create_budget_proposal(
        self,
        proposal_id: str,
        recipient: str,
        amount: int,
        token: str,
        purpose: str,
        governance_proposal_id: str
    ) -> str:
        """Create budget proposal after governance approval"""
        tx = self.treasury.functions.createBudgetProposal(
            proposalId=bytes.fromhex(proposal_id),
            recipient=recipient,
            amount=amount,
            token=token,
            purpose=purpose,
            governanceProposalId=bytes.fromhex(governance_proposal_id)
        ).build_transaction({
            'from': self.governance.address,
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.governance.address)
        })

        signed_tx = self.w3.eth.account.sign_transaction(tx, GOVERNANCE_PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        return tx_hash.hex()

    async def execute_budget_proposal(self, proposal_id: str) -> str:
        """Execute budget proposal after timelock"""
        tx = self.treasury.functions.executeBudgetProposal(
            proposalId=bytes.fromhex(proposal_id)
        ).build_transaction({
            'from': EXECUTOR_ADDRESS,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(EXECUTOR_ADDRESS)
        })

        signed_tx = self.w3.eth.account.sign_transaction(tx, EXECUTOR_PRIVATE_KEY)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        return tx_hash.hex()

    async def get_treasury_balance(self, token: str = "0x0") -> int:
        """Get treasury balance"""
        return self.treasury.functions.getBalance(token).call()
```

**Deliverable**: Multi-sig treasury with governance control and automated distribution.

---

## Track 2.2: Automated Proposal Execution (3-5 days)

### Objective

Automatically execute approved governance proposals on-chain.

### Execution Types

1. **Parameter Changes**: Update FL coordinator parameters
2. **Budget Proposals**: Release funds from treasury
3. **Participant Bans**: Update access control lists
4. **Emergency Actions**: Execute emergency stops/resumes

### Implementation

```python
class ProposalExecutor:
    """Automatically execute approved proposals"""

    async def execute_proposal(self, proposal_id: str):
        # Get proposal from governance
        proposal = await governance_coordinator.proposal_mgr.get_proposal(proposal_id)

        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError(f"Proposal not approved: {proposal.status}")

        # Route to appropriate executor
        if proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
            return await self._execute_parameter_change(proposal)
        elif proposal.proposal_type == ProposalType.PARTICIPANT_MANAGEMENT:
            return await self._execute_participant_management(proposal)
        elif proposal.proposal_type == ProposalType.BUDGET_PROPOSAL:
            return await self._execute_budget_proposal(proposal)
        elif proposal.proposal_type == ProposalType.EMERGENCY_ACTION:
            return await self._execute_emergency_action(proposal)

    async def _execute_parameter_change(self, proposal: ProposalData):
        """Execute parameter change"""
        params = proposal.execution_params
        parameter = params["parameter"]
        new_value = params["new_value"]

        # Update FL coordinator
        await fl_coordinator.update_parameter(parameter, new_value)

        # Log execution
        await self._log_execution(proposal.proposal_id, success=True, result={
            "parameter": parameter,
            "old_value": params.get("current_value"),
            "new_value": new_value
        })

    async def _execute_budget_proposal(self, proposal: ProposalData):
        """Execute budget proposal"""
        params = proposal.execution_params
        recipient = params["recipient"]
        amount = params["amount"]
        token = params.get("token", "0x0")

        # Create treasury proposal
        tx_hash = await treasury_manager.create_budget_proposal(
            proposal_id=proposal.proposal_id,
            recipient=recipient,
            amount=amount,
            token=token,
            purpose=proposal.title,
            governance_proposal_id=proposal.proposal_id
        )

        # Wait 48 hours for timelock
        await asyncio.sleep(48 * 3600)

        # Execute distribution
        exec_tx_hash = await treasury_manager.execute_budget_proposal(proposal.proposal_id)

        await self._log_execution(proposal.proposal_id, success=True, result={
            "recipient": recipient,
            "amount": amount,
            "tx_hash": exec_tx_hash
        })
```

**Deliverable**: Automated execution for all proposal types.

---

## Track 2.3: Token Economics (Optional) (3-5 days)

### Objective

Optional governance token for additional features (staking, rewards, etc.).

**Note**: This is **optional** - current reputation-weighted voting works without a token.

### Governance Token Design

```solidity
// ZeroTrustMLToken.sol
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";

contract ZeroTrustMLToken is ERC20, ERC20Votes {
    constructor() ERC20("ZeroTrustML", "ZTM") ERC20Permit("ZeroTrustML") {
        _mint(msg.sender, 1_000_000_000 * 10**18);  // 1 billion tokens
    }

    // Required overrides
    function _afterTokenTransfer(address from, address to, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._afterTokenTransfer(from, to, amount);
    }

    function _mint(address to, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._mint(to, amount);
    }

    function _burn(address account, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._burn(account, amount);
    }
}
```

### Token Use Cases

1. **Staking for Guardians**: Guardians stake tokens to signal commitment
2. **Rewards for FL Participants**: Additional rewards beyond base compensation
3. **Voting Power Boost**: Token holders get small voting power boost (e.g., 1.1x)
4. **Governance Treasury**: Tokens can be granted for ecosystem development

**Recommendation**: Start without token, add later if community requests.

---

## Track 2.4: Budget Proposal System (3-5 days)

### Objective

Allow community to propose and vote on budget allocation.

### Budget Proposal Flow

```
1. Community member creates budget proposal
2. Proposal includes: recipient, amount, purpose, milestones
3. Community votes on proposal (7-14 days)
4. If approved: funds allocated with milestones
5. Recipient reports progress on milestones
6. Community can vote to cancel if milestones not met
```

### Implementation

```python
class BudgetProposalManager:
    async def create_budget_proposal(
        self,
        proposer_id: str,
        recipient: str,
        amount: int,
        token: str,
        purpose: str,
        milestones: List[Dict],
        duration_days: int = 90
    ) -> str:
        """Create budget proposal"""

        # Create governance proposal
        success, message, proposal_id = await governance_coordinator.create_governance_proposal(
            proposer_participant_id=proposer_id,
            proposal_type=ProposalType.BUDGET_PROPOSAL,
            title=f"Budget: {purpose}",
            description=f"""
            ## Budget Request

            **Recipient**: {recipient}
            **Amount**: {amount} tokens
            **Duration**: {duration_days} days
            **Purpose**: {purpose}

            ## Milestones
            {self._format_milestones(milestones)}

            ## Expected Impact
            [Proposer should detail expected impact]
            """,
            execution_params={
                "recipient": recipient,
                "amount": amount,
                "token": token,
                "milestones": milestones,
                "duration_days": duration_days
            },
            voting_duration_days=14  # Longer for budget proposals
        )

        return proposal_id

    async def report_milestone_completion(
        self,
        proposal_id: str,
        milestone_id: int,
        evidence_cid: str  # IPFS CID of completion evidence
    ):
        """Report milestone completion"""
        # Store milestone completion
        await governance_coordinator.store_milestone_completion(
            proposal_id=proposal_id,
            milestone_id=milestone_id,
            evidence_cid=evidence_cid,
            timestamp=int(time.time())
        )

        # Trigger next payment if milestone approved
        milestone = await self._get_milestone(proposal_id, milestone_id)
        if milestone["payment_amount"] > 0:
            await treasury_manager.release_milestone_payment(
                proposal_id=proposal_id,
                milestone_id=milestone_id,
                recipient=milestone["recipient"],
                amount=milestone["payment_amount"]
            )
```

**Deliverable**: Community-driven budget allocation system.

---

## Track 2.5: Testing & Documentation (2-3 days)

### Testing

1. **Treasury Tests** (1 day)
   - Test deposit/withdrawal
   - Test multi-sig authorization
   - Test timelock mechanism
   - Test emergency withdrawal

2. **Execution Tests** (1 day)
   - Test automated proposal execution
   - Test execution failure handling
   - Test execution result logging

3. **Integration Tests** (0.5 day)
   - Test end-to-end budget proposal flow
   - Test milestone-based payments
   - Test proposal cancellation

### Documentation (0.5-1 day)

- Treasury management guide
- Budget proposal guide
- Token economics (if implemented)
- Emergency procedures

**Deliverable**: Complete test coverage and documentation.

---

## Summary: Track 2 (DAO Infrastructure)

### Timeline

| Component | Duration | Key Deliverable |
|-----------|----------|-----------------|
| Treasury Contracts | 5-7 days | Multi-sig treasury |
| Automated Execution | 3-5 days | Proposal execution |
| Token Economics (Optional) | 3-5 days | Governance token |
| Budget Proposals | 3-5 days | Community funding |
| Testing & Docs | 2-3 days | Complete coverage |
| **Total** | **16-25 days** | Full DAO |

### Cost Estimate

**Development**:
- Smart contract audits: $5-20k (recommended for treasury)
- Testing on Mumbai: Minimal

**Operational**:
- Treasury holding costs: $0 (just gas for transactions)
- Execution gas: ~$0.01-0.05 per execution
- Monitoring: $20-50/month

**Total**: ~$30-100/month operational + one-time audit

### Success Criteria

- ✅ Treasury can hold and distribute funds
- ✅ Multi-sig emergency withdrawal works
- ✅ Automated proposal execution successful
- ✅ Budget proposals create and execute correctly
- ✅ Milestone-based payments work
- ✅ Smart contracts audited (if handling significant funds)
- ✅ Complete documentation

---

**Status**: Planning Complete
**Recommendation**: Only implement after Track 1 (Production Readiness) and initial community activity
**Next**: Evaluate community need for treasury before implementation
