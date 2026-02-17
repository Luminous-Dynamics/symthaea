// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/governance/IGovernor.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/**
 * @title GovernanceExtensions
 * @notice Extended governance utilities for the Mycelix DAO
 * @dev Implements vote delegation, proposal batching, and optimistic execution
 */

/**
 * @title VoteDelegation
 * @notice Advanced vote delegation with partial delegation support
 */
contract VoteDelegation is EIP712 {
    using ECDSA for bytes32;

    // ============================================================
    // Types
    // ============================================================

    struct Delegation {
        address delegate;
        uint96 percentage;   // Basis points (10000 = 100%)
        uint64 expiry;       // Delegation expiry timestamp
        bool active;
    }

    struct DelegationSignature {
        address delegator;
        address delegate;
        uint96 percentage;
        uint64 expiry;
        uint256 nonce;
        uint256 deadline;
        bytes signature;
    }

    // ============================================================
    // Storage
    // ============================================================

    // delegator => delegate => Delegation
    mapping(address => mapping(address => Delegation)) public delegations;

    // delegator => total delegated percentage
    mapping(address => uint96) public totalDelegated;

    // delegator => list of delegates
    mapping(address => address[]) public delegateList;

    // Nonces for signature replay protection
    mapping(address => uint256) public nonces;

    bytes32 public constant DELEGATION_TYPEHASH = keccak256(
        "Delegation(address delegator,address delegate,uint96 percentage,uint64 expiry,uint256 nonce,uint256 deadline)"
    );

    // ============================================================
    // Events
    // ============================================================

    event DelegationCreated(
        address indexed delegator,
        address indexed delegate,
        uint96 percentage,
        uint64 expiry
    );
    event DelegationRevoked(address indexed delegator, address indexed delegate);
    event DelegationExpired(address indexed delegator, address indexed delegate);

    // ============================================================
    // Errors
    // ============================================================

    error InvalidPercentage();
    error DelegationOverflow();
    error DelegationExpiredError();
    error InvalidSignature();
    error DeadlineExpired();

    // ============================================================
    // Constructor
    // ============================================================

    constructor() EIP712("MycelixVoteDelegation", "1") {}

    // ============================================================
    // Delegation Functions
    // ============================================================

    /**
     * @notice Create or update a delegation
     * @param delegate The address to delegate to
     * @param percentage The percentage of votes to delegate (in basis points)
     * @param expiry When the delegation expires
     */
    function createDelegation(
        address delegate,
        uint96 percentage,
        uint64 expiry
    ) external {
        _createDelegation(msg.sender, delegate, percentage, expiry);
    }

    /**
     * @notice Create delegation via signature (gasless)
     */
    function createDelegationBySig(DelegationSignature calldata sig) external {
        if (block.timestamp > sig.deadline) revert DeadlineExpired();

        bytes32 structHash = keccak256(abi.encode(
            DELEGATION_TYPEHASH,
            sig.delegator,
            sig.delegate,
            sig.percentage,
            sig.expiry,
            sig.nonce,
            sig.deadline
        ));

        bytes32 hash = _hashTypedDataV4(structHash);
        address signer = hash.recover(sig.signature);

        if (signer != sig.delegator) revert InvalidSignature();
        if (nonces[sig.delegator] != sig.nonce) revert InvalidSignature();

        nonces[sig.delegator]++;
        _createDelegation(sig.delegator, sig.delegate, sig.percentage, sig.expiry);
    }

    function _createDelegation(
        address delegator,
        address delegate,
        uint96 percentage,
        uint64 expiry
    ) internal {
        if (percentage > 10000) revert InvalidPercentage();
        if (expiry <= block.timestamp) revert DelegationExpiredError();

        Delegation storage existing = delegations[delegator][delegate];

        // Update total delegated percentage
        if (existing.active) {
            unchecked {
                totalDelegated[delegator] -= existing.percentage;
            }
        } else {
            delegateList[delegator].push(delegate);
        }

        if (totalDelegated[delegator] + percentage > 10000) {
            revert DelegationOverflow();
        }

        delegations[delegator][delegate] = Delegation({
            delegate: delegate,
            percentage: percentage,
            expiry: expiry,
            active: true
        });

        unchecked {
            totalDelegated[delegator] += percentage;
        }

        emit DelegationCreated(delegator, delegate, percentage, expiry);
    }

    /**
     * @notice Revoke a delegation
     */
    function revokeDelegation(address delegate) external {
        Delegation storage del = delegations[msg.sender][delegate];

        if (del.active) {
            unchecked {
                totalDelegated[msg.sender] -= del.percentage;
            }
            del.active = false;
            emit DelegationRevoked(msg.sender, delegate);
        }
    }

    /**
     * @notice Get effective voting power for an account
     * @param voter The voter address
     * @param baseVotes The voter's base voting power
     */
    function getEffectiveVotes(
        address voter,
        uint256 baseVotes
    ) external view returns (uint256 effectiveVotes) {
        // Start with undelegated portion
        effectiveVotes = (baseVotes * (10000 - totalDelegated[voter])) / 10000;

        // This would be called by the governor to add delegated votes
        // The actual implementation depends on integration with the token
    }

    /**
     * @notice Get all active delegations for an account
     */
    function getActiveDelegations(address delegator)
        external
        view
        returns (Delegation[] memory active)
    {
        address[] memory delegates = delegateList[delegator];
        uint256 count;

        // Count active delegations
        for (uint256 i; i < delegates.length; i++) {
            Delegation storage del = delegations[delegator][delegates[i]];
            if (del.active && del.expiry > block.timestamp) {
                count++;
            }
        }

        active = new Delegation[](count);
        uint256 index;

        for (uint256 i; i < delegates.length; i++) {
            Delegation storage del = delegations[delegator][delegates[i]];
            if (del.active && del.expiry > block.timestamp) {
                active[index++] = del;
            }
        }
    }
}

/**
 * @title ProposalBatcher
 * @notice Batch multiple proposals into a single meta-proposal
 */
contract ProposalBatcher {
    // ============================================================
    // Types
    // ============================================================

    struct BatchedProposal {
        uint256[] proposalIds;
        string description;
        address creator;
        uint256 createdAt;
        bool executed;
    }

    // ============================================================
    // Storage
    // ============================================================

    IGovernor public governor;
    mapping(uint256 => BatchedProposal) public batches;
    uint256 public batchCount;

    // ============================================================
    // Events
    // ============================================================

    event BatchCreated(uint256 indexed batchId, uint256[] proposalIds, string description);
    event BatchExecuted(uint256 indexed batchId);

    // ============================================================
    // Constructor
    // ============================================================

    constructor(address _governor) {
        governor = IGovernor(_governor);
    }

    // ============================================================
    // Functions
    // ============================================================

    /**
     * @notice Create a batch of proposals
     */
    function createBatch(
        address[][] memory targets,
        uint256[][] memory values,
        bytes[][] memory calldatas,
        string[] memory descriptions,
        string memory batchDescription
    ) external returns (uint256 batchId, uint256[] memory proposalIds) {
        require(targets.length == values.length && targets.length == calldatas.length, "Length mismatch");
        require(targets.length == descriptions.length, "Description mismatch");

        batchId = ++batchCount;
        proposalIds = new uint256[](targets.length);

        for (uint256 i; i < targets.length; i++) {
            proposalIds[i] = governor.propose(
                targets[i],
                values[i],
                calldatas[i],
                descriptions[i]
            );
        }

        batches[batchId] = BatchedProposal({
            proposalIds: proposalIds,
            description: batchDescription,
            creator: msg.sender,
            createdAt: block.timestamp,
            executed: false
        });

        emit BatchCreated(batchId, proposalIds, batchDescription);
    }

    /**
     * @notice Execute all proposals in a batch
     */
    function executeBatch(
        uint256 batchId,
        address[][] memory targets,
        uint256[][] memory values,
        bytes[][] memory calldatas,
        bytes32[] memory descriptionHashes
    ) external {
        BatchedProposal storage batch = batches[batchId];
        require(!batch.executed, "Batch already executed");
        require(batch.proposalIds.length == targets.length, "Length mismatch");

        for (uint256 i; i < batch.proposalIds.length; i++) {
            governor.execute(
                targets[i],
                values[i],
                calldatas[i],
                descriptionHashes[i]
            );
        }

        batch.executed = true;
        emit BatchExecuted(batchId);
    }

    /**
     * @notice Get batch details
     */
    function getBatch(uint256 batchId) external view returns (BatchedProposal memory) {
        return batches[batchId];
    }
}

/**
 * @title OptimisticGovernance
 * @notice Allows optimistic execution of proposals with a challenge period
 */
contract OptimisticGovernance {
    // ============================================================
    // Types
    // ============================================================

    enum OptimisticState {
        Pending,
        Challenged,
        Executed,
        Cancelled
    }

    struct OptimisticProposal {
        bytes32 proposalHash;
        address proposer;
        uint256 createdAt;
        uint256 executionTime;
        uint256 challengeStake;
        address challenger;
        OptimisticState state;
    }

    // ============================================================
    // Storage
    // ============================================================

    IGovernor public governor;
    uint256 public challengePeriod = 2 days;
    uint256 public minChallengeStake = 1 ether;

    mapping(bytes32 => OptimisticProposal) public optimisticProposals;

    // ============================================================
    // Events
    // ============================================================

    event OptimisticProposalCreated(bytes32 indexed proposalHash, address proposer);
    event ProposalChallenged(bytes32 indexed proposalHash, address challenger);
    event ProposalExecuted(bytes32 indexed proposalHash);
    event ProposalCancelled(bytes32 indexed proposalHash);

    // ============================================================
    // Constructor
    // ============================================================

    constructor(address _governor) {
        governor = IGovernor(_governor);
    }

    // ============================================================
    // Functions
    // ============================================================

    /**
     * @notice Create an optimistic proposal
     */
    function createOptimisticProposal(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description
    ) external returns (bytes32 proposalHash) {
        proposalHash = keccak256(abi.encode(targets, values, calldatas, keccak256(bytes(description))));

        optimisticProposals[proposalHash] = OptimisticProposal({
            proposalHash: proposalHash,
            proposer: msg.sender,
            createdAt: block.timestamp,
            executionTime: block.timestamp + challengePeriod,
            challengeStake: 0,
            challenger: address(0),
            state: OptimisticState.Pending
        });

        emit OptimisticProposalCreated(proposalHash, msg.sender);
    }

    /**
     * @notice Challenge an optimistic proposal
     */
    function challengeProposal(bytes32 proposalHash) external payable {
        OptimisticProposal storage prop = optimisticProposals[proposalHash];

        require(prop.state == OptimisticState.Pending, "Invalid state");
        require(block.timestamp < prop.executionTime, "Challenge period ended");
        require(msg.value >= minChallengeStake, "Insufficient stake");

        prop.state = OptimisticState.Challenged;
        prop.challengeStake = msg.value;
        prop.challenger = msg.sender;

        emit ProposalChallenged(proposalHash, msg.sender);
    }

    /**
     * @notice Execute an unchallenged optimistic proposal
     */
    function executeOptimistic(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) external {
        bytes32 proposalHash = keccak256(abi.encode(targets, values, calldatas, descriptionHash));
        OptimisticProposal storage prop = optimisticProposals[proposalHash];

        require(prop.state == OptimisticState.Pending, "Invalid state");
        require(block.timestamp >= prop.executionTime, "Challenge period active");

        prop.state = OptimisticState.Executed;

        // Execute via governor
        governor.execute(targets, values, calldatas, descriptionHash);

        emit ProposalExecuted(proposalHash);
    }
}
