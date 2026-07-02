// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/governance/GovernanceExtensions.sol";

contract VoteDelegationTest is Test {
    VoteDelegation delegation;

    address alice = makeAddr("alice");
    address bob = makeAddr("bob");
    address charlie = makeAddr("charlie");

    uint256 aliceKey = 0x1;

    function setUp() public {
        delegation = new VoteDelegation();
    }

    // ============================================================
    // createDelegation Tests
    // ============================================================

    function test_createDelegation() public {
        vm.prank(alice);
        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 30 days));

        (address delegate, uint96 percentage, uint64 expiry, bool active) =
            delegation.delegations(alice, bob);

        assertEq(delegate, bob);
        assertEq(percentage, 5000);
        assertEq(expiry, uint64(block.timestamp + 30 days));
        assertTrue(active);
    }

    function test_createDelegation_emitsEvent() public {
        vm.expectEmit(true, true, false, true);
        emit VoteDelegation.DelegationCreated(
            alice,
            bob,
            5000,
            uint64(block.timestamp + 30 days)
        );

        vm.prank(alice);
        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 30 days));
    }

    function test_createDelegation_multipleDelegates() public {
        vm.startPrank(alice);

        delegation.createDelegation(bob, 3000, uint64(block.timestamp + 30 days));
        delegation.createDelegation(charlie, 2000, uint64(block.timestamp + 30 days));

        vm.stopPrank();

        assertEq(delegation.totalDelegated(alice), 5000);
    }

    function test_createDelegation_updateExisting() public {
        vm.startPrank(alice);

        delegation.createDelegation(bob, 3000, uint64(block.timestamp + 30 days));
        assertEq(delegation.totalDelegated(alice), 3000);

        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 60 days));
        assertEq(delegation.totalDelegated(alice), 5000);

        vm.stopPrank();
    }

    function test_createDelegation_revertsIfPercentageTooHigh() public {
        vm.prank(alice);
        vm.expectRevert(VoteDelegation.InvalidPercentage.selector);
        delegation.createDelegation(bob, 10001, uint64(block.timestamp + 30 days));
    }

    function test_createDelegation_revertsIfExpired() public {
        vm.prank(alice);
        vm.expectRevert(VoteDelegation.DelegationExpiredError.selector);
        delegation.createDelegation(bob, 5000, uint64(block.timestamp - 1));
    }

    function test_createDelegation_revertsIfOverflow() public {
        vm.startPrank(alice);

        delegation.createDelegation(bob, 6000, uint64(block.timestamp + 30 days));

        vm.expectRevert(VoteDelegation.DelegationOverflow.selector);
        delegation.createDelegation(charlie, 5000, uint64(block.timestamp + 30 days));

        vm.stopPrank();
    }

    // ============================================================
    // revokeDelegation Tests
    // ============================================================

    function test_revokeDelegation() public {
        vm.startPrank(alice);

        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 30 days));
        assertEq(delegation.totalDelegated(alice), 5000);

        delegation.revokeDelegation(bob);
        assertEq(delegation.totalDelegated(alice), 0);

        (, , , bool active) = delegation.delegations(alice, bob);
        assertFalse(active);

        vm.stopPrank();
    }

    function test_revokeDelegation_emitsEvent() public {
        vm.prank(alice);
        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 30 days));

        vm.expectEmit(true, true, false, false);
        emit VoteDelegation.DelegationRevoked(alice, bob);

        vm.prank(alice);
        delegation.revokeDelegation(bob);
    }

    function test_revokeDelegation_nonExistent() public {
        // Should not revert, just do nothing
        vm.prank(alice);
        delegation.revokeDelegation(bob);

        assertEq(delegation.totalDelegated(alice), 0);
    }

    // ============================================================
    // getEffectiveVotes Tests
    // ============================================================

    function test_getEffectiveVotes_noDelegation() public view {
        uint256 baseVotes = 1000;
        uint256 effective = delegation.getEffectiveVotes(alice, baseVotes);

        assertEq(effective, baseVotes);
    }

    function test_getEffectiveVotes_withDelegation() public {
        vm.prank(alice);
        delegation.createDelegation(bob, 5000, uint64(block.timestamp + 30 days));

        uint256 baseVotes = 1000;
        uint256 effective = delegation.getEffectiveVotes(alice, baseVotes);

        // 50% delegated, so 50% remaining
        assertEq(effective, 500);
    }

    function test_getEffectiveVotes_fullDelegation() public {
        vm.prank(alice);
        delegation.createDelegation(bob, 10000, uint64(block.timestamp + 30 days));

        uint256 baseVotes = 1000;
        uint256 effective = delegation.getEffectiveVotes(alice, baseVotes);

        assertEq(effective, 0);
    }

    // ============================================================
    // getActiveDelegations Tests
    // ============================================================

    function test_getActiveDelegations() public {
        vm.startPrank(alice);
        delegation.createDelegation(bob, 3000, uint64(block.timestamp + 30 days));
        delegation.createDelegation(charlie, 2000, uint64(block.timestamp + 30 days));
        vm.stopPrank();

        VoteDelegation.Delegation[] memory active = delegation.getActiveDelegations(alice);

        assertEq(active.length, 2);
    }

    function test_getActiveDelegations_excludesExpired() public {
        vm.startPrank(alice);
        delegation.createDelegation(bob, 3000, uint64(block.timestamp + 30 days));
        delegation.createDelegation(charlie, 2000, uint64(block.timestamp + 10 days));
        vm.stopPrank();

        // Advance time past charlie's delegation
        vm.warp(block.timestamp + 15 days);

        VoteDelegation.Delegation[] memory active = delegation.getActiveDelegations(alice);

        assertEq(active.length, 1);
        assertEq(active[0].delegate, bob);
    }

    function test_getActiveDelegations_excludesRevoked() public {
        vm.startPrank(alice);
        delegation.createDelegation(bob, 3000, uint64(block.timestamp + 30 days));
        delegation.createDelegation(charlie, 2000, uint64(block.timestamp + 30 days));
        delegation.revokeDelegation(charlie);
        vm.stopPrank();

        VoteDelegation.Delegation[] memory active = delegation.getActiveDelegations(alice);

        assertEq(active.length, 1);
        assertEq(active[0].delegate, bob);
    }

    // ============================================================
    // createDelegationBySig Tests
    // ============================================================

    function test_createDelegationBySig() public {
        uint256 privateKey = 0xA11CE;
        address signer = vm.addr(privateKey);

        uint96 percentage = 5000;
        uint64 expiry = uint64(block.timestamp + 30 days);
        uint256 nonce = delegation.nonces(signer);
        uint256 deadline = block.timestamp + 1 hours;

        bytes32 structHash = keccak256(abi.encode(
            delegation.DELEGATION_TYPEHASH(),
            signer,
            bob,
            percentage,
            expiry,
            nonce,
            deadline
        ));

        bytes32 digest = keccak256(abi.encodePacked(
            "\x19\x01",
            delegation.DOMAIN_SEPARATOR(),
            structHash
        ));

        (uint8 v, bytes32 r, bytes32 s) = vm.sign(privateKey, digest);

        VoteDelegation.DelegationSignature memory sig = VoteDelegation.DelegationSignature({
            delegator: signer,
            delegate: bob,
            percentage: percentage,
            expiry: expiry,
            nonce: nonce,
            deadline: deadline,
            signature: abi.encodePacked(r, s, v)
        });

        delegation.createDelegationBySig(sig);

        (address delegate, uint96 pct, , bool active) = delegation.delegations(signer, bob);
        assertEq(delegate, bob);
        assertEq(pct, percentage);
        assertTrue(active);
    }

    function test_createDelegationBySig_revertsIfDeadlineExpired() public {
        VoteDelegation.DelegationSignature memory sig = VoteDelegation.DelegationSignature({
            delegator: alice,
            delegate: bob,
            percentage: 5000,
            expiry: uint64(block.timestamp + 30 days),
            nonce: 0,
            deadline: block.timestamp - 1,
            signature: ""
        });

        vm.expectRevert(VoteDelegation.DeadlineExpired.selector);
        delegation.createDelegationBySig(sig);
    }
}

contract ProposalBatcherTest is Test {
    ProposalBatcher batcher;
    MockGovernor mockGovernor;

    address alice = makeAddr("alice");

    function setUp() public {
        mockGovernor = new MockGovernor();
        batcher = new ProposalBatcher(address(mockGovernor));
    }

    function test_createBatch() public {
        address[][] memory targets = new address[][](2);
        targets[0] = new address[](1);
        targets[0][0] = address(0x1);
        targets[1] = new address[](1);
        targets[1][0] = address(0x2);

        uint256[][] memory values = new uint256[][](2);
        values[0] = new uint256[](1);
        values[1] = new uint256[](1);

        bytes[][] memory calldatas = new bytes[][](2);
        calldatas[0] = new bytes[](1);
        calldatas[1] = new bytes[](1);

        string[] memory descriptions = new string[](2);
        descriptions[0] = "Proposal 1";
        descriptions[1] = "Proposal 2";

        vm.prank(alice);
        (uint256 batchId, uint256[] memory proposalIds) = batcher.createBatch(
            targets,
            values,
            calldatas,
            descriptions,
            "Batch Description"
        );

        assertEq(batchId, 1);
        assertEq(proposalIds.length, 2);

        ProposalBatcher.BatchedProposal memory batch = batcher.getBatch(batchId);
        assertEq(batch.creator, alice);
        assertEq(batch.description, "Batch Description");
        assertFalse(batch.executed);
    }

    function test_executeBatch() public {
        // Create batch first
        address[][] memory targets = new address[][](1);
        targets[0] = new address[](1);
        targets[0][0] = address(0x1);

        uint256[][] memory values = new uint256[][](1);
        values[0] = new uint256[](1);

        bytes[][] memory calldatas = new bytes[][](1);
        calldatas[0] = new bytes[](1);

        string[] memory descriptions = new string[](1);
        descriptions[0] = "Proposal 1";

        (uint256 batchId, ) = batcher.createBatch(
            targets,
            values,
            calldatas,
            descriptions,
            "Batch"
        );

        bytes32[] memory descriptionHashes = new bytes32[](1);
        descriptionHashes[0] = keccak256(bytes("Proposal 1"));

        batcher.executeBatch(batchId, targets, values, calldatas, descriptionHashes);

        ProposalBatcher.BatchedProposal memory batch = batcher.getBatch(batchId);
        assertTrue(batch.executed);
    }

    function test_executeBatch_revertsIfAlreadyExecuted() public {
        // Setup and execute once
        address[][] memory targets = new address[][](1);
        targets[0] = new address[](1);
        targets[0][0] = address(0x1);

        uint256[][] memory values = new uint256[][](1);
        values[0] = new uint256[](1);

        bytes[][] memory calldatas = new bytes[][](1);
        calldatas[0] = new bytes[](1);

        string[] memory descriptions = new string[](1);
        descriptions[0] = "Proposal 1";

        (uint256 batchId, ) = batcher.createBatch(targets, values, calldatas, descriptions, "Batch");

        bytes32[] memory descriptionHashes = new bytes32[](1);
        descriptionHashes[0] = keccak256(bytes("Proposal 1"));

        batcher.executeBatch(batchId, targets, values, calldatas, descriptionHashes);

        // Try to execute again
        vm.expectRevert("Batch already executed");
        batcher.executeBatch(batchId, targets, values, calldatas, descriptionHashes);
    }
}

contract OptimisticGovernanceTest is Test {
    OptimisticGovernance optimistic;
    MockGovernor mockGovernor;

    address alice = makeAddr("alice");
    address bob = makeAddr("bob");

    function setUp() public {
        mockGovernor = new MockGovernor();
        optimistic = new OptimisticGovernance(address(mockGovernor));
    }

    function test_createOptimisticProposal() public {
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);

        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);

        vm.prank(alice);
        bytes32 proposalHash = optimistic.createOptimisticProposal(
            targets,
            values,
            calldatas,
            "Test Proposal"
        );

        (
            bytes32 hash,
            address proposer,
            uint256 createdAt,
            uint256 executionTime,
            ,
            ,
            OptimisticGovernance.OptimisticState state
        ) = optimistic.optimisticProposals(proposalHash);

        assertEq(hash, proposalHash);
        assertEq(proposer, alice);
        assertEq(createdAt, block.timestamp);
        assertEq(executionTime, block.timestamp + 2 days);
        assertEq(uint8(state), uint8(OptimisticGovernance.OptimisticState.Pending));
    }

    function test_challengeProposal() public {
        // Create proposal
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);

        vm.prank(alice);
        bytes32 proposalHash = optimistic.createOptimisticProposal(
            targets, values, calldatas, "Test"
        );

        // Challenge
        vm.deal(bob, 2 ether);
        vm.prank(bob);
        optimistic.challengeProposal{value: 1 ether}(proposalHash);

        (
            , , , ,
            uint256 challengeStake,
            address challenger,
            OptimisticGovernance.OptimisticState state
        ) = optimistic.optimisticProposals(proposalHash);

        assertEq(challengeStake, 1 ether);
        assertEq(challenger, bob);
        assertEq(uint8(state), uint8(OptimisticGovernance.OptimisticState.Challenged));
    }

    function test_challengeProposal_revertsIfNotEnoughStake() public {
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);

        vm.prank(alice);
        bytes32 proposalHash = optimistic.createOptimisticProposal(
            targets, values, calldatas, "Test"
        );

        vm.deal(bob, 0.5 ether);
        vm.prank(bob);
        vm.expectRevert("Insufficient stake");
        optimistic.challengeProposal{value: 0.5 ether}(proposalHash);
    }

    function test_challengeProposal_revertsIfChallengePeriodEnded() public {
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);

        vm.prank(alice);
        bytes32 proposalHash = optimistic.createOptimisticProposal(
            targets, values, calldatas, "Test"
        );

        // Advance past challenge period
        vm.warp(block.timestamp + 3 days);

        vm.deal(bob, 2 ether);
        vm.prank(bob);
        vm.expectRevert("Challenge period ended");
        optimistic.challengeProposal{value: 1 ether}(proposalHash);
    }

    function test_executeOptimistic() public {
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);
        string memory description = "Test";

        vm.prank(alice);
        bytes32 proposalHash = optimistic.createOptimisticProposal(
            targets, values, calldatas, description
        );

        // Advance past challenge period
        vm.warp(block.timestamp + 3 days);

        bytes32 descriptionHash = keccak256(bytes(description));
        optimistic.executeOptimistic(targets, values, calldatas, descriptionHash);

        (, , , , , , OptimisticGovernance.OptimisticState state) =
            optimistic.optimisticProposals(proposalHash);

        assertEq(uint8(state), uint8(OptimisticGovernance.OptimisticState.Executed));
    }

    function test_executeOptimistic_revertsIfChallengePeriodActive() public {
        address[] memory targets = new address[](1);
        targets[0] = address(0x1);
        uint256[] memory values = new uint256[](1);
        bytes[] memory calldatas = new bytes[](1);
        string memory description = "Test";

        vm.prank(alice);
        optimistic.createOptimisticProposal(targets, values, calldatas, description);

        bytes32 descriptionHash = keccak256(bytes(description));

        vm.expectRevert("Challenge period active");
        optimistic.executeOptimistic(targets, values, calldatas, descriptionHash);
    }
}

// Mock Governor for testing
contract MockGovernor is IGovernor {
    uint256 public proposalCount;

    function name() external pure override returns (string memory) {
        return "MockGovernor";
    }

    function version() external pure override returns (string memory) {
        return "1";
    }

    function COUNTING_MODE() external pure override returns (string memory) {
        return "support=bravo&quorum=for,abstain";
    }

    function hashProposal(
        address[] memory,
        uint256[] memory,
        bytes[] memory,
        bytes32
    ) external pure override returns (uint256) {
        return 1;
    }

    function state(uint256) external pure override returns (ProposalState) {
        return ProposalState.Active;
    }

    function proposalThreshold() external pure override returns (uint256) {
        return 0;
    }

    function proposalSnapshot(uint256) external pure override returns (uint256) {
        return block.number;
    }

    function proposalDeadline(uint256) external pure override returns (uint256) {
        return block.number + 100;
    }

    function proposalProposer(uint256) external pure override returns (address) {
        return address(0);
    }

    function proposalEta(uint256) external pure override returns (uint256) {
        return 0;
    }

    function proposalNeedsQueuing(uint256) external pure override returns (bool) {
        return false;
    }

    function votingDelay() external pure override returns (uint256) {
        return 1;
    }

    function votingPeriod() external pure override returns (uint256) {
        return 100;
    }

    function quorum(uint256) external pure override returns (uint256) {
        return 0;
    }

    function getVotes(address, uint256) external pure override returns (uint256) {
        return 100;
    }

    function getVotesWithParams(
        address,
        uint256,
        bytes memory
    ) external pure override returns (uint256) {
        return 100;
    }

    function hasVoted(uint256, address) external pure override returns (bool) {
        return false;
    }

    function propose(
        address[] memory,
        uint256[] memory,
        bytes[] memory,
        string memory
    ) external override returns (uint256) {
        return ++proposalCount;
    }

    function queue(
        address[] memory,
        uint256[] memory,
        bytes[] memory,
        bytes32
    ) external pure override returns (uint256) {
        return 1;
    }

    function execute(
        address[] memory,
        uint256[] memory,
        bytes[] memory,
        bytes32
    ) external payable override returns (uint256) {
        return 1;
    }

    function cancel(
        address[] memory,
        uint256[] memory,
        bytes[] memory,
        bytes32
    ) external pure override returns (uint256) {
        return 1;
    }

    function castVote(uint256, uint8) external pure override returns (uint256) {
        return 100;
    }

    function castVoteWithReason(
        uint256,
        uint8,
        string calldata
    ) external pure override returns (uint256) {
        return 100;
    }

    function castVoteWithReasonAndParams(
        uint256,
        uint8,
        string calldata,
        bytes memory
    ) external pure override returns (uint256) {
        return 100;
    }

    function castVoteBySig(
        uint256,
        uint8,
        address,
        bytes memory
    ) external pure override returns (uint256) {
        return 100;
    }

    function castVoteWithReasonAndParamsBySig(
        uint256,
        uint8,
        address,
        string calldata,
        bytes memory,
        bytes memory
    ) external pure override returns (uint256) {
        return 100;
    }
}
