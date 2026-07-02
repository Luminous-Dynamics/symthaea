// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {ContributionRegistry} from "../src/ContributionRegistry.sol";
import {ModelRegistry} from "../src/ModelRegistry.sol";

contract ContributionRegistryTest is Test {
    ContributionRegistry public contributionRegistry;
    ModelRegistry public modelRegistry;

    address public owner = address(1);
    address public recorder = address(2);
    address public validator = address(3);
    address public participant1 = address(4);
    address public participant2 = address(5);
    address public participant3 = address(6);
    address public user1 = address(7);

    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");

    bytes32 public modelId1;
    bytes32 public modelId2;
    bytes32 public gradientHash1;
    bytes32 public gradientHash2;
    bytes32 public gradientHash3;

    function setUp() public {
        // Set block.timestamp to a reasonable value to avoid underflow in tests
        // that use (block.timestamp - X) for historical timestamps
        vm.warp(1000);

        // Deploy ModelRegistry first
        vm.prank(owner);
        modelRegistry = new ModelRegistry(owner);

        // Deploy ContributionRegistry with ModelRegistry
        vm.prank(owner);
        contributionRegistry = new ContributionRegistry(address(modelRegistry), owner);

        // Grant roles
        vm.startPrank(owner);
        contributionRegistry.grantRole(RECORDER_ROLE, recorder);
        contributionRegistry.grantRole(VALIDATOR_ROLE, validator);
        vm.stopPrank();

        // Setup test data
        modelId1 = keccak256("model:mycelix:test1");
        modelId2 = keccak256("model:mycelix:test2");
        gradientHash1 = keccak256("gradient_participant1");
        gradientHash2 = keccak256("gradient_participant2");
        gradientHash3 = keccak256("gradient_participant3");

        // Register models in ModelRegistry
        vm.startPrank(owner);
        modelRegistry.registerModel(modelId1, "ipfs://model1", keccak256("weights1"));
        modelRegistry.registerModel(modelId2, "ipfs://model2", keccak256("weights2"));
        vm.stopPrank();
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertTrue(contributionRegistry.hasRole(contributionRegistry.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(contributionRegistry.hasRole(GOVERNANCE_ROLE, owner));
        assertTrue(contributionRegistry.hasRole(RECORDER_ROLE, owner));
        assertTrue(contributionRegistry.hasRole(VALIDATOR_ROLE, owner));
        assertEq(address(contributionRegistry.modelRegistry()), address(modelRegistry));
        assertEq(contributionRegistry.totalContributions(), 0);
    }

    function test_Constructor_RevertZeroModelRegistry() public {
        vm.expectRevert(ContributionRegistry.ZeroAddress.selector);
        new ContributionRegistry(address(0), owner);
    }

    function test_Constructor_RevertZeroAdmin() public {
        vm.expectRevert(ContributionRegistry.ZeroAddress.selector);
        new ContributionRegistry(address(modelRegistry), address(0));
    }

    function test_Constructor_RevertInvalidModelRegistry() public {
        // EOA is not a valid contract
        vm.expectRevert(ContributionRegistry.InvalidModelRegistry.selector);
        new ContributionRegistry(address(0x1234), owner);
    }

    // ============ Contribution Recording Tests ============

    function test_RecordContribution() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            timestamp
        );

        ContributionRegistry.Contribution memory contribution = contributionRegistry.getContribution(
            modelId1,
            1,
            participant1
        );

        assertEq(contribution.modelId, modelId1);
        assertEq(contribution.round, 1);
        assertEq(contribution.participant, participant1);
        assertEq(contribution.gradientHash, gradientHash1);
        assertEq(contribution.timestamp, timestamp);
        assertTrue(contribution.exists);
        assertEq(contributionRegistry.totalContributions(), 1);
    }

    function test_RecordContribution_MultipleParticipants() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant3, gradientHash3, timestamp);
        vm.stopPrank();

        assertEq(contributionRegistry.totalContributions(), 3);
        assertEq(contributionRegistry.getRoundParticipantCount(modelId1, 1), 3);
    }

    function test_RecordContribution_MultipleRounds() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 2, participant1, keccak256("hash_r2"), timestamp);
        contributionRegistry.recordContribution(modelId1, 3, participant1, keccak256("hash_r3"), timestamp);
        vm.stopPrank();

        assertEq(contributionRegistry.getParticipantRoundCount(modelId1, participant1), 3);
    }

    function test_RecordContribution_RevertUnauthorized() public {
        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertModelNotFound() public {
        bytes32 nonExistentModel = keccak256("nonexistent");

        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.ModelNotFound.selector);
        contributionRegistry.recordContribution(
            nonExistentModel,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertModelNotActive() public {
        vm.prank(owner);
        modelRegistry.deactivateModel(modelId1);

        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.ModelNotActive.selector);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertZeroParticipant() public {
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.ZeroAddress.selector);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            address(0),
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertInvalidGradientHash() public {
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.InvalidGradientHash.selector);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            bytes32(0),
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertInvalidTimestamp_Zero() public {
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.InvalidTimestamp.selector);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            0
        );
    }

    function test_RecordContribution_RevertInvalidTimestamp_Future() public {
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.InvalidTimestamp.selector);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp + 1000
        );
    }

    function test_RecordContribution_RevertInvalidRound() public {
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.InvalidRound.selector);
        contributionRegistry.recordContribution(
            modelId1,
            0,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_RecordContribution_RevertAlreadyExists() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.expectRevert(ContributionRegistry.ContributionAlreadyExists.selector);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash2, timestamp);
        vm.stopPrank();
    }

    // ============ Batch Operations Tests ============

    function test_RecordContributionsBatch() public {
        uint256 timestamp = block.timestamp - 100;

        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;

        bytes32[] memory gradientHashes = new bytes32[](3);
        gradientHashes[0] = gradientHash1;
        gradientHashes[1] = gradientHash2;
        gradientHashes[2] = gradientHash3;

        uint256[] memory timestamps = new uint256[](3);
        timestamps[0] = timestamp;
        timestamps[1] = timestamp;
        timestamps[2] = timestamp;

        vm.prank(recorder);
        contributionRegistry.recordContributionsBatch(
            modelId1,
            1,
            participants,
            gradientHashes,
            timestamps
        );

        assertEq(contributionRegistry.totalContributions(), 3);
        assertEq(contributionRegistry.getRoundParticipantCount(modelId1, 1), 3);

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));
        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant2));
        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant3));
    }

    function test_RecordContributionsBatch_RevertArrayLengthMismatch() public {
        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;

        bytes32[] memory gradientHashes = new bytes32[](2); // Mismatched length
        gradientHashes[0] = gradientHash1;
        gradientHashes[1] = gradientHash2;

        uint256[] memory timestamps = new uint256[](3);
        timestamps[0] = block.timestamp - 100;
        timestamps[1] = block.timestamp - 100;
        timestamps[2] = block.timestamp - 100;

        vm.prank(recorder);
        vm.expectRevert("Array length mismatch");
        contributionRegistry.recordContributionsBatch(
            modelId1,
            1,
            participants,
            gradientHashes,
            timestamps
        );
    }

    function test_RecordContributionsBatch_RevertUnauthorized() public {
        address[] memory participants = new address[](1);
        participants[0] = participant1;

        bytes32[] memory gradientHashes = new bytes32[](1);
        gradientHashes[0] = gradientHash1;

        uint256[] memory timestamps = new uint256[](1);
        timestamps[0] = block.timestamp - 100;

        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.recordContributionsBatch(
            modelId1,
            1,
            participants,
            gradientHashes,
            timestamps
        );
    }

    // ============ Validation Marking Tests ============

    function test_MarkContributionValid_Valid() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.prank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        ContributionRegistry.ValidationStatus memory status = contributionRegistry.getValidationStatus(
            modelId1,
            1,
            participant1
        );

        assertTrue(status.validated);
        assertTrue(status.isValid);
        assertEq(status.validatedBy, validator);
        assertTrue(status.validatedAt > 0);
    }

    function test_MarkContributionValid_Invalid() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.prank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, false);

        ContributionRegistry.ValidationStatus memory status = contributionRegistry.getValidationStatus(
            modelId1,
            1,
            participant1
        );

        assertTrue(status.validated);
        assertFalse(status.isValid);
    }

    function test_MarkContributionValid_UpdatesStats() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        vm.stopPrank();

        vm.startPrank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);
        contributionRegistry.markContributionValid(modelId1, 1, participant2, false);
        vm.stopPrank();

        // Check participant stats
        ContributionRegistry.ParticipantStats memory p1Stats = contributionRegistry.getParticipantStatsDetailed(
            modelId1,
            participant1
        );
        assertEq(p1Stats.validContributions, 1);
        assertEq(p1Stats.invalidContributions, 0);

        ContributionRegistry.ParticipantStats memory p2Stats = contributionRegistry.getParticipantStatsDetailed(
            modelId1,
            participant2
        );
        assertEq(p2Stats.validContributions, 0);
        assertEq(p2Stats.invalidContributions, 1);

        // Check round stats
        ContributionRegistry.RoundStats memory roundStats = contributionRegistry.getRoundStatsDetailed(modelId1, 1);
        assertEq(roundStats.validatedCount, 2);
        assertEq(roundStats.validCount, 1);
        assertEq(roundStats.invalidCount, 1);
    }

    function test_MarkContributionValid_RevertUnauthorized() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);
    }

    function test_MarkContributionValid_RevertContributionNotFound() public {
        vm.prank(validator);
        vm.expectRevert(ContributionRegistry.ContributionNotFound.selector);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);
    }

    function test_MarkContributionValid_RevertAlreadyValidated() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.startPrank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        vm.expectRevert(ContributionRegistry.AlreadyValidated.selector);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, false);
        vm.stopPrank();
    }

    function test_MarkContributionsValidBatch() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant3, gradientHash3, timestamp);
        vm.stopPrank();

        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;

        bool[] memory validities = new bool[](3);
        validities[0] = true;
        validities[1] = true;
        validities[2] = false;

        vm.prank(validator);
        contributionRegistry.markContributionsValidBatch(modelId1, 1, participants, validities);

        (bool validated1, bool isValid1) = contributionRegistry.isContributionValidated(modelId1, 1, participant1);
        (bool validated2, bool isValid2) = contributionRegistry.isContributionValidated(modelId1, 1, participant2);
        (bool validated3, bool isValid3) = contributionRegistry.isContributionValidated(modelId1, 1, participant3);

        assertTrue(validated1);
        assertTrue(isValid1);
        assertTrue(validated2);
        assertTrue(isValid2);
        assertTrue(validated3);
        assertFalse(isValid3);
    }

    function test_MarkContributionsValidBatch_RevertArrayLengthMismatch() public {
        address[] memory participants = new address[](2);
        participants[0] = participant1;
        participants[1] = participant2;

        bool[] memory validities = new bool[](3); // Mismatched
        validities[0] = true;
        validities[1] = true;
        validities[2] = false;

        vm.prank(validator);
        vm.expectRevert("Array length mismatch");
        contributionRegistry.markContributionsValidBatch(modelId1, 1, participants, validities);
    }

    // ============ Statistics Query Tests ============

    function test_GetParticipantStats() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 2, participant1, keccak256("hash2"), timestamp);
        contributionRegistry.recordContribution(modelId1, 3, participant1, keccak256("hash3"), timestamp);
        vm.stopPrank();

        (uint256 totalRounds, uint256 totalContributions) = contributionRegistry.getParticipantStats(
            modelId1,
            participant1
        );

        assertEq(totalRounds, 3);
        assertEq(totalContributions, 3);
    }

    function test_GetParticipantStatsDetailed() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 2, participant1, keccak256("hash2"), timestamp);
        vm.stopPrank();

        vm.startPrank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);
        contributionRegistry.markContributionValid(modelId1, 2, participant1, false);
        vm.stopPrank();

        ContributionRegistry.ParticipantStats memory stats = contributionRegistry.getParticipantStatsDetailed(
            modelId1,
            participant1
        );

        assertEq(stats.totalRounds, 2);
        assertEq(stats.totalContributions, 2);
        assertEq(stats.validContributions, 1);
        assertEq(stats.invalidContributions, 1);
        assertTrue(stats.firstContributionAt > 0);
        assertTrue(stats.lastContributionAt > 0);
    }

    function test_GetRoundStats() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant3, gradientHash3, timestamp);
        vm.stopPrank();

        (uint256 participantCount, uint256 contributionCount) = contributionRegistry.getRoundStats(modelId1, 1);

        assertEq(participantCount, 3);
        assertEq(contributionCount, 3);
    }

    function test_GetRoundStatsDetailed() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        vm.stopPrank();

        vm.startPrank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);
        contributionRegistry.markContributionValid(modelId1, 1, participant2, false);
        vm.stopPrank();

        ContributionRegistry.RoundStats memory stats = contributionRegistry.getRoundStatsDetailed(modelId1, 1);

        assertEq(stats.participantCount, 2);
        assertEq(stats.contributionCount, 2);
        assertEq(stats.validatedCount, 2);
        assertEq(stats.validCount, 1);
        assertEq(stats.invalidCount, 1);
    }

    // ============ ModelRegistry Integration Tests ============

    function test_Integration_ModelDeactivationBlocksContributions() public {
        // Record a contribution while model is active
        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, block.timestamp - 100);

        // Deactivate model
        vm.prank(owner);
        modelRegistry.deactivateModel(modelId1);

        // New contributions should be blocked
        vm.prank(recorder);
        vm.expectRevert(ContributionRegistry.ModelNotActive.selector);
        contributionRegistry.recordContribution(modelId1, 2, participant2, gradientHash2, block.timestamp - 50);

        // Reactivate model
        vm.prank(owner);
        modelRegistry.reactivateModel(modelId1);

        // Contributions should work again
        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 2, participant2, gradientHash2, block.timestamp - 50);

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 2, participant2));
    }

    function test_Integration_DifferentModels() public {
        uint256 timestamp = block.timestamp - 100;

        // Record contributions for different models
        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId2, 1, participant1, keccak256("model2_hash"), timestamp);
        vm.stopPrank();

        // Verify contributions are separate
        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));
        assertTrue(contributionRegistry.contributionExistsCheck(modelId2, 1, participant1));

        // Stats should be separate per model
        (uint256 rounds1, ) = contributionRegistry.getParticipantStats(modelId1, participant1);
        (uint256 rounds2, ) = contributionRegistry.getParticipantStats(modelId2, participant1);

        assertEq(rounds1, 1);
        assertEq(rounds2, 1);
    }

    // ============ Access Control Tests ============

    function test_AccessControl_RecorderRole() public {
        // User without role cannot record
        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );

        // Grant role
        vm.prank(owner);
        contributionRegistry.grantRole(RECORDER_ROLE, user1);

        // Now can record
        vm.prank(user1);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));
    }

    function test_AccessControl_ValidatorRole() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        // User without role cannot validate
        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        // Grant role
        vm.prank(owner);
        contributionRegistry.grantRole(VALIDATOR_ROLE, user1);

        // Now can validate
        vm.prank(user1);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        (bool validated, ) = contributionRegistry.isContributionValidated(modelId1, 1, participant1);
        assertTrue(validated);
    }

    // ============ Pause/Unpause Tests ============

    function test_Pause() public {
        vm.prank(owner);
        contributionRegistry.pause();

        assertTrue(contributionRegistry.paused());

        vm.prank(recorder);
        vm.expectRevert();
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );
    }

    function test_Unpause() public {
        vm.startPrank(owner);
        contributionRegistry.pause();
        contributionRegistry.unpause();
        vm.stopPrank();

        assertFalse(contributionRegistry.paused());

        vm.prank(recorder);
        contributionRegistry.recordContribution(
            modelId1,
            1,
            participant1,
            gradientHash1,
            block.timestamp - 100
        );

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));
    }

    function test_Pause_RevertUnauthorized() public {
        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.pause();
    }

    function test_Unpause_RevertUnauthorized() public {
        vm.prank(owner);
        contributionRegistry.pause();

        vm.prank(user1);
        vm.expectRevert();
        contributionRegistry.unpause();
    }

    function test_Pause_BlocksAllMutations() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        vm.prank(owner);
        contributionRegistry.pause();

        // Recording blocked
        vm.prank(recorder);
        vm.expectRevert();
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);

        // Validation blocked
        vm.prank(validator);
        vm.expectRevert();
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        // Batch recording blocked
        address[] memory participants = new address[](1);
        participants[0] = participant2;
        bytes32[] memory hashes = new bytes32[](1);
        hashes[0] = gradientHash2;
        uint256[] memory timestamps = new uint256[](1);
        timestamps[0] = timestamp;

        vm.prank(recorder);
        vm.expectRevert();
        contributionRegistry.recordContributionsBatch(modelId1, 1, participants, hashes, timestamps);
    }

    // ============ View Function Tests ============

    function test_GetContribution() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        ContributionRegistry.Contribution memory contribution = contributionRegistry.getContribution(
            modelId1,
            1,
            participant1
        );

        assertEq(contribution.modelId, modelId1);
        assertEq(contribution.round, 1);
        assertEq(contribution.participant, participant1);
        assertEq(contribution.gradientHash, gradientHash1);
    }

    function test_GetContributionsByRound() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        vm.stopPrank();

        ContributionRegistry.Contribution[] memory contributions = contributionRegistry.getContributionsByRound(
            modelId1,
            1
        );

        assertEq(contributions.length, 2);
    }

    function test_GetContributionsByParticipant() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 2, participant1, keccak256("hash2"), timestamp);
        contributionRegistry.recordContribution(modelId1, 3, participant1, keccak256("hash3"), timestamp);
        vm.stopPrank();

        ContributionRegistry.Contribution[] memory contributions = contributionRegistry.getContributionsByParticipant(
            modelId1,
            participant1
        );

        assertEq(contributions.length, 3);
    }

    function test_GetRoundParticipants() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, timestamp);
        vm.stopPrank();

        address[] memory participants = contributionRegistry.getRoundParticipants(modelId1, 1);

        assertEq(participants.length, 2);
        assertEq(participants[0], participant1);
        assertEq(participants[1], participant2);
    }

    function test_GetParticipantRounds() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);
        contributionRegistry.recordContribution(modelId1, 5, participant1, keccak256("hash5"), timestamp);
        contributionRegistry.recordContribution(modelId1, 10, participant1, keccak256("hash10"), timestamp);
        vm.stopPrank();

        uint256[] memory rounds = contributionRegistry.getParticipantRounds(modelId1, participant1);

        assertEq(rounds.length, 3);
        assertEq(rounds[0], 1);
        assertEq(rounds[1], 5);
        assertEq(rounds[2], 10);
    }

    function test_ContributionExistsCheck() public {
        assertFalse(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, block.timestamp - 100);

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, 1, participant1));
    }

    function test_IsContributionValidated() public {
        uint256 timestamp = block.timestamp - 100;

        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, timestamp);

        (bool validated1, ) = contributionRegistry.isContributionValidated(modelId1, 1, participant1);
        assertFalse(validated1);

        vm.prank(validator);
        contributionRegistry.markContributionValid(modelId1, 1, participant1, true);

        (bool validated2, bool isValid2) = contributionRegistry.isContributionValidated(modelId1, 1, participant1);
        assertTrue(validated2);
        assertTrue(isValid2);
    }

    // ============ Edge Cases ============

    function test_EdgeCase_EmptyRoundQueries() public {
        address[] memory participants = contributionRegistry.getRoundParticipants(modelId1, 999);
        assertEq(participants.length, 0);

        ContributionRegistry.Contribution[] memory contributions = contributionRegistry.getContributionsByRound(
            modelId1,
            999
        );
        assertEq(contributions.length, 0);
    }

    function test_EdgeCase_EmptyParticipantQueries() public {
        uint256[] memory rounds = contributionRegistry.getParticipantRounds(modelId1, participant1);
        assertEq(rounds.length, 0);

        ContributionRegistry.Contribution[] memory contributions = contributionRegistry.getContributionsByParticipant(
            modelId1,
            participant1
        );
        assertEq(contributions.length, 0);
    }

    function test_EdgeCase_NonExistentContributionValidation() public {
        ContributionRegistry.ValidationStatus memory status = contributionRegistry.getValidationStatus(
            modelId1,
            999,
            participant1
        );

        assertFalse(status.validated);
        assertFalse(status.isValid);
        assertEq(status.validatedBy, address(0));
    }

    function test_EdgeCase_SameParticipantDifferentRounds() public {
        uint256 timestamp = block.timestamp - 100;

        vm.startPrank(recorder);
        // Participant can contribute to multiple rounds with different hashes
        contributionRegistry.recordContribution(modelId1, 1, participant1, keccak256("r1"), timestamp);
        contributionRegistry.recordContribution(modelId1, 2, participant1, keccak256("r2"), timestamp);
        contributionRegistry.recordContribution(modelId1, 3, participant1, keccak256("r3"), timestamp);
        vm.stopPrank();

        // Each round should have its own contribution
        ContributionRegistry.Contribution memory c1 = contributionRegistry.getContribution(modelId1, 1, participant1);
        ContributionRegistry.Contribution memory c2 = contributionRegistry.getContribution(modelId1, 2, participant1);
        ContributionRegistry.Contribution memory c3 = contributionRegistry.getContribution(modelId1, 3, participant1);

        assertEq(c1.gradientHash, keccak256("r1"));
        assertEq(c2.gradientHash, keccak256("r2"));
        assertEq(c3.gradientHash, keccak256("r3"));
    }

    // ============ Fuzz Tests ============

    function testFuzz_RecordContribution(uint256 round, uint256 timestamp) public {
        vm.assume(round > 0 && round < type(uint256).max);
        vm.assume(timestamp > 0 && timestamp <= block.timestamp);

        vm.prank(recorder);
        contributionRegistry.recordContribution(
            modelId1,
            round,
            participant1,
            gradientHash1,
            timestamp
        );

        assertTrue(contributionRegistry.contributionExistsCheck(modelId1, round, participant1));
    }

    function testFuzz_BatchRecording(uint8 count) public {
        vm.assume(count > 0 && count <= 50);

        address[] memory participants = new address[](count);
        bytes32[] memory gradientHashes = new bytes32[](count);
        uint256[] memory timestamps = new uint256[](count);

        for (uint8 i = 0; i < count; i++) {
            participants[i] = address(uint160(1000 + i));
            gradientHashes[i] = keccak256(abi.encodePacked("gradient_", i));
            timestamps[i] = block.timestamp - 100;
        }

        vm.prank(recorder);
        contributionRegistry.recordContributionsBatch(
            modelId1,
            1,
            participants,
            gradientHashes,
            timestamps
        );

        assertEq(contributionRegistry.getRoundParticipantCount(modelId1, 1), count);
    }

    function testFuzz_ValidationStats(uint8 validCount, uint8 invalidCount) public {
        vm.assume(validCount <= 25 && invalidCount <= 25);
        uint256 total = uint256(validCount) + uint256(invalidCount);
        vm.assume(total > 0);

        // Record contributions
        for (uint256 i = 0; i < total; i++) {
            vm.prank(recorder);
            contributionRegistry.recordContribution(
                modelId1,
                1,
                address(uint160(2000 + i)),
                keccak256(abi.encodePacked("hash_", i)),
                block.timestamp - 100
            );
        }

        // Validate contributions
        for (uint256 i = 0; i < validCount; i++) {
            vm.prank(validator);
            contributionRegistry.markContributionValid(
                modelId1,
                1,
                address(uint160(2000 + i)),
                true
            );
        }

        for (uint256 i = 0; i < invalidCount; i++) {
            vm.prank(validator);
            contributionRegistry.markContributionValid(
                modelId1,
                1,
                address(uint160(2000 + validCount + i)),
                false
            );
        }

        ContributionRegistry.RoundStats memory stats = contributionRegistry.getRoundStatsDetailed(modelId1, 1);

        assertEq(stats.participantCount, total);
        assertEq(stats.validatedCount, total);
        assertEq(stats.validCount, validCount);
        assertEq(stats.invalidCount, invalidCount);
    }

    // ============ M-02 Pagination Tests ============

    function test_GetContributionsByRoundPaginated() public {
        // Record 5 contributions
        for (uint256 i = 0; i < 5; i++) {
            vm.prank(recorder);
            contributionRegistry.recordContribution(
                modelId1,
                1,
                address(uint160(100 + i)),
                keccak256(abi.encodePacked("hash_", i)),
                block.timestamp - 100
            );
        }

        // Get first page (offset 0, limit 2)
        (ContributionRegistry.Contribution[] memory page1, uint256 total1, bool hasMore1) =
            contributionRegistry.getContributionsByRoundPaginated(modelId1, 1, 0, 2);

        assertEq(page1.length, 2, "First page should have 2 items");
        assertEq(total1, 5, "Total should be 5");
        assertTrue(hasMore1, "Should have more pages");

        // Get second page (offset 2, limit 2)
        (ContributionRegistry.Contribution[] memory page2, uint256 total2, bool hasMore2) =
            contributionRegistry.getContributionsByRoundPaginated(modelId1, 1, 2, 2);

        assertEq(page2.length, 2, "Second page should have 2 items");
        assertEq(total2, 5, "Total should still be 5");
        assertTrue(hasMore2, "Should have more pages");

        // Get last page (offset 4, limit 2)
        (ContributionRegistry.Contribution[] memory page3, uint256 total3, bool hasMore3) =
            contributionRegistry.getContributionsByRoundPaginated(modelId1, 1, 4, 2);

        assertEq(page3.length, 1, "Last page should have 1 item");
        assertEq(total3, 5, "Total should still be 5");
        assertFalse(hasMore3, "Should not have more pages");
    }

    function test_GetContributionsByParticipantPaginated() public {
        // Record contributions across 4 rounds for participant1
        for (uint256 round = 1; round <= 4; round++) {
            vm.prank(recorder);
            contributionRegistry.recordContribution(
                modelId1,
                round,
                participant1,
                keccak256(abi.encodePacked("hash_round_", round)),
                block.timestamp - 100
            );
        }

        // Get paginated results
        (ContributionRegistry.Contribution[] memory page1, uint256 total, bool hasMore) =
            contributionRegistry.getContributionsByParticipantPaginated(modelId1, participant1, 0, 2);

        assertEq(page1.length, 2, "Page should have 2 items");
        assertEq(total, 4, "Total should be 4");
        assertTrue(hasMore, "Should have more pages");

        // Get remaining
        (ContributionRegistry.Contribution[] memory page2, , bool hasMore2) =
            contributionRegistry.getContributionsByParticipantPaginated(modelId1, participant1, 2, 10);

        assertEq(page2.length, 2, "Remaining page should have 2 items");
        assertFalse(hasMore2, "Should not have more pages");
    }

    function test_GetRoundParticipantsPaginated() public {
        // Record contributions from 3 participants
        address[] memory participants = new address[](3);
        participants[0] = participant1;
        participants[1] = participant2;
        participants[2] = participant3;

        for (uint256 i = 0; i < 3; i++) {
            vm.prank(recorder);
            contributionRegistry.recordContribution(
                modelId1,
                1,
                participants[i],
                keccak256(abi.encodePacked("hash_", i)),
                block.timestamp - 100
            );
        }

        // Get paginated participants
        (address[] memory page1, uint256 total, bool hasMore) =
            contributionRegistry.getRoundParticipantsPaginated(modelId1, 1, 0, 2);

        assertEq(page1.length, 2, "Page should have 2 participants");
        assertEq(total, 3, "Total should be 3");
        assertTrue(hasMore, "Should have more pages");

        // Get last participant
        (address[] memory page2, , bool hasMore2) =
            contributionRegistry.getRoundParticipantsPaginated(modelId1, 1, 2, 10);

        assertEq(page2.length, 1, "Last page should have 1 participant");
        assertFalse(hasMore2, "Should not have more pages");
    }

    function test_GetParticipantRoundsPaginated() public {
        // Record contributions across 5 rounds
        for (uint256 round = 1; round <= 5; round++) {
            vm.prank(recorder);
            contributionRegistry.recordContribution(
                modelId1,
                round,
                participant1,
                keccak256(abi.encodePacked("hash_", round)),
                block.timestamp - 100
            );
        }

        // Get paginated rounds
        (uint256[] memory page1, uint256 total, bool hasMore) =
            contributionRegistry.getParticipantRoundsPaginated(modelId1, participant1, 0, 3);

        assertEq(page1.length, 3, "Page should have 3 rounds");
        assertEq(total, 5, "Total should be 5");
        assertTrue(hasMore, "Should have more pages");

        // Verify round numbers
        assertEq(page1[0], 1, "First round should be 1");
        assertEq(page1[1], 2, "Second round should be 2");
        assertEq(page1[2], 3, "Third round should be 3");
    }

    function test_Pagination_EmptyResults() public {
        // Query non-existent data
        (ContributionRegistry.Contribution[] memory empty, uint256 total, bool hasMore) =
            contributionRegistry.getContributionsByRoundPaginated(modelId1, 999, 0, 10);

        assertEq(empty.length, 0, "Should return empty array");
        assertEq(total, 0, "Total should be 0");
        assertFalse(hasMore, "Should not have more pages");
    }

    function test_Pagination_OffsetBeyondTotal() public {
        // Record 2 contributions
        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant1, gradientHash1, block.timestamp - 100);
        vm.prank(recorder);
        contributionRegistry.recordContribution(modelId1, 1, participant2, gradientHash2, block.timestamp - 100);

        // Query with offset beyond total
        (ContributionRegistry.Contribution[] memory result, uint256 total, bool hasMore) =
            contributionRegistry.getContributionsByRoundPaginated(modelId1, 1, 10, 5);

        assertEq(result.length, 0, "Should return empty array");
        assertEq(total, 2, "Total should still be 2");
        assertFalse(hasMore, "Should not have more pages");
    }
}
