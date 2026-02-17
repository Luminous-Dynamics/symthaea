// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {ModelRegistry} from "../src/ModelRegistry.sol";

contract ModelRegistryTest is Test {
    ModelRegistry public registry;

    address public owner = address(1);
    address public operator = address(2);
    address public recorder = address(3);
    address public user1 = address(4);
    address public user2 = address(5);
    address public validator1 = address(6);
    address public validator2 = address(7);
    address public validator3 = address(8);

    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");

    bytes32 public modelId1;
    bytes32 public modelId2;
    bytes32 public weightsHash1;
    bytes32 public weightsHash2;
    string public metadata1 = "ipfs://QmTestModel1Metadata";
    string public metadata2 = "ipfs://QmTestModel2Metadata";

    function setUp() public {
        vm.prank(owner);
        registry = new ModelRegistry(owner);

        // Grant roles
        vm.startPrank(owner);
        registry.grantRole(OPERATOR_ROLE, operator);
        registry.grantRole(RECORDER_ROLE, recorder);
        vm.stopPrank();

        // Setup test data
        modelId1 = keccak256("model:mycelix:test1");
        modelId2 = keccak256("model:mycelix:test2");
        weightsHash1 = keccak256("initial_weights_v1");
        weightsHash2 = keccak256("initial_weights_v2");
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertTrue(registry.hasRole(registry.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(registry.hasRole(GOVERNANCE_ROLE, owner));
        assertTrue(registry.hasRole(OPERATOR_ROLE, owner));
        assertTrue(registry.hasRole(RECORDER_ROLE, owner));
        assertEq(registry.totalModels(), 0);
    }

    function test_Constructor_RevertZeroOwner() public {
        // OpenZeppelin's Ownable checks for zero address before our custom check
        vm.expectRevert(abi.encodeWithSignature("OwnableInvalidOwner(address)", address(0)));
        new ModelRegistry(address(0));
    }

    // ============ Model Registration Tests ============

    function test_RegisterModel() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        ModelRegistry.Model memory model = registry.getModel(modelId1);

        assertEq(model.modelId, modelId1);
        assertEq(model.owner, user1);
        assertEq(model.metadata, metadata1);
        assertEq(model.currentWeightsHash, weightsHash1);
        assertEq(model.currentRound, 0);
        assertTrue(model.active);
        assertTrue(model.createdAt > 0);
        assertEq(registry.totalModels(), 1);
    }

    function test_RegisterModel_Multiple() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        registry.registerModel(modelId2, metadata2, weightsHash2);

        assertEq(registry.totalModels(), 2);
        assertTrue(registry.modelExistsCheck(modelId1));
        assertTrue(registry.modelExistsCheck(modelId2));
    }

    function test_RegisterModel_RevertAlreadyExists() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.ModelAlreadyExists.selector);
        registry.registerModel(modelId1, metadata2, weightsHash2);
    }

    function test_RegisterModel_RevertInvalidModelId() public {
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidModelId.selector);
        registry.registerModel(bytes32(0), metadata1, weightsHash1);
    }

    function test_RegisterModel_RevertInvalidWeightsHash() public {
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidWeightsHash.selector);
        registry.registerModel(modelId1, metadata1, bytes32(0));
    }

    function test_RegisterModel_RevertInvalidMetadata() public {
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidMetadata.selector);
        registry.registerModel(modelId1, "", weightsHash1);
    }

    function test_RegisterModel_RevertMetadataTooLong() public {
        // Create a string longer than MAX_METADATA_LENGTH (2048)
        bytes memory longMetadata = new bytes(2049);
        for (uint256 i = 0; i < 2049; i++) {
            longMetadata[i] = "a";
        }

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.MetadataTooLong.selector);
        registry.registerModel(modelId1, string(longMetadata), weightsHash1);
    }

    function test_RegisterModel_RecordsInitialVersion() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        assertEq(registry.modelVersionCount(modelId1), 1);

        ModelRegistry.ModelVersion memory version = registry.getModelVersion(modelId1, 0);
        assertEq(version.weightsHash, weightsHash1);
        assertEq(version.round, 0);
        assertEq(version.updatedBy, user1);
    }

    // ============ Model Update Tests ============

    function test_UpdateModel_ByOwner() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 newWeightsHash = keccak256("updated_weights_v1");

        vm.prank(user1);
        registry.updateModel(modelId1, newWeightsHash, 1);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.currentWeightsHash, newWeightsHash);
        assertEq(model.currentRound, 1);
        assertEq(registry.modelVersionCount(modelId1), 2);
    }

    function test_UpdateModel_ByOperator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 newWeightsHash = keccak256("updated_weights_v1");

        vm.prank(operator);
        registry.updateModel(modelId1, newWeightsHash, 1);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.currentWeightsHash, newWeightsHash);
        assertEq(model.currentRound, 1);
    }

    function test_UpdateModel_MultipleRounds() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        for (uint256 i = 1; i <= 5; i++) {
            bytes32 newWeightsHash = keccak256(abi.encodePacked("weights_round_", i));
            vm.prank(user1);
            registry.updateModel(modelId1, newWeightsHash, i);
        }

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.currentRound, 5);
        assertEq(registry.modelVersionCount(modelId1), 6); // Initial + 5 updates
    }

    function test_UpdateModel_RevertNotOwnerOrOperator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 newWeightsHash = keccak256("updated_weights");

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.NotModelOwnerOrOperator.selector);
        registry.updateModel(modelId1, newWeightsHash, 1);
    }

    function test_UpdateModel_RevertModelNotFound() public {
        bytes32 newWeightsHash = keccak256("updated_weights");

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.updateModel(modelId1, newWeightsHash, 1);
    }

    function test_UpdateModel_RevertModelNotActive() public {
        vm.startPrank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);
        registry.deactivateModel(modelId1);
        vm.stopPrank();

        bytes32 newWeightsHash = keccak256("updated_weights");

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotActive.selector);
        registry.updateModel(modelId1, newWeightsHash, 1);
    }

    function test_UpdateModel_RevertInvalidWeightsHash() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidWeightsHash.selector);
        registry.updateModel(modelId1, bytes32(0), 1);
    }

    function test_UpdateModel_RevertInvalidRound() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 newWeightsHash = keccak256("updated_weights");

        // Try to update with same round (0)
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidRound.selector);
        registry.updateModel(modelId1, newWeightsHash, 0);
    }

    function test_UpdateModel_RevertRoundMustIncrease() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 newWeightsHash1 = keccak256("updated_weights_1");
        bytes32 newWeightsHash2 = keccak256("updated_weights_2");

        vm.startPrank(user1);
        registry.updateModel(modelId1, newWeightsHash1, 5);

        // Try to update with lower round
        vm.expectRevert(ModelRegistry.InvalidRound.selector);
        registry.updateModel(modelId1, newWeightsHash2, 3);
        vm.stopPrank();
    }

    // ============ Model Metadata Update Tests ============

    function test_UpdateMetadata() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        string memory newMetadata = "ipfs://QmNewMetadata";

        vm.prank(user1);
        registry.updateMetadata(modelId1, newMetadata);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.metadata, newMetadata);
    }

    function test_UpdateMetadata_RevertNotOwner() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.NotModelOwner.selector);
        registry.updateMetadata(modelId1, "new metadata");
    }

    function test_UpdateMetadata_RevertInvalidMetadata() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.InvalidMetadata.selector);
        registry.updateMetadata(modelId1, "");
    }

    function test_UpdateMetadata_RevertMetadataTooLong() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes memory longMetadata = new bytes(2049);
        for (uint256 i = 0; i < 2049; i++) {
            longMetadata[i] = "x";
        }

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.MetadataTooLong.selector);
        registry.updateMetadata(modelId1, string(longMetadata));
    }

    // ============ Model Ownership Transfer Tests ============

    function test_TransferModelOwnership() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.transferModelOwnership(modelId1, user2);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.owner, user2);
    }

    function test_TransferModelOwnership_RevertNotOwner() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.NotModelOwner.selector);
        registry.transferModelOwnership(modelId1, user2);
    }

    function test_TransferModelOwnership_RevertZeroAddress() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ZeroAddress.selector);
        registry.transferModelOwnership(modelId1, address(0));
    }

    function test_TransferModelOwnership_RevertSameAddress() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.SameAddress.selector);
        registry.transferModelOwnership(modelId1, user1);
    }

    // ============ Model Deactivation/Reactivation Tests ============

    function test_DeactivateModel() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.deactivateModel(modelId1);

        assertFalse(registry.isModelActive(modelId1));
    }

    function test_ReactivateModel() public {
        vm.startPrank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);
        registry.deactivateModel(modelId1);
        registry.reactivateModel(modelId1);
        vm.stopPrank();

        assertTrue(registry.isModelActive(modelId1));
    }

    function test_DeactivateModel_RevertNotOwner() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.NotModelOwner.selector);
        registry.deactivateModel(modelId1);
    }

    function test_EmergencyDeactivate() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(owner);
        registry.emergencyDeactivate(modelId1);

        assertFalse(registry.isModelActive(modelId1));
    }

    function test_EmergencyDeactivate_RevertUnauthorized() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert();
        registry.emergencyDeactivate(modelId1);
    }

    // ============ Validator Management Tests ============

    function test_AddValidator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.addValidator(modelId1, validator1);

        assertTrue(registry.isValidator(modelId1, validator1));
        assertEq(registry.getValidatorCount(modelId1), 1);
    }

    function test_AddValidator_Multiple() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.addValidator(modelId1, validator1);
        registry.addValidator(modelId1, validator2);
        registry.addValidator(modelId1, validator3);
        vm.stopPrank();

        address[] memory validators = registry.getValidators(modelId1);
        assertEq(validators.length, 3);
        assertTrue(registry.isValidator(modelId1, validator1));
        assertTrue(registry.isValidator(modelId1, validator2));
        assertTrue(registry.isValidator(modelId1, validator3));
    }

    function test_AddValidator_ByOperator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(operator);
        registry.addValidator(modelId1, validator1);

        assertTrue(registry.isValidator(modelId1, validator1));
    }

    function test_AddValidator_RevertZeroAddress() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ZeroAddress.selector);
        registry.addValidator(modelId1, address(0));
    }

    function test_AddValidator_RevertAlreadyExists() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.addValidator(modelId1, validator1);

        vm.expectRevert(ModelRegistry.ValidatorAlreadyExists.selector);
        registry.addValidator(modelId1, validator1);
        vm.stopPrank();
    }

    function test_AddValidator_RevertTooManyValidators() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        // Add max validators (100)
        for (uint256 i = 0; i < 100; i++) {
            registry.addValidator(modelId1, address(uint160(1000 + i)));
        }

        // Try to add one more
        vm.expectRevert(ModelRegistry.TooManyValidators.selector);
        registry.addValidator(modelId1, address(2000));
        vm.stopPrank();
    }

    function test_AddValidator_RevertNotActive() public {
        vm.startPrank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);
        registry.deactivateModel(modelId1);
        vm.stopPrank();

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotActive.selector);
        registry.addValidator(modelId1, validator1);
    }

    function test_RemoveValidator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.addValidator(modelId1, validator1);
        registry.removeValidator(modelId1, validator1);
        vm.stopPrank();

        assertFalse(registry.isValidator(modelId1, validator1));
        assertEq(registry.getValidatorCount(modelId1), 0);
    }

    function test_RemoveValidator_PreservesOrder() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.addValidator(modelId1, validator1);
        registry.addValidator(modelId1, validator2);
        registry.addValidator(modelId1, validator3);

        // Remove middle validator
        registry.removeValidator(modelId1, validator2);
        vm.stopPrank();

        address[] memory validators = registry.getValidators(modelId1);
        assertEq(validators.length, 2);
        assertFalse(registry.isValidator(modelId1, validator2));
        assertTrue(registry.isValidator(modelId1, validator1));
        assertTrue(registry.isValidator(modelId1, validator3));
    }

    function test_RemoveValidator_RevertNotFound() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ValidatorNotFound.selector);
        registry.removeValidator(modelId1, validator1);
    }

    function test_RemoveValidator_ByOperator() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.addValidator(modelId1, validator1);

        vm.prank(operator);
        registry.removeValidator(modelId1, validator1);

        assertFalse(registry.isValidator(modelId1, validator1));
    }

    // ============ Round Recording Tests ============

    function test_RecordRound() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        bytes32 aggregatedHash = keccak256("aggregated_weights_round_1");

        vm.prank(recorder);
        registry.recordRound(modelId1, 1, aggregatedHash, 10);

        ModelRegistry.Round memory round = registry.getRound(modelId1, 1);
        assertEq(round.roundNumber, 1);
        assertEq(round.aggregatedHash, aggregatedHash);
        assertEq(round.participantCount, 10);
        assertEq(round.recordedBy, recorder);
        assertTrue(round.finalized);
    }

    function test_RecordRound_Sequential() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(recorder);
        registry.recordRound(modelId1, 1, keccak256("hash_1"), 10);

        // Must update model round first
        vm.stopPrank();
        vm.prank(user1);
        registry.updateModel(modelId1, keccak256("weights_1"), 1);

        vm.prank(recorder);
        registry.recordRound(modelId1, 2, keccak256("hash_2"), 12);
        vm.stopPrank();

        ModelRegistry.Round memory round2 = registry.getRound(modelId1, 2);
        assertEq(round2.roundNumber, 2);
        assertEq(round2.participantCount, 12);
    }

    function test_RecordRound_RevertUnauthorized() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user2);
        vm.expectRevert();
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);
    }

    function test_RecordRound_RevertInvalidWeightsHash() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(recorder);
        vm.expectRevert(ModelRegistry.InvalidWeightsHash.selector);
        registry.recordRound(modelId1, 1, bytes32(0), 10);
    }

    function test_RecordRound_RevertInvalidParticipantCount() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(recorder);
        vm.expectRevert(ModelRegistry.InvalidParticipantCount.selector);
        registry.recordRound(modelId1, 1, keccak256("hash"), 0);
    }

    function test_RecordRound_RevertAlreadyRecorded() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(recorder);
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);

        vm.expectRevert(ModelRegistry.RoundAlreadyRecorded.selector);
        registry.recordRound(modelId1, 1, keccak256("hash2"), 15);
        vm.stopPrank();
    }

    function test_RecordRound_RevertNotSequential() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        // Try to record round 3 when round 1 is expected
        vm.prank(recorder);
        vm.expectRevert(ModelRegistry.RoundNotSequential.selector);
        registry.recordRound(modelId1, 3, keccak256("hash"), 10);
    }

    function test_RecordRound_RevertModelNotActive() public {
        vm.startPrank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);
        registry.deactivateModel(modelId1);
        vm.stopPrank();

        vm.prank(recorder);
        vm.expectRevert(ModelRegistry.ModelNotActive.selector);
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);
    }

    function test_GetLatestRound() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(recorder);
        registry.recordRound(modelId1, 1, keccak256("hash_1"), 10);

        (ModelRegistry.Round memory round, uint256 roundNumber) = registry.getLatestRound(modelId1);
        assertEq(roundNumber, 1);
        assertEq(round.participantCount, 10);
    }

    // ============ Access Control Tests ============

    function test_AccessControl_RecorderRole() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        // User without role cannot record
        vm.prank(user2);
        vm.expectRevert();
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);

        // Grant role
        vm.prank(owner);
        registry.grantRole(RECORDER_ROLE, user2);

        // Now can record
        vm.prank(user2);
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);
    }

    function test_AccessControl_OperatorRole() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        // User without role cannot update
        vm.prank(user2);
        vm.expectRevert(ModelRegistry.NotModelOwnerOrOperator.selector);
        registry.updateModel(modelId1, keccak256("hash"), 1);

        // Grant role
        vm.prank(owner);
        registry.grantRole(OPERATOR_ROLE, user2);

        // Now can update
        vm.prank(user2);
        registry.updateModel(modelId1, keccak256("hash"), 1);
    }

    // ============ Pause/Unpause Tests ============

    function test_Pause() public {
        vm.prank(owner);
        registry.pause();

        assertTrue(registry.paused());

        vm.prank(user1);
        vm.expectRevert();
        registry.registerModel(modelId1, metadata1, weightsHash1);
    }

    function test_Unpause() public {
        vm.startPrank(owner);
        registry.pause();
        registry.unpause();
        vm.stopPrank();

        assertFalse(registry.paused());

        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        assertTrue(registry.modelExistsCheck(modelId1));
    }

    function test_Pause_RevertUnauthorized() public {
        vm.prank(user1);
        vm.expectRevert();
        registry.pause();
    }

    function test_Unpause_RevertUnauthorized() public {
        vm.prank(owner);
        registry.pause();

        vm.prank(user1);
        vm.expectRevert();
        registry.unpause();
    }

    function test_Pause_BlocksAllOperations() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(owner);
        registry.pause();

        // Test various operations are blocked
        vm.startPrank(user1);

        vm.expectRevert();
        registry.updateModel(modelId1, keccak256("hash"), 1);

        vm.expectRevert();
        registry.updateMetadata(modelId1, "new metadata");

        vm.expectRevert();
        registry.addValidator(modelId1, validator1);

        vm.expectRevert();
        registry.transferModelOwnership(modelId1, user2);

        vm.stopPrank();

        vm.prank(recorder);
        vm.expectRevert();
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);
    }

    // ============ View Function Tests ============

    function test_GetModel_RevertNotFound() public {
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.getModel(modelId1);
    }

    function test_GetModelHistory() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.updateModel(modelId1, keccak256("weights_1"), 1);
        registry.updateModel(modelId1, keccak256("weights_2"), 2);
        vm.stopPrank();

        ModelRegistry.ModelVersion[] memory history = registry.getModelHistory(modelId1);
        assertEq(history.length, 3);
        assertEq(history[0].round, 0);
        assertEq(history[1].round, 1);
        assertEq(history[2].round, 2);
    }

    function test_GetModelVersion_RevertInvalidIndex() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.expectRevert(ModelRegistry.InvalidRound.selector);
        registry.getModelVersion(modelId1, 5);
    }

    function test_GetModelState() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.startPrank(user1);
        registry.addValidator(modelId1, validator1);
        registry.addValidator(modelId1, validator2);
        registry.updateModel(modelId1, keccak256("updated"), 1);
        vm.stopPrank();

        (
            bytes32 currentWeightsHash,
            uint256 currentRound,
            bool active,
            uint256 validatorCount
        ) = registry.getModelState(modelId1);

        assertEq(currentWeightsHash, keccak256("updated"));
        assertEq(currentRound, 1);
        assertTrue(active);
        assertEq(validatorCount, 2);
    }

    function test_ModelExistsCheck() public {
        assertFalse(registry.modelExistsCheck(modelId1));

        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        assertTrue(registry.modelExistsCheck(modelId1));
    }

    function test_IsModelActive() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        assertTrue(registry.isModelActive(modelId1));

        vm.prank(user1);
        registry.deactivateModel(modelId1);

        assertFalse(registry.isModelActive(modelId1));
    }

    // ============ Edge Cases ============

    function test_EdgeCase_RegisterAfterDeactivate() public {
        // Model ID should be unique - cannot re-register even if deactivated
        vm.startPrank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);
        registry.deactivateModel(modelId1);
        vm.stopPrank();

        vm.prank(user2);
        vm.expectRevert(ModelRegistry.ModelAlreadyExists.selector);
        registry.registerModel(modelId1, metadata2, weightsHash2);
    }

    function test_EdgeCase_TransferThenUpdate() public {
        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.transferModelOwnership(modelId1, user2);

        // Old owner cannot update
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.NotModelOwnerOrOperator.selector);
        registry.updateModel(modelId1, keccak256("hash"), 1);

        // New owner can update
        vm.prank(user2);
        registry.updateModel(modelId1, keccak256("hash"), 1);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.currentRound, 1);
    }

    function test_EdgeCase_ValidatorOperationsOnNonexistentModel() public {
        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.addValidator(modelId1, validator1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.removeValidator(modelId1, validator1);

        vm.prank(user1);
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.getValidators(modelId1);
    }

    function test_EdgeCase_RoundOperationsOnNonexistentModel() public {
        vm.prank(recorder);
        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.recordRound(modelId1, 1, keccak256("hash"), 10);

        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.getRound(modelId1, 1);

        vm.expectRevert(ModelRegistry.ModelNotFound.selector);
        registry.getLatestRound(modelId1);
    }

    // ============ Fuzz Tests ============

    function testFuzz_RegisterModel(bytes32 _modelId, bytes32 _weightsHash) public {
        vm.assume(_modelId != bytes32(0));
        vm.assume(_weightsHash != bytes32(0));

        vm.prank(user1);
        registry.registerModel(_modelId, metadata1, _weightsHash);

        assertTrue(registry.modelExistsCheck(_modelId));
    }

    function testFuzz_UpdateModel(uint256 round) public {
        vm.assume(round > 0 && round < type(uint256).max);

        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(user1);
        registry.updateModel(modelId1, keccak256(abi.encodePacked(round)), round);

        ModelRegistry.Model memory model = registry.getModel(modelId1);
        assertEq(model.currentRound, round);
    }

    function testFuzz_RecordRound(uint256 participantCount) public {
        vm.assume(participantCount > 0 && participantCount < 10000);

        vm.prank(user1);
        registry.registerModel(modelId1, metadata1, weightsHash1);

        vm.prank(recorder);
        registry.recordRound(modelId1, 1, keccak256("hash"), participantCount);

        ModelRegistry.Round memory round = registry.getRound(modelId1, 1);
        assertEq(round.participantCount, participantCount);
    }
}
