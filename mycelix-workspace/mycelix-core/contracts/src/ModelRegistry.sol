// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title ModelRegistry
 * @author Mycelix Network
 * @notice Registry for tracking Federated Learning (FL) model state on-chain
 * @dev Provides model lifecycle management, validator coordination, and round tracking
 *
 * Features:
 * - Model registration with metadata and initial weights hash
 * - Version history tracking for model updates
 * - Validator management per model
 * - Round tracking with aggregated weights and participant counts
 */
contract ModelRegistry is Ownable, AccessControl, ReentrancyGuard, Pausable {
    // ============ Constants ============

    /// @notice Role for governance actions
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    /// @notice Role for model operators (can update models they own)
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    /// @notice Role for round recorders (FL aggregation nodes)
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");

    /// @notice Maximum validators per model
    uint256 public constant MAX_VALIDATORS = 100;

    /// @notice Maximum metadata length
    uint256 public constant MAX_METADATA_LENGTH = 2048;

    // ============ Structs ============

    /// @notice Model record containing core model information
    struct Model {
        bytes32 modelId;
        address owner;
        string metadata;
        bytes32 currentWeightsHash;
        uint256 currentRound;
        uint256 createdAt;
        uint256 updatedAt;
        bool active;
    }

    /// @notice Model version record for tracking history
    struct ModelVersion {
        bytes32 weightsHash;
        uint256 round;
        uint256 timestamp;
        address updatedBy;
    }

    /// @notice Round record containing aggregation results
    struct Round {
        uint256 roundNumber;
        bytes32 aggregatedHash;
        uint256 participantCount;
        uint256 timestamp;
        address recordedBy;
        bool finalized;
    }

    // ============ State Variables ============

    /// @notice Mapping of model ID to Model record
    mapping(bytes32 => Model) public models;

    /// @notice Mapping of model ID to version history (version index => ModelVersion)
    mapping(bytes32 => mapping(uint256 => ModelVersion)) public modelVersions;

    /// @notice Mapping of model ID to version count
    mapping(bytes32 => uint256) public modelVersionCount;

    /// @notice Mapping of model ID to validators (validator address => is validator)
    mapping(bytes32 => mapping(address => bool)) public modelValidators;

    /// @notice Mapping of model ID to validator list
    mapping(bytes32 => address[]) internal _validatorLists;

    /// @notice Mapping of model ID to round data (round number => Round)
    mapping(bytes32 => mapping(uint256 => Round)) public rounds;

    /// @notice Total registered models
    uint256 public totalModels;

    // ============ Events ============

    /// @notice Emitted when a new model is registered
    event ModelRegistered(
        bytes32 indexed modelId,
        address indexed owner,
        string metadata,
        bytes32 initialWeightsHash,
        uint256 timestamp
    );

    /// @notice Emitted when a model is updated
    event ModelUpdated(
        bytes32 indexed modelId,
        bytes32 indexed newWeightsHash,
        uint256 indexed round,
        address updatedBy,
        uint256 timestamp
    );

    /// @notice Emitted when a model is deactivated
    event ModelDeactivated(
        bytes32 indexed modelId,
        address indexed deactivatedBy,
        uint256 timestamp
    );

    /// @notice Emitted when a model is reactivated
    event ModelReactivated(
        bytes32 indexed modelId,
        address indexed reactivatedBy,
        uint256 timestamp
    );

    /// @notice Emitted when a validator is added to a model
    event ValidatorAdded(
        bytes32 indexed modelId,
        address indexed validator,
        address indexed addedBy,
        uint256 timestamp
    );

    /// @notice Emitted when a validator is removed from a model
    event ValidatorRemoved(
        bytes32 indexed modelId,
        address indexed validator,
        address indexed removedBy,
        uint256 timestamp
    );

    /// @notice Emitted when a round is recorded
    event RoundRecorded(
        bytes32 indexed modelId,
        uint256 indexed round,
        bytes32 aggregatedHash,
        uint256 participantCount,
        address indexed recordedBy,
        uint256 timestamp
    );

    /// @notice Emitted when model ownership is transferred
    event ModelOwnershipTransferred(
        bytes32 indexed modelId,
        address indexed previousOwner,
        address indexed newOwner,
        uint256 timestamp
    );

    /// @notice Emitted when model metadata is updated
    event ModelMetadataUpdated(
        bytes32 indexed modelId,
        string newMetadata,
        uint256 timestamp
    );

    // ============ Errors ============

    error ModelAlreadyExists();
    error ModelNotFound();
    error ModelNotActive();
    error NotModelOwner();
    error NotModelOwnerOrOperator();
    error InvalidModelId();
    error InvalidWeightsHash();
    error InvalidMetadata();
    error MetadataTooLong();
    error ValidatorAlreadyExists();
    error ValidatorNotFound();
    error TooManyValidators();
    error InvalidRound();
    error RoundAlreadyRecorded();
    error RoundNotSequential();
    error ZeroAddress();
    error SameAddress();
    error InvalidParticipantCount();

    // ============ Modifiers ============

    /// @notice Ensures the caller is the model owner
    modifier onlyModelOwner(bytes32 modelId) {
        if (models[modelId].owner != msg.sender) {
            revert NotModelOwner();
        }
        _;
    }

    /// @notice Ensures the caller is the model owner or has operator role
    modifier onlyModelOwnerOrOperator(bytes32 modelId) {
        if (models[modelId].owner != msg.sender && !hasRole(OPERATOR_ROLE, msg.sender)) {
            revert NotModelOwnerOrOperator();
        }
        _;
    }

    /// @notice Ensures the model exists
    modifier modelExists(bytes32 modelId) {
        if (models[modelId].owner == address(0)) {
            revert ModelNotFound();
        }
        _;
    }

    /// @notice Ensures the model is active
    modifier modelActive(bytes32 modelId) {
        if (!models[modelId].active) {
            revert ModelNotActive();
        }
        _;
    }

    // ============ Constructor ============

    /**
     * @notice Initializes the ModelRegistry contract
     * @param initialOwner Address of the contract owner
     */
    constructor(address initialOwner) Ownable(initialOwner) {
        if (initialOwner == address(0)) {
            revert ZeroAddress();
        }

        _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
        _grantRole(GOVERNANCE_ROLE, initialOwner);
        _grantRole(OPERATOR_ROLE, initialOwner);
        _grantRole(RECORDER_ROLE, initialOwner);
    }

    // ============ Model Management Functions ============

    /**
     * @notice Registers a new FL model
     * @param modelId Unique identifier for the model
     * @param metadata Model metadata (e.g., IPFS hash, description)
     * @param initialWeightsHash Hash of the initial model weights
     * @dev Model ID should be a keccak256 hash of unique model properties
     */
    function registerModel(
        bytes32 modelId,
        string calldata metadata,
        bytes32 initialWeightsHash
    ) external nonReentrant whenNotPaused {
        if (modelId == bytes32(0)) {
            revert InvalidModelId();
        }
        if (initialWeightsHash == bytes32(0)) {
            revert InvalidWeightsHash();
        }
        if (bytes(metadata).length == 0) {
            revert InvalidMetadata();
        }
        if (bytes(metadata).length > MAX_METADATA_LENGTH) {
            revert MetadataTooLong();
        }
        if (models[modelId].owner != address(0)) {
            revert ModelAlreadyExists();
        }

        models[modelId] = Model({
            modelId: modelId,
            owner: msg.sender,
            metadata: metadata,
            currentWeightsHash: initialWeightsHash,
            currentRound: 0,
            createdAt: block.timestamp,
            updatedAt: block.timestamp,
            active: true
        });

        // Record initial version
        modelVersions[modelId][0] = ModelVersion({
            weightsHash: initialWeightsHash,
            round: 0,
            timestamp: block.timestamp,
            updatedBy: msg.sender
        });
        modelVersionCount[modelId] = 1;

        totalModels++;

        emit ModelRegistered(
            modelId,
            msg.sender,
            metadata,
            initialWeightsHash,
            block.timestamp
        );
    }

    /**
     * @notice Updates a model with new weights hash
     * @param modelId The model to update
     * @param newWeightsHash Hash of the new model weights
     * @param round The round number this update corresponds to
     * @dev Round must be greater than current round
     */
    function updateModel(
        bytes32 modelId,
        bytes32 newWeightsHash,
        uint256 round
    ) external nonReentrant whenNotPaused modelExists(modelId) modelActive(modelId) onlyModelOwnerOrOperator(modelId) {
        if (newWeightsHash == bytes32(0)) {
            revert InvalidWeightsHash();
        }

        Model storage model = models[modelId];

        if (round <= model.currentRound) {
            revert InvalidRound();
        }

        // Update model state
        model.currentWeightsHash = newWeightsHash;
        model.currentRound = round;
        model.updatedAt = block.timestamp;

        // Record version history
        uint256 versionIndex = modelVersionCount[modelId];
        modelVersions[modelId][versionIndex] = ModelVersion({
            weightsHash: newWeightsHash,
            round: round,
            timestamp: block.timestamp,
            updatedBy: msg.sender
        });
        modelVersionCount[modelId]++;

        emit ModelUpdated(
            modelId,
            newWeightsHash,
            round,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Gets full model details
     * @param modelId The model to query
     * @return model The complete model record
     */
    function getModel(bytes32 modelId) external view returns (Model memory) {
        if (models[modelId].owner == address(0)) {
            revert ModelNotFound();
        }
        return models[modelId];
    }

    /**
     * @notice Gets the version history of a model
     * @param modelId The model to query
     * @return versions Array of all model versions
     */
    function getModelHistory(bytes32 modelId) external view modelExists(modelId) returns (ModelVersion[] memory versions) {
        uint256 count = modelVersionCount[modelId];
        versions = new ModelVersion[](count);

        for (uint256 i = 0; i < count; i++) {
            versions[i] = modelVersions[modelId][i];
        }

        return versions;
    }

    /**
     * @notice Updates model metadata
     * @param modelId The model to update
     * @param newMetadata The new metadata string
     */
    function updateMetadata(
        bytes32 modelId,
        string calldata newMetadata
    ) external nonReentrant whenNotPaused modelExists(modelId) onlyModelOwner(modelId) {
        if (bytes(newMetadata).length == 0) {
            revert InvalidMetadata();
        }
        if (bytes(newMetadata).length > MAX_METADATA_LENGTH) {
            revert MetadataTooLong();
        }

        models[modelId].metadata = newMetadata;
        models[modelId].updatedAt = block.timestamp;

        emit ModelMetadataUpdated(modelId, newMetadata, block.timestamp);
    }

    /**
     * @notice Transfers model ownership
     * @param modelId The model to transfer
     * @param newOwner The new owner address
     */
    function transferModelOwnership(
        bytes32 modelId,
        address newOwner
    ) external nonReentrant whenNotPaused modelExists(modelId) onlyModelOwner(modelId) {
        if (newOwner == address(0)) {
            revert ZeroAddress();
        }
        if (newOwner == msg.sender) {
            revert SameAddress();
        }

        address previousOwner = models[modelId].owner;
        models[modelId].owner = newOwner;
        models[modelId].updatedAt = block.timestamp;

        emit ModelOwnershipTransferred(modelId, previousOwner, newOwner, block.timestamp);
    }

    /**
     * @notice Deactivates a model
     * @param modelId The model to deactivate
     */
    function deactivateModel(
        bytes32 modelId
    ) external nonReentrant whenNotPaused modelExists(modelId) onlyModelOwner(modelId) {
        models[modelId].active = false;
        models[modelId].updatedAt = block.timestamp;

        emit ModelDeactivated(modelId, msg.sender, block.timestamp);
    }

    /**
     * @notice Reactivates a deactivated model
     * @param modelId The model to reactivate
     */
    function reactivateModel(
        bytes32 modelId
    ) external nonReentrant whenNotPaused modelExists(modelId) onlyModelOwner(modelId) {
        models[modelId].active = true;
        models[modelId].updatedAt = block.timestamp;

        emit ModelReactivated(modelId, msg.sender, block.timestamp);
    }

    // ============ Validator Management Functions ============

    /**
     * @notice Adds a validator to a model
     * @param modelId The model to add validator to
     * @param validator The validator address to add
     */
    function addValidator(
        bytes32 modelId,
        address validator
    ) external nonReentrant whenNotPaused modelExists(modelId) modelActive(modelId) onlyModelOwnerOrOperator(modelId) {
        if (validator == address(0)) {
            revert ZeroAddress();
        }
        if (modelValidators[modelId][validator]) {
            revert ValidatorAlreadyExists();
        }
        if (_validatorLists[modelId].length >= MAX_VALIDATORS) {
            revert TooManyValidators();
        }

        modelValidators[modelId][validator] = true;
        _validatorLists[modelId].push(validator);

        emit ValidatorAdded(modelId, validator, msg.sender, block.timestamp);
    }

    /**
     * @notice Removes a validator from a model
     * @param modelId The model to remove validator from
     * @param validator The validator address to remove
     */
    function removeValidator(
        bytes32 modelId,
        address validator
    ) external nonReentrant whenNotPaused modelExists(modelId) onlyModelOwnerOrOperator(modelId) {
        if (!modelValidators[modelId][validator]) {
            revert ValidatorNotFound();
        }

        modelValidators[modelId][validator] = false;

        // Remove from validator list (swap and pop)
        address[] storage validators = _validatorLists[modelId];
        uint256 length = validators.length;
        for (uint256 i = 0; i < length; i++) {
            if (validators[i] == validator) {
                validators[i] = validators[length - 1];
                validators.pop();
                break;
            }
        }

        emit ValidatorRemoved(modelId, validator, msg.sender, block.timestamp);
    }

    /**
     * @notice Checks if an address is a validator for a model
     * @param modelId The model to check
     * @param validator The address to check
     * @return isValidatorStatus True if the address is a validator
     */
    function isValidator(
        bytes32 modelId,
        address validator
    ) external view returns (bool isValidatorStatus) {
        return modelValidators[modelId][validator];
    }

    /**
     * @notice Gets all validators for a model
     * @param modelId The model to query
     * @return validators Array of validator addresses
     */
    function getValidators(bytes32 modelId) external view modelExists(modelId) returns (address[] memory validators) {
        return _validatorLists[modelId];
    }

    /**
     * @notice Gets the validator count for a model
     * @param modelId The model to query
     * @return count Number of validators
     */
    function getValidatorCount(bytes32 modelId) external view returns (uint256 count) {
        return _validatorLists[modelId].length;
    }

    // ============ Round Tracking Functions ============

    /**
     * @notice Records a completed FL round
     * @param modelId The model the round belongs to
     * @param round The round number
     * @param aggregatedHash Hash of the aggregated model weights
     * @param participantCount Number of participants in this round
     * @dev Only callable by addresses with RECORDER_ROLE
     */
    function recordRound(
        bytes32 modelId,
        uint256 round,
        bytes32 aggregatedHash,
        uint256 participantCount
    ) external nonReentrant whenNotPaused modelExists(modelId) modelActive(modelId) onlyRole(RECORDER_ROLE) {
        if (aggregatedHash == bytes32(0)) {
            revert InvalidWeightsHash();
        }
        if (participantCount == 0) {
            revert InvalidParticipantCount();
        }
        if (rounds[modelId][round].finalized) {
            revert RoundAlreadyRecorded();
        }

        // Ensure rounds are recorded sequentially (allow for round 1 to start or next expected round)
        Model storage model = models[modelId];
        uint256 expectedRound = model.currentRound + 1;
        if (round != expectedRound && !(model.currentRound == 0 && round == 1)) {
            revert RoundNotSequential();
        }

        rounds[modelId][round] = Round({
            roundNumber: round,
            aggregatedHash: aggregatedHash,
            participantCount: participantCount,
            timestamp: block.timestamp,
            recordedBy: msg.sender,
            finalized: true
        });

        emit RoundRecorded(
            modelId,
            round,
            aggregatedHash,
            participantCount,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Gets round data for a specific round
     * @param modelId The model to query
     * @param round The round number to query
     * @return roundData The round record
     */
    function getRound(
        bytes32 modelId,
        uint256 round
    ) external view modelExists(modelId) returns (Round memory roundData) {
        return rounds[modelId][round];
    }

    /**
     * @notice Gets the latest round for a model
     * @param modelId The model to query
     * @return roundData The latest round record
     * @return roundNumber The latest round number
     */
    function getLatestRound(bytes32 modelId) external view modelExists(modelId) returns (Round memory roundData, uint256 roundNumber) {
        uint256 currentRound = models[modelId].currentRound;

        // If current round is 0 and there's a round 1 recorded, return round 1
        if (currentRound == 0 && rounds[modelId][1].finalized) {
            return (rounds[modelId][1], 1);
        }

        // Otherwise return the current round or the next recorded round
        if (rounds[modelId][currentRound].finalized) {
            return (rounds[modelId][currentRound], currentRound);
        }

        // Check if there's a recorded round at currentRound + 1
        uint256 nextRound = currentRound + 1;
        if (rounds[modelId][nextRound].finalized) {
            return (rounds[modelId][nextRound], nextRound);
        }

        // Return empty round with current round number
        return (rounds[modelId][currentRound], currentRound);
    }

    // ============ Admin Functions ============

    /**
     * @notice Pauses the contract
     */
    function pause() external onlyRole(GOVERNANCE_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses the contract
     */
    function unpause() external onlyRole(GOVERNANCE_ROLE) {
        _unpause();
    }

    /**
     * @notice Emergency deactivation of a model (governance only)
     * @param modelId The model to deactivate
     */
    function emergencyDeactivate(bytes32 modelId) external onlyRole(GOVERNANCE_ROLE) modelExists(modelId) {
        models[modelId].active = false;
        models[modelId].updatedAt = block.timestamp;

        emit ModelDeactivated(modelId, msg.sender, block.timestamp);
    }

    // ============ View Functions ============

    /**
     * @notice Gets a specific version of a model
     * @param modelId The model to query
     * @param versionIndex The version index
     * @return version The model version record
     */
    function getModelVersion(
        bytes32 modelId,
        uint256 versionIndex
    ) external view modelExists(modelId) returns (ModelVersion memory version) {
        if (versionIndex >= modelVersionCount[modelId]) {
            revert InvalidRound();
        }
        return modelVersions[modelId][versionIndex];
    }

    /**
     * @notice Checks if a model exists
     * @param modelId The model to check
     * @return exists True if the model exists
     */
    function modelExistsCheck(bytes32 modelId) external view returns (bool exists) {
        return models[modelId].owner != address(0);
    }

    /**
     * @notice Checks if a model is active
     * @param modelId The model to check
     * @return active True if the model is active
     */
    function isModelActive(bytes32 modelId) external view returns (bool active) {
        return models[modelId].active;
    }

    /**
     * @notice Gets the current state of a model
     * @param modelId The model to query
     * @return currentWeightsHash The current weights hash
     * @return currentRound The current round number
     * @return active Whether the model is active
     * @return validatorCount Number of validators
     */
    function getModelState(bytes32 modelId)
        external
        view
        modelExists(modelId)
        returns (
            bytes32 currentWeightsHash,
            uint256 currentRound,
            bool active,
            uint256 validatorCount
        )
    {
        Model storage model = models[modelId];
        return (
            model.currentWeightsHash,
            model.currentRound,
            model.active,
            _validatorLists[modelId].length
        );
    }
}
