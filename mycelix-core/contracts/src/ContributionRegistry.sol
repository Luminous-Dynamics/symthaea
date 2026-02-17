// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title IModelRegistry
 * @notice Interface for ModelRegistry contract
 */
interface IModelRegistry {
    function modelExistsCheck(bytes32 modelId) external view returns (bool exists);
    function isModelActive(bytes32 modelId) external view returns (bool active);
}

/**
 * @title ContributionRegistry
 * @author Mycelix Network
 * @notice Registry for tracking Federated Learning (FL) participant contributions on-chain
 * @dev Provides contribution recording, validation, and aggregated statistics for FL rounds
 *
 * Features:
 * - Contribution recording with gradient hashes and timestamps
 * - Validation status tracking per contribution
 * - Aggregated statistics per participant and per round
 * - Integration with ModelRegistry for model validation
 */
contract ContributionRegistry is AccessControl, ReentrancyGuard, Pausable {
    // ============ Constants ============

    /// @notice Role for recording contributions
    bytes32 public constant RECORDER_ROLE = keccak256("RECORDER_ROLE");

    /// @notice Role for validating contributions
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");

    /// @notice Role for governance actions
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    // ============ Structs ============

    /// @notice Contribution record containing participant contribution details
    struct Contribution {
        bytes32 modelId;
        uint256 round;
        address participant;
        bytes32 gradientHash;
        uint256 timestamp;
        uint256 recordedAt;
        bool exists;
    }

    /// @notice Validation status for a contribution
    struct ValidationStatus {
        bool validated;
        bool isValid;
        address validatedBy;
        uint256 validatedAt;
    }

    /// @notice Aggregated participant statistics for a model
    struct ParticipantStats {
        uint256 totalRounds;
        uint256 totalContributions;
        uint256 validContributions;
        uint256 invalidContributions;
        uint256 firstContributionAt;
        uint256 lastContributionAt;
    }

    /// @notice Aggregated round statistics for a model
    struct RoundStats {
        uint256 participantCount;
        uint256 contributionCount;
        uint256 validatedCount;
        uint256 validCount;
        uint256 invalidCount;
    }

    // ============ State Variables ============

    /// @notice Reference to the ModelRegistry contract
    IModelRegistry public immutable modelRegistry;

    /// @notice Mapping of model ID => round => participant => Contribution
    mapping(bytes32 => mapping(uint256 => mapping(address => Contribution))) public contributions;

    /// @notice Mapping of model ID => round => participant => ValidationStatus
    mapping(bytes32 => mapping(uint256 => mapping(address => ValidationStatus))) public validationStatuses;

    /// @notice Mapping of model ID => round => list of participant addresses
    mapping(bytes32 => mapping(uint256 => address[])) internal _roundParticipants;

    /// @notice Mapping of model ID => participant => list of rounds participated in
    mapping(bytes32 => mapping(address => uint256[])) internal _participantRounds;

    /// @notice Mapping of model ID => participant => ParticipantStats
    mapping(bytes32 => mapping(address => ParticipantStats)) internal _participantStats;

    /// @notice Mapping of model ID => round => RoundStats
    mapping(bytes32 => mapping(uint256 => RoundStats)) internal _roundStats;

    /// @notice Total contributions across all models
    uint256 public totalContributions;

    // ============ Events ============

    /// @notice Emitted when a contribution is recorded
    event ContributionRecorded(
        bytes32 indexed modelId,
        uint256 indexed round,
        address indexed participant,
        bytes32 gradientHash,
        uint256 timestamp,
        uint256 recordedAt
    );

    /// @notice Emitted when a contribution is validated
    event ContributionValidated(
        bytes32 indexed modelId,
        uint256 indexed round,
        address indexed participant,
        bool isValid,
        address validatedBy,
        uint256 validatedAt
    );

    /// @notice Emitted when the ModelRegistry reference is updated (governance only)
    event ModelRegistryUpdated(
        address indexed previousRegistry,
        address indexed newRegistry,
        uint256 timestamp
    );

    // ============ Errors ============

    error ZeroAddress();
    error ModelNotFound();
    error ModelNotActive();
    error ContributionAlreadyExists();
    error ContributionNotFound();
    error InvalidGradientHash();
    error InvalidTimestamp();
    error InvalidRound();
    error AlreadyValidated();
    error InvalidModelRegistry();

    // ============ Modifiers ============

    /// @notice Ensures the model exists in ModelRegistry
    modifier modelExists(bytes32 modelId) {
        if (!modelRegistry.modelExistsCheck(modelId)) {
            revert ModelNotFound();
        }
        _;
    }

    /// @notice Ensures the model is active in ModelRegistry
    modifier modelActive(bytes32 modelId) {
        if (!modelRegistry.isModelActive(modelId)) {
            revert ModelNotActive();
        }
        _;
    }

    /// @notice Ensures the contribution exists
    modifier contributionExists(bytes32 modelId, uint256 round, address participant) {
        if (!contributions[modelId][round][participant].exists) {
            revert ContributionNotFound();
        }
        _;
    }

    // ============ Constructor ============

    /**
     * @notice Initializes the ContributionRegistry contract
     * @param _modelRegistry Address of the ModelRegistry contract
     * @param initialAdmin Address of the initial admin
     */
    constructor(address _modelRegistry, address initialAdmin) {
        if (_modelRegistry == address(0)) {
            revert ZeroAddress();
        }
        if (initialAdmin == address(0)) {
            revert ZeroAddress();
        }

        // Verify the ModelRegistry is a valid contract
        // Note: This doesn't guarantee it's a ModelRegistry, but ensures it's a contract
        if (_modelRegistry.code.length == 0) {
            revert InvalidModelRegistry();
        }

        modelRegistry = IModelRegistry(_modelRegistry);

        _grantRole(DEFAULT_ADMIN_ROLE, initialAdmin);
        _grantRole(GOVERNANCE_ROLE, initialAdmin);
        _grantRole(RECORDER_ROLE, initialAdmin);
        _grantRole(VALIDATOR_ROLE, initialAdmin);
    }

    // ============ Contribution Recording Functions ============

    /**
     * @notice Records a participant contribution for a FL round
     * @param modelId The model ID the contribution is for
     * @param round The round number
     * @param participant The participant address
     * @param gradientHash Hash of the gradient/model update
     * @param timestamp The timestamp when the contribution was made
     * @dev Only callable by addresses with RECORDER_ROLE
     */
    function recordContribution(
        bytes32 modelId,
        uint256 round,
        address participant,
        bytes32 gradientHash,
        uint256 timestamp
    ) external nonReentrant whenNotPaused modelExists(modelId) modelActive(modelId) onlyRole(RECORDER_ROLE) {
        if (participant == address(0)) {
            revert ZeroAddress();
        }
        if (gradientHash == bytes32(0)) {
            revert InvalidGradientHash();
        }
        if (timestamp == 0 || timestamp > block.timestamp) {
            revert InvalidTimestamp();
        }
        if (round == 0) {
            revert InvalidRound();
        }
        if (contributions[modelId][round][participant].exists) {
            revert ContributionAlreadyExists();
        }

        // Record the contribution
        contributions[modelId][round][participant] = Contribution({
            modelId: modelId,
            round: round,
            participant: participant,
            gradientHash: gradientHash,
            timestamp: timestamp,
            recordedAt: block.timestamp,
            exists: true
        });

        // Update round participants list
        _roundParticipants[modelId][round].push(participant);

        // Update participant rounds list
        _participantRounds[modelId][participant].push(round);

        // Update participant stats
        ParticipantStats storage pStats = _participantStats[modelId][participant];
        if (pStats.firstContributionAt == 0) {
            pStats.firstContributionAt = block.timestamp;
        }
        pStats.totalRounds++;
        pStats.totalContributions++;
        pStats.lastContributionAt = block.timestamp;

        // Update round stats
        RoundStats storage rStats = _roundStats[modelId][round];
        rStats.participantCount++;
        rStats.contributionCount++;

        totalContributions++;

        emit ContributionRecorded(
            modelId,
            round,
            participant,
            gradientHash,
            timestamp,
            block.timestamp
        );
    }

    /**
     * @notice Gets a specific contribution
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @return contribution The contribution record
     */
    function getContribution(
        bytes32 modelId,
        uint256 round,
        address participant
    ) external view returns (Contribution memory contribution) {
        return contributions[modelId][round][participant];
    }

    /**
     * @notice Gets all contributions for a specific round
     * @param modelId The model ID
     * @param round The round number
     * @return contributionList Array of contributions for the round
     *
     * @dev WARNING: This function can run out of gas with many participants.
     *      Use getContributionsByRoundPaginated for large datasets.
     */
    function getContributionsByRound(
        bytes32 modelId,
        uint256 round
    ) external view returns (Contribution[] memory contributionList) {
        address[] memory participants = _roundParticipants[modelId][round];
        uint256 count = participants.length;

        contributionList = new Contribution[](count);
        for (uint256 i = 0; i < count; i++) {
            contributionList[i] = contributions[modelId][round][participants[i]];
        }

        return contributionList;
    }

    /**
     * @notice M-02 remediation: Gets contributions for a round with pagination
     * @param modelId The model ID
     * @param round The round number
     * @param offset Starting index (0-based)
     * @param limit Maximum number of results to return
     * @return contributionList Array of contributions
     * @return total Total number of contributions in this round
     * @return hasMore Whether there are more contributions after this page
     */
    function getContributionsByRoundPaginated(
        bytes32 modelId,
        uint256 round,
        uint256 offset,
        uint256 limit
    ) external view returns (
        Contribution[] memory contributionList,
        uint256 total,
        bool hasMore
    ) {
        address[] storage participants = _roundParticipants[modelId][round];
        total = participants.length;

        if (offset >= total || limit == 0) {
            return (new Contribution[](0), total, false);
        }

        uint256 remaining = total - offset;
        uint256 resultCount = remaining < limit ? remaining : limit;
        hasMore = offset + resultCount < total;

        contributionList = new Contribution[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            contributionList[i] = contributions[modelId][round][participants[offset + i]];
        }

        return (contributionList, total, hasMore);
    }

    /**
     * @notice Gets all contributions by a specific participant for a model
     * @param modelId The model ID
     * @param participant The participant address
     * @return contributionList Array of contributions by the participant
     *
     * @dev WARNING: This function can run out of gas with many contributions.
     *      Use getContributionsByParticipantPaginated for large datasets.
     */
    function getContributionsByParticipant(
        bytes32 modelId,
        address participant
    ) external view returns (Contribution[] memory contributionList) {
        uint256[] memory rounds = _participantRounds[modelId][participant];
        uint256 count = rounds.length;

        contributionList = new Contribution[](count);
        for (uint256 i = 0; i < count; i++) {
            contributionList[i] = contributions[modelId][rounds[i]][participant];
        }

        return contributionList;
    }

    /**
     * @notice M-02 remediation: Gets contributions by participant with pagination
     * @param modelId The model ID
     * @param participant The participant address
     * @param offset Starting index (0-based)
     * @param limit Maximum number of results to return
     * @return contributionList Array of contributions
     * @return total Total number of contributions by this participant
     * @return hasMore Whether there are more contributions after this page
     */
    function getContributionsByParticipantPaginated(
        bytes32 modelId,
        address participant,
        uint256 offset,
        uint256 limit
    ) external view returns (
        Contribution[] memory contributionList,
        uint256 total,
        bool hasMore
    ) {
        uint256[] storage rounds = _participantRounds[modelId][participant];
        total = rounds.length;

        if (offset >= total || limit == 0) {
            return (new Contribution[](0), total, false);
        }

        uint256 remaining = total - offset;
        uint256 resultCount = remaining < limit ? remaining : limit;
        hasMore = offset + resultCount < total;

        contributionList = new Contribution[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            contributionList[i] = contributions[modelId][rounds[offset + i]][participant];
        }

        return (contributionList, total, hasMore);
    }

    /**
     * @notice Gets the list of participants for a specific round
     * @param modelId The model ID
     * @param round The round number
     * @return participants Array of participant addresses
     *
     * @dev WARNING: This function can run out of gas with many participants.
     *      Use getRoundParticipantsPaginated for large datasets.
     */
    function getRoundParticipants(
        bytes32 modelId,
        uint256 round
    ) external view returns (address[] memory participants) {
        return _roundParticipants[modelId][round];
    }

    /**
     * @notice M-02 remediation: Gets round participants with pagination
     * @param modelId The model ID
     * @param round The round number
     * @param offset Starting index (0-based)
     * @param limit Maximum number of results to return
     * @return participants Array of participant addresses
     * @return total Total number of participants in this round
     * @return hasMore Whether there are more participants after this page
     */
    function getRoundParticipantsPaginated(
        bytes32 modelId,
        uint256 round,
        uint256 offset,
        uint256 limit
    ) external view returns (
        address[] memory participants,
        uint256 total,
        bool hasMore
    ) {
        address[] storage allParticipants = _roundParticipants[modelId][round];
        total = allParticipants.length;

        if (offset >= total || limit == 0) {
            return (new address[](0), total, false);
        }

        uint256 remaining = total - offset;
        uint256 resultCount = remaining < limit ? remaining : limit;
        hasMore = offset + resultCount < total;

        participants = new address[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            participants[i] = allParticipants[offset + i];
        }

        return (participants, total, hasMore);
    }

    /**
     * @notice Gets the list of rounds a participant contributed to
     * @param modelId The model ID
     * @param participant The participant address
     * @return rounds Array of round numbers
     *
     * @dev WARNING: This function can run out of gas with many rounds.
     *      Use getParticipantRoundsPaginated for large datasets.
     */
    function getParticipantRounds(
        bytes32 modelId,
        address participant
    ) external view returns (uint256[] memory rounds) {
        return _participantRounds[modelId][participant];
    }

    /**
     * @notice M-02 remediation: Gets participant rounds with pagination
     * @param modelId The model ID
     * @param participant The participant address
     * @param offset Starting index (0-based)
     * @param limit Maximum number of results to return
     * @return rounds Array of round numbers
     * @return total Total number of rounds participated in
     * @return hasMore Whether there are more rounds after this page
     */
    function getParticipantRoundsPaginated(
        bytes32 modelId,
        address participant,
        uint256 offset,
        uint256 limit
    ) external view returns (
        uint256[] memory rounds,
        uint256 total,
        bool hasMore
    ) {
        uint256[] storage allRounds = _participantRounds[modelId][participant];
        total = allRounds.length;

        if (offset >= total || limit == 0) {
            return (new uint256[](0), total, false);
        }

        uint256 remaining = total - offset;
        uint256 resultCount = remaining < limit ? remaining : limit;
        hasMore = offset + resultCount < total;

        rounds = new uint256[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            rounds[i] = allRounds[offset + i];
        }

        return (rounds, total, hasMore);
    }

    // ============ Aggregated Statistics Functions ============

    /**
     * @notice Gets aggregated statistics for a participant
     * @param modelId The model ID
     * @param participant The participant address
     * @return totalRounds Total number of rounds participated in
     * @return totalContributions Total number of contributions made
     */
    function getParticipantStats(
        bytes32 modelId,
        address participant
    ) external view returns (uint256 totalRounds, uint256 totalContributions) {
        ParticipantStats memory stats = _participantStats[modelId][participant];
        return (stats.totalRounds, stats.totalContributions);
    }

    /**
     * @notice Gets detailed statistics for a participant
     * @param modelId The model ID
     * @param participant The participant address
     * @return stats The full participant statistics
     */
    function getParticipantStatsDetailed(
        bytes32 modelId,
        address participant
    ) external view returns (ParticipantStats memory stats) {
        return _participantStats[modelId][participant];
    }

    /**
     * @notice Gets aggregated statistics for a round
     * @param modelId The model ID
     * @param round The round number
     * @return participantCount Number of unique participants
     * @return contributionCount Total number of contributions
     */
    function getRoundStats(
        bytes32 modelId,
        uint256 round
    ) external view returns (uint256 participantCount, uint256 contributionCount) {
        RoundStats memory stats = _roundStats[modelId][round];
        return (stats.participantCount, stats.contributionCount);
    }

    /**
     * @notice Gets detailed statistics for a round
     * @param modelId The model ID
     * @param round The round number
     * @return stats The full round statistics
     */
    function getRoundStatsDetailed(
        bytes32 modelId,
        uint256 round
    ) external view returns (RoundStats memory stats) {
        return _roundStats[modelId][round];
    }

    // ============ Validation Functions ============

    /**
     * @notice Marks a contribution as valid or invalid
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @param isValid Whether the contribution is valid
     * @dev Only callable by addresses with VALIDATOR_ROLE
     */
    function markContributionValid(
        bytes32 modelId,
        uint256 round,
        address participant,
        bool isValid
    ) external nonReentrant whenNotPaused contributionExists(modelId, round, participant) onlyRole(VALIDATOR_ROLE) {
        ValidationStatus storage status = validationStatuses[modelId][round][participant];

        if (status.validated) {
            revert AlreadyValidated();
        }

        status.validated = true;
        status.isValid = isValid;
        status.validatedBy = msg.sender;
        status.validatedAt = block.timestamp;

        // Update participant stats
        ParticipantStats storage pStats = _participantStats[modelId][participant];
        if (isValid) {
            pStats.validContributions++;
        } else {
            pStats.invalidContributions++;
        }

        // Update round stats
        RoundStats storage rStats = _roundStats[modelId][round];
        rStats.validatedCount++;
        if (isValid) {
            rStats.validCount++;
        } else {
            rStats.invalidCount++;
        }

        emit ContributionValidated(
            modelId,
            round,
            participant,
            isValid,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Gets the validation status of a contribution
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @return status The validation status record
     */
    function getValidationStatus(
        bytes32 modelId,
        uint256 round,
        address participant
    ) external view returns (ValidationStatus memory status) {
        return validationStatuses[modelId][round][participant];
    }

    /**
     * @notice Checks if a contribution has been validated
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @return validated Whether the contribution has been validated
     * @return isValid Whether the contribution is valid (only meaningful if validated is true)
     */
    function isContributionValidated(
        bytes32 modelId,
        uint256 round,
        address participant
    ) external view returns (bool validated, bool isValid) {
        ValidationStatus memory status = validationStatuses[modelId][round][participant];
        return (status.validated, status.isValid);
    }

    // ============ Batch Operations ============

    /**
     * @notice Records multiple contributions in a single transaction
     * @param modelId The model ID
     * @param round The round number
     * @param participants Array of participant addresses
     * @param gradientHashes Array of gradient hashes
     * @param timestamps Array of timestamps
     * @dev Only callable by addresses with RECORDER_ROLE
     */
    function recordContributionsBatch(
        bytes32 modelId,
        uint256 round,
        address[] calldata participants,
        bytes32[] calldata gradientHashes,
        uint256[] calldata timestamps
    ) external nonReentrant whenNotPaused modelExists(modelId) modelActive(modelId) onlyRole(RECORDER_ROLE) {
        uint256 count = participants.length;
        require(count == gradientHashes.length && count == timestamps.length, "Array length mismatch");

        for (uint256 i = 0; i < count; i++) {
            _recordContributionInternal(
                modelId,
                round,
                participants[i],
                gradientHashes[i],
                timestamps[i]
            );
        }
    }

    /**
     * @notice Internal function to record a contribution
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @param gradientHash Hash of the gradient
     * @param timestamp The timestamp
     */
    function _recordContributionInternal(
        bytes32 modelId,
        uint256 round,
        address participant,
        bytes32 gradientHash,
        uint256 timestamp
    ) internal {
        if (participant == address(0)) {
            revert ZeroAddress();
        }
        if (gradientHash == bytes32(0)) {
            revert InvalidGradientHash();
        }
        if (timestamp == 0 || timestamp > block.timestamp) {
            revert InvalidTimestamp();
        }
        if (round == 0) {
            revert InvalidRound();
        }
        if (contributions[modelId][round][participant].exists) {
            revert ContributionAlreadyExists();
        }

        contributions[modelId][round][participant] = Contribution({
            modelId: modelId,
            round: round,
            participant: participant,
            gradientHash: gradientHash,
            timestamp: timestamp,
            recordedAt: block.timestamp,
            exists: true
        });

        _roundParticipants[modelId][round].push(participant);
        _participantRounds[modelId][participant].push(round);

        ParticipantStats storage pStats = _participantStats[modelId][participant];
        if (pStats.firstContributionAt == 0) {
            pStats.firstContributionAt = block.timestamp;
        }
        pStats.totalRounds++;
        pStats.totalContributions++;
        pStats.lastContributionAt = block.timestamp;

        RoundStats storage rStats = _roundStats[modelId][round];
        rStats.participantCount++;
        rStats.contributionCount++;

        totalContributions++;

        emit ContributionRecorded(
            modelId,
            round,
            participant,
            gradientHash,
            timestamp,
            block.timestamp
        );
    }

    /**
     * @notice Validates multiple contributions in a single transaction
     * @param modelId The model ID
     * @param round The round number
     * @param participants Array of participant addresses
     * @param validities Array of validity flags
     * @dev Only callable by addresses with VALIDATOR_ROLE
     */
    function markContributionsValidBatch(
        bytes32 modelId,
        uint256 round,
        address[] calldata participants,
        bool[] calldata validities
    ) external nonReentrant whenNotPaused onlyRole(VALIDATOR_ROLE) {
        uint256 count = participants.length;
        require(count == validities.length, "Array length mismatch");

        for (uint256 i = 0; i < count; i++) {
            _markContributionValidInternal(modelId, round, participants[i], validities[i]);
        }
    }

    /**
     * @notice Internal function to validate a contribution
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @param isValid Whether the contribution is valid
     */
    function _markContributionValidInternal(
        bytes32 modelId,
        uint256 round,
        address participant,
        bool isValid
    ) internal {
        if (!contributions[modelId][round][participant].exists) {
            revert ContributionNotFound();
        }

        ValidationStatus storage status = validationStatuses[modelId][round][participant];

        if (status.validated) {
            revert AlreadyValidated();
        }

        status.validated = true;
        status.isValid = isValid;
        status.validatedBy = msg.sender;
        status.validatedAt = block.timestamp;

        ParticipantStats storage pStats = _participantStats[modelId][participant];
        if (isValid) {
            pStats.validContributions++;
        } else {
            pStats.invalidContributions++;
        }

        RoundStats storage rStats = _roundStats[modelId][round];
        rStats.validatedCount++;
        if (isValid) {
            rStats.validCount++;
        } else {
            rStats.invalidCount++;
        }

        emit ContributionValidated(
            modelId,
            round,
            participant,
            isValid,
            msg.sender,
            block.timestamp
        );
    }

    // ============ Admin Functions ============

    /**
     * @notice Pauses the contract
     * @dev Only callable by addresses with GOVERNANCE_ROLE
     */
    function pause() external onlyRole(GOVERNANCE_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses the contract
     * @dev Only callable by addresses with GOVERNANCE_ROLE
     */
    function unpause() external onlyRole(GOVERNANCE_ROLE) {
        _unpause();
    }

    // ============ View Functions ============

    /**
     * @notice Checks if a contribution exists
     * @param modelId The model ID
     * @param round The round number
     * @param participant The participant address
     * @return exists True if the contribution exists
     */
    function contributionExistsCheck(
        bytes32 modelId,
        uint256 round,
        address participant
    ) external view returns (bool exists) {
        return contributions[modelId][round][participant].exists;
    }

    /**
     * @notice Gets the number of participants in a round
     * @param modelId The model ID
     * @param round The round number
     * @return count Number of participants
     */
    function getRoundParticipantCount(
        bytes32 modelId,
        uint256 round
    ) external view returns (uint256 count) {
        return _roundParticipants[modelId][round].length;
    }

    /**
     * @notice Gets the number of rounds a participant has contributed to
     * @param modelId The model ID
     * @param participant The participant address
     * @return count Number of rounds
     */
    function getParticipantRoundCount(
        bytes32 modelId,
        address participant
    ) external view returns (uint256 count) {
        return _participantRounds[modelId][participant].length;
    }
}
