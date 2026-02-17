// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {MerkleProof} from "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title ReputationAnchor
 * @author Mycelix Network
 * @notice Anchors PoGQ (Proof of Genuine Quality) reputation scores on-chain
 * @dev Uses Merkle trees to efficiently store and verify reputation data from Holochain
 *
 * The reputation system follows PoGQ principles:
 * - Scores represent genuine quality contributions tracked off-chain
 * - On-chain anchoring provides settlement layer for disputes and payments
 * - Merkle proofs allow gas-efficient verification of any agent's score
 */
contract ReputationAnchor is Ownable, AccessControl, ReentrancyGuard, Pausable {
    // ============ Constants ============

    /// @notice Role for authorized reputation anchors (Holochain bridge nodes)
    bytes32 public constant ANCHOR_ROLE = keccak256("ANCHOR_ROLE");

    /// @notice Role for governance actions
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    /// @notice Maximum reputation score (scaled by 1e18)
    uint256 public constant MAX_SCORE = 1000 * 1e18;

    /// @notice Minimum time between root updates (prevents spam)
    uint256 public constant MIN_UPDATE_INTERVAL = 1 hours;

    // ============ State Variables ============

    /// @notice Current Merkle root of all reputation scores
    bytes32 public currentRoot;

    /// @notice Previous Merkle root (for transition period)
    bytes32 public previousRoot;

    /// @notice Timestamp of last root update
    uint256 public lastUpdateTime;

    /// @notice Epoch counter for tracking updates
    uint256 public epoch;

    /// @notice Grace period where both roots are valid
    /// @dev H-10 SECURITY WARNING: During the grace period (default 24 hours), proofs
    /// against the PREVIOUS root are still accepted. This allows for smooth transitions
    /// but creates a window where agents could use old (potentially higher) scores.
    ///
    /// MITIGATIONS:
    /// 1. Nonces in leaf hashes prevent replay across epochs IF properly incremented
    /// 2. Downstream contracts should check epochTimestamps and apply additional validation
    /// 3. Consider reducing grace period for high-stakes operations (setGracePeriod)
    /// 4. For critical decisions, verify against currentRoot only:
    ///    `MerkleProof.verify(proof, currentRoot, leaf)` without grace period fallback
    uint256 public gracePeriod = 24 hours;

    /// @notice Mapping of epoch to root for historical lookups
    mapping(uint256 => bytes32) public epochRoots;

    /// @notice Mapping of epoch to timestamp
    mapping(uint256 => uint256) public epochTimestamps;

    /// @notice Nonce tracking for agents (managed off-chain in Holochain, used in Merkle leaves)
    /// @dev Nonces are included in Merkle tree leaves to prevent cross-epoch replay
    ///      Format: leaf = keccak256(abi.encodePacked(agent, score, nonce))
    ///      Each new epoch should increment nonces for updated agents
    mapping(address => uint256) public agentNonces;

    /// @notice Verification count per agent per epoch (rate limiting)
    mapping(address => mapping(uint256 => uint256)) public verificationCount;

    /// @notice Maximum verifications per agent per epoch
    uint256 public constant MAX_VERIFICATIONS_PER_EPOCH = 100;

    /// @notice Cached verified scores (optional, for gas optimization)
    mapping(address => uint256) public cachedScores;

    /// @notice Timestamp of last cached score update
    mapping(address => uint256) public cacheTimestamp;

    // ============ Events ============

    /// @notice Emitted when a new reputation root is anchored
    event ReputationRootAnchored(
        bytes32 indexed newRoot,
        bytes32 indexed previousRoot,
        uint256 indexed epoch,
        address anchor,
        uint256 timestamp
    );

    /// @notice Emitted when a reputation score is verified
    event ReputationVerified(
        address indexed agent,
        uint256 score,
        uint256 indexed epoch,
        bool cached
    );

    /// @notice Emitted when grace period is updated
    event GracePeriodUpdated(uint256 oldPeriod, uint256 newPeriod);

    /// @notice Emitted when a score is cached
    event ScoreCached(address indexed agent, uint256 score, uint256 timestamp);

    /// @notice Emitted when a cache is invalidated
    event CacheInvalidated(address indexed agent, address indexed invalidatedBy);

    // ============ Errors ============

    error InvalidRoot();
    error TooSoonToUpdate();
    error InvalidProof();
    error ScoreTooHigh();
    error InvalidGracePeriod();
    error ZeroAddress();
    error StaleCache();
    error RateLimitExceeded();

    // ============ Constructor ============

    /**
     * @notice Initializes the ReputationAnchor contract
     * @param initialOwner Address of the contract owner
     * @param initialAnchor Address of the initial anchor node
     */
    constructor(address initialOwner, address initialAnchor) Ownable(initialOwner) {
        if (initialOwner == address(0) || initialAnchor == address(0)) {
            revert ZeroAddress();
        }

        _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
        _grantRole(GOVERNANCE_ROLE, initialOwner);
        _grantRole(ANCHOR_ROLE, initialAnchor);

        lastUpdateTime = block.timestamp;
    }

    // ============ External Functions ============

    /**
     * @notice Stores a new Merkle root of reputation scores
     * @param merkleRoot The new Merkle root containing all agent scores
     * @dev Only callable by addresses with ANCHOR_ROLE
     *
     * The Merkle tree structure is:
     * - Leaf: keccak256(abi.encodePacked(agent, score, nonce))
     * - Tree is sorted by agent address for deterministic construction
     */
    function storeReputationRoot(bytes32 merkleRoot)
        external
        onlyRole(ANCHOR_ROLE)
        whenNotPaused
    {
        if (merkleRoot == bytes32(0)) {
            revert InvalidRoot();
        }

        if (block.timestamp < lastUpdateTime + MIN_UPDATE_INTERVAL) {
            revert TooSoonToUpdate();
        }

        // Rotate roots
        previousRoot = currentRoot;
        currentRoot = merkleRoot;

        // Update epoch and timestamps
        epoch++;
        epochRoots[epoch] = merkleRoot;
        epochTimestamps[epoch] = block.timestamp;
        lastUpdateTime = block.timestamp;

        emit ReputationRootAnchored(
            merkleRoot,
            previousRoot,
            epoch,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Verifies an agent's reputation score using a Merkle proof
     * @param agent The address of the agent
     * @param score The claimed reputation score (scaled by 1e18)
     * @param nonce The agent's nonce at the time of root creation
     * @param proof The Merkle proof
     * @return valid True if the proof is valid
     *
     * @dev Checks against both current and previous root during grace period
     */
    function verifyReputation(
        address agent,
        uint256 score,
        uint256 nonce,
        bytes32[] calldata proof
    ) external view returns (bool valid) {
        if (score > MAX_SCORE) {
            revert ScoreTooHigh();
        }

        bytes32 leaf = keccak256(abi.encodePacked(agent, score, nonce));

        // Try current root first
        if (MerkleProof.verify(proof, currentRoot, leaf)) {
            return true;
        }

        // During grace period, also accept previous root
        if (
            previousRoot != bytes32(0) &&
            block.timestamp <= epochTimestamps[epoch] + gracePeriod
        ) {
            return MerkleProof.verify(proof, previousRoot, leaf);
        }

        return false;
    }

    /**
     * @notice Verifies reputation and emits an event (state-changing version)
     * @param agent The address of the agent
     * @param score The claimed reputation score
     * @param nonce The agent's nonce
     * @param proof The Merkle proof
     * @param useCache Whether to use and update cache
     * @return valid True if verified
     */
    function verifyAndRecord(
        address agent,
        uint256 score,
        uint256 nonce,
        bytes32[] calldata proof,
        bool useCache
    ) external nonReentrant whenNotPaused returns (bool valid) {
        if (score > MAX_SCORE) {
            revert ScoreTooHigh();
        }

        // Rate limiting: prevent spam verification attacks
        if (verificationCount[agent][epoch] >= MAX_VERIFICATIONS_PER_EPOCH) {
            revert RateLimitExceeded();
        }
        verificationCount[agent][epoch]++;

        bytes32 leaf = keccak256(abi.encodePacked(agent, score, nonce));
        bool verified = false;
        bool usedCache = false;

        // Check cache first if requested
        if (useCache && cacheTimestamp[agent] >= epochTimestamps[epoch]) {
            if (cachedScores[agent] == score) {
                verified = true;
                usedCache = true;
            }
        }

        // If not cached, verify against Merkle root
        if (!verified) {
            if (MerkleProof.verify(proof, currentRoot, leaf)) {
                verified = true;
            } else if (
                previousRoot != bytes32(0) &&
                block.timestamp <= epochTimestamps[epoch] + gracePeriod
            ) {
                verified = MerkleProof.verify(proof, previousRoot, leaf);
            }

            // Update cache if verified
            if (verified && useCache) {
                cachedScores[agent] = score;
                cacheTimestamp[agent] = block.timestamp;
                emit ScoreCached(agent, score, block.timestamp);
            }
        }

        if (verified) {
            emit ReputationVerified(agent, score, epoch, usedCache);
        }

        return verified;
    }

    /**
     * @notice Gets the reputation score at a specific epoch
     * @param epochNumber The epoch to query
     * @return root The Merkle root at that epoch
     * @return timestamp The timestamp of that epoch
     */
    function getEpochData(uint256 epochNumber)
        external
        view
        returns (bytes32 root, uint256 timestamp)
    {
        return (epochRoots[epochNumber], epochTimestamps[epochNumber]);
    }

    /**
     * @notice Verifies against a specific historical epoch
     * @param agent The agent address
     * @param score The score to verify
     * @param nonce The nonce
     * @param proof The Merkle proof
     * @param epochNumber The epoch to verify against
     * @return valid True if valid for that epoch
     */
    function verifyAtEpoch(
        address agent,
        uint256 score,
        uint256 nonce,
        bytes32[] calldata proof,
        uint256 epochNumber
    ) external view returns (bool valid) {
        bytes32 root = epochRoots[epochNumber];
        if (root == bytes32(0)) {
            return false;
        }

        bytes32 leaf = keccak256(abi.encodePacked(agent, score, nonce));
        return MerkleProof.verify(proof, root, leaf);
    }

    // ============ Admin Functions ============

    /**
     * @notice Updates the grace period for root transitions
     * @param newGracePeriod The new grace period in seconds
     */
    function setGracePeriod(uint256 newGracePeriod)
        external
        onlyRole(GOVERNANCE_ROLE)
    {
        if (newGracePeriod < 1 hours || newGracePeriod > 7 days) {
            revert InvalidGracePeriod();
        }

        uint256 oldPeriod = gracePeriod;
        gracePeriod = newGracePeriod;

        emit GracePeriodUpdated(oldPeriod, newGracePeriod);
    }

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
     * @notice Invalidates a cached score (emergency use)
     * @param agent The agent whose cache to invalidate
     */
    function invalidateCache(address agent) external onlyRole(GOVERNANCE_ROLE) {
        delete cachedScores[agent];
        delete cacheTimestamp[agent];
        emit CacheInvalidated(agent, msg.sender);
    }

    // ============ View Functions ============

    /**
     * @notice Returns the current epoch and root
     */
    function getCurrentState()
        external
        view
        returns (
            bytes32 root,
            uint256 currentEpoch,
            uint256 lastUpdate,
            bool inGracePeriod
        )
    {
        return (
            currentRoot,
            epoch,
            lastUpdateTime,
            block.timestamp <= epochTimestamps[epoch] + gracePeriod
        );
    }

    /**
     * @notice Checks if an address has the anchor role
     */
    function isAnchor(address account) external view returns (bool) {
        return hasRole(ANCHOR_ROLE, account);
    }

    /**
     * @notice Gets cached score if valid
     * @param agent The agent to query
     * @return score The cached score (0 if not cached or stale)
     * @return isValid Whether the cache is valid
     */
    function getCachedScore(address agent)
        external
        view
        returns (uint256 score, bool isValid)
    {
        if (cacheTimestamp[agent] >= epochTimestamps[epoch]) {
            return (cachedScores[agent], true);
        }
        return (0, false);
    }
}
