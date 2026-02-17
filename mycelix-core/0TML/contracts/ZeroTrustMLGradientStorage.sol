// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Zero-TrustML Gradient Storage Contract
 * @notice Immutable storage for federated learning gradients, credits, and Byzantine events
 * @dev Designed for Zero-TrustML Phase 10 Coordinator with Ethereum/Polygon L2 deployment
 */
contract ZeroTrustMLGradientStorage {

    // ============================================
    // DATA STRUCTURES
    // ============================================

    struct Gradient {
        string gradientId;          // Unique gradient identifier
        bytes32 nodeIdHash;         // Hashed node ID (privacy)
        uint256 roundNum;           // FL round number
        string gradientHash;        // Hash of gradient data (stored off-chain)
        uint256 pogqScore;          // PoGQ score scaled by 1000 (0-1000)
        bool zkpocVerified;         // ZK-PoC verification status
        uint256 timestamp;          // Block timestamp
        address submitter;          // Address that submitted gradient
        bool exists;                // Existence flag
    }

    struct Credit {
        bytes32 holderHash;         // Hashed holder ID
        uint256 amount;             // Credit amount
        string earnedFrom;          // Source of credit
        uint256 timestamp;          // Block timestamp
        uint256 creditId;           // Unique credit ID
    }

    struct ByzantineEvent {
        bytes32 nodeIdHash;         // Hashed node ID
        uint256 roundNum;           // FL round number
        string detectionMethod;     // How it was detected
        string severity;            // "low", "medium", "high", "critical"
        string details;             // JSON details
        uint256 timestamp;          // Block timestamp
        uint256 eventId;            // Unique event ID
    }

    struct Reputation {
        uint256 totalGradientsSubmitted;
        uint256 totalCreditsEarned;
        uint256 byzantineEventCount;
        uint256 averagePogqScore;   // Scaled by 1000
        uint256 lastActivityTimestamp;
    }

    // ============================================
    // STATE VARIABLES
    // ============================================

    // Storage mappings
    mapping(string => Gradient) private gradients;                    // gradientId => Gradient
    mapping(uint256 => string[]) private gradientsByRound;           // roundNum => gradientIds[]
    mapping(bytes32 => Credit[]) private creditHistory;              // holderHash => Credit[]
    mapping(bytes32 => uint256) private creditBalances;              // holderHash => balance
    mapping(bytes32 => ByzantineEvent[]) private byzantineEvents;    // nodeIdHash => ByzantineEvent[]
    mapping(bytes32 => Reputation) private reputations;              // nodeIdHash => Reputation

    // Global counters
    uint256 public totalGradients;
    uint256 public totalCreditsIssued;
    uint256 public totalByzantineEvents;
    uint256 private nextCreditId = 1;
    uint256 private nextEventId = 1;

    // Contract metadata
    address public owner;
    string public version = "1.0.0";

    // ============================================
    // EVENTS
    // ============================================

    event GradientStored(
        string indexed gradientId,
        bytes32 indexed nodeIdHash,
        uint256 indexed roundNum,
        uint256 pogqScore,
        bool zkpocVerified
    );

    event CreditIssued(
        bytes32 indexed holderHash,
        uint256 amount,
        string earnedFrom,
        uint256 creditId
    );

    event ByzantineEventLogged(
        bytes32 indexed nodeIdHash,
        uint256 indexed roundNum,
        string severity,
        uint256 eventId
    );

    event ReputationUpdated(
        bytes32 indexed nodeIdHash,
        uint256 totalGradients,
        uint256 totalCredits,
        uint256 byzantineEvents
    );

    // ============================================
    // MODIFIERS
    // ============================================

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    // ============================================
    // CONSTRUCTOR
    // ============================================

    constructor() {
        owner = msg.sender;
    }

    // ============================================
    // GRADIENT FUNCTIONS
    // ============================================

    /**
     * @notice Store a gradient on-chain (hash only for gas efficiency)
     * @param gradientId Unique gradient identifier
     * @param nodeIdHash Keccak256 hash of node ID
     * @param roundNum FL round number
     * @param gradientHash Hash of gradient data
     * @param pogqScore PoGQ score scaled by 1000 (0-1000)
     * @param zkpocVerified ZK-PoC verification status
     */
    function storeGradient(
        string memory gradientId,
        bytes32 nodeIdHash,
        uint256 roundNum,
        string memory gradientHash,
        uint256 pogqScore,
        bool zkpocVerified
    ) external {
        require(!gradients[gradientId].exists, "Gradient already exists");
        require(pogqScore <= 1000, "PoGQ score must be <= 1000");
        require(bytes(gradientId).length > 0, "Gradient ID cannot be empty");
        require(bytes(gradientHash).length > 0, "Gradient hash cannot be empty");

        // Create gradient
        gradients[gradientId] = Gradient({
            gradientId: gradientId,
            nodeIdHash: nodeIdHash,
            roundNum: roundNum,
            gradientHash: gradientHash,
            pogqScore: pogqScore,
            zkpocVerified: zkpocVerified,
            timestamp: block.timestamp,
            submitter: msg.sender,
            exists: true
        });

        // Index by round
        gradientsByRound[roundNum].push(gradientId);

        // Update stats
        totalGradients++;

        // Update reputation
        Reputation storage rep = reputations[nodeIdHash];
        rep.totalGradientsSubmitted++;

        // Update average PoGQ score (weighted average)
        if (rep.averagePogqScore == 0) {
            rep.averagePogqScore = pogqScore;
        } else {
            rep.averagePogqScore = (rep.averagePogqScore * (rep.totalGradientsSubmitted - 1) + pogqScore) / rep.totalGradientsSubmitted;
        }

        rep.lastActivityTimestamp = block.timestamp;

        emit GradientStored(gradientId, nodeIdHash, roundNum, pogqScore, zkpocVerified);
        emit ReputationUpdated(nodeIdHash, rep.totalGradientsSubmitted, rep.totalCreditsEarned, rep.byzantineEventCount);
    }

    /**
     * @notice Get gradient by ID
     * @param gradientId Gradient identifier
     * @return Gradient data
     */
    function getGradient(string memory gradientId) external view returns (
        string memory,
        bytes32,
        uint256,
        string memory,
        uint256,
        bool,
        uint256,
        address
    ) {
        Gradient memory g = gradients[gradientId];
        require(g.exists, "Gradient does not exist");

        return (
            g.gradientId,
            g.nodeIdHash,
            g.roundNum,
            g.gradientHash,
            g.pogqScore,
            g.zkpocVerified,
            g.timestamp,
            g.submitter
        );
    }

    /**
     * @notice Get all gradient IDs for a specific round
     * @param roundNum FL round number
     * @return Array of gradient IDs
     */
    function getGradientsByRound(uint256 roundNum) external view returns (string[] memory) {
        return gradientsByRound[roundNum];
    }

    /**
     * @notice Check if gradient exists
     * @param gradientId Gradient identifier
     * @return True if gradient exists
     */
    function gradientExists(string memory gradientId) external view returns (bool) {
        return gradients[gradientId].exists;
    }

    // ============================================
    // CREDIT FUNCTIONS
    // ============================================

    /**
     * @notice Issue credit to a holder
     * @param holderHash Keccak256 hash of holder ID
     * @param amount Credit amount
     * @param earnedFrom Source of credit
     */
    function issueCredit(
        bytes32 holderHash,
        uint256 amount,
        string memory earnedFrom
    ) external {
        require(amount > 0, "Amount must be positive");
        require(bytes(earnedFrom).length > 0, "Earned from cannot be empty");

        uint256 creditId = nextCreditId++;

        // Create credit record
        Credit memory credit = Credit({
            holderHash: holderHash,
            amount: amount,
            earnedFrom: earnedFrom,
            timestamp: block.timestamp,
            creditId: creditId
        });

        // Add to history
        creditHistory[holderHash].push(credit);

        // Update balance
        creditBalances[holderHash] += amount;

        // Update stats
        totalCreditsIssued += amount;

        // Update reputation
        Reputation storage rep = reputations[holderHash];
        rep.totalCreditsEarned += amount;
        rep.lastActivityTimestamp = block.timestamp;

        emit CreditIssued(holderHash, amount, earnedFrom, creditId);
        emit ReputationUpdated(holderHash, rep.totalGradientsSubmitted, rep.totalCreditsEarned, rep.byzantineEventCount);
    }

    /**
     * @notice Get credit balance for a holder
     * @param holderHash Keccak256 hash of holder ID
     * @return Credit balance
     */
    function getCreditBalance(bytes32 holderHash) external view returns (uint256) {
        return creditBalances[holderHash];
    }

    /**
     * @notice Get credit history for a holder
     * @param holderHash Keccak256 hash of holder ID
     * @return Array of Credit structs
     */
    function getCreditHistory(bytes32 holderHash) external view returns (Credit[] memory) {
        return creditHistory[holderHash];
    }

    /**
     * @notice Get credit count for a holder
     * @param holderHash Keccak256 hash of holder ID
     * @return Number of credit records
     */
    function getCreditCount(bytes32 holderHash) external view returns (uint256) {
        return creditHistory[holderHash].length;
    }

    // ============================================
    // BYZANTINE EVENT FUNCTIONS
    // ============================================

    /**
     * @notice Log a Byzantine event (immutable audit trail)
     * @param nodeIdHash Keccak256 hash of node ID
     * @param roundNum FL round number
     * @param detectionMethod How it was detected
     * @param severity Event severity
     * @param details JSON details
     */
    function logByzantineEvent(
        bytes32 nodeIdHash,
        uint256 roundNum,
        string memory detectionMethod,
        string memory severity,
        string memory details
    ) external {
        require(bytes(detectionMethod).length > 0, "Detection method cannot be empty");
        require(bytes(severity).length > 0, "Severity cannot be empty");

        uint256 eventId = nextEventId++;

        // Create event
        ByzantineEvent memory evt = ByzantineEvent({
            nodeIdHash: nodeIdHash,
            roundNum: roundNum,
            detectionMethod: detectionMethod,
            severity: severity,
            details: details,
            timestamp: block.timestamp,
            eventId: eventId
        });

        // Add to node's event history
        byzantineEvents[nodeIdHash].push(evt);

        // Update stats
        totalByzantineEvents++;

        // Update reputation
        Reputation storage rep = reputations[nodeIdHash];
        rep.byzantineEventCount++;
        rep.lastActivityTimestamp = block.timestamp;

        emit ByzantineEventLogged(nodeIdHash, roundNum, severity, eventId);
        emit ReputationUpdated(nodeIdHash, rep.totalGradientsSubmitted, rep.totalCreditsEarned, rep.byzantineEventCount);
    }

    /**
     * @notice Get all Byzantine events for a node
     * @param nodeIdHash Keccak256 hash of node ID
     * @return Array of ByzantineEvent structs
     */
    function getByzantineEvents(bytes32 nodeIdHash) external view returns (ByzantineEvent[] memory) {
        return byzantineEvents[nodeIdHash];
    }

    /**
     * @notice Get Byzantine events for a node in a specific round
     * @param nodeIdHash Keccak256 hash of node ID
     * @param roundNum FL round number
     * @return Array of ByzantineEvent structs
     */
    function getByzantineEventsByRound(
        bytes32 nodeIdHash,
        uint256 roundNum
    ) external view returns (ByzantineEvent[] memory) {
        ByzantineEvent[] memory allEvents = byzantineEvents[nodeIdHash];

        // Count events in this round
        uint256 count = 0;
        for (uint256 i = 0; i < allEvents.length; i++) {
            if (allEvents[i].roundNum == roundNum) {
                count++;
            }
        }

        // Create result array
        ByzantineEvent[] memory result = new ByzantineEvent[](count);
        uint256 index = 0;

        // Fill result array
        for (uint256 i = 0; i < allEvents.length; i++) {
            if (allEvents[i].roundNum == roundNum) {
                result[index] = allEvents[i];
                index++;
            }
        }

        return result;
    }

    // ============================================
    // REPUTATION FUNCTIONS
    // ============================================

    /**
     * @notice Get reputation data for a node
     * @param nodeIdHash Keccak256 hash of node ID
     * @return totalGradientsSubmitted Total gradients submitted
     * @return totalCreditsEarned Total credits earned
     * @return byzantineEventCount Total Byzantine events
     * @return averagePogqScore Average PoGQ score (scaled by 1000)
     * @return lastActivityTimestamp Last activity timestamp
     */
    function getReputation(bytes32 nodeIdHash) external view returns (
        uint256 totalGradientsSubmitted,
        uint256 totalCreditsEarned,
        uint256 byzantineEventCount,
        uint256 averagePogqScore,
        uint256 lastActivityTimestamp
    ) {
        Reputation memory rep = reputations[nodeIdHash];
        return (
            rep.totalGradientsSubmitted,
            rep.totalCreditsEarned,
            rep.byzantineEventCount,
            rep.averagePogqScore,
            rep.lastActivityTimestamp
        );
    }

    // ============================================
    // STATISTICS FUNCTIONS
    // ============================================

    /**
     * @notice Get global statistics
     * @return totalGradients Total gradients stored
     * @return totalCreditsIssued Total credits issued
     * @return totalByzantineEvents Total Byzantine events logged
     */
    function getStats() external view returns (
        uint256,
        uint256,
        uint256
    ) {
        return (
            totalGradients,
            totalCreditsIssued,
            totalByzantineEvents
        );
    }

    /**
     * @notice Get contract version
     * @return Version string
     */
    function getVersion() external view returns (string memory) {
        return version;
    }

    // ============================================
    // ADMIN FUNCTIONS
    // ============================================

    /**
     * @notice Transfer ownership
     * @param newOwner New owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        owner = newOwner;
    }
}
