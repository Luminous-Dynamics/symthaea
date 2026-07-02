// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title Economic Strategy Router
 * @notice Allows each song to have its own economic operating system
 * @dev Routes payments through pluggable economic strategy modules
 */
contract EconomicStrategyRouter is Ownable {

    // ========== TYPES ==========

    enum PaymentType {
        STREAM,
        DOWNLOAD,
        TIP,
        PATRONAGE,
        NFT_ACCESS
    }

    struct Split {
        address recipient;
        uint256 basisPoints;  // 10000 = 100%
        string role;
    }

    struct Payment {
        bytes32 songId;
        address listener;
        uint256 amount;
        PaymentType paymentType;
        uint256 timestamp;
    }

    // ========== STATE ==========

    // Song ID => Economic Strategy Contract
    mapping(bytes32 => address) public songStrategy;

    // Strategy ID => Strategy Contract
    mapping(bytes32 => address) public registeredStrategies;

    // Song ID => Artist DID (authorization)
    mapping(bytes32 => address) public songArtist;

    // FLOW token contract
    IERC20 public flowToken;

    // Payment history (for analytics)
    mapping(bytes32 => Payment[]) public paymentHistory;

    // ========== EVENTS ==========

    event StrategyRegistered(bytes32 indexed strategyId, address strategyAddress);
    event StrategySet(bytes32 indexed songId, bytes32 indexed strategyId);
    event PaymentProcessed(bytes32 indexed songId, address indexed listener, uint256 amount, PaymentType paymentType);

    // ========== CONSTRUCTOR ==========

    constructor(address _flowToken) Ownable(msg.sender) {
        flowToken = IERC20(_flowToken);
    }

    // ========== ADMIN FUNCTIONS ==========

    /**
     * @notice Register a new economic strategy
     * @param strategyId Unique identifier for this strategy
     * @param strategyAddress Contract implementing IEconomicStrategy
     */
    function registerStrategy(
        bytes32 strategyId,
        address strategyAddress
    ) external onlyOwner {
        require(strategyAddress != address(0), "Invalid strategy address");
        require(registeredStrategies[strategyId] == address(0), "Strategy already registered");

        registeredStrategies[strategyId] = strategyAddress;
        emit StrategyRegistered(strategyId, strategyAddress);
    }

    // ========== ARTIST FUNCTIONS ==========

    /**
     * @notice Artist registers their song and sets initial strategy
     * @param songId Unique song identifier (hash from DKG)
     * @param strategyId Economic strategy to use
     */
    function registerSong(
        bytes32 songId,
        bytes32 strategyId
    ) external {
        require(songArtist[songId] == address(0), "Song already registered");
        require(registeredStrategies[strategyId] != address(0), "Strategy not found");

        songArtist[songId] = msg.sender;
        songStrategy[songId] = registeredStrategies[strategyId];

        emit StrategySet(songId, strategyId);
    }

    /**
     * @notice Artist changes economic strategy for their song
     * @param songId Song to update
     * @param strategyId New strategy to use
     */
    function changeSongStrategy(
        bytes32 songId,
        bytes32 strategyId
    ) external {
        require(songArtist[songId] == msg.sender, "Not song owner");
        require(registeredStrategies[strategyId] != address(0), "Strategy not found");

        songStrategy[songId] = registeredStrategies[strategyId];
        emit StrategySet(songId, strategyId);
    }

    // ========== LISTENER FUNCTIONS ==========

    /**
     * @notice Process a payment for listening to a song
     * @param songId Song being played
     * @param amount FLOW tokens to pay
     * @param paymentType Type of payment (stream, download, tip, etc.)
     */
    function processPayment(
        bytes32 songId,
        uint256 amount,
        PaymentType paymentType
    ) external {
        require(songStrategy[songId] != address(0), "Song not registered");
        require(amount > 0, "Amount must be positive");

        // Transfer FLOW from listener to this contract
        require(
            flowToken.transferFrom(msg.sender, address(this), amount),
            "Transfer failed"
        );

        // Delegate to strategy contract
        IEconomicStrategy strategy = IEconomicStrategy(songStrategy[songId]);
        strategy.processPayment(songId, msg.sender, amount, paymentType);

        // Record payment
        paymentHistory[songId].push(Payment({
            songId: songId,
            listener: msg.sender,
            amount: amount,
            paymentType: paymentType,
            timestamp: block.timestamp
        }));

        emit PaymentProcessed(songId, msg.sender, amount, paymentType);
    }

    /**
     * @notice Batch process multiple payments (gas optimization)
     */
    function batchProcessPayments(
        bytes32[] calldata songIds,
        uint256[] calldata amounts,
        PaymentType[] calldata paymentTypes
    ) external {
        require(songIds.length == amounts.length, "Length mismatch");
        require(songIds.length == paymentTypes.length, "Length mismatch");

        for (uint256 i = 0; i < songIds.length; i++) {
            processPayment(songIds[i], amounts[i], paymentTypes[i]);
        }
    }

    // ========== VIEW FUNCTIONS ==========

    /**
     * @notice Get payment history for a song
     */
    function getPaymentHistory(bytes32 songId) external view returns (Payment[] memory) {
        return paymentHistory[songId];
    }

    /**
     * @notice Calculate how much each recipient would receive
     */
    function previewSplits(
        bytes32 songId,
        uint256 amount
    ) external view returns (Split[] memory) {
        require(songStrategy[songId] != address(0), "Song not registered");

        IEconomicStrategy strategy = IEconomicStrategy(songStrategy[songId]);
        return strategy.calculateSplits(songId, amount);
    }
}

/**
 * @title Economic Strategy Interface
 * @notice All economic strategy contracts must implement this interface
 */
interface IEconomicStrategy {
    /**
     * @notice Process a payment according to this strategy's rules
     * @param songId Song being paid for
     * @param listener Address of the listener paying
     * @param amount Amount of FLOW tokens
     * @param paymentType Type of payment
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        EconomicStrategyRouter.PaymentType paymentType
    ) external;

    /**
     * @notice Calculate how payment would be split
     * @param songId Song ID
     * @param amount Amount to split
     * @return Array of splits
     */
    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) external view returns (EconomicStrategyRouter.Split[] memory);
}
