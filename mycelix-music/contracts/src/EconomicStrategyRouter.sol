// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title EconomicStrategyRouter
 * @notice Core routing contract that delegates listener payments to pluggable economic strategy contracts
 * @dev Artists can register each song with a different strategy. The router collects FLOW, applies protocol fees,
 *      and forwards net payments plus context to the selected strategy. It also exposes helpful read methods for the UI.
 */
contract EconomicStrategyRouter is Ownable, ReentrancyGuard {
    // ============================================================
    // Types
    // ============================================================

    enum PaymentType {
        STREAM,
        DOWNLOAD,
        TIP,
        PATRONAGE,
        NFT_ACCESS
    }

    struct Split {
        address recipient;
        uint256 basisPoints; // 10000 = 100%
        string role;
    }

    struct Payment {
        bytes32 songId;
        address listener;
        uint256 amount; // Gross amount paid before protocol fee
        PaymentType paymentType;
        uint256 timestamp;
    }

    // ============================================================
    // State
    // ============================================================

    IERC20 public immutable flowToken;

    // Strategy registry
    mapping(bytes32 => address) public registeredStrategies;

    // Song configuration
    mapping(bytes32 => address) public songStrategy;      // songId => strategy contract address
    mapping(bytes32 => bytes32) public songStrategyId;    // songId => strategy identifier
    mapping(bytes32 => address) public songArtist;        // songId => artist wallet

    // Protocol economics
    uint256 public protocolFeeBps = 100; // 1%
    address public protocolTreasury;

    // Payment history
    mapping(bytes32 => Payment[]) private paymentHistory;

    // ============================================================
    // Events
    // ============================================================

    event StrategyRegistered(bytes32 indexed strategyId, address strategyAddress);
    event SongRegistered(bytes32 indexed songId, bytes32 indexed strategyId, address indexed artist);
    event SongStrategyChanged(bytes32 indexed songId, bytes32 indexed strategyId);
    event PaymentProcessed(
        bytes32 indexed songId,
        address indexed listener,
        uint256 amount,
        PaymentType paymentType
    );
    event ProtocolFeeUpdated(uint256 newFeeBps);
    event ProtocolTreasuryUpdated(address newTreasury);

    // ============================================================
    // Constructor
    // ============================================================

    constructor(address _flowToken, address _protocolTreasury) {
        require(_flowToken != address(0), "Invalid FLOW token address");
        require(_protocolTreasury != address(0), "Invalid treasury address");

        flowToken = IERC20(_flowToken);
        protocolTreasury = _protocolTreasury;
    }

    // ============================================================
    // Strategy Registry
    // ============================================================

    function registerStrategy(bytes32 strategyId, address strategyAddress) external onlyOwner {
        require(strategyAddress != address(0), "Invalid strategy address");
        require(registeredStrategies[strategyId] == address(0), "Strategy already registered");

        registeredStrategies[strategyId] = strategyAddress;
        emit StrategyRegistered(strategyId, strategyAddress);
    }

    // ============================================================
    // Artist Actions
    // ============================================================

    /**
     * @notice Register a song with a specific economic strategy
     * @dev Caller becomes the artist of record for this song
     */
    function registerSong(bytes32 songId, bytes32 strategyId) external {
        require(songStrategy[songId] == address(0), "Song already registered");

        address strategyAddress = registeredStrategies[strategyId];
        require(strategyAddress != address(0), "Strategy not registered");

        songArtist[songId] = msg.sender;
        songStrategy[songId] = strategyAddress;
        songStrategyId[songId] = strategyId;

        emit SongRegistered(songId, strategyId, msg.sender);
    }

    /**
     * @notice Change the economic strategy for an existing song
     */
    function changeSongStrategy(bytes32 songId, bytes32 strategyId) external {
        require(songArtist[songId] == msg.sender, "Not song artist");

        address strategyAddress = registeredStrategies[strategyId];
        require(strategyAddress != address(0), "Strategy not registered");

        songStrategy[songId] = strategyAddress;
        songStrategyId[songId] = strategyId;

        emit SongStrategyChanged(songId, strategyId);
    }

    // ============================================================
    // Listener Payments
    // ============================================================

    function processPayment(
        bytes32 songId,
        uint256 amount,
        PaymentType paymentType
    ) external nonReentrant {
        _processPayment(songId, amount, paymentType);
    }

    function batchProcessPayments(
        bytes32[] calldata songIds,
        uint256[] calldata amounts,
        PaymentType[] calldata paymentTypes
    ) external nonReentrant {
        require(
            songIds.length == amounts.length && songIds.length == paymentTypes.length,
            "Length mismatch"
        );

        for (uint256 i = 0; i < songIds.length; i++) {
            _processPayment(songIds[i], amounts[i], paymentTypes[i]);
        }
    }

    function _processPayment(
        bytes32 songId,
        uint256 amount,
        PaymentType paymentType
    ) internal {
        address strategyAddress = songStrategy[songId];
        require(strategyAddress != address(0), "Song not registered");
        require(amount > 0, "Amount must be positive");

        IEconomicStrategy strategy = IEconomicStrategy(strategyAddress);

        uint256 minPayment = strategy.getMinPayment(songId, paymentType);
        require(amount >= minPayment, "Payment below minimum");

        require(
            flowToken.transferFrom(msg.sender, address(this), amount),
            "FLOW transfer failed"
        );

        uint256 protocolFee = (amount * protocolFeeBps) / 10000;
        uint256 netAmount = amount - protocolFee;

        if (protocolFee > 0) {
            require(
                flowToken.transfer(protocolTreasury, protocolFee),
                "Protocol fee transfer failed"
            );
        }

        require(flowToken.approve(strategyAddress, netAmount), "Approval failed");
        strategy.processPayment(songId, msg.sender, netAmount, paymentType);

        paymentHistory[songId].push(
            Payment({
                songId: songId,
                listener: msg.sender,
                amount: amount,
                paymentType: paymentType,
                timestamp: block.timestamp
            })
        );

        emit PaymentProcessed(songId, msg.sender, amount, paymentType);
    }

    // ============================================================
    // View Helpers
    // ============================================================

    function getPaymentHistory(bytes32 songId) external view returns (Payment[] memory) {
        return paymentHistory[songId];
    }

    function previewSplits(bytes32 songId, uint256 amount) external view returns (Split[] memory) {
        address strategyAddress = songStrategy[songId];
        require(strategyAddress != address(0), "Song not registered");

        return IEconomicStrategy(strategyAddress).calculateSplits(songId, amount);
    }

    function isAuthorized(bytes32 songId, address listener) external view returns (bool) {
        address strategyAddress = songStrategy[songId];
        if (strategyAddress == address(0)) return false;

        return IEconomicStrategy(strategyAddress).isAuthorized(songId, listener);
    }

    function getStrategyForSong(bytes32 songId) external view returns (address) {
        return songStrategy[songId];
    }

    // ============================================================
    // Protocol Admin
    // ============================================================

    function updateProtocolFee(uint256 newFeeBps) external onlyOwner {
        require(newFeeBps <= 500, "Fee too high (max 5%)");
        protocolFeeBps = newFeeBps;
        emit ProtocolFeeUpdated(newFeeBps);
    }

    function updateProtocolTreasury(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), "Invalid treasury address");
        protocolTreasury = newTreasury;
        emit ProtocolTreasuryUpdated(newTreasury);
    }
}

/**
 * @title IEconomicStrategy
 * @notice Interface implemented by every economic strategy
 */
interface IEconomicStrategy {
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        EconomicStrategyRouter.PaymentType paymentType
    ) external;

    function getMinPayment(
        bytes32 songId,
        EconomicStrategyRouter.PaymentType paymentType
    ) external view returns (uint256);

    function isAuthorized(bytes32 songId, address listener) external view returns (bool);

    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) external view returns (EconomicStrategyRouter.Split[] memory);
}
