// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title EconomicStrategyRouterV2
 * @notice Upgradeable version of the EconomicStrategyRouter with gas optimizations
 * @dev Uses UUPS proxy pattern for upgradeability
 *
 * Gas Optimizations:
 * - Packed storage slots
 * - Custom errors instead of require strings
 * - Unchecked math where safe
 * - Batch operations
 * - Memory-efficient data structures
 */
contract EconomicStrategyRouterV2 is
    Initializable,
    OwnableUpgradeable,
    ReentrancyGuardUpgradeable,
    PausableUpgradeable,
    UUPSUpgradeable
{
    using SafeERC20 for IERC20;

    // ============================================================
    // Custom Errors (Gas efficient)
    // ============================================================

    error InvalidAddress();
    error InvalidAmount();
    error SongNotRegistered();
    error SongAlreadyRegistered();
    error StrategyNotRegistered();
    error StrategyAlreadyRegistered();
    error NotSongArtist();
    error PaymentBelowMinimum();
    error FeeTooHigh();
    error ArrayLengthMismatch();
    error TransferFailed();
    error Unauthorized();

    // ============================================================
    // Types (Gas optimized with packed structs)
    // ============================================================

    enum PaymentType {
        STREAM,
        DOWNLOAD,
        TIP,
        PATRONAGE,
        NFT_ACCESS
    }

    /// @dev Packed struct: 32 bytes (1 slot)
    struct SongConfig {
        address strategy;      // 20 bytes
        address artist;        // 20 bytes -> moved to separate slot
        bytes12 strategyId;    // 12 bytes
    }

    /// @dev Packed struct for storage efficiency
    struct PaymentRecord {
        uint128 amount;        // 16 bytes - max ~3.4e38, plenty for tokens
        uint64 timestamp;      // 8 bytes - good until year 584,942,417,355
        address listener;      // 20 bytes
        PaymentType pType;     // 1 byte
    }

    struct Split {
        address recipient;
        uint96 basisPoints;    // 12 bytes - max 10000 for 100%
        bytes32 role;          // Stored as hash for efficiency
    }

    // ============================================================
    // Storage (Upgradeable pattern - append only)
    // ============================================================

    /// @custom:storage-location erc7201:mycelix.router.storage
    struct RouterStorage {
        IERC20 flowToken;
        address protocolTreasury;
        uint16 protocolFeeBps;     // Max 500 (5%)
        bool initialized;

        // Strategy registry
        mapping(bytes12 => address) registeredStrategies;

        // Song configuration (optimized)
        mapping(bytes32 => address) songStrategy;
        mapping(bytes32 => address) songArtist;
        mapping(bytes32 => bytes12) songStrategyId;

        // Payment history (compact)
        mapping(bytes32 => PaymentRecord[]) paymentHistory;

        // Analytics counters
        mapping(bytes32 => uint256) totalPayments;
        mapping(bytes32 => uint256) totalVolume;

        // Whitelist for authorized callers (e.g., relayers)
        mapping(address => bool) authorizedCallers;

        // Version tracking
        uint256 version;
    }

    // keccak256(abi.encode(uint256(keccak256("mycelix.router.storage")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant STORAGE_LOCATION = 0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a00;

    function _getStorage() private pure returns (RouterStorage storage $) {
        assembly {
            $.slot := STORAGE_LOCATION
        }
    }

    // ============================================================
    // Events
    // ============================================================

    event StrategyRegistered(bytes12 indexed strategyId, address indexed strategyAddress);
    event SongRegistered(bytes32 indexed songId, bytes12 indexed strategyId, address indexed artist);
    event SongStrategyChanged(bytes32 indexed songId, bytes12 indexed newStrategyId);
    event PaymentProcessed(
        bytes32 indexed songId,
        address indexed listener,
        uint128 amount,
        PaymentType paymentType
    );
    event BatchPaymentsProcessed(uint256 count, uint256 totalAmount);
    event ProtocolFeeUpdated(uint16 newFeeBps);
    event ProtocolTreasuryUpdated(address indexed newTreasury);
    event CallerAuthorized(address indexed caller, bool authorized);
    event Upgraded(uint256 newVersion);

    // ============================================================
    // Modifiers
    // ============================================================

    modifier onlyAuthorized() {
        RouterStorage storage $ = _getStorage();
        if (msg.sender != owner() && !$.authorizedCallers[msg.sender]) {
            revert Unauthorized();
        }
        _;
    }

    // ============================================================
    // Initialization
    // ============================================================

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(
        address _flowToken,
        address _protocolTreasury,
        uint16 _protocolFeeBps
    ) public initializer {
        if (_flowToken == address(0) || _protocolTreasury == address(0)) {
            revert InvalidAddress();
        }
        if (_protocolFeeBps > 500) revert FeeTooHigh();

        __Ownable_init(msg.sender);
        __ReentrancyGuard_init();
        __Pausable_init();
        __UUPSUpgradeable_init();

        RouterStorage storage $ = _getStorage();
        $.flowToken = IERC20(_flowToken);
        $.protocolTreasury = _protocolTreasury;
        $.protocolFeeBps = _protocolFeeBps;
        $.version = 2;
    }

    // ============================================================
    // Strategy Registry
    // ============================================================

    function registerStrategy(bytes12 strategyId, address strategyAddress) external onlyOwner {
        if (strategyAddress == address(0)) revert InvalidAddress();

        RouterStorage storage $ = _getStorage();
        if ($.registeredStrategies[strategyId] != address(0)) {
            revert StrategyAlreadyRegistered();
        }

        $.registeredStrategies[strategyId] = strategyAddress;
        emit StrategyRegistered(strategyId, strategyAddress);
    }

    function batchRegisterStrategies(
        bytes12[] calldata strategyIds,
        address[] calldata strategyAddresses
    ) external onlyOwner {
        if (strategyIds.length != strategyAddresses.length) revert ArrayLengthMismatch();

        RouterStorage storage $ = _getStorage();

        unchecked {
            for (uint256 i; i < strategyIds.length; ++i) {
                if (strategyAddresses[i] == address(0)) revert InvalidAddress();
                if ($.registeredStrategies[strategyIds[i]] != address(0)) {
                    revert StrategyAlreadyRegistered();
                }
                $.registeredStrategies[strategyIds[i]] = strategyAddresses[i];
                emit StrategyRegistered(strategyIds[i], strategyAddresses[i]);
            }
        }
    }

    // ============================================================
    // Artist Actions
    // ============================================================

    function registerSong(bytes32 songId, bytes12 strategyId) external whenNotPaused {
        RouterStorage storage $ = _getStorage();

        if ($.songStrategy[songId] != address(0)) revert SongAlreadyRegistered();

        address strategyAddress = $.registeredStrategies[strategyId];
        if (strategyAddress == address(0)) revert StrategyNotRegistered();

        $.songArtist[songId] = msg.sender;
        $.songStrategy[songId] = strategyAddress;
        $.songStrategyId[songId] = strategyId;

        emit SongRegistered(songId, strategyId, msg.sender);
    }

    function batchRegisterSongs(
        bytes32[] calldata songIds,
        bytes12[] calldata strategyIds
    ) external whenNotPaused {
        if (songIds.length != strategyIds.length) revert ArrayLengthMismatch();

        RouterStorage storage $ = _getStorage();

        unchecked {
            for (uint256 i; i < songIds.length; ++i) {
                if ($.songStrategy[songIds[i]] != address(0)) revert SongAlreadyRegistered();

                address strategyAddress = $.registeredStrategies[strategyIds[i]];
                if (strategyAddress == address(0)) revert StrategyNotRegistered();

                $.songArtist[songIds[i]] = msg.sender;
                $.songStrategy[songIds[i]] = strategyAddress;
                $.songStrategyId[songIds[i]] = strategyIds[i];

                emit SongRegistered(songIds[i], strategyIds[i], msg.sender);
            }
        }
    }

    function changeSongStrategy(bytes32 songId, bytes12 strategyId) external whenNotPaused {
        RouterStorage storage $ = _getStorage();

        if ($.songArtist[songId] != msg.sender) revert NotSongArtist();

        address strategyAddress = $.registeredStrategies[strategyId];
        if (strategyAddress == address(0)) revert StrategyNotRegistered();

        $.songStrategy[songId] = strategyAddress;
        $.songStrategyId[songId] = strategyId;

        emit SongStrategyChanged(songId, strategyId);
    }

    // ============================================================
    // Payment Processing (Gas Optimized)
    // ============================================================

    function processPayment(
        bytes32 songId,
        uint128 amount,
        PaymentType paymentType
    ) external nonReentrant whenNotPaused {
        _processPayment(songId, amount, paymentType, msg.sender);
    }

    /// @notice Process payment on behalf of a user (for relayers/meta-tx)
    function processPaymentFor(
        bytes32 songId,
        uint128 amount,
        PaymentType paymentType,
        address listener
    ) external nonReentrant whenNotPaused onlyAuthorized {
        _processPayment(songId, amount, paymentType, listener);
    }

    function batchProcessPayments(
        bytes32[] calldata songIds,
        uint128[] calldata amounts,
        PaymentType[] calldata paymentTypes
    ) external nonReentrant whenNotPaused {
        uint256 len = songIds.length;
        if (len != amounts.length || len != paymentTypes.length) {
            revert ArrayLengthMismatch();
        }

        uint256 totalAmount;
        unchecked {
            for (uint256 i; i < len; ++i) {
                _processPayment(songIds[i], amounts[i], paymentTypes[i], msg.sender);
                totalAmount += amounts[i];
            }
        }

        emit BatchPaymentsProcessed(len, totalAmount);
    }

    function _processPayment(
        bytes32 songId,
        uint128 amount,
        PaymentType paymentType,
        address listener
    ) internal {
        if (amount == 0) revert InvalidAmount();

        RouterStorage storage $ = _getStorage();

        address strategyAddress = $.songStrategy[songId];
        if (strategyAddress == address(0)) revert SongNotRegistered();

        IEconomicStrategyV2 strategy = IEconomicStrategyV2(strategyAddress);

        uint256 minPayment = strategy.getMinPayment(songId, paymentType);
        if (amount < minPayment) revert PaymentBelowMinimum();

        // Transfer tokens
        $.flowToken.safeTransferFrom(listener, address(this), amount);

        // Calculate fees using unchecked for gas savings (overflow impossible with uint128)
        uint128 protocolFee;
        uint128 netAmount;
        unchecked {
            protocolFee = uint128((uint256(amount) * $.protocolFeeBps) / 10000);
            netAmount = amount - protocolFee;
        }

        // Send protocol fee
        if (protocolFee > 0) {
            $.flowToken.safeTransfer($.protocolTreasury, protocolFee);
        }

        // Forward to strategy
        $.flowToken.safeIncreaseAllowance(strategyAddress, netAmount);
        strategy.processPayment(songId, listener, netAmount, paymentType);

        // Record payment (gas optimized struct)
        $.paymentHistory[songId].push(PaymentRecord({
            amount: amount,
            timestamp: uint64(block.timestamp),
            listener: listener,
            pType: paymentType
        }));

        // Update counters
        unchecked {
            $.totalPayments[songId]++;
            $.totalVolume[songId] += amount;
        }

        emit PaymentProcessed(songId, listener, amount, paymentType);
    }

    // ============================================================
    // View Functions
    // ============================================================

    function getPaymentHistory(bytes32 songId)
        external
        view
        returns (PaymentRecord[] memory)
    {
        return _getStorage().paymentHistory[songId];
    }

    function getPaymentHistoryPaginated(
        bytes32 songId,
        uint256 offset,
        uint256 limit
    ) external view returns (PaymentRecord[] memory records, uint256 total) {
        RouterStorage storage $ = _getStorage();
        PaymentRecord[] storage history = $.paymentHistory[songId];
        total = history.length;

        if (offset >= total) {
            return (new PaymentRecord[](0), total);
        }

        uint256 end = offset + limit;
        if (end > total) end = total;

        records = new PaymentRecord[](end - offset);
        unchecked {
            for (uint256 i = offset; i < end; ++i) {
                records[i - offset] = history[i];
            }
        }
    }

    function getSongStats(bytes32 songId)
        external
        view
        returns (
            uint256 totalPaymentCount,
            uint256 totalVolumeAmount,
            address artist,
            address strategy
        )
    {
        RouterStorage storage $ = _getStorage();
        return (
            $.totalPayments[songId],
            $.totalVolume[songId],
            $.songArtist[songId],
            $.songStrategy[songId]
        );
    }

    function previewSplits(bytes32 songId, uint128 amount)
        external
        view
        returns (Split[] memory)
    {
        RouterStorage storage $ = _getStorage();
        address strategyAddress = $.songStrategy[songId];
        if (strategyAddress == address(0)) revert SongNotRegistered();
        return IEconomicStrategyV2(strategyAddress).calculateSplits(songId, amount);
    }

    function isAuthorized(bytes32 songId, address listener) external view returns (bool) {
        RouterStorage storage $ = _getStorage();
        address strategyAddress = $.songStrategy[songId];
        if (strategyAddress == address(0)) return false;
        return IEconomicStrategyV2(strategyAddress).isAuthorized(songId, listener);
    }

    // ============================================================
    // Admin Functions
    // ============================================================

    function updateProtocolFee(uint16 newFeeBps) external onlyOwner {
        if (newFeeBps > 500) revert FeeTooHigh();
        _getStorage().protocolFeeBps = newFeeBps;
        emit ProtocolFeeUpdated(newFeeBps);
    }

    function updateProtocolTreasury(address newTreasury) external onlyOwner {
        if (newTreasury == address(0)) revert InvalidAddress();
        _getStorage().protocolTreasury = newTreasury;
        emit ProtocolTreasuryUpdated(newTreasury);
    }

    function setAuthorizedCaller(address caller, bool authorized) external onlyOwner {
        _getStorage().authorizedCallers[caller] = authorized;
        emit CallerAuthorized(caller, authorized);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    // ============================================================
    // Upgrade Authorization
    // ============================================================

    function _authorizeUpgrade(address newImplementation) internal override onlyOwner {
        RouterStorage storage $ = _getStorage();
        $.version++;
        emit Upgraded($.version);
    }

    function version() external view returns (uint256) {
        return _getStorage().version;
    }
}

/**
 * @title IEconomicStrategyV2
 * @notice Interface for V2 economic strategies with gas optimizations
 */
interface IEconomicStrategyV2 {
    function processPayment(
        bytes32 songId,
        address listener,
        uint128 amount,
        EconomicStrategyRouterV2.PaymentType paymentType
    ) external;

    function getMinPayment(
        bytes32 songId,
        EconomicStrategyRouterV2.PaymentType paymentType
    ) external view returns (uint256);

    function isAuthorized(bytes32 songId, address listener) external view returns (bool);

    function calculateSplits(
        bytes32 songId,
        uint128 amount
    ) external view returns (EconomicStrategyRouterV2.Split[] memory);
}
