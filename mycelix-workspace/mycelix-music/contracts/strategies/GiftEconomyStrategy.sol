// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title Gift Economy Strategy
 * @notice Free streaming with optional gifting using CGC tokens
 * @dev Implements attention-weighted distribution where engaged listeners contribute more
 */
contract GiftEconomyStrategy is IEconomicStrategy {

    // ========== TYPES ==========

    struct GiftConfig {
        bool acceptsGifts;
        uint256 minGiftAmount;
        address[] defaultRecipients;  // Where gifts go by default
        uint256[] defaultSplits;
        bool initialized;
    }

    struct ListenerProfile {
        uint256 totalGiftsGiven;
        uint256 totalStreamsCount;
        uint256 lastInteraction;
        uint256 cgcBalance;  // Claimable CGC rewards
    }

    // ========== STATE ==========

    IERC20 public flowToken;
    IERC20 public cgcToken;
    address public router;

    // Song ID => Gift configuration
    mapping(bytes32 => GiftConfig) public giftConfigs;

    // Artist => Listener => Profile
    mapping(address => mapping(address => ListenerProfile)) public listenerProfiles;

    // CGC distribution per listen (set by DAO)
    uint256 public cgcPerListen = 1 ether;  // 1 CGC per listen

    // Bonus multipliers for engaged listeners
    uint256 public repeatListenerMultiplier = 150;  // 1.5x for repeat listens
    uint256 public earlyListenerBonus = 10 ether;   // 10 CGC for first 100 listeners

    // Track early listeners
    mapping(bytes32 => uint256) public listenerCount;
    uint256 public constant EARLY_LISTENER_THRESHOLD = 100;

    // ========== EVENTS ==========

    event GiftReceived(bytes32 indexed songId, address indexed listener, uint256 amount);
    event CGCDistributed(address indexed listener, uint256 amount);
    event ListenerRewarded(address indexed listener, uint256 bonus, string reason);

    // ========== CONSTRUCTOR ==========

    constructor(
        address _flowToken,
        address _cgcToken,
        address _router
    ) {
        flowToken = IERC20(_flowToken);
        cgcToken = IERC20(_cgcToken);
        router = _router;
    }

    // ========== MODIFIERS ==========

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    // ========== CONFIGURATION ==========

    /**
     * @notice Artist configures gift acceptance for their song
     */
    function configureGifts(
        bytes32 songId,
        bool acceptsGifts,
        uint256 minGiftAmount,
        address[] calldata recipients,
        uint256[] calldata splits
    ) external {
        require(recipients.length == splits.length, "Length mismatch");
        require(!giftConfigs[songId].initialized, "Already configured");

        // Validate splits sum to 10000
        uint256 total = 0;
        for (uint256 i = 0; i < splits.length; i++) {
            total += splits[i];
        }
        require(total == 10000, "Must sum to 100%");

        giftConfigs[songId] = GiftConfig({
            acceptsGifts: acceptsGifts,
            minGiftAmount: minGiftAmount,
            defaultRecipients: recipients,
            defaultSplits: splits,
            initialized: true
        });
    }

    // ========== PAYMENT PROCESSING ==========

    /**
     * @notice Process "payment" - actually free listening with CGC rewards
     * @dev Listeners don't pay, they receive CGC for listening
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        EconomicStrategyRouter.PaymentType paymentType
    ) external override onlyRouter {
        require(giftConfigs[songId].initialized, "Not configured");

        GiftConfig memory config = giftConfigs[songId];

        if (paymentType == EconomicStrategyRouter.PaymentType.STREAM) {
            // FREE LISTENING - reward listener with CGC
            _rewardListener(songId, listener);

        } else if (paymentType == EconomicStrategyRouter.PaymentType.TIP) {
            // OPTIONAL GIFT - listener voluntarily tips artist
            require(config.acceptsGifts, "Gifts not accepted");
            require(amount >= config.minGiftAmount, "Gift too small");

            _distributeGift(songId, amount, config);

            emit GiftReceived(songId, listener, amount);
        }

        // Update listener profile
        ListenerProfile storage profile = listenerProfiles[
            config.defaultRecipients[0]  // Primary artist
        ][listener];

        profile.totalStreamsCount++;
        profile.lastInteraction = block.timestamp;
    }

    /**
     * @notice Reward listener with CGC for engaging with music
     */
    function _rewardListener(bytes32 songId, address listener) internal {
        uint256 reward = cgcPerListen;

        // Early listener bonus
        if (listenerCount[songId] < EARLY_LISTENER_THRESHOLD) {
            reward += earlyListenerBonus;
            listenerCount[songId]++;

            emit ListenerRewarded(listener, earlyListenerBonus, "early_listener");
        }

        // Repeat listener bonus
        GiftConfig memory config = giftConfigs[songId];
        ListenerProfile storage profile = listenerProfiles[
            config.defaultRecipients[0]
        ][listener];

        if (profile.totalStreamsCount > 5) {
            uint256 bonus = (reward * repeatListenerMultiplier) / 100 - reward;
            reward += bonus;

            emit ListenerRewarded(listener, bonus, "repeat_listener");
        }

        // Credit CGC to listener (they can claim later)
        profile.cgcBalance += reward;

        emit CGCDistributed(listener, reward);
    }

    /**
     * @notice Distribute voluntary gift to artists
     */
    function _distributeGift(
        bytes32 songId,
        uint256 amount,
        GiftConfig memory config
    ) internal {
        for (uint256 i = 0; i < config.defaultRecipients.length; i++) {
            uint256 recipientAmount = (amount * config.defaultSplits[i]) / 10000;

            require(
                flowToken.transfer(config.defaultRecipients[i], recipientAmount),
                "Gift transfer failed"
            );
        }
    }

    // ========== LISTENER FUNCTIONS ==========

    /**
     * @notice Listener claims their accumulated CGC rewards
     */
    function claimCGCRewards(address artist) external {
        ListenerProfile storage profile = listenerProfiles[artist][msg.sender];
        uint256 claimable = profile.cgcBalance;

        require(claimable > 0, "Nothing to claim");

        profile.cgcBalance = 0;

        require(
            cgcToken.transfer(msg.sender, claimable),
            "CGC transfer failed"
        );
    }

    // ========== VIEW FUNCTIONS ==========

    /**
     * @notice Calculate splits (for preview)
     */
    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) external view override returns (EconomicStrategyRouter.Split[] memory) {
        require(giftConfigs[songId].initialized, "Not configured");

        GiftConfig memory config = giftConfigs[songId];
        EconomicStrategyRouter.Split[] memory result =
            new EconomicStrategyRouter.Split[](config.defaultRecipients.length);

        for (uint256 i = 0; i < config.defaultRecipients.length; i++) {
            result[i] = EconomicStrategyRouter.Split({
                recipient: config.defaultRecipients[i],
                basisPoints: config.defaultSplits[i],
                role: "artist"  // Simplified for gift economy
            });
        }

        return result;
    }

    /**
     * @notice Get listener's CGC balance for an artist
     */
    function getCGCBalance(
        address artist,
        address listener
    ) external view returns (uint256) {
        return listenerProfiles[artist][listener].cgcBalance;
    }

    /**
     * @notice Get listener stats
     */
    function getListenerStats(
        address artist,
        address listener
    ) external view returns (
        uint256 totalGifts,
        uint256 totalStreams,
        uint256 cgcBalance
    ) {
        ListenerProfile memory profile = listenerProfiles[artist][listener];
        return (
            profile.totalGiftsGiven,
            profile.totalStreamsCount,
            profile.cgcBalance
        );
    }
}
