// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title GiftEconomyStrategy
 * @notice Revolutionary model: Free listening + listeners earn CGC tokens + voluntary tips
 * @dev Artists receive voluntary tips, listeners earn CGC for engagement
 */
contract GiftEconomyStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    // CGC Registry interface (from commons-charter)
    interface ICGCRegistry {
        function awardCGC(address recipient, uint256 amount, string memory reason) external;
    }

    ICGCRegistry public cgcRegistry;

    // Configuration per song
    struct GiftConfig {
        address artist;
        uint256 cgcPerListen; // CGC awarded to listener per stream
        uint256 earlyListenerBonus; // Extra CGC for first N listeners
        uint256 earlyListenerThreshold; // Number of early listeners
        uint256 repeatListenerMultiplier; // Multiplier for loyal fans (in basis points)
        bool initialized;
    }

    mapping(bytes32 => GiftConfig) public giftConfigs;

    // Listener tracking
    struct ListenerProfile {
        uint256 totalStreamsCount;
        uint256 lastStreamTimestamp;
        uint256 cgcBalance;
        bool isEarlyListener;
    }

    mapping(bytes32 => mapping(address => ListenerProfile)) public listenerProfiles;
    mapping(bytes32 => uint256) public listenerCount;
    mapping(bytes32 => uint256) public totalTipsReceived;

    // Events
    event GiftConfigured(
        bytes32 indexed songId,
        address artist,
        uint256 cgcPerListen,
        uint256 earlyListenerBonus
    );
    event StreamReward(
        bytes32 indexed songId,
        address indexed listener,
        uint256 cgcAmount,
        bool isEarlyListener,
        bool isRepeatListener
    );
    event TipReceived(bytes32 indexed songId, address indexed listener, uint256 amount);
    event CGCDistributed(address indexed listener, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router, address _cgcRegistry) {
        require(_flowToken != address(0), "Invalid FLOW token");
        require(_router != address(0), "Invalid router");
        require(_cgcRegistry != address(0), "Invalid CGC registry");

        flowToken = IERC20(_flowToken);
        router = _router;
        cgcRegistry = ICGCRegistry(_cgcRegistry);
    }

    /**
     * @notice Configure gift economy parameters for a song
     * @param songId Unique identifier for the song
     * @param artist Address of the artist
     * @param cgcPerListen CGC awarded per stream
     * @param earlyListenerBonus Extra CGC for early listeners
     * @param earlyListenerThreshold Number of early listeners
     * @param repeatListenerMultiplier Multiplier for repeat listeners (10000 = 1x)
     */
    function configureGiftEconomy(
        bytes32 songId,
        address artist,
        uint256 cgcPerListen,
        uint256 earlyListenerBonus,
        uint256 earlyListenerThreshold,
        uint256 repeatListenerMultiplier
    ) external {
        require(!giftConfigs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(cgcPerListen > 0, "CGC per listen must be > 0");

        giftConfigs[songId] = GiftConfig({
            artist: artist,
            cgcPerListen: cgcPerListen,
            earlyListenerBonus: earlyListenerBonus,
            earlyListenerThreshold: earlyListenerThreshold,
            repeatListenerMultiplier: repeatListenerMultiplier,
            initialized: true
        });

        emit GiftConfigured(songId, artist, cgcPerListen, earlyListenerBonus);
    }

    /**
     * @notice Process a payment (either free stream or tip)
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @param amount Amount of FLOW tokens (0 for free stream, >0 for tip)
     * @param paymentType Type of payment
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        GiftConfig memory config = giftConfigs[songId];
        require(config.initialized, "Gift economy not configured");

        if (amount == 0 || paymentType == PaymentType.STREAM) {
            // Free stream - reward listener with CGC
            _rewardListener(songId, listener, config);
        } else {
            // Voluntary tip - transfer to artist
            require(
                flowToken.transferFrom(router, config.artist, amount),
                "Tip transfer failed"
            );

            totalTipsReceived[songId] += amount;
            emit TipReceived(songId, listener, amount);

            // Also reward listener for tipping (double reward!)
            _rewardListener(songId, listener, config);
        }
    }

    /**
     * @notice Internal function to reward listener with CGC
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @param config Gift economy configuration
     */
    function _rewardListener(
        bytes32 songId,
        address listener,
        GiftConfig memory config
    ) internal {
        ListenerProfile storage profile = listenerProfiles[songId][listener];

        uint256 reward = config.cgcPerListen;
        bool isEarly = false;
        bool isRepeat = false;

        // Early listener bonus
        if (listenerCount[songId] < config.earlyListenerThreshold && !profile.isEarlyListener) {
            reward += config.earlyListenerBonus;
            profile.isEarlyListener = true;
            isEarly = true;
        }

        // Repeat listener multiplier
        if (profile.totalStreamsCount > 5) {
            reward = (reward * config.repeatListenerMultiplier) / 10000;
            isRepeat = true;
        }

        // Update profile
        if (profile.totalStreamsCount == 0) {
            listenerCount[songId]++;
        }

        profile.totalStreamsCount++;
        profile.lastStreamTimestamp = block.timestamp;
        profile.cgcBalance += reward;

        // Award CGC through commons-charter registry
        cgcRegistry.awardCGC(
            listener,
            reward,
            string(abi.encodePacked("Listened to song: ", bytes32ToString(songId)))
        );

        emit StreamReward(songId, listener, reward, isEarly, isRepeat);
        emit CGCDistributed(listener, reward);
    }

    /**
     * @notice Get minimum payment (always 0 for gift economy)
     * @param songId Unique identifier for the song
     * @param paymentType Type of payment
     * @return Always 0 (listening is free)
     */
    function getMinPayment(bytes32 songId, PaymentType paymentType)
        external
        pure
        override
        returns (uint256)
    {
        // Unused parameters
        songId;
        paymentType;

        return 0; // Free!
    }

    /**
     * @notice Check if listener is authorized (always true in gift economy)
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @return Always true (free for everyone)
     */
    function isAuthorized(bytes32 songId, address listener) external pure override returns (bool) {
        // Unused parameters
        songId;
        listener;

        return true; // Free for all!
    }

    /**
     * @notice Get listener profile
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @return profile Listener's engagement profile
     */
    function getListenerProfile(bytes32 songId, address listener)
        external
        view
        returns (ListenerProfile memory)
    {
        return listenerProfiles[songId][listener];
    }

    /**
     * @notice Get gift economy stats for a song
     * @param songId Unique identifier for the song
     * @return config Gift economy configuration
     * @return listeners Total number of unique listeners
     * @return totalTips Total tips received by artist
     */
    function getSongStats(bytes32 songId)
        external
        view
        returns (
            GiftConfig memory config,
            uint256 listeners,
            uint256 totalTips
        )
    {
        return (giftConfigs[songId], listenerCount[songId], totalTipsReceived[songId]);
    }

    /**
     * @notice Convert bytes32 to string (helper function)
     * @param _bytes32 Bytes32 to convert
     * @return String representation
     */
    function bytes32ToString(bytes32 _bytes32) internal pure returns (string memory) {
        uint8 i = 0;
        while (i < 32 && _bytes32[i] != 0) {
            i++;
        }
        bytes memory bytesArray = new bytes(i);
        for (i = 0; i < 32 && _bytes32[i] != 0; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }

    /**
     * @notice Update CGC registry address (owner only)
     * @param newRegistry New CGC registry address
     */
    function updateCGCRegistry(address newRegistry) external onlyOwner {
        require(newRegistry != address(0), "Invalid registry");
        cgcRegistry = ICGCRegistry(newRegistry);
    }
}
