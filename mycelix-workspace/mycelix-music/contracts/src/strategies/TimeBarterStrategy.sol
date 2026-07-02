// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title TimeBarterStrategy
 * @notice Exchange time listening for time creating - circular gift economy
 * @dev Listeners earn time credits by listening, spend them on premium content
 */
contract TimeBarterStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    // Time credits (in seconds)
    struct TimeConfig {
        address artist;
        uint256 creditsPerMinute; // Credits earned per minute listened
        uint256 creditsRequired; // Credits required to access this song
        bool isPremium; // Does this song require credits?
        bool grantsCredits; // Does listening to this grant credits?
        bool initialized;
    }

    struct TimeAccount {
        uint256 timeCredits;
        uint256 totalListenTime; // In seconds
        uint256 lastListenTimestamp;
        uint256 creditsSpent;
    }

    mapping(bytes32 => TimeConfig) public configs;
    mapping(address => TimeAccount) public accounts;
    mapping(bytes32 => uint256) public songListenTime; // Total listen time per song
    mapping(bytes32 => mapping(address => uint256)) public userSongTime; // Per-user listen time

    // Default credits per minute (60 = 1 credit per second)
    uint256 public defaultCreditsPerMinute = 60;

    event TimeConfigured(bytes32 indexed songId, uint256 creditsRequired, bool isPremium);
    event CreditsEarned(address indexed listener, bytes32 indexed songId, uint256 credits, uint256 listenTime);
    event CreditsSpent(address indexed listener, bytes32 indexed songId, uint256 credits);
    event ListenSessionRecorded(bytes32 indexed songId, address indexed listener, uint256 duration);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configureTimeBarter(
        bytes32 songId,
        address artist,
        uint256 creditsPerMinute,
        uint256 creditsRequired,
        bool isPremium,
        bool grantsCredits
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");

        configs[songId] = TimeConfig({
            artist: artist,
            creditsPerMinute: creditsPerMinute > 0 ? creditsPerMinute : defaultCreditsPerMinute,
            creditsRequired: creditsRequired,
            isPremium: isPremium,
            grantsCredits: grantsCredits,
            initialized: true
        });

        emit TimeConfigured(songId, creditsRequired, isPremium);
    }

    /**
     * @notice Record listening time (called by frontend/SDK)
     * @param songId Song that was listened to
     * @param listener Listener address
     * @param durationSeconds Listening duration in seconds
     */
    function recordListenTime(bytes32 songId, address listener, uint256 durationSeconds) external {
        TimeConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        TimeAccount storage account = accounts[listener];

        // Record listen time
        userSongTime[songId][listener] += durationSeconds;
        songListenTime[songId] += durationSeconds;
        account.totalListenTime += durationSeconds;
        account.lastListenTimestamp = block.timestamp;

        // Award credits if song grants them
        if (config.grantsCredits) {
            uint256 creditsEarned = (durationSeconds * config.creditsPerMinute) / 60;
            account.timeCredits += creditsEarned;
            emit CreditsEarned(listener, songId, creditsEarned, durationSeconds);
        }

        emit ListenSessionRecorded(songId, listener, durationSeconds);
    }

    /**
     * @notice Spend credits to access premium content
     */
    function spendCredits(bytes32 songId) external {
        TimeConfig storage config = configs[songId];
        require(config.initialized, "Not configured");
        require(config.isPremium, "Not premium");

        TimeAccount storage account = accounts[msg.sender];
        require(account.timeCredits >= config.creditsRequired, "Insufficient credits");

        account.timeCredits -= config.creditsRequired;
        account.creditsSpent += config.creditsRequired;

        emit CreditsSpent(msg.sender, songId, config.creditsRequired);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        TimeConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        if (config.isPremium) {
            // Check if listener has spent credits
            TimeAccount storage account = accounts[listener];

            if (account.timeCredits >= config.creditsRequired) {
                // Deduct credits
                account.timeCredits -= config.creditsRequired;
                account.creditsSpent += config.creditsRequired;
                emit CreditsSpent(listener, songId, config.creditsRequired);
            } else if (amount > 0) {
                // Pay with tokens instead
                require(flowToken.transferFrom(router, config.artist, amount), "Transfer failed");
            } else {
                revert("Credits or payment required");
            }
        }

        // Record that a stream happened (time will be recorded separately)
        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        TimeConfig storage config = configs[songId];
        // Return 0 if user can pay with time credits
        if (!config.isPremium) return 0;
        // Premium songs need either credits or payment
        return 0; // Credits are the primary payment method
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        TimeConfig storage config = configs[songId];
        if (!config.initialized) return false;

        // Non-premium songs always authorized
        if (!config.isPremium) return true;

        // Check if listener has enough credits
        return accounts[listener].timeCredits >= config.creditsRequired;
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        TimeConfig storage config = configs[songId];

        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](1);
        splits[0] = EconomicStrategyRouter.Split({
            recipient: config.artist,
            basisPoints: 10000,
            role: "Artist"
        });

        return splits;
    }

    function getTimeAccount(address user) external view returns (TimeAccount memory) {
        return accounts[user];
    }

    function getListenStats(bytes32 songId, address user)
        external view returns (uint256 songTotal, uint256 userTime, uint256 userCredits)
    {
        return (songListenTime[songId], userSongTime[songId][user], accounts[user].timeCredits);
    }

    /**
     * @notice Grant bonus credits (for promotions, etc.)
     */
    function grantBonusCredits(address recipient, uint256 credits) external onlyOwner {
        accounts[recipient].timeCredits += credits;
        emit CreditsEarned(recipient, bytes32(0), credits, 0);
    }
}
