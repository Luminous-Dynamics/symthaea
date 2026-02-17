// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title SubscriptionStrategy
 * @notice Monthly subscription model - subscribers get unlimited access to artist's catalog
 * @dev Subscribers pay monthly fee, can stream any song from subscribed artists
 */
contract SubscriptionStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    // Subscription tiers
    struct SubscriptionTier {
        uint256 monthlyPrice;
        uint256 maxStreamsPerMonth; // 0 = unlimited
        bool active;
    }

    // Artist subscription configuration
    struct ArtistConfig {
        address artist;
        SubscriptionTier[] tiers;
        bool initialized;
    }

    // Subscriber info
    struct Subscription {
        uint256 tierIndex;
        uint256 startTime;
        uint256 expiryTime;
        uint256 streamsThisMonth;
        uint256 monthStartTime;
    }

    mapping(bytes32 => ArtistConfig) public artistConfigs; // artistId => config
    mapping(bytes32 => mapping(address => Subscription)) public subscriptions; // artistId => subscriber => sub
    mapping(bytes32 => bytes32) public songToArtist; // songId => artistId
    mapping(bytes32 => uint256) public artistEarnings;

    event ArtistConfigured(bytes32 indexed artistId, address artist);
    event TierAdded(bytes32 indexed artistId, uint256 tierIndex, uint256 monthlyPrice);
    event SubscriptionCreated(bytes32 indexed artistId, address indexed subscriber, uint256 tierIndex);
    event SubscriptionRenewed(bytes32 indexed artistId, address indexed subscriber, uint256 newExpiry);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        require(_flowToken != address(0), "Invalid FLOW token");
        require(_router != address(0), "Invalid router");
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    /**
     * @notice Configure subscription for an artist
     */
    function configureArtist(
        bytes32 artistId,
        address artist,
        uint256[] calldata monthlyPrices,
        uint256[] calldata maxStreams
    ) external {
        require(!artistConfigs[artistId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(monthlyPrices.length == maxStreams.length, "Length mismatch");
        require(monthlyPrices.length > 0, "Need at least one tier");

        ArtistConfig storage config = artistConfigs[artistId];
        config.artist = artist;
        config.initialized = true;

        for (uint256 i = 0; i < monthlyPrices.length; i++) {
            config.tiers.push(SubscriptionTier({
                monthlyPrice: monthlyPrices[i],
                maxStreamsPerMonth: maxStreams[i],
                active: true
            }));
            emit TierAdded(artistId, i, monthlyPrices[i]);
        }

        emit ArtistConfigured(artistId, artist);
    }

    /**
     * @notice Register a song under an artist
     */
    function registerSong(bytes32 songId, bytes32 artistId) external {
        require(artistConfigs[artistId].initialized, "Artist not configured");
        songToArtist[songId] = artistId;
    }

    /**
     * @notice Subscribe to an artist
     */
    function subscribe(bytes32 artistId, uint256 tierIndex) external {
        ArtistConfig storage config = artistConfigs[artistId];
        require(config.initialized, "Artist not configured");
        require(tierIndex < config.tiers.length, "Invalid tier");
        require(config.tiers[tierIndex].active, "Tier not active");

        uint256 price = config.tiers[tierIndex].monthlyPrice;
        require(flowToken.transferFrom(msg.sender, config.artist, price), "Payment failed");

        subscriptions[artistId][msg.sender] = Subscription({
            tierIndex: tierIndex,
            startTime: block.timestamp,
            expiryTime: block.timestamp + 30 days,
            streamsThisMonth: 0,
            monthStartTime: block.timestamp
        });

        artistEarnings[artistId] += price;
        emit SubscriptionCreated(artistId, msg.sender, tierIndex);
    }

    /**
     * @notice Process a stream (check subscription validity)
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        bytes32 artistId = songToArtist[songId];
        require(artistId != bytes32(0), "Song not registered");

        Subscription storage sub = subscriptions[artistId][listener];
        require(sub.expiryTime > block.timestamp, "Subscription expired");

        ArtistConfig storage config = artistConfigs[artistId];
        SubscriptionTier storage tier = config.tiers[sub.tierIndex];

        // Reset monthly counter if new month
        if (block.timestamp > sub.monthStartTime + 30 days) {
            sub.streamsThisMonth = 0;
            sub.monthStartTime = block.timestamp;
        }

        // Check stream limit
        if (tier.maxStreamsPerMonth > 0) {
            require(sub.streamsThisMonth < tier.maxStreamsPerMonth, "Monthly limit reached");
        }

        sub.streamsThisMonth++;

        // Amount should be 0 for subscription streams
        if (amount > 0) {
            // Extra payment goes to artist (tip)
            require(flowToken.transferFrom(router, config.artist, amount), "Transfer failed");
            artistEarnings[artistId] += amount;
        }

        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external pure override returns (uint256) {
        songId;
        return 0; // Free for subscribers
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        bytes32 artistId = songToArtist[songId];
        if (artistId == bytes32(0)) return false;

        Subscription storage sub = subscriptions[artistId][listener];
        return sub.expiryTime > block.timestamp;
    }

    function calculateSplits(bytes32 songId, uint256 amount)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        bytes32 artistId = songToArtist[songId];
        ArtistConfig storage config = artistConfigs[artistId];

        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](1);
        splits[0] = EconomicStrategyRouter.Split({
            recipient: config.artist,
            basisPoints: 10000,
            role: "Artist"
        });

        return splits;
    }

    function getSubscription(bytes32 artistId, address subscriber)
        external view returns (Subscription memory)
    {
        return subscriptions[artistId][subscriber];
    }
}
