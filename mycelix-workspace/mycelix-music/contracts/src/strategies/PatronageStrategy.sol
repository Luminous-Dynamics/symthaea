// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title PatronageStrategy
 * @notice Fans become patrons with recurring support - like Patreon for music
 * @dev Monthly patronage with tiered benefits
 */
contract PatronageStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct PatronTier {
        string name;
        uint256 monthlyAmount;
        bool exclusiveContent; // Can access patron-only songs
        uint256 bonusDownloads; // Free downloads per month
        bool active;
    }

    struct ArtistPatronage {
        address artist;
        PatronTier[] tiers;
        uint256 totalPatrons;
        uint256 totalEarnings;
        bool initialized;
    }

    struct Patron {
        uint256 tierIndex;
        uint256 startTime;
        uint256 lastPaymentTime;
        uint256 totalContributed;
        uint256 downloadsUsed;
    }

    mapping(bytes32 => ArtistPatronage) public artistPatronage; // artistId => patronage
    mapping(bytes32 => mapping(address => Patron)) public patrons; // artistId => patron => info
    mapping(bytes32 => bytes32) public songToArtist; // songId => artistId
    mapping(bytes32 => bool) public patronOnlySongs; // songId => is patron-only

    event PatronageConfigured(bytes32 indexed artistId, address artist);
    event TierAdded(bytes32 indexed artistId, string name, uint256 monthlyAmount);
    event PatronJoined(bytes32 indexed artistId, address indexed patron, uint256 tierIndex);
    event PatronPayment(bytes32 indexed artistId, address indexed patron, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configurePatronage(
        bytes32 artistId,
        address artist,
        string[] calldata tierNames,
        uint256[] calldata monthlyAmounts,
        bool[] calldata exclusiveContent,
        uint256[] calldata bonusDownloads
    ) external {
        require(!artistPatronage[artistId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(tierNames.length == monthlyAmounts.length, "Length mismatch");

        ArtistPatronage storage patronage = artistPatronage[artistId];
        patronage.artist = artist;
        patronage.initialized = true;

        for (uint256 i = 0; i < tierNames.length; i++) {
            patronage.tiers.push(PatronTier({
                name: tierNames[i],
                monthlyAmount: monthlyAmounts[i],
                exclusiveContent: exclusiveContent[i],
                bonusDownloads: bonusDownloads[i],
                active: true
            }));
            emit TierAdded(artistId, tierNames[i], monthlyAmounts[i]);
        }

        emit PatronageConfigured(artistId, artist);
    }

    function registerSong(bytes32 songId, bytes32 artistId, bool patronOnly) external {
        require(artistPatronage[artistId].initialized, "Artist not configured");
        songToArtist[songId] = artistId;
        patronOnlySongs[songId] = patronOnly;
    }

    function becomePatron(bytes32 artistId, uint256 tierIndex) external {
        ArtistPatronage storage patronage = artistPatronage[artistId];
        require(patronage.initialized, "Artist not configured");
        require(tierIndex < patronage.tiers.length, "Invalid tier");
        require(patronage.tiers[tierIndex].active, "Tier not active");

        uint256 amount = patronage.tiers[tierIndex].monthlyAmount;
        require(flowToken.transferFrom(msg.sender, patronage.artist, amount), "Payment failed");

        patrons[artistId][msg.sender] = Patron({
            tierIndex: tierIndex,
            startTime: block.timestamp,
            lastPaymentTime: block.timestamp,
            totalContributed: amount,
            downloadsUsed: 0
        });

        patronage.totalPatrons++;
        patronage.totalEarnings += amount;

        emit PatronJoined(artistId, msg.sender, tierIndex);
        emit PatronPayment(artistId, msg.sender, amount);
    }

    function renewPatronage(bytes32 artistId) external {
        ArtistPatronage storage patronage = artistPatronage[artistId];
        Patron storage patron = patrons[artistId][msg.sender];
        require(patron.startTime > 0, "Not a patron");

        PatronTier storage tier = patronage.tiers[patron.tierIndex];
        require(flowToken.transferFrom(msg.sender, patronage.artist, tier.monthlyAmount), "Payment failed");

        patron.lastPaymentTime = block.timestamp;
        patron.totalContributed += tier.monthlyAmount;
        patron.downloadsUsed = 0; // Reset monthly downloads

        patronage.totalEarnings += tier.monthlyAmount;
        emit PatronPayment(artistId, msg.sender, tier.monthlyAmount);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        bytes32 artistId = songToArtist[songId];
        require(artistId != bytes32(0), "Song not registered");

        bool isPatronOnly = patronOnlySongs[songId];
        Patron storage patron = patrons[artistId][listener];

        if (isPatronOnly) {
            // Must be active patron with exclusive content access
            require(patron.startTime > 0, "Patron access required");
            require(
                patron.lastPaymentTime + 35 days > block.timestamp,
                "Patronage expired"
            );

            ArtistPatronage storage patronage = artistPatronage[artistId];
            require(
                patronage.tiers[patron.tierIndex].exclusiveContent,
                "Tier doesn't include exclusive content"
            );
        }

        // Extra payment (tip) goes to artist
        if (amount > 0) {
            ArtistPatronage storage patronage = artistPatronage[artistId];
            require(flowToken.transferFrom(router, patronage.artist, amount), "Transfer failed");
            patronage.totalEarnings += amount;
        }

        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        songId;
        return 0; // Free for patrons
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        bytes32 artistId = songToArtist[songId];
        if (artistId == bytes32(0)) return false;

        // Public songs always authorized
        if (!patronOnlySongs[songId]) return true;

        // Check patron status
        Patron storage patron = patrons[artistId][listener];
        if (patron.startTime == 0) return false;
        if (patron.lastPaymentTime + 35 days <= block.timestamp) return false;

        ArtistPatronage storage patronage = artistPatronage[artistId];
        return patronage.tiers[patron.tierIndex].exclusiveContent;
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        bytes32 artistId = songToArtist[songId];
        ArtistPatronage storage patronage = artistPatronage[artistId];

        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](1);
        splits[0] = EconomicStrategyRouter.Split({
            recipient: patronage.artist,
            basisPoints: 10000,
            role: "Artist"
        });

        return splits;
    }

    function getPatronInfo(bytes32 artistId, address patronAddr)
        external view returns (Patron memory, bool isActive)
    {
        Patron storage patron = patrons[artistId][patronAddr];
        isActive = patron.startTime > 0 && patron.lastPaymentTime + 35 days > block.timestamp;
        return (patron, isActive);
    }

    function getArtistPatronage(bytes32 artistId)
        external view returns (address artist, uint256 totalPatrons, uint256 totalEarnings)
    {
        ArtistPatronage storage patronage = artistPatronage[artistId];
        return (patronage.artist, patronage.totalPatrons, patronage.totalEarnings);
    }
}
