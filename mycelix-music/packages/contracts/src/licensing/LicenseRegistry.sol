// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title LicenseRegistry
 * @notice NFT-based licensing system for music content
 * @dev Each license is an NFT with specific usage rights
 *
 * License Types:
 * - Personal: Non-commercial personal use
 * - Creator: Social media, YouTube, streams
 * - Commercial: Advertising, film, TV
 * - Sync: Film/TV/advertising sync rights
 * - Exclusive: Full exclusive rights
 *
 * Features:
 * - Tiered pricing by license type
 * - Time-limited or perpetual licenses
 * - Geographic restrictions
 * - Usage tracking and verification
 * - Automatic royalty distribution
 */
contract LicenseRegistry is ERC721, ERC721URIStorage, AccessControl, ReentrancyGuard {
    using Counters for Counters.Counter;

    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant LICENSOR_ROLE = keccak256("LICENSOR_ROLE");

    Counters.Counter private _licenseIdCounter;

    enum LicenseType {
        PERSONAL,       // Non-commercial personal use
        CREATOR,        // Social media, YouTube, podcasts
        COMMERCIAL,     // Advertising, products, services
        SYNC,           // Film, TV, advertising sync
        EXCLUSIVE       // Full exclusive rights
    }

    enum Duration {
        ONE_MONTH,
        THREE_MONTHS,
        ONE_YEAR,
        PERPETUAL
    }

    struct LicenseTerms {
        uint256 contentId;           // Track/stem ID being licensed
        address contentContract;     // NFT contract of the content
        LicenseType licenseType;
        Duration duration;
        uint256 expiresAt;           // 0 for perpetual
        string[] territories;         // Empty = worldwide
        uint256 maxStreams;          // 0 = unlimited
        uint256 maxRevenue;          // 0 = unlimited (in cents)
        bool isTransferable;
        bool isExclusive;
        string customTermsUri;       // IPFS hash of custom legal terms
    }

    struct LicenseOffer {
        uint256 id;
        address licensor;
        uint256 contentId;
        address contentContract;
        LicenseType licenseType;
        uint256 price;               // Base price
        Duration[] allowedDurations;
        uint256[] durationPriceMultipliers; // Basis points (10000 = 1x)
        string[] allowedTerritories;
        uint256 maxStreams;
        uint256 maxRevenue;
        bool transferable;
        bool active;
    }

    // License NFT ID => Terms
    mapping(uint256 => LicenseTerms) public licenseTerms;

    // Offer ID => Offer
    mapping(uint256 => LicenseOffer) public licenseOffers;
    uint256 public offerCount;

    // Content ID => License Type => Offer ID
    mapping(uint256 => mapping(LicenseType => uint256)) public contentOffers;

    // Content ID => Has exclusive license
    mapping(uint256 => bool) public hasExclusiveLicense;

    // License ID => Usage metrics
    mapping(uint256 => uint256) public licenseStreams;
    mapping(uint256 => uint256) public licenseRevenue;

    // Platform fee (basis points)
    uint256 public platformFee = 500; // 5%
    address public platformFeeRecipient;

    // ============================================================================
    // Events
    // ============================================================================

    event LicenseOfferCreated(
        uint256 indexed offerId,
        address indexed licensor,
        uint256 contentId,
        LicenseType licenseType,
        uint256 price
    );

    event LicenseOfferUpdated(uint256 indexed offerId);

    event LicenseOfferDeactivated(uint256 indexed offerId);

    event LicensePurchased(
        uint256 indexed licenseId,
        uint256 indexed offerId,
        address indexed licensee,
        uint256 price
    );

    event LicenseUsageReported(
        uint256 indexed licenseId,
        uint256 streams,
        uint256 revenue
    );

    event LicenseExpired(uint256 indexed licenseId);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(address _platformFeeRecipient)
        ERC721("Mycelix Music License", "MXLIC")
    {
        platformFeeRecipient = _platformFeeRecipient;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(LICENSOR_ROLE, msg.sender);
    }

    // ============================================================================
    // License Offers
    // ============================================================================

    /**
     * @notice Create a license offer for content
     */
    function createLicenseOffer(
        uint256 contentId,
        address contentContract,
        LicenseType licenseType,
        uint256 price,
        Duration[] calldata allowedDurations,
        uint256[] calldata durationMultipliers,
        string[] calldata territories,
        uint256 maxStreams,
        uint256 maxRevenue,
        bool transferable
    ) external returns (uint256) {
        require(allowedDurations.length == durationMultipliers.length, "Length mismatch");

        uint256 offerId = ++offerCount;

        licenseOffers[offerId] = LicenseOffer({
            id: offerId,
            licensor: msg.sender,
            contentId: contentId,
            contentContract: contentContract,
            licenseType: licenseType,
            price: price,
            allowedDurations: allowedDurations,
            durationPriceMultipliers: durationMultipliers,
            allowedTerritories: territories,
            maxStreams: maxStreams,
            maxRevenue: maxRevenue,
            transferable: transferable,
            active: true
        });

        contentOffers[contentId][licenseType] = offerId;

        emit LicenseOfferCreated(offerId, msg.sender, contentId, licenseType, price);

        return offerId;
    }

    /**
     * @notice Update license offer price
     */
    function updateOfferPrice(uint256 offerId, uint256 newPrice) external {
        LicenseOffer storage offer = licenseOffers[offerId];
        require(offer.licensor == msg.sender, "Not licensor");
        offer.price = newPrice;
        emit LicenseOfferUpdated(offerId);
    }

    /**
     * @notice Deactivate a license offer
     */
    function deactivateOffer(uint256 offerId) external {
        LicenseOffer storage offer = licenseOffers[offerId];
        require(offer.licensor == msg.sender, "Not licensor");
        offer.active = false;
        emit LicenseOfferDeactivated(offerId);
    }

    // ============================================================================
    // License Purchase
    // ============================================================================

    /**
     * @notice Purchase a license
     */
    function purchaseLicense(
        uint256 offerId,
        Duration duration,
        string[] calldata territories
    ) external payable nonReentrant returns (uint256) {
        LicenseOffer storage offer = licenseOffers[offerId];
        require(offer.active, "Offer not active");

        // Check for exclusive conflict
        if (offer.licenseType == LicenseType.EXCLUSIVE) {
            require(!hasExclusiveLicense[offer.contentId], "Exclusive license exists");
        }

        // Validate duration
        bool validDuration = false;
        uint256 multiplier = 10000;
        for (uint256 i = 0; i < offer.allowedDurations.length; i++) {
            if (offer.allowedDurations[i] == duration) {
                validDuration = true;
                multiplier = offer.durationPriceMultipliers[i];
                break;
            }
        }
        require(validDuration, "Duration not allowed");

        // Validate territories
        if (offer.allowedTerritories.length > 0) {
            for (uint256 i = 0; i < territories.length; i++) {
                bool found = false;
                for (uint256 j = 0; j < offer.allowedTerritories.length; j++) {
                    if (keccak256(bytes(territories[i])) == keccak256(bytes(offer.allowedTerritories[j]))) {
                        found = true;
                        break;
                    }
                }
                require(found, "Territory not allowed");
            }
        }

        // Calculate price
        uint256 price = (offer.price * multiplier) / 10000;
        require(msg.value >= price, "Insufficient payment");

        // Calculate expiration
        uint256 expiresAt = 0;
        if (duration == Duration.ONE_MONTH) {
            expiresAt = block.timestamp + 30 days;
        } else if (duration == Duration.THREE_MONTHS) {
            expiresAt = block.timestamp + 90 days;
        } else if (duration == Duration.ONE_YEAR) {
            expiresAt = block.timestamp + 365 days;
        }
        // PERPETUAL = 0 (never expires)

        // Mint license NFT
        uint256 licenseId = _licenseIdCounter.current();
        _licenseIdCounter.increment();
        _safeMint(msg.sender, licenseId);

        // Store terms
        licenseTerms[licenseId] = LicenseTerms({
            contentId: offer.contentId,
            contentContract: offer.contentContract,
            licenseType: offer.licenseType,
            duration: duration,
            expiresAt: expiresAt,
            territories: territories.length > 0 ? territories : offer.allowedTerritories,
            maxStreams: offer.maxStreams,
            maxRevenue: offer.maxRevenue,
            isTransferable: offer.transferable,
            isExclusive: offer.licenseType == LicenseType.EXCLUSIVE,
            customTermsUri: ""
        });

        // Mark exclusive if applicable
        if (offer.licenseType == LicenseType.EXCLUSIVE) {
            hasExclusiveLicense[offer.contentId] = true;
        }

        // Distribute payment
        uint256 platformCut = (price * platformFee) / 10000;
        uint256 licensorAmount = price - platformCut;

        (bool success1, ) = payable(platformFeeRecipient).call{value: platformCut}("");
        require(success1, "Platform fee failed");

        (bool success2, ) = payable(offer.licensor).call{value: licensorAmount}("");
        require(success2, "Licensor payment failed");

        // Refund excess
        if (msg.value > price) {
            (bool success3, ) = payable(msg.sender).call{value: msg.value - price}("");
            require(success3, "Refund failed");
        }

        emit LicensePurchased(licenseId, offerId, msg.sender, price);

        return licenseId;
    }

    // ============================================================================
    // License Verification
    // ============================================================================

    /**
     * @notice Check if a license is valid
     */
    function isLicenseValid(uint256 licenseId) public view returns (bool) {
        if (!_exists(licenseId)) return false;

        LicenseTerms storage terms = licenseTerms[licenseId];

        // Check expiration
        if (terms.expiresAt > 0 && block.timestamp > terms.expiresAt) {
            return false;
        }

        // Check usage limits
        if (terms.maxStreams > 0 && licenseStreams[licenseId] >= terms.maxStreams) {
            return false;
        }

        if (terms.maxRevenue > 0 && licenseRevenue[licenseId] >= terms.maxRevenue) {
            return false;
        }

        return true;
    }

    /**
     * @notice Check if license covers a territory
     */
    function isLicenseValidForTerritory(
        uint256 licenseId,
        string calldata territory
    ) external view returns (bool) {
        if (!isLicenseValid(licenseId)) return false;

        LicenseTerms storage terms = licenseTerms[licenseId];

        // Empty territories = worldwide
        if (terms.territories.length == 0) return true;

        for (uint256 i = 0; i < terms.territories.length; i++) {
            if (keccak256(bytes(terms.territories[i])) == keccak256(bytes(territory))) {
                return true;
            }
        }

        return false;
    }

    /**
     * @notice Get license terms
     */
    function getLicenseTerms(uint256 licenseId) external view returns (
        uint256 contentId,
        address contentContract,
        LicenseType licenseType,
        Duration duration,
        uint256 expiresAt,
        uint256 maxStreams,
        uint256 maxRevenue,
        bool isTransferable,
        bool isExclusive
    ) {
        LicenseTerms storage terms = licenseTerms[licenseId];
        return (
            terms.contentId,
            terms.contentContract,
            terms.licenseType,
            terms.duration,
            terms.expiresAt,
            terms.maxStreams,
            terms.maxRevenue,
            terms.isTransferable,
            terms.isExclusive
        );
    }

    // ============================================================================
    // Usage Tracking
    // ============================================================================

    /**
     * @notice Report usage metrics (called by authorized oracles)
     */
    function reportUsage(
        uint256 licenseId,
        uint256 additionalStreams,
        uint256 additionalRevenue
    ) external onlyRole(LICENSOR_ROLE) {
        require(_exists(licenseId), "License does not exist");

        licenseStreams[licenseId] += additionalStreams;
        licenseRevenue[licenseId] += additionalRevenue;

        emit LicenseUsageReported(licenseId, additionalStreams, additionalRevenue);
    }

    // ============================================================================
    // Transfer Restrictions
    // ============================================================================

    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);

        // Allow minting
        if (from == address(0)) return;

        // Check if license is transferable
        LicenseTerms storage terms = licenseTerms[tokenId];
        require(terms.isTransferable, "License not transferable");
    }

    // ============================================================================
    // Admin
    // ============================================================================

    function setPlatformFee(uint256 feeBps) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(feeBps <= 2000, "Max 20%");
        platformFee = feeBps;
    }

    function setPlatformFeeRecipient(address recipient) external onlyRole(DEFAULT_ADMIN_ROLE) {
        platformFeeRecipient = recipient;
    }

    // ============================================================================
    // Overrides
    // ============================================================================

    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage, AccessControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }

    function _exists(uint256 tokenId) internal view returns (bool) {
        return _ownerOf(tokenId) != address(0);
    }
}
