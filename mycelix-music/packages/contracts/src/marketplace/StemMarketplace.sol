// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC1155/IERC1155.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title StemMarketplace
 * @notice Marketplace for buying, selling, and licensing music stems
 * @dev Supports fixed price, auctions, and offers
 *
 * Features:
 * - List stems for sale (fixed price or auction)
 * - Buy stems outright or license for specific uses
 * - Make/accept offers on unlisted stems
 * - Royalty distribution on resales
 * - Escrow for safe transactions
 */
contract StemMarketplace is AccessControl, ReentrancyGuard {
    using ECDSA for bytes32;

    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    enum ListingType {
        FIXED_PRICE,
        AUCTION,
        LICENSE_ONLY
    }

    enum ListingStatus {
        ACTIVE,
        SOLD,
        CANCELLED,
        EXPIRED
    }

    struct Listing {
        uint256 id;
        address seller;
        address nftContract;
        uint256 tokenId;
        uint256 amount;          // For ERC1155
        ListingType listingType;
        ListingStatus status;
        uint256 price;           // Fixed price or starting bid
        uint256 minBidIncrement;
        uint256 reservePrice;    // For auctions
        uint256 startTime;
        uint256 endTime;
        address currency;        // address(0) for ETH
        address highestBidder;
        uint256 highestBid;
        string metadataUri;      // IPFS URI with stem details
    }

    struct Offer {
        uint256 id;
        address buyer;
        address nftContract;
        uint256 tokenId;
        uint256 amount;
        uint256 price;
        address currency;
        uint256 expiration;
        bool isActive;
    }

    // Listing ID => Listing
    mapping(uint256 => Listing) public listings;
    uint256 public listingCount;

    // Offer ID => Offer
    mapping(uint256 => Offer) public offers;
    uint256 public offerCount;

    // NFT Contract => Token ID => Active Listing ID
    mapping(address => mapping(uint256 => uint256)) public activeListings;

    // Escrow: Offer ID => Escrowed amount
    mapping(uint256 => uint256) public escrow;

    // Platform fee (basis points)
    uint256 public platformFee = 250; // 2.5%
    address public platformFeeRecipient;

    // Royalty info: NFT Contract => Royalty recipient => Royalty BPS
    mapping(address => address) public royaltyRecipients;
    mapping(address => uint256) public royaltyBps;

    // ============================================================================
    // Events
    // ============================================================================

    event ListingCreated(
        uint256 indexed listingId,
        address indexed seller,
        address nftContract,
        uint256 tokenId,
        uint256 price,
        ListingType listingType
    );

    event ListingCancelled(uint256 indexed listingId);

    event ListingSold(
        uint256 indexed listingId,
        address indexed buyer,
        uint256 price
    );

    event BidPlaced(
        uint256 indexed listingId,
        address indexed bidder,
        uint256 bid
    );

    event OfferCreated(
        uint256 indexed offerId,
        address indexed buyer,
        address nftContract,
        uint256 tokenId,
        uint256 price
    );

    event OfferAccepted(
        uint256 indexed offerId,
        address indexed seller
    );

    event OfferCancelled(uint256 indexed offerId);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(address _platformFeeRecipient) {
        platformFeeRecipient = _platformFeeRecipient;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
    }

    // ============================================================================
    // Listing Management
    // ============================================================================

    /**
     * @notice Create a fixed-price listing
     */
    function createListing(
        address nftContract,
        uint256 tokenId,
        uint256 amount,
        uint256 price,
        address currency,
        uint256 duration,
        string calldata metadataUri
    ) external returns (uint256) {
        require(price > 0, "Price must be > 0");
        require(duration > 0, "Invalid duration");

        // Transfer NFT to marketplace
        IERC1155(nftContract).safeTransferFrom(
            msg.sender,
            address(this),
            tokenId,
            amount,
            ""
        );

        uint256 listingId = ++listingCount;

        listings[listingId] = Listing({
            id: listingId,
            seller: msg.sender,
            nftContract: nftContract,
            tokenId: tokenId,
            amount: amount,
            listingType: ListingType.FIXED_PRICE,
            status: ListingStatus.ACTIVE,
            price: price,
            minBidIncrement: 0,
            reservePrice: 0,
            startTime: block.timestamp,
            endTime: block.timestamp + duration,
            currency: currency,
            highestBidder: address(0),
            highestBid: 0,
            metadataUri: metadataUri
        });

        activeListings[nftContract][tokenId] = listingId;

        emit ListingCreated(listingId, msg.sender, nftContract, tokenId, price, ListingType.FIXED_PRICE);

        return listingId;
    }

    /**
     * @notice Create an auction listing
     */
    function createAuction(
        address nftContract,
        uint256 tokenId,
        uint256 amount,
        uint256 startingPrice,
        uint256 reservePrice,
        uint256 minBidIncrement,
        address currency,
        uint256 duration,
        string calldata metadataUri
    ) external returns (uint256) {
        require(startingPrice > 0, "Starting price must be > 0");
        require(duration >= 1 hours, "Min duration 1 hour");
        require(minBidIncrement > 0, "Invalid bid increment");

        IERC1155(nftContract).safeTransferFrom(
            msg.sender,
            address(this),
            tokenId,
            amount,
            ""
        );

        uint256 listingId = ++listingCount;

        listings[listingId] = Listing({
            id: listingId,
            seller: msg.sender,
            nftContract: nftContract,
            tokenId: tokenId,
            amount: amount,
            listingType: ListingType.AUCTION,
            status: ListingStatus.ACTIVE,
            price: startingPrice,
            minBidIncrement: minBidIncrement,
            reservePrice: reservePrice,
            startTime: block.timestamp,
            endTime: block.timestamp + duration,
            currency: currency,
            highestBidder: address(0),
            highestBid: 0,
            metadataUri: metadataUri
        });

        activeListings[nftContract][tokenId] = listingId;

        emit ListingCreated(listingId, msg.sender, nftContract, tokenId, startingPrice, ListingType.AUCTION);

        return listingId;
    }

    /**
     * @notice Cancel a listing
     */
    function cancelListing(uint256 listingId) external nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not seller");
        require(listing.status == ListingStatus.ACTIVE, "Not active");
        require(listing.highestBidder == address(0), "Has bids");

        listing.status = ListingStatus.CANCELLED;
        delete activeListings[listing.nftContract][listing.tokenId];

        // Return NFT
        IERC1155(listing.nftContract).safeTransferFrom(
            address(this),
            msg.sender,
            listing.tokenId,
            listing.amount,
            ""
        );

        emit ListingCancelled(listingId);
    }

    // ============================================================================
    // Buying
    // ============================================================================

    /**
     * @notice Buy a fixed-price listing
     */
    function buy(uint256 listingId) external payable nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.status == ListingStatus.ACTIVE, "Not active");
        require(listing.listingType == ListingType.FIXED_PRICE, "Not fixed price");
        require(block.timestamp <= listing.endTime, "Expired");

        uint256 price = listing.price;

        if (listing.currency == address(0)) {
            require(msg.value >= price, "Insufficient payment");
        } else {
            IERC20(listing.currency).transferFrom(msg.sender, address(this), price);
        }

        listing.status = ListingStatus.SOLD;
        delete activeListings[listing.nftContract][listing.tokenId];

        // Distribute payment
        _distributePayment(listing.nftContract, listing.seller, price, listing.currency);

        // Transfer NFT
        IERC1155(listing.nftContract).safeTransferFrom(
            address(this),
            msg.sender,
            listing.tokenId,
            listing.amount,
            ""
        );

        // Refund excess ETH
        if (listing.currency == address(0) && msg.value > price) {
            (bool success, ) = payable(msg.sender).call{value: msg.value - price}("");
            require(success, "Refund failed");
        }

        emit ListingSold(listingId, msg.sender, price);
    }

    // ============================================================================
    // Auction
    // ============================================================================

    /**
     * @notice Place a bid on an auction
     */
    function bid(uint256 listingId) external payable nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.status == ListingStatus.ACTIVE, "Not active");
        require(listing.listingType == ListingType.AUCTION, "Not auction");
        require(block.timestamp <= listing.endTime, "Auction ended");
        require(listing.currency == address(0), "ETH only for now");

        uint256 minBid = listing.highestBid == 0
            ? listing.price
            : listing.highestBid + listing.minBidIncrement;

        require(msg.value >= minBid, "Bid too low");

        // Refund previous bidder
        if (listing.highestBidder != address(0)) {
            (bool success, ) = payable(listing.highestBidder).call{value: listing.highestBid}("");
            require(success, "Refund failed");
        }

        listing.highestBidder = msg.sender;
        listing.highestBid = msg.value;

        // Extend auction if bid in last 10 minutes
        if (listing.endTime - block.timestamp < 10 minutes) {
            listing.endTime = block.timestamp + 10 minutes;
        }

        emit BidPlaced(listingId, msg.sender, msg.value);
    }

    /**
     * @notice Settle an auction after it ends
     */
    function settleAuction(uint256 listingId) external nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.status == ListingStatus.ACTIVE, "Not active");
        require(listing.listingType == ListingType.AUCTION, "Not auction");
        require(block.timestamp > listing.endTime, "Auction not ended");

        listing.status = ListingStatus.SOLD;
        delete activeListings[listing.nftContract][listing.tokenId];

        if (listing.highestBidder != address(0) && listing.highestBid >= listing.reservePrice) {
            // Distribute payment
            _distributePayment(listing.nftContract, listing.seller, listing.highestBid, listing.currency);

            // Transfer NFT to winner
            IERC1155(listing.nftContract).safeTransferFrom(
                address(this),
                listing.highestBidder,
                listing.tokenId,
                listing.amount,
                ""
            );

            emit ListingSold(listingId, listing.highestBidder, listing.highestBid);
        } else {
            // Reserve not met or no bids, return NFT
            IERC1155(listing.nftContract).safeTransferFrom(
                address(this),
                listing.seller,
                listing.tokenId,
                listing.amount,
                ""
            );

            // Refund highest bidder if any
            if (listing.highestBidder != address(0)) {
                (bool success, ) = payable(listing.highestBidder).call{value: listing.highestBid}("");
                require(success, "Refund failed");
            }

            emit ListingCancelled(listingId);
        }
    }

    // ============================================================================
    // Offers
    // ============================================================================

    /**
     * @notice Make an offer on any stem (listed or not)
     */
    function makeOffer(
        address nftContract,
        uint256 tokenId,
        uint256 amount,
        uint256 price,
        address currency,
        uint256 duration
    ) external payable nonReentrant returns (uint256) {
        require(price > 0, "Price must be > 0");
        require(duration >= 1 hours, "Min duration 1 hour");

        // Escrow payment
        if (currency == address(0)) {
            require(msg.value >= price, "Insufficient payment");
            escrow[offerCount + 1] = price;
        } else {
            IERC20(currency).transferFrom(msg.sender, address(this), price);
            escrow[offerCount + 1] = price;
        }

        uint256 offerId = ++offerCount;

        offers[offerId] = Offer({
            id: offerId,
            buyer: msg.sender,
            nftContract: nftContract,
            tokenId: tokenId,
            amount: amount,
            price: price,
            currency: currency,
            expiration: block.timestamp + duration,
            isActive: true
        });

        emit OfferCreated(offerId, msg.sender, nftContract, tokenId, price);

        return offerId;
    }

    /**
     * @notice Accept an offer
     */
    function acceptOffer(uint256 offerId) external nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.isActive, "Offer not active");
        require(block.timestamp <= offer.expiration, "Offer expired");

        // Verify seller owns the token
        require(
            IERC1155(offer.nftContract).balanceOf(msg.sender, offer.tokenId) >= offer.amount,
            "Insufficient balance"
        );

        offer.isActive = false;
        uint256 escrowed = escrow[offerId];
        delete escrow[offerId];

        // Transfer NFT to buyer
        IERC1155(offer.nftContract).safeTransferFrom(
            msg.sender,
            offer.buyer,
            offer.tokenId,
            offer.amount,
            ""
        );

        // Distribute payment from escrow
        _distributePayment(offer.nftContract, msg.sender, escrowed, offer.currency);

        emit OfferAccepted(offerId, msg.sender);
    }

    /**
     * @notice Cancel an offer and get refund
     */
    function cancelOffer(uint256 offerId) external nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.buyer == msg.sender, "Not buyer");
        require(offer.isActive, "Not active");

        offer.isActive = false;
        uint256 escrowed = escrow[offerId];
        delete escrow[offerId];

        // Refund
        if (offer.currency == address(0)) {
            (bool success, ) = payable(msg.sender).call{value: escrowed}("");
            require(success, "Refund failed");
        } else {
            IERC20(offer.currency).transfer(msg.sender, escrowed);
        }

        emit OfferCancelled(offerId);
    }

    // ============================================================================
    // Payment Distribution
    // ============================================================================

    function _distributePayment(
        address nftContract,
        address seller,
        uint256 amount,
        address currency
    ) internal {
        // Calculate fees
        uint256 platformCut = (amount * platformFee) / 10000;
        uint256 royaltyCut = 0;

        if (royaltyRecipients[nftContract] != address(0)) {
            royaltyCut = (amount * royaltyBps[nftContract]) / 10000;
        }

        uint256 sellerAmount = amount - platformCut - royaltyCut;

        if (currency == address(0)) {
            // ETH payments
            if (platformCut > 0) {
                (bool success1, ) = payable(platformFeeRecipient).call{value: platformCut}("");
                require(success1, "Platform fee failed");
            }

            if (royaltyCut > 0) {
                (bool success2, ) = payable(royaltyRecipients[nftContract]).call{value: royaltyCut}("");
                require(success2, "Royalty failed");
            }

            (bool success3, ) = payable(seller).call{value: sellerAmount}("");
            require(success3, "Seller payment failed");
        } else {
            // ERC20 payments
            if (platformCut > 0) {
                IERC20(currency).transfer(platformFeeRecipient, platformCut);
            }

            if (royaltyCut > 0) {
                IERC20(currency).transfer(royaltyRecipients[nftContract], royaltyCut);
            }

            IERC20(currency).transfer(seller, sellerAmount);
        }
    }

    // ============================================================================
    // Admin
    // ============================================================================

    function setPlatformFee(uint256 feeBps) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(feeBps <= 1000, "Max 10%");
        platformFee = feeBps;
    }

    function setPlatformFeeRecipient(address recipient) external onlyRole(DEFAULT_ADMIN_ROLE) {
        platformFeeRecipient = recipient;
    }

    function setRoyaltyInfo(
        address nftContract,
        address recipient,
        uint256 bps
    ) external onlyRole(OPERATOR_ROLE) {
        require(bps <= 2500, "Max 25%");
        royaltyRecipients[nftContract] = recipient;
        royaltyBps[nftContract] = bps;
    }

    // ============================================================================
    // ERC1155 Receiver
    // ============================================================================

    function onERC1155Received(
        address,
        address,
        uint256,
        uint256,
        bytes calldata
    ) external pure returns (bytes4) {
        return this.onERC1155Received.selector;
    }

    function onERC1155BatchReceived(
        address,
        address,
        uint256[] calldata,
        uint256[] calldata,
        bytes calldata
    ) external pure returns (bytes4) {
        return this.onERC1155BatchReceived.selector;
    }
}
