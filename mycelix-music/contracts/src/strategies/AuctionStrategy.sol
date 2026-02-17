// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title AuctionStrategy
 * @notice Auction model for exclusive or limited releases
 * @dev Artists can auction exclusive access to tracks or albums
 */
contract AuctionStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct Auction {
        address artist;
        uint256 startTime;
        uint256 endTime;
        uint256 startingPrice;
        uint256 reservePrice;
        uint256 buyNowPrice; // 0 = no buy now option
        address highestBidder;
        uint256 highestBid;
        uint256 totalBids;
        uint256 accessSlots; // Number of winners (1 = single winner, N = N winners)
        bool finalized;
        address[] winners;
    }

    struct Bid {
        address bidder;
        uint256 amount;
        uint256 timestamp;
    }

    mapping(bytes32 => Auction) public auctions;
    mapping(bytes32 => Bid[]) public bids;
    mapping(bytes32 => mapping(address => uint256)) public userBids;
    mapping(bytes32 => mapping(address => bool)) public hasAccess;

    event AuctionCreated(bytes32 indexed songId, uint256 startTime, uint256 endTime, uint256 startingPrice);
    event BidPlaced(bytes32 indexed songId, address indexed bidder, uint256 amount);
    event BidOutbid(bytes32 indexed songId, address indexed previousBidder, uint256 refundAmount);
    event AuctionFinalized(bytes32 indexed songId, address[] winners, uint256 totalRaised);
    event BuyNowPurchased(bytes32 indexed songId, address indexed buyer, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function createAuction(
        bytes32 songId,
        address artist,
        uint256 startTime,
        uint256 duration,
        uint256 startingPrice,
        uint256 reservePrice,
        uint256 buyNowPrice,
        uint256 accessSlots
    ) external {
        require(auctions[songId].artist == address(0), "Auction exists");
        require(artist != address(0), "Invalid artist");
        require(startTime >= block.timestamp, "Invalid start time");
        require(duration >= 1 hours, "Duration too short");
        require(accessSlots > 0, "Need at least 1 slot");

        auctions[songId] = Auction({
            artist: artist,
            startTime: startTime,
            endTime: startTime + duration,
            startingPrice: startingPrice,
            reservePrice: reservePrice,
            buyNowPrice: buyNowPrice,
            highestBidder: address(0),
            highestBid: 0,
            totalBids: 0,
            accessSlots: accessSlots,
            finalized: false,
            winners: new address[](0)
        });

        emit AuctionCreated(songId, startTime, startTime + duration, startingPrice);
    }

    function placeBid(bytes32 songId, uint256 amount) external {
        Auction storage auction = auctions[songId];
        require(auction.artist != address(0), "Auction not found");
        require(block.timestamp >= auction.startTime, "Auction not started");
        require(block.timestamp < auction.endTime, "Auction ended");
        require(!auction.finalized, "Auction finalized");
        require(amount >= auction.startingPrice, "Below starting price");
        require(amount > auction.highestBid, "Bid too low");

        // Refund previous highest bidder
        if (auction.highestBidder != address(0)) {
            require(flowToken.transfer(auction.highestBidder, auction.highestBid), "Refund failed");
            emit BidOutbid(songId, auction.highestBidder, auction.highestBid);
        }

        // Take new bid
        require(flowToken.transferFrom(msg.sender, address(this), amount), "Payment failed");

        auction.highestBidder = msg.sender;
        auction.highestBid = amount;
        auction.totalBids++;

        userBids[songId][msg.sender] = amount;

        bids[songId].push(Bid({
            bidder: msg.sender,
            amount: amount,
            timestamp: block.timestamp
        }));

        emit BidPlaced(songId, msg.sender, amount);

        // Extend auction if bid in last 10 minutes
        if (auction.endTime - block.timestamp < 10 minutes) {
            auction.endTime += 10 minutes;
        }
    }

    function buyNow(bytes32 songId) external {
        Auction storage auction = auctions[songId];
        require(auction.artist != address(0), "Auction not found");
        require(auction.buyNowPrice > 0, "No buy now option");
        require(block.timestamp >= auction.startTime, "Auction not started");
        require(block.timestamp < auction.endTime, "Auction ended");
        require(!auction.finalized, "Auction finalized");

        require(flowToken.transferFrom(msg.sender, auction.artist, auction.buyNowPrice), "Payment failed");

        hasAccess[songId][msg.sender] = true;

        emit BuyNowPurchased(songId, msg.sender, auction.buyNowPrice);

        // If single slot auction, end it
        if (auction.accessSlots == 1) {
            auction.finalized = true;
            auction.winners.push(msg.sender);
        }
    }

    function finalizeAuction(bytes32 songId) external {
        Auction storage auction = auctions[songId];
        require(auction.artist != address(0), "Auction not found");
        require(block.timestamp >= auction.endTime, "Auction not ended");
        require(!auction.finalized, "Already finalized");

        auction.finalized = true;

        if (auction.highestBid >= auction.reservePrice && auction.highestBidder != address(0)) {
            // Transfer funds to artist
            require(flowToken.transfer(auction.artist, auction.highestBid), "Transfer failed");

            // Grant access to winner
            auction.winners.push(auction.highestBidder);
            hasAccess[songId][auction.highestBidder] = true;

            emit AuctionFinalized(songId, auction.winners, auction.highestBid);
        } else {
            // Reserve not met, refund highest bidder
            if (auction.highestBidder != address(0)) {
                require(flowToken.transfer(auction.highestBidder, auction.highestBid), "Refund failed");
            }

            emit AuctionFinalized(songId, new address[](0), 0);
        }
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        require(hasAccess[songId][listener], "No auction access");
        // Winners can stream for free
        amount;
        paymentType;
    }

    function getMinPayment(bytes32, PaymentType) external pure override returns (uint256) {
        return 0; // Free for auction winners
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        return hasAccess[songId][listener];
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        Auction storage auction = auctions[songId];

        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](1);
        splits[0] = EconomicStrategyRouter.Split({
            recipient: auction.artist,
            basisPoints: 10000,
            role: "Artist"
        });

        return splits;
    }

    function getAuction(bytes32 songId) external view returns (Auction memory) {
        return auctions[songId];
    }

    function getBids(bytes32 songId) external view returns (Bid[] memory) {
        return bids[songId];
    }

    function getTimeRemaining(bytes32 songId) external view returns (uint256) {
        Auction storage auction = auctions[songId];
        if (block.timestamp >= auction.endTime) return 0;
        return auction.endTime - block.timestamp;
    }
}
