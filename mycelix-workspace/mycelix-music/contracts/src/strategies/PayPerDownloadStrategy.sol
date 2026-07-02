// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title PayPerDownloadStrategy
 * @notice Classic digital purchase model - pay once, own forever
 * @dev One-time purchase grants permanent download rights
 */
contract PayPerDownloadStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct DownloadConfig {
        address artist;
        uint256 downloadPrice;
        uint256 streamPrice; // Optional streaming price (can be 0 for preview)
        uint256 bundlePrice; // Album/collection price
        address[] recipients;
        uint256[] basisPoints;
        bool initialized;
    }

    struct PurchaseRecord {
        bool hasDownloadRights;
        uint256 purchaseTime;
        uint256 totalPaid;
    }

    mapping(bytes32 => DownloadConfig) public configs;
    mapping(bytes32 => mapping(address => PurchaseRecord)) public purchases;
    mapping(bytes32 => uint256) public totalDownloads;
    mapping(bytes32 => uint256) public totalEarnings;

    event DownloadConfigured(bytes32 indexed songId, uint256 downloadPrice, uint256 streamPrice);
    event DownloadPurchased(bytes32 indexed songId, address indexed buyer, uint256 amount);
    event StreamPurchased(bytes32 indexed songId, address indexed listener, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configureDownload(
        bytes32 songId,
        address artist,
        uint256 downloadPrice,
        uint256 streamPrice,
        address[] calldata recipients,
        uint256[] calldata basisPoints
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(downloadPrice > 0, "Download price must be > 0");
        require(recipients.length == basisPoints.length, "Length mismatch");

        uint256 sum = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            sum += basisPoints[i];
        }
        require(sum == 10000, "Must sum to 100%");

        configs[songId] = DownloadConfig({
            artist: artist,
            downloadPrice: downloadPrice,
            streamPrice: streamPrice,
            bundlePrice: 0,
            recipients: recipients,
            basisPoints: basisPoints,
            initialized: true
        });

        emit DownloadConfigured(songId, downloadPrice, streamPrice);
    }

    function purchaseDownload(bytes32 songId) external {
        DownloadConfig storage config = configs[songId];
        require(config.initialized, "Not configured");
        require(!purchases[songId][msg.sender].hasDownloadRights, "Already purchased");

        require(flowToken.transferFrom(msg.sender, address(this), config.downloadPrice), "Payment failed");

        // Distribute payment
        for (uint256 i = 0; i < config.recipients.length; i++) {
            uint256 share = (config.downloadPrice * config.basisPoints[i]) / 10000;
            require(flowToken.transfer(config.recipients[i], share), "Transfer failed");
        }

        purchases[songId][msg.sender] = PurchaseRecord({
            hasDownloadRights: true,
            purchaseTime: block.timestamp,
            totalPaid: config.downloadPrice
        });

        totalDownloads[songId]++;
        totalEarnings[songId] += config.downloadPrice;

        emit DownloadPurchased(songId, msg.sender, config.downloadPrice);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        DownloadConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        PurchaseRecord storage purchase = purchases[songId][listener];

        if (paymentType == PaymentType.DOWNLOAD) {
            // Full download purchase through router
            require(amount >= config.downloadPrice, "Insufficient payment");
            require(!purchase.hasDownloadRights, "Already purchased");

            for (uint256 i = 0; i < config.recipients.length; i++) {
                uint256 share = (amount * config.basisPoints[i]) / 10000;
                require(flowToken.transferFrom(router, config.recipients[i], share), "Transfer failed");
            }

            purchase.hasDownloadRights = true;
            purchase.purchaseTime = block.timestamp;
            purchase.totalPaid = amount;

            totalDownloads[songId]++;
            totalEarnings[songId] += amount;

            emit DownloadPurchased(songId, listener, amount);
        } else {
            // Stream (preview or paid stream)
            if (purchase.hasDownloadRights) {
                // Free streaming for owners
                return;
            }

            // Pay for stream preview
            if (config.streamPrice > 0) {
                require(amount >= config.streamPrice, "Insufficient payment");

                for (uint256 i = 0; i < config.recipients.length; i++) {
                    uint256 share = (amount * config.basisPoints[i]) / 10000;
                    require(flowToken.transferFrom(router, config.recipients[i], share), "Transfer failed");
                }

                totalEarnings[songId] += amount;
                emit StreamPurchased(songId, listener, amount);
            }
        }
    }

    function getMinPayment(bytes32 songId, PaymentType paymentType) external view override returns (uint256) {
        DownloadConfig storage config = configs[songId];

        if (paymentType == PaymentType.DOWNLOAD) {
            return config.downloadPrice;
        }

        return config.streamPrice;
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        DownloadConfig storage config = configs[songId];
        if (!config.initialized) return false;

        // Owners always authorized
        if (purchases[songId][listener].hasDownloadRights) return true;

        // Non-owners can preview if stream price exists
        return config.streamPrice > 0 || config.streamPrice == 0;
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        DownloadConfig storage config = configs[songId];
        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](config.recipients.length);

        for (uint256 i = 0; i < config.recipients.length; i++) {
            splits[i] = EconomicStrategyRouter.Split({
                recipient: config.recipients[i],
                basisPoints: config.basisPoints[i],
                role: "Recipient"
            });
        }

        return splits;
    }

    function hasPurchased(bytes32 songId, address buyer) external view returns (bool) {
        return purchases[songId][buyer].hasDownloadRights;
    }

    function getPurchaseInfo(bytes32 songId, address buyer) external view returns (PurchaseRecord memory) {
        return purchases[songId][buyer];
    }
}
