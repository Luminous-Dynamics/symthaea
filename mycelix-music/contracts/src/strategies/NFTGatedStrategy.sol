// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title NFTGatedStrategy
 * @notice NFT holders get exclusive access to music
 * @dev Listeners must hold specific NFT to stream, optional payment for non-holders
 */
contract NFTGatedStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct NFTGateConfig {
        address artist;
        address nftContract;
        uint256 requiredTokenId; // 0 = any token from collection
        bool anyTokenAllowed;
        uint256 nonHolderPrice; // Price for non-holders (0 = not allowed)
        address[] recipients;
        uint256[] basisPoints;
        bool initialized;
    }

    mapping(bytes32 => NFTGateConfig) public configs;
    mapping(bytes32 => uint256) public totalEarnings;
    mapping(bytes32 => uint256) public holderPlays;
    mapping(bytes32 => uint256) public paidPlays;

    event NFTGateConfigured(bytes32 indexed songId, address nftContract, uint256 nonHolderPrice);
    event HolderAccess(bytes32 indexed songId, address indexed listener, uint256 tokenId);
    event NonHolderPurchase(bytes32 indexed songId, address indexed listener, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configureNFTGate(
        bytes32 songId,
        address artist,
        address nftContract,
        uint256 requiredTokenId,
        bool anyTokenAllowed,
        uint256 nonHolderPrice,
        address[] calldata recipients,
        uint256[] calldata basisPoints
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(nftContract != address(0), "Invalid NFT contract");
        require(recipients.length == basisPoints.length, "Length mismatch");

        uint256 sum = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            sum += basisPoints[i];
        }
        require(sum == 10000, "Must sum to 100%");

        configs[songId] = NFTGateConfig({
            artist: artist,
            nftContract: nftContract,
            requiredTokenId: requiredTokenId,
            anyTokenAllowed: anyTokenAllowed,
            nonHolderPrice: nonHolderPrice,
            recipients: recipients,
            basisPoints: basisPoints,
            initialized: true
        });

        emit NFTGateConfigured(songId, nftContract, nonHolderPrice);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        NFTGateConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        bool isHolder = _checkNFTHolder(config, listener);

        if (isHolder) {
            // Free access for holders
            holderPlays[songId]++;
            emit HolderAccess(songId, listener, 0);
        } else {
            // Non-holder must pay
            require(config.nonHolderPrice > 0, "NFT required");
            require(amount >= config.nonHolderPrice, "Insufficient payment");

            // Distribute payment
            for (uint256 i = 0; i < config.recipients.length; i++) {
                uint256 share = (amount * config.basisPoints[i]) / 10000;
                require(flowToken.transferFrom(router, config.recipients[i], share), "Transfer failed");
            }

            paidPlays[songId]++;
            totalEarnings[songId] += amount;
            emit NonHolderPurchase(songId, listener, amount);
        }

        paymentType; // Silence unused warning
    }

    function _checkNFTHolder(NFTGateConfig storage config, address listener) internal view returns (bool) {
        IERC721 nft = IERC721(config.nftContract);

        if (config.anyTokenAllowed) {
            return nft.balanceOf(listener) > 0;
        } else {
            try nft.ownerOf(config.requiredTokenId) returns (address owner) {
                return owner == listener;
            } catch {
                return false;
            }
        }
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        NFTGateConfig storage config = configs[songId];
        // Return 0 if holder, otherwise return non-holder price
        return config.nonHolderPrice;
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        NFTGateConfig storage config = configs[songId];
        if (!config.initialized) return false;

        // Holders always authorized
        if (_checkNFTHolder(config, listener)) return true;

        // Non-holders authorized if payment option exists
        return config.nonHolderPrice > 0;
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        NFTGateConfig storage config = configs[songId];
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

    function checkNFTAccess(bytes32 songId, address listener) external view returns (bool isHolder, uint256 price) {
        NFTGateConfig storage config = configs[songId];
        isHolder = _checkNFTHolder(config, listener);
        price = isHolder ? 0 : config.nonHolderPrice;
    }
}
