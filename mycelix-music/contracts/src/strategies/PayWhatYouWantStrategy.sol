// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title PayWhatYouWantStrategy
 * @notice Listener chooses the amount - optional minimum, suggested amounts
 * @dev Free to listen with optional payment, artists can set suggested amounts
 */
contract PayWhatYouWantStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct PWYWConfig {
        address artist;
        uint256 minimumPayment; // 0 = truly free
        uint256 suggestedPayment;
        address[] recipients;
        uint256[] basisPoints;
        bool initialized;
    }

    struct SongStats {
        uint256 totalPlays;
        uint256 paidPlays;
        uint256 totalEarnings;
        uint256 averagePayment;
    }

    mapping(bytes32 => PWYWConfig) public configs;
    mapping(bytes32 => SongStats) public stats;

    event PWYWConfigured(bytes32 indexed songId, uint256 minimum, uint256 suggested);
    event PaymentReceived(bytes32 indexed songId, address indexed listener, uint256 amount, bool isPaid);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configurePWYW(
        bytes32 songId,
        address artist,
        uint256 minimumPayment,
        uint256 suggestedPayment,
        address[] calldata recipients,
        uint256[] calldata basisPoints
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(recipients.length == basisPoints.length, "Length mismatch");

        uint256 sum = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            sum += basisPoints[i];
        }
        require(sum == 10000, "Must sum to 100%");

        configs[songId] = PWYWConfig({
            artist: artist,
            minimumPayment: minimumPayment,
            suggestedPayment: suggestedPayment,
            recipients: recipients,
            basisPoints: basisPoints,
            initialized: true
        });

        emit PWYWConfigured(songId, minimumPayment, suggestedPayment);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        PWYWConfig storage config = configs[songId];
        require(config.initialized, "Not configured");
        require(amount >= config.minimumPayment, "Below minimum");

        SongStats storage songStats = stats[songId];
        songStats.totalPlays++;

        if (amount > 0) {
            // Distribute payment
            for (uint256 i = 0; i < config.recipients.length; i++) {
                uint256 share = (amount * config.basisPoints[i]) / 10000;
                require(flowToken.transferFrom(router, config.recipients[i], share), "Transfer failed");
            }

            songStats.paidPlays++;
            songStats.totalEarnings += amount;
            songStats.averagePayment = songStats.totalEarnings / songStats.paidPlays;

            emit PaymentReceived(songId, listener, amount, true);
        } else {
            emit PaymentReceived(songId, listener, 0, false);
        }

        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        return configs[songId].minimumPayment;
    }

    function isAuthorized(bytes32, address) external pure override returns (bool) {
        return true; // Open to all
    }

    function calculateSplits(bytes32 songId, uint256 amount)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        PWYWConfig storage config = configs[songId];
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

    function getSuggestedPayment(bytes32 songId) external view returns (uint256) {
        return configs[songId].suggestedPayment;
    }

    function getSongStats(bytes32 songId) external view returns (SongStats memory) {
        return stats[songId];
    }
}
