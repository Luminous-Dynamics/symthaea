// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title FreemiumStrategy
 * @notice Free plays with limit, then pay-per-stream - try before you buy model
 * @dev Listeners get N free plays, then pay per stream
 */
contract FreemiumStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    struct FreemiumConfig {
        address artist;
        uint256 freePlays; // Number of free plays per listener
        uint256 priceAfterFree; // Price after free plays exhausted
        address[] recipients;
        uint256[] basisPoints;
        bool initialized;
    }

    struct ListenerState {
        uint256 playsUsed;
        uint256 totalPaid;
    }

    mapping(bytes32 => FreemiumConfig) public configs;
    mapping(bytes32 => mapping(address => ListenerState)) public listenerStates;
    mapping(bytes32 => uint256) public totalEarnings;
    mapping(bytes32 => uint256) public totalFreePlays;

    event FreemiumConfigured(bytes32 indexed songId, uint256 freePlays, uint256 priceAfterFree);
    event FreePlayUsed(bytes32 indexed songId, address indexed listener, uint256 remaining);
    event PaidPlay(bytes32 indexed songId, address indexed listener, uint256 amount);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    function configureFreemium(
        bytes32 songId,
        address artist,
        uint256 freePlays,
        uint256 priceAfterFree,
        address[] calldata recipients,
        uint256[] calldata basisPoints
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(freePlays > 0, "Need at least 1 free play");
        require(priceAfterFree > 0, "Price must be > 0");
        require(recipients.length == basisPoints.length, "Length mismatch");

        uint256 sum = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            sum += basisPoints[i];
        }
        require(sum == 10000, "Must sum to 100%");

        configs[songId] = FreemiumConfig({
            artist: artist,
            freePlays: freePlays,
            priceAfterFree: priceAfterFree,
            recipients: recipients,
            basisPoints: basisPoints,
            initialized: true
        });

        emit FreemiumConfigured(songId, freePlays, priceAfterFree);
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        FreemiumConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        ListenerState storage state = listenerStates[songId][listener];

        if (state.playsUsed < config.freePlays) {
            // Free play
            state.playsUsed++;
            totalFreePlays[songId]++;
            emit FreePlayUsed(songId, listener, config.freePlays - state.playsUsed);
        } else {
            // Paid play
            require(amount >= config.priceAfterFree, "Insufficient payment");

            // Distribute to recipients
            for (uint256 i = 0; i < config.recipients.length; i++) {
                uint256 share = (amount * config.basisPoints[i]) / 10000;
                require(flowToken.transferFrom(router, config.recipients[i], share), "Transfer failed");
            }

            state.playsUsed++;
            state.totalPaid += amount;
            totalEarnings[songId] += amount;

            emit PaidPlay(songId, listener, amount);
        }

        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        FreemiumConfig storage config = configs[songId];
        // Return 0 if listener still has free plays (checked in processPayment)
        return config.priceAfterFree;
    }

    function isAuthorized(bytes32, address) external pure override returns (bool) {
        return true; // Anyone can play (free or paid)
    }

    function calculateSplits(bytes32 songId, uint256 amount)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        FreemiumConfig storage config = configs[songId];
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

    function getListenerState(bytes32 songId, address listener)
        external view returns (uint256 playsUsed, uint256 freePlaysRemaining, uint256 totalPaid)
    {
        FreemiumConfig storage config = configs[songId];
        ListenerState storage state = listenerStates[songId][listener];

        playsUsed = state.playsUsed;
        freePlaysRemaining = state.playsUsed < config.freePlays
            ? config.freePlays - state.playsUsed
            : 0;
        totalPaid = state.totalPaid;
    }
}
