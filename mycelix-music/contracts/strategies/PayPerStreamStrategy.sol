// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title Pay Per Stream Strategy
 * @notice Simple pay-per-play model with fixed royalty splits
 * @dev Artists earn FLOW tokens instantly for each stream
 */
contract PayPerStreamStrategy is IEconomicStrategy {

    // ========== TYPES ==========

    struct RoyaltySplit {
        address[] recipients;
        uint256[] basisPoints;  // Must sum to 10000
        string[] roles;
        bool initialized;
    }

    // ========== STATE ==========

    IERC20 public flowToken;
    address public router;

    // Song ID => Royalty splits
    mapping(bytes32 => RoyaltySplit) public royaltySplits;

    // Minimum payment per stream (anti-spam)
    uint256 public constant MIN_PAYMENT = 0.001 ether;  // 0.001 FLOW

    // ========== EVENTS ==========

    event RoyaltySplitSet(bytes32 indexed songId, address[] recipients, uint256[] basisPoints);
    event PaymentDistributed(bytes32 indexed songId, address indexed recipient, uint256 amount, string role);

    // ========== CONSTRUCTOR ==========

    constructor(address _flowToken, address _router) {
        flowToken = IERC20(_flowToken);
        router = _router;
    }

    // ========== MODIFIERS ==========

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    // ========== CONFIGURATION ==========

    /**
     * @notice Artist sets royalty splits for their song
     * @param songId Song identifier
     * @param recipients Array of recipient addresses
     * @param basisPoints Array of basis points (must sum to 10000)
     * @param roles Array of role labels (artist, producer, etc.)
     */
    function setRoyaltySplit(
        bytes32 songId,
        address[] calldata recipients,
        uint256[] calldata basisPoints,
        string[] calldata roles
    ) external {
        require(recipients.length == basisPoints.length, "Length mismatch");
        require(recipients.length == roles.length, "Length mismatch");
        require(!royaltySplits[songId].initialized, "Already configured");

        // Validate basis points sum to 10000 (100%)
        uint256 total = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            total += basisPoints[i];
            require(recipients[i] != address(0), "Invalid recipient");
        }
        require(total == 10000, "Must sum to 100%");

        royaltySplits[songId] = RoyaltySplit({
            recipients: recipients,
            basisPoints: basisPoints,
            roles: roles,
            initialized: true
        });

        emit RoyaltySplitSet(songId, recipients, basisPoints);
    }

    // ========== PAYMENT PROCESSING ==========

    /**
     * @notice Process payment and distribute to recipients
     * @dev Called by router contract
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        EconomicStrategyRouter.PaymentType paymentType
    ) external override onlyRouter {
        require(amount >= MIN_PAYMENT, "Payment too small");
        require(royaltySplits[songId].initialized, "Splits not configured");

        RoyaltySplit memory split = royaltySplits[songId];

        // Distribute payment according to splits
        for (uint256 i = 0; i < split.recipients.length; i++) {
            uint256 recipientAmount = (amount * split.basisPoints[i]) / 10000;

            // Transfer FLOW tokens
            require(
                flowToken.transfer(split.recipients[i], recipientAmount),
                "Transfer failed"
            );

            emit PaymentDistributed(
                songId,
                split.recipients[i],
                recipientAmount,
                split.roles[i]
            );
        }
    }

    /**
     * @notice Calculate how payment would be split (view function)
     */
    function calculateSplits(
        bytes32 songId,
        uint256 amount
    ) external view override returns (EconomicStrategyRouter.Split[] memory) {
        require(royaltySplits[songId].initialized, "Splits not configured");

        RoyaltySplit memory split = royaltySplits[songId];
        EconomicStrategyRouter.Split[] memory result =
            new EconomicStrategyRouter.Split[](split.recipients.length);

        for (uint256 i = 0; i < split.recipients.length; i++) {
            result[i] = EconomicStrategyRouter.Split({
                recipient: split.recipients[i],
                basisPoints: split.basisPoints[i],
                role: split.roles[i]
            });
        }

        return result;
    }

    // ========== VIEW FUNCTIONS ==========

    /**
     * @notice Get royalty split configuration for a song
     */
    function getRoyaltySplit(bytes32 songId) external view returns (
        address[] memory recipients,
        uint256[] memory basisPoints,
        string[] memory roles
    ) {
        RoyaltySplit memory split = royaltySplits[songId];
        return (split.recipients, split.basisPoints, split.roles);
    }
}
