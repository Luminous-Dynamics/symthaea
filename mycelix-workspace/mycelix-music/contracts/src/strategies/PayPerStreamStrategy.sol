// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title PayPerStreamStrategy
 * @notice Traditional pay-per-stream model: $0.01 per stream with instant splits
 * @dev Artists configure royalty splits (e.g., 4 band members + producer)
 */
contract PayPerStreamStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    address public immutable router;

    // Minimum payment: 0.01 FLOW (assuming 1 FLOW = $1)
    uint256 public constant MIN_PAYMENT = 0.01 ether;

    // Royalty split configuration per song
    struct RoyaltySplit {
        address[] recipients;
        uint256[] basisPoints; // Must sum to 10000 (100%)
        string[] roles; // e.g., ["Lead Singer", "Guitarist", "Producer"]
        bool initialized;
    }

    // Mapping: songId => royalty split configuration
    mapping(bytes32 => RoyaltySplit) public royaltySplits;

    // Payment history tracking
    struct PaymentRecord {
        address listener;
        uint256 amount;
        uint256 timestamp;
        PaymentType paymentType;
    }

    mapping(bytes32 => PaymentRecord[]) public paymentHistory;
    mapping(bytes32 => uint256) public totalEarnings;

    // Events
    event RoyaltySplitConfigured(bytes32 indexed songId, address[] recipients, uint256[] basisPoints);
    event PaymentDistributed(
        bytes32 indexed songId,
        address indexed listener,
        uint256 amount,
        address[] recipients,
        uint256[] amounts
    );

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
     * @notice Configure royalty split for a song
     * @param songId Unique identifier for the song
     * @param recipients Addresses to receive royalties
     * @param basisPoints Percentage in basis points (100 = 1%)
     * @param roles Descriptions of each recipient's role
     */
    function configureRoyaltySplit(
        bytes32 songId,
        address[] calldata recipients,
        uint256[] calldata basisPoints,
        string[] calldata roles
    ) external {
        require(!royaltySplits[songId].initialized, "Already configured");
        require(recipients.length > 0, "No recipients");
        require(recipients.length == basisPoints.length, "Length mismatch");
        require(recipients.length == roles.length, "Role length mismatch");

        // Verify sum is 10000 (100%)
        uint256 sum = 0;
        for (uint256 i = 0; i < basisPoints.length; i++) {
            require(recipients[i] != address(0), "Invalid recipient");
            require(basisPoints[i] > 0, "Zero basis points");
            sum += basisPoints[i];
        }
        require(sum == 10000, "Must sum to 100%");

        royaltySplits[songId] = RoyaltySplit({
            recipients: recipients,
            basisPoints: basisPoints,
            roles: roles,
            initialized: true
        });

        emit RoyaltySplitConfigured(songId, recipients, basisPoints);
    }

    /**
     * @notice Process a payment according to pay-per-stream rules
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @param amount Amount of FLOW tokens (after protocol fee)
     * @param paymentType Type of payment
     */
    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        require(amount >= MIN_PAYMENT, "Payment too small");

        RoyaltySplit memory split = royaltySplits[songId];
        require(split.initialized, "Royalty split not configured");

        // Calculate and distribute royalties
        uint256[] memory recipientAmounts = new uint256[](split.recipients.length);

        for (uint256 i = 0; i < split.recipients.length; i++) {
            uint256 recipientAmount = (amount * split.basisPoints[i]) / 10000;
            recipientAmounts[i] = recipientAmount;

            // Transfer from router (which holds the tokens)
            require(
                flowToken.transferFrom(router, split.recipients[i], recipientAmount),
                "Transfer failed"
            );
        }

        // Record payment
        paymentHistory[songId].push(
            PaymentRecord({
                listener: listener,
                amount: amount,
                timestamp: block.timestamp,
                paymentType: paymentType
            })
        );

        totalEarnings[songId] += amount;

        emit PaymentDistributed(songId, listener, amount, split.recipients, recipientAmounts);
    }

    /**
     * @notice Get minimum payment amount
     * @param songId Unique identifier for the song
     * @param paymentType Type of payment
     * @return Minimum payment amount
     */
    function getMinPayment(bytes32 songId, PaymentType paymentType)
        external
        pure
        override
        returns (uint256)
    {
        // Unused parameters
        songId;
        paymentType;

        return MIN_PAYMENT;
    }

    /**
     * @notice Check if listener is authorized (always true for pay-per-stream)
     * @param songId Unique identifier for the song
     * @param listener Address of the listener
     * @return Always true (anyone can pay and listen)
     */
    function isAuthorized(bytes32 songId, address listener) external pure override returns (bool) {
        // Unused parameters
        songId;
        listener;

        return true; // Anyone can pay and stream
    }

    /**
     * @notice Get royalty split configuration
     * @param songId Unique identifier for the song
     * @return recipients Array of recipient addresses
     * @return basisPoints Array of basis points
     * @return roles Array of role descriptions
     */
    function getRoyaltySplit(bytes32 songId)
        external
        view
        returns (
            address[] memory recipients,
            uint256[] memory basisPoints,
            string[] memory roles
        )
    {
        RoyaltySplit memory split = royaltySplits[songId];
        return (split.recipients, split.basisPoints, split.roles);
    }

    /**
     * @notice Get payment history for a song
     * @param songId Unique identifier for the song
     * @return Array of payment records
     */
    function getPaymentHistory(bytes32 songId) external view returns (PaymentRecord[] memory) {
        return paymentHistory[songId];
    }

    /**
     * @notice Get total earnings for a song
     * @param songId Unique identifier for the song
     * @return Total amount earned
     */
    function getTotalEarnings(bytes32 songId) external view returns (uint256) {
        return totalEarnings[songId];
    }
}
