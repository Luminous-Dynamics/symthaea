// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Address} from "@openzeppelin/contracts/utils/Address.sol";

/**
 * @title PaymentRouter
 * @author Mycelix Network
 * @notice Routes payments for marketplace and music streaming with escrow and dispute resolution
 * @dev Supports both native tokens (ETH/MATIC) and ERC20 tokens
 *
 * Features:
 * - Split payments across multiple recipients
 * - Escrow functionality with time-based release
 * - Dispute resolution with arbitration hooks
 * - Integration with ReputationAnchor for reputation-weighted payments
 */
contract PaymentRouter is Ownable, AccessControl, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;
    using Address for address payable;

    // ============ Constants ============

    /// @notice Role for dispute arbitrators
    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");

    /// @notice Role for trusted payment initiators (marketplace contracts)
    bytes32 public constant ROUTER_ROLE = keccak256("ROUTER_ROLE");

    /// @notice Role for governance actions
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    /// @notice Maximum number of recipients in a single payment
    uint256 public constant MAX_RECIPIENTS = 100;

    /// @notice Basis points denominator (10000 = 100%)
    uint256 public constant BASIS_POINTS = 10_000;

    /// @notice Maximum platform fee (5%)
    uint256 public constant MAX_PLATFORM_FEE = 500;

    /// @notice Minimum escrow duration
    uint256 public constant MIN_ESCROW_DURATION = 1 hours;

    /// @notice Maximum escrow duration
    uint256 public constant MAX_ESCROW_DURATION = 365 days;

    // ============ Enums ============

    enum EscrowState {
        Active,
        Released,
        Disputed,
        Refunded,
        Resolved
    }

    enum DisputeResolution {
        Pending,
        ReleaseToPayer,
        ReleaseToPayee,
        Split
    }

    // ============ Structs ============

    struct PaymentSplit {
        address payable recipient;
        uint256 shareBps; // Share in basis points
    }

    struct Escrow {
        bytes32 paymentId;
        address payer;
        address payable payee;
        address token; // address(0) for native token
        uint256 amount;
        uint256 releaseTime;
        EscrowState state;
        bytes32 metadataHash; // IPFS hash of payment metadata
    }

    struct Dispute {
        uint256 escrowId;
        address initiator;
        bytes32 evidenceHash;
        DisputeResolution resolution;
        uint256 splitPercentPayer; // Only used if resolution is Split
        uint256 timestamp;
        address arbitrator;
    }

    // ============ State Variables ============

    /// @notice Platform fee in basis points
    uint256 public platformFeeBps = 250; // 2.5%

    /// @notice Address receiving platform fees
    address payable public feeRecipient;

    /// @notice Counter for escrow IDs
    uint256 public escrowCounter;

    /// @notice Counter for dispute IDs
    uint256 public disputeCounter;

    /// @notice Mapping of escrow ID to Escrow data
    mapping(uint256 => Escrow) public escrows;

    /// @notice Mapping of dispute ID to Dispute data
    mapping(uint256 => Dispute) public disputes;

    /// @notice Mapping of escrow ID to dispute ID (if disputed)
    mapping(uint256 => uint256) public escrowDisputes;

    /// @notice Mapping of payment ID to escrow ID for lookup
    mapping(bytes32 => uint256) public paymentToEscrow;

    /// @notice Accumulated fees by token address
    mapping(address => uint256) public accumulatedFees;

    /// @notice Whitelisted tokens for payments
    mapping(address => bool) public whitelistedTokens;

    /// @notice M-01 remediation: Pending withdrawals for pull-payment pattern
    /// @dev Maps address => token => amount available for withdrawal
    mapping(address => mapping(address => uint256)) public pendingWithdrawals;

    // ============ Events ============

    event PaymentRouted(
        bytes32 indexed paymentId,
        address indexed payer,
        address token,
        uint256 totalAmount,
        uint256 feeAmount,
        uint256 recipientCount
    );

    event RecipientPaid(
        bytes32 indexed paymentId,
        address indexed recipient,
        uint256 amount,
        uint256 shareBps
    );

    event EscrowCreated(
        uint256 indexed escrowId,
        bytes32 indexed paymentId,
        address indexed payer,
        address payee,
        address token,
        uint256 amount,
        uint256 releaseTime
    );

    event EscrowReleased(
        uint256 indexed escrowId,
        address indexed payee,
        uint256 amount
    );

    event EscrowRefunded(
        uint256 indexed escrowId,
        address indexed payer,
        uint256 amount
    );

    event DisputeOpened(
        uint256 indexed disputeId,
        uint256 indexed escrowId,
        address indexed initiator,
        bytes32 evidenceHash
    );

    event DisputeResolved(
        uint256 indexed disputeId,
        uint256 indexed escrowId,
        DisputeResolution resolution,
        address arbitrator
    );

    event PlatformFeeUpdated(uint256 oldFee, uint256 newFee);
    event FeeRecipientUpdated(address oldRecipient, address newRecipient);
    event TokenWhitelisted(address token, bool status);
    event FeesWithdrawn(address token, uint256 amount, address recipient);

    /// @notice M-01 remediation: Emitted when funds are credited for withdrawal
    event PendingWithdrawal(address indexed recipient, address indexed token, uint256 amount);

    /// @notice M-01 remediation: Emitted when pending funds are withdrawn
    event WithdrawalClaimed(address indexed recipient, address indexed token, uint256 amount);

    // ============ Errors ============

    error InvalidRecipients();
    error InvalidShares();
    error ZeroAmount();
    error ZeroAddress();
    error TokenNotWhitelisted();
    error EscrowNotActive();
    error EscrowNotReleasable();
    error EscrowAlreadyDisputed();
    error NotEscrowParticipant();
    error DisputeNotPending();
    error InvalidResolution();
    error InvalidDuration();
    error FeeTooHigh();
    error TransferFailed();
    error InsufficientValue();
    error NothingToWithdraw();

    // ============ Constructor ============

    /**
     * @notice Initializes the PaymentRouter
     * @param initialOwner Contract owner address
     * @param initialFeeRecipient Address to receive platform fees
     */
    constructor(
        address initialOwner,
        address payable initialFeeRecipient
    ) Ownable(initialOwner) {
        if (initialOwner == address(0) || initialFeeRecipient == address(0)) {
            revert ZeroAddress();
        }

        feeRecipient = initialFeeRecipient;

        _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
        _grantRole(GOVERNANCE_ROLE, initialOwner);
        _grantRole(ARBITRATOR_ROLE, initialOwner);
        _grantRole(ROUTER_ROLE, initialOwner);

        // Whitelist native token by default
        whitelistedTokens[address(0)] = true;
    }

    // ============ External Functions ============

    /**
     * @notice Routes a payment to multiple recipients with specified shares
     * @param paymentId Unique identifier for this payment
     * @param recipients Array of recipient addresses and their shares
     * @param token Token address (address(0) for native token)
     * @param totalAmount Total amount to distribute
     *
     * @dev Shares must sum to BASIS_POINTS (10000)
     * @dev Platform fee is deducted before distribution
     */
    function routePayment(
        bytes32 paymentId,
        PaymentSplit[] calldata recipients,
        address token,
        uint256 totalAmount
    ) external payable nonReentrant whenNotPaused {
        _validatePayment(recipients, token, totalAmount);

        // Handle token transfer
        if (token == address(0)) {
            if (msg.value < totalAmount) {
                revert InsufficientValue();
            }
        } else {
            IERC20(token).safeTransferFrom(msg.sender, address(this), totalAmount);
        }

        // Calculate and collect platform fee
        uint256 feeAmount = (totalAmount * platformFeeBps) / BASIS_POINTS;
        uint256 distributable = totalAmount - feeAmount;
        accumulatedFees[token] += feeAmount;

        // H-09 Fix: Track total distributed to handle dust from integer division
        // Distribute to recipients, giving any dust to the last recipient
        uint256 totalDistributed = 0;
        uint256 lastIndex = recipients.length - 1;

        for (uint256 i = 0; i < recipients.length; i++) {
            uint256 share;
            if (i == lastIndex) {
                // Last recipient gets remaining amount to avoid dust accumulation
                share = distributable - totalDistributed;
            } else {
                share = (distributable * recipients[i].shareBps) / BASIS_POINTS;
                totalDistributed += share;
            }
            _transferFunds(recipients[i].recipient, token, share);

            emit RecipientPaid(paymentId, recipients[i].recipient, share, recipients[i].shareBps);
        }

        emit PaymentRouted(
            paymentId,
            msg.sender,
            token,
            totalAmount,
            feeAmount,
            recipients.length
        );

        // Refund excess ETH
        if (token == address(0) && msg.value > totalAmount) {
            payable(msg.sender).sendValue(msg.value - totalAmount);
        }
    }

    /**
     * @notice Creates an escrow for a payment
     * @param paymentId Unique payment identifier
     * @param payee Address to receive funds on release
     * @param token Token address (address(0) for native)
     * @param amount Amount to escrow
     * @param duration Time until auto-release (in seconds)
     * @param metadataHash IPFS hash of payment metadata
     * @return escrowId The ID of the created escrow
     */
    function createEscrow(
        bytes32 paymentId,
        address payable payee,
        address token,
        uint256 amount,
        uint256 duration,
        bytes32 metadataHash
    ) external payable nonReentrant whenNotPaused returns (uint256 escrowId) {
        if (payee == address(0)) revert ZeroAddress();
        if (amount == 0) revert ZeroAmount();
        if (!whitelistedTokens[token]) revert TokenNotWhitelisted();
        if (duration < MIN_ESCROW_DURATION || duration > MAX_ESCROW_DURATION) {
            revert InvalidDuration();
        }

        // Handle token transfer
        if (token == address(0)) {
            if (msg.value < amount) revert InsufficientValue();
        } else {
            IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        }

        escrowId = ++escrowCounter;

        escrows[escrowId] = Escrow({
            paymentId: paymentId,
            payer: msg.sender,
            payee: payee,
            token: token,
            amount: amount,
            releaseTime: block.timestamp + duration,
            state: EscrowState.Active,
            metadataHash: metadataHash
        });

        paymentToEscrow[paymentId] = escrowId;

        emit EscrowCreated(
            escrowId,
            paymentId,
            msg.sender,
            payee,
            token,
            amount,
            block.timestamp + duration
        );

        // Refund excess ETH
        if (token == address(0) && msg.value > amount) {
            payable(msg.sender).sendValue(msg.value - amount);
        }

        return escrowId;
    }

    /**
     * @notice Releases escrow funds to the payee
     * @param escrowId The escrow to release
     * @dev Can be called by payer at any time, or by anyone after releaseTime
     */
    function releaseEscrow(uint256 escrowId) external nonReentrant whenNotPaused {
        Escrow storage escrow = escrows[escrowId];

        if (escrow.state != EscrowState.Active) revert EscrowNotActive();

        // Only payer can release early, anyone can release after time
        if (block.timestamp < escrow.releaseTime && msg.sender != escrow.payer) {
            revert EscrowNotReleasable();
        }

        escrow.state = EscrowState.Released;

        // Calculate and deduct fee
        uint256 feeAmount = (escrow.amount * platformFeeBps) / BASIS_POINTS;
        uint256 payeeAmount = escrow.amount - feeAmount;
        accumulatedFees[escrow.token] += feeAmount;

        _transferFunds(escrow.payee, escrow.token, payeeAmount);

        emit EscrowReleased(escrowId, escrow.payee, payeeAmount);
    }

    /**
     * @notice Opens a dispute on an active escrow
     * @param escrowId The escrow to dispute
     * @param evidenceHash IPFS hash of dispute evidence
     * @return disputeId The ID of the opened dispute
     */
    function openDispute(
        uint256 escrowId,
        bytes32 evidenceHash
    ) external nonReentrant whenNotPaused returns (uint256 disputeId) {
        Escrow storage escrow = escrows[escrowId];

        if (escrow.state != EscrowState.Active) revert EscrowNotActive();
        if (msg.sender != escrow.payer && msg.sender != escrow.payee) {
            revert NotEscrowParticipant();
        }

        escrow.state = EscrowState.Disputed;
        disputeId = ++disputeCounter;

        disputes[disputeId] = Dispute({
            escrowId: escrowId,
            initiator: msg.sender,
            evidenceHash: evidenceHash,
            resolution: DisputeResolution.Pending,
            splitPercentPayer: 0,
            timestamp: block.timestamp,
            arbitrator: address(0)
        });

        escrowDisputes[escrowId] = disputeId;

        emit DisputeOpened(disputeId, escrowId, msg.sender, evidenceHash);

        return disputeId;
    }

    /**
     * @notice Resolves a dispute (arbitrator only)
     * @param disputeId The dispute to resolve
     * @param resolution The resolution type
     * @param splitPercentPayer Payer's percentage if resolution is Split (0-100)
     *
     * @dev M-01 remediation: Uses pull-payment pattern for dispute resolutions
     *      to prevent reentrancy attacks during fund distribution.
     *      Recipients must call withdrawPending() to claim their funds.
     */
    function resolveDispute(
        uint256 disputeId,
        DisputeResolution resolution,
        uint256 splitPercentPayer
    ) external onlyRole(ARBITRATOR_ROLE) nonReentrant whenNotPaused {
        Dispute storage dispute = disputes[disputeId];
        Escrow storage escrow = escrows[dispute.escrowId];

        if (dispute.resolution != DisputeResolution.Pending) {
            revert DisputeNotPending();
        }
        if (resolution == DisputeResolution.Pending) {
            revert InvalidResolution();
        }
        if (resolution == DisputeResolution.Split && splitPercentPayer > 100) {
            revert InvalidResolution();
        }

        // M-01: All state changes happen BEFORE any potential external interactions
        dispute.resolution = resolution;
        dispute.splitPercentPayer = splitPercentPayer;
        dispute.arbitrator = msg.sender;
        escrow.state = EscrowState.Resolved;

        // Calculate fee first (effects)
        uint256 feeAmount = (escrow.amount * platformFeeBps) / BASIS_POINTS;
        uint256 distributable = escrow.amount - feeAmount;
        accumulatedFees[escrow.token] += feeAmount;

        // M-01: Credit pending withdrawals instead of direct transfer (pull-payment pattern)
        // This prevents reentrancy and griefing attacks during dispute resolution
        if (resolution == DisputeResolution.ReleaseToPayer) {
            pendingWithdrawals[escrow.payer][escrow.token] += distributable;
            emit PendingWithdrawal(escrow.payer, escrow.token, distributable);
        } else if (resolution == DisputeResolution.ReleaseToPayee) {
            pendingWithdrawals[escrow.payee][escrow.token] += distributable;
            emit PendingWithdrawal(escrow.payee, escrow.token, distributable);
        } else if (resolution == DisputeResolution.Split) {
            uint256 payerAmount = (distributable * splitPercentPayer) / 100;
            uint256 payeeAmount = distributable - payerAmount;

            if (payerAmount > 0) {
                pendingWithdrawals[escrow.payer][escrow.token] += payerAmount;
                emit PendingWithdrawal(escrow.payer, escrow.token, payerAmount);
            }
            if (payeeAmount > 0) {
                pendingWithdrawals[escrow.payee][escrow.token] += payeeAmount;
                emit PendingWithdrawal(escrow.payee, escrow.token, payeeAmount);
            }
        }

        emit DisputeResolved(disputeId, dispute.escrowId, resolution, msg.sender);
    }

    /**
     * @notice M-01 remediation: Withdraw pending funds (pull-payment pattern)
     * @param token Token to withdraw (address(0) for native)
     *
     * @dev Allows recipients to safely claim funds credited from dispute resolutions.
     *      This prevents reentrancy and griefing attacks.
     */
    function withdrawPending(address token) external nonReentrant whenNotPaused {
        uint256 amount = pendingWithdrawals[msg.sender][token];
        if (amount == 0) revert NothingToWithdraw();

        // M-01: Clear state BEFORE external call (CEI pattern)
        pendingWithdrawals[msg.sender][token] = 0;

        // Now safe to make external call
        _transferFunds(payable(msg.sender), token, amount);

        emit WithdrawalClaimed(msg.sender, token, amount);
    }

    /**
     * @notice Refunds an escrow to the payer (payee consent required)
     * @param escrowId The escrow to refund
     * @dev Only payee can call this for active escrows
     */
    function refundEscrow(uint256 escrowId) external nonReentrant whenNotPaused {
        Escrow storage escrow = escrows[escrowId];

        if (escrow.state != EscrowState.Active) revert EscrowNotActive();
        if (msg.sender != escrow.payee) revert NotEscrowParticipant();

        escrow.state = EscrowState.Refunded;

        // No fee on refunds
        _transferFunds(payable(escrow.payer), escrow.token, escrow.amount);

        emit EscrowRefunded(escrowId, escrow.payer, escrow.amount);
    }

    // ============ Admin Functions ============

    /**
     * @notice Updates the platform fee
     * @param newFeeBps New fee in basis points
     */
    function setPlatformFee(uint256 newFeeBps) external onlyRole(GOVERNANCE_ROLE) {
        if (newFeeBps > MAX_PLATFORM_FEE) revert FeeTooHigh();

        uint256 oldFee = platformFeeBps;
        platformFeeBps = newFeeBps;

        emit PlatformFeeUpdated(oldFee, newFeeBps);
    }

    /**
     * @notice Updates the fee recipient
     * @param newRecipient New fee recipient address
     */
    function setFeeRecipient(
        address payable newRecipient
    ) external onlyRole(GOVERNANCE_ROLE) {
        if (newRecipient == address(0)) revert ZeroAddress();

        address oldRecipient = feeRecipient;
        feeRecipient = newRecipient;

        emit FeeRecipientUpdated(oldRecipient, newRecipient);
    }

    /**
     * @notice Whitelists or de-whitelists a token
     * @param token Token address
     * @param status Whether to whitelist
     */
    function setTokenWhitelist(
        address token,
        bool status
    ) external onlyRole(GOVERNANCE_ROLE) {
        whitelistedTokens[token] = status;
        emit TokenWhitelisted(token, status);
    }

    /**
     * @notice Withdraws accumulated platform fees
     * @param token Token to withdraw (address(0) for native)
     */
    function withdrawFees(address token) external onlyRole(GOVERNANCE_ROLE) nonReentrant {
        uint256 amount = accumulatedFees[token];
        if (amount == 0) revert ZeroAmount();

        accumulatedFees[token] = 0;
        _transferFunds(feeRecipient, token, amount);

        emit FeesWithdrawn(token, amount, feeRecipient);
    }

    /**
     * @notice Pauses the contract
     */
    function pause() external onlyRole(GOVERNANCE_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses the contract
     */
    function unpause() external onlyRole(GOVERNANCE_ROLE) {
        _unpause();
    }

    // ============ View Functions ============

    /**
     * @notice Gets escrow details
     * @param escrowId The escrow ID
     */
    function getEscrow(uint256 escrowId) external view returns (Escrow memory) {
        return escrows[escrowId];
    }

    /**
     * @notice Gets dispute details
     * @param disputeId The dispute ID
     */
    function getDispute(uint256 disputeId) external view returns (Dispute memory) {
        return disputes[disputeId];
    }

    /**
     * @notice Checks if an escrow is releasable
     * @param escrowId The escrow to check
     */
    function isEscrowReleasable(uint256 escrowId) external view returns (bool) {
        Escrow storage escrow = escrows[escrowId];
        return escrow.state == EscrowState.Active && block.timestamp >= escrow.releaseTime;
    }

    /**
     * @notice Calculates the fee for an amount
     * @param amount The amount to calculate fee for
     */
    function calculateFee(uint256 amount) external view returns (uint256) {
        return (amount * platformFeeBps) / BASIS_POINTS;
    }

    /**
     * @notice M-01 remediation: Gets pending withdrawal balance
     * @param account Address to check
     * @param token Token address (address(0) for native)
     */
    function getPendingWithdrawal(address account, address token) external view returns (uint256) {
        return pendingWithdrawals[account][token];
    }

    // ============ Internal Functions ============

    /**
     * @dev Validates payment parameters
     */
    function _validatePayment(
        PaymentSplit[] calldata recipients,
        address token,
        uint256 totalAmount
    ) internal view {
        if (recipients.length == 0 || recipients.length > MAX_RECIPIENTS) {
            revert InvalidRecipients();
        }
        if (totalAmount == 0) revert ZeroAmount();
        if (!whitelistedTokens[token]) revert TokenNotWhitelisted();

        uint256 totalShares = 0;
        for (uint256 i = 0; i < recipients.length; i++) {
            if (recipients[i].recipient == address(0)) revert ZeroAddress();
            totalShares += recipients[i].shareBps;
        }

        if (totalShares != BASIS_POINTS) revert InvalidShares();
    }

    /**
     * @dev Transfers funds (native or ERC20)
     */
    function _transferFunds(
        address payable recipient,
        address token,
        uint256 amount
    ) internal {
        if (amount == 0) return;

        if (token == address(0)) {
            recipient.sendValue(amount);
        } else {
            IERC20(token).safeTransfer(recipient, amount);
        }
    }

    /**
     * @dev Allows receiving native tokens
     */
    receive() external payable {}
}
