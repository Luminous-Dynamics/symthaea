// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title RoyaltySplitter
 * @notice Automatic revenue distribution for collaborative works
 * @dev Immutable splits defined at creation, transparent distribution
 *
 * Features:
 * - Define splits at track/album creation
 * - Automatic distribution when funds received
 * - Support for ETH and ERC20 tokens
 * - On-chain transparency for all payments
 * - Batch claiming for gas efficiency
 */
contract RoyaltySplitter is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ============================================================================
    // State
    // ============================================================================

    struct Split {
        address payee;
        uint256 shares;     // Basis points (10000 = 100%)
        string role;        // e.g., "Producer", "Vocalist", "Songwriter"
    }

    struct SplitConfig {
        bytes32 id;
        string name;            // Track/album name
        string contentUri;      // IPFS URI of the content
        Split[] splits;
        uint256 totalShares;
        uint256 createdAt;
        bool finalized;
    }

    struct PayeeBalance {
        uint256 ethBalance;
        mapping(address => uint256) tokenBalances;
    }

    // Split config ID => config
    mapping(bytes32 => SplitConfig) public splitConfigs;

    // Payee => balances
    mapping(address => PayeeBalance) private payeeBalances;

    // Track all tokens ever received
    address[] public knownTokens;
    mapping(address => bool) public isKnownToken;

    // Split config IDs for enumeration
    bytes32[] public allSplitIds;

    // Factory for creating new splits
    address public splitFactory;

    // ============================================================================
    // Events
    // ============================================================================

    event SplitCreated(bytes32 indexed splitId, string name, address indexed creator);
    event SplitFinalized(bytes32 indexed splitId);
    event PaymentReceived(bytes32 indexed splitId, address token, uint256 amount);
    event PaymentDistributed(bytes32 indexed splitId, address indexed payee, address token, uint256 amount);
    event FundsWithdrawn(address indexed payee, address token, uint256 amount);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor() Ownable(msg.sender) {}

    // ============================================================================
    // Split Management
    // ============================================================================

    /**
     * @notice Create a new royalty split configuration
     * @param name Name of the track/album
     * @param contentUri IPFS URI of the content
     * @param payees Array of payee addresses
     * @param shares Array of shares (basis points, must sum to 10000)
     * @param roles Array of role descriptions
     */
    function createSplit(
        string calldata name,
        string calldata contentUri,
        address[] calldata payees,
        uint256[] calldata shares,
        string[] calldata roles
    ) external returns (bytes32) {
        require(payees.length == shares.length, "Length mismatch");
        require(payees.length == roles.length, "Length mismatch");
        require(payees.length > 0, "No payees");

        uint256 totalShares = 0;
        for (uint256 i = 0; i < shares.length; i++) {
            require(payees[i] != address(0), "Invalid payee");
            require(shares[i] > 0, "Zero shares");
            totalShares += shares[i];
        }
        require(totalShares == 10000, "Shares must equal 10000");

        bytes32 splitId = keccak256(abi.encodePacked(name, contentUri, block.timestamp, msg.sender));

        SplitConfig storage config = splitConfigs[splitId];
        config.id = splitId;
        config.name = name;
        config.contentUri = contentUri;
        config.totalShares = totalShares;
        config.createdAt = block.timestamp;
        config.finalized = false;

        for (uint256 i = 0; i < payees.length; i++) {
            config.splits.push(Split({
                payee: payees[i],
                shares: shares[i],
                role: roles[i]
            }));
        }

        allSplitIds.push(splitId);
        emit SplitCreated(splitId, name, msg.sender);

        return splitId;
    }

    /**
     * @notice Finalize a split (prevents modifications)
     */
    function finalizeSplit(bytes32 splitId) external {
        SplitConfig storage config = splitConfigs[splitId];
        require(config.createdAt > 0, "Split doesn't exist");
        require(!config.finalized, "Already finalized");

        // Only a payee can finalize
        bool isPayee = false;
        for (uint256 i = 0; i < config.splits.length; i++) {
            if (config.splits[i].payee == msg.sender) {
                isPayee = true;
                break;
            }
        }
        require(isPayee, "Not a payee");

        config.finalized = true;
        emit SplitFinalized(splitId);
    }

    // ============================================================================
    // Payment Distribution
    // ============================================================================

    /**
     * @notice Receive and distribute ETH payment
     */
    function distributeETH(bytes32 splitId) external payable nonReentrant {
        require(msg.value > 0, "No ETH sent");

        SplitConfig storage config = splitConfigs[splitId];
        require(config.createdAt > 0, "Split doesn't exist");

        _distributePayment(splitId, address(0), msg.value);
    }

    /**
     * @notice Distribute ERC20 payment (must approve first)
     */
    function distributeToken(bytes32 splitId, IERC20 token, uint256 amount) external nonReentrant {
        require(amount > 0, "No tokens");

        SplitConfig storage config = splitConfigs[splitId];
        require(config.createdAt > 0, "Split doesn't exist");

        token.safeTransferFrom(msg.sender, address(this), amount);

        // Track token
        if (!isKnownToken[address(token)]) {
            knownTokens.push(address(token));
            isKnownToken[address(token)] = true;
        }

        _distributePayment(splitId, address(token), amount);
    }

    function _distributePayment(bytes32 splitId, address token, uint256 amount) internal {
        SplitConfig storage config = splitConfigs[splitId];

        emit PaymentReceived(splitId, token, amount);

        for (uint256 i = 0; i < config.splits.length; i++) {
            Split storage split = config.splits[i];
            uint256 payeeAmount = (amount * split.shares) / 10000;

            if (payeeAmount > 0) {
                if (token == address(0)) {
                    payeeBalances[split.payee].ethBalance += payeeAmount;
                } else {
                    payeeBalances[split.payee].tokenBalances[token] += payeeAmount;
                }

                emit PaymentDistributed(splitId, split.payee, token, payeeAmount);
            }
        }
    }

    // ============================================================================
    // Withdrawals
    // ============================================================================

    /**
     * @notice Withdraw all ETH balance
     */
    function withdrawETH() external nonReentrant {
        uint256 amount = payeeBalances[msg.sender].ethBalance;
        require(amount > 0, "No ETH balance");

        payeeBalances[msg.sender].ethBalance = 0;

        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");

        emit FundsWithdrawn(msg.sender, address(0), amount);
    }

    /**
     * @notice Withdraw specific token balance
     */
    function withdrawToken(IERC20 token) external nonReentrant {
        uint256 amount = payeeBalances[msg.sender].tokenBalances[address(token)];
        require(amount > 0, "No token balance");

        payeeBalances[msg.sender].tokenBalances[address(token)] = 0;
        token.safeTransfer(msg.sender, amount);

        emit FundsWithdrawn(msg.sender, address(token), amount);
    }

    /**
     * @notice Withdraw all balances (ETH + all known tokens)
     */
    function withdrawAll() external nonReentrant {
        // Withdraw ETH
        uint256 ethAmount = payeeBalances[msg.sender].ethBalance;
        if (ethAmount > 0) {
            payeeBalances[msg.sender].ethBalance = 0;
            (bool success, ) = payable(msg.sender).call{value: ethAmount}("");
            require(success, "ETH transfer failed");
            emit FundsWithdrawn(msg.sender, address(0), ethAmount);
        }

        // Withdraw all tokens
        for (uint256 i = 0; i < knownTokens.length; i++) {
            address tokenAddr = knownTokens[i];
            uint256 tokenAmount = payeeBalances[msg.sender].tokenBalances[tokenAddr];

            if (tokenAmount > 0) {
                payeeBalances[msg.sender].tokenBalances[tokenAddr] = 0;
                IERC20(tokenAddr).safeTransfer(msg.sender, tokenAmount);
                emit FundsWithdrawn(msg.sender, tokenAddr, tokenAmount);
            }
        }
    }

    // ============================================================================
    // View Functions
    // ============================================================================

    function getSplit(bytes32 splitId) external view returns (
        string memory name,
        string memory contentUri,
        uint256 createdAt,
        bool finalized,
        Split[] memory splits
    ) {
        SplitConfig storage config = splitConfigs[splitId];
        return (config.name, config.contentUri, config.createdAt, config.finalized, config.splits);
    }

    function getPayeeETHBalance(address payee) external view returns (uint256) {
        return payeeBalances[payee].ethBalance;
    }

    function getPayeeTokenBalance(address payee, address token) external view returns (uint256) {
        return payeeBalances[payee].tokenBalances[token];
    }

    function getSplitCount() external view returns (uint256) {
        return allSplitIds.length;
    }

    function getSplitsForPayee(address payee) external view returns (bytes32[] memory) {
        uint256 count = 0;

        // Count splits involving this payee
        for (uint256 i = 0; i < allSplitIds.length; i++) {
            SplitConfig storage config = splitConfigs[allSplitIds[i]];
            for (uint256 j = 0; j < config.splits.length; j++) {
                if (config.splits[j].payee == payee) {
                    count++;
                    break;
                }
            }
        }

        // Build result array
        bytes32[] memory result = new bytes32[](count);
        uint256 index = 0;

        for (uint256 i = 0; i < allSplitIds.length; i++) {
            SplitConfig storage config = splitConfigs[allSplitIds[i]];
            for (uint256 j = 0; j < config.splits.length; j++) {
                if (config.splits[j].payee == payee) {
                    result[index++] = allSplitIds[i];
                    break;
                }
            }
        }

        return result;
    }

    // ============================================================================
    // Receive
    // ============================================================================

    receive() external payable {
        // Accept ETH but don't auto-distribute (needs splitId)
    }
}
