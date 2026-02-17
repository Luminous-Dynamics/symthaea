// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/TimelockController.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title MycelixTreasury
 * @notice DAO-controlled treasury for Mycelix
 * @dev Extends TimelockController for governance-approved spending
 *
 * Treasury funds come from:
 * - Platform fees
 * - Artist percentage donations
 * - Community funding rounds
 * - Partner contributions
 *
 * Funds are allocated for:
 * - Artist grants
 * - Development bounties
 * - Community initiatives
 * - Infrastructure costs
 */
contract MycelixTreasury is TimelockController {
    using SafeERC20 for IERC20;

    // ============================================================================
    // State
    // ============================================================================

    // Budget allocations (in basis points, 10000 = 100%)
    struct BudgetAllocation {
        uint256 artistGrants;
        uint256 development;
        uint256 community;
        uint256 infrastructure;
        uint256 reserve;
    }

    BudgetAllocation public budgetAllocation;

    // Grant program
    struct Grant {
        address recipient;
        uint256 amount;
        string description;
        uint256 createdAt;
        uint256 claimedAt;
        bool active;
    }

    mapping(uint256 => Grant) public grants;
    uint256 public grantCounter;

    // Spending tracking
    mapping(address => uint256) public monthlySpending;
    mapping(address => uint256) public lastSpendingReset;
    mapping(address => uint256) public spendingLimit;

    // ============================================================================
    // Events
    // ============================================================================

    event GrantCreated(uint256 indexed grantId, address recipient, uint256 amount, string description);
    event GrantClaimed(uint256 indexed grantId, address recipient, uint256 amount);
    event GrantRevoked(uint256 indexed grantId);
    event BudgetUpdated(BudgetAllocation newAllocation);
    event DonationReceived(address indexed donor, uint256 amount, string message);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(
        uint256 minDelay,
        address[] memory proposers,
        address[] memory executors,
        address admin
    ) TimelockController(minDelay, proposers, executors, admin) {
        // Default budget allocation
        budgetAllocation = BudgetAllocation({
            artistGrants: 4000,     // 40%
            development: 2500,      // 25%
            community: 2000,        // 20%
            infrastructure: 1000,   // 10%
            reserve: 500            // 5%
        });
    }

    // ============================================================================
    // Receiving Funds
    // ============================================================================

    receive() external payable {
        emit DonationReceived(msg.sender, msg.value, "");
    }

    /**
     * @notice Donate with a message
     */
    function donate(string calldata message) external payable {
        require(msg.value > 0, "No value sent");
        emit DonationReceived(msg.sender, msg.value, message);
    }

    /**
     * @notice Donate ERC20 tokens
     */
    function donateToken(
        IERC20 token,
        uint256 amount,
        string calldata message
    ) external {
        require(amount > 0, "No amount specified");
        token.safeTransferFrom(msg.sender, address(this), amount);
        emit DonationReceived(msg.sender, amount, message);
    }

    // ============================================================================
    // Grant Management
    // ============================================================================

    /**
     * @notice Create a new grant (requires governance approval via execute)
     */
    function createGrant(
        address recipient,
        uint256 amount,
        string calldata description
    ) external onlyRole(EXECUTOR_ROLE) returns (uint256) {
        require(recipient != address(0), "Invalid recipient");
        require(amount > 0, "Invalid amount");
        require(address(this).balance >= amount, "Insufficient treasury balance");

        uint256 grantId = grantCounter++;

        grants[grantId] = Grant({
            recipient: recipient,
            amount: amount,
            description: description,
            createdAt: block.timestamp,
            claimedAt: 0,
            active: true
        });

        emit GrantCreated(grantId, recipient, amount, description);
        return grantId;
    }

    /**
     * @notice Claim an approved grant
     */
    function claimGrant(uint256 grantId) external {
        Grant storage grant = grants[grantId];

        require(grant.active, "Grant not active");
        require(grant.recipient == msg.sender, "Not grant recipient");
        require(grant.claimedAt == 0, "Already claimed");
        require(address(this).balance >= grant.amount, "Insufficient balance");

        grant.claimedAt = block.timestamp;
        grant.active = false;

        (bool success, ) = payable(msg.sender).call{value: grant.amount}("");
        require(success, "Transfer failed");

        emit GrantClaimed(grantId, msg.sender, grant.amount);
    }

    /**
     * @notice Revoke an unclaimed grant
     */
    function revokeGrant(uint256 grantId) external onlyRole(EXECUTOR_ROLE) {
        Grant storage grant = grants[grantId];
        require(grant.active, "Grant not active");
        require(grant.claimedAt == 0, "Already claimed");

        grant.active = false;
        emit GrantRevoked(grantId);
    }

    // ============================================================================
    // Budget Management
    // ============================================================================

    /**
     * @notice Update budget allocation
     */
    function updateBudget(BudgetAllocation calldata newAllocation)
        external
        onlyRole(EXECUTOR_ROLE)
    {
        require(
            newAllocation.artistGrants +
            newAllocation.development +
            newAllocation.community +
            newAllocation.infrastructure +
            newAllocation.reserve == 10000,
            "Must total 100%"
        );

        budgetAllocation = newAllocation;
        emit BudgetUpdated(newAllocation);
    }

    /**
     * @notice Get available budget for a category
     */
    function getAvailableBudget(string calldata category)
        external
        view
        returns (uint256)
    {
        uint256 totalBalance = address(this).balance;
        bytes32 categoryHash = keccak256(bytes(category));

        if (categoryHash == keccak256("artistGrants")) {
            return (totalBalance * budgetAllocation.artistGrants) / 10000;
        } else if (categoryHash == keccak256("development")) {
            return (totalBalance * budgetAllocation.development) / 10000;
        } else if (categoryHash == keccak256("community")) {
            return (totalBalance * budgetAllocation.community) / 10000;
        } else if (categoryHash == keccak256("infrastructure")) {
            return (totalBalance * budgetAllocation.infrastructure) / 10000;
        } else if (categoryHash == keccak256("reserve")) {
            return (totalBalance * budgetAllocation.reserve) / 10000;
        }

        return 0;
    }

    // ============================================================================
    // View Functions
    // ============================================================================

    /**
     * @notice Get treasury balance
     */
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    /**
     * @notice Get ERC20 token balance
     */
    function getTokenBalance(IERC20 token) external view returns (uint256) {
        return token.balanceOf(address(this));
    }

    /**
     * @notice Get grant details
     */
    function getGrant(uint256 grantId) external view returns (
        address recipient,
        uint256 amount,
        string memory description,
        uint256 createdAt,
        uint256 claimedAt,
        bool active
    ) {
        Grant storage grant = grants[grantId];
        return (
            grant.recipient,
            grant.amount,
            grant.description,
            grant.createdAt,
            grant.claimedAt,
            grant.active
        );
    }
}
