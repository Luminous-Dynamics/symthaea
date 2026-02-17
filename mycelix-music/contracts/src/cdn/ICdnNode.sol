// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ICdnNode - Interface for CDN Node operations
/// @notice Defines the interface for community-owned CDN node incentives
interface ICdnNode {
    /// @notice Node status
    enum NodeStatus {
        Inactive,
        Active,
        Slashed,
        Exiting
    }

    /// @notice Node information
    struct Node {
        address operator;
        string ipfsPeerId;
        string region;
        uint256 stakedAmount;
        uint256 registeredAt;
        uint256 lastActiveAt;
        uint256 totalBytesServed;
        uint256 successfulRequests;
        uint256 failedRequests;
        uint256 pogqScore; // Basis points (0-10000)
        uint256 pendingRewards;
        uint256 slashCount;
        NodeStatus status;
    }

    /// @notice Emitted when a new node is registered
    event NodeRegistered(
        address indexed operator,
        string ipfsPeerId,
        string region,
        uint256 stakedAmount
    );

    /// @notice Emitted when a node is updated
    event NodeUpdated(
        address indexed operator,
        uint256 pogqScore,
        uint256 totalBytesServed
    );

    /// @notice Emitted when rewards are distributed
    event RewardsDistributed(
        address indexed operator,
        uint256 amount,
        uint256 epoch
    );

    /// @notice Emitted when a node is slashed
    event NodeSlashed(
        address indexed operator,
        uint256 amount,
        string reason
    );

    /// @notice Emitted when a node exits
    event NodeExited(
        address indexed operator,
        uint256 returnedStake
    );

    /// @notice Register as a CDN node
    /// @param ipfsPeerId The IPFS peer ID for this node
    /// @param region Geographic region code
    function registerNode(string calldata ipfsPeerId, string calldata region) external payable;

    /// @notice Update node metrics (called by oracle)
    /// @param operator Node operator address
    /// @param pogqScore New PoGQ score (basis points)
    /// @param bytesServed Bytes served since last update
    /// @param successful Successful requests since last update
    /// @param failed Failed requests since last update
    function updateNodeMetrics(
        address operator,
        uint256 pogqScore,
        uint256 bytesServed,
        uint256 successful,
        uint256 failed
    ) external;

    /// @notice Claim pending rewards
    function claimRewards() external;

    /// @notice Initiate node exit (starts unbonding period)
    function initiateExit() external;

    /// @notice Complete exit and withdraw stake
    function completeExit() external;

    /// @notice Get node information
    /// @param operator Node operator address
    /// @return Node struct with all information
    function getNode(address operator) external view returns (Node memory);

    /// @notice Get all active nodes
    /// @return Array of active node addresses
    function getActiveNodes() external view returns (address[] memory);

    /// @notice Get minimum stake required
    /// @return Minimum stake in wei
    function minStake() external view returns (uint256);
}
