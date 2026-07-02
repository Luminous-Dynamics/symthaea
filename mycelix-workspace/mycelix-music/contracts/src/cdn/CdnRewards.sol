// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./CdnRegistry.sol";

/// @title CdnRewards - Reward Distribution for CDN Nodes
/// @notice Distributes rewards to CDN nodes based on PoGQ scores and service metrics
/// @dev Rewards come from protocol fees and optional TEND token inflation
contract CdnRewards {
    // ============ State Variables ============

    /// @notice Registry contract
    CdnRegistry public registry;

    /// @notice Owner address
    address public owner;

    /// @notice Total rewards available for distribution
    uint256 public rewardsPool;

    /// @notice Epoch duration (how often rewards are distributed)
    uint256 public epochDuration = 1 days;

    /// @notice Last epoch timestamp
    uint256 public lastEpoch;

    /// @notice Current epoch number
    uint256 public currentEpoch;

    /// @notice Rewards per epoch (base amount)
    uint256 public rewardsPerEpoch = 100 ether; // 100 xDAI per epoch

    /// @notice Minimum PoGQ score to receive rewards (basis points)
    uint256 public minPogqForRewards = 5000; // 50%

    /// @notice Weight factors for reward calculation
    uint256 public pogqWeight = 50; // 50% weight on PoGQ score
    uint256 public bytesWeight = 30; // 30% weight on bytes served
    uint256 public uptimeWeight = 20; // 20% weight on uptime

    /// @notice Bytes served per epoch per node (for relative calculation)
    mapping(address => uint256) public epochBytesServed;

    /// @notice Previous epoch bytes (for delta calculation)
    mapping(address => uint256) public prevEpochBytes;

    // ============ Events ============

    event EpochCompleted(uint256 indexed epoch, uint256 totalDistributed, uint256 nodeCount);
    event RewardsPoolFunded(address indexed funder, uint256 amount);
    event RewardsParametersUpdated(uint256 rewardsPerEpoch, uint256 minPogq);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    // ============ Constructor ============

    constructor(address _registry) {
        owner = msg.sender;
        registry = CdnRegistry(payable(_registry));
        lastEpoch = block.timestamp;
    }

    // ============ External Functions ============

    /// @notice Distribute rewards for the current epoch
    /// @dev Can be called by anyone once epoch duration has passed
    function distributeRewards() external {
        require(
            block.timestamp >= lastEpoch + epochDuration,
            "Epoch not complete"
        );

        address[] memory activeNodes = registry.getActiveNodes();
        require(activeNodes.length > 0, "No active nodes");

        // Calculate total weight across all eligible nodes
        uint256 totalWeight = 0;
        uint256[] memory weights = new uint256[](activeNodes.length);
        uint256 totalBytesThisEpoch = 0;

        for (uint256 i = 0; i < activeNodes.length; i++) {
            ICdnNode.Node memory node = registry.getNode(activeNodes[i]);

            // Skip nodes below minimum PoGQ
            if (node.pogqScore < minPogqForRewards) {
                continue;
            }

            // Calculate bytes served this epoch
            uint256 bytesThisEpoch = node.totalBytesServed - prevEpochBytes[activeNodes[i]];
            epochBytesServed[activeNodes[i]] = bytesThisEpoch;
            totalBytesThisEpoch += bytesThisEpoch;
        }

        // Calculate weights for each node
        for (uint256 i = 0; i < activeNodes.length; i++) {
            ICdnNode.Node memory node = registry.getNode(activeNodes[i]);

            if (node.pogqScore < minPogqForRewards) {
                weights[i] = 0;
                continue;
            }

            // PoGQ component (0-10000 basis points, scale to 100)
            uint256 pogqComponent = (node.pogqScore * pogqWeight) / 100;

            // Bytes component (relative to total)
            uint256 bytesComponent = 0;
            if (totalBytesThisEpoch > 0) {
                bytesComponent = (epochBytesServed[activeNodes[i]] * bytesWeight * 100) / totalBytesThisEpoch;
            }

            // Uptime component (success rate)
            uint256 uptimeComponent = 0;
            uint256 totalRequests = node.successfulRequests + node.failedRequests;
            if (totalRequests > 0) {
                uint256 successRate = (node.successfulRequests * 10000) / totalRequests;
                uptimeComponent = (successRate * uptimeWeight) / 100;
            }

            weights[i] = pogqComponent + bytesComponent + uptimeComponent;
            totalWeight += weights[i];

            // Update prev epoch bytes for next calculation
            prevEpochBytes[activeNodes[i]] = node.totalBytesServed;
        }

        // Determine actual rewards to distribute
        uint256 toDistribute = rewardsPerEpoch;
        if (toDistribute > rewardsPool) {
            toDistribute = rewardsPool;
        }

        // Distribute rewards proportionally
        uint256 distributed = 0;
        for (uint256 i = 0; i < activeNodes.length; i++) {
            if (weights[i] == 0 || totalWeight == 0) {
                continue;
            }

            uint256 nodeReward = (toDistribute * weights[i]) / totalWeight;
            if (nodeReward > 0) {
                registry.addRewards(activeNodes[i], nodeReward);
                distributed += nodeReward;
            }
        }

        rewardsPool -= distributed;
        lastEpoch = block.timestamp;
        currentEpoch++;

        emit EpochCompleted(currentEpoch, distributed, activeNodes.length);
    }

    /// @notice Fund the rewards pool
    function fundRewardsPool() external payable {
        rewardsPool += msg.value;
        emit RewardsPoolFunded(msg.sender, msg.value);
    }

    /// @notice Get estimated rewards for a node for current epoch
    /// @param operator Node operator address
    /// @return Estimated reward amount
    function estimateRewards(address operator) external view returns (uint256) {
        ICdnNode.Node memory node = registry.getNode(operator);

        if (node.pogqScore < minPogqForRewards) {
            return 0;
        }

        address[] memory activeNodes = registry.getActiveNodes();
        uint256 eligibleNodes = 0;

        for (uint256 i = 0; i < activeNodes.length; i++) {
            ICdnNode.Node memory n = registry.getNode(activeNodes[i]);
            if (n.pogqScore >= minPogqForRewards) {
                eligibleNodes++;
            }
        }

        if (eligibleNodes == 0) {
            return 0;
        }

        // Simplified estimation based on equal share adjusted by PoGQ
        uint256 baseShare = rewardsPerEpoch / eligibleNodes;
        uint256 pogqMultiplier = node.pogqScore; // Basis points

        return (baseShare * pogqMultiplier) / 10000;
    }

    /// @notice Get time until next epoch
    /// @return Seconds until next epoch
    function timeUntilNextEpoch() external view returns (uint256) {
        uint256 nextEpoch = lastEpoch + epochDuration;
        if (block.timestamp >= nextEpoch) {
            return 0;
        }
        return nextEpoch - block.timestamp;
    }

    // ============ Admin Functions ============

    /// @notice Set rewards per epoch
    /// @param _rewardsPerEpoch New rewards per epoch
    function setRewardsPerEpoch(uint256 _rewardsPerEpoch) external onlyOwner {
        rewardsPerEpoch = _rewardsPerEpoch;
        emit RewardsParametersUpdated(_rewardsPerEpoch, minPogqForRewards);
    }

    /// @notice Set minimum PoGQ score for rewards
    /// @param _minPogq New minimum PoGQ (basis points)
    function setMinPogqForRewards(uint256 _minPogq) external onlyOwner {
        require(_minPogq <= 10000, "Invalid basis points");
        minPogqForRewards = _minPogq;
        emit RewardsParametersUpdated(rewardsPerEpoch, _minPogq);
    }

    /// @notice Set epoch duration
    /// @param _duration New epoch duration
    function setEpochDuration(uint256 _duration) external onlyOwner {
        require(_duration >= 1 hours, "Too short");
        epochDuration = _duration;
    }

    /// @notice Set weight factors
    /// @param _pogq PoGQ weight (percentage)
    /// @param _bytes Bytes weight (percentage)
    /// @param _uptime Uptime weight (percentage)
    function setWeights(uint256 _pogq, uint256 _bytes, uint256 _uptime) external onlyOwner {
        require(_pogq + _bytes + _uptime == 100, "Weights must sum to 100");
        pogqWeight = _pogq;
        bytesWeight = _bytes;
        uptimeWeight = _uptime;
    }

    /// @notice Update registry reference
    /// @param _registry New registry address
    function setRegistry(address _registry) external onlyOwner {
        registry = CdnRegistry(payable(_registry));
    }

    // ============ Receive ============

    /// @notice Receive xDAI for rewards pool
    receive() external payable {
        rewardsPool += msg.value;
        emit RewardsPoolFunded(msg.sender, msg.value);
    }
}
