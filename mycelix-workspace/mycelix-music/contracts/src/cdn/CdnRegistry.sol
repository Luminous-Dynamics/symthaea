// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./ICdnNode.sol";

/// @title CdnRegistry - CDN Node Registration and Staking
/// @notice Manages the registry of community-owned CDN nodes
/// @dev Nodes stake xDAI to participate, earn rewards based on PoGQ scores
contract CdnRegistry is ICdnNode {
    // ============ State Variables ============

    /// @notice Minimum stake required to register as a node
    uint256 public override minStake = 100 ether; // 100 xDAI

    /// @notice Unbonding period for node exits
    uint256 public unbondingPeriod = 7 days;

    /// @notice Oracle address for metric updates
    address public oracle;

    /// @notice Owner address for admin functions
    address public owner;

    /// @notice Mapping of operator address to node data
    mapping(address => Node) public nodes;

    /// @notice Array of all registered node operators
    address[] public nodeOperators;

    /// @notice Mapping of operator to their index in nodeOperators
    mapping(address => uint256) public operatorIndex;

    /// @notice Exit timestamps for unbonding
    mapping(address => uint256) public exitTimestamps;

    /// @notice Total staked across all nodes
    uint256 public totalStaked;

    /// @notice Total active nodes
    uint256 public activeNodeCount;

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == oracle, "Not oracle");
        _;
    }

    modifier onlyActiveNode() {
        require(nodes[msg.sender].status == NodeStatus.Active, "Not active node");
        _;
    }

    // ============ Constructor ============

    constructor(address _oracle) {
        owner = msg.sender;
        oracle = _oracle;
    }

    // ============ External Functions ============

    /// @inheritdoc ICdnNode
    function registerNode(
        string calldata ipfsPeerId,
        string calldata region
    ) external payable override {
        require(msg.value >= minStake, "Insufficient stake");
        require(nodes[msg.sender].operator == address(0), "Already registered");
        require(bytes(ipfsPeerId).length > 0, "Invalid IPFS peer ID");
        require(bytes(region).length > 0, "Invalid region");

        nodes[msg.sender] = Node({
            operator: msg.sender,
            ipfsPeerId: ipfsPeerId,
            region: region,
            stakedAmount: msg.value,
            registeredAt: block.timestamp,
            lastActiveAt: block.timestamp,
            totalBytesServed: 0,
            successfulRequests: 0,
            failedRequests: 0,
            pogqScore: 10000, // Start with perfect score
            pendingRewards: 0,
            slashCount: 0,
            status: NodeStatus.Active
        });

        operatorIndex[msg.sender] = nodeOperators.length;
        nodeOperators.push(msg.sender);
        totalStaked += msg.value;
        activeNodeCount++;

        emit NodeRegistered(msg.sender, ipfsPeerId, region, msg.value);
    }

    /// @inheritdoc ICdnNode
    function updateNodeMetrics(
        address operator,
        uint256 pogqScore,
        uint256 bytesServed,
        uint256 successful,
        uint256 failed
    ) external override onlyOracle {
        Node storage node = nodes[operator];
        require(node.operator != address(0), "Node not found");
        require(node.status == NodeStatus.Active, "Node not active");

        node.pogqScore = pogqScore;
        node.totalBytesServed += bytesServed;
        node.successfulRequests += successful;
        node.failedRequests += failed;
        node.lastActiveAt = block.timestamp;

        emit NodeUpdated(operator, pogqScore, node.totalBytesServed);
    }

    /// @inheritdoc ICdnNode
    function claimRewards() external override onlyActiveNode {
        Node storage node = nodes[msg.sender];
        uint256 rewards = node.pendingRewards;
        require(rewards > 0, "No pending rewards");

        node.pendingRewards = 0;

        (bool success, ) = msg.sender.call{value: rewards}("");
        require(success, "Transfer failed");

        emit RewardsDistributed(msg.sender, rewards, block.timestamp);
    }

    /// @inheritdoc ICdnNode
    function initiateExit() external override onlyActiveNode {
        Node storage node = nodes[msg.sender];
        node.status = NodeStatus.Exiting;
        exitTimestamps[msg.sender] = block.timestamp;
        activeNodeCount--;
    }

    /// @inheritdoc ICdnNode
    function completeExit() external override {
        Node storage node = nodes[msg.sender];
        require(node.status == NodeStatus.Exiting, "Not exiting");
        require(
            block.timestamp >= exitTimestamps[msg.sender] + unbondingPeriod,
            "Unbonding period not complete"
        );

        uint256 returnAmount = node.stakedAmount + node.pendingRewards;
        node.status = NodeStatus.Inactive;
        node.stakedAmount = 0;
        node.pendingRewards = 0;
        totalStaked -= returnAmount;

        (bool success, ) = msg.sender.call{value: returnAmount}("");
        require(success, "Transfer failed");

        emit NodeExited(msg.sender, returnAmount);
    }

    /// @inheritdoc ICdnNode
    function getNode(address operator) external view override returns (Node memory) {
        return nodes[operator];
    }

    /// @inheritdoc ICdnNode
    function getActiveNodes() external view override returns (address[] memory) {
        address[] memory activeNodes = new address[](activeNodeCount);
        uint256 index = 0;

        for (uint256 i = 0; i < nodeOperators.length; i++) {
            if (nodes[nodeOperators[i]].status == NodeStatus.Active) {
                activeNodes[index] = nodeOperators[i];
                index++;
            }
        }

        return activeNodes;
    }

    /// @notice Get nodes by region
    /// @param region Region code to filter by
    /// @return Array of node addresses in that region
    function getNodesByRegion(string calldata region) external view returns (address[] memory) {
        // First, count matching nodes
        uint256 count = 0;
        for (uint256 i = 0; i < nodeOperators.length; i++) {
            Node storage node = nodes[nodeOperators[i]];
            if (
                node.status == NodeStatus.Active &&
                keccak256(bytes(node.region)) == keccak256(bytes(region))
            ) {
                count++;
            }
        }

        // Then populate array
        address[] memory regionNodes = new address[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < nodeOperators.length; i++) {
            Node storage node = nodes[nodeOperators[i]];
            if (
                node.status == NodeStatus.Active &&
                keccak256(bytes(node.region)) == keccak256(bytes(region))
            ) {
                regionNodes[index] = nodeOperators[i];
                index++;
            }
        }

        return regionNodes;
    }

    /// @notice Get top nodes by PoGQ score
    /// @param count Number of nodes to return
    /// @return Array of top node addresses
    function getTopNodes(uint256 count) external view returns (address[] memory) {
        // Get all active nodes
        address[] memory activeNodes = new address[](activeNodeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < nodeOperators.length; i++) {
            if (nodes[nodeOperators[i]].status == NodeStatus.Active) {
                activeNodes[index] = nodeOperators[i];
                index++;
            }
        }

        // Simple bubble sort by PoGQ score (fine for small sets)
        for (uint256 i = 0; i < activeNodes.length; i++) {
            for (uint256 j = i + 1; j < activeNodes.length; j++) {
                if (nodes[activeNodes[j]].pogqScore > nodes[activeNodes[i]].pogqScore) {
                    address temp = activeNodes[i];
                    activeNodes[i] = activeNodes[j];
                    activeNodes[j] = temp;
                }
            }
        }

        // Return top N
        uint256 resultCount = count < activeNodes.length ? count : activeNodes.length;
        address[] memory topNodes = new address[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            topNodes[i] = activeNodes[i];
        }

        return topNodes;
    }

    // ============ Admin Functions ============

    /// @notice Set the oracle address
    /// @param _oracle New oracle address
    function setOracle(address _oracle) external onlyOwner {
        oracle = _oracle;
    }

    /// @notice Set minimum stake
    /// @param _minStake New minimum stake
    function setMinStake(uint256 _minStake) external onlyOwner {
        minStake = _minStake;
    }

    /// @notice Set unbonding period
    /// @param _period New unbonding period
    function setUnbondingPeriod(uint256 _period) external onlyOwner {
        unbondingPeriod = _period;
    }

    /// @notice Add rewards to a node (called by rewards contract)
    /// @param operator Node operator
    /// @param amount Reward amount
    function addRewards(address operator, uint256 amount) external {
        // In production, this would be restricted to the rewards contract
        nodes[operator].pendingRewards += amount;
    }

    /// @notice Slash a node's stake (called by slashing contract)
    /// @param operator Node operator
    /// @param amount Slash amount
    /// @param reason Reason for slashing
    function slash(address operator, uint256 amount, string calldata reason) external {
        // In production, this would be restricted to the slashing contract
        Node storage node = nodes[operator];
        require(node.stakedAmount >= amount, "Insufficient stake");

        node.stakedAmount -= amount;
        node.slashCount++;
        totalStaked -= amount;

        // If stake falls below minimum, force exit
        if (node.stakedAmount < minStake) {
            node.status = NodeStatus.Slashed;
            activeNodeCount--;
        }

        emit NodeSlashed(operator, amount, reason);
    }

    // ============ Receive ============

    /// @notice Receive xDAI for rewards pool
    receive() external payable {}
}
