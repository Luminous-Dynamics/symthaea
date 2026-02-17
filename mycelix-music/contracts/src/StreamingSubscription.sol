// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title StreamingSubscription
 * @notice On-chain subscription management for streaming access
 * @dev Handles subscription lifecycle and artist pool distribution
 */
contract StreamingSubscription is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ==================== Structs ====================

    struct Plan {
        string name;
        uint256 price;           // Monthly price
        uint256 duration;        // Duration in seconds
        uint256 streamAllowance; // Max streams per period (0 = unlimited)
        bool active;
        bool exists;
    }

    struct Subscription {
        uint256 planId;
        uint256 startTime;
        uint256 endTime;
        uint256 streamsUsed;
        bool autoRenew;
        bool active;
    }

    struct ArtistPool {
        uint256 totalStreams;
        uint256 totalEarnings;
        uint256 lastDistribution;
    }

    // ==================== State ====================

    /// @notice Payment token
    IERC20 public paymentToken;

    /// @notice Subscription plans
    mapping(uint256 => Plan) public plans;
    uint256 public nextPlanId = 1;

    /// @notice User subscriptions
    mapping(address => Subscription) public subscriptions;

    /// @notice Artist streaming pool
    mapping(address => ArtistPool) public artistPools;

    /// @notice Current period stream counts per artist
    mapping(uint256 => mapping(address => uint256)) public periodStreams;

    /// @notice Current period ID
    uint256 public currentPeriod;

    /// @notice Period duration (default: 30 days)
    uint256 public periodDuration = 30 days;

    /// @notice Period start time
    uint256 public periodStartTime;

    /// @notice Revenue pool for current period
    uint256 public currentPeriodRevenue;

    /// @notice Platform fee
    uint256 public platformFee = 2000; // 20%

    /// @notice Treasury
    address public treasury;

    /// @notice Authorized stream recorders
    mapping(address => bool) public streamRecorders;

    // ==================== Events ====================

    event PlanCreated(uint256 indexed planId, string name, uint256 price, uint256 duration);
    event PlanUpdated(uint256 indexed planId);
    event PlanDeactivated(uint256 indexed planId);

    event Subscribed(
        address indexed user,
        uint256 indexed planId,
        uint256 startTime,
        uint256 endTime
    );

    event SubscriptionRenewed(
        address indexed user,
        uint256 indexed planId,
        uint256 newEndTime
    );

    event SubscriptionCancelled(address indexed user);

    event StreamRecorded(
        address indexed user,
        address indexed artist,
        string songId,
        uint256 period
    );

    event PeriodDistributed(
        uint256 indexed period,
        uint256 totalRevenue,
        uint256 totalStreams,
        uint256 artistCount
    );

    event ArtistPaid(
        address indexed artist,
        uint256 indexed period,
        uint256 amount,
        uint256 streams
    );

    // ==================== Errors ====================

    error PlanNotFound();
    error PlanNotActive();
    error AlreadySubscribed();
    error NotSubscribed();
    error SubscriptionExpired();
    error StreamAllowanceExceeded();
    error Unauthorized();
    error InvalidInput();
    error PeriodNotEnded();

    // ==================== Constructor ====================

    constructor(address _paymentToken, address _treasury) {
        paymentToken = IERC20(_paymentToken);
        treasury = _treasury;
        periodStartTime = block.timestamp;
        currentPeriod = 1;
    }

    // ==================== Plan Management ====================

    /**
     * @notice Create a subscription plan
     * @param name Plan name
     * @param price Monthly price
     * @param duration Duration in seconds
     * @param streamAllowance Max streams (0 = unlimited)
     */
    function createPlan(
        string calldata name,
        uint256 price,
        uint256 duration,
        uint256 streamAllowance
    ) external onlyOwner returns (uint256 planId) {
        planId = nextPlanId++;

        plans[planId] = Plan({
            name: name,
            price: price,
            duration: duration,
            streamAllowance: streamAllowance,
            active: true,
            exists: true
        });

        emit PlanCreated(planId, name, price, duration);
        return planId;
    }

    /**
     * @notice Update a plan
     * @param planId Plan ID
     * @param price New price
     * @param streamAllowance New allowance
     */
    function updatePlan(
        uint256 planId,
        uint256 price,
        uint256 streamAllowance
    ) external onlyOwner {
        Plan storage plan = plans[planId];
        if (!plan.exists) revert PlanNotFound();

        plan.price = price;
        plan.streamAllowance = streamAllowance;

        emit PlanUpdated(planId);
    }

    /**
     * @notice Deactivate a plan
     * @param planId Plan ID
     */
    function deactivatePlan(uint256 planId) external onlyOwner {
        Plan storage plan = plans[planId];
        if (!plan.exists) revert PlanNotFound();

        plan.active = false;

        emit PlanDeactivated(planId);
    }

    // ==================== Subscription Management ====================

    /**
     * @notice Subscribe to a plan
     * @param planId Plan ID
     * @param autoRenew Auto-renew flag
     */
    function subscribe(uint256 planId, bool autoRenew) external nonReentrant {
        Plan storage plan = plans[planId];
        if (!plan.exists) revert PlanNotFound();
        if (!plan.active) revert PlanNotActive();

        Subscription storage sub = subscriptions[msg.sender];
        if (sub.active && sub.endTime > block.timestamp) {
            revert AlreadySubscribed();
        }

        // Take payment
        paymentToken.safeTransferFrom(msg.sender, address(this), plan.price);

        // Add to period revenue (after platform fee)
        uint256 fee = (plan.price * platformFee) / 10000;
        uint256 poolAmount = plan.price - fee;

        paymentToken.safeTransfer(treasury, fee);
        currentPeriodRevenue += poolAmount;

        // Create subscription
        uint256 startTime = block.timestamp;
        uint256 endTime = startTime + plan.duration;

        subscriptions[msg.sender] = Subscription({
            planId: planId,
            startTime: startTime,
            endTime: endTime,
            streamsUsed: 0,
            autoRenew: autoRenew,
            active: true
        });

        emit Subscribed(msg.sender, planId, startTime, endTime);
    }

    /**
     * @notice Renew subscription
     */
    function renew() external nonReentrant {
        Subscription storage sub = subscriptions[msg.sender];
        if (!sub.active) revert NotSubscribed();

        Plan storage plan = plans[sub.planId];
        if (!plan.exists || !plan.active) revert PlanNotActive();

        // Take payment
        paymentToken.safeTransferFrom(msg.sender, address(this), plan.price);

        // Add to period revenue
        uint256 fee = (plan.price * platformFee) / 10000;
        uint256 poolAmount = plan.price - fee;

        paymentToken.safeTransfer(treasury, fee);
        currentPeriodRevenue += poolAmount;

        // Extend subscription
        uint256 newEndTime = (sub.endTime > block.timestamp)
            ? sub.endTime + plan.duration
            : block.timestamp + plan.duration;

        sub.endTime = newEndTime;
        sub.streamsUsed = 0;

        emit SubscriptionRenewed(msg.sender, sub.planId, newEndTime);
    }

    /**
     * @notice Cancel auto-renewal
     */
    function cancelAutoRenew() external {
        Subscription storage sub = subscriptions[msg.sender];
        if (!sub.active) revert NotSubscribed();

        sub.autoRenew = false;

        emit SubscriptionCancelled(msg.sender);
    }

    /**
     * @notice Check if user has active subscription
     * @param user User address
     */
    function isSubscribed(address user) public view returns (bool) {
        Subscription storage sub = subscriptions[user];
        return sub.active && sub.endTime > block.timestamp;
    }

    /**
     * @notice Get subscription details
     * @param user User address
     */
    function getSubscription(address user) external view returns (Subscription memory) {
        return subscriptions[user];
    }

    // ==================== Stream Recording ====================

    /**
     * @notice Set stream recorder authorization
     * @param recorder Recorder address
     * @param authorized Authorization status
     */
    function setStreamRecorder(address recorder, bool authorized) external onlyOwner {
        streamRecorders[recorder] = authorized;
    }

    /**
     * @notice Record a stream (called by authorized backend)
     * @param user User address
     * @param artist Artist address
     * @param songId Song ID
     */
    function recordStream(
        address user,
        address artist,
        string calldata songId
    ) external {
        if (!streamRecorders[msg.sender]) revert Unauthorized();
        if (!isSubscribed(user)) revert NotSubscribed();

        Subscription storage sub = subscriptions[user];
        Plan storage plan = plans[sub.planId];

        // Check stream allowance
        if (plan.streamAllowance > 0 && sub.streamsUsed >= plan.streamAllowance) {
            revert StreamAllowanceExceeded();
        }

        // Increment counters
        sub.streamsUsed++;
        periodStreams[currentPeriod][artist]++;
        artistPools[artist].totalStreams++;

        emit StreamRecorded(user, artist, songId, currentPeriod);
    }

    /**
     * @notice Batch record streams
     * @param users User addresses
     * @param artists Artist addresses
     * @param songIds Song IDs
     */
    function batchRecordStreams(
        address[] calldata users,
        address[] calldata artists,
        string[] calldata songIds
    ) external {
        if (!streamRecorders[msg.sender]) revert Unauthorized();
        if (users.length != artists.length || users.length != songIds.length) {
            revert InvalidInput();
        }

        for (uint256 i = 0; i < users.length; i++) {
            address user = users[i];
            address artist = artists[i];

            if (!isSubscribed(user)) continue;

            Subscription storage sub = subscriptions[user];
            Plan storage plan = plans[sub.planId];

            if (plan.streamAllowance > 0 && sub.streamsUsed >= plan.streamAllowance) {
                continue;
            }

            sub.streamsUsed++;
            periodStreams[currentPeriod][artist]++;
            artistPools[artist].totalStreams++;

            emit StreamRecorded(user, artist, songIds[i], currentPeriod);
        }
    }

    // ==================== Distribution ====================

    /**
     * @notice Distribute period revenue to artists
     * @param artists Artists to distribute to
     */
    function distributePeriodRevenue(address[] calldata artists) external nonReentrant {
        if (block.timestamp < periodStartTime + periodDuration) {
            revert PeriodNotEnded();
        }

        uint256 period = currentPeriod;
        uint256 totalRevenue = currentPeriodRevenue;

        if (totalRevenue == 0 || artists.length == 0) {
            // Move to next period even if no revenue
            _advancePeriod();
            return;
        }

        // Calculate total streams for this period
        uint256 totalStreams = 0;
        for (uint256 i = 0; i < artists.length; i++) {
            totalStreams += periodStreams[period][artists[i]];
        }

        if (totalStreams == 0) {
            _advancePeriod();
            return;
        }

        // Distribute proportionally
        uint256 distributed = 0;
        for (uint256 i = 0; i < artists.length; i++) {
            address artist = artists[i];
            uint256 artistStreams = periodStreams[period][artist];

            if (artistStreams == 0) continue;

            uint256 artistShare = (totalRevenue * artistStreams) / totalStreams;

            if (artistShare > 0) {
                paymentToken.safeTransfer(artist, artistShare);
                artistPools[artist].totalEarnings += artistShare;
                artistPools[artist].lastDistribution = block.timestamp;
                distributed += artistShare;

                emit ArtistPaid(artist, period, artistShare, artistStreams);
            }
        }

        emit PeriodDistributed(period, distributed, totalStreams, artists.length);

        _advancePeriod();
    }

    /**
     * @notice Advance to next period
     */
    function _advancePeriod() internal {
        currentPeriod++;
        periodStartTime = block.timestamp;
        currentPeriodRevenue = 0;
    }

    // ==================== View Functions ====================

    /**
     * @notice Get plan details
     * @param planId Plan ID
     */
    function getPlan(uint256 planId) external view returns (Plan memory) {
        return plans[planId];
    }

    /**
     * @notice Get artist pool data
     * @param artist Artist address
     */
    function getArtistPool(address artist) external view returns (ArtistPool memory) {
        return artistPools[artist];
    }

    /**
     * @notice Get artist streams for current period
     * @param artist Artist address
     */
    function getArtistPeriodStreams(address artist) external view returns (uint256) {
        return periodStreams[currentPeriod][artist];
    }

    /**
     * @notice Get current period info
     */
    function getCurrentPeriodInfo()
        external
        view
        returns (
            uint256 period,
            uint256 startTime,
            uint256 endTime,
            uint256 revenue
        )
    {
        return (
            currentPeriod,
            periodStartTime,
            periodStartTime + periodDuration,
            currentPeriodRevenue
        );
    }

    // ==================== Admin Functions ====================

    /**
     * @notice Update platform fee
     * @param newFee New fee in basis points
     */
    function setPlatformFee(uint256 newFee) external onlyOwner {
        if (newFee > 5000) revert InvalidInput(); // Max 50%
        platformFee = newFee;
    }

    /**
     * @notice Update period duration
     * @param newDuration New duration in seconds
     */
    function setPeriodDuration(uint256 newDuration) external onlyOwner {
        if (newDuration < 1 days) revert InvalidInput();
        periodDuration = newDuration;
    }

    /**
     * @notice Update treasury
     * @param newTreasury New treasury address
     */
    function setTreasury(address newTreasury) external onlyOwner {
        treasury = newTreasury;
    }

    /**
     * @notice Emergency withdraw
     * @param token Token address
     * @param amount Amount
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
    }
}
