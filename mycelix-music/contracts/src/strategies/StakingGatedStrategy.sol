// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "../EconomicStrategyRouter.sol";

/**
 * @title StakingGatedStrategy
 * @notice Stake tokens to access music - earn yield while listening
 * @dev Listeners stake tokens for access, earn rewards, artists get staking income
 */
contract StakingGatedStrategy is IEconomicStrategy, Ownable {
    IERC20 public immutable flowToken;
    IERC20 public immutable stakingToken; // TEND or similar
    address public immutable router;

    struct StakingConfig {
        address artist;
        uint256 minStakeAmount;
        uint256 stakingPeriod; // Minimum staking duration
        uint256 artistRewardBps; // Basis points of staking rewards to artist
        uint256 nonStakerPrice; // Price for non-stakers (0 = not allowed)
        bool initialized;
    }

    struct Stake {
        uint256 amount;
        uint256 startTime;
        uint256 lastRewardTime;
        uint256 rewardsEarned;
    }

    mapping(bytes32 => StakingConfig) public configs;
    mapping(bytes32 => mapping(address => Stake)) public stakes;
    mapping(bytes32 => uint256) public totalStaked;
    mapping(bytes32 => uint256) public artistEarnings;

    // Reward rate: 10% APY in basis points
    uint256 public constant REWARD_RATE_BPS = 1000;

    event StakingConfigured(bytes32 indexed songId, uint256 minStake, uint256 period);
    event Staked(bytes32 indexed songId, address indexed staker, uint256 amount);
    event Unstaked(bytes32 indexed songId, address indexed staker, uint256 amount, uint256 rewards);
    event RewardsClaimed(bytes32 indexed songId, address indexed staker, uint256 rewards);

    modifier onlyRouter() {
        require(msg.sender == router, "Only router can call");
        _;
    }

    constructor(address _flowToken, address _stakingToken, address _router) {
        flowToken = IERC20(_flowToken);
        stakingToken = IERC20(_stakingToken);
        router = _router;
    }

    function configureStaking(
        bytes32 songId,
        address artist,
        uint256 minStakeAmount,
        uint256 stakingPeriod,
        uint256 artistRewardBps,
        uint256 nonStakerPrice
    ) external {
        require(!configs[songId].initialized, "Already configured");
        require(artist != address(0), "Invalid artist");
        require(minStakeAmount > 0, "Min stake must be > 0");
        require(artistRewardBps <= 5000, "Max 50% to artist");

        configs[songId] = StakingConfig({
            artist: artist,
            minStakeAmount: minStakeAmount,
            stakingPeriod: stakingPeriod,
            artistRewardBps: artistRewardBps,
            nonStakerPrice: nonStakerPrice,
            initialized: true
        });

        emit StakingConfigured(songId, minStakeAmount, stakingPeriod);
    }

    function stake(bytes32 songId, uint256 amount) external {
        StakingConfig storage config = configs[songId];
        require(config.initialized, "Not configured");
        require(amount >= config.minStakeAmount, "Below minimum stake");

        Stake storage userStake = stakes[songId][msg.sender];

        // Claim pending rewards if restaking
        if (userStake.amount > 0) {
            _claimRewards(songId, msg.sender);
        }

        require(stakingToken.transferFrom(msg.sender, address(this), amount), "Transfer failed");

        userStake.amount += amount;
        userStake.startTime = block.timestamp;
        userStake.lastRewardTime = block.timestamp;

        totalStaked[songId] += amount;

        emit Staked(songId, msg.sender, amount);
    }

    function unstake(bytes32 songId) external {
        StakingConfig storage config = configs[songId];
        Stake storage userStake = stakes[songId][msg.sender];

        require(userStake.amount > 0, "No stake");
        require(
            block.timestamp >= userStake.startTime + config.stakingPeriod,
            "Staking period not complete"
        );

        uint256 rewards = _claimRewards(songId, msg.sender);
        uint256 amount = userStake.amount;

        totalStaked[songId] -= amount;
        userStake.amount = 0;

        require(stakingToken.transfer(msg.sender, amount), "Transfer failed");

        emit Unstaked(songId, msg.sender, amount, rewards);
    }

    function claimRewards(bytes32 songId) external {
        _claimRewards(songId, msg.sender);
    }

    function _claimRewards(bytes32 songId, address staker) internal returns (uint256) {
        StakingConfig storage config = configs[songId];
        Stake storage userStake = stakes[songId][staker];

        if (userStake.amount == 0) return 0;

        uint256 timeElapsed = block.timestamp - userStake.lastRewardTime;
        uint256 rewards = (userStake.amount * REWARD_RATE_BPS * timeElapsed) / (10000 * 365 days);

        if (rewards > 0) {
            userStake.lastRewardTime = block.timestamp;
            userStake.rewardsEarned += rewards;

            // Split rewards between staker and artist
            uint256 artistShare = (rewards * config.artistRewardBps) / 10000;
            uint256 stakerShare = rewards - artistShare;

            // Note: In production, rewards would come from a rewards pool
            artistEarnings[songId] += artistShare;

            emit RewardsClaimed(songId, staker, stakerShare);
        }

        return rewards;
    }

    function processPayment(
        bytes32 songId,
        address listener,
        uint256 amount,
        PaymentType paymentType
    ) external override onlyRouter {
        StakingConfig storage config = configs[songId];
        require(config.initialized, "Not configured");

        Stake storage userStake = stakes[songId][listener];

        if (userStake.amount >= config.minStakeAmount) {
            // Free access for stakers
            return;
        }

        // Non-staker must pay
        require(config.nonStakerPrice > 0, "Staking required");
        require(amount >= config.nonStakerPrice, "Insufficient payment");

        require(flowToken.transferFrom(router, config.artist, amount), "Transfer failed");
        artistEarnings[songId] += amount;

        paymentType; // Silence unused warning
    }

    function getMinPayment(bytes32 songId, PaymentType) external view override returns (uint256) {
        StakingConfig storage config = configs[songId];
        return config.nonStakerPrice;
    }

    function isAuthorized(bytes32 songId, address listener) external view override returns (bool) {
        StakingConfig storage config = configs[songId];
        if (!config.initialized) return false;

        Stake storage userStake = stakes[songId][listener];
        if (userStake.amount >= config.minStakeAmount) return true;

        return config.nonStakerPrice > 0;
    }

    function calculateSplits(bytes32 songId, uint256)
        external view override returns (EconomicStrategyRouter.Split[] memory)
    {
        StakingConfig storage config = configs[songId];

        EconomicStrategyRouter.Split[] memory splits = new EconomicStrategyRouter.Split[](1);
        splits[0] = EconomicStrategyRouter.Split({
            recipient: config.artist,
            basisPoints: 10000,
            role: "Artist"
        });

        return splits;
    }

    function getStakeInfo(bytes32 songId, address staker)
        external view returns (Stake memory, uint256 pendingRewards)
    {
        Stake storage userStake = stakes[songId][staker];
        uint256 timeElapsed = block.timestamp - userStake.lastRewardTime;
        pendingRewards = (userStake.amount * REWARD_RATE_BPS * timeElapsed) / (10000 * 365 days);
        return (userStake, pendingRewards);
    }
}
