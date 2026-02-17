// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ISporeToken
 * @notice Interface for non-transferable engagement tokens
 * @dev Spores represent genuine participation in the ecosystem.
 * They CANNOT be bought, sold, or transferred - only earned through
 * authentic engagement and contribution.
 *
 * Philosophy: Value should flow to those who participate, not those
 * who speculate. Spores decay over time to prevent hoarding and
 * ensure ongoing contribution is required for influence.
 */
interface ISporeToken {
    /// @notice Categories of spore-earning activities
    enum SporeActivity {
        Listening,       // Hours of genuine listening
        EarlyDiscovery,  // Supporting artists before popularity
        PlaylistCuration,// Creating valuable playlists
        CommunityHelp,   // Helping other users
        EventHosting,    // Organizing community events
        ContentCreation, // Reviews, articles, art
        BugReporting,    // Platform improvement
        Governance       // Participating in votes
    }

    /// @notice Spore balance with decay tracking
    struct SporeBalance {
        uint256 total;
        uint256 lastActivity;
        uint256 decayRate;
        uint256 tier; // Calculated from total (determines governance weight)
    }

    /// @notice Emitted when spores are earned
    event SporesEarned(
        address indexed user,
        SporeActivity activity,
        uint256 amount,
        string description
    );

    /// @notice Emitted when spores decay due to inactivity
    event SporesDecayed(
        address indexed user,
        uint256 decayedAmount,
        uint256 remaining
    );

    /// @notice Emitted when governance tier changes
    event TierChanged(
        address indexed user,
        uint256 fromTier,
        uint256 toTier
    );

    /**
     * @notice Award spores for an activity (only callable by authorized contracts)
     * @param user The recipient
     * @param activity The type of activity
     * @param amount Base amount (may be modified by multipliers)
     * @param description Human-readable description
     */
    function awardSpores(
        address user,
        SporeActivity activity,
        uint256 amount,
        string calldata description
    ) external;

    /**
     * @notice Get current spore balance after decay
     * @param user The user to check
     */
    function balanceOf(address user) external view returns (uint256);

    /**
     * @notice Get detailed spore info
     * @param user The user to check
     */
    function getSporeInfo(address user) external view returns (SporeBalance memory);

    /**
     * @notice Get governance voting weight
     * @param user The user to check
     */
    function getVotingWeight(address user) external view returns (uint256);

    /**
     * @notice Get tier thresholds
     * @param tier The tier level (1-5)
     */
    function getTierThreshold(uint256 tier) external view returns (uint256);

    /**
     * @notice Calculate decay amount for a user
     * @param user The user to check
     */
    function calculateDecay(address user) external view returns (uint256);

    /**
     * @notice Apply decay (can be called by anyone, gas refunded)
     * @param user The user to decay
     */
    function applyDecay(address user) external;

    /**
     * @notice Check if user can perform governance action
     * @param user The user to check
     * @param requiredTier Minimum tier needed
     */
    function canParticipateInGovernance(address user, uint256 requiredTier)
        external
        view
        returns (bool);

    /**
     * @notice Get activity multiplier (some activities worth more)
     * @param activity The activity type
     */
    function getActivityMultiplier(SporeActivity activity)
        external
        view
        returns (uint256);

    // Note: No transfer functions! Spores are soulbound.
}
