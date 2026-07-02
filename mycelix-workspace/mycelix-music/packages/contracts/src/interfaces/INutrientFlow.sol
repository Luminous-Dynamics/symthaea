// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title INutrientFlow
 * @notice Interface for the regenerative value distribution system
 * @dev Like mycelium distributing nutrients through a forest, this contract
 * ensures value flows to where it's needed most in the ecosystem.
 *
 * Philosophy: In nature, old-growth trees support seedlings through nutrient
 * sharing. Similarly, successful artists automatically contribute to emerging
 * artist development, creating a self-sustaining creative ecosystem.
 */
interface INutrientFlow {
    /// @notice Distribution allocation for a transaction
    struct Distribution {
        uint256 artistShare;        // Primary artist (85%)
        uint256 collaboratorShare;  // All collaborators (10%)
        uint256 treasuryShare;      // Community treasury (5%)
    }

    /// @notice Treasury allocation breakdown
    struct TreasuryAllocation {
        uint256 emergingArtistFund;   // 40% - Support new artists
        uint256 educationGrants;       // 20% - Learning resources
        uint256 infrastructureFund;    // 20% - Platform development
        uint256 communityEvents;       // 10% - Community building
        uint256 reserve;               // 10% - Emergency/opportunity
    }

    /// @notice Collaborator credit on a work
    struct Collaborator {
        address wallet;
        uint256 splitBps;  // Basis points (100 = 1%)
        string role;       // "producer", "engineer", "vocalist", etc.
    }

    /// @notice Emitted when revenue is distributed
    event RevenueDistributed(
        bytes32 indexed songId,
        uint256 totalAmount,
        address artist,
        uint256 artistAmount,
        uint256 collaboratorAmount,
        uint256 treasuryAmount
    );

    /// @notice Emitted when collaborator receives share
    event CollaboratorPaid(
        bytes32 indexed songId,
        address indexed collaborator,
        uint256 amount,
        string role
    );

    /// @notice Emitted when treasury receives funds
    event TreasuryFunded(
        uint256 amount,
        uint256 emergingFund,
        uint256 education,
        uint256 infrastructure,
        uint256 community,
        uint256 reserve
    );

    /// @notice Emitted when emerging artist receives grant
    event EmergingArtistGrant(
        address indexed artist,
        uint256 amount,
        string reason
    );

    /**
     * @notice Distribute revenue for a song
     * @param songId The song identifier
     * @param amount Total revenue to distribute
     */
    function distributeRevenue(bytes32 songId, uint256 amount) external;

    /**
     * @notice Register collaborators for a song
     * @param songId The song identifier
     * @param collaborators Array of collaborator credits
     */
    function registerCollaborators(
        bytes32 songId,
        Collaborator[] calldata collaborators
    ) external;

    /**
     * @notice Get current treasury balances
     */
    function getTreasuryBalances() external view returns (TreasuryAllocation memory);

    /**
     * @notice Get emerging artist fund balance
     */
    function getEmergingArtistFund() external view returns (uint256);

    /**
     * @notice Apply for emerging artist grant
     * @param amount Requested amount
     * @param proposal Description of how funds will be used
     */
    function applyForGrant(uint256 amount, string calldata proposal) external;

    /**
     * @notice Vote on grant application (requires Spore tokens)
     * @param applicationId The grant application
     * @param support True to approve, false to reject
     */
    function voteOnGrant(uint256 applicationId, bool support) external;

    /**
     * @notice Execute approved grant
     * @param applicationId The approved application
     */
    function executeGrant(uint256 applicationId) external;

    /**
     * @notice Get song collaborators
     * @param songId The song to query
     */
    function getCollaborators(bytes32 songId)
        external
        view
        returns (Collaborator[] memory);

    /**
     * @notice Calculate distribution for amount
     * @param amount The revenue amount
     */
    function calculateDistribution(uint256 amount)
        external
        pure
        returns (Distribution memory);

    /**
     * @notice Get total revenue distributed through system
     */
    function getTotalDistributed() external view returns (uint256);

    /**
     * @notice Get total grants given to emerging artists
     */
    function getTotalGrantsGiven() external view returns (uint256);
}
