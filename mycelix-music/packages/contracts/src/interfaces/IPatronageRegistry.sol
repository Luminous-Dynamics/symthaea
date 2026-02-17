// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IPatronageRegistry
 * @notice Interface for the Mycelix Patronage system
 * @dev Patronage represents ongoing support, not speculative ownership
 *
 * Philosophy: Patronage is a relationship, not an asset. These tokens
 * represent genuine connection between listener and artist, with
 * time-based commitment and anti-speculation mechanisms.
 */
interface IPatronageRegistry {
    /// @notice Patronage commitment levels
    enum PatronageTier {
        Seedling,  // Monthly commitment
        Sapling,   // Quarterly commitment
        Grove,     // Annual commitment
        Ancient    // Multi-year bond
    }

    /// @notice Patronage record for an artist-patron relationship
    struct Patronage {
        address patron;
        address artist;
        PatronageTier tier;
        uint256 startTime;
        uint256 commitment; // Amount committed per period
        uint256 totalContributed;
        uint256 sporesEarned;
        bool active;
    }

    /// @notice Emitted when a new patronage relationship begins
    event PatronageStarted(
        address indexed patron,
        address indexed artist,
        PatronageTier tier,
        uint256 commitment
    );

    /// @notice Emitted when patronage is renewed
    event PatronageRenewed(
        address indexed patron,
        address indexed artist,
        uint256 period,
        uint256 amount
    );

    /// @notice Emitted when patronage tier is upgraded
    event PatronageUpgraded(
        address indexed patron,
        address indexed artist,
        PatronageTier fromTier,
        PatronageTier toTier
    );

    /// @notice Emitted when patronage ends (not transferred - relationships can't be sold)
    event PatronageEnded(
        address indexed patron,
        address indexed artist,
        uint256 totalContributed,
        uint256 duration
    );

    /**
     * @notice Begin a patronage relationship with an artist
     * @param artist The artist to support
     * @param tier The commitment level
     * @param periods Number of periods to commit upfront
     */
    function startPatronage(
        address artist,
        PatronageTier tier,
        uint256 periods
    ) external payable;

    /**
     * @notice Renew an existing patronage
     * @param artist The artist being supported
     * @param periods Additional periods to commit
     */
    function renewPatronage(address artist, uint256 periods) external payable;

    /**
     * @notice Upgrade to a higher commitment tier
     * @param artist The artist being supported
     * @param newTier The new commitment level
     */
    function upgradeTier(address artist, PatronageTier newTier) external payable;

    /**
     * @notice Get patronage details
     * @param patron The supporter address
     * @param artist The artist address
     */
    function getPatronage(address patron, address artist)
        external
        view
        returns (Patronage memory);

    /**
     * @notice Get all patrons for an artist
     * @param artist The artist address
     * @param tier Optional tier filter (or max uint for all)
     */
    function getArtistPatrons(address artist, PatronageTier tier)
        external
        view
        returns (address[] memory);

    /**
     * @notice Get total patronage revenue for an artist
     * @param artist The artist address
     */
    function getArtistPatronageRevenue(address artist)
        external
        view
        returns (uint256);

    /**
     * @notice Check if address is an active patron
     * @param patron The supporter address
     * @param artist The artist address
     */
    function isActivePatron(address patron, address artist)
        external
        view
        returns (bool);

    /**
     * @notice Get the commitment period for a tier
     * @param tier The patronage tier
     */
    function getTierPeriod(PatronageTier tier)
        external
        pure
        returns (uint256);

    /**
     * @notice Get minimum commitment for a tier
     * @param tier The patronage tier
     */
    function getTierMinimum(PatronageTier tier)
        external
        view
        returns (uint256);
}
