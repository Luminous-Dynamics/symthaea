// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IProofOfPresence
 * @notice Non-transferable tokens (Soulbound) that commemorate shared musical experiences
 * @dev These tokens cannot be transferred or sold - they represent authentic moments
 *
 * Philosophy: Presence is the gift of attention. These tokens capture moments
 * when listeners truly showed up - at concerts, listening parties, album drops,
 * and artist milestones. Unlike traditional NFTs, these cannot be bought or sold.
 * They can only be earned by being present.
 */
interface IProofOfPresence {
    /// @notice Types of presence that can be commemorated
    enum PresenceType {
        LIVE_CONCERT,           // Physical concert attendance
        VIRTUAL_CONCERT,        // Virtual/streamed concert
        ALBUM_RELEASE,          // Present at album release moment
        FIRST_LISTEN,           // Among first 1000 to hear a song
        LISTENING_CIRCLE,       // Participated in group listening
        ARTIST_MILESTONE,       // Artist hit a milestone (1M plays, etc)
        COLLABORATIVE_SESSION,  // Part of a collaborative creation
        COMMUNITY_EVENT,        // Community gathering/event
        SEASONAL_GATHERING,     // Equinox, solstice celebrations
        PATRONAGE_RENEWAL       // Annual patronage renewal ceremony
    }

    /// @notice A Proof of Presence token
    struct Presence {
        uint256 id;
        address holder;
        PresenceType presenceType;
        uint256 eventId;           // Reference to the specific event
        string eventName;
        uint256 timestamp;
        uint256 duration;          // How long they were present (seconds)
        string artistId;           // Primary artist involved
        string[] collaboratorIds;  // Other artists/participants
        string location;           // Physical or virtual location
        bytes32 experienceHash;    // Unique hash of the experience
        string metadataUri;        // IPFS URI for additional metadata
        uint256 resonanceLevel;    // 1-100, based on engagement
    }

    /// @notice Emitted when a new Proof of Presence is minted
    event PresenceMinted(
        uint256 indexed tokenId,
        address indexed holder,
        PresenceType presenceType,
        uint256 eventId,
        string artistId,
        uint256 resonanceLevel
    );

    /// @notice Emitted when resonance level is boosted (cannot decrease)
    event ResonanceBoosted(
        uint256 indexed tokenId,
        uint256 previousLevel,
        uint256 newLevel
    );

    /// @notice Emitted when a presence unlocks a special ability
    event PresenceAbilityUnlocked(
        uint256 indexed tokenId,
        string abilityId,
        string description
    );

    /// @notice Mints a Proof of Presence to a participant
    /// @param holder The address receiving the proof
    /// @param presenceType The type of presence being commemorated
    /// @param eventId The unique identifier of the event
    /// @param eventName Human-readable name of the event
    /// @param duration How long the holder was present
    /// @param artistId The primary artist's identifier
    /// @param collaboratorIds Other participants in the experience
    /// @param location Where the experience took place
    /// @param metadataUri IPFS URI for additional data
    function mintPresence(
        address holder,
        PresenceType presenceType,
        uint256 eventId,
        string calldata eventName,
        uint256 duration,
        string calldata artistId,
        string[] calldata collaboratorIds,
        string calldata location,
        string calldata metadataUri
    ) external returns (uint256 tokenId);

    /// @notice Batch mint presence tokens for multiple participants
    /// @param holders Array of addresses to receive proofs
    /// @param presenceType The type of presence for all recipients
    /// @param eventId The unique identifier of the event
    /// @param eventName Human-readable name of the event
    /// @param durations Duration for each participant
    /// @param artistId The primary artist's identifier
    /// @param metadataUri IPFS URI for additional data
    function batchMintPresence(
        address[] calldata holders,
        PresenceType presenceType,
        uint256 eventId,
        string calldata eventName,
        uint256[] calldata durations,
        string calldata artistId,
        string calldata metadataUri
    ) external returns (uint256[] memory tokenIds);

    /// @notice Get a presence token by ID
    function getPresence(uint256 tokenId) external view returns (Presence memory);

    /// @notice Get all presence tokens for a holder
    function getPresencesByHolder(address holder) external view returns (Presence[] memory);

    /// @notice Get all presence tokens for an event
    function getPresencesByEvent(uint256 eventId) external view returns (Presence[] memory);

    /// @notice Get presence tokens for a holder with a specific artist
    function getPresencesByArtist(address holder, string calldata artistId)
        external view returns (Presence[] memory);

    /// @notice Calculate total resonance score for a holder
    function getTotalResonance(address holder) external view returns (uint256);

    /// @notice Check if holder has a specific type of presence
    function hasPresenceType(address holder, PresenceType presenceType)
        external view returns (bool);

    /// @notice Count how many times holder attended an artist's events
    function getArtistPresenceCount(address holder, string calldata artistId)
        external view returns (uint256);

    /// @notice Boost the resonance level of a presence (can only increase)
    /// @param tokenId The token to boost
    /// @param additionalResonance The amount to add
    function boostResonance(uint256 tokenId, uint256 additionalResonance) external;

    /// @notice Check what abilities/perks a presence unlocks
    function getUnlockedAbilities(uint256 tokenId) external view returns (string[] memory);

    /// @notice Verify authenticity of a presence token
    function verifyPresence(uint256 tokenId, bytes32 experienceHash)
        external view returns (bool);

    /**
     * @notice SOULBOUND: This function always reverts
     * @dev Presence tokens are non-transferable
     */
    function transferFrom(address from, address to, uint256 tokenId) external;

    /**
     * @notice SOULBOUND: This function always reverts
     * @dev Presence tokens cannot be approved for transfer
     */
    function approve(address to, uint256 tokenId) external;
}

/**
 * @title IPresenceRegistry
 * @notice Registry for managing events that can generate Presence tokens
 */
interface IPresenceRegistry {
    /// @notice An event that can generate presence tokens
    struct PresenceEvent {
        uint256 id;
        string name;
        IProofOfPresence.PresenceType presenceType;
        string artistId;
        uint256 startTime;
        uint256 endTime;
        string location;
        uint256 maxParticipants;
        uint256 currentParticipants;
        bool isActive;
        bytes32 verificationKey;  // For validating attendance
    }

    /// @notice Create a new event
    function createEvent(
        string calldata name,
        IProofOfPresence.PresenceType presenceType,
        string calldata artistId,
        uint256 startTime,
        uint256 endTime,
        string calldata location,
        uint256 maxParticipants
    ) external returns (uint256 eventId);

    /// @notice Register attendance at an event
    function registerAttendance(
        uint256 eventId,
        address attendee,
        bytes calldata proof
    ) external;

    /// @notice Finalize event and mint presence tokens
    function finalizeEvent(uint256 eventId) external;

    /// @notice Get event details
    function getEvent(uint256 eventId) external view returns (PresenceEvent memory);

    /// @notice Get upcoming events for an artist
    function getArtistEvents(string calldata artistId)
        external view returns (PresenceEvent[] memory);
}
