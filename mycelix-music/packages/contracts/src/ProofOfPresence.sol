// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "./interfaces/IProofOfPresence.sol";

/**
 * @title ProofOfPresence
 * @notice Soulbound tokens that commemorate shared musical experiences
 * @dev These tokens are non-transferable - they can only be earned by being present
 *
 * Philosophy: Presence is the gift of attention. These tokens capture moments
 * when listeners truly showed up - at concerts, listening parties, album drops,
 * and artist milestones. They cannot be bought or sold, only earned.
 */
contract ProofOfPresence is IProofOfPresence, AccessControl {
    using Counters for Counters.Counter;
    using Strings for uint256;

    // ========================================================================
    // Roles
    // ========================================================================

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant RESONANCE_BOOSTER_ROLE = keccak256("RESONANCE_BOOSTER_ROLE");

    // ========================================================================
    // State
    // ========================================================================

    Counters.Counter private _tokenIdCounter;

    // Token ID to Presence mapping
    mapping(uint256 => Presence) private _presences;

    // Holder to token IDs mapping
    mapping(address => uint256[]) private _holderTokens;

    // Event ID to token IDs mapping
    mapping(uint256 => uint256[]) private _eventTokens;

    // Artist ID to holder to token IDs mapping
    mapping(string => mapping(address => uint256[])) private _artistPresences;

    // Token ID to unlocked abilities
    mapping(uint256 => string[]) private _tokenAbilities;

    // Base URI for metadata
    string private _baseTokenURI;

    // ========================================================================
    // Constructor
    // ========================================================================

    constructor(string memory baseUri) {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(RESONANCE_BOOSTER_ROLE, msg.sender);
        _baseTokenURI = baseUri;
    }

    // ========================================================================
    // Minting
    // ========================================================================

    /**
     * @notice Mints a Proof of Presence to a participant
     */
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
    ) external onlyRole(MINTER_ROLE) returns (uint256 tokenId) {
        _tokenIdCounter.increment();
        tokenId = _tokenIdCounter.current();

        // Calculate initial resonance based on duration and presence type
        uint256 resonance = _calculateInitialResonance(presenceType, duration);

        // Generate experience hash
        bytes32 experienceHash = keccak256(abi.encodePacked(
            holder,
            presenceType,
            eventId,
            block.timestamp,
            duration
        ));

        _presences[tokenId] = Presence({
            id: tokenId,
            holder: holder,
            presenceType: presenceType,
            eventId: eventId,
            eventName: eventName,
            timestamp: block.timestamp,
            duration: duration,
            artistId: artistId,
            collaboratorIds: collaboratorIds,
            location: location,
            experienceHash: experienceHash,
            metadataUri: metadataUri,
            resonanceLevel: resonance
        });

        // Update indices
        _holderTokens[holder].push(tokenId);
        _eventTokens[eventId].push(tokenId);
        _artistPresences[artistId][holder].push(tokenId);

        // Unlock default abilities based on presence type
        _unlockDefaultAbilities(tokenId, presenceType, resonance);

        emit PresenceMinted(tokenId, holder, presenceType, eventId, artistId, resonance);

        return tokenId;
    }

    /**
     * @notice Batch mint presence tokens for multiple participants
     */
    function batchMintPresence(
        address[] calldata holders,
        PresenceType presenceType,
        uint256 eventId,
        string calldata eventName,
        uint256[] calldata durations,
        string calldata artistId,
        string calldata metadataUri
    ) external onlyRole(MINTER_ROLE) returns (uint256[] memory tokenIds) {
        require(holders.length == durations.length, "Length mismatch");

        tokenIds = new uint256[](holders.length);
        string[] memory empty = new string[](0);

        for (uint256 i = 0; i < holders.length; i++) {
            tokenIds[i] = this.mintPresence(
                holders[i],
                presenceType,
                eventId,
                eventName,
                durations[i],
                artistId,
                empty,
                "",
                metadataUri
            );
        }

        return tokenIds;
    }

    // ========================================================================
    // Resonance
    // ========================================================================

    /**
     * @notice Boost the resonance level of a presence (can only increase)
     */
    function boostResonance(
        uint256 tokenId,
        uint256 additionalResonance
    ) external onlyRole(RESONANCE_BOOSTER_ROLE) {
        require(_presences[tokenId].id != 0, "Token does not exist");

        uint256 previousLevel = _presences[tokenId].resonanceLevel;
        uint256 newLevel = previousLevel + additionalResonance;

        // Cap at 100
        if (newLevel > 100) newLevel = 100;

        _presences[tokenId].resonanceLevel = newLevel;

        // Check for ability unlocks at new resonance levels
        _checkResonanceAbilities(tokenId, previousLevel, newLevel);

        emit ResonanceBoosted(tokenId, previousLevel, newLevel);
    }

    // ========================================================================
    // View Functions
    // ========================================================================

    /**
     * @notice Get a presence token by ID
     */
    function getPresence(uint256 tokenId) external view returns (Presence memory) {
        require(_presences[tokenId].id != 0, "Token does not exist");
        return _presences[tokenId];
    }

    /**
     * @notice Get all presence tokens for a holder
     */
    function getPresencesByHolder(address holder) external view returns (Presence[] memory) {
        uint256[] memory tokenIds = _holderTokens[holder];
        Presence[] memory presences = new Presence[](tokenIds.length);

        for (uint256 i = 0; i < tokenIds.length; i++) {
            presences[i] = _presences[tokenIds[i]];
        }

        return presences;
    }

    /**
     * @notice Get all presence tokens for an event
     */
    function getPresencesByEvent(uint256 eventId) external view returns (Presence[] memory) {
        uint256[] memory tokenIds = _eventTokens[eventId];
        Presence[] memory presences = new Presence[](tokenIds.length);

        for (uint256 i = 0; i < tokenIds.length; i++) {
            presences[i] = _presences[tokenIds[i]];
        }

        return presences;
    }

    /**
     * @notice Get presence tokens for a holder with a specific artist
     */
    function getPresencesByArtist(
        address holder,
        string calldata artistId
    ) external view returns (Presence[] memory) {
        uint256[] memory tokenIds = _artistPresences[artistId][holder];
        Presence[] memory presences = new Presence[](tokenIds.length);

        for (uint256 i = 0; i < tokenIds.length; i++) {
            presences[i] = _presences[tokenIds[i]];
        }

        return presences;
    }

    /**
     * @notice Calculate total resonance score for a holder
     */
    function getTotalResonance(address holder) external view returns (uint256) {
        uint256[] memory tokenIds = _holderTokens[holder];
        uint256 total = 0;

        for (uint256 i = 0; i < tokenIds.length; i++) {
            total += _presences[tokenIds[i]].resonanceLevel;
        }

        return total;
    }

    /**
     * @notice Check if holder has a specific type of presence
     */
    function hasPresenceType(
        address holder,
        PresenceType presenceType
    ) external view returns (bool) {
        uint256[] memory tokenIds = _holderTokens[holder];

        for (uint256 i = 0; i < tokenIds.length; i++) {
            if (_presences[tokenIds[i]].presenceType == presenceType) {
                return true;
            }
        }

        return false;
    }

    /**
     * @notice Count how many times holder attended an artist's events
     */
    function getArtistPresenceCount(
        address holder,
        string calldata artistId
    ) external view returns (uint256) {
        return _artistPresences[artistId][holder].length;
    }

    /**
     * @notice Check what abilities/perks a presence unlocks
     */
    function getUnlockedAbilities(uint256 tokenId) external view returns (string[] memory) {
        return _tokenAbilities[tokenId];
    }

    /**
     * @notice Verify authenticity of a presence token
     */
    function verifyPresence(
        uint256 tokenId,
        bytes32 experienceHash
    ) external view returns (bool) {
        if (_presences[tokenId].id == 0) return false;
        return _presences[tokenId].experienceHash == experienceHash;
    }

    // ========================================================================
    // Soulbound Enforcement
    // ========================================================================

    /**
     * @notice SOULBOUND: This function always reverts
     */
    function transferFrom(address, address, uint256) external pure {
        revert("Presence tokens are soulbound and cannot be transferred");
    }

    /**
     * @notice SOULBOUND: This function always reverts
     */
    function approve(address, uint256) external pure {
        revert("Presence tokens are soulbound and cannot be approved");
    }

    // ========================================================================
    // Internal Functions
    // ========================================================================

    /**
     * @dev Calculate initial resonance based on presence type and duration
     */
    function _calculateInitialResonance(
        PresenceType presenceType,
        uint256 duration
    ) internal pure returns (uint256) {
        uint256 baseResonance;

        // Base resonance by type
        if (presenceType == PresenceType.LIVE_CONCERT) {
            baseResonance = 50;
        } else if (presenceType == PresenceType.VIRTUAL_CONCERT) {
            baseResonance = 35;
        } else if (presenceType == PresenceType.ALBUM_RELEASE) {
            baseResonance = 40;
        } else if (presenceType == PresenceType.FIRST_LISTEN) {
            baseResonance = 45;
        } else if (presenceType == PresenceType.LISTENING_CIRCLE) {
            baseResonance = 30;
        } else if (presenceType == PresenceType.ARTIST_MILESTONE) {
            baseResonance = 40;
        } else if (presenceType == PresenceType.COLLABORATIVE_SESSION) {
            baseResonance = 55;
        } else if (presenceType == PresenceType.COMMUNITY_EVENT) {
            baseResonance = 35;
        } else if (presenceType == PresenceType.SEASONAL_GATHERING) {
            baseResonance = 45;
        } else if (presenceType == PresenceType.PATRONAGE_RENEWAL) {
            baseResonance = 50;
        } else {
            baseResonance = 25;
        }

        // Duration bonus (up to 25% extra for long presence)
        uint256 durationBonus = 0;
        if (duration > 3600) { // > 1 hour
            durationBonus = 10;
        } else if (duration > 7200) { // > 2 hours
            durationBonus = 20;
        } else if (duration > 14400) { // > 4 hours
            durationBonus = 25;
        }

        uint256 total = baseResonance + durationBonus;
        return total > 100 ? 100 : total;
    }

    /**
     * @dev Unlock default abilities based on presence type
     */
    function _unlockDefaultAbilities(
        uint256 tokenId,
        PresenceType presenceType,
        uint256 resonance
    ) internal {
        string[] storage abilities = _tokenAbilities[tokenId];

        // Type-based abilities
        if (presenceType == PresenceType.LIVE_CONCERT) {
            abilities.push("concert_veteran");
            if (resonance >= 70) abilities.push("backstage_eligible");
        } else if (presenceType == PresenceType.FIRST_LISTEN) {
            abilities.push("early_supporter");
            abilities.push("first_1000_badge");
        } else if (presenceType == PresenceType.COLLABORATIVE_SESSION) {
            abilities.push("contributor_credit");
        } else if (presenceType == PresenceType.ARTIST_MILESTONE) {
            abilities.push("milestone_witness");
        } else if (presenceType == PresenceType.PATRONAGE_RENEWAL) {
            abilities.push("loyal_patron");
        }

        // Resonance-based abilities
        if (resonance >= 80) {
            abilities.push("high_resonance");
        }

        // Emit events for each ability
        for (uint256 i = 0; i < abilities.length; i++) {
            emit PresenceAbilityUnlocked(tokenId, abilities[i], abilities[i]);
        }
    }

    /**
     * @dev Check for new ability unlocks when resonance increases
     */
    function _checkResonanceAbilities(
        uint256 tokenId,
        uint256 previousLevel,
        uint256 newLevel
    ) internal {
        // Unlock abilities at resonance thresholds
        if (previousLevel < 50 && newLevel >= 50) {
            _tokenAbilities[tokenId].push("dedicated_listener");
            emit PresenceAbilityUnlocked(tokenId, "dedicated_listener", "Reached 50 resonance");
        }

        if (previousLevel < 75 && newLevel >= 75) {
            _tokenAbilities[tokenId].push("deep_connection");
            emit PresenceAbilityUnlocked(tokenId, "deep_connection", "Reached 75 resonance");
        }

        if (previousLevel < 90 && newLevel >= 90) {
            _tokenAbilities[tokenId].push("soul_bond");
            emit PresenceAbilityUnlocked(tokenId, "soul_bond", "Reached 90 resonance");
        }

        if (previousLevel < 100 && newLevel >= 100) {
            _tokenAbilities[tokenId].push("perfect_resonance");
            emit PresenceAbilityUnlocked(tokenId, "perfect_resonance", "Achieved perfect resonance");
        }
    }

    /**
     * @dev Returns the token URI for a given token ID
     */
    function tokenURI(uint256 tokenId) external view returns (string memory) {
        require(_presences[tokenId].id != 0, "Token does not exist");

        if (bytes(_presences[tokenId].metadataUri).length > 0) {
            return _presences[tokenId].metadataUri;
        }

        return string(abi.encodePacked(_baseTokenURI, tokenId.toString()));
    }

    /**
     * @notice Get the total number of presence tokens minted
     */
    function totalSupply() external view returns (uint256) {
        return _tokenIdCounter.current();
    }

    /**
     * @notice Get the number of presence tokens held by an address
     */
    function balanceOf(address holder) external view returns (uint256) {
        return _holderTokens[holder].length;
    }

    /**
     * @notice Set the base URI for token metadata
     */
    function setBaseURI(string calldata baseUri) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _baseTokenURI = baseUri;
    }
}
