// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/**
 * @title SoulRegistry
 * @notice On-chain registry for Mycelix Soul identities
 * @dev Souls are soulbound (non-transferable) and represent authentic presence
 *
 * Philosophy:
 * - One address, one soul (enforced on-chain)
 * - Souls cannot be transferred or sold
 * - Resonance accumulates through genuine engagement
 * - Credentials are issued by trusted attesters
 */
contract SoulRegistry is AccessControl, EIP712 {
    using Counters for Counters.Counter;
    using ECDSA for bytes32;

    // ========================================================================
    // Roles
    // ========================================================================

    bytes32 public constant ATTESTER_ROLE = keccak256("ATTESTER_ROLE");
    bytes32 public constant RESONANCE_MANAGER_ROLE = keccak256("RESONANCE_MANAGER_ROLE");

    // ========================================================================
    // Types
    // ========================================================================

    struct Soul {
        uint256 id;
        address owner;
        string profileUri;      // IPFS URI for off-chain profile data
        uint256 totalResonance;
        uint256 connections;
        uint256 bornAt;
        bool exists;
    }

    struct Credential {
        bytes32 credentialType;
        address issuer;
        uint256 issuedAt;
        uint256 expiresAt;      // 0 = never expires
        bytes32 dataHash;       // Hash of credential data
        bool revoked;
    }

    // ========================================================================
    // State
    // ========================================================================

    Counters.Counter private _soulIdCounter;

    // Address to Soul mapping
    mapping(address => Soul) public souls;

    // Soul ID to address mapping (for lookups)
    mapping(uint256 => address) public soulOwners;

    // Credentials: soulId => credentialType => Credential
    mapping(uint256 => mapping(bytes32 => Credential)) public credentials;

    // Credential types held by a soul
    mapping(uint256 => bytes32[]) public soulCredentialTypes;

    // Connections between souls: soulId => connectedSoulId => exists
    mapping(uint256 => mapping(uint256 => bool)) public connections;

    // ========================================================================
    // Events
    // ========================================================================

    event SoulCreated(
        uint256 indexed soulId,
        address indexed owner,
        string profileUri,
        uint256 timestamp
    );

    event ProfileUpdated(
        uint256 indexed soulId,
        string newProfileUri
    );

    event ResonanceAdded(
        uint256 indexed soulId,
        uint256 amount,
        string source,
        uint256 newTotal
    );

    event CredentialIssued(
        uint256 indexed soulId,
        bytes32 indexed credentialType,
        address indexed issuer,
        uint256 expiresAt
    );

    event CredentialRevoked(
        uint256 indexed soulId,
        bytes32 indexed credentialType,
        address indexed revoker
    );

    event ConnectionFormed(
        uint256 indexed soulId1,
        uint256 indexed soulId2,
        uint256 timestamp
    );

    // ========================================================================
    // Errors
    // ========================================================================

    error SoulAlreadyExists();
    error SoulDoesNotExist();
    error NotSoulOwner();
    error CredentialAlreadyExists();
    error CredentialDoesNotExist();
    error CredentialExpired();
    error ConnectionAlreadyExists();
    error CannotConnectToSelf();

    // ========================================================================
    // Constructor
    // ========================================================================

    constructor() EIP712("MycelixSoulRegistry", "1") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ATTESTER_ROLE, msg.sender);
        _grantRole(RESONANCE_MANAGER_ROLE, msg.sender);
    }

    // ========================================================================
    // Soul Management
    // ========================================================================

    /**
     * @notice Create a new soul for the caller
     * @param profileUri IPFS URI containing profile data
     */
    function createSoul(string calldata profileUri) external returns (uint256) {
        if (souls[msg.sender].exists) revert SoulAlreadyExists();

        _soulIdCounter.increment();
        uint256 soulId = _soulIdCounter.current();

        souls[msg.sender] = Soul({
            id: soulId,
            owner: msg.sender,
            profileUri: profileUri,
            totalResonance: 0,
            connections: 0,
            bornAt: block.timestamp,
            exists: true
        });

        soulOwners[soulId] = msg.sender;

        emit SoulCreated(soulId, msg.sender, profileUri, block.timestamp);

        return soulId;
    }

    /**
     * @notice Update soul profile URI
     * @param newProfileUri New IPFS URI for profile
     */
    function updateProfile(string calldata newProfileUri) external {
        Soul storage soul = souls[msg.sender];
        if (!soul.exists) revert SoulDoesNotExist();

        soul.profileUri = newProfileUri;

        emit ProfileUpdated(soul.id, newProfileUri);
    }

    /**
     * @notice Get soul by address
     */
    function getSoul(address owner) external view returns (Soul memory) {
        if (!souls[owner].exists) revert SoulDoesNotExist();
        return souls[owner];
    }

    /**
     * @notice Get soul by ID
     */
    function getSoulById(uint256 soulId) external view returns (Soul memory) {
        address owner = soulOwners[soulId];
        if (owner == address(0)) revert SoulDoesNotExist();
        return souls[owner];
    }

    /**
     * @notice Check if an address has a soul
     */
    function hasSoul(address owner) external view returns (bool) {
        return souls[owner].exists;
    }

    // ========================================================================
    // Resonance
    // ========================================================================

    /**
     * @notice Add resonance to a soul (only callable by RESONANCE_MANAGER_ROLE)
     * @param soulId The soul to add resonance to
     * @param amount Amount of resonance to add
     * @param source Description of resonance source
     */
    function addResonance(
        uint256 soulId,
        uint256 amount,
        string calldata source
    ) external onlyRole(RESONANCE_MANAGER_ROLE) {
        address owner = soulOwners[soulId];
        if (owner == address(0)) revert SoulDoesNotExist();

        Soul storage soul = souls[owner];
        soul.totalResonance += amount;

        emit ResonanceAdded(soulId, amount, source, soul.totalResonance);
    }

    /**
     * @notice Get total resonance for a soul
     */
    function getResonance(uint256 soulId) external view returns (uint256) {
        address owner = soulOwners[soulId];
        if (owner == address(0)) revert SoulDoesNotExist();
        return souls[owner].totalResonance;
    }

    // ========================================================================
    // Credentials
    // ========================================================================

    /**
     * @notice Issue a credential to a soul (only callable by ATTESTER_ROLE)
     * @param soulId The soul receiving the credential
     * @param credentialType Type identifier for the credential
     * @param expiresAt Expiration timestamp (0 = never)
     * @param dataHash Hash of off-chain credential data
     */
    function issueCredential(
        uint256 soulId,
        bytes32 credentialType,
        uint256 expiresAt,
        bytes32 dataHash
    ) external onlyRole(ATTESTER_ROLE) {
        address owner = soulOwners[soulId];
        if (owner == address(0)) revert SoulDoesNotExist();

        Credential storage cred = credentials[soulId][credentialType];
        if (cred.issuedAt != 0 && !cred.revoked) revert CredentialAlreadyExists();

        credentials[soulId][credentialType] = Credential({
            credentialType: credentialType,
            issuer: msg.sender,
            issuedAt: block.timestamp,
            expiresAt: expiresAt,
            dataHash: dataHash,
            revoked: false
        });

        // Track credential types for this soul
        if (cred.issuedAt == 0) {
            soulCredentialTypes[soulId].push(credentialType);
        }

        emit CredentialIssued(soulId, credentialType, msg.sender, expiresAt);
    }

    /**
     * @notice Revoke a credential (only callable by original issuer or admin)
     * @param soulId The soul losing the credential
     * @param credentialType Type of credential to revoke
     */
    function revokeCredential(
        uint256 soulId,
        bytes32 credentialType
    ) external {
        Credential storage cred = credentials[soulId][credentialType];
        if (cred.issuedAt == 0) revert CredentialDoesNotExist();

        // Only issuer or admin can revoke
        require(
            cred.issuer == msg.sender || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Not authorized to revoke"
        );

        cred.revoked = true;

        emit CredentialRevoked(soulId, credentialType, msg.sender);
    }

    /**
     * @notice Check if a soul has a valid (non-expired, non-revoked) credential
     * @param soulId The soul to check
     * @param credentialType The credential type to check
     */
    function hasValidCredential(
        uint256 soulId,
        bytes32 credentialType
    ) external view returns (bool) {
        Credential storage cred = credentials[soulId][credentialType];

        if (cred.issuedAt == 0) return false;
        if (cred.revoked) return false;
        if (cred.expiresAt != 0 && block.timestamp > cred.expiresAt) return false;

        return true;
    }

    /**
     * @notice Get all credential types for a soul
     */
    function getCredentialTypes(uint256 soulId) external view returns (bytes32[] memory) {
        return soulCredentialTypes[soulId];
    }

    // ========================================================================
    // Connections
    // ========================================================================

    /**
     * @notice Form a connection between two souls
     * @param otherSoulId The soul to connect with
     */
    function connect(uint256 otherSoulId) external {
        Soul storage mySoul = souls[msg.sender];
        if (!mySoul.exists) revert SoulDoesNotExist();

        address otherOwner = soulOwners[otherSoulId];
        if (otherOwner == address(0)) revert SoulDoesNotExist();
        if (otherOwner == msg.sender) revert CannotConnectToSelf();

        uint256 mySoulId = mySoul.id;
        if (connections[mySoulId][otherSoulId]) revert ConnectionAlreadyExists();

        // Form bidirectional connection
        connections[mySoulId][otherSoulId] = true;
        connections[otherSoulId][mySoulId] = true;

        // Update connection counts
        mySoul.connections++;
        souls[otherOwner].connections++;

        emit ConnectionFormed(mySoulId, otherSoulId, block.timestamp);
    }

    /**
     * @notice Check if two souls are connected
     */
    function areConnected(uint256 soulId1, uint256 soulId2) external view returns (bool) {
        return connections[soulId1][soulId2];
    }

    // ========================================================================
    // Soulbound Enforcement
    // ========================================================================

    /**
     * @notice Souls are non-transferable. This function always reverts.
     */
    function transfer(address) external pure {
        revert("Souls are soulbound and cannot be transferred");
    }

    // ========================================================================
    // View Functions
    // ========================================================================

    /**
     * @notice Get total number of souls created
     */
    function totalSouls() external view returns (uint256) {
        return _soulIdCounter.current();
    }

    /**
     * @notice Get soul ID for an address
     */
    function soulIdOf(address owner) external view returns (uint256) {
        Soul storage soul = souls[owner];
        if (!soul.exists) revert SoulDoesNotExist();
        return soul.id;
    }
}
