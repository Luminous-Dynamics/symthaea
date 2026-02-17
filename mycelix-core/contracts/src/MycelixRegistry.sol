// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import {Address} from "@openzeppelin/contracts/utils/Address.sol";

/**
 * @title MycelixRegistry
 * @author Mycelix Network
 * @notice Registry for DIDs (Decentralized Identifiers) and hApp addresses
 * @dev Bridges Holochain agent IDs to Ethereum addresses
 *
 * DID Format: did:mycelix:<holochain-agent-pubkey>
 * The bytes32 did parameter is keccak256 of the full DID string
 *
 * Features:
 * - Register DIDs with proof of ownership
 * - Resolve DIDs to Ethereum addresses
 * - Support for multiple addresses per DID (primary + alternates)
 * - hApp contract address registry
 * - Revocation and key rotation support
 */
contract MycelixRegistry is Ownable, AccessControl, ReentrancyGuard, Pausable {
    using ECDSA for bytes32;
    using MessageHashUtils for bytes32;

    // ============ Constants ============

    /// @notice Role for governance actions
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    /// @notice Role for registering hApps
    bytes32 public constant HAPP_REGISTRAR_ROLE = keccak256("HAPP_REGISTRAR_ROLE");

    /// @notice Maximum alternate addresses per DID
    uint256 public constant MAX_ALTERNATES = 5;

    /// @notice Registration fee (can be 0)
    uint256 public registrationFee = 0;

    /// @notice Domain separator for EIP-712 signatures
    bytes32 public immutable DOMAIN_SEPARATOR;

    /// @notice Typehash for DID registration
    bytes32 public constant REGISTER_TYPEHASH =
        keccak256("RegisterDID(bytes32 did,address owner,uint256 nonce,uint256 deadline)");

    /// @notice Typehash for address updates
    bytes32 public constant UPDATE_TYPEHASH =
        keccak256("UpdateAddress(bytes32 did,address newOwner,uint256 nonce,uint256 deadline)");

    // ============ Structs ============

    struct DIDRecord {
        address owner; // Primary owner address
        address[] alternates; // Alternate addresses
        uint256 registeredAt; // Registration timestamp
        uint256 updatedAt; // Last update timestamp
        bytes32 metadataHash; // IPFS hash of metadata (profile, etc.)
        bool revoked; // Revocation status
    }

    struct HAppRecord {
        bytes32 happId; // Holochain hApp ID
        address contractAddress; // Ethereum contract address
        address registrar; // Who registered this
        uint256 registeredAt;
        bool active;
        bytes32 metadataHash; // IPFS hash of hApp metadata
    }

    // ============ State Variables ============

    /// @notice Mapping of DID hash to DID record
    mapping(bytes32 => DIDRecord) public didRecords;

    /// @notice Mapping of address to DID (reverse lookup)
    mapping(address => bytes32) public addressToDid;

    /// @notice Mapping of hApp ID to hApp record
    mapping(bytes32 => HAppRecord) public happRecords;

    /// @notice Mapping of contract address to hApp ID (reverse lookup)
    mapping(address => bytes32) public contractToHapp;

    /// @notice Nonces for EIP-712 signatures
    mapping(address => uint256) public nonces;

    /// @notice Total registered DIDs
    uint256 public totalDids;

    /// @notice Total registered hApps
    uint256 public totalHapps;

    /// @notice Fee recipient
    address payable public feeRecipient;

    // ============ Events ============

    event DIDRegistered(
        bytes32 indexed did,
        address indexed owner,
        uint256 timestamp
    );

    event DIDUpdated(
        bytes32 indexed did,
        address indexed oldOwner,
        address indexed newOwner,
        uint256 timestamp
    );

    event DIDRevoked(bytes32 indexed did, address indexed owner, uint256 timestamp);

    event AlternateAdded(
        bytes32 indexed did,
        address indexed alternate,
        uint256 timestamp
    );

    event AlternateRemoved(
        bytes32 indexed did,
        address indexed alternate,
        uint256 timestamp
    );

    event MetadataUpdated(bytes32 indexed did, bytes32 metadataHash);

    event HAppRegistered(
        bytes32 indexed happId,
        address indexed contractAddress,
        address registrar
    );

    event HAppUpdated(
        bytes32 indexed happId,
        address indexed oldContract,
        address indexed newContract
    );

    event HAppDeactivated(bytes32 indexed happId);

    event RegistrationFeeUpdated(uint256 oldFee, uint256 newFee);
    event FeeRecipientUpdated(address oldRecipient, address newRecipient);

    // ============ Errors ============

    error DIDAlreadyRegistered();
    error DIDNotRegistered();
    error DIDIsRevoked();
    error NotDIDOwner();
    error AddressAlreadyLinked();
    error TooManyAlternates();
    error AlternateNotFound();
    error HAppAlreadyRegistered();
    error HAppNotRegistered();
    error HAppNotActive();
    error InvalidSignature();
    error DeadlineExpired();
    error InsufficientFee();
    error ZeroAddress();
    error SameAddress();
    /// @dev C-04 Fix: Explicit error when trying to add owner as their own alternate
    error CannotAddOwnerAsAlternate();

    // ============ Constructor ============

    /**
     * @notice Initializes the MycelixRegistry
     * @param initialOwner Contract owner
     * @param initialFeeRecipient Fee recipient address
     */
    constructor(
        address initialOwner,
        address payable initialFeeRecipient
    ) Ownable(initialOwner) {
        if (initialOwner == address(0) || initialFeeRecipient == address(0)) {
            revert ZeroAddress();
        }

        feeRecipient = initialFeeRecipient;

        _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
        _grantRole(GOVERNANCE_ROLE, initialOwner);
        _grantRole(HAPP_REGISTRAR_ROLE, initialOwner);

        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256(
                    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
                ),
                keccak256("MycelixRegistry"),
                keccak256("1"),
                block.chainid,
                address(this)
            )
        );
    }

    // ============ DID Functions ============

    /**
     * @notice Registers a new DID with the caller as owner
     * @param did The keccak256 hash of the DID string
     * @param metadataHash IPFS hash of DID metadata
     */
    function registerDID(
        bytes32 did,
        bytes32 metadataHash
    ) external payable nonReentrant whenNotPaused {
        _registerDID(did, msg.sender, metadataHash);
    }

    /**
     * @notice Registers a DID with a signature (meta-transaction)
     * @param did The DID hash
     * @param owner The owner address
     * @param deadline Signature expiry
     * @param v Signature v
     * @param r Signature r
     * @param s Signature s
     * @param metadataHash IPFS metadata hash
     */
    function registerDIDWithSignature(
        bytes32 did,
        address owner,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s,
        bytes32 metadataHash
    ) external payable nonReentrant whenNotPaused {
        if (block.timestamp > deadline) revert DeadlineExpired();

        bytes32 structHash = keccak256(
            abi.encode(REGISTER_TYPEHASH, did, owner, nonces[owner]++, deadline)
        );
        bytes32 digest = keccak256(abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash));
        address signer = ECDSA.recover(digest, v, r, s);

        if (signer != owner) revert InvalidSignature();

        _registerDID(did, owner, metadataHash);
    }

    /**
     * @notice Resolves a DID to its owner address
     * @param did The DID hash
     * @return owner The owner address
     * @return alternates Array of alternate addresses
     * @return isActive Whether the DID is active (not revoked)
     */
    function resolveAddress(bytes32 did)
        external
        view
        returns (
            address owner,
            address[] memory alternates,
            bool isActive
        )
    {
        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();

        return (record.owner, record.alternates, !record.revoked);
    }

    /**
     * @notice Updates the primary owner of a DID
     * @param did The DID hash
     * @param newOwner The new owner address
     */
    function updateOwner(
        bytes32 did,
        address newOwner
    ) external nonReentrant whenNotPaused {
        if (newOwner == address(0)) revert ZeroAddress();

        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();
        if (record.revoked) revert DIDIsRevoked();
        if (record.owner != msg.sender) revert NotDIDOwner();
        if (newOwner == msg.sender) revert SameAddress();
        if (addressToDid[newOwner] != bytes32(0)) revert AddressAlreadyLinked();

        address oldOwner = record.owner;

        // Update mappings
        delete addressToDid[oldOwner];
        addressToDid[newOwner] = did;

        record.owner = newOwner;
        record.updatedAt = block.timestamp;

        emit DIDUpdated(did, oldOwner, newOwner, block.timestamp);
    }

    /**
     * @notice Adds an alternate address to a DID
     * @param did The DID hash
     * @param alternate The alternate address to add
     * @dev C-04 Fix: Explicitly checks that alternate is not the owner to prevent
     * address aliasing where the same address could be both owner and alternate
     */
    function addAlternate(
        bytes32 did,
        address alternate
    ) external nonReentrant whenNotPaused {
        if (alternate == address(0)) revert ZeroAddress();

        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();
        if (record.revoked) revert DIDIsRevoked();
        if (record.owner != msg.sender) revert NotDIDOwner();
        // C-04: Explicitly prevent adding owner as alternate (clearer than generic AddressAlreadyLinked)
        if (alternate == record.owner) revert CannotAddOwnerAsAlternate();
        if (record.alternates.length >= MAX_ALTERNATES) revert TooManyAlternates();
        if (addressToDid[alternate] != bytes32(0)) revert AddressAlreadyLinked();

        record.alternates.push(alternate);
        addressToDid[alternate] = did;
        record.updatedAt = block.timestamp;

        emit AlternateAdded(did, alternate, block.timestamp);
    }

    /**
     * @notice Removes an alternate address from a DID
     * @param did The DID hash
     * @param alternate The alternate address to remove
     */
    function removeAlternate(
        bytes32 did,
        address alternate
    ) external nonReentrant whenNotPaused {
        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();
        if (record.owner != msg.sender) revert NotDIDOwner();

        bool found = false;
        uint256 length = record.alternates.length;

        for (uint256 i = 0; i < length; i++) {
            if (record.alternates[i] == alternate) {
                // Swap with last and pop
                record.alternates[i] = record.alternates[length - 1];
                record.alternates.pop();
                found = true;
                break;
            }
        }

        if (!found) revert AlternateNotFound();

        delete addressToDid[alternate];
        record.updatedAt = block.timestamp;

        emit AlternateRemoved(did, alternate, block.timestamp);
    }

    /**
     * @notice Updates DID metadata
     * @param did The DID hash
     * @param metadataHash New IPFS metadata hash
     */
    function updateMetadata(
        bytes32 did,
        bytes32 metadataHash
    ) external nonReentrant whenNotPaused {
        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();
        if (record.revoked) revert DIDIsRevoked();
        if (record.owner != msg.sender) revert NotDIDOwner();

        record.metadataHash = metadataHash;
        record.updatedAt = block.timestamp;

        emit MetadataUpdated(did, metadataHash);
    }

    /**
     * @notice Revokes a DID
     * @param did The DID hash
     */
    function revokeDID(bytes32 did) external nonReentrant whenNotPaused {
        DIDRecord storage record = didRecords[did];
        if (record.owner == address(0)) revert DIDNotRegistered();
        if (record.owner != msg.sender) revert NotDIDOwner();

        record.revoked = true;
        record.updatedAt = block.timestamp;

        // Clear reverse lookups
        delete addressToDid[record.owner];
        for (uint256 i = 0; i < record.alternates.length; i++) {
            delete addressToDid[record.alternates[i]];
        }

        emit DIDRevoked(did, msg.sender, block.timestamp);
    }

    // ============ hApp Registry Functions ============

    /**
     * @notice Registers a hApp contract
     * @param happId The Holochain hApp ID hash
     * @param contractAddress The Ethereum contract address
     * @param metadataHash IPFS metadata hash
     */
    function registerHApp(
        bytes32 happId,
        address contractAddress,
        bytes32 metadataHash
    ) external onlyRole(HAPP_REGISTRAR_ROLE) nonReentrant whenNotPaused {
        if (contractAddress == address(0)) revert ZeroAddress();
        if (happRecords[happId].contractAddress != address(0)) {
            revert HAppAlreadyRegistered();
        }
        if (contractToHapp[contractAddress] != bytes32(0)) {
            revert AddressAlreadyLinked();
        }

        happRecords[happId] = HAppRecord({
            happId: happId,
            contractAddress: contractAddress,
            registrar: msg.sender,
            registeredAt: block.timestamp,
            active: true,
            metadataHash: metadataHash
        });

        contractToHapp[contractAddress] = happId;
        totalHapps++;

        emit HAppRegistered(happId, contractAddress, msg.sender);
    }

    /**
     * @notice Updates a hApp contract address
     * @param happId The hApp ID
     * @param newContractAddress The new contract address
     */
    function updateHApp(
        bytes32 happId,
        address newContractAddress
    ) external onlyRole(HAPP_REGISTRAR_ROLE) nonReentrant whenNotPaused {
        if (newContractAddress == address(0)) revert ZeroAddress();

        HAppRecord storage record = happRecords[happId];
        if (record.contractAddress == address(0)) revert HAppNotRegistered();
        if (!record.active) revert HAppNotActive();
        if (contractToHapp[newContractAddress] != bytes32(0)) {
            revert AddressAlreadyLinked();
        }

        address oldContract = record.contractAddress;

        // Update mappings
        delete contractToHapp[oldContract];
        contractToHapp[newContractAddress] = happId;
        record.contractAddress = newContractAddress;

        emit HAppUpdated(happId, oldContract, newContractAddress);
    }

    /**
     * @notice Deactivates a hApp
     * @param happId The hApp ID
     */
    function deactivateHApp(
        bytes32 happId
    ) external onlyRole(HAPP_REGISTRAR_ROLE) nonReentrant whenNotPaused {
        HAppRecord storage record = happRecords[happId];
        if (record.contractAddress == address(0)) revert HAppNotRegistered();

        record.active = false;
        delete contractToHapp[record.contractAddress];

        emit HAppDeactivated(happId);
    }

    /**
     * @notice Resolves a hApp ID to its contract address
     * @param happId The hApp ID
     * @return contractAddress The contract address
     * @return isActive Whether the hApp is active
     */
    function resolveHApp(bytes32 happId)
        external
        view
        returns (address contractAddress, bool isActive)
    {
        HAppRecord storage record = happRecords[happId];
        if (record.contractAddress == address(0)) revert HAppNotRegistered();

        return (record.contractAddress, record.active);
    }

    // ============ Admin Functions ============

    /**
     * @notice Updates the registration fee
     * @param newFee The new fee amount
     */
    function setRegistrationFee(uint256 newFee) external onlyRole(GOVERNANCE_ROLE) {
        uint256 oldFee = registrationFee;
        registrationFee = newFee;
        emit RegistrationFeeUpdated(oldFee, newFee);
    }

    /**
     * @notice Updates the fee recipient
     * @param newRecipient The new recipient address
     */
    function setFeeRecipient(
        address payable newRecipient
    ) external onlyRole(GOVERNANCE_ROLE) {
        if (newRecipient == address(0)) revert ZeroAddress();

        address oldRecipient = feeRecipient;
        feeRecipient = newRecipient;
        emit FeeRecipientUpdated(oldRecipient, newRecipient);
    }

    /**
     * @notice Withdraws accumulated fees
     * @dev C-03 Fix: Uses Address.sendValue() instead of raw .call{} for safer transfers.
     * While nonReentrant protects against reentrancy, sendValue provides additional
     * safety by reverting with a clear error if the transfer fails.
     */
    function withdrawFees() external onlyRole(GOVERNANCE_ROLE) nonReentrant {
        uint256 balance = address(this).balance;
        if (balance > 0) {
            Address.sendValue(feeRecipient, balance);
        }
    }

    /**
     * @notice Pauses the contract
     */
    function pause() external onlyRole(GOVERNANCE_ROLE) {
        _pause();
    }

    /**
     * @notice Unpauses the contract
     */
    function unpause() external onlyRole(GOVERNANCE_ROLE) {
        _unpause();
    }

    // ============ View Functions ============

    /**
     * @notice Gets full DID record
     * @param did The DID hash
     */
    function getDIDRecord(bytes32 did) external view returns (DIDRecord memory) {
        return didRecords[did];
    }

    /**
     * @notice Gets full hApp record
     * @param happId The hApp ID
     */
    function getHAppRecord(bytes32 happId) external view returns (HAppRecord memory) {
        return happRecords[happId];
    }

    /**
     * @notice Checks if an address is linked to any DID
     * @param addr The address to check
     */
    function isAddressLinked(address addr) external view returns (bool) {
        return addressToDid[addr] != bytes32(0);
    }

    /**
     * @notice Gets the DID for an address
     * @param addr The address to lookup
     */
    function getDIDForAddress(address addr) external view returns (bytes32) {
        return addressToDid[addr];
    }

    // ============ Internal Functions ============

    /**
     * @dev Internal DID registration logic
     */
    function _registerDID(
        bytes32 did,
        address owner,
        bytes32 metadataHash
    ) internal {
        if (msg.value < registrationFee) revert InsufficientFee();
        if (didRecords[did].owner != address(0)) revert DIDAlreadyRegistered();
        if (addressToDid[owner] != bytes32(0)) revert AddressAlreadyLinked();

        didRecords[did] = DIDRecord({
            owner: owner,
            alternates: new address[](0),
            registeredAt: block.timestamp,
            updatedAt: block.timestamp,
            metadataHash: metadataHash,
            revoked: false
        });

        addressToDid[owner] = did;
        totalDids++;

        emit DIDRegistered(did, owner, block.timestamp);

        // Refund excess
        if (msg.value > registrationFee) {
            (bool success, ) = msg.sender.call{value: msg.value - registrationFee}("");
            require(success, "Refund failed");
        }
    }

    /**
     * @dev Allows receiving native tokens
     */
    receive() external payable {}
}
