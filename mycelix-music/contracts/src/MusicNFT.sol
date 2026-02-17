// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/token/ERC1155/extensions/ERC1155Supply.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/token/common/ERC2981.sol";

/**
 * @title MusicNFT
 * @notice Music collectibles with tiered editions and on-chain royalties
 * @dev ERC1155 for efficient multi-edition support with ERC2981 royalties
 */
contract MusicNFT is
    ERC1155,
    ERC1155Supply,
    ERC2981,
    AccessControl,
    ReentrancyGuard
{
    using SafeERC20 for IERC20;
    using Strings for uint256;

    // ==================== Roles ====================

    bytes32 public constant ARTIST_ROLE = keccak256("ARTIST_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    // ==================== Structs ====================

    enum EditionType {
        STANDARD,    // Regular edition
        LIMITED,     // Limited quantity
        EXCLUSIVE,   // 1/1 or very rare
        FREE         // Free claim
    }

    struct Release {
        string songId;
        address artist;
        string metadataUri;
        EditionType editionType;
        uint256 maxSupply;     // 0 = unlimited
        uint256 price;
        address paymentToken;  // address(0) = native token
        uint256 startTime;
        uint256 endTime;       // 0 = no end
        bool paused;
        bool exists;
    }

    struct MintStats {
        uint256 totalMinted;
        uint256 totalRevenue;
        uint256 uniqueHolders;
    }

    // ==================== State ====================

    /// @notice Base URI for metadata
    string public baseUri;

    /// @notice Next token ID
    uint256 public nextTokenId = 1;

    /// @notice Release data by token ID
    mapping(uint256 => Release) public releases;

    /// @notice Mint stats by token ID
    mapping(uint256 => MintStats) public mintStats;

    /// @notice Artist to token IDs
    mapping(address => uint256[]) public artistReleases;

    /// @notice Holder tracking per token
    mapping(uint256 => mapping(address => bool)) public hasToken;

    /// @notice Platform treasury
    address public treasury;

    /// @notice Platform fee in basis points
    uint256 public platformFee = 500; // 5%

    /// @notice Default royalty in basis points
    uint96 public defaultRoyalty = 1000; // 10%

    /// @notice Allowlist per token
    mapping(uint256 => mapping(address => bool)) public allowlist;

    /// @notice Allowlist required per token
    mapping(uint256 => bool) public allowlistRequired;

    /// @notice Max per wallet per token
    mapping(uint256 => uint256) public maxPerWallet;

    /// @notice Mints per wallet per token
    mapping(uint256 => mapping(address => uint256)) public mintsPerWallet;

    // ==================== Events ====================

    event ReleaseCreated(
        uint256 indexed tokenId,
        address indexed artist,
        string songId,
        EditionType editionType,
        uint256 maxSupply,
        uint256 price
    );

    event ReleaseMinted(
        uint256 indexed tokenId,
        address indexed minter,
        uint256 quantity,
        uint256 totalPaid
    );

    event ReleaseUpdated(uint256 indexed tokenId);

    event ReleasePaused(uint256 indexed tokenId, bool paused);

    event AllowlistUpdated(uint256 indexed tokenId, address[] addresses, bool status);

    // ==================== Errors ====================

    error ReleaseNotFound();
    error ReleaseNotActive();
    error ReleasePaused();
    error MaxSupplyReached();
    error InsufficientPayment();
    error Unauthorized();
    error InvalidInput();
    error NotAllowlisted();
    error MaxPerWalletReached();
    error TransferFailed();

    // ==================== Constructor ====================

    constructor(
        string memory _baseUri,
        address _treasury
    ) ERC1155(_baseUri) {
        baseUri = _baseUri;
        treasury = _treasury;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    // ==================== Release Creation ====================

    /**
     * @notice Create a new music release
     * @param songId Platform song ID
     * @param metadataUri Token metadata URI
     * @param editionType Type of edition
     * @param maxSupply Maximum supply (0 for unlimited)
     * @param price Mint price
     * @param paymentToken Payment token (address(0) for native)
     * @param startTime Sale start time
     * @param endTime Sale end time (0 for no end)
     */
    function createRelease(
        string calldata songId,
        string calldata metadataUri,
        EditionType editionType,
        uint256 maxSupply,
        uint256 price,
        address paymentToken,
        uint256 startTime,
        uint256 endTime,
        uint256 _maxPerWallet
    ) external returns (uint256 tokenId) {
        // Validate exclusive editions
        if (editionType == EditionType.EXCLUSIVE && maxSupply != 1) {
            revert InvalidInput();
        }

        tokenId = nextTokenId++;

        releases[tokenId] = Release({
            songId: songId,
            artist: msg.sender,
            metadataUri: metadataUri,
            editionType: editionType,
            maxSupply: maxSupply,
            price: price,
            paymentToken: paymentToken,
            startTime: startTime,
            endTime: endTime,
            paused: false,
            exists: true
        });

        if (_maxPerWallet > 0) {
            maxPerWallet[tokenId] = _maxPerWallet;
        }

        artistReleases[msg.sender].push(tokenId);

        // Set royalty for this token
        _setTokenRoyalty(tokenId, msg.sender, defaultRoyalty);

        _grantRole(ARTIST_ROLE, msg.sender);

        emit ReleaseCreated(
            tokenId,
            msg.sender,
            songId,
            editionType,
            maxSupply,
            price
        );

        return tokenId;
    }

    /**
     * @notice Update release settings
     * @param tokenId Token ID to update
     * @param price New price
     * @param startTime New start time
     * @param endTime New end time
     */
    function updateRelease(
        uint256 tokenId,
        uint256 price,
        uint256 startTime,
        uint256 endTime
    ) external {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (msg.sender != release.artist && !hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            revert Unauthorized();
        }

        release.price = price;
        release.startTime = startTime;
        release.endTime = endTime;

        emit ReleaseUpdated(tokenId);
    }

    /**
     * @notice Pause/unpause a release
     * @param tokenId Token ID
     * @param paused Paused state
     */
    function pauseRelease(uint256 tokenId, bool paused) external {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (msg.sender != release.artist && !hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            revert Unauthorized();
        }

        release.paused = paused;

        emit ReleasePaused(tokenId, paused);
    }

    // ==================== Minting ====================

    /**
     * @notice Mint a release
     * @param tokenId Token ID to mint
     * @param quantity Number to mint
     */
    function mint(
        uint256 tokenId,
        uint256 quantity
    ) external payable nonReentrant {
        Release storage release = releases[tokenId];

        // Validations
        if (!release.exists) revert ReleaseNotFound();
        if (release.paused) revert ReleasePaused();
        if (block.timestamp < release.startTime) revert ReleaseNotActive();
        if (release.endTime > 0 && block.timestamp > release.endTime) {
            revert ReleaseNotActive();
        }

        // Check supply
        if (release.maxSupply > 0) {
            if (totalSupply(tokenId) + quantity > release.maxSupply) {
                revert MaxSupplyReached();
            }
        }

        // Check allowlist
        if (allowlistRequired[tokenId] && !allowlist[tokenId][msg.sender]) {
            revert NotAllowlisted();
        }

        // Check max per wallet
        if (maxPerWallet[tokenId] > 0) {
            if (mintsPerWallet[tokenId][msg.sender] + quantity > maxPerWallet[tokenId]) {
                revert MaxPerWalletReached();
            }
        }

        // Calculate payment
        uint256 totalPrice = release.price * quantity;

        if (totalPrice > 0) {
            if (release.paymentToken == address(0)) {
                // Native token payment
                if (msg.value < totalPrice) revert InsufficientPayment();

                // Platform fee
                uint256 fee = (totalPrice * platformFee) / 10000;
                uint256 artistAmount = totalPrice - fee;

                (bool feeSuccess, ) = treasury.call{value: fee}("");
                (bool artistSuccess, ) = release.artist.call{value: artistAmount}("");

                if (!feeSuccess || !artistSuccess) revert TransferFailed();

                // Refund excess
                if (msg.value > totalPrice) {
                    (bool refundSuccess, ) = msg.sender.call{value: msg.value - totalPrice}("");
                    if (!refundSuccess) revert TransferFailed();
                }
            } else {
                // ERC20 payment
                IERC20 token = IERC20(release.paymentToken);

                uint256 fee = (totalPrice * platformFee) / 10000;
                uint256 artistAmount = totalPrice - fee;

                token.safeTransferFrom(msg.sender, treasury, fee);
                token.safeTransferFrom(msg.sender, release.artist, artistAmount);
            }
        }

        // Update tracking
        mintsPerWallet[tokenId][msg.sender] += quantity;

        if (!hasToken[tokenId][msg.sender]) {
            hasToken[tokenId][msg.sender] = true;
            mintStats[tokenId].uniqueHolders++;
        }

        mintStats[tokenId].totalMinted += quantity;
        mintStats[tokenId].totalRevenue += totalPrice;

        // Mint
        _mint(msg.sender, tokenId, quantity, "");

        emit ReleaseMinted(tokenId, msg.sender, quantity, totalPrice);
    }

    /**
     * @notice Free mint for allowlisted addresses
     * @param tokenId Token ID to mint
     */
    function freeMint(uint256 tokenId) external nonReentrant {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (release.editionType != EditionType.FREE) revert InvalidInput();
        if (release.paused) revert ReleasePaused();

        if (allowlistRequired[tokenId] && !allowlist[tokenId][msg.sender]) {
            revert NotAllowlisted();
        }

        // One per wallet for free mints
        if (mintsPerWallet[tokenId][msg.sender] > 0) {
            revert MaxPerWalletReached();
        }

        if (release.maxSupply > 0 && totalSupply(tokenId) >= release.maxSupply) {
            revert MaxSupplyReached();
        }

        mintsPerWallet[tokenId][msg.sender] = 1;

        if (!hasToken[tokenId][msg.sender]) {
            hasToken[tokenId][msg.sender] = true;
            mintStats[tokenId].uniqueHolders++;
        }

        mintStats[tokenId].totalMinted++;

        _mint(msg.sender, tokenId, 1, "");

        emit ReleaseMinted(tokenId, msg.sender, 1, 0);
    }

    /**
     * @notice Artist airdrop
     * @param tokenId Token ID
     * @param recipients Recipients
     * @param quantities Quantities per recipient
     */
    function airdrop(
        uint256 tokenId,
        address[] calldata recipients,
        uint256[] calldata quantities
    ) external {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (msg.sender != release.artist && !hasRole(MINTER_ROLE, msg.sender)) {
            revert Unauthorized();
        }

        if (recipients.length != quantities.length) revert InvalidInput();

        uint256 totalQuantity = 0;
        for (uint256 i = 0; i < quantities.length; i++) {
            totalQuantity += quantities[i];
        }

        if (release.maxSupply > 0) {
            if (totalSupply(tokenId) + totalQuantity > release.maxSupply) {
                revert MaxSupplyReached();
            }
        }

        for (uint256 i = 0; i < recipients.length; i++) {
            address recipient = recipients[i];
            uint256 quantity = quantities[i];

            if (!hasToken[tokenId][recipient]) {
                hasToken[tokenId][recipient] = true;
                mintStats[tokenId].uniqueHolders++;
            }

            mintStats[tokenId].totalMinted += quantity;
            _mint(recipient, tokenId, quantity, "");
        }
    }

    // ==================== Allowlist ====================

    /**
     * @notice Set allowlist for a release
     * @param tokenId Token ID
     * @param addresses Addresses to add/remove
     * @param status Allowlist status
     */
    function setAllowlist(
        uint256 tokenId,
        address[] calldata addresses,
        bool status
    ) external {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (msg.sender != release.artist && !hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            revert Unauthorized();
        }

        for (uint256 i = 0; i < addresses.length; i++) {
            allowlist[tokenId][addresses[i]] = status;
        }

        emit AllowlistUpdated(tokenId, addresses, status);
    }

    /**
     * @notice Set allowlist requirement
     * @param tokenId Token ID
     * @param required Whether allowlist is required
     */
    function setAllowlistRequired(uint256 tokenId, bool required) external {
        Release storage release = releases[tokenId];

        if (!release.exists) revert ReleaseNotFound();
        if (msg.sender != release.artist && !hasRole(DEFAULT_ADMIN_ROLE, msg.sender)) {
            revert Unauthorized();
        }

        allowlistRequired[tokenId] = required;
    }

    // ==================== View Functions ====================

    /**
     * @notice Get release details
     * @param tokenId Token ID
     */
    function getRelease(uint256 tokenId) external view returns (Release memory) {
        return releases[tokenId];
    }

    /**
     * @notice Get releases by artist
     * @param artist Artist address
     */
    function getArtistReleases(address artist) external view returns (uint256[] memory) {
        return artistReleases[artist];
    }

    /**
     * @notice Get mint stats
     * @param tokenId Token ID
     */
    function getMintStats(uint256 tokenId) external view returns (MintStats memory) {
        return mintStats[tokenId];
    }

    /**
     * @notice Check if release is mintable
     * @param tokenId Token ID
     */
    function isMintable(uint256 tokenId) external view returns (bool) {
        Release storage release = releases[tokenId];

        if (!release.exists) return false;
        if (release.paused) return false;
        if (block.timestamp < release.startTime) return false;
        if (release.endTime > 0 && block.timestamp > release.endTime) return false;
        if (release.maxSupply > 0 && totalSupply(tokenId) >= release.maxSupply) return false;

        return true;
    }

    /**
     * @notice Token URI
     * @param tokenId Token ID
     */
    function uri(uint256 tokenId) public view override returns (string memory) {
        Release storage release = releases[tokenId];

        if (!release.exists) return "";

        if (bytes(release.metadataUri).length > 0) {
            return release.metadataUri;
        }

        return string(abi.encodePacked(baseUri, tokenId.toString()));
    }

    // ==================== Admin Functions ====================

    /**
     * @notice Update platform fee
     * @param newFee New fee in basis points
     */
    function setPlatformFee(uint256 newFee) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newFee > 2000) revert InvalidInput(); // Max 20%
        platformFee = newFee;
    }

    /**
     * @notice Update treasury
     * @param newTreasury New treasury address
     */
    function setTreasury(address newTreasury) external onlyRole(DEFAULT_ADMIN_ROLE) {
        treasury = newTreasury;
    }

    /**
     * @notice Update base URI
     * @param newBaseUri New base URI
     */
    function setBaseUri(string calldata newBaseUri) external onlyRole(DEFAULT_ADMIN_ROLE) {
        baseUri = newBaseUri;
    }

    /**
     * @notice Update default royalty
     * @param newRoyalty New royalty in basis points
     */
    function setDefaultRoyalty(uint96 newRoyalty) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newRoyalty > 2500) revert InvalidInput(); // Max 25%
        defaultRoyalty = newRoyalty;
    }

    // ==================== Overrides ====================

    function supportsInterface(
        bytes4 interfaceId
    ) public view override(ERC1155, ERC2981, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }

    function _beforeTokenTransfer(
        address operator,
        address from,
        address to,
        uint256[] memory ids,
        uint256[] memory amounts,
        bytes memory data
    ) internal override(ERC1155, ERC1155Supply) {
        super._beforeTokenTransfer(operator, from, to, ids, amounts, data);

        // Update holder tracking on transfer
        for (uint256 i = 0; i < ids.length; i++) {
            uint256 tokenId = ids[i];

            if (from != address(0) && balanceOf(from, tokenId) == amounts[i]) {
                hasToken[tokenId][from] = false;
                if (mintStats[tokenId].uniqueHolders > 0) {
                    mintStats[tokenId].uniqueHolders--;
                }
            }

            if (to != address(0) && !hasToken[tokenId][to]) {
                hasToken[tokenId][to] = true;
                mintStats[tokenId].uniqueHolders++;
            }
        }
    }
}
