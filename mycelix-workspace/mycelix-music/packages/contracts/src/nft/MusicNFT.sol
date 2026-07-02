// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/token/ERC1155/extensions/ERC1155Supply.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title MusicNFT
 * @notice ERC-1155 Multi-token for music releases
 * @dev Supports editions, stems, and collectibles with royalty splits
 *
 * Token types:
 * - Full Track: The complete audio file
 * - Stems: Individual components (drums, bass, vocals, etc.)
 * - Cover Art: Album/track artwork
 * - Collectibles: Special edition memorabilia
 *
 * Features:
 * - Limited editions with dynamic pricing
 * - Royalty splits for collaborators
 * - Unlockable content for holders
 * - On-chain provenance
 */
contract MusicNFT is ERC1155, ERC1155Supply, AccessControl, ReentrancyGuard {
    using Strings for uint256;

    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ARTIST_ROLE = keccak256("ARTIST_ROLE");

    enum TokenType {
        FULL_TRACK,
        STEM_DRUMS,
        STEM_BASS,
        STEM_VOCALS,
        STEM_OTHER,
        COVER_ART,
        COLLECTIBLE
    }

    struct Release {
        uint256 id;
        address artist;
        string title;
        string metadataUri;       // IPFS URI for metadata
        string audioUri;          // IPFS URI for audio
        uint256 releaseDate;
        uint256 totalEditions;
        uint256 mintedEditions;
        uint256 price;
        bool isActive;
        RoyaltySplit[] royalties;
    }

    struct RoyaltySplit {
        address recipient;
        uint256 share;            // Basis points (10000 = 100%)
        string role;
    }

    struct TokenMetadata {
        uint256 releaseId;
        TokenType tokenType;
        uint256 edition;          // Edition number within release
        uint256 mintedAt;
        string unlockableUri;     // Additional content for holders
    }

    // Release ID => Release
    mapping(uint256 => Release) public releases;
    uint256 public releaseCount;

    // Token ID => Metadata
    mapping(uint256 => TokenMetadata) public tokenMetadata;

    // Release ID => Token Type => Token ID
    mapping(uint256 => mapping(TokenType => uint256)) public releaseTokens;

    // Base URI for metadata
    string private _baseUri;

    // Platform fee (basis points)
    uint256 public platformFee = 250; // 2.5%
    address public platformFeeRecipient;

    // Secondary sale royalty (EIP-2981)
    uint256 public defaultRoyaltyBps = 1000; // 10%

    // ============================================================================
    // Events
    // ============================================================================

    event ReleaseCreated(
        uint256 indexed releaseId,
        address indexed artist,
        string title,
        uint256 totalEditions,
        uint256 price
    );

    event TokenMinted(
        uint256 indexed tokenId,
        uint256 indexed releaseId,
        address indexed minter,
        TokenType tokenType,
        uint256 edition
    );

    event RoyaltiesDistributed(
        uint256 indexed releaseId,
        uint256 amount,
        address[] recipients,
        uint256[] amounts
    );

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(
        string memory baseUri,
        address _platformFeeRecipient
    ) ERC1155(baseUri) {
        _baseUri = baseUri;
        platformFeeRecipient = _platformFeeRecipient;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    // ============================================================================
    // Release Management
    // ============================================================================

    /**
     * @notice Create a new music release
     */
    function createRelease(
        string calldata title,
        string calldata metadataUri,
        string calldata audioUri,
        uint256 totalEditions,
        uint256 price,
        address[] calldata royaltyRecipients,
        uint256[] calldata royaltyShares,
        string[] calldata royaltyRoles
    ) external returns (uint256) {
        require(royaltyRecipients.length == royaltyShares.length, "Length mismatch");
        require(royaltyRecipients.length == royaltyRoles.length, "Length mismatch");

        uint256 totalShares = 0;
        for (uint256 i = 0; i < royaltyShares.length; i++) {
            totalShares += royaltyShares[i];
        }
        require(totalShares == 10000, "Shares must equal 10000");

        uint256 releaseId = releaseCount++;

        Release storage release = releases[releaseId];
        release.id = releaseId;
        release.artist = msg.sender;
        release.title = title;
        release.metadataUri = metadataUri;
        release.audioUri = audioUri;
        release.releaseDate = block.timestamp;
        release.totalEditions = totalEditions;
        release.mintedEditions = 0;
        release.price = price;
        release.isActive = true;

        for (uint256 i = 0; i < royaltyRecipients.length; i++) {
            release.royalties.push(RoyaltySplit({
                recipient: royaltyRecipients[i],
                share: royaltyShares[i],
                role: royaltyRoles[i]
            }));
        }

        _grantRole(ARTIST_ROLE, msg.sender);

        emit ReleaseCreated(releaseId, msg.sender, title, totalEditions, price);

        return releaseId;
    }

    /**
     * @notice Update release metadata
     */
    function updateRelease(
        uint256 releaseId,
        string calldata metadataUri,
        bool isActive
    ) external {
        Release storage release = releases[releaseId];
        require(release.artist == msg.sender, "Not artist");

        release.metadataUri = metadataUri;
        release.isActive = isActive;
    }

    // ============================================================================
    // Minting
    // ============================================================================

    /**
     * @notice Mint a token from a release
     */
    function mint(
        uint256 releaseId,
        TokenType tokenType
    ) external payable nonReentrant {
        Release storage release = releases[releaseId];
        require(release.isActive, "Release not active");
        require(release.mintedEditions < release.totalEditions, "Sold out");
        require(msg.value >= release.price, "Insufficient payment");

        // Generate token ID
        uint256 tokenId = _generateTokenId(releaseId, tokenType, release.mintedEditions);

        // Create metadata
        tokenMetadata[tokenId] = TokenMetadata({
            releaseId: releaseId,
            tokenType: tokenType,
            edition: release.mintedEditions + 1,
            mintedAt: block.timestamp,
            unlockableUri: ""
        });

        release.mintedEditions++;

        // Distribute payment
        _distributePayment(releaseId, msg.value);

        // Mint token
        _mint(msg.sender, tokenId, 1, "");

        emit TokenMinted(tokenId, releaseId, msg.sender, tokenType, release.mintedEditions);

        // Refund excess
        if (msg.value > release.price) {
            (bool success, ) = payable(msg.sender).call{value: msg.value - release.price}("");
            require(success, "Refund failed");
        }
    }

    /**
     * @notice Batch mint multiple token types
     */
    function mintBatch(
        uint256 releaseId,
        TokenType[] calldata tokenTypes
    ) external payable nonReentrant {
        Release storage release = releases[releaseId];
        require(release.isActive, "Release not active");

        uint256 totalPrice = release.price * tokenTypes.length;
        require(msg.value >= totalPrice, "Insufficient payment");
        require(release.mintedEditions + tokenTypes.length <= release.totalEditions, "Not enough editions");

        uint256[] memory tokenIds = new uint256[](tokenTypes.length);
        uint256[] memory amounts = new uint256[](tokenTypes.length);

        for (uint256 i = 0; i < tokenTypes.length; i++) {
            uint256 tokenId = _generateTokenId(releaseId, tokenTypes[i], release.mintedEditions + i);

            tokenMetadata[tokenId] = TokenMetadata({
                releaseId: releaseId,
                tokenType: tokenTypes[i],
                edition: release.mintedEditions + i + 1,
                mintedAt: block.timestamp,
                unlockableUri: ""
            });

            tokenIds[i] = tokenId;
            amounts[i] = 1;
        }

        release.mintedEditions += tokenTypes.length;

        // Distribute payment
        _distributePayment(releaseId, totalPrice);

        // Mint tokens
        _mintBatch(msg.sender, tokenIds, amounts, "");

        // Refund excess
        if (msg.value > totalPrice) {
            (bool success, ) = payable(msg.sender).call{value: msg.value - totalPrice}("");
            require(success, "Refund failed");
        }
    }

    /**
     * @notice Artist mint (free for artist)
     */
    function artistMint(
        uint256 releaseId,
        TokenType tokenType,
        address to,
        uint256 amount
    ) external {
        Release storage release = releases[releaseId];
        require(release.artist == msg.sender, "Not artist");
        require(release.mintedEditions + amount <= release.totalEditions, "Not enough editions");

        for (uint256 i = 0; i < amount; i++) {
            uint256 tokenId = _generateTokenId(releaseId, tokenType, release.mintedEditions + i);

            tokenMetadata[tokenId] = TokenMetadata({
                releaseId: releaseId,
                tokenType: tokenType,
                edition: release.mintedEditions + i + 1,
                mintedAt: block.timestamp,
                unlockableUri: ""
            });

            _mint(to, tokenId, 1, "");
        }

        release.mintedEditions += amount;
    }

    // ============================================================================
    // Payment Distribution
    // ============================================================================

    function _distributePayment(uint256 releaseId, uint256 amount) internal {
        Release storage release = releases[releaseId];

        // Platform fee
        uint256 fee = (amount * platformFee) / 10000;
        if (fee > 0) {
            (bool feeSuccess, ) = payable(platformFeeRecipient).call{value: fee}("");
            require(feeSuccess, "Fee transfer failed");
        }

        uint256 remaining = amount - fee;

        // Distribute to royalty recipients
        address[] memory recipients = new address[](release.royalties.length);
        uint256[] memory amounts = new uint256[](release.royalties.length);

        for (uint256 i = 0; i < release.royalties.length; i++) {
            RoyaltySplit storage split = release.royalties[i];
            uint256 recipientAmount = (remaining * split.share) / 10000;

            if (recipientAmount > 0) {
                (bool success, ) = payable(split.recipient).call{value: recipientAmount}("");
                require(success, "Royalty transfer failed");
            }

            recipients[i] = split.recipient;
            amounts[i] = recipientAmount;
        }

        emit RoyaltiesDistributed(releaseId, amount, recipients, amounts);
    }

    // ============================================================================
    // Unlockable Content
    // ============================================================================

    /**
     * @notice Set unlockable content for token holders
     */
    function setUnlockable(uint256 tokenId, string calldata unlockableUri) external {
        TokenMetadata storage metadata = tokenMetadata[tokenId];
        Release storage release = releases[metadata.releaseId];
        require(release.artist == msg.sender, "Not artist");

        metadata.unlockableUri = unlockableUri;
    }

    /**
     * @notice Get unlockable content (only for token holders)
     */
    function getUnlockable(uint256 tokenId) external view returns (string memory) {
        require(balanceOf(msg.sender, tokenId) > 0, "Not a holder");
        return tokenMetadata[tokenId].unlockableUri;
    }

    // ============================================================================
    // Royalties (EIP-2981)
    // ============================================================================

    function royaltyInfo(
        uint256 tokenId,
        uint256 salePrice
    ) external view returns (address receiver, uint256 royaltyAmount) {
        TokenMetadata storage metadata = tokenMetadata[tokenId];
        Release storage release = releases[metadata.releaseId];

        // Primary recipient gets the royalty, then redistributes
        receiver = release.artist;
        royaltyAmount = (salePrice * defaultRoyaltyBps) / 10000;
    }

    // ============================================================================
    // URI Management
    // ============================================================================

    function uri(uint256 tokenId) public view override returns (string memory) {
        TokenMetadata storage metadata = tokenMetadata[tokenId];
        Release storage release = releases[metadata.releaseId];

        // Return release metadata URI with token-specific query params
        return string(abi.encodePacked(
            release.metadataUri,
            "?tokenId=",
            tokenId.toString(),
            "&type=",
            uint256(metadata.tokenType).toString(),
            "&edition=",
            metadata.edition.toString()
        ));
    }

    function setBaseUri(string calldata newUri) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _baseUri = newUri;
    }

    // ============================================================================
    // View Functions
    // ============================================================================

    function getRelease(uint256 releaseId) external view returns (
        address artist,
        string memory title,
        string memory metadataUri,
        uint256 totalEditions,
        uint256 mintedEditions,
        uint256 price,
        bool isActive
    ) {
        Release storage release = releases[releaseId];
        return (
            release.artist,
            release.title,
            release.metadataUri,
            release.totalEditions,
            release.mintedEditions,
            release.price,
            release.isActive
        );
    }

    function getReleaseRoyalties(uint256 releaseId) external view returns (
        address[] memory recipients,
        uint256[] memory shares,
        string[] memory roles
    ) {
        Release storage release = releases[releaseId];
        uint256 len = release.royalties.length;

        recipients = new address[](len);
        shares = new uint256[](len);
        roles = new string[](len);

        for (uint256 i = 0; i < len; i++) {
            recipients[i] = release.royalties[i].recipient;
            shares[i] = release.royalties[i].share;
            roles[i] = release.royalties[i].role;
        }
    }

    function getTokenMetadata(uint256 tokenId) external view returns (TokenMetadata memory) {
        return tokenMetadata[tokenId];
    }

    // ============================================================================
    // Internal
    // ============================================================================

    function _generateTokenId(
        uint256 releaseId,
        TokenType tokenType,
        uint256 edition
    ) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(releaseId, tokenType, edition)));
    }

    // ============================================================================
    // Required Overrides
    // ============================================================================

    function _update(
        address from,
        address to,
        uint256[] memory ids,
        uint256[] memory values
    ) internal override(ERC1155, ERC1155Supply) {
        super._update(from, to, ids, values);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC1155, AccessControl)
        returns (bool)
    {
        // EIP-2981 royalty standard
        return interfaceId == 0x2a55205a || super.supportsInterface(interfaceId);
    }
}
