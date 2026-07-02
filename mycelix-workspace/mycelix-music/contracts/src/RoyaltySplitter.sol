// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title RoyaltySplitter
 * @notice Automatic royalty distribution for music streaming
 * @dev Handles proportional payment splits between collaborators
 */
contract RoyaltySplitter is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ==================== Structs ====================

    struct Split {
        address payable recipient;
        uint256 share; // Basis points (1/100 of a percent, so 10000 = 100%)
    }

    struct Song {
        string songId;
        address primaryArtist;
        Split[] splits;
        uint256 totalEarned;
        uint256 totalDistributed;
        bool active;
        uint256 createdAt;
    }

    // ==================== State ====================

    /// @notice Payment token (e.g., USDC)
    IERC20 public paymentToken;

    /// @notice Platform fee in basis points
    uint256 public platformFee = 1500; // 15%

    /// @notice Platform treasury address
    address public treasury;

    /// @notice Minimum payout amount
    uint256 public minPayout = 1e6; // 1 USDC (6 decimals)

    /// @notice Songs registry
    mapping(bytes32 => Song) public songs;

    /// @notice Artist to song IDs
    mapping(address => bytes32[]) public artistSongs;

    /// @notice Pending balances per address
    mapping(address => uint256) public pendingBalances;

    /// @notice Total pending across all addresses
    uint256 public totalPending;

    // ==================== Events ====================

    event SongRegistered(
        bytes32 indexed songHash,
        string songId,
        address indexed primaryArtist,
        uint256 splitCount
    );

    event SongUpdated(bytes32 indexed songHash, uint256 newSplitCount);

    event RoyaltyReceived(
        bytes32 indexed songHash,
        uint256 amount,
        uint256 platformCut,
        uint256 artistCut
    );

    event RoyaltyDistributed(
        bytes32 indexed songHash,
        address indexed recipient,
        uint256 amount
    );

    event PayoutClaimed(address indexed recipient, uint256 amount);

    event PlatformFeeUpdated(uint256 oldFee, uint256 newFee);

    event TreasuryUpdated(address oldTreasury, address newTreasury);

    // ==================== Errors ====================

    error InvalidSplits();
    error SongNotFound();
    error SongAlreadyExists();
    error Unauthorized();
    error InsufficientBalance();
    error TransferFailed();
    error InvalidFee();
    error ZeroAddress();

    // ==================== Constructor ====================

    constructor(address _paymentToken, address _treasury) {
        if (_paymentToken == address(0) || _treasury == address(0)) {
            revert ZeroAddress();
        }
        paymentToken = IERC20(_paymentToken);
        treasury = _treasury;
    }

    // ==================== Song Management ====================

    /**
     * @notice Register a new song with royalty splits
     * @param songId Unique song identifier from the platform
     * @param splits Array of royalty splits (must sum to 10000)
     */
    function registerSong(
        string calldata songId,
        Split[] calldata splits
    ) external returns (bytes32 songHash) {
        songHash = keccak256(abi.encodePacked(songId));

        if (songs[songHash].createdAt != 0) {
            revert SongAlreadyExists();
        }

        _validateSplits(splits);

        Song storage song = songs[songHash];
        song.songId = songId;
        song.primaryArtist = msg.sender;
        song.active = true;
        song.createdAt = block.timestamp;

        for (uint256 i = 0; i < splits.length; i++) {
            song.splits.push(splits[i]);
        }

        artistSongs[msg.sender].push(songHash);

        emit SongRegistered(songHash, songId, msg.sender, splits.length);

        return songHash;
    }

    /**
     * @notice Update splits for an existing song
     * @param songHash Hash of the song
     * @param newSplits New splits array
     */
    function updateSplits(
        bytes32 songHash,
        Split[] calldata newSplits
    ) external {
        Song storage song = songs[songHash];

        if (song.createdAt == 0) {
            revert SongNotFound();
        }

        if (msg.sender != song.primaryArtist && msg.sender != owner()) {
            revert Unauthorized();
        }

        _validateSplits(newSplits);

        // Clear existing splits
        delete song.splits;

        // Add new splits
        for (uint256 i = 0; i < newSplits.length; i++) {
            song.splits.push(newSplits[i]);
        }

        emit SongUpdated(songHash, newSplits.length);
    }

    /**
     * @notice Deactivate a song
     * @param songHash Hash of the song
     */
    function deactivateSong(bytes32 songHash) external {
        Song storage song = songs[songHash];

        if (song.createdAt == 0) {
            revert SongNotFound();
        }

        if (msg.sender != song.primaryArtist && msg.sender != owner()) {
            revert Unauthorized();
        }

        song.active = false;
    }

    // ==================== Royalty Distribution ====================

    /**
     * @notice Receive and distribute royalty payment for a song
     * @param songHash Hash of the song
     * @param amount Amount of payment token
     */
    function receiveRoyalty(
        bytes32 songHash,
        uint256 amount
    ) external nonReentrant {
        Song storage song = songs[songHash];

        if (song.createdAt == 0) {
            revert SongNotFound();
        }

        // Transfer tokens from sender
        paymentToken.safeTransferFrom(msg.sender, address(this), amount);

        // Calculate platform cut
        uint256 platformCut = (amount * platformFee) / 10000;
        uint256 artistCut = amount - platformCut;

        // Send platform fee to treasury
        if (platformCut > 0) {
            paymentToken.safeTransfer(treasury, platformCut);
        }

        // Distribute to splits
        for (uint256 i = 0; i < song.splits.length; i++) {
            Split memory split = song.splits[i];
            uint256 splitAmount = (artistCut * split.share) / 10000;

            if (splitAmount > 0) {
                pendingBalances[split.recipient] += splitAmount;
                totalPending += splitAmount;

                emit RoyaltyDistributed(songHash, split.recipient, splitAmount);
            }
        }

        song.totalEarned += amount;
        song.totalDistributed += artistCut;

        emit RoyaltyReceived(songHash, amount, platformCut, artistCut);
    }

    /**
     * @notice Batch receive royalties for multiple songs
     * @param songHashes Array of song hashes
     * @param amounts Array of amounts
     */
    function batchReceiveRoyalties(
        bytes32[] calldata songHashes,
        uint256[] calldata amounts
    ) external nonReentrant {
        require(songHashes.length == amounts.length, "Length mismatch");

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }

        // Single transfer for all royalties
        paymentToken.safeTransferFrom(msg.sender, address(this), totalAmount);

        for (uint256 i = 0; i < songHashes.length; i++) {
            _processRoyalty(songHashes[i], amounts[i]);
        }
    }

    /**
     * @notice Claim pending balance
     */
    function claimPayout() external nonReentrant {
        uint256 balance = pendingBalances[msg.sender];

        if (balance < minPayout) {
            revert InsufficientBalance();
        }

        pendingBalances[msg.sender] = 0;
        totalPending -= balance;

        paymentToken.safeTransfer(msg.sender, balance);

        emit PayoutClaimed(msg.sender, balance);
    }

    /**
     * @notice Check claimable balance
     * @param account Address to check
     */
    function claimableBalance(address account) external view returns (uint256) {
        return pendingBalances[account];
    }

    // ==================== View Functions ====================

    /**
     * @notice Get song details
     * @param songHash Hash of the song
     */
    function getSong(
        bytes32 songHash
    )
        external
        view
        returns (
            string memory songId,
            address primaryArtist,
            uint256 totalEarned,
            uint256 totalDistributed,
            bool active,
            uint256 splitCount
        )
    {
        Song storage song = songs[songHash];
        return (
            song.songId,
            song.primaryArtist,
            song.totalEarned,
            song.totalDistributed,
            song.active,
            song.splits.length
        );
    }

    /**
     * @notice Get splits for a song
     * @param songHash Hash of the song
     */
    function getSplits(bytes32 songHash) external view returns (Split[] memory) {
        return songs[songHash].splits;
    }

    /**
     * @notice Get songs by artist
     * @param artist Artist address
     */
    function getArtistSongs(address artist) external view returns (bytes32[] memory) {
        return artistSongs[artist];
    }

    // ==================== Admin Functions ====================

    /**
     * @notice Update platform fee
     * @param newFee New fee in basis points (max 3000 = 30%)
     */
    function setPlatformFee(uint256 newFee) external onlyOwner {
        if (newFee > 3000) {
            revert InvalidFee();
        }

        emit PlatformFeeUpdated(platformFee, newFee);
        platformFee = newFee;
    }

    /**
     * @notice Update treasury address
     * @param newTreasury New treasury address
     */
    function setTreasury(address newTreasury) external onlyOwner {
        if (newTreasury == address(0)) {
            revert ZeroAddress();
        }

        emit TreasuryUpdated(treasury, newTreasury);
        treasury = newTreasury;
    }

    /**
     * @notice Update minimum payout
     * @param newMinPayout New minimum payout amount
     */
    function setMinPayout(uint256 newMinPayout) external onlyOwner {
        minPayout = newMinPayout;
    }

    /**
     * @notice Emergency withdraw (only unclaimed tokens)
     * @param token Token to withdraw
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(
        address token,
        uint256 amount
    ) external onlyOwner {
        if (token == address(paymentToken)) {
            // Cannot withdraw more than excess (non-pending) tokens
            uint256 excess = paymentToken.balanceOf(address(this)) - totalPending;
            require(amount <= excess, "Cannot withdraw pending funds");
        }

        IERC20(token).safeTransfer(owner(), amount);
    }

    // ==================== Internal Functions ====================

    function _validateSplits(Split[] calldata splits) internal pure {
        if (splits.length == 0 || splits.length > 10) {
            revert InvalidSplits();
        }

        uint256 totalShares = 0;
        for (uint256 i = 0; i < splits.length; i++) {
            if (splits[i].recipient == address(0)) {
                revert ZeroAddress();
            }
            if (splits[i].share == 0) {
                revert InvalidSplits();
            }
            totalShares += splits[i].share;
        }

        if (totalShares != 10000) {
            revert InvalidSplits();
        }
    }

    function _processRoyalty(bytes32 songHash, uint256 amount) internal {
        Song storage song = songs[songHash];

        if (song.createdAt == 0) {
            revert SongNotFound();
        }

        // Calculate platform cut
        uint256 platformCut = (amount * platformFee) / 10000;
        uint256 artistCut = amount - platformCut;

        // Send platform fee to treasury
        if (platformCut > 0) {
            paymentToken.safeTransfer(treasury, platformCut);
        }

        // Distribute to splits
        for (uint256 i = 0; i < song.splits.length; i++) {
            Split memory split = song.splits[i];
            uint256 splitAmount = (artistCut * split.share) / 10000;

            if (splitAmount > 0) {
                pendingBalances[split.recipient] += splitAmount;
                totalPending += splitAmount;

                emit RoyaltyDistributed(songHash, split.recipient, splitAmount);
            }
        }

        song.totalEarned += amount;
        song.totalDistributed += artistCut;

        emit RoyaltyReceived(songHash, amount, platformCut, artistCut);
    }
}
