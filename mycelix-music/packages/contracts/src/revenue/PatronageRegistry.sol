// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title PatronageRegistry
 * @notice Subscription-based artist support system
 * @dev Enables recurring patronage with tiered benefits
 *
 * Features:
 * - Multiple tiers per artist (e.g., Listener, Supporter, Patron, Benefactor)
 * - Monthly subscription model with streaming payments
 * - Exclusive content access based on tier
 * - Patron recognition and on-chain history
 */
contract PatronageRegistry is AccessControl, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant ARTIST_ROLE = keccak256("ARTIST_ROLE");

    struct PatronageTier {
        uint256 id;
        string name;
        uint256 monthlyAmount;      // In wei or token units
        uint256 maxPatrons;         // 0 = unlimited
        uint256 currentPatrons;
        string benefitsUri;         // IPFS URI describing benefits
        bool active;
    }

    struct Patronage {
        address patron;
        address artist;
        uint256 tierId;
        uint256 startedAt;
        uint256 lastPaymentAt;
        uint256 totalContributed;
        bool active;
    }

    struct ArtistProfile {
        address artistAddress;
        string profileUri;
        uint256 totalPatrons;
        uint256 totalEarnings;
        uint256 withdrawableBalance;
        bool registered;
    }

    // Artist address => tier ID => tier details
    mapping(address => mapping(uint256 => PatronageTier)) public tiers;
    mapping(address => uint256) public tierCount;

    // Patron address => artist address => patronage details
    mapping(address => mapping(address => Patronage)) public patronages;

    // Artist profiles
    mapping(address => ArtistProfile) public artists;

    // Platform fee (basis points, 500 = 5%)
    uint256 public platformFee = 500;
    address public feeRecipient;

    // Accepted payment token (address(0) = native ETH)
    IERC20 public paymentToken;

    // Grace period for late payments (7 days)
    uint256 public constant GRACE_PERIOD = 7 days;
    uint256 public constant SUBSCRIPTION_PERIOD = 30 days;

    // ============================================================================
    // Events
    // ============================================================================

    event ArtistRegistered(address indexed artist, string profileUri);
    event TierCreated(address indexed artist, uint256 tierId, string name, uint256 monthlyAmount);
    event TierUpdated(address indexed artist, uint256 tierId);
    event PatronageStarted(address indexed patron, address indexed artist, uint256 tierId);
    event PatronageRenewed(address indexed patron, address indexed artist, uint256 amount);
    event PatronageCanceled(address indexed patron, address indexed artist);
    event ArtistWithdrawal(address indexed artist, uint256 amount);
    event PlatformFeeUpdated(uint256 newFee);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(address _paymentToken, address _feeRecipient) {
        paymentToken = IERC20(_paymentToken);
        feeRecipient = _feeRecipient;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    // ============================================================================
    // Artist Management
    // ============================================================================

    /**
     * @notice Register as an artist
     */
    function registerArtist(string calldata profileUri) external {
        require(!artists[msg.sender].registered, "Already registered");

        artists[msg.sender] = ArtistProfile({
            artistAddress: msg.sender,
            profileUri: profileUri,
            totalPatrons: 0,
            totalEarnings: 0,
            withdrawableBalance: 0,
            registered: true
        });

        _grantRole(ARTIST_ROLE, msg.sender);
        emit ArtistRegistered(msg.sender, profileUri);
    }

    /**
     * @notice Create a patronage tier
     */
    function createTier(
        string calldata name,
        uint256 monthlyAmount,
        uint256 maxPatrons,
        string calldata benefitsUri
    ) external onlyRole(ARTIST_ROLE) returns (uint256) {
        require(monthlyAmount > 0, "Amount must be positive");

        uint256 tierId = tierCount[msg.sender]++;

        tiers[msg.sender][tierId] = PatronageTier({
            id: tierId,
            name: name,
            monthlyAmount: monthlyAmount,
            maxPatrons: maxPatrons,
            currentPatrons: 0,
            benefitsUri: benefitsUri,
            active: true
        });

        emit TierCreated(msg.sender, tierId, name, monthlyAmount);
        return tierId;
    }

    /**
     * @notice Update tier details (not price - that would affect existing patrons)
     */
    function updateTier(
        uint256 tierId,
        string calldata name,
        string calldata benefitsUri,
        bool active
    ) external onlyRole(ARTIST_ROLE) {
        PatronageTier storage tier = tiers[msg.sender][tierId];
        require(tier.monthlyAmount > 0, "Tier doesn't exist");

        tier.name = name;
        tier.benefitsUri = benefitsUri;
        tier.active = active;

        emit TierUpdated(msg.sender, tierId);
    }

    // ============================================================================
    // Patronage Management
    // ============================================================================

    /**
     * @notice Start supporting an artist
     */
    function subscribe(address artist, uint256 tierId) external payable nonReentrant {
        PatronageTier storage tier = tiers[artist][tierId];
        require(tier.active, "Tier not active");
        require(tier.maxPatrons == 0 || tier.currentPatrons < tier.maxPatrons, "Tier full");
        require(!patronages[msg.sender][artist].active, "Already subscribed");

        uint256 amount = tier.monthlyAmount;
        _processPayment(msg.sender, amount);

        // Calculate platform fee
        uint256 fee = (amount * platformFee) / 10000;
        uint256 artistAmount = amount - fee;

        // Update artist balance
        artists[artist].withdrawableBalance += artistAmount;
        artists[artist].totalEarnings += artistAmount;
        artists[artist].totalPatrons++;

        // Update tier
        tier.currentPatrons++;

        // Create patronage record
        patronages[msg.sender][artist] = Patronage({
            patron: msg.sender,
            artist: artist,
            tierId: tierId,
            startedAt: block.timestamp,
            lastPaymentAt: block.timestamp,
            totalContributed: amount,
            active: true
        });

        // Transfer platform fee
        if (fee > 0) {
            _transferFunds(feeRecipient, fee);
        }

        emit PatronageStarted(msg.sender, artist, tierId);
    }

    /**
     * @notice Renew subscription (can be called by anyone to renew)
     */
    function renew(address patron, address artist) external payable nonReentrant {
        Patronage storage patronage = patronages[patron][artist];
        require(patronage.active, "No active patronage");

        PatronageTier storage tier = tiers[artist][patronage.tierId];
        uint256 amount = tier.monthlyAmount;

        // Check if renewal is due
        require(
            block.timestamp >= patronage.lastPaymentAt + SUBSCRIPTION_PERIOD - 1 days,
            "Too early to renew"
        );

        _processPayment(patron, amount);

        // Calculate platform fee
        uint256 fee = (amount * platformFee) / 10000;
        uint256 artistAmount = amount - fee;

        // Update records
        artists[artist].withdrawableBalance += artistAmount;
        artists[artist].totalEarnings += artistAmount;
        patronage.lastPaymentAt = block.timestamp;
        patronage.totalContributed += amount;

        // Transfer platform fee
        if (fee > 0) {
            _transferFunds(feeRecipient, fee);
        }

        emit PatronageRenewed(patron, artist, amount);
    }

    /**
     * @notice Cancel subscription
     */
    function unsubscribe(address artist) external {
        Patronage storage patronage = patronages[msg.sender][artist];
        require(patronage.active, "No active patronage");

        patronage.active = false;
        tiers[artist][patronage.tierId].currentPatrons--;
        artists[artist].totalPatrons--;

        emit PatronageCanceled(msg.sender, artist);
    }

    /**
     * @notice Check if patronage is still active (within grace period)
     */
    function isPatronageActive(address patron, address artist) external view returns (bool) {
        Patronage storage patronage = patronages[patron][artist];
        if (!patronage.active) return false;

        uint256 deadline = patronage.lastPaymentAt + SUBSCRIPTION_PERIOD + GRACE_PERIOD;
        return block.timestamp <= deadline;
    }

    /**
     * @notice Get patron's tier for an artist
     */
    function getPatronTier(address patron, address artist) external view returns (uint256) {
        Patronage storage patronage = patronages[patron][artist];
        if (!patronage.active) return type(uint256).max; // No tier

        uint256 deadline = patronage.lastPaymentAt + SUBSCRIPTION_PERIOD + GRACE_PERIOD;
        if (block.timestamp > deadline) return type(uint256).max; // Expired

        return patronage.tierId;
    }

    // ============================================================================
    // Artist Withdrawals
    // ============================================================================

    /**
     * @notice Withdraw accumulated earnings
     */
    function withdraw() external nonReentrant onlyRole(ARTIST_ROLE) {
        uint256 amount = artists[msg.sender].withdrawableBalance;
        require(amount > 0, "Nothing to withdraw");

        artists[msg.sender].withdrawableBalance = 0;
        _transferFunds(msg.sender, amount);

        emit ArtistWithdrawal(msg.sender, amount);
    }

    /**
     * @notice Withdraw specific amount
     */
    function withdrawAmount(uint256 amount) external nonReentrant onlyRole(ARTIST_ROLE) {
        require(artists[msg.sender].withdrawableBalance >= amount, "Insufficient balance");

        artists[msg.sender].withdrawableBalance -= amount;
        _transferFunds(msg.sender, amount);

        emit ArtistWithdrawal(msg.sender, amount);
    }

    // ============================================================================
    // Admin Functions
    // ============================================================================

    function setPlatformFee(uint256 newFee) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newFee <= 1000, "Max 10% fee");
        platformFee = newFee;
        emit PlatformFeeUpdated(newFee);
    }

    function setFeeRecipient(address newRecipient) external onlyRole(DEFAULT_ADMIN_ROLE) {
        feeRecipient = newRecipient;
    }

    // ============================================================================
    // Internal Functions
    // ============================================================================

    function _processPayment(address from, uint256 amount) internal {
        if (address(paymentToken) == address(0)) {
            require(msg.value >= amount, "Insufficient ETH");
            // Refund excess
            if (msg.value > amount) {
                (bool success, ) = payable(from).call{value: msg.value - amount}("");
                require(success, "Refund failed");
            }
        } else {
            paymentToken.safeTransferFrom(from, address(this), amount);
        }
    }

    function _transferFunds(address to, uint256 amount) internal {
        if (address(paymentToken) == address(0)) {
            (bool success, ) = payable(to).call{value: amount}("");
            require(success, "Transfer failed");
        } else {
            paymentToken.safeTransfer(to, amount);
        }
    }

    // ============================================================================
    // View Functions
    // ============================================================================

    function getArtistTiers(address artist) external view returns (PatronageTier[] memory) {
        uint256 count = tierCount[artist];
        PatronageTier[] memory result = new PatronageTier[](count);

        for (uint256 i = 0; i < count; i++) {
            result[i] = tiers[artist][i];
        }

        return result;
    }

    function getPatronage(address patron, address artist) external view returns (Patronage memory) {
        return patronages[patron][artist];
    }
}
