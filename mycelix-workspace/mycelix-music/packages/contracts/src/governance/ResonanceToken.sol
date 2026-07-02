// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title ResonanceToken
 * @notice Governance token for Mycelix based on accumulated resonance
 * @dev Non-transferable token representing voting power
 *
 * Resonance is earned through:
 * - Listening activity
 * - Artist support/patronage
 * - Community participation
 * - Proof of Presence collection
 * - Collaborative creation
 *
 * Resonance cannot be transferred (soulbound governance)
 * but can be delegated for voting.
 */
contract ResonanceToken is ERC20, ERC20Votes, ERC20Permit, AccessControl {
    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");

    // Reference to Soul Registry for soul verification
    address public soulRegistry;

    // Resonance sources and their weights
    enum ResonanceSource {
        LISTENING,          // Passive listening
        PATRONAGE,          // Artist support
        COMMUNITY,          // Circle participation
        PRESENCE,           // Event attendance
        COLLABORATION,      // Studio contributions
        CREATION            // Publishing original work
    }

    mapping(ResonanceSource => uint256) public sourceWeights;

    // Track resonance by source per user
    mapping(address => mapping(ResonanceSource => uint256)) public resonanceBySource;

    // Decay settings
    uint256 public decayRatePerYear = 500; // 5% annual decay (in basis points)
    mapping(address => uint256) public lastActivityTimestamp;

    // ============================================================================
    // Events
    // ============================================================================

    event ResonanceEarned(
        address indexed account,
        ResonanceSource source,
        uint256 amount,
        string reason
    );

    event ResonanceDecayed(address indexed account, uint256 amount);
    event SourceWeightUpdated(ResonanceSource source, uint256 weight);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(address _soulRegistry)
        ERC20("Mycelix Resonance", "RESO")
        ERC20Permit("Mycelix Resonance")
    {
        soulRegistry = _soulRegistry;

        // Set default source weights (in basis points, 10000 = 1x)
        sourceWeights[ResonanceSource.LISTENING] = 10000;       // 1x
        sourceWeights[ResonanceSource.PATRONAGE] = 20000;       // 2x
        sourceWeights[ResonanceSource.COMMUNITY] = 15000;       // 1.5x
        sourceWeights[ResonanceSource.PRESENCE] = 25000;        // 2.5x
        sourceWeights[ResonanceSource.COLLABORATION] = 30000;   // 3x
        sourceWeights[ResonanceSource.CREATION] = 50000;        // 5x

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    // ============================================================================
    // Resonance Management
    // ============================================================================

    /**
     * @notice Award resonance to a user
     * @dev Only callable by authorized minters (contracts or admin)
     */
    function awardResonance(
        address account,
        ResonanceSource source,
        uint256 baseAmount,
        string calldata reason
    ) external onlyRole(MINTER_ROLE) {
        require(account != address(0), "Invalid account");
        require(baseAmount > 0, "Amount must be positive");

        // Apply source weight
        uint256 weightedAmount = (baseAmount * sourceWeights[source]) / 10000;

        // Apply decay first
        _applyDecay(account);

        // Mint new resonance
        _mint(account, weightedAmount);
        resonanceBySource[account][source] += weightedAmount;
        lastActivityTimestamp[account] = block.timestamp;

        emit ResonanceEarned(account, source, weightedAmount, reason);
    }

    /**
     * @notice Burn resonance (for penalties or redemption)
     */
    function burnResonance(
        address account,
        uint256 amount,
        ResonanceSource source
    ) external onlyRole(BURNER_ROLE) {
        require(balanceOf(account) >= amount, "Insufficient resonance");

        _burn(account, amount);

        if (resonanceBySource[account][source] >= amount) {
            resonanceBySource[account][source] -= amount;
        } else {
            resonanceBySource[account][source] = 0;
        }
    }

    /**
     * @notice Apply time-based decay to account
     */
    function applyDecay(address account) external {
        _applyDecay(account);
    }

    function _applyDecay(address account) internal {
        uint256 lastActivity = lastActivityTimestamp[account];
        if (lastActivity == 0 || lastActivity >= block.timestamp) return;

        uint256 balance = balanceOf(account);
        if (balance == 0) return;

        uint256 timePassed = block.timestamp - lastActivity;
        uint256 yearlyDecay = (balance * decayRatePerYear) / 10000;
        uint256 decayAmount = (yearlyDecay * timePassed) / 365 days;

        if (decayAmount > 0 && decayAmount <= balance) {
            _burn(account, decayAmount);
            emit ResonanceDecayed(account, decayAmount);
        }

        lastActivityTimestamp[account] = block.timestamp;
    }

    // ============================================================================
    // Voting Power
    // ============================================================================

    /**
     * @notice Get current voting power
     */
    function getVotingPower(address account) external view returns (uint256) {
        return getVotes(account);
    }

    /**
     * @notice Get resonance breakdown by source
     */
    function getResonanceBreakdown(address account) external view returns (
        uint256 listening,
        uint256 patronage,
        uint256 community,
        uint256 presence,
        uint256 collaboration,
        uint256 creation
    ) {
        return (
            resonanceBySource[account][ResonanceSource.LISTENING],
            resonanceBySource[account][ResonanceSource.PATRONAGE],
            resonanceBySource[account][ResonanceSource.COMMUNITY],
            resonanceBySource[account][ResonanceSource.PRESENCE],
            resonanceBySource[account][ResonanceSource.COLLABORATION],
            resonanceBySource[account][ResonanceSource.CREATION]
        );
    }

    // ============================================================================
    // Admin Functions
    // ============================================================================

    /**
     * @notice Update source weight
     */
    function setSourceWeight(
        ResonanceSource source,
        uint256 weight
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        sourceWeights[source] = weight;
        emit SourceWeightUpdated(source, weight);
    }

    /**
     * @notice Update decay rate
     */
    function setDecayRate(uint256 newRate) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newRate <= 5000, "Max 50% decay rate");
        decayRatePerYear = newRate;
    }

    /**
     * @notice Update soul registry reference
     */
    function setSoulRegistry(address newRegistry) external onlyRole(DEFAULT_ADMIN_ROLE) {
        soulRegistry = newRegistry;
    }

    // ============================================================================
    // Transfer Restrictions (Soulbound)
    // ============================================================================

    /**
     * @notice Override transfer to prevent non-delegation transfers
     * @dev Resonance is soulbound - cannot be transferred
     */
    function transfer(address, uint256) public pure override returns (bool) {
        revert("Resonance is soulbound");
    }

    function transferFrom(address, address, uint256) public pure override returns (bool) {
        revert("Resonance is soulbound");
    }

    // ============================================================================
    // Required Overrides
    // ============================================================================

    function _update(
        address from,
        address to,
        uint256 value
    ) internal override(ERC20, ERC20Votes) {
        super._update(from, to, value);
    }

    function nonces(address owner) public view override(ERC20Permit, Nonces) returns (uint256) {
        return super.nonces(owner);
    }
}
