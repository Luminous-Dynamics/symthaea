// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/governance/Governor.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorSettings.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorCountingSimple.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorStorage.sol";
import "@openzeppelin/contracts/governance/extensions/GovernorTimelockControl.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title MycelixGovernor
 * @notice Governance contract for Mycelix DAO
 * @dev Uses resonance-weighted voting based on soul participation
 *
 * Key features:
 * - Voting power based on total resonance
 * - Quadratic voting option for fairer representation
 * - Proposal categories for different quorum requirements
 * - Delegation support for vote representation
 */
contract MycelixGovernor is
    Governor,
    GovernorSettings,
    GovernorCountingSimple,
    GovernorStorage,
    GovernorTimelockControl,
    AccessControl
{
    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");

    // Reference to the resonance token for voting power
    IResonanceVotes public immutable resonanceToken;

    // Proposal categories with different quorum requirements
    enum ProposalCategory {
        STANDARD,           // 4% quorum - general proposals
        TREASURY,           // 10% quorum - treasury spending
        PROTOCOL,           // 15% quorum - protocol changes
        CONSTITUTIONAL      // 25% quorum - fundamental changes
    }

    mapping(uint256 => ProposalCategory) public proposalCategories;

    // Quorum percentages (in basis points, 10000 = 100%)
    mapping(ProposalCategory => uint256) public categoryQuorums;

    // Quadratic voting mode
    bool public quadraticVotingEnabled;

    // ============================================================================
    // Events
    // ============================================================================

    event ProposalCategorized(uint256 proposalId, ProposalCategory category);
    event QuadraticVotingToggled(bool enabled);
    event QuorumUpdated(ProposalCategory category, uint256 newQuorum);

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor(
        IResonanceVotes _resonanceToken,
        TimelockController _timelock,
        uint48 _votingDelay,
        uint32 _votingPeriod,
        uint256 _proposalThreshold
    )
        Governor("Mycelix Governor")
        GovernorSettings(_votingDelay, _votingPeriod, _proposalThreshold)
        GovernorTimelockControl(_timelock)
    {
        resonanceToken = _resonanceToken;

        // Set default quorums
        categoryQuorums[ProposalCategory.STANDARD] = 400;      // 4%
        categoryQuorums[ProposalCategory.TREASURY] = 1000;     // 10%
        categoryQuorums[ProposalCategory.PROTOCOL] = 1500;     // 15%
        categoryQuorums[ProposalCategory.CONSTITUTIONAL] = 2500; // 25%

        // Grant roles
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(PROPOSER_ROLE, msg.sender);
        _grantRole(GUARDIAN_ROLE, msg.sender);
    }

    // ============================================================================
    // Proposal Functions
    // ============================================================================

    /**
     * @notice Create a categorized proposal
     */
    function proposeWithCategory(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        ProposalCategory category
    ) public returns (uint256) {
        uint256 proposalId = propose(targets, values, calldatas, description);
        proposalCategories[proposalId] = category;
        emit ProposalCategorized(proposalId, category);
        return proposalId;
    }

    /**
     * @notice Emergency cancel proposal (guardian only)
     */
    function emergencyCancel(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) public onlyRole(GUARDIAN_ROLE) returns (uint256) {
        return _cancel(targets, values, calldatas, descriptionHash);
    }

    // ============================================================================
    // Voting Power
    // ============================================================================

    /**
     * @notice Get voting power for an account
     * @dev Applies quadratic formula if enabled
     */
    function _getVotes(
        address account,
        uint256 timepoint,
        bytes memory /* params */
    ) internal view override returns (uint256) {
        uint256 rawVotes = resonanceToken.getPastVotes(account, timepoint);

        if (quadraticVotingEnabled && rawVotes > 0) {
            // Quadratic voting: sqrt(votes)
            return sqrt(rawVotes);
        }

        return rawVotes;
    }

    /**
     * @notice Toggle quadratic voting mode
     */
    function setQuadraticVoting(bool enabled) external onlyRole(DEFAULT_ADMIN_ROLE) {
        quadraticVotingEnabled = enabled;
        emit QuadraticVotingToggled(enabled);
    }

    // ============================================================================
    // Quorum
    // ============================================================================

    /**
     * @notice Get quorum for a specific proposal
     */
    function quorum(uint256 proposalId) public view override returns (uint256) {
        ProposalCategory category = proposalCategories[proposalId];
        uint256 totalSupply = resonanceToken.getPastTotalSupply(proposalSnapshot(proposalId));
        return (totalSupply * categoryQuorums[category]) / 10000;
    }

    /**
     * @notice Update quorum for a category
     */
    function setQuorum(ProposalCategory category, uint256 newQuorum)
        external
        onlyRole(DEFAULT_ADMIN_ROLE)
    {
        require(newQuorum <= 10000, "Quorum cannot exceed 100%");
        categoryQuorums[category] = newQuorum;
        emit QuorumUpdated(category, newQuorum);
    }

    // ============================================================================
    // Required Overrides
    // ============================================================================

    function votingDelay()
        public
        view
        override(Governor, GovernorSettings)
        returns (uint256)
    {
        return super.votingDelay();
    }

    function votingPeriod()
        public
        view
        override(Governor, GovernorSettings)
        returns (uint256)
    {
        return super.votingPeriod();
    }

    function proposalThreshold()
        public
        view
        override(Governor, GovernorSettings)
        returns (uint256)
    {
        return super.proposalThreshold();
    }

    function state(uint256 proposalId)
        public
        view
        override(Governor, GovernorTimelockControl)
        returns (ProposalState)
    {
        return super.state(proposalId);
    }

    function proposalNeedsQueuing(uint256 proposalId)
        public
        view
        override(Governor, GovernorTimelockControl)
        returns (bool)
    {
        return super.proposalNeedsQueuing(proposalId);
    }

    function _queueOperations(
        uint256 proposalId,
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(Governor, GovernorTimelockControl) returns (uint48) {
        return super._queueOperations(proposalId, targets, values, calldatas, descriptionHash);
    }

    function _executeOperations(
        uint256 proposalId,
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(Governor, GovernorTimelockControl) {
        super._executeOperations(proposalId, targets, values, calldatas, descriptionHash);
    }

    function _cancel(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        bytes32 descriptionHash
    ) internal override(Governor, GovernorTimelockControl) returns (uint256) {
        return super._cancel(targets, values, calldatas, descriptionHash);
    }

    function _executor()
        internal
        view
        override(Governor, GovernorTimelockControl)
        returns (address)
    {
        return super._executor();
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(Governor, AccessControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }

    function _propose(
        address[] memory targets,
        uint256[] memory values,
        bytes[] memory calldatas,
        string memory description,
        address proposer
    ) internal override(Governor, GovernorStorage) returns (uint256) {
        return super._propose(targets, values, calldatas, description, proposer);
    }

    // ============================================================================
    // Utility
    // ============================================================================

    /**
     * @notice Integer square root (Babylonian method)
     */
    function sqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }
}

/**
 * @title IResonanceVotes
 * @notice Interface for the resonance-based voting token
 */
interface IResonanceVotes {
    function getPastVotes(address account, uint256 timepoint) external view returns (uint256);
    function getPastTotalSupply(uint256 timepoint) external view returns (uint256);
    function delegates(address account) external view returns (address);
    function delegate(address delegatee) external;
}
