// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./CdnRegistry.sol";

/// @title CdnSlashing - Penalty Mechanism for CDN Node Misbehavior
/// @notice Handles reporting, verification, and slashing of misbehaving CDN nodes
/// @dev Integrates with Mycelix-Core Byzantine detection for off-chain verification
contract CdnSlashing {
    // ============ Enums ============

    /// @notice Types of slashable offenses
    enum OffenseType {
        ContentCorruption,   // Serving corrupted content
        FakeMetrics,         // Reporting false metrics
        Unavailability,      // Extended downtime
        SlowResponse,        // Consistently slow responses
        SybilAttack,         // Multiple fake identities
        ByzantineBehavior    // General Byzantine behavior
    }

    /// @notice Status of a slash report
    enum ReportStatus {
        Pending,
        UnderReview,
        Confirmed,
        Rejected,
        Executed
    }

    // ============ Structs ============

    /// @notice Slash report
    struct SlashReport {
        uint256 id;
        address reporter;
        address accused;
        OffenseType offense;
        string evidence;
        uint256 reportedAt;
        ReportStatus status;
        uint256 slashAmount;
        uint256 reviewerCount;
        uint256 confirmVotes;
        uint256 rejectVotes;
    }

    // ============ State Variables ============

    /// @notice Registry contract
    CdnRegistry public registry;

    /// @notice Owner address
    address public owner;

    /// @notice Reviewers who can vote on reports
    mapping(address => bool) public isReviewer;

    /// @notice Report counter
    uint256 public reportCount;

    /// @notice All reports
    mapping(uint256 => SlashReport) public reports;

    /// @notice Reports by accused node
    mapping(address => uint256[]) public reportsByNode;

    /// @notice Votes cast by reviewers on reports
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    /// @notice Slash amounts per offense type (basis points of stake)
    mapping(OffenseType => uint256) public slashAmounts;

    /// @notice Required votes to confirm a report
    uint256 public requiredVotes = 3;

    /// @notice Review period before auto-rejection
    uint256 public reviewPeriod = 3 days;

    /// @notice Reward for valid reports (basis points of slash)
    uint256 public reporterRewardBps = 1000; // 10%

    /// @notice Minimum reporter stake
    uint256 public minReporterStake = 10 ether;

    /// @notice Stakes by reporter
    mapping(address => uint256) public reporterStakes;

    // ============ Events ============

    event ReportSubmitted(
        uint256 indexed reportId,
        address indexed reporter,
        address indexed accused,
        OffenseType offense
    );

    event ReportReviewed(
        uint256 indexed reportId,
        address indexed reviewer,
        bool confirmed
    );

    event SlashExecuted(
        uint256 indexed reportId,
        address indexed accused,
        uint256 amount
    );

    event ReportRejected(
        uint256 indexed reportId,
        address indexed accused
    );

    event ReviewerAdded(address indexed reviewer);
    event ReviewerRemoved(address indexed reviewer);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyReviewer() {
        require(isReviewer[msg.sender], "Not reviewer");
        _;
    }

    // ============ Constructor ============

    constructor(address _registry) {
        owner = msg.sender;
        registry = CdnRegistry(payable(_registry));

        // Set default slash amounts (basis points of stake)
        slashAmounts[OffenseType.ContentCorruption] = 5000;  // 50%
        slashAmounts[OffenseType.FakeMetrics] = 3000;        // 30%
        slashAmounts[OffenseType.Unavailability] = 1000;     // 10%
        slashAmounts[OffenseType.SlowResponse] = 500;        // 5%
        slashAmounts[OffenseType.SybilAttack] = 10000;       // 100%
        slashAmounts[OffenseType.ByzantineBehavior] = 5000;  // 50%
    }

    // ============ External Functions ============

    /// @notice Submit a slash report
    /// @param accused Address of the accused node
    /// @param offense Type of offense
    /// @param evidence IPFS hash or URL to evidence
    function submitReport(
        address accused,
        OffenseType offense,
        string calldata evidence
    ) external payable returns (uint256) {
        require(msg.value >= minReporterStake, "Insufficient stake");
        require(bytes(evidence).length > 0, "Evidence required");

        ICdnNode.Node memory node = registry.getNode(accused);
        require(node.operator != address(0), "Node not found");

        reportCount++;
        uint256 reportId = reportCount;

        uint256 potentialSlash = (node.stakedAmount * slashAmounts[offense]) / 10000;

        reports[reportId] = SlashReport({
            id: reportId,
            reporter: msg.sender,
            accused: accused,
            offense: offense,
            evidence: evidence,
            reportedAt: block.timestamp,
            status: ReportStatus.Pending,
            slashAmount: potentialSlash,
            reviewerCount: 0,
            confirmVotes: 0,
            rejectVotes: 0
        });

        reportsByNode[accused].push(reportId);
        reporterStakes[msg.sender] += msg.value;

        emit ReportSubmitted(reportId, msg.sender, accused, offense);

        return reportId;
    }

    /// @notice Vote on a slash report
    /// @param reportId Report ID to vote on
    /// @param confirm True to confirm, false to reject
    function voteOnReport(uint256 reportId, bool confirm) external onlyReviewer {
        SlashReport storage report = reports[reportId];
        require(report.id != 0, "Report not found");
        require(report.status == ReportStatus.Pending || report.status == ReportStatus.UnderReview, "Cannot vote");
        require(!hasVoted[reportId][msg.sender], "Already voted");

        hasVoted[reportId][msg.sender] = true;
        report.reviewerCount++;

        if (confirm) {
            report.confirmVotes++;
        } else {
            report.rejectVotes++;
        }

        report.status = ReportStatus.UnderReview;

        emit ReportReviewed(reportId, msg.sender, confirm);

        // Check if we have enough votes to decide
        if (report.confirmVotes >= requiredVotes) {
            _executeSlash(reportId);
        } else if (report.rejectVotes >= requiredVotes) {
            _rejectReport(reportId);
        }
    }

    /// @notice Auto-reject expired reports
    /// @param reportId Report ID to check
    function checkExpiredReport(uint256 reportId) external {
        SlashReport storage report = reports[reportId];
        require(report.id != 0, "Report not found");
        require(
            report.status == ReportStatus.Pending || report.status == ReportStatus.UnderReview,
            "Already resolved"
        );
        require(
            block.timestamp > report.reportedAt + reviewPeriod,
            "Review period not over"
        );

        // If not enough confirmations, reject
        if (report.confirmVotes < requiredVotes) {
            _rejectReport(reportId);
        }
    }

    /// @notice Get reports for a node
    /// @param node Node address
    /// @return Array of report IDs
    function getReportsForNode(address node) external view returns (uint256[] memory) {
        return reportsByNode[node];
    }

    /// @notice Get report details
    /// @param reportId Report ID
    /// @return Report struct
    function getReport(uint256 reportId) external view returns (SlashReport memory) {
        return reports[reportId];
    }

    /// @notice Withdraw reporter stake after resolution
    /// @param amount Amount to withdraw
    function withdrawReporterStake(uint256 amount) external {
        require(reporterStakes[msg.sender] >= amount, "Insufficient stake");
        reporterStakes[msg.sender] -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }

    // ============ Internal Functions ============

    /// @notice Execute a confirmed slash
    /// @param reportId Report ID
    function _executeSlash(uint256 reportId) internal {
        SlashReport storage report = reports[reportId];
        report.status = ReportStatus.Executed;

        // Execute slash in registry
        registry.slash(
            report.accused,
            report.slashAmount,
            _offenseToString(report.offense)
        );

        // Reward reporter
        uint256 reporterReward = (report.slashAmount * reporterRewardBps) / 10000;
        reporterStakes[report.reporter] += reporterReward;

        emit SlashExecuted(reportId, report.accused, report.slashAmount);
    }

    /// @notice Reject a report
    /// @param reportId Report ID
    function _rejectReport(uint256 reportId) internal {
        SlashReport storage report = reports[reportId];
        report.status = ReportStatus.Rejected;

        // Reporter loses stake as penalty for false report
        uint256 penalty = reporterStakes[report.reporter] / 2;
        reporterStakes[report.reporter] -= penalty;

        emit ReportRejected(reportId, report.accused);
    }

    /// @notice Convert offense type to string
    function _offenseToString(OffenseType offense) internal pure returns (string memory) {
        if (offense == OffenseType.ContentCorruption) return "ContentCorruption";
        if (offense == OffenseType.FakeMetrics) return "FakeMetrics";
        if (offense == OffenseType.Unavailability) return "Unavailability";
        if (offense == OffenseType.SlowResponse) return "SlowResponse";
        if (offense == OffenseType.SybilAttack) return "SybilAttack";
        return "ByzantineBehavior";
    }

    // ============ Admin Functions ============

    /// @notice Add a reviewer
    /// @param reviewer Address to add
    function addReviewer(address reviewer) external onlyOwner {
        isReviewer[reviewer] = true;
        emit ReviewerAdded(reviewer);
    }

    /// @notice Remove a reviewer
    /// @param reviewer Address to remove
    function removeReviewer(address reviewer) external onlyOwner {
        isReviewer[reviewer] = false;
        emit ReviewerRemoved(reviewer);
    }

    /// @notice Set slash amount for offense type
    /// @param offense Offense type
    /// @param bps Basis points of stake to slash
    function setSlashAmount(OffenseType offense, uint256 bps) external onlyOwner {
        require(bps <= 10000, "Invalid basis points");
        slashAmounts[offense] = bps;
    }

    /// @notice Set required votes
    /// @param _requiredVotes New required votes
    function setRequiredVotes(uint256 _requiredVotes) external onlyOwner {
        require(_requiredVotes > 0, "Must require at least 1 vote");
        requiredVotes = _requiredVotes;
    }

    /// @notice Set review period
    /// @param _period New review period
    function setReviewPeriod(uint256 _period) external onlyOwner {
        reviewPeriod = _period;
    }

    /// @notice Set reporter reward
    /// @param _bps New reporter reward in basis points
    function setReporterReward(uint256 _bps) external onlyOwner {
        require(_bps <= 5000, "Max 50% reward");
        reporterRewardBps = _bps;
    }

    /// @notice Set minimum reporter stake
    /// @param _stake New minimum stake
    function setMinReporterStake(uint256 _stake) external onlyOwner {
        minReporterStake = _stake;
    }

    // ============ Receive ============

    receive() external payable {}
}
