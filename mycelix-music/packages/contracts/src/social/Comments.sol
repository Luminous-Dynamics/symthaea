// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title Comments
 * @notice On-chain commenting system for music content
 * @dev Stores comment hashes on-chain with content on IPFS
 *
 * Features:
 * - Comments on tracks, albums, releases
 * - Nested replies (threaded discussions)
 * - Tipping comments
 * - Upvotes/downvotes (reputation-weighted)
 * - Moderation by content owners
 */
contract Comments is AccessControl, ReentrancyGuard {
    // ============================================================================
    // State
    // ============================================================================

    bytes32 public constant MODERATOR_ROLE = keccak256("MODERATOR_ROLE");

    enum ContentType {
        TRACK,
        ALBUM,
        RELEASE,
        ARTIST_POST,
        CIRCLE
    }

    struct Comment {
        uint256 id;
        address author;
        bytes32 contentHash;      // IPFS hash of comment content
        ContentType contentType;
        uint256 contentId;        // ID of the content being commented on
        uint256 parentId;         // 0 if top-level, otherwise parent comment ID
        uint256 timestamp;
        uint256 upvotes;
        uint256 downvotes;
        uint256 tipAmount;
        bool isHidden;
        bool isDeleted;
    }

    // Comment ID => Comment
    mapping(uint256 => Comment) public comments;
    uint256 public commentCount;

    // Content => Comment IDs (for fetching comments on content)
    mapping(ContentType => mapping(uint256 => uint256[])) public contentComments;

    // Comment ID => Reply IDs
    mapping(uint256 => uint256[]) public commentReplies;

    // User => Comment ID => Vote (-1, 0, 1)
    mapping(address => mapping(uint256 => int8)) public userVotes;

    // User => Total upvotes received (reputation)
    mapping(address => uint256) public userReputation;

    // Content owner mapping (for moderation)
    mapping(ContentType => mapping(uint256 => address)) public contentOwners;

    // Minimum reputation to downvote (prevents brigading)
    uint256 public minReputationToDownvote = 10;

    // ============================================================================
    // Events
    // ============================================================================

    event CommentCreated(
        uint256 indexed commentId,
        address indexed author,
        ContentType contentType,
        uint256 indexed contentId,
        uint256 parentId,
        bytes32 contentHash
    );

    event CommentVoted(
        uint256 indexed commentId,
        address indexed voter,
        int8 vote
    );

    event CommentTipped(
        uint256 indexed commentId,
        address indexed tipper,
        uint256 amount
    );

    event CommentModerated(
        uint256 indexed commentId,
        address indexed moderator,
        bool isHidden
    );

    event CommentDeleted(
        uint256 indexed commentId,
        address indexed deletedBy
    );

    // ============================================================================
    // Constructor
    // ============================================================================

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MODERATOR_ROLE, msg.sender);
    }

    // ============================================================================
    // Comment Creation
    // ============================================================================

    /**
     * @notice Create a new comment on content
     */
    function createComment(
        bytes32 contentHash,
        ContentType contentType,
        uint256 contentId,
        uint256 parentId
    ) external returns (uint256) {
        require(contentHash != bytes32(0), "Empty content");

        // If reply, verify parent exists and belongs to same content
        if (parentId > 0) {
            Comment storage parent = comments[parentId];
            require(!parent.isDeleted, "Parent deleted");
            require(parent.contentType == contentType, "Wrong content type");
            require(parent.contentId == contentId, "Wrong content");
        }

        uint256 commentId = ++commentCount;

        comments[commentId] = Comment({
            id: commentId,
            author: msg.sender,
            contentHash: contentHash,
            contentType: contentType,
            contentId: contentId,
            parentId: parentId,
            timestamp: block.timestamp,
            upvotes: 0,
            downvotes: 0,
            tipAmount: 0,
            isHidden: false,
            isDeleted: false
        });

        // Index by content
        contentComments[contentType][contentId].push(commentId);

        // Index replies
        if (parentId > 0) {
            commentReplies[parentId].push(commentId);
        }

        emit CommentCreated(commentId, msg.sender, contentType, contentId, parentId, contentHash);

        return commentId;
    }

    /**
     * @notice Edit comment (only content hash changes, keeps history)
     */
    function editComment(uint256 commentId, bytes32 newContentHash) external {
        Comment storage comment = comments[commentId];
        require(comment.author == msg.sender, "Not author");
        require(!comment.isDeleted, "Deleted");
        require(newContentHash != bytes32(0), "Empty content");

        comment.contentHash = newContentHash;
    }

    /**
     * @notice Delete own comment
     */
    function deleteComment(uint256 commentId) external {
        Comment storage comment = comments[commentId];
        require(
            comment.author == msg.sender ||
            hasRole(MODERATOR_ROLE, msg.sender) ||
            contentOwners[comment.contentType][comment.contentId] == msg.sender,
            "Not authorized"
        );
        require(!comment.isDeleted, "Already deleted");

        comment.isDeleted = true;

        emit CommentDeleted(commentId, msg.sender);
    }

    // ============================================================================
    // Voting
    // ============================================================================

    /**
     * @notice Upvote a comment
     */
    function upvote(uint256 commentId) external {
        Comment storage comment = comments[commentId];
        require(!comment.isDeleted, "Deleted");
        require(comment.author != msg.sender, "Cannot vote own comment");

        int8 currentVote = userVotes[msg.sender][commentId];

        if (currentVote == 1) {
            // Remove upvote
            comment.upvotes--;
            userVotes[msg.sender][commentId] = 0;
            userReputation[comment.author]--;
        } else if (currentVote == -1) {
            // Switch from downvote to upvote
            comment.downvotes--;
            comment.upvotes++;
            userVotes[msg.sender][commentId] = 1;
            userReputation[comment.author] += 2;
        } else {
            // New upvote
            comment.upvotes++;
            userVotes[msg.sender][commentId] = 1;
            userReputation[comment.author]++;
        }

        emit CommentVoted(commentId, msg.sender, userVotes[msg.sender][commentId]);
    }

    /**
     * @notice Downvote a comment (requires minimum reputation)
     */
    function downvote(uint256 commentId) external {
        require(userReputation[msg.sender] >= minReputationToDownvote, "Insufficient reputation");

        Comment storage comment = comments[commentId];
        require(!comment.isDeleted, "Deleted");
        require(comment.author != msg.sender, "Cannot vote own comment");

        int8 currentVote = userVotes[msg.sender][commentId];

        if (currentVote == -1) {
            // Remove downvote
            comment.downvotes--;
            userVotes[msg.sender][commentId] = 0;
            userReputation[comment.author]++;
        } else if (currentVote == 1) {
            // Switch from upvote to downvote
            comment.upvotes--;
            comment.downvotes++;
            userVotes[msg.sender][commentId] = -1;
            userReputation[comment.author] -= 2;
        } else {
            // New downvote
            comment.downvotes++;
            userVotes[msg.sender][commentId] = -1;
            userReputation[comment.author]--;
        }

        emit CommentVoted(commentId, msg.sender, userVotes[msg.sender][commentId]);
    }

    // ============================================================================
    // Tipping
    // ============================================================================

    /**
     * @notice Tip a comment author
     */
    function tipComment(uint256 commentId) external payable nonReentrant {
        require(msg.value > 0, "No tip");

        Comment storage comment = comments[commentId];
        require(!comment.isDeleted, "Deleted");
        require(comment.author != msg.sender, "Cannot tip self");

        comment.tipAmount += msg.value;

        // Transfer tip to author
        (bool success, ) = payable(comment.author).call{value: msg.value}("");
        require(success, "Transfer failed");

        emit CommentTipped(commentId, msg.sender, msg.value);
    }

    // ============================================================================
    // Moderation
    // ============================================================================

    /**
     * @notice Hide/unhide a comment (moderation)
     */
    function moderateComment(uint256 commentId, bool hide) external {
        Comment storage comment = comments[commentId];
        require(
            hasRole(MODERATOR_ROLE, msg.sender) ||
            contentOwners[comment.contentType][comment.contentId] == msg.sender,
            "Not moderator"
        );

        comment.isHidden = hide;

        emit CommentModerated(commentId, msg.sender, hide);
    }

    /**
     * @notice Set content owner (can moderate comments on their content)
     */
    function setContentOwner(
        ContentType contentType,
        uint256 contentId,
        address owner
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        contentOwners[contentType][contentId] = owner;
    }

    /**
     * @notice Update minimum reputation for downvoting
     */
    function setMinReputationToDownvote(uint256 minRep) external onlyRole(DEFAULT_ADMIN_ROLE) {
        minReputationToDownvote = minRep;
    }

    // ============================================================================
    // View Functions
    // ============================================================================

    /**
     * @notice Get comments for content
     */
    function getContentCommentIds(
        ContentType contentType,
        uint256 contentId
    ) external view returns (uint256[] memory) {
        return contentComments[contentType][contentId];
    }

    /**
     * @notice Get reply IDs for a comment
     */
    function getReplyIds(uint256 commentId) external view returns (uint256[] memory) {
        return commentReplies[commentId];
    }

    /**
     * @notice Get comment details
     */
    function getComment(uint256 commentId) external view returns (
        address author,
        bytes32 contentHash,
        ContentType contentType,
        uint256 contentId,
        uint256 parentId,
        uint256 timestamp,
        uint256 upvotes,
        uint256 downvotes,
        uint256 tipAmount,
        bool isHidden,
        bool isDeleted
    ) {
        Comment storage c = comments[commentId];
        return (
            c.author,
            c.contentHash,
            c.contentType,
            c.contentId,
            c.parentId,
            c.timestamp,
            c.upvotes,
            c.downvotes,
            c.tipAmount,
            c.isHidden,
            c.isDeleted
        );
    }

    /**
     * @notice Get user's vote on a comment
     */
    function getUserVote(address user, uint256 commentId) external view returns (int8) {
        return userVotes[user][commentId];
    }

    /**
     * @notice Get comment count for content
     */
    function getCommentCount(
        ContentType contentType,
        uint256 contentId
    ) external view returns (uint256) {
        return contentComments[contentType][contentId].length;
    }

    /**
     * @notice Get top-level comments (paginated)
     */
    function getTopLevelComments(
        ContentType contentType,
        uint256 contentId,
        uint256 offset,
        uint256 limit
    ) external view returns (uint256[] memory) {
        uint256[] storage all = contentComments[contentType][contentId];
        uint256 count = 0;

        // Count top-level comments
        for (uint256 i = 0; i < all.length; i++) {
            if (comments[all[i]].parentId == 0) count++;
        }

        // Collect with pagination
        uint256 resultSize = limit;
        if (offset + limit > count) {
            resultSize = count > offset ? count - offset : 0;
        }

        uint256[] memory result = new uint256[](resultSize);
        uint256 found = 0;
        uint256 skipped = 0;

        for (uint256 i = 0; i < all.length && found < resultSize; i++) {
            if (comments[all[i]].parentId == 0) {
                if (skipped >= offset) {
                    result[found++] = all[i];
                } else {
                    skipped++;
                }
            }
        }

        return result;
    }

    /**
     * @notice Calculate comment score (upvotes - downvotes, weighted by age)
     */
    function getCommentScore(uint256 commentId) external view returns (int256) {
        Comment storage c = comments[commentId];
        int256 rawScore = int256(c.upvotes) - int256(c.downvotes);

        // Time decay factor (older comments get slight boost to prevent new comment domination)
        uint256 age = block.timestamp - c.timestamp;
        uint256 ageBoost = age / 1 hours; // +1 per hour

        return rawScore + int256(ageBoost > 24 ? 24 : ageBoost);
    }
}
