// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {ReputationAnchor} from "../src/ReputationAnchor.sol";
import {Merkle} from "murky/Merkle.sol";

contract ReputationAnchorTest is Test {
    ReputationAnchor public anchor;
    Merkle public merkle;

    address public owner = address(1);
    address public anchorNode = address(2);
    address public agent1 = address(3);
    address public agent2 = address(4);
    address public unauthorized = address(5);

    bytes32 public constant ANCHOR_ROLE = keccak256("ANCHOR_ROLE");
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    function setUp() public {
        vm.prank(owner);
        anchor = new ReputationAnchor(owner, anchorNode);
        merkle = new Merkle();
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertTrue(anchor.hasRole(anchor.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(anchor.hasRole(GOVERNANCE_ROLE, owner));
        assertTrue(anchor.hasRole(ANCHOR_ROLE, anchorNode));
    }

    function test_Constructor_RevertZeroOwner() public {
        vm.expectRevert(abi.encodeWithSignature("OwnableInvalidOwner(address)", address(0)));
        new ReputationAnchor(address(0), anchorNode);
    }

    function test_Constructor_RevertZeroAnchor() public {
        vm.expectRevert(ReputationAnchor.ZeroAddress.selector);
        new ReputationAnchor(owner, address(0));
    }

    // ============ Store Reputation Root Tests ============

    function test_StoreReputationRoot() public {
        bytes32 root = keccak256("test root");

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours); // Wait for min interval
        anchor.storeReputationRoot(root);

        assertEq(anchor.currentRoot(), root);
        assertEq(anchor.epoch(), 1);
    }

    function test_StoreReputationRoot_RotatesRoots() public {
        bytes32 root1 = keccak256("root 1");
        bytes32 root2 = keccak256("root 2");

        vm.startPrank(anchorNode);

        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root1);
        assertEq(anchor.currentRoot(), root1);
        assertEq(anchor.previousRoot(), bytes32(0));

        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root2);
        assertEq(anchor.currentRoot(), root2);
        assertEq(anchor.previousRoot(), root1);

        vm.stopPrank();
    }

    function test_StoreReputationRoot_RevertUnauthorized() public {
        bytes32 root = keccak256("test root");

        vm.prank(unauthorized);
        vm.expectRevert();
        anchor.storeReputationRoot(root);
    }

    function test_StoreReputationRoot_RevertInvalidRoot() public {
        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        vm.expectRevert(ReputationAnchor.InvalidRoot.selector);
        anchor.storeReputationRoot(bytes32(0));
    }

    function test_StoreReputationRoot_RevertTooSoon() public {
        bytes32 root1 = keccak256("root 1");
        bytes32 root2 = keccak256("root 2");

        vm.startPrank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root1);

        vm.expectRevert(ReputationAnchor.TooSoonToUpdate.selector);
        anchor.storeReputationRoot(root2);
        vm.stopPrank();
    }

    // ============ Verify Reputation Tests ============

    function test_VerifyReputation() public {
        // Create test data
        uint256 score = 850 * 1e18;
        uint256 nonce = 1;

        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = keccak256(abi.encodePacked(agent2, uint256(500 * 1e18), uint256(1)));

        bytes32 root = merkle.getRoot(data);
        bytes32[] memory proof = merkle.getProof(data, 0);

        // Store the root
        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        // Verify
        bool valid = anchor.verifyReputation(agent1, score, nonce, proof);
        assertTrue(valid);
    }

    function test_VerifyReputation_InvalidProof() public {
        bytes32 root = keccak256("some root");
        bytes32[] memory invalidProof = new bytes32[](1);
        invalidProof[0] = keccak256("wrong");

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        bool valid = anchor.verifyReputation(agent1, 850 * 1e18, 1, invalidProof);
        assertFalse(valid);
    }

    function test_VerifyReputation_RevertScoreTooHigh() public {
        uint256 tooHighScore = 1001 * 1e18;
        bytes32[] memory proof = new bytes32[](0);

        vm.expectRevert(ReputationAnchor.ScoreTooHigh.selector);
        anchor.verifyReputation(agent1, tooHighScore, 1, proof);
    }

    function test_VerifyReputation_GracePeriod() public {
        uint256 score = 850 * 1e18;
        uint256 nonce = 1;

        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32 leaf2 = keccak256(abi.encodePacked(agent2, uint256(500 * 1e18), uint256(2)));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = leaf2;

        bytes32 root1 = merkle.getRoot(data);
        bytes32[] memory proof = merkle.getProof(data, 0);

        // Store first root - advance to 2 hours from now
        uint256 firstStoreTime = block.timestamp + 2 hours;
        vm.warp(firstStoreTime);
        vm.prank(anchorNode);
        anchor.storeReputationRoot(root1);

        // Store second root (different) - advance another 2 hours
        uint256 secondStoreTime = firstStoreTime + 2 hours;
        vm.warp(secondStoreTime);
        vm.prank(anchorNode);
        anchor.storeReputationRoot(keccak256("new root"));

        // During grace period, old proof should still work
        bool valid = anchor.verifyReputation(agent1, score, nonce, proof);
        assertTrue(valid);

        // After grace period, old proof should fail
        vm.warp(block.timestamp + 25 hours);
        valid = anchor.verifyReputation(agent1, score, nonce, proof);
        assertFalse(valid);
    }

    // ============ Admin Function Tests ============

    function test_SetGracePeriod() public {
        uint256 newPeriod = 12 hours;

        vm.prank(owner);
        anchor.setGracePeriod(newPeriod);

        assertEq(anchor.gracePeriod(), newPeriod);
    }

    function test_SetGracePeriod_RevertInvalid() public {
        vm.prank(owner);
        vm.expectRevert(ReputationAnchor.InvalidGracePeriod.selector);
        anchor.setGracePeriod(30 minutes); // Too short
    }

    function test_Pause() public {
        vm.prank(owner);
        anchor.pause();

        assertTrue(anchor.paused());
    }

    function test_StoreReputationRoot_RevertWhenPaused() public {
        vm.prank(owner);
        anchor.pause();

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        vm.expectRevert();
        anchor.storeReputationRoot(keccak256("root"));
    }

    // ============ View Function Tests ============

    function test_GetCurrentState() public {
        bytes32 root = keccak256("test");

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        (bytes32 r, uint256 e, uint256 lastUpdate, bool inGrace) = anchor.getCurrentState();

        assertEq(r, root);
        assertEq(e, 1);
        assertTrue(lastUpdate > 0);
        assertTrue(inGrace);
    }

    function test_GetEpochData() public {
        bytes32 root = keccak256("test");

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        (bytes32 r, uint256 ts) = anchor.getEpochData(1);
        assertEq(r, root);
        assertTrue(ts > 0);
    }

    function test_IsAnchor() public view {
        assertTrue(anchor.isAnchor(anchorNode));
        assertFalse(anchor.isAnchor(unauthorized));
    }

    // ============ Rate Limiting Tests ============

    function test_RateLimiting() public {
        uint256 score = 850 * 1e18;
        uint256 nonce = 1;

        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = keccak256(abi.encodePacked(agent2, uint256(500 * 1e18), uint256(1)));

        bytes32 root = merkle.getRoot(data);
        bytes32[] memory proof = merkle.getProof(data, 0);

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        // Call verifyAndRecord up to the limit
        for (uint256 i = 0; i < 100; i++) {
            anchor.verifyAndRecord(agent1, score, nonce, proof, false);
        }

        // 101st call should revert
        vm.expectRevert(ReputationAnchor.RateLimitExceeded.selector);
        anchor.verifyAndRecord(agent1, score, nonce, proof, false);
    }

    function test_RateLimitResetsOnNewEpoch() public {
        uint256 score = 850 * 1e18;
        uint256 nonce = 1;

        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = keccak256(abi.encodePacked(agent2, uint256(500 * 1e18), uint256(1)));

        bytes32 root = merkle.getRoot(data);
        bytes32[] memory proof = merkle.getProof(data, 0);

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        // Use up rate limit
        for (uint256 i = 0; i < 100; i++) {
            anchor.verifyAndRecord(agent1, score, nonce, proof, false);
        }

        // New epoch
        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(keccak256("new root"));

        // Should work again (different epoch rate limit)
        anchor.verifyAndRecord(agent1, score, nonce, proof, false);
    }

    // ============ Cache Invalidation Event Test ============

    function test_CacheInvalidationEmitsEvent() public {
        uint256 score = 850 * 1e18;
        uint256 nonce = 1;

        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = keccak256(abi.encodePacked(agent2, uint256(500 * 1e18), uint256(1)));

        bytes32 root = merkle.getRoot(data);
        bytes32[] memory proof = merkle.getProof(data, 0);

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        // Cache a score
        anchor.verifyAndRecord(agent1, score, nonce, proof, true);

        // Invalidate and check event
        vm.prank(owner);
        vm.expectEmit(true, true, false, false);
        emit ReputationAnchor.CacheInvalidated(agent1, owner);
        anchor.invalidateCache(agent1);
    }

    // ============ Fuzz Tests ============

    function testFuzz_StoreAndVerify(uint256 score, uint256 nonce) public {
        vm.assume(score <= anchor.MAX_SCORE());
        vm.assume(nonce < type(uint128).max);

        // Murky requires at least 2 leaves for merkle tree generation
        bytes32 leaf = keccak256(abi.encodePacked(agent1, score, nonce));
        bytes32 dummyLeaf = keccak256(abi.encodePacked(agent2, uint256(0), uint256(0)));
        bytes32[] memory data = new bytes32[](2);
        data[0] = leaf;
        data[1] = dummyLeaf;

        bytes32 root = merkle.getRoot(data);

        vm.prank(anchorNode);
        vm.warp(block.timestamp + 2 hours);
        anchor.storeReputationRoot(root);

        bytes32[] memory proof = merkle.getProof(data, 0);
        bool valid = anchor.verifyReputation(agent1, score, nonce, proof);
        assertTrue(valid);
    }
}
