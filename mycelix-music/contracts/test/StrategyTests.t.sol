// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/EconomicStrategyRouter.sol";
import "../src/strategies/FreemiumStrategy.sol";
import "../src/strategies/PatronageStrategy.sol";
import "../src/strategies/NFTGatedStrategy.sol";
import "../src/strategies/PayWhatYouWantStrategy.sol";
import "../src/strategies/PayPerDownloadStrategy.sol";
import "../src/strategies/SubscriptionStrategy.sol";
import "../script/DeployLocal.s.sol";

/**
 * @title StrategyTests
 * @notice Comprehensive tests for all economic strategy contracts
 */
contract StrategyTests is Test {
    EconomicStrategyRouter public router;
    FreemiumStrategy public freemium;
    PatronageStrategy public patronage;
    NFTGatedStrategy public nftGated;
    MockFLOWToken public flowToken;
    MockNFT public mockNFT;

    address public artist = address(0x1);
    address public listener = address(0x2);
    address public listener2 = address(0x3);
    address public protocolTreasury = address(0x4);

    bytes32 public constant FREEMIUM_ID = keccak256("freemium-v1");
    bytes32 public constant PATRONAGE_ID = keccak256("patronage-v1");
    bytes32 public constant NFT_GATED_ID = keccak256("nft-gated-v1");

    function setUp() public {
        // Deploy mock tokens
        flowToken = new MockFLOWToken();
        mockNFT = new MockNFT();

        // Deploy router
        router = new EconomicStrategyRouter(address(flowToken), protocolTreasury);

        // Deploy strategies
        freemium = new FreemiumStrategy(address(flowToken), address(router));
        patronage = new PatronageStrategy(address(flowToken), address(router));
        nftGated = new NFTGatedStrategy(address(flowToken), address(router));

        // Register strategies
        router.registerStrategy(FREEMIUM_ID, address(freemium));
        router.registerStrategy(PATRONAGE_ID, address(patronage));
        router.registerStrategy(NFT_GATED_ID, address(nftGated));

        // Fund test accounts
        flowToken.mint(listener, 1000 ether);
        flowToken.mint(listener2, 1000 ether);

        vm.label(artist, "Artist");
        vm.label(listener, "Listener");
        vm.label(listener2, "Listener2");
        vm.label(protocolTreasury, "Protocol Treasury");
    }

    // ============================================================
    // Freemium Strategy Tests
    // ============================================================

    function testFreemiumFreePlaysThenPaid() public {
        bytes32 songId = keccak256("freemium-song");

        // Register song
        vm.prank(artist);
        router.registerSong(songId, FREEMIUM_ID);

        // Configure freemium: 3 free plays, then 0.01 ETH per play
        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        vm.prank(artist);
        freemium.configureFreemium(
            songId,
            artist,
            3,              // 3 free plays
            0.01 ether,     // price after free
            recipients,
            basisPoints
        );

        // First 3 plays should be free
        for (uint256 i = 0; i < 3; i++) {
            vm.prank(address(router));
            freemium.processPayment(songId, listener, 0, IEconomicStrategy.PaymentType.STREAM);
        }

        // Check listener state
        (uint256 playsUsed, uint256 freePlaysRemaining, uint256 totalPaid) =
            freemium.getListenerState(songId, listener);
        assertEq(playsUsed, 3);
        assertEq(freePlaysRemaining, 0);
        assertEq(totalPaid, 0);

        // 4th play should require payment
        uint256 paymentAmount = 0.01 ether;
        uint256 protocolFee = (paymentAmount * 100) / 10000;
        uint256 netAmount = paymentAmount - protocolFee;

        uint256 artistBalanceBefore = flowToken.balanceOf(artist);

        vm.startPrank(listener);
        flowToken.approve(address(router), paymentAmount);
        router.processPayment(songId, paymentAmount, IEconomicStrategy.PaymentType.STREAM);
        vm.stopPrank();

        // Verify artist received payment
        assertEq(flowToken.balanceOf(artist), artistBalanceBefore + netAmount);
    }

    function testFreemiumMultipleListeners() public {
        bytes32 songId = keccak256("freemium-multi");

        vm.prank(artist);
        router.registerSong(songId, FREEMIUM_ID);

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        vm.prank(artist);
        freemium.configureFreemium(songId, artist, 2, 0.01 ether, recipients, basisPoints);

        // Both listeners should have independent free play counts
        vm.prank(address(router));
        freemium.processPayment(songId, listener, 0, IEconomicStrategy.PaymentType.STREAM);

        vm.prank(address(router));
        freemium.processPayment(songId, listener2, 0, IEconomicStrategy.PaymentType.STREAM);

        // Each listener should have 1 play used, 1 remaining
        (uint256 plays1, uint256 remaining1, ) = freemium.getListenerState(songId, listener);
        (uint256 plays2, uint256 remaining2, ) = freemium.getListenerState(songId, listener2);

        assertEq(plays1, 1);
        assertEq(remaining1, 1);
        assertEq(plays2, 1);
        assertEq(remaining2, 1);
    }

    function testCannotConfigureFreemiumTwice() public {
        bytes32 songId = keccak256("freemium-duplicate");

        vm.prank(artist);
        router.registerSong(songId, FREEMIUM_ID);

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        vm.prank(artist);
        freemium.configureFreemium(songId, artist, 3, 0.01 ether, recipients, basisPoints);

        vm.expectRevert("Already configured");
        vm.prank(artist);
        freemium.configureFreemium(songId, artist, 5, 0.02 ether, recipients, basisPoints);
    }

    // ============================================================
    // Patronage Strategy Tests
    // ============================================================

    function testPatronageJoinTier() public {
        bytes32 artistId = keccak256("artist-1");

        // Configure patronage with tiers
        string[] memory tierNames = new string[](2);
        tierNames[0] = "Basic";
        tierNames[1] = "Premium";

        uint256[] memory monthlyAmounts = new uint256[](2);
        monthlyAmounts[0] = 5 ether;
        monthlyAmounts[1] = 15 ether;

        bool[] memory exclusiveContent = new bool[](2);
        exclusiveContent[0] = false;
        exclusiveContent[1] = true;

        uint256[] memory bonusDownloads = new uint256[](2);
        bonusDownloads[0] = 2;
        bonusDownloads[1] = 10;

        vm.prank(artist);
        patronage.configurePatronage(
            artistId,
            artist,
            tierNames,
            monthlyAmounts,
            exclusiveContent,
            bonusDownloads
        );

        // Listener joins basic tier
        uint256 artistBalanceBefore = flowToken.balanceOf(artist);

        vm.startPrank(listener);
        flowToken.approve(address(patronage), 5 ether);
        patronage.becomePatron(artistId, 0); // Basic tier
        vm.stopPrank();

        // Verify artist received payment
        assertEq(flowToken.balanceOf(artist), artistBalanceBefore + 5 ether);

        // Verify patron info
        (PatronageStrategy.Patron memory patronInfo, bool isActive) =
            patronage.getPatronInfo(artistId, listener);
        assertTrue(isActive);
        assertEq(patronInfo.tierIndex, 0);
        assertEq(patronInfo.totalContributed, 5 ether);
    }

    function testPatronageExclusiveContent() public {
        bytes32 artistId = keccak256("artist-exclusive");
        bytes32 songId = keccak256("exclusive-song");

        // Configure patronage
        string[] memory tierNames = new string[](1);
        tierNames[0] = "Premium";
        uint256[] memory monthlyAmounts = new uint256[](1);
        monthlyAmounts[0] = 10 ether;
        bool[] memory exclusiveContent = new bool[](1);
        exclusiveContent[0] = true;
        uint256[] memory bonusDownloads = new uint256[](1);
        bonusDownloads[0] = 5;

        vm.prank(artist);
        patronage.configurePatronage(artistId, artist, tierNames, monthlyAmounts, exclusiveContent, bonusDownloads);

        // Register patron-only song
        vm.prank(artist);
        patronage.registerSong(songId, artistId, true); // patron only

        // Non-patron should not be authorized
        assertFalse(patronage.isAuthorized(songId, listener));

        // Listener becomes patron
        vm.startPrank(listener);
        flowToken.approve(address(patronage), 10 ether);
        patronage.becomePatron(artistId, 0);
        vm.stopPrank();

        // Patron should be authorized
        assertTrue(patronage.isAuthorized(songId, listener));
    }

    function testPatronageRenewal() public {
        bytes32 artistId = keccak256("artist-renew");

        string[] memory tierNames = new string[](1);
        tierNames[0] = "Monthly";
        uint256[] memory monthlyAmounts = new uint256[](1);
        monthlyAmounts[0] = 5 ether;
        bool[] memory exclusiveContent = new bool[](1);
        exclusiveContent[0] = true;
        uint256[] memory bonusDownloads = new uint256[](1);
        bonusDownloads[0] = 0;

        vm.prank(artist);
        patronage.configurePatronage(artistId, artist, tierNames, monthlyAmounts, exclusiveContent, bonusDownloads);

        // Become patron
        vm.startPrank(listener);
        flowToken.approve(address(patronage), 100 ether);
        patronage.becomePatron(artistId, 0);

        // Fast forward 30 days
        vm.warp(block.timestamp + 30 days);

        // Renew
        patronage.renewPatronage(artistId);
        vm.stopPrank();

        // Check total contributed
        (PatronageStrategy.Patron memory patronInfo, bool isActive) =
            patronage.getPatronInfo(artistId, listener);
        assertTrue(isActive);
        assertEq(patronInfo.totalContributed, 10 ether); // 2 months
    }

    // ============================================================
    // NFT Gated Strategy Tests
    // ============================================================

    function testNFTGatedHolderAccess() public {
        bytes32 songId = keccak256("nft-song");

        // Configure NFT gate
        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        vm.prank(artist);
        nftGated.configureNFTGate(
            songId,
            artist,
            address(mockNFT),
            0,              // any token
            true,           // any token allowed
            0.05 ether,     // non-holder price
            recipients,
            basisPoints
        );

        // Mint NFT to listener
        mockNFT.mint(listener, 1);

        // Check access
        (bool isHolder, uint256 price) = nftGated.checkNFTAccess(songId, listener);
        assertTrue(isHolder);
        assertEq(price, 0);

        // Non-holder should pay
        (bool isHolder2, uint256 price2) = nftGated.checkNFTAccess(songId, listener2);
        assertFalse(isHolder2);
        assertEq(price2, 0.05 ether);
    }

    function testNFTGatedNonHolderPayment() public {
        bytes32 songId = keccak256("nft-paid");

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        vm.prank(artist);
        router.registerSong(songId, NFT_GATED_ID);

        vm.prank(artist);
        nftGated.configureNFTGate(
            songId,
            artist,
            address(mockNFT),
            0,
            true,
            0.05 ether,
            recipients,
            basisPoints
        );

        // Non-holder pays to listen
        uint256 paymentAmount = 0.05 ether;
        uint256 protocolFee = (paymentAmount * 100) / 10000;
        uint256 netAmount = paymentAmount - protocolFee;

        uint256 artistBalanceBefore = flowToken.balanceOf(artist);

        vm.startPrank(listener2); // No NFT
        flowToken.approve(address(router), paymentAmount);
        router.processPayment(songId, paymentAmount, IEconomicStrategy.PaymentType.STREAM);
        vm.stopPrank();

        // Artist should receive payment
        assertEq(flowToken.balanceOf(artist), artistBalanceBefore + netAmount);
    }

    function testNFTGatedSpecificToken() public {
        bytes32 songId = keccak256("nft-specific");

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        // Configure to require specific token ID 42
        vm.prank(artist);
        nftGated.configureNFTGate(
            songId,
            artist,
            address(mockNFT),
            42,             // specific token ID
            false,          // NOT any token
            0,              // no non-holder option
            recipients,
            basisPoints
        );

        // Mint token 1 to listener - should NOT have access
        mockNFT.mint(listener, 1);
        (bool isHolder1, ) = nftGated.checkNFTAccess(songId, listener);
        assertFalse(isHolder1);

        // Mint token 42 to listener2 - should have access
        mockNFT.mint(listener2, 42);
        (bool isHolder2, ) = nftGated.checkNFTAccess(songId, listener2);
        assertTrue(isHolder2);
    }

    // ============================================================
    // Integration Tests
    // ============================================================

    function testChangeSongStrategy() public {
        bytes32 songId = keccak256("strategy-change");

        // Start with freemium
        vm.prank(artist);
        router.registerSong(songId, FREEMIUM_ID);
        assertEq(router.songStrategy(songId), address(freemium));

        // Change to NFT-gated
        vm.prank(artist);
        router.changeSongStrategy(songId, NFT_GATED_ID);
        assertEq(router.songStrategy(songId), address(nftGated));
    }

    function testOnlyArtistCanChangeStrategy() public {
        bytes32 songId = keccak256("artist-only");

        vm.prank(artist);
        router.registerSong(songId, FREEMIUM_ID);

        // Non-artist should fail
        vm.expectRevert("Not song artist");
        vm.prank(listener);
        router.changeSongStrategy(songId, NFT_GATED_ID);
    }

    function testBatchPayments() public {
        bytes32 song1 = keccak256("batch-1");
        bytes32 song2 = keccak256("batch-2");

        // Register two freemium songs
        vm.startPrank(artist);
        router.registerSong(song1, FREEMIUM_ID);
        router.registerSong(song2, FREEMIUM_ID);

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        freemium.configureFreemium(song1, artist, 3, 0.01 ether, recipients, basisPoints);
        freemium.configureFreemium(song2, artist, 3, 0.01 ether, recipients, basisPoints);
        vm.stopPrank();

        // Batch free plays
        bytes32[] memory songIds = new bytes32[](2);
        songIds[0] = song1;
        songIds[1] = song2;

        uint256[] memory amounts = new uint256[](2);
        amounts[0] = 0;
        amounts[1] = 0;

        IEconomicStrategy.PaymentType[] memory types = new IEconomicStrategy.PaymentType[](2);
        types[0] = IEconomicStrategy.PaymentType.STREAM;
        types[1] = IEconomicStrategy.PaymentType.STREAM;

        vm.prank(listener);
        router.batchProcessPayments(songIds, amounts, types);

        // Both songs should have 1 play each
        (uint256 plays1, , ) = freemium.getListenerState(song1, listener);
        (uint256 plays2, , ) = freemium.getListenerState(song2, listener);
        assertEq(plays1, 1);
        assertEq(plays2, 1);
    }
}

// ============================================================
// Mock Contracts
// ============================================================

contract MockNFT is IERC721 {
    mapping(uint256 => address) private _owners;
    mapping(address => uint256) private _balances;

    function mint(address to, uint256 tokenId) external {
        _owners[tokenId] = to;
        _balances[to]++;
    }

    function balanceOf(address owner) external view override returns (uint256) {
        return _balances[owner];
    }

    function ownerOf(uint256 tokenId) external view override returns (address) {
        address owner = _owners[tokenId];
        require(owner != address(0), "Token does not exist");
        return owner;
    }

    // Minimal ERC721 implementation for testing
    function safeTransferFrom(address, address, uint256, bytes calldata) external override {}
    function safeTransferFrom(address, address, uint256) external override {}
    function transferFrom(address, address, uint256) external override {}
    function approve(address, uint256) external override {}
    function setApprovalForAll(address, bool) external override {}
    function getApproved(uint256) external pure override returns (address) { return address(0); }
    function isApprovedForAll(address, address) external pure override returns (bool) { return false; }
    function supportsInterface(bytes4) external pure override returns (bool) { return true; }
}
