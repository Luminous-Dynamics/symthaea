// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/EconomicStrategyRouter.sol";
import "../src/strategies/PayPerStreamStrategy.sol";
import "../src/strategies/GiftEconomyStrategy.sol";
import "../script/DeployLocal.s.sol";

contract EconomicStrategyRouterTest is Test {
    EconomicStrategyRouter public router;
    PayPerStreamStrategy public payPerStream;
    GiftEconomyStrategy public giftEconomy;
    MockFLOWToken public flowToken;
    MockCGCRegistry public cgcRegistry;

    address public artist = address(0x1);
    address public listener = address(0x2);
    address public protocolTreasury = address(0x3);

    bytes32 public constant PAY_PER_STREAM_ID = keccak256("pay-per-stream-v1");
    bytes32 public constant GIFT_ECONOMY_ID = keccak256("gift-economy-v1");

    function setUp() public {
        // Deploy mock tokens
        flowToken = new MockFLOWToken();
        cgcRegistry = new MockCGCRegistry();

        // Deploy router
        router = new EconomicStrategyRouter(address(flowToken), protocolTreasury);

        // Deploy strategies
        payPerStream = new PayPerStreamStrategy(address(flowToken), address(router));
        giftEconomy = new GiftEconomyStrategy(address(flowToken), address(router), address(cgcRegistry));

        // Register strategies
        router.registerStrategy(PAY_PER_STREAM_ID, address(payPerStream));
        router.registerStrategy(GIFT_ECONOMY_ID, address(giftEconomy));

        // Fund test accounts
        flowToken.mint(listener, 1000 ether);

        vm.label(artist, "Artist");
        vm.label(listener, "Listener");
        vm.label(protocolTreasury, "Protocol Treasury");
    }

    // ============================================================
    // Router Tests
    // ============================================================

    function testRegisterStrategy() public {
        bytes32 newStrategyId = keccak256("new-strategy");
        address newStrategy = address(0x999);

        router.registerStrategy(newStrategyId, newStrategy);

        assertEq(router.registeredStrategies(newStrategyId), newStrategy);
    }

    function testCannotRegisterStrategyTwice() public {
        bytes32 strategyId = keccak256("duplicate");
        address strategy = address(0x999);

        router.registerStrategy(strategyId, strategy);

        vm.expectRevert("Strategy already registered");
        router.registerStrategy(strategyId, strategy);
    }

    function testRegisterSong() public {
        bytes32 songId = keccak256("test-song");

        vm.prank(artist);
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);

        assertEq(router.songStrategy(songId), PAY_PER_STREAM_ID);
        assertEq(router.songArtist(songId), artist);
    }

    function testCannotRegisterSongTwice() public {
        bytes32 songId = keccak256("test-song");

        vm.prank(artist);
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);

        vm.prank(artist);
        vm.expectRevert("Song already registered");
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);
    }

    // ============================================================
    // Pay Per Stream Tests
    // ============================================================

    function testPayPerStreamPayment() public {
        bytes32 songId = keccak256("pay-per-stream-song");

        // Register song
        vm.prank(artist);
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);

        // Configure royalty split (100% to artist)
        address[] memory recipients = new address[](1);
        recipients[0] = artist;

        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;

        string[] memory roles = new string[](1);
        roles[0] = "Artist";

        vm.prank(artist);
        payPerStream.configureRoyaltySplit(songId, recipients, basisPoints, roles);

        // Process payment
        uint256 paymentAmount = 0.01 ether;
        uint256 protocolFee = (paymentAmount * 100) / 10000; // 1%
        uint256 netAmount = paymentAmount - protocolFee;

        uint256 artistBalanceBefore = flowToken.balanceOf(artist);
        uint256 treasuryBalanceBefore = flowToken.balanceOf(protocolTreasury);

        vm.startPrank(listener);
        flowToken.approve(address(router), paymentAmount);
        router.processPayment(songId, paymentAmount, IEconomicStrategy.PaymentType.STREAM);
        vm.stopPrank();

        // Verify balances
        assertEq(flowToken.balanceOf(artist), artistBalanceBefore + netAmount);
        assertEq(flowToken.balanceOf(protocolTreasury), treasuryBalanceBefore + protocolFee);
    }

    function testPayPerStreamWithMultipleSplits() public {
        bytes32 songId = keccak256("band-song");
        address member1 = address(0x11);
        address member2 = address(0x12);

        // Register song
        vm.prank(artist);
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);

        // Configure royalty split (50/50)
        address[] memory recipients = new address[](2);
        recipients[0] = member1;
        recipients[1] = member2;

        uint256[] memory basisPoints = new uint256[](2);
        basisPoints[0] = 5000;
        basisPoints[1] = 5000;

        string[] memory roles = new string[](2);
        roles[0] = "Singer";
        roles[1] = "Guitarist";

        vm.prank(artist);
        payPerStream.configureRoyaltySplit(songId, recipients, basisPoints, roles);

        // Process payment
        uint256 paymentAmount = 1 ether;
        uint256 protocolFee = (paymentAmount * 100) / 10000; // 1%
        uint256 netAmount = paymentAmount - protocolFee;

        vm.startPrank(listener);
        flowToken.approve(address(router), paymentAmount);
        router.processPayment(songId, paymentAmount, IEconomicStrategy.PaymentType.STREAM);
        vm.stopPrank();

        // Each member should get 50% of net amount
        assertEq(flowToken.balanceOf(member1), netAmount / 2);
        assertEq(flowToken.balanceOf(member2), netAmount / 2);
    }

    // ============================================================
    // Gift Economy Tests
    // ============================================================

    function testGiftEconomyFreeListening() public {
        bytes32 songId = keccak256("gift-song");

        // Register song
        vm.prank(artist);
        router.registerSong(songId, GIFT_ECONOMY_ID, artist);

        // Configure gift economy
        vm.prank(artist);
        giftEconomy.configureGiftEconomy(
            songId,
            artist,
            1 ether,  // 1 CGC per listen
            10 ether, // 10 CGC early listener bonus
            100,      // First 100 listeners
            15000     // 1.5x multiplier
        );

        // Process free stream (0 payment)
        uint256 cgcBefore = cgcRegistry.cgcBalance(listener);

        vm.prank(address(router));
        giftEconomy.processPayment(songId, listener, 0, IEconomicStrategy.PaymentType.STREAM);

        // Listener should receive CGC + early listener bonus
        uint256 cgcAfter = cgcRegistry.cgcBalance(listener);
        assertEq(cgcAfter, cgcBefore + 11 ether); // 1 CGC + 10 CGC bonus
    }

    function testGiftEconomyTip() public {
        bytes32 songId = keccak256("gift-song");

        // Register song
        vm.prank(artist);
        router.registerSong(songId, GIFT_ECONOMY_ID, artist);

        // Configure gift economy
        vm.prank(artist);
        giftEconomy.configureGiftEconomy(songId, artist, 1 ether, 10 ether, 100, 15000);

        // Process tip
        uint256 tipAmount = 5 ether;
        uint256 protocolFee = (tipAmount * 100) / 10000; // 1%
        uint256 netAmount = tipAmount - protocolFee;

        uint256 artistBalanceBefore = flowToken.balanceOf(artist);

        vm.startPrank(listener);
        flowToken.approve(address(router), tipAmount);
        router.processPayment(songId, tipAmount, IEconomicStrategy.PaymentType.TIP);
        vm.stopPrank();

        // Artist should receive tip (minus protocol fee)
        assertEq(flowToken.balanceOf(artist), artistBalanceBefore + netAmount);

        // Listener should also receive CGC for tipping
        assertGt(cgcRegistry.cgcBalance(listener), 0);
    }

    function testGiftEconomyRepeatListenerBonus() public {
        bytes32 songId = keccak256("gift-song");

        // Register and configure
        vm.prank(artist);
        router.registerSong(songId, GIFT_ECONOMY_ID, artist);

        vm.prank(artist);
        giftEconomy.configureGiftEconomy(songId, artist, 1 ether, 10 ether, 100, 15000);

        // First 6 listens (to trigger repeat listener multiplier)
        for (uint i = 0; i < 6; i++) {
            vm.prank(address(router));
            giftEconomy.processPayment(songId, listener, 0, IEconomicStrategy.PaymentType.STREAM);
        }

        // Check listener profile
        (uint256 totalStreams, , uint256 cgcBalance, bool isEarly) =
            giftEconomy.getListenerProfile(songId, listener);

        assertEq(totalStreams, 6);
        assertGt(cgcBalance, 6 ether); // Should be more due to early + repeat bonuses
        assertTrue(isEarly);
    }

    // ============================================================
    // Protocol Fee Tests
    // ============================================================

    function testProtocolFeeCollection() public {
        bytes32 songId = keccak256("test-song");

        vm.prank(artist);
        router.registerSong(songId, PAY_PER_STREAM_ID, artist);

        address[] memory recipients = new address[](1);
        recipients[0] = artist;
        uint256[] memory basisPoints = new uint256[](1);
        basisPoints[0] = 10000;
        string[] memory roles = new string[](1);
        roles[0] = "Artist";

        vm.prank(artist);
        payPerStream.configureRoyaltySplit(songId, recipients, basisPoints, roles);

        uint256 paymentAmount = 100 ether;
        uint256 expectedFee = 1 ether; // 1% of 100 = 1

        uint256 treasuryBefore = flowToken.balanceOf(protocolTreasury);

        vm.startPrank(listener);
        flowToken.approve(address(router), paymentAmount);
        router.processPayment(songId, paymentAmount, IEconomicStrategy.PaymentType.STREAM);
        vm.stopPrank();

        assertEq(flowToken.balanceOf(protocolTreasury), treasuryBefore + expectedFee);
    }

    function testUpdateProtocolFee() public {
        uint256 newFee = 200; // 2%

        router.updateProtocolFee(newFee);

        assertEq(router.protocolFeeBps(), newFee);
    }

    function testCannotSetFeeTooHigh() public {
        uint256 tooHighFee = 600; // 6% (max is 5%)

        vm.expectRevert("Fee too high (max 5%)");
        router.updateProtocolFee(tooHighFee);
    }
}
