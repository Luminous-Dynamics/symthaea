// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {PaymentRouter} from "../src/PaymentRouter.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

// Mock ERC20 for testing
contract MockERC20 is IERC20 {
    string public name = "Mock Token";
    string public symbol = "MOCK";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract PaymentRouterTest is Test {
    PaymentRouter public router;
    MockERC20 public token;

    address public owner = address(1);
    address payable public feeRecipient = payable(address(2));
    address payable public recipient1 = payable(address(3));
    address payable public recipient2 = payable(address(4));
    address public payer = address(5);
    address payable public payee = payable(address(6));

    bytes32 public constant ARBITRATOR_ROLE = keccak256("ARBITRATOR_ROLE");
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    function setUp() public {
        vm.prank(owner);
        router = new PaymentRouter(owner, feeRecipient);

        token = new MockERC20();

        // Whitelist the mock token
        vm.prank(owner);
        router.setTokenWhitelist(address(token), true);

        // Fund payer
        vm.deal(payer, 100 ether);
        token.mint(payer, 100 ether);
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertTrue(router.hasRole(router.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(router.hasRole(GOVERNANCE_ROLE, owner));
        assertTrue(router.hasRole(ARBITRATOR_ROLE, owner));
        assertEq(router.feeRecipient(), feeRecipient);
        assertEq(router.platformFeeBps(), 250); // 2.5%
    }

    function test_Constructor_RevertZeroOwner() public {
        vm.expectRevert(abi.encodeWithSignature("OwnableInvalidOwner(address)", address(0)));
        new PaymentRouter(address(0), feeRecipient);
    }

    // ============ Route Payment Tests ============

    function test_RoutePayment_Native() public {
        bytes32 paymentId = keccak256("payment1");
        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](2);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 7000});
        splits[1] = PaymentRouter.PaymentSplit({recipient: recipient2, shareBps: 3000});

        uint256 amount = 1 ether;
        uint256 expectedFee = (amount * 250) / 10000; // 2.5%
        uint256 distributable = amount - expectedFee;

        uint256 r1Before = recipient1.balance;
        uint256 r2Before = recipient2.balance;

        vm.prank(payer);
        router.routePayment{value: amount}(paymentId, splits, address(0), amount);

        // Check recipient balances
        assertEq(recipient1.balance - r1Before, (distributable * 7000) / 10000);
        assertEq(recipient2.balance - r2Before, (distributable * 3000) / 10000);

        // Check accumulated fees
        assertEq(router.accumulatedFees(address(0)), expectedFee);
    }

    function test_RoutePayment_ERC20() public {
        bytes32 paymentId = keccak256("payment1");
        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](1);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 10000});

        uint256 amount = 10 ether;
        uint256 expectedFee = (amount * 250) / 10000;
        uint256 distributable = amount - expectedFee;

        vm.startPrank(payer);
        token.approve(address(router), amount);
        router.routePayment(paymentId, splits, address(token), amount);
        vm.stopPrank();

        assertEq(token.balanceOf(recipient1), distributable);
        assertEq(router.accumulatedFees(address(token)), expectedFee);
    }

    function test_RoutePayment_RevertInvalidShares() public {
        bytes32 paymentId = keccak256("payment1");
        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](2);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 5000});
        splits[1] = PaymentRouter.PaymentSplit({recipient: recipient2, shareBps: 4000}); // Total 9000, not 10000

        vm.prank(payer);
        vm.expectRevert(PaymentRouter.InvalidShares.selector);
        router.routePayment{value: 1 ether}(paymentId, splits, address(0), 1 ether);
    }

    function test_RoutePayment_RevertTokenNotWhitelisted() public {
        MockERC20 unwhitelistedToken = new MockERC20();
        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](1);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 10000});

        vm.prank(payer);
        vm.expectRevert(PaymentRouter.TokenNotWhitelisted.selector);
        router.routePayment(keccak256("test"), splits, address(unwhitelistedToken), 1 ether);
    }

    // ============ Escrow Tests ============

    function test_CreateEscrow() public {
        bytes32 paymentId = keccak256("escrow1");
        uint256 amount = 1 ether;
        uint256 duration = 7 days;

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: amount}(
            paymentId,
            payee,
            address(0),
            amount,
            duration,
            keccak256("metadata")
        );

        assertEq(escrowId, 1);

        PaymentRouter.Escrow memory escrow = router.getEscrow(escrowId);
        assertEq(escrow.payer, payer);
        assertEq(escrow.payee, payee);
        assertEq(escrow.amount, amount);
        assertEq(uint256(escrow.state), uint256(PaymentRouter.EscrowState.Active));
    }

    function test_ReleaseEscrow_ByPayer() public {
        bytes32 paymentId = keccak256("escrow1");
        uint256 amount = 1 ether;

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: amount}(
            paymentId,
            payee,
            address(0),
            amount,
            7 days,
            keccak256("metadata")
        );

        uint256 payeeBefore = payee.balance;
        uint256 expectedFee = (amount * 250) / 10000;

        vm.prank(payer);
        router.releaseEscrow(escrowId);

        assertEq(payee.balance - payeeBefore, amount - expectedFee);

        PaymentRouter.Escrow memory escrow = router.getEscrow(escrowId);
        assertEq(uint256(escrow.state), uint256(PaymentRouter.EscrowState.Released));
    }

    function test_ReleaseEscrow_AfterTime() public {
        bytes32 paymentId = keccak256("escrow1");
        uint256 amount = 1 ether;
        uint256 duration = 7 days;

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: amount}(
            paymentId,
            payee,
            address(0),
            amount,
            duration,
            keccak256("metadata")
        );

        // Warp past release time
        vm.warp(block.timestamp + duration + 1);

        // Anyone can release after time
        address anyone = address(0x999);
        vm.prank(anyone);
        router.releaseEscrow(escrowId);

        PaymentRouter.Escrow memory escrow = router.getEscrow(escrowId);
        assertEq(uint256(escrow.state), uint256(PaymentRouter.EscrowState.Released));
    }

    function test_ReleaseEscrow_RevertNotReleasable() public {
        bytes32 paymentId = keccak256("escrow1");

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: 1 ether}(
            paymentId,
            payee,
            address(0),
            1 ether,
            7 days,
            keccak256("metadata")
        );

        // Non-payer tries to release before time
        address anyone = address(0x999);
        vm.prank(anyone);
        vm.expectRevert(PaymentRouter.EscrowNotReleasable.selector);
        router.releaseEscrow(escrowId);
    }

    function test_RefundEscrow() public {
        uint256 amount = 1 ether;

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: amount}(
            keccak256("escrow1"),
            payee,
            address(0),
            amount,
            7 days,
            keccak256("metadata")
        );

        uint256 payerBefore = payer.balance;

        // Payee refunds
        vm.prank(payee);
        router.refundEscrow(escrowId);

        // Payer gets full amount back (no fee on refund)
        assertEq(payer.balance - payerBefore, amount);

        PaymentRouter.Escrow memory escrow = router.getEscrow(escrowId);
        assertEq(uint256(escrow.state), uint256(PaymentRouter.EscrowState.Refunded));
    }

    // ============ Dispute Tests ============

    function test_OpenDispute() public {
        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: 1 ether}(
            keccak256("escrow1"),
            payee,
            address(0),
            1 ether,
            7 days,
            keccak256("metadata")
        );

        bytes32 evidenceHash = keccak256("evidence");

        vm.prank(payer);
        uint256 disputeId = router.openDispute(escrowId, evidenceHash);

        assertEq(disputeId, 1);

        PaymentRouter.Dispute memory dispute = router.getDispute(disputeId);
        assertEq(dispute.escrowId, escrowId);
        assertEq(dispute.initiator, payer);
        assertEq(dispute.evidenceHash, evidenceHash);
        assertEq(uint256(dispute.resolution), uint256(PaymentRouter.DisputeResolution.Pending));

        PaymentRouter.Escrow memory escrow = router.getEscrow(escrowId);
        assertEq(uint256(escrow.state), uint256(PaymentRouter.EscrowState.Disputed));
    }

    function test_ResolveDispute_ReleaseToPayee() public {
        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: 1 ether}(
            keccak256("escrow1"),
            payee,
            address(0),
            1 ether,
            7 days,
            keccak256("metadata")
        );

        vm.prank(payer);
        uint256 disputeId = router.openDispute(escrowId, keccak256("evidence"));

        uint256 payeeBefore = payee.balance;

        vm.prank(owner); // Owner has arbitrator role
        router.resolveDispute(disputeId, PaymentRouter.DisputeResolution.ReleaseToPayee, 0);

        PaymentRouter.Dispute memory dispute = router.getDispute(disputeId);
        assertEq(
            uint256(dispute.resolution),
            uint256(PaymentRouter.DisputeResolution.ReleaseToPayee)
        );

        // M-01: Funds are now in pending withdrawals (pull-payment pattern)
        uint256 pendingAmount = router.getPendingWithdrawal(payee, address(0));
        assertTrue(pendingAmount > 0, "Pending withdrawal should be credited");

        // M-01: Payee must withdraw their funds
        vm.prank(payee);
        router.withdrawPending(address(0));

        // Payee received funds (minus fee)
        assertTrue(payee.balance > payeeBefore, "Payee should have received funds");
        assertEq(router.getPendingWithdrawal(payee, address(0)), 0, "Pending should be cleared");
    }

    function test_ResolveDispute_Split() public {
        uint256 amount = 1 ether;

        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: amount}(
            keccak256("escrow1"),
            payee,
            address(0),
            amount,
            7 days,
            keccak256("metadata")
        );

        vm.prank(payer);
        uint256 disputeId = router.openDispute(escrowId, keccak256("evidence"));

        uint256 payerBefore = payer.balance;
        uint256 payeeBefore = payee.balance;

        // 60% to payer, 40% to payee
        vm.prank(owner);
        router.resolveDispute(disputeId, PaymentRouter.DisputeResolution.Split, 60);

        uint256 fee = (amount * 250) / 10000;
        uint256 distributable = amount - fee;

        // M-01: Verify pending withdrawals were credited (pull-payment pattern)
        uint256 payerPending = router.getPendingWithdrawal(payer, address(0));
        uint256 payeePending = router.getPendingWithdrawal(payee, address(0));
        assertTrue(payerPending > 0, "Payer pending should be credited");
        assertTrue(payeePending > 0, "Payee pending should be credited");

        // M-01: Both parties withdraw their funds
        vm.prank(payer);
        router.withdrawPending(address(0));
        vm.prank(payee);
        router.withdrawPending(address(0));

        assertTrue(payer.balance > payerBefore, "Payer should have received funds");
        assertTrue(payee.balance > payeeBefore, "Payee should have received funds");
    }

    // ============ Admin Tests ============

    function test_SetPlatformFee() public {
        vm.prank(owner);
        router.setPlatformFee(300); // 3%

        assertEq(router.platformFeeBps(), 300);
    }

    function test_SetPlatformFee_RevertTooHigh() public {
        vm.prank(owner);
        vm.expectRevert(PaymentRouter.FeeTooHigh.selector);
        router.setPlatformFee(600); // 6% > max 5%
    }

    function test_WithdrawFees() public {
        // Create some fees
        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](1);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 10000});

        vm.prank(payer);
        router.routePayment{value: 1 ether}(keccak256("p1"), splits, address(0), 1 ether);

        uint256 fees = router.accumulatedFees(address(0));
        assertTrue(fees > 0);

        uint256 recipientBefore = feeRecipient.balance;

        vm.prank(owner);
        router.withdrawFees(address(0));

        assertEq(feeRecipient.balance - recipientBefore, fees);
        assertEq(router.accumulatedFees(address(0)), 0);
    }

    function test_Pause() public {
        vm.prank(owner);
        router.pause();

        assertTrue(router.paused());

        PaymentRouter.PaymentSplit[] memory splits = new PaymentRouter.PaymentSplit[](1);
        splits[0] = PaymentRouter.PaymentSplit({recipient: recipient1, shareBps: 10000});

        vm.prank(payer);
        vm.expectRevert();
        router.routePayment{value: 1 ether}(keccak256("p1"), splits, address(0), 1 ether);
    }

    // ============ View Function Tests ============

    function test_CalculateFee() public view {
        uint256 fee = router.calculateFee(1 ether);
        assertEq(fee, 0.025 ether); // 2.5%
    }

    function test_IsEscrowReleasable() public {
        vm.prank(payer);
        uint256 escrowId = router.createEscrow{value: 1 ether}(
            keccak256("escrow1"),
            payee,
            address(0),
            1 ether,
            7 days,
            keccak256("metadata")
        );

        assertFalse(router.isEscrowReleasable(escrowId));

        vm.warp(block.timestamp + 8 days);
        assertTrue(router.isEscrowReleasable(escrowId));
    }
}
