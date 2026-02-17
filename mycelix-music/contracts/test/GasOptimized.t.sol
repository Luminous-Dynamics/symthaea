// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";
import "../src/libraries/GasOptimized.sol";

contract GasOptimizedTest is Test {
    using GasOptimized for uint256;
    using GasOptimized for address[];

    // ============================================================
    // Math Operations
    // ============================================================

    function test_uncheckedInc() public pure {
        assertEq(GasOptimized.uncheckedInc(0), 1);
        assertEq(GasOptimized.uncheckedInc(100), 101);
        assertEq(GasOptimized.uncheckedInc(type(uint256).max - 1), type(uint256).max);
    }

    function test_uncheckedDec() public pure {
        assertEq(GasOptimized.uncheckedDec(1), 0);
        assertEq(GasOptimized.uncheckedDec(100), 99);
    }

    function test_percentage() public pure {
        // 50% of 100 = 50
        assertEq(GasOptimized.percentage(100, 5000), 50);

        // 100% of 100 = 100
        assertEq(GasOptimized.percentage(100, 10000), 100);

        // 25% of 1000 = 250
        assertEq(GasOptimized.percentage(1000, 2500), 250);

        // 1% of 10000 = 100
        assertEq(GasOptimized.percentage(10000, 100), 100);

        // 0% = 0
        assertEq(GasOptimized.percentage(1000, 0), 0);
    }

    function testFuzz_percentage(uint256 amount, uint256 bps) public pure {
        vm.assume(amount < type(uint128).max);
        vm.assume(bps <= 10000);

        uint256 result = GasOptimized.percentage(amount, bps);
        assertLe(result, amount);
    }

    function test_calculateShare() public pure {
        // 1/4 of 100 = 25
        assertEq(GasOptimized.calculateShare(100, 1, 4, false), 25);

        // 1/3 of 100 = 33 (rounded down)
        assertEq(GasOptimized.calculateShare(100, 1, 3, false), 33);

        // 1/3 of 100 = 34 (rounded up)
        assertEq(GasOptimized.calculateShare(100, 1, 3, true), 34);

        // 2/3 of 99 = 66
        assertEq(GasOptimized.calculateShare(99, 2, 3, false), 66);

        // 2/3 of 99 = 66 (rounded up)
        assertEq(GasOptimized.calculateShare(99, 2, 3, true), 66);
    }

    function test_sqrt() public pure {
        assertEq(GasOptimized.sqrt(0), 0);
        assertEq(GasOptimized.sqrt(1), 1);
        assertEq(GasOptimized.sqrt(4), 2);
        assertEq(GasOptimized.sqrt(9), 3);
        assertEq(GasOptimized.sqrt(16), 4);
        assertEq(GasOptimized.sqrt(100), 10);
        assertEq(GasOptimized.sqrt(10000), 100);

        // Non-perfect squares (should floor)
        assertEq(GasOptimized.sqrt(2), 1);
        assertEq(GasOptimized.sqrt(3), 1);
        assertEq(GasOptimized.sqrt(5), 2);
        assertEq(GasOptimized.sqrt(10), 3);
        assertEq(GasOptimized.sqrt(99), 9);
    }

    function testFuzz_sqrt(uint256 x) public pure {
        uint256 y = GasOptimized.sqrt(x);

        // y^2 <= x
        assertTrue(y * y <= x);

        // (y+1)^2 > x (unless overflow)
        if (y < type(uint128).max) {
            assertTrue((y + 1) * (y + 1) > x);
        }
    }

    function test_min() public pure {
        assertEq(GasOptimized.min(1, 2), 1);
        assertEq(GasOptimized.min(2, 1), 1);
        assertEq(GasOptimized.min(5, 5), 5);
        assertEq(GasOptimized.min(0, type(uint256).max), 0);
    }

    function test_max() public pure {
        assertEq(GasOptimized.max(1, 2), 2);
        assertEq(GasOptimized.max(2, 1), 2);
        assertEq(GasOptimized.max(5, 5), 5);
        assertEq(GasOptimized.max(0, type(uint256).max), type(uint256).max);
    }

    // ============================================================
    // Array Operations
    // ============================================================

    function test_sum() public pure {
        uint256[] memory arr = new uint256[](4);
        arr[0] = 10;
        arr[1] = 20;
        arr[2] = 30;
        arr[3] = 40;

        assertEq(GasOptimized.sum(arr), 100);
    }

    function test_sumEmptyArray() public pure {
        uint256[] memory arr = new uint256[](0);
        assertEq(GasOptimized.sum(arr), 0);
    }

    function test_contains() public pure {
        address[] memory arr = new address[](3);
        arr[0] = address(0x1);
        arr[1] = address(0x2);
        arr[2] = address(0x3);

        assertTrue(GasOptimized.contains(arr, address(0x2)));
        assertFalse(GasOptimized.contains(arr, address(0x4)));
    }

    function test_indexOf() public pure {
        address[] memory arr = new address[](3);
        arr[0] = address(0x1);
        arr[1] = address(0x2);
        arr[2] = address(0x3);

        assertEq(GasOptimized.indexOf(arr, address(0x1)), 0);
        assertEq(GasOptimized.indexOf(arr, address(0x2)), 1);
        assertEq(GasOptimized.indexOf(arr, address(0x3)), 2);
        assertEq(GasOptimized.indexOf(arr, address(0x4)), type(uint256).max);
    }

    // ============================================================
    // Bit Operations
    // ============================================================

    function test_getBit() public pure {
        uint256 bitmap = 0b1010; // bits 1 and 3 are set

        assertFalse(GasOptimized.getBit(bitmap, 0));
        assertTrue(GasOptimized.getBit(bitmap, 1));
        assertFalse(GasOptimized.getBit(bitmap, 2));
        assertTrue(GasOptimized.getBit(bitmap, 3));
    }

    function test_setBit() public pure {
        uint256 bitmap = 0;

        bitmap = GasOptimized.setBit(bitmap, 0);
        assertEq(bitmap, 1);

        bitmap = GasOptimized.setBit(bitmap, 3);
        assertEq(bitmap, 0b1001);
    }

    function test_clearBit() public pure {
        uint256 bitmap = 0b1111;

        bitmap = GasOptimized.clearBit(bitmap, 1);
        assertEq(bitmap, 0b1101);

        bitmap = GasOptimized.clearBit(bitmap, 3);
        assertEq(bitmap, 0b0101);
    }

    function test_toggleBit() public pure {
        uint256 bitmap = 0b1010;

        // Toggle bit 0 (off -> on)
        bitmap = GasOptimized.toggleBit(bitmap, 0);
        assertEq(bitmap, 0b1011);

        // Toggle bit 1 (on -> off)
        bitmap = GasOptimized.toggleBit(bitmap, 1);
        assertEq(bitmap, 0b1001);
    }

    function test_popCount() public pure {
        assertEq(GasOptimized.popCount(0), 0);
        assertEq(GasOptimized.popCount(1), 1);
        assertEq(GasOptimized.popCount(0b1010), 2);
        assertEq(GasOptimized.popCount(0b1111), 4);
        assertEq(GasOptimized.popCount(0xFF), 8);
        assertEq(GasOptimized.popCount(type(uint256).max), 256);
    }

    // ============================================================
    // Hashing
    // ============================================================

    function test_hash2Addresses() public pure {
        address a = address(0x1);
        address b = address(0x2);

        bytes32 hash1 = GasOptimized.hash2Addresses(a, b);
        bytes32 hash2 = GasOptimized.hash2Addresses(a, b);
        bytes32 hash3 = GasOptimized.hash2Addresses(b, a);

        // Same inputs = same hash
        assertEq(hash1, hash2);

        // Order matters
        assertTrue(hash1 != hash3);
    }

    function test_hashAddressUint() public pure {
        address a = address(0x1);
        uint256 n = 123;

        bytes32 hash = GasOptimized.hashAddressUint(a, n);
        assertEq(hash, keccak256(abi.encodePacked(a, n)));
    }

    function test_createId() public {
        address creator = address(0x1);
        uint256 nonce = 1;
        uint256 timestamp = block.timestamp;

        bytes32 id = GasOptimized.createId(creator, nonce, timestamp);

        // Should include chain ID
        assertEq(id, keccak256(abi.encodePacked(creator, nonce, timestamp, block.chainid)));
    }

    // ============================================================
    // Address Utilities
    // ============================================================

    function test_isContract() public {
        // EOA should return false
        assertFalse(GasOptimized.isContract(address(0x1)));

        // This test contract should return true
        assertTrue(GasOptimized.isContract(address(this)));
    }

    // ============================================================
    // Time Utilities
    // ============================================================

    function test_isPast() public {
        assertTrue(GasOptimized.isPast(block.timestamp - 1));
        assertFalse(GasOptimized.isPast(block.timestamp));
        assertFalse(GasOptimized.isPast(block.timestamp + 1));
    }

    function test_isFuture() public {
        assertFalse(GasOptimized.isFuture(block.timestamp - 1));
        assertFalse(GasOptimized.isFuture(block.timestamp));
        assertTrue(GasOptimized.isFuture(block.timestamp + 1));
    }

    function test_getPeriod() public {
        uint256 dayLength = 1 days;
        uint256 period = GasOptimized.getPeriod(dayLength);

        // Should be current day number since epoch
        assertEq(period, block.timestamp / dayLength);
    }

    // ============================================================
    // Storage Utilities
    // ============================================================

    function test_pack128() public pure {
        uint128 a = 12345;
        uint128 b = 67890;

        uint256 packed = GasOptimized.pack128(a, b);

        assertEq(uint128(packed), a);
        assertEq(uint128(packed >> 128), b);
    }

    function test_unpack128() public pure {
        uint128 a = 12345;
        uint128 b = 67890;

        uint256 packed = GasOptimized.pack128(a, b);
        (uint128 unpacked_a, uint128 unpacked_b) = GasOptimized.unpack128(packed);

        assertEq(unpacked_a, a);
        assertEq(unpacked_b, b);
    }

    function testFuzz_pack128Roundtrip(uint128 a, uint128 b) public pure {
        uint256 packed = GasOptimized.pack128(a, b);
        (uint128 unpacked_a, uint128 unpacked_b) = GasOptimized.unpack128(packed);

        assertEq(unpacked_a, a);
        assertEq(unpacked_b, b);
    }

    function test_pack64() public pure {
        uint64 a = 1;
        uint64 b = 2;
        uint64 c = 3;
        uint64 d = 4;

        uint256 packed = GasOptimized.pack64(a, b, c, d);

        assertEq(uint64(packed), a);
        assertEq(uint64(packed >> 64), b);
        assertEq(uint64(packed >> 128), c);
        assertEq(uint64(packed >> 192), d);
    }

    function test_unpack64() public pure {
        uint64 a = 1;
        uint64 b = 2;
        uint64 c = 3;
        uint64 d = 4;

        uint256 packed = GasOptimized.pack64(a, b, c, d);
        (
            uint64 unpacked_a,
            uint64 unpacked_b,
            uint64 unpacked_c,
            uint64 unpacked_d
        ) = GasOptimized.unpack64(packed);

        assertEq(unpacked_a, a);
        assertEq(unpacked_b, b);
        assertEq(unpacked_c, c);
        assertEq(unpacked_d, d);
    }

    function testFuzz_pack64Roundtrip(uint64 a, uint64 b, uint64 c, uint64 d) public pure {
        uint256 packed = GasOptimized.pack64(a, b, c, d);
        (
            uint64 unpacked_a,
            uint64 unpacked_b,
            uint64 unpacked_c,
            uint64 unpacked_d
        ) = GasOptimized.unpack64(packed);

        assertEq(unpacked_a, a);
        assertEq(unpacked_b, b);
        assertEq(unpacked_c, c);
        assertEq(unpacked_d, d);
    }

    // ============================================================
    // Gas Benchmarks
    // ============================================================

    function test_gasComparison_uncheckedLoop() public pure {
        // This demonstrates gas savings from unchecked increment
        uint256 sum;
        for (uint256 i; i < 100;) {
            sum += i;
            unchecked { ++i; }
        }
        assertEq(sum, 4950);
    }

    function test_gasComparison_checkedLoop() public pure {
        // Standard loop with checked arithmetic
        uint256 sum;
        for (uint256 i = 0; i < 100; i++) {
            sum += i;
        }
        assertEq(sum, 4950);
    }
}

// Test Multicall contract
contract MulticallTest is Test {
    MulticallMock multicall;

    function setUp() public {
        multicall = new MulticallMock();
    }

    function test_multicall() public {
        bytes[] memory data = new bytes[](3);
        data[0] = abi.encodeWithSelector(MulticallMock.increment.selector);
        data[1] = abi.encodeWithSelector(MulticallMock.increment.selector);
        data[2] = abi.encodeWithSelector(MulticallMock.getValue.selector);

        bytes[] memory results = multicall.multicall(data);

        // Last result should be the counter value (2)
        uint256 value = abi.decode(results[2], (uint256));
        assertEq(value, 2);
    }

    function test_multicallReverts() public {
        bytes[] memory data = new bytes[](2);
        data[0] = abi.encodeWithSelector(MulticallMock.increment.selector);
        data[1] = abi.encodeWithSelector(MulticallMock.revertCall.selector);

        vm.expectRevert("Test revert");
        multicall.multicall(data);
    }
}

// Mock contract for Multicall testing
contract MulticallMock is Multicall {
    uint256 public counter;

    function increment() external returns (uint256) {
        return ++counter;
    }

    function getValue() external view returns (uint256) {
        return counter;
    }

    function revertCall() external pure {
        revert("Test revert");
    }
}
